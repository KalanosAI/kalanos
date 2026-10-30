"""Lazy video payloads and the optional decoder behind them.

Lives outside any one adapter because more than one format (LeRobot, MCAP, RLDS)
carries frames as an mp4 alongside timestamped metadata, and each wants the same lazy,
optional decoder.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import threading
from collections import OrderedDict
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# External
from upath import UPath

# Internal
from kalanos.analysis.entry_points import MissingDependency
from kalanos.analysis.optional import load_numpy


if TYPE_CHECKING:
    import numpy as np

# Dependencies added by the video extra
# A missing decoder fails only the reads that need it, not the import:
# every LeRobot dataset loads through this module, video or not.
try:
    import av
    import av.error
except ImportError:
    av = None


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Opening a video re-reads its index, through many small range requests when remote,
# and many payloads may share one file. So files stay open across payloads,
# keyed on their path, which assumes a file does not change during a run.
# A container has one read position, so a read checks it out of the pool
# and puts it back when done; no two reads ever share one.
# Each remote handle's block cache holds at most 32 blocks (fsspec's default)
# of this size.
_REMOTE_BLOCK_SIZE = 1 << 20
_MAX_OPEN_FILES = 4
# Idle files only, least recently used first.
_OPEN_FILES: "OrderedDict[str, _VideoFile]" = OrderedDict()
# Guards `_OPEN_FILES` alone, never held during a read.
_OPEN_LOCK = threading.Lock()

# FFmpeg's names for its colour codes, as libavutil/pixdesc.c gives them.
# Unspecified and reserved codes are left out, so they read as unstated.
_COLOR_RANGES = {1: "tv", 2: "pc"}
_COLOR_PRIMARIES = {
    1: "bt709",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "film",
    9: "bt2020",
    10: "smpte428",
    11: "smpte431",
    12: "smpte432",
    22: "ebu3213",
    256: "vgamut",
}
_COLOR_TRANSFERS = {
    1: "bt709",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "linear",
    9: "log100",
    10: "log316",
    11: "iec61966-2-4",
    12: "bt1361e",
    13: "iec61966-2-1",
    14: "bt2020-10",
    15: "bt2020-12",
    16: "smpte2084",
    17: "smpte428",
    18: "arib-std-b67",
    256: "vlog",
}
_COLOR_SPACES = {
    0: "gbr",
    1: "bt709",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycgco",
    9: "bt2020nc",
    10: "bt2020c",
    11: "smpte2085",
    12: "chroma-derived-nc",
    13: "chroma-derived-c",
    14: "ictcp",
    15: "ipt-c2",
    16: "ycgco-re",
    17: "ycgco-ro",
}

# The container tags read, by allowlist: location and other personal tags,
# which phones and cameras also write, are never read.
_MAKE_TAGS = ("com.apple.quicktime.make", "make", "com.android.manufacturer")
_MODEL_TAGS = ("com.apple.quicktime.model", "model", "com.android.model")
_ENCODER_TAGS = ("encoder",)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class DecoderUnavailable(MissingDependency):
    """Raised when a video payload is decoded without the decoder installed."""

    def __init__(self, path: UPath) -> None:
        """Name the payload that could not be decoded and the extra that decodes it.

        Parameters
        ----------
        path : UPath
            The video file the read was asked for.
        """

        super().__init__("kalanos", "video")
        self.msg = f"{path}: decoding a video payload needs PyAV; add {self.target}"


class DecodeFailed(RuntimeError):
    """Raised when PyAV is installed but a video file cannot be opened or decoded."""


class ReadInterrupted(DecodeFailed):
    """Raised when a video file's read fails outside the file, such as on the network.

    The file may well be readable on the next attempt,
    so the failure says nothing about the file itself.
    """


class DecodeLimitReached(RuntimeError):
    """Raised when a read stops at `max_decode_frames` or `max_pixels`."""


@runtime_checkable
class SampledFrames(Protocol):
    """A payload that can hand out a few frames as small gray images.

    The only way a vision metric reads pixels,
    so a payload without `gray_windows` is never decoded.
    """

    def __len__(self) -> int:
        """Count the payload's frames."""

        ...

    def gray_windows(
        self,
        windows: Sequence[tuple[int, int]],
        size: int,
        native: frozenset[int] = frozenset(),
        max_decode_frames: int | None = None,
        max_pixels: int | None = None,
    ) -> (
        "Generator[tuple[int, np.ndarray, np.ndarray | None, float | None], None, None]"
    ):
        """Yield the consecutive frames of each `[start, end)` window as gray images.

        Frames are yielded as they decode, so a caller that keeps only
        what it measures of each frame reads a whole segment in flat memory.
        A caller that stops early closes the iterator,
        which releases the file it holds open.

        Parameters
        ----------
        windows : Sequence[tuple[int, int]]
            Ascending, non-overlapping ranges of the payload's own frames.
        size : int
            The side of each yielded square image, in pixels.
        native : frozenset[int]
            The frame indices within the payload
            that also come back at native resolution.
        max_decode_frames : int | None
            The most frames the read may decode, seek preroll included.
        max_pixels : int | None
            The most pixels a frame in `native` may hold.

        Yields
        ------
        tuple[int, numpy.ndarray, numpy.ndarray | None, float | None]
            One frame, in window order and frame order within each window:

            - the window's position in `windows`;
            - the frame as a `(size, size)` uint8 image;
            - the frame's `(height, width)` uint8 luminance
              when its index is in `native`, `None` otherwise;
            - its presentation time in seconds when the container has one,
              `None` otherwise.

            A window is cut short only where decoding runs out before the segment ends.

        Raises
        ------
        DecodeLimitReached
            If the read would pass `max_decode_frames`,
            or a frame in `native` holds more than `max_pixels`.
        """

        ...

    def rgb_frame(self, index: int) -> "np.ndarray":
        """Decode one frame of the payload at native resolution as RGB, for a preview.

        Raises the same errors as `gray_windows`.
        """

        ...


@dataclass(frozen=True)
class SegmentPackets:
    """The packets a video segment holds, as the container demuxes them.

    Attributes
    ----------
    pts : tuple of int
        Every counted packet's presentation timestamp, sorted, in `time_base` ticks.
    time_base : float
        Seconds per tick.
    fps : float
        The rate the segment was placed by:
        the segment's own, its frame count over its span,
        or the container's average rate when the segment has no frames or no span.
    """

    pts: tuple[int, ...]
    time_base: float
    fps: float


@dataclass(frozen=True)
class DeclaredFormat:
    """The format a dataset declares for one camera's video files.

    Each field is what the dataset's own metadata states; `None` means undeclared.

    Attributes
    ----------
    width : int or None
        The frame width in pixels.
    height : int or None
        The frame height in pixels.
    fps : float or None
        The frame rate.
    pix_fmt : str or None
        The pixel format, as FFmpeg names it (`yuv420p`).
    codec : str or None
        The video codec, as the dataset names it (`av1`).
    channels : int or None
        How many channels each frame holds.
    is_depth_map : bool or None
        Whether the camera records depth rather than colour.
    has_audio : bool or None
        Whether the video files carry an audio track.
    """

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    pix_fmt: str | None = None
    codec: str | None = None
    channels: int | None = None
    is_depth_map: bool | None = None
    has_audio: bool | None = None


@dataclass(frozen=True)
class ContainerHeader:
    """The format a video file's container states for its first video stream.

    It describes the whole file, which in a LeRobot v3.0 dataset holds several episodes.
    Each field after `pix_fmt` is `None` where the container states nothing.

    Attributes
    ----------
    width : int
        The frame width in pixels.
    height : int
        The frame height in pixels.
    fps : float or None
        The container's average frame rate, `None` when it states none.
    pix_fmt : str or None
        The pixel format, `None` when the codec parameters state none.
    codec : str or None
        The codec, as FFmpeg names it (`h264`).
    profile : str or None
        The codec profile (`High`).
    bit_rate : int or None
        The stream's bitrate in bits per second.
    frame_count : int or None
        How many frames the stream holds.
    duration_s : float or None
        The stream's duration in seconds.
    color_range : str or None
        The colour range, as FFmpeg names it (`tv`).
    color_primaries : str or None
        The colour primaries, as FFmpeg names them (`bt709`).
    color_transfer : str or None
        The transfer characteristic, as FFmpeg names it (`bt709`).
    color_space : str or None
        The colour space, as FFmpeg names it (`bt709`).
    camera_make : str or None
        The camera maker a container tag names.
    camera_model : str or None
        The camera model a container tag names.
    encoder : str or None
        The software that wrote the file, as its `encoder` tag names it.
    """

    width: int
    height: int
    fps: float | None
    pix_fmt: str | None
    codec: str | None = None
    profile: str | None = None
    bit_rate: int | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    color_range: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_space: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    encoder: str | None = None


@dataclass
class _VideoFile:
    """An open video file, as the pool of open files holds it.

    Attributes
    ----------
    container : Any
        The PyAV input container.
    handle : Any or None
        The fsspec file object a remote container reads, `None` for a local file.
    header : ContainerHeader or None
        The header read at open, `None` when the file has no video stream.
        Decoding updates the codec context from the bitstream,
        so a header read after a decode could differ from one read at open.
    """

    container: Any
    handle: Any | None
    header: ContainerHeader | None

    def close(self) -> None:
        """Close the container, then the file object it reads."""

        try:
            self.container.close()
        finally:
            if self.handle is not None:
                self.handle.close()


@dataclass(frozen=True)
class VideoPayload:
    """A video stream's frames, decoded only when `fetch` is called.

    Attributes
    ----------
    path : UPath
        The video file this payload's frames come from.
    frame_count : int
        How many frames the stream declares, without opening `path`.
    start_s : float
        Where in `path` this stream's frames begin.
    end_s : float
        Where in `path` this stream's frames end.
    whole_file : bool
        Whether `path` holds this episode alone,
        so frames past `end_s` still belong to it.
    declared : DeclaredFormat or None
        The format the dataset declares for this camera,
        `None` when the adapter reads no declaration.
    """

    path: UPath
    frame_count: int
    start_s: float
    end_s: float
    whole_file: bool = False
    declared: DeclaredFormat | None = None

    def __len__(self) -> int:
        """Return the declared frame count, reading nothing.

        Returns
        -------
        int
            `frame_count`.
        """

        return self.frame_count

    @contextmanager
    def _open(self, av: ModuleType) -> Generator["_VideoFile", None, None]:
        """Check `path` out of the pool of open files, opening it when none is idle.

        The file goes back into the pool when the read ends or is abandoned.
        A read that fails may leave the container broken, so it is closed instead.
        """

        file = _check_out(self.path) or _open_file(av, self.path)
        try:
            yield file
        except GeneratorExit:
            _check_in(self.path, file)
            raise
        except BaseException:
            file.close()
            raise
        _check_in(self.path, file)

    def _bounds(self, stream: Any) -> tuple[int, int, float]:
        """Return this segment's `[lo, hi)` in the stream's time base, and its fps.

        Both bounds sit half a frame early:
        a float `start_s`/`end_s` does not land exactly on a frame's pts,
        and strict bounds misread 11 of 40 toto episodes by one frame.
        The frame is the segment's own, its frame count over its span, when it has one:
        a stall anywhere in a shared file lowers the container's average rate,
        and once it falls to half the real rate or below,
        half of that longer frame spans a whole real one,
        and every segment is pulled a frame early.
        """

        span = self.end_s - self.start_s
        if self.frame_count > 0 and span > 0:
            fps = self.frame_count / span
        else:
            fps = float(stream.average_rate or 0)
            if fps <= 0:
                raise DecodeFailed(
                    f"{self.path}: no frame rate in the file or the segment"
                )
        time_base = float(stream.time_base)
        lo = round((self.start_s - 0.5 / fps) / time_base)
        hi = round((self.end_s - 0.5 / fps) / time_base)
        return lo, hi, fps

    def gray_windows(
        self,
        windows: Sequence[tuple[int, int]],
        size: int,
        native: frozenset[int] = frozenset(),
        max_decode_frames: int | None = None,
        max_pixels: int | None = None,
    ) -> (
        "Generator[tuple[int, np.ndarray, np.ndarray | None, float | None], None, None]"
    ):
        """Decode every frame of each `[start, end)` window as `(size, size)` gray.

        Each window costs one seek back to a keyframe and a decode forward through it,
        so a full scan is the single window `[(0, frame_count)]`.
        The file stays checked out, so no other read reuses its container,
        until the iterator is exhausted or closed.

        Parameters
        ----------
        windows : Sequence[tuple[int, int]]
            Ascending ranges of positions within this segment,
            `0 <= start < end <= frame_count`.
        size : int
            The side of each returned square image, in pixels.
        native : frozenset[int]
            The frame indices within this segment
            that also come back at native resolution.
        max_decode_frames : int | None
            The most frames the read may decode, seek preroll included.
        max_pixels : int | None
            The most pixels a frame in `native` may hold.

        Yields
        ------
        tuple[int, numpy.ndarray, numpy.ndarray | None, float | None]
            One frame, in window order and frame order within each window:

            - the window's position in `windows`;
            - the frame as a `(size, size)` uint8 image;
            - the frame's native `(height, width)` luminance
              when its index is in `native`, `None` otherwise;
            - its presentation time in seconds when the container has one,
              `None` otherwise.

            A window is cut short only where decoding runs out before the segment ends.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        DecodeFailed
            If the file cannot be opened, seeked or decoded,
            or has no frame rate to place the segment by.
        DecodeLimitReached
            If the read would pass `max_decode_frames`,
            or a frame in `native` holds more than `max_pixels`.
        """

        return self._windows(
            windows, size, "gray", native, max_decode_frames, max_pixels
        )

    def rgb_frame(self, index: int) -> "np.ndarray":
        """Decode one frame of the segment at native resolution as RGB, for a preview.

        Returns
        -------
        numpy.ndarray
            The `(height, width, 3)` uint8 frame at position `index`.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        DecodeFailed
            If the file cannot be opened, seeked or decoded,
            or the segment holds no frame at `index`.
        """

        # Closed here rather than left to the collector,
        # which would keep the file checked out and its container unshared.
        with closing(self._windows([(index, index + 1)], 0, None)) as frames:
            for _, frame, _native, _pts in frames:
                return frame
        raise DecodeFailed(f"{self.path}: no frame at position {index}")

    def rgb_frames(self, size: int | None) -> "list[np.ndarray]":
        """Decode every frame of the segment as an RGB image.

        Parameters
        ----------
        size : int or None
            The side of each returned square image, in pixels,
            or `None` for frames at native resolution.

        Returns
        -------
        list of numpy.ndarray
            One uint8 frame per position, in order;
            short only where decoding runs out before the segment ends.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        DecodeFailed
            If the file cannot be opened, seeked or decoded,
            or has no frame rate to place the segment by.
        """

        if size is None:
            windows = self._windows([(0, self.frame_count)], 0, None)
        else:
            windows = self._windows([(0, self.frame_count)], size, "rgb24")
        return [frame for _, frame, _native, _pts in windows]

    def _windows(
        self,
        windows: Sequence[tuple[int, int]],
        size: int,
        pixel_format: str | None,
        native: frozenset[int] = frozenset(),
        max_decode_frames: int | None = None,
        max_pixels: int | None = None,
    ) -> (
        "Generator[tuple[int, np.ndarray, np.ndarray | None, float | None], None, None]"
    ):
        """Decode each window's frames as `(size, size)` images in `pixel_format`.

        A `pixel_format` of `None` yields each frame unresized, as RGB.
        """

        if av is None:
            raise DecoderUnavailable(self.path)
        decoded = 0
        try:
            with self._open(av) as file:
                container = file.container
                stream = container.streams.video[0]
                lo, hi, fps = self._bounds(stream)
                time_base = float(stream.time_base)
                for position, (start, end) in enumerate(windows):
                    # lo already sits half a frame early, so both bounds do too.
                    start_pts = min(lo + round(start / fps / time_base), hi - 1)
                    end_pts = min(lo + round(end / fps / time_base), hi)
                    container.seek(start_pts, stream=stream)
                    index = start
                    for frame in container.decode(stream):
                        decoded += 1
                        if (
                            max_decode_frames is not None
                            and decoded > max_decode_frames
                        ):
                            raise DecodeLimitReached("max_decode_frames reached")
                        if frame.pts is None or frame.pts < start_pts:
                            continue
                        if frame.pts >= end_pts:
                            break
                        if (
                            max_pixels is not None
                            and index in native
                            and frame.width * frame.height > max_pixels
                        ):
                            raise DecodeLimitReached("frame exceeds max_pixels")
                        if pixel_format is None:
                            image = frame.to_ndarray(format="rgb24")
                        else:
                            image = frame.reformat(
                                width=size, height=size, format=pixel_format
                            ).to_ndarray()
                        full = _luminance(frame) if index in native else None
                        pts = float(frame.time) if frame.time is not None else None
                        yield position, image, full, pts
                        index += 1
        # A remote open fails with OSError from fsspec, not with an FFmpegError.
        except (av.error.FFmpegError, OSError) as exc:
            raise _read_failure(self.path, exc) from exc

    def header(self) -> ContainerHeader:
        """Read the format the container states for its first video stream.

        Decodes nothing.

        Returns
        -------
        ContainerHeader
            The stream's frame size, rate, pixel format, codec, bitrate,
            length and colour, and the camera and encoder its tags name,
            all for the whole file.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        DecodeFailed
            If the file cannot be opened, or holds no video stream.
        """

        if av is None:
            raise DecoderUnavailable(self.path)
        try:
            with self._open(av) as file:
                if file.header is None:
                    raise DecodeFailed(f"{self.path}: no video stream")
                return file.header
        # A remote open fails with OSError from fsspec, not with an FFmpegError.
        except (av.error.FFmpegError, OSError) as exc:
            raise _read_failure(self.path, exc) from exc

    def packet_times(self) -> SegmentPackets:
        """Read the timestamp of every packet in this segment, decoding nothing.

        The packets are what the container holds, not capped at `frame_count`.
        On a `whole_file` payload the segment runs to the end of the file.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        DecodeFailed
            If the file cannot be opened or demuxed,
            or has no frame rate to place the segment by.
        """

        if av is None:
            raise DecoderUnavailable(self.path)
        collected: list[int] = []
        try:
            with self._open(av) as file:
                container = file.container
                stream = container.streams.video[0]
                lo, hi, fps = self._bounds(stream)
                time_base = float(stream.time_base)
                container.seek(lo, stream=stream)
                for packet in container.demux(stream):
                    if packet.pts is None or packet.size == 0:
                        continue
                    if self.whole_file:
                        if lo <= packet.pts:
                            collected.append(packet.pts)
                        continue
                    # Packets arrive in decode order, so with B-frames a pts past hi
                    # can precede ones below it; dts never exceeds pts and only grows.
                    if packet.dts is not None and packet.dts >= hi:
                        break
                    if lo <= packet.pts < hi:
                        collected.append(packet.pts)
        # A remote open fails with OSError from fsspec, not with an FFmpegError.
        except (av.error.FFmpegError, OSError) as exc:
            raise _read_failure(self.path, exc) from exc
        return SegmentPackets(
            pts=tuple(sorted(collected)), time_base=time_base, fps=fps
        )

    def fetch(self) -> "list[np.ndarray]":
        """Decode and return every frame between `start_s` and `end_s`.

        Returns
        -------
        list of numpy.ndarray
            One `(height, width, 3)` RGB array per frame, in order.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        DecodeFailed
            If the video stream has no time base to seek by.
        """

        if av is None:
            raise DecoderUnavailable(self.path)

        container = av.open(str(self.path))
        try:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise DecodeFailed(f"{self.path}: the video stream has no time base")
            container.seek(int(self.start_s / stream.time_base), stream=stream)
            # end_s is the exclusive upper bound: it is the next segment's own start_s,
            # so a frame landing exactly on it belongs there, not here.
            # The count is also capped at frame_count, so a float-precision mismatch
            # can't pull in an extra frame.
            frames = []
            for frame in container.decode(stream):
                if len(frames) >= self.frame_count:
                    break
                if frame.time is not None and frame.time >= self.end_s:
                    break
                if frame.time is not None and frame.time < self.start_s:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
        finally:
            container.close()

        if len(frames) != self.frame_count:
            logger.warning(
                "%s: declared %d frame(s) but decoded %d between %.6fs and %.6fs",
                self.path,
                self.frame_count,
                len(frames),
                self.start_s,
                self.end_s,
            )
        return frames


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _color_name(codec: Any, attribute: str, names: dict[int, str]) -> str | None:
    """Name the colour code a codec context holds, as FFmpeg names it.

    Parameters
    ----------
    codec : Any
        The stream's PyAV codec context.
    attribute : str
        The codec context attribute holding the code, such as `color_trc`.
    names : dict
        FFmpeg's name for each code.

    Returns
    -------
    str or None
        The code's name, `None` when the attribute is missing,
        or the code is unspecified, reserved or unknown.
    """

    # Older PyAV releases within the supported range lack some of these attributes.
    code = getattr(codec, attribute, None)
    if code is None:
        return None
    try:
        return names.get(int(code))
    except (TypeError, ValueError):
        return None


def _tag(container: Any, stream: Any, keys: Sequence[str]) -> str | None:
    """Read the first of `keys` that the container's tags, then the stream's, hold.

    Parameters
    ----------
    container : Any
        The open PyAV container.
    stream : Any
        Its first video stream.
    keys : Sequence of str
        The tag names to look up, in order of preference.

    Returns
    -------
    str or None
        The first non-blank string value, `None` when no key holds one.
    """

    for tags in (container.metadata, stream.metadata):
        for key in keys:
            value = tags.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _open_file(av: ModuleType, path: UPath) -> _VideoFile:
    """Open `path` as a PyAV container, local or remote, and read its header.

    Parameters
    ----------
    av : ModuleType
        The PyAV module.
    path : UPath
        The video file to open.

    Returns
    -------
    _VideoFile
        The open container, its file object when remote, and its header.
    """

    if path.protocol in ("", "file"):
        handle = None
        container = av.open(str(path))
    else:
        # PyAV cannot resolve an fsspec URL such as hf://, so it reads a file object.
        handle = path.open("rb", block_size=_REMOTE_BLOCK_SIZE, cache_type="blockcache")
        try:
            container = av.open(handle)
        except BaseException:
            handle.close()
            raise
    file = _VideoFile(container=container, handle=handle, header=None)
    try:
        if container.streams.video:
            file.header = _read_header(container)
    except BaseException:
        file.close()
        raise
    return file


def _read_header(container: Any) -> ContainerHeader:
    """Read the format an open container states for its first video stream.

    Parameters
    ----------
    container : Any
        A PyAV container with at least one video stream, not yet decoded from.

    Returns
    -------
    ContainerHeader
        The stream's frame size, rate, pixel format, codec, bitrate,
        length and colour, and the camera and encoder its tags name.
    """

    stream = container.streams.video[0]
    codec = stream.codec_context
    rate = stream.average_rate
    # PyAV 17 segfaults reading codec.pix_fmt on a stream with none set;
    # codec.format is None there instead.
    pixel_format = codec.format
    bit_rate = codec.bit_rate or stream.bit_rate
    duration = stream.duration
    time_base = stream.time_base
    return ContainerHeader(
        width=int(codec.width),
        height=int(codec.height),
        fps=float(rate) if rate else None,
        pix_fmt=pixel_format.name if pixel_format is not None else None,
        # codec.name is the decoder FFmpeg picked, libdav1d for AV1.
        # PyAV 13 has no canonical_name, so the codec reads as unstated there.
        codec=getattr(codec.codec, "canonical_name", None) or None,
        profile=codec.profile or None,
        # An unset count, length or bitrate reads as 0, not as missing.
        bit_rate=int(bit_rate) if bit_rate else None,
        frame_count=int(stream.frames) if stream.frames else None,
        duration_s=float(duration * time_base) if duration and time_base else None,
        color_range=_color_name(codec, "color_range", _COLOR_RANGES),
        color_primaries=_color_name(codec, "color_primaries", _COLOR_PRIMARIES),
        color_transfer=_color_name(codec, "color_trc", _COLOR_TRANSFERS),
        color_space=_color_name(codec, "colorspace", _COLOR_SPACES),
        camera_make=_tag(container, stream, _MAKE_TAGS),
        camera_model=_tag(container, stream, _MODEL_TAGS),
        encoder=_tag(container, stream, _ENCODER_TAGS),
    )


def _check_out(path: UPath) -> _VideoFile | None:
    """Take the idle open file for `path` out of the pool, if there is one.

    Parameters
    ----------
    path : UPath
        The video file a read wants.

    Returns
    -------
    _VideoFile or None
        The file, now held by the caller alone, or `None` when none is idle.
    """

    with _OPEN_LOCK:
        return _OPEN_FILES.pop(str(path), None)


def _check_in(path: UPath, file: _VideoFile) -> None:
    """Put an open file back into the pool as the most recently used.

    When two reads of `path` ran at once, the pool already holds an idle file for it
    and the incoming one is closed.
    Files past `_MAX_OPEN_FILES` are closed oldest first.

    Parameters
    ----------
    path : UPath
        The video file `file` reads.
    file : _VideoFile
        The open file, no longer used by any read.
    """

    evicted: list[_VideoFile] = []
    key = str(path)
    with _OPEN_LOCK:
        if key in _OPEN_FILES:
            evicted.append(file)
        else:
            _OPEN_FILES[key] = file
            while len(_OPEN_FILES) > _MAX_OPEN_FILES:
                evicted.append(_OPEN_FILES.popitem(last=False)[1])
    for stale in evicted:
        stale.close()


def close_video_files() -> None:
    """Close and forget every pooled video file, at the end of a run."""

    with _OPEN_LOCK:
        files = list(_OPEN_FILES.values())
        _OPEN_FILES.clear()
    for file in files:
        file.close()


def _read_failure(path: UPath, exc: Exception) -> DecodeFailed:
    """Classify a failed read as the file's fault or an interruption outside it.

    Parameters
    ----------
    path : UPath
        The video file the read was for.
    exc : Exception
        What the read raised: an FFmpeg error, or an OSError from fsspec.

    Returns
    -------
    DecodeFailed
        `DecodeFailed` when FFmpeg rejected the bytes or the file does not exist,
        `ReadInterrupted` for any other OSError,
        which a remote file raises on a dropped connection or an HTTP error status.
    """

    # FFmpeg's errors subclass OSError too, so they are matched first.
    if isinstance(exc, FileNotFoundError) or (
        av is not None and isinstance(exc, av.error.FFmpegError)
    ):
        return DecodeFailed(f"{path}: {exc}")
    return ReadInterrupted(f"{path}: {exc}")


def _luminance(frame: Any) -> "np.ndarray":
    """The frame's native-resolution luminance, the Y plane itself for planar YUV."""

    numpy = load_numpy()
    assert numpy is not None

    pixel_format = frame.format
    # A deeper format stores each Y sample in two bytes,
    # so only an 8-bit plane reads as is.
    if (
        pixel_format.name.startswith("yuv")
        and pixel_format.is_planar
        and pixel_format.components[0].bits == 8
    ):
        plane = frame.planes[0]
        rows = numpy.frombuffer(plane, dtype=numpy.uint8).reshape(
            plane.height, plane.line_size
        )
        return rows[:, : plane.width].copy()
    return frame.reformat(format="gray").to_ndarray().copy()
