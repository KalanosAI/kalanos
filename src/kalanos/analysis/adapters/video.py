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
from kalanos.analysis.optional import load_av, load_numpy


if TYPE_CHECKING:
    import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

logger = logging.getLogger(__name__)

# Opening a remote video re-reads its index through many small range requests,
# and many payloads may share one file. So remote files stay open across payloads,
# keyed on their URL, which assumes a remote file does not change during a run.
# Each handle's block cache holds at most 32 blocks (fsspec's default) of this size.
_REMOTE_BLOCK_SIZE = 1 << 20
_MAX_REMOTE_HANDLES = 4
_REMOTE_HANDLES: "OrderedDict[str, Any]" = OrderedDict()
# A shared handle has one read position, so only one container may use it at a time.
_REMOTE_LOCK = threading.Lock()


class DecoderUnavailable(RuntimeError):
    """Raised when a video payload is fetched without the decoder installed."""


class DecodeFailed(RuntimeError):
    """Raised when PyAV is installed but a video file cannot be opened or decoded."""


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


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _remote_handle(path: UPath) -> Any:
    """Return an open, block-cached file object for `path`, reusing a cached one.

    The caller holds `_REMOTE_LOCK`.
    """

    key = str(path)
    if key in _REMOTE_HANDLES:
        _REMOTE_HANDLES.move_to_end(key)
        return _REMOTE_HANDLES[key]

    handle = path.open("rb", block_size=_REMOTE_BLOCK_SIZE, cache_type="blockcache")
    _REMOTE_HANDLES[key] = handle
    while len(_REMOTE_HANDLES) > _MAX_REMOTE_HANDLES:
        _key, evicted = _REMOTE_HANDLES.popitem(last=False)
        evicted.close()
    return handle


def close_remote_handles() -> None:
    """Close and forget every cached remote file object, at the end of a run."""

    with _REMOTE_LOCK:
        for handle in _REMOTE_HANDLES.values():
            handle.close()
        _REMOTE_HANDLES.clear()


def _forget_remote_handle(path: UPath) -> None:
    """Close and forget the cached file object for `path`, if there is one."""

    with _REMOTE_LOCK:
        handle = _REMOTE_HANDLES.pop(str(path), None)
        if handle is not None:
            handle.close()


def decoder_available() -> bool:
    """Whether a video payload can be decoded in this process.

    Returns
    -------
    bool
        Whether PyAV imports successfully.
    """

    return load_av() is not None


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


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


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
    """

    path: UPath
    frame_count: int
    start_s: float
    end_s: float

    def __len__(self) -> int:
        """Return the declared frame count, reading nothing.

        Returns
        -------
        int
            `frame_count`.
        """

        return self.frame_count

    def _av(self) -> ModuleType:
        """Import PyAV, raising `DecoderUnavailable` when it is not installed."""

        av = load_av()
        if av is None:
            raise DecoderUnavailable(
                f"{self.path}: decoding a video payload needs PyAV; add kalanos[video]"
            )
        return av

    @contextmanager
    def _open(self, av: ModuleType) -> Generator[Any, None, None]:
        """Open `path` as a PyAV container, local or remote, and close it on exit.

        A remote file object outlives the container, cached for the next payload.
        """

        if self.path.protocol in ("", "file"):
            container = av.open(str(self.path))
            try:
                yield container
            finally:
                container.close()
            return

        # PyAV cannot resolve an fsspec URL such as hf://, so it reads a file object.
        with _REMOTE_LOCK:
            handle = _remote_handle(self.path)
            handle.seek(0)
            container = av.open(handle)
            try:
                yield container
            finally:
                container.close()

    def _bounds(self, stream: Any) -> tuple[int, int, float]:
        """Return this segment's `[lo, hi)` in the stream's time base, and its fps.

        Both bounds sit half a frame early:
        a float `start_s`/`end_s` does not land exactly on a frame's pts,
        and strict bounds misread 11 of 40 toto episodes by one frame.
        """

        rate = float(stream.average_rate or 0)
        if rate <= 0 and (self.frame_count <= 0 or self.end_s <= self.start_s):
            raise DecodeFailed(f"{self.path}: no frame rate in the file or the segment")
        fps = rate if rate > 0 else self.frame_count / (self.end_s - self.start_s)
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
        The file stays open, and a remote one locked, until the iterator is exhausted
        or closed.

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

        # Closed here rather than left to the collector, which would hold a remote lock.
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

        av = self._av()
        decoded = 0
        try:
            with self._open(av) as container:
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
            # A handle that failed mid-read may be broken; the next payload reopens.
            _forget_remote_handle(self.path)
            raise DecodeFailed(f"{self.path}: {exc}") from exc

    def count_frames(self) -> int:
        """Count the packets in this segment, decoding nothing.

        The count is what the container holds, not capped at `frame_count`.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        DecodeFailed
            If the file cannot be opened or demuxed,
            or has no frame rate to place the segment by.
        """

        av = self._av()
        count = 0
        try:
            with self._open(av) as container:
                stream = container.streams.video[0]
                lo, hi, _fps = self._bounds(stream)
                container.seek(lo, stream=stream)
                for packet in container.demux(stream):
                    if packet.pts is None or packet.size == 0:
                        continue
                    # Packets arrive in decode order, so with B-frames a pts past hi
                    # can precede ones below it; dts never exceeds pts and only grows.
                    if packet.dts is not None and packet.dts >= hi:
                        break
                    if lo <= packet.pts < hi:
                        count += 1
        # A remote open fails with OSError from fsspec, not with an FFmpegError.
        except (av.error.FFmpegError, OSError) as exc:
            # A handle that failed mid-read may be broken; the next payload reopens.
            _forget_remote_handle(self.path)
            raise DecodeFailed(f"{self.path}: {exc}") from exc
        return count

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
        """

        av = load_av()
        if av is None:
            raise DecoderUnavailable(
                f"{self.path}: decoding a video payload needs PyAV; add kalanos[video]"
            )

        container = av.open(str(self.path))
        try:
            stream = container.streams.video[0]
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
