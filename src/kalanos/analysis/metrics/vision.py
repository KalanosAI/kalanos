"""The vision family.

Fourteen metrics measure camera footage:

- sharpness_score, exposure_shift_pct and exposure_level measure
  `StreamContext.vision_samples` evenly spaced frames of a camera stream
  (10 by default);
- blank_frame_pct, contrast_score, dark_frame_pct and bright_frame_pct
  read the statistics blur and exposure keep of those same frames,
  decoding nothing more, and the default policy reports them without a band;
- frozen_frame_pct reads that many windows of about 1 s of consecutive frames,
  and needs an action stream in the same episode;
- frame_count_vs_timebase, pts_defect_pct and max_gap_s share one packet probe,
  decoding nothing, under a full scan too;
- camera_sync_skew_ms compares the cameras of one episode through the same probe;
- camera_stream_present and video_format_consistency share one read of the
  container header, and presence also decodes the segment's first frame.

Blur and exposure each judge one episode by itself,
and `kalanos.analysis.scoring.cameras` compares one camera across episodes.
Under `StreamContext.full_frame_scan`, blur, exposure and freezes read every frame.

The three share one decode:

- when frozen_frame_pct runs, the sampled frames are taken from inside its
  windows, and otherwise only the sampled frames are decoded;
- each frame is decoded as a 128x128 gray image for exposure and freezes,
  and each sampled frame also at native resolution for blur;
- every frame is measured and dropped, so a full scan runs in flat memory.

`camera_frames` publishes the shared read as evidence,
which the `vision` diagnostic reads,
and `footage_vector` publishes the sampled frames, pooled,
for the cross-episode comparison of a camera's footage.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import functools
import hashlib
import logging
import math
from collections import OrderedDict
from contextlib import closing
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

# External
import polars as pl

# Internal
from kalanos.analysis.adapters.video import (
    ContainerHeader,
    DeclaredFormat,
    DecodeFailed,
    DecodeLimitReached,
    DecoderUnavailable,
    ReadInterrupted,
    SampledFrames,
    SegmentPackets,
    VideoPayload,
)
from kalanos.analysis.execution import reads_numeric_payloads
from kalanos.analysis.localization import support_for
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.dictionary import Modality
from kalanos.analysis.models.domain import ClockInfo, Kind, Payload, Stream
from kalanos.analysis.models.metrics import (
    EpisodeContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.models.report import (
    CameraFormat,
    CameraFrame,
    CameraFrames,
    VideoFormat,
)
from kalanos.analysis.optional import load_numpy
from kalanos.assets.dictionary import load_dictionary


if TYPE_CHECKING:
    import numpy as np

    # The frame cache's key: what was read, and how it was read.
    _Key = tuple[
        int,  # id(payload)
        tuple[tuple[int, int], ...],  # the [start, end) windows read
        tuple[int, ...],  # the sampled frame positions
        tuple[int, ...],  # the evidence frame positions
        int,  # VisionSpec.dark_level
        int,  # VisionSpec.bright_level
        int | None,  # the max_decode_frames cap, None under a full scan
        int,  # VisionSpec.max_pixels
    ]


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Small enough to convert cheaply, large enough for exposure and freeze thumbnails
# (rda's size).
_SAMPLE_SIZE = 128
# Blur is the share of edge energy a further 3x3 box blur removes:
# a sharp frame loses most of it, an already soft one little, whatever its content.
_REBLUR = 3

# Exposure is judged against the camera's own typical frame, not fixed gray levels:
# a simulator's white background reads exactly like a camera blown out 4x.
# A frame whose mean moves by more than this share of the typical mean is exposed
# differently, the way an auto-exposure jump or a light switching would leave it.
_SHIFT_SHARE = 0.35
# Nor by less than this many gray levels, so sensor noise in a dark scene is no shift.
_SHIFT_FLOOR = 20
# A frame clipping this much more of its pixels than usual is blown out (rda's share).
_CLIP_RISE = 0.20
# A P95 - P5 spread below this is a blank frame on any camera.
# Contrast is otherwise no exposure signal: it follows what is in view,
# and on lerobot/pusht a third of clean frames hold half the typical spread.
_BLANK_SPREAD = 5
# A frame is dark or bright when at least this share of its pixels is crushed
# or clipped, the share `kalanos.analysis.scoring.cameras` uses for a camera's traits.
_EXPOSED_SHARE = 0.5
# A frame is also dark when its P95 is at or below this level:
# a dim camera crushes few pixels to `VisionSpec.dark_level`,
# but keeps nearly all of them low.
# On the calibration datasets, no clean frame had a P95 below 141,
# and every frame dimmed to a tenth read 25 or less.
_DIM_P95 = 40

# rda's thumbnail side for frame differences,
# reached by averaging 2x2 blocks of the shared 128x128 decode.
_FREEZE_SIZE = 64
# The side the footage-variation vector pools each sampled frame to,
# measured on six LeRobot datasets (docs/METRICS.md, "Across episodes").
_VARIATION_SIZE = 16
# A shorter repeat is a frame-rate conversion or a dropped grab (rda).
_FREEZE_MIN_S = 0.5
# Frames differing by less than this are one frame, however clean the camera (rda).
_FREEZE_EPS_FLOOR = 0.10
# Or by less than this share of the 10th-percentile difference, its noise (rda).
_FREEZE_EPS_SHARE = 0.25
_FREEZE_NOISE_PCT = 10
# Freeze differences are scaled to this contrast,
# so a dim camera's smaller frame changes are not mistaken for repeats.
_FREEZE_CONTRAST = 255
# A step moves if its mean |d action| exceeds this share of the 90th percentile (rda).
_MOVING_SHARE = 0.05
_MOVING_PCT = 90
# A run counts when the actions move over more than half of it (rda).
_MOVING_MAJORITY = 0.5

# A presentation step longer than this many declared frame periods has lost a frame:
# half a frame of slack, as `VideoPayload` allows when it places a segment.
_GAP_PERIODS = 1.5

# Two frame rates differ when further apart than this share of the declared one.
# Candidate: it absorbs 30000/1001 against a declared 29.97,
# and catches any change larger than 0.1%.
_FPS_REL_TOL = 1e-3

# The metrics every decodable camera stream computes; the others can be
# not_applicable by design (no action stream, no video container).
CORE_METRICS = ("sharpness_score", "exposure_shift_pct")

# One episode's camera streams are graded back to back,
# so the cache only needs to span one episode's cameras.
_CACHE_SIZE = 8

# Keyed on id(payload), the frames read and how they were read.
# Two invariants hold it together:
# - the value holds the payload itself,
#   which stops its id being reused while the entry lives;
# - payloads are immutable in normal use,
#   so an in-place edit of a SyntheticFrames.frames list would read a stale sample.
_CACHE: "OrderedDict[_Key, tuple[SampledFrames, _Frames | MetricResult]]" = (
    OrderedDict()
)

# The packet probe, keyed on id(payload).
# The value holds the payload so its id is not reused while cached, as in `_CACHE`.
_PACKETS: "OrderedDict[int, tuple[VideoPayload, SegmentPackets | DecodeFailed]]" = (
    OrderedDict()
)

# The container header, keyed on id(payload).
# The value holds the payload so its id is not reused while cached, as in `_CACHE`.
_HEADERS: "OrderedDict[int, tuple[VideoPayload, ContainerHeader | DecodeFailed]]" = (
    OrderedDict()
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class _Frames:
    """What the vision metrics keep of one read of a camera stream.

    Attributes
    ----------
    windows : list of tuple of (int, int)
        The `[start, end)` ranges of frames decoded.
    n_read : int
        How many frames were decoded.
    sampled : list of int
        The positions of the frames blur and exposure measure, among those decoded.
    loss : list of float
        Each sampled frame's share of Laplacian variance
        lost under a `_REBLUR` box blur.
    laplacian : list of float
        Each sampled frame's native Laplacian variance.
    mean, p5, p95 : list of float
        Each sampled frame's mean, 5th and 95th percentile gray level.
    white_share : list of float
        Each sampled frame's share of pixels at or above `VisionSpec.bright_level`.
    dark_share : list of float
        Each sampled frame's share of pixels at or below `VisionSpec.dark_level`.
    diffs : list of tuple of (int, numpy.ndarray)
        Per range that decoded at least two frames, its start
        and the contrast-scaled mean absolute difference between each pair
        of neighbouring thumbnails, NaN where either frame is blank.
    luma_steps : list of tuple of (int, numpy.ndarray)
        The same ranges' starts, and the unscaled mean absolute difference
        between each pair of neighbouring thumbnails, blank or not.
    pts : list of float or None
        Each sampled frame's presentation time, `None` where the payload has none.
    shape : list of tuple of (int, int)
        Each sampled frame's native `(height, width)`.
    luma_sha256 : list of str
        SHA-256 of each sampled frame's native luminance.
    requested : list of int
        The positions `_layout` asked blur and exposure to measure.
    evidence_positions : list of int
        The positions a sampled read would measure,
        equal to `requested` unless the read is a full scan.
    pooled : dict of int to numpy.ndarray
        Each decoded evidence frame's `_VARIATION_SIZE` block mean
        of its `_SAMPLE_SIZE` gray image, scaled to 0-1, keyed by position.
    stopped : tuple of (Availability, str) or None
        Why the read stopped early, `None` when it ran to the end.
    """

    windows: list[tuple[int, int]]
    n_read: int
    sampled: list[int]
    loss: list[float]
    laplacian: list[float]
    mean: list[float]
    p5: list[float]
    p95: list[float]
    white_share: list[float]
    dark_share: list[float]
    diffs: "list[tuple[int, np.ndarray]]"
    luma_steps: "list[tuple[int, np.ndarray]]"
    pts: list[float | None]
    shape: list[tuple[int, int]]
    luma_sha256: list[str]
    requested: list[int]
    evidence_positions: list[int]
    pooled: "dict[int, np.ndarray]"
    stopped: tuple[Availability, str] | None = None


@dataclass(frozen=True)
class _Exposure:
    """exposure_shift_pct's verdict on each sampled frame."""

    counts: dict[str, int]
    bad: list[int]
    worst_index: int | None
    typical_mean: float
    typical_white: float


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@functools.cache
def _depth_types() -> frozenset[str]:
    """The taxonomy types the packaged dictionary marks as depth."""

    # Metrics are not handed the deployment's own dictionary,
    # so only the packaged one is read.
    dictionary = load_dictionary(None)
    return frozenset(
        taxonomy_type
        for taxonomy_type, entry in dictionary.entries.items()
        if entry.modality == Modality.DEPTH
    )


def is_camera_footage(kind: str, taxonomy_type: str) -> bool:
    """Whether a stream of this `Kind` value and taxonomy type is camera footage.

    Depth maps are not.
    """

    return (
        kind in (Kind.IMAGE.value, Kind.VIDEO.value)
        and taxonomy_type not in _depth_types()
    )


def _camera_payload(ctx: StreamContext) -> Payload | MetricResult:
    """Return the stream's payload, or `not_applicable` if it is not camera footage."""

    if ctx.stream.kind not in (Kind.IMAGE, Kind.VIDEO):
        return not_applicable("not a camera stream")
    if not ctx.reads_payloads:
        return not_applicable("metadata tier does not read frames")
    if ctx.payload is None:
        return not_applicable("the stream carries no frames")
    if ctx.taxonomy_type in _depth_types():
        return not_applicable("the stream is a depth map, not camera footage")
    # Frame positions are read as source rows, which a reordered stream breaks.
    if (
        not ctx.stream.source_order.preserved
        or ctx.stream.source_order.original_index is not None
    ):
        return not_applicable(
            "sampled video requires the adapter's original frame order"
        )
    return ctx.payload


def _readable(ctx: StreamContext) -> SampledFrames | MetricResult:
    """Return the stream's payload if its frames can be read, or `not_applicable`."""

    payload = _camera_payload(ctx)
    if isinstance(payload, MetricResult):
        return payload
    if not isinstance(payload, SampledFrames):
        return not_applicable("the stream's frames cannot be sampled")
    if load_numpy() is None:
        return not_applicable("sampling frames needs numpy; add kalanos[video]")
    return payload


def _frame_rate(ctx: StreamContext) -> float | None:
    """The stream's frame rate from its median timestamp gap, or `None` without one."""

    numpy = load_numpy()
    assert numpy is not None

    timestamps = ctx.timestamps.to_numpy()
    if len(timestamps) < 2:
        return None
    gap = float(numpy.median(numpy.diff(timestamps)))
    return 1 / gap if gap > 0 else None


def _min_run(fps: float) -> int:
    """How many frozen transitions in a row make a freeze at `fps`."""

    return max(round(_FREEZE_MIN_S * fps), 2)


def _spread(n: int, count: int) -> list[int]:
    """Up to `count` evenly spaced positions among `n` frames, both ends included."""

    if n <= count:
        return list(range(n))
    step = (n - 1) / max(count - 1, 1)
    return sorted({round(k * step) for k in range(count)})


def _layout(
    ctx: StreamContext, full_frame_scan: bool
) -> tuple[list[tuple[int, int]], list[int]]:
    """Pick frozen frames' windows and the frames blur and exposure measure.

    Frozen frames read `vision_samples` evenly spaced windows,
    each two minimum freeze runs long (about 1 s),
    or the whole stream when the windows would cover it.
    Blur and exposure measure `vision_samples` evenly spaced frames,
    each inside a window, so one decode serves all three.
    Under `full_frame_scan`, all three read every frame.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to lay out.
    full_frame_scan : bool
        Whether to read every frame; passed apart from `ctx`
        so a full scan can also find the frames a sample would take.

    Returns
    -------
    tuple of (list of tuple of (int, int), list of int)
        The `[start, end)` windows, and the sampled frame positions.
    """

    n = ctx.n_samples
    count = ctx.vision_samples
    fps = _frame_rate(ctx)
    length = 2 * _min_run(fps) if fps is not None else 1

    if full_frame_scan:
        return [(0, n)], list(range(n))
    if n <= count * length:
        return [(0, n)], _spread(n, count)
    if count == 1:
        start = (n - length) // 2
        return [(start, start + length)], [start + length // 2]
    # The k-th sample sits k/(count - 1) of the way through the k-th window,
    # which spaces the samples as evenly as the windows.
    starts = [round(k * (n - length) / (count - 1)) for k in range(count)]
    offsets = [round(k * (length - 1) / (count - 1)) for k in range(count)]
    return (
        [(start, start + length) for start in starts],
        [start + offset for start, offset in zip(starts, offsets, strict=True)],
    )


def _freeze_reads(ctx: StreamContext) -> bool:
    """Whether frozen_frame_pct gets far enough on this stream to read its windows."""

    fps = _frame_rate(ctx)
    if fps is None or ctx.n_samples < 2 * _min_run(fps):
        return False
    return any(
        stream.taxonomy_type.startswith("action.")
        and stream.payload is not None
        and bool(stream.channels)
        for stream in ctx.episode_streams or []
    )


def _ranges(positions: list[int]) -> list[tuple[int, int]]:
    """Merge sorted frame positions into `[start, end)` runs of consecutive ones."""

    ranges: list[tuple[int, int]] = []
    for position in positions:
        if ranges and ranges[-1][1] == position:
            ranges[-1] = (ranges[-1][0], position + 1)
        else:
            ranges.append((position, position + 1))
    return ranges


def _frames(ctx: StreamContext) -> "_Frames | MetricResult":
    """Return what the vision metrics keep of the stream's frames, or `not_applicable`.

    Whichever vision metric asks first decodes for every one of them:
    the windows when frozen_frame_pct will read them,
    the sampled frames alone otherwise.
    The stream is checked on every call; only the read is cached,
    so two streams sharing one payload are each judged on their own type.
    """

    payload = _readable(ctx)
    if isinstance(payload, MetricResult):
        return payload

    windows, sampled = _layout(ctx, ctx.full_frame_scan)
    evidence = _layout(ctx, False)[1] if ctx.full_frame_scan else sampled
    if not _freeze_reads(ctx):
        windows = _ranges(sampled)
    spec = ctx.vision
    # A full scan is asked to read every frame, so no frame budget stops it.
    max_decode_frames = None if ctx.full_frame_scan else spec.max_decode_frames
    key = (
        id(payload),
        tuple(windows),
        tuple(sampled),
        tuple(evidence),
        spec.dark_level,
        spec.bright_level,
        max_decode_frames,
        spec.max_pixels,
    )
    if key in _CACHE and _CACHE[key][0] is payload:
        _CACHE.move_to_end(key)
        return _CACHE[key][1]

    result = _read(
        payload,
        windows,
        sampled,
        evidence,
        dark_level=spec.dark_level,
        bright_level=spec.bright_level,
        max_decode_frames=max_decode_frames,
        max_pixels=spec.max_pixels,
    )
    _CACHE[key] = (payload, result)
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return result


def _read(
    payload: SampledFrames,
    windows: list[tuple[int, int]],
    sampled: list[int],
    evidence: list[int],
    *,
    dark_level: int,
    bright_level: int,
    max_decode_frames: int | None,
    max_pixels: int,
) -> "_Frames | MetricResult":
    """Decode `windows` of `payload` once, keeping only what the metrics measure.

    A read stopped by a decode cap, or by a decode failure after its first frame,
    keeps what it read, with the reason in `stopped`.
    A failure before any frame decoded, such as a file that cannot be opened,
    is `not_applicable`.
    """

    numpy = load_numpy()
    assert numpy is not None

    wanted = set(sampled)
    evidence_set = set(evidence)
    kept: dict[str, list[float]] = {
        name: []
        for name in (
            "loss",
            "laplacian",
            "mean",
            "p5",
            "p95",
            "white_share",
            "dark_share",
        )
    }
    measured: list[int] = []
    pts: list[float | None] = []
    shapes: list[tuple[int, int]] = []
    hashes: list[str] = []
    pooled: dict[int, np.ndarray] = {}
    stopped: tuple[Availability, str] | None = None
    counts = [0] * len(windows)
    diffs: list[list[float]] = [[] for _ in windows]
    steps: list[list[float]] = [[] for _ in windows]
    previous: np.ndarray | None = None
    previous_position = -1
    previous_spread = 0.0
    levels = numpy.arange(256)
    try:
        with closing(
            payload.gray_windows(
                windows,
                _SAMPLE_SIZE,
                native=frozenset(sampled),
                max_decode_frames=max_decode_frames,
                max_pixels=max_pixels,
            )
        ) as frames:
            for position, frame, full, frame_pts in frames:
                # A window is only ever cut short at its end,
                # so the frames it did yield are its first ones.
                index = windows[position][0] + counts[position]
                counts[position] += 1

                histogram = numpy.bincount(frame.ravel(), minlength=256)
                cumulative = numpy.cumsum(histogram)
                p5, p95 = _percentile(cumulative, 5), _percentile(cumulative, 95)
                spread = p95 - p5
                if index in wanted:
                    assert full is not None
                    laplacian = _laplacian_var(full)
                    measured.append(index)
                    pts.append(frame_pts)
                    shapes.append((int(full.shape[0]), int(full.shape[1])))
                    hashes.append(hashlib.sha256(full.tobytes()).hexdigest())
                    kept["loss"].append(_reblur_loss(full, laplacian))
                    kept["laplacian"].append(laplacian)
                    kept["mean"].append(int(histogram @ levels) / frame.size)
                    kept["p5"].append(p5)
                    kept["p95"].append(p95)
                    # A pixel at or above the bright level is clipped white (rda),
                    # one at or below the dark level crushed black.
                    kept["white_share"].append(
                        int(histogram[bright_level:].sum()) / frame.size
                    )
                    kept["dark_share"].append(
                        int(histogram[: dark_level + 1].sum()) / frame.size
                    )

                # Only the frames a sample measures are pooled,
                # so a full scan's vector matches a sampled read's.
                if index in evidence_set:
                    pooled[index] = _thumbnail(frame, _VARIATION_SIZE) / 255

                thumbnail = _thumbnail(frame)
                if position == previous_position and previous is not None:
                    difference = float(numpy.abs(thumbnail - previous).mean())
                    steps[position].append(difference)
                    contrast = min(spread, previous_spread)
                    if contrast < _BLANK_SPREAD:
                        diffs[position].append(float("nan"))
                    else:
                        diffs[position].append(difference * _FREEZE_CONTRAST / contrast)
                previous, previous_position = thumbnail, position
                previous_spread = spread
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    except DecodeLimitReached as exc:
        stopped = (Availability.UNAVAILABLE, str(exc))
    except DecodeFailed as exc:
        # Only a file that started decoding was readable and then broke;
        # one that never yields a frame was never camera footage to read.
        if not sum(counts):
            return not_applicable(str(exc))
        stopped = (Availability.ERROR, str(exc))
    if not sum(counts) and stopped is None:
        return not_applicable("no frame could be decoded")

    return _Frames(
        windows=windows,
        n_read=sum(counts),
        sampled=measured,
        diffs=[
            (start, numpy.asarray(window_diffs, dtype=numpy.float32))
            for (start, _end), window_diffs in zip(windows, diffs, strict=True)
            if window_diffs
        ],
        luma_steps=[
            (start, numpy.asarray(window_steps, dtype=numpy.float32))
            for (start, _end), window_steps in zip(windows, steps, strict=True)
            if window_steps
        ],
        pts=pts,
        shape=shapes,
        luma_sha256=hashes,
        requested=list(sampled),
        evidence_positions=list(evidence),
        pooled=pooled,
        stopped=stopped,
        **kept,
    )


def _percentile(cumulative: "np.ndarray", q: float) -> float:
    """Read a percentile of a gray frame from its cumulative histogram.

    Parameters
    ----------
    cumulative : numpy.ndarray
        The `cumsum` of the frame's 256-bin gray-level histogram.
    q : float
        The percentile, in percent.

    Returns
    -------
    float
        The percentile `numpy.percentile` gives on the frame itself.
    """

    numpy = load_numpy()
    assert numpy is not None

    n = int(cumulative[-1])
    virtual = (n - 1) * (q / 100)
    below = math.floor(virtual)
    t = virtual - below
    a = int(numpy.searchsorted(cumulative, below, side="right"))
    b = int(numpy.searchsorted(cumulative, min(below + 1, n - 1), side="right"))
    if a == b:
        return float(a)
    # numpy's linear interpolation in its two branches,
    # so the result equals `numpy.percentile` to the bit.
    if t < 0.5:
        return a + (b - a) * t
    return b - (b - a) * (1 - t)


def _thumbnail(frame: "np.ndarray", size: int = _FREEZE_SIZE) -> "np.ndarray":
    """Average a `(_SAMPLE_SIZE, _SAMPLE_SIZE)` gray frame down to a thumbnail.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame as `gray_windows` yields it at `_SAMPLE_SIZE`.
    size : int
        The thumbnail's side, a divisor of `_SAMPLE_SIZE // 2`:
        `_FREEZE_SIZE` for freezes, `_VARIATION_SIZE` for footage variation.

    Returns
    -------
    numpy.ndarray
        The `(size, size)` float32 mean of each block of pixels.
    """

    numpy = load_numpy()
    assert numpy is not None

    wide = frame.astype(numpy.uint16)
    sums = wide[0::2, 0::2] + wide[1::2, 0::2] + wide[0::2, 1::2] + wide[1::2, 1::2]
    factor = sums.shape[0] // size
    assert factor * size == sums.shape[0], f"{size} does not divide {sums.shape[0]}"
    if factor > 1:
        sums = sums.reshape(size, factor, size, factor).sum(
            axis=(1, 3), dtype=numpy.uint32
        )
    # Block sums are integers well below 2**24, so their float32 value is exact
    # and the one float32 division equals the float32 block mean to the bit,
    # which freeze differences and footage vectors depend on.
    return sums.astype(numpy.float32) / numpy.float32((_SAMPLE_SIZE // size) ** 2)


def clear_cache() -> None:
    """Empty the frame, packet and header caches."""

    _CACHE.clear()
    _PACKETS.clear()
    _HEADERS.clear()


def _packets(payload: VideoPayload) -> SegmentPackets | DecodeFailed:
    """Probe the payload's packets once, keeping a failure as the result.

    Raises
    ------
    DecoderUnavailable
        If PyAV is not installed in this process.
    """

    entry = _PACKETS.get(id(payload))
    if entry is not None and entry[0] is payload:
        _PACKETS.move_to_end(id(payload))
        return entry[1]
    try:
        result: SegmentPackets | DecodeFailed = payload.packet_times()
    except DecodeFailed as exc:
        result = exc
    _PACKETS[id(payload)] = (payload, result)
    while len(_PACKETS) > _CACHE_SIZE:
        _PACKETS.popitem(last=False)
    return result


def _declared_fps(payload: VideoPayload, packets: SegmentPackets) -> float:
    """Return the segment's declared frame rate, or the container's without one.

    Parameters
    ----------
    payload : VideoPayload
        The segment, with its declared frame count and bounds.
    packets : SegmentPackets
        The segment's probed packets, whose rate is the fallback.

    Returns
    -------
    float
        Frames per second.
    """

    # The container's average rate falls with every dropped frame,
    # so a steady drop would widen its own threshold;
    # the dataset's declared rate does not move.
    span = payload.end_s - payload.start_s
    return (
        payload.frame_count / span
        if payload.frame_count > 0 and span > 0
        else packets.fps
    )


def _header(payload: VideoPayload) -> ContainerHeader | DecodeFailed:
    """Read the payload's container header once, keeping a failure as the result.

    A `ReadInterrupted` is returned but not kept, so the next metric reads again.

    Parameters
    ----------
    payload : VideoPayload
        The video stream whose header to read.

    Returns
    -------
    ContainerHeader or DecodeFailed
        The header, or the failure that stopped the read.

    Raises
    ------
    DecoderUnavailable
        If PyAV is not installed in this process.
    """

    entry = _HEADERS.get(id(payload))
    if entry is not None and entry[0] is payload:
        _HEADERS.move_to_end(id(payload))
        return entry[1]
    try:
        result: ContainerHeader | DecodeFailed = payload.header()
    except ReadInterrupted as exc:
        return exc
    except DecodeFailed as exc:
        result = exc
    _HEADERS[id(payload)] = (payload, result)
    while len(_HEADERS) > _CACHE_SIZE:
        _HEADERS.popitem(last=False)
    return result


def _video_file(ctx: StreamContext) -> VideoPayload | MetricResult:
    """Return the video payload behind the stream, or `not_applicable` if none.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to check.

    Returns
    -------
    VideoPayload or MetricResult
        The payload, or `not_applicable` when the stream is not camera footage,
        its frames do not come from a video file, or it carries no timestamps.
    """

    payload = _camera_payload(ctx)
    if isinstance(payload, MetricResult):
        return payload
    if not isinstance(payload, VideoPayload):
        return not_applicable("the stream's frames do not come from a video file")
    if payload.frame_count == 0:
        return not_applicable("the stream carries no timestamps")
    return payload


def _laplacian_var(frame: "np.ndarray") -> float:
    """Variance of the 3x3 Laplacian over the whole frame."""

    image = frame.astype("float64")
    laplacian = (
        image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
        - 4 * image[1:-1, 1:-1]
    )
    return float(laplacian.var())


def _reblur_loss(frame: "np.ndarray", laplacian: float) -> float:
    """Share of `laplacian` a `_REBLUR` box blur of the frame removes.

    0.0 for a frame with no edges.
    """

    if laplacian == 0:
        return 0.0
    numpy = load_numpy()
    assert numpy is not None

    height, width = frame.shape
    pad = _REBLUR // 2
    padded = numpy.pad(frame.astype("float64"), pad, mode="edge")
    blurred = numpy.zeros((height, width))
    for dy in range(_REBLUR):
        for dx in range(_REBLUR):
            blurred += padded[dy : dy + height, dx : dx + width]
    blurred /= _REBLUR * _REBLUR
    return 1 - _laplacian_var(blurred) / laplacian


def _stopped_result(stopped: tuple[Availability, str]) -> MetricResult:
    """The ungraded result of a read that stopped before it finished."""

    availability, reason = stopped
    return MetricResult(
        value=None,
        unit=None,
        status=MetricStatus.NOT_APPLICABLE,
        availability=availability,
        evidence={"reason": reason},
    )


def _exposure(frames: _Frames) -> _Exposure:
    """Judge each sampled frame against the episode's typical one.

    The rule exposure_shift_pct grades, shared with `camera_frames`.
    """

    numpy = load_numpy()
    assert numpy is not None

    spreads = [p95 - p5 for p5, p95 in zip(frames.p5, frames.p95, strict=True)]
    typical_mean = float(numpy.median(frames.mean))
    typical_white = float(numpy.median(frames.white_share))
    shift = max(_SHIFT_SHARE * typical_mean, _SHIFT_FLOOR)

    counts = {"darker": 0, "brighter": 0, "clipped": 0, "blank": 0}
    bad: list[int] = []
    worst_index: int | None = None
    worst_excess = -1.0
    for index, mean, spread, white_share in zip(
        frames.sampled, frames.mean, spreads, frames.white_share, strict=True
    ):
        # Each rule's distance past its threshold, relative to the threshold,
        # so the rules can be compared when picking the worst frame.
        excess = {
            "darker": (typical_mean - mean - shift) / shift,
            "brighter": (mean - typical_mean - shift) / shift,
            "clipped": (white_share - typical_white - _CLIP_RISE) / _CLIP_RISE,
            "blank": (_BLANK_SPREAD - spread) / _BLANK_SPREAD,
        }
        broken = {rule: value for rule, value in excess.items() if value > 0}
        for rule in broken:
            counts[rule] += 1
        if broken:
            bad.append(int(index))
            if max(broken.values()) > worst_excess:
                worst_excess = max(broken.values())
                worst_index = int(index)
    return _Exposure(
        counts=counts,
        bad=bad,
        worst_index=worst_index,
        typical_mean=typical_mean,
        typical_white=typical_white,
    )


@metric(level=Level.STREAM, family=Family.VISION, label="low sharpness")
def sharpness_score(ctx: StreamContext) -> MetricResult:
    """Median share of the sampled frames' edge energy a 3x3 box re-blur removes.

    Higher is sharper.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only`, or `not_applicable` when the stream cannot be sampled.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")
    numpy = load_numpy()
    assert numpy is not None

    losses = frames.loss
    worst = min(range(len(losses)), key=losses.__getitem__)
    return MetricResult(
        value=float(numpy.median(losses)),
        unit="fraction",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_sampled": len(losses),
            "n_frames": len(ctx.timestamps),
            "laplacian_var": float(numpy.median(frames.laplacian)),
            "min_loss": float(losses[worst]),
            "worst_index": int(frames.sampled[worst]),
        },
    )


@metric(level=Level.STREAM, family=Family.VISION, label="brightness changes")
def exposure_shift_pct(ctx: StreamContext) -> MetricResult:
    """Share of sampled frames exposed unlike the episode's typical frame, or blank.

    The typical frame is the median, over the sampled frames, of each frame's
    mean gray level and share of pixels at or above `VisionSpec.bright_level`.
    A frame is bad when it is:

    - darker or brighter: its mean moves from the typical mean
      by more than `_SHIFT_SHARE` of it and at least `_SHIFT_FLOOR` levels;
    - clipped: its white share exceeds the typical one by more than `_CLIP_RISE`;
    - blank: its P95 - P5 spread is below `_BLANK_SPREAD`, whatever the camera.

    A camera exposed the same way in most sampled frames is its own typical frame,
    so it reads clean whether it films a white simulator scene
    or is overexposed throughout; only blank frames fail on their own.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        The share in percent, with each bad frame as a support interval,
        or `not_applicable` when the stream cannot be sampled.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")
    exposure = _exposure(frames)
    n_frames = len(ctx.timestamps)
    return MetricResult(
        value=float(100 * len(exposure.bad) / len(frames.sampled)),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_sampled": len(frames.sampled),
            "n_frames": n_frames,
            **exposure.counts,
            "typical_mean": exposure.typical_mean,
            "typical_white_share": exposure.typical_white,
            "worst_index": exposure.worst_index,
            "bad_indices": exposure.bad,
        },
        support=support_for(
            list(range(n_frames)), [(index, index + 1) for index in exposure.bad]
        ),
    )


@metric(level=Level.STREAM, family=Family.VISION, label="exposure level")
def exposure_level(ctx: StreamContext) -> MetricResult:
    """Median mean gray of the sampled frames, 0 to 255; reported, never graded.

    It describes the episode by itself:
    a white simulator scene and an overexposed camera read alike here,
    and `Report.cameras` compares it across the camera's episodes.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only`, or `not_applicable` when the stream cannot be sampled.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")
    numpy = load_numpy()
    assert numpy is not None

    return MetricResult(
        value=float(numpy.median(frames.mean)),
        unit="gray level",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_sampled": len(frames.sampled),
            "n_frames": len(ctx.timestamps),
            "dark_share": float(numpy.median(frames.dark_share)),
            "bright_share": float(numpy.median(frames.white_share)),
        },
    )


def _frame_pct(
    ctx: StreamContext, frames: _Frames, bad: list[int], **evidence: object
) -> MetricResult:
    """The report-only share of sampled frames that break a per-frame rule.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream measured.
    frames : _Frames
        The stream's shared read, with at least one sampled frame.
    bad : list of int
        The positions of the sampled frames that break the rule.
    **evidence : object
        Further evidence the metric publishes beside the counts.

    Returns
    -------
    MetricResult
        The share in percent, with each bad frame as a support interval.
    """

    n_frames = len(ctx.timestamps)
    return MetricResult(
        value=float(100 * len(bad) / len(frames.sampled)),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_sampled": len(frames.sampled),
            "n_frames": n_frames,
            "bad_indices": bad,
            **evidence,
        },
        support=support_for(
            list(range(n_frames)), [(index, index + 1) for index in bad]
        ),
    )


@metric(level=Level.STREAM, family=Family.VISION, label="blank frames")
def blank_frame_pct(ctx: StreamContext) -> MetricResult:
    """Share of sampled frames whose P95 - P5 spread is below `_BLANK_SPREAD`.

    exposure_shift_pct counts these frames too, among its other rules;
    this metric counts them alone, so a stream blank throughout reads 100.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only`, the share in percent with each blank frame as a support
        interval, or `not_applicable` when the stream cannot be sampled.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")

    spreads = [p95 - p5 for p5, p95 in zip(frames.p5, frames.p95, strict=True)]
    bad = [
        int(index)
        for index, spread in zip(frames.sampled, spreads, strict=True)
        if spread < _BLANK_SPREAD
    ]
    return _frame_pct(ctx, frames, bad, min_spread=float(min(spreads)))


@metric(level=Level.STREAM, family=Family.VISION, label="low contrast")
def contrast_score(ctx: StreamContext) -> MetricResult:
    """Median P95 - P5 spread of the sampled frames, 0 to 255 gray levels.

    Lower is flatter. It follows what is in view as much as the camera,
    so it is reported and not graded.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only`, or `not_applicable` when the stream cannot be sampled.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")
    numpy = load_numpy()
    assert numpy is not None

    spreads = [p95 - p5 for p5, p95 in zip(frames.p5, frames.p95, strict=True)]
    worst = min(range(len(spreads)), key=spreads.__getitem__)
    return MetricResult(
        value=float(numpy.median(spreads)),
        unit="gray level",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_sampled": len(frames.sampled),
            "n_frames": len(ctx.timestamps),
            "min_spread": float(spreads[worst]),
            "worst_index": int(frames.sampled[worst]),
        },
    )


@metric(level=Level.STREAM, family=Family.VISION, label="dark frames")
def dark_frame_pct(ctx: StreamContext) -> MetricResult:
    """Share of sampled frames that are crushed or dim.

    A frame is crushed when at least `_EXPOSED_SHARE` of its pixels are
    at or below `VisionSpec.dark_level`,
    and dim when its P95 is at or below `_DIM_P95`.
    Unlike exposure_shift_pct, it judges each frame by itself,
    so a camera dark or dim throughout reads 100 here,
    and clean there unless its frames are blank.
    A deliberately dark scene reads 100 too, as a white simulator scene does
    on bright_frame_pct, so it is reported and not graded.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only`, the share in percent with each dark frame as a support
        interval, or `not_applicable` when the stream cannot be sampled.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")

    bad = [
        int(index)
        for index, share, p95 in zip(
            frames.sampled, frames.dark_share, frames.p95, strict=True
        )
        if share >= _EXPOSED_SHARE or p95 <= _DIM_P95
    ]
    return _frame_pct(
        ctx, frames, bad, dark_level=ctx.vision.dark_level, dim_p95=_DIM_P95
    )


@metric(level=Level.STREAM, family=Family.VISION, label="bright frames")
def bright_frame_pct(ctx: StreamContext) -> MetricResult:
    """Share of sampled frames with at least `_EXPOSED_SHARE` of pixels clipped.

    A pixel is clipped at or above `VisionSpec.bright_level`.
    It judges each frame by itself, so a white simulator scene reads 100
    just as a camera blown out throughout does,
    which is why it is reported and not graded.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only`, the share in percent with each bright frame as a support
        interval, or `not_applicable` when the stream cannot be sampled.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")

    bad = [
        int(index)
        for index, share in zip(frames.sampled, frames.white_share, strict=True)
        if share >= _EXPOSED_SHARE
    ]
    return _frame_pct(ctx, frames, bad, bright_level=ctx.vision.bright_level)


@metric(level=Level.STREAM, family=Family.VISION, label="frame-count mismatch")
def frame_count_vs_timebase(ctx: StreamContext) -> MetricResult:
    """Deviation between the frames the video holds and the frames its timestamps imply.

    `present` counts the container's packets in the segment, decoding nothing,
    under a full scan too: a decode is bounded by the timestamps,
    so it could never find a frame past them.
    On a whole-file payload the count runs to the end of the file,
    so surplus frames past the timestamps show.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only` with `|present - implied| / implied`,
        or `not_applicable` when the stream is not backed by a video file.
    """

    payload = _camera_payload(ctx)
    if isinstance(payload, MetricResult):
        return payload
    # An in-memory frame list has no container to disagree with its timestamps.
    if not isinstance(payload, VideoPayload):
        return not_applicable("the stream's frames do not come from a video file")
    implied = len(ctx.timestamps)
    if implied == 0:
        return not_applicable("the stream carries no timestamps")

    try:
        packets = _packets(payload)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(packets, DecodeFailed):
        return not_applicable(str(packets))
    present = len(packets.pts)

    return MetricResult(
        value=float(abs(present - implied) / implied),
        unit="fraction",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "present": int(present),
            "implied": int(implied),
            "truncated": bool(present < implied),
        },
    )


@metric(level=Level.STREAM, family=Family.VISION, label="frame timestamp defects")
def pts_defect_pct(ctx: StreamContext) -> MetricResult:
    """Share of presentation intervals that repeat a timestamp or skip a frame.

    The intervals are the steps between the segment's sorted packet timestamps.
    A step of zero is a duplicate;
    one longer than `_GAP_PERIODS` declared frame periods is a gap.

    Returns
    -------
    MetricResult
        `report_only` with the percentage of defective intervals,
        or `not_applicable` when there is no readable video file
        or fewer than two frames.
    """

    payload = _camera_payload(ctx)
    if isinstance(payload, MetricResult):
        return payload
    if not isinstance(payload, VideoPayload):
        return not_applicable("the stream's frames do not come from a video file")
    numpy = load_numpy()
    if numpy is None:
        return not_applicable("spacing frames needs numpy; add kalanos[video]")
    try:
        packets = _packets(payload)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(packets, DecodeFailed):
        return not_applicable(str(packets))
    if len(packets.pts) < 2:
        return not_applicable("fewer than two frames to space")

    fps = _declared_fps(payload, packets)
    steps = numpy.diff(numpy.asarray(packets.pts))
    duplicates = int((steps == 0).sum())
    gaps = int((steps * packets.time_base > _GAP_PERIODS / fps).sum())
    return MetricResult(
        value=100 * (duplicates + gaps) / len(steps),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "intervals": len(steps),
            "duplicates": duplicates,
            "gaps": gaps,
            "period_s": 1 / fps,
        },
    )


@metric(level=Level.STREAM, family=Family.VISION, label="longest frame gap")
def max_gap_s(ctx: StreamContext) -> MetricResult:
    """Longest step between the segment's sorted packet timestamps, in seconds.

    Only steps between packets count:
    the segment's edges come from metadata, so a hole there may be rounding.

    Returns
    -------
    MetricResult
        `report_only` with the longest step and where it sits,
        or `not_applicable` when there is no readable video file
        or fewer than two frames.
    """

    payload = _camera_payload(ctx)
    if isinstance(payload, MetricResult):
        return payload
    if not isinstance(payload, VideoPayload):
        return not_applicable("the stream's frames do not come from a video file")
    numpy = load_numpy()
    if numpy is None:
        return not_applicable("spacing frames needs numpy; add kalanos[video]")
    try:
        packets = _packets(payload)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(packets, DecodeFailed):
        return not_applicable(str(packets))
    if len(packets.pts) < 2:
        return not_applicable("fewer than two frames to space")

    period = 1 / _declared_fps(payload, packets)
    pts = numpy.asarray(packets.pts)
    steps = numpy.diff(pts)
    index = int(numpy.argmax(steps))
    gap = float(steps[index] * packets.time_base)
    # The bounds sit half a frame early, so the first packet can precede start_s;
    # a gap is never placed before its segment begins.
    gap_start = max(0.0, float(pts[index] * packets.time_base - payload.start_s))
    return MetricResult(
        value=gap,
        unit="s",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "after_row": index,
            "gap_start_s": gap_start,
            "gap_end_s": float(pts[index + 1] * packets.time_base - payload.start_s),
            "period_s": period,
            "gap_periods": gap / period,
            "intervals": len(steps),
        },
    )


@metric(level=Level.EPISODE, family=Family.VISION, label="camera timestamp offset")
def camera_sync_skew_ms(ctx: EpisodeContext) -> MetricResult:
    """Spread of the episode's cameras' offsets against their recorded timestamps.

    A camera's offset is the median, over frames paired by index,
    of its packet presentation time less its recorded timestamp.
    Three kinds of camera are set aside under `skipped`:
    - a segment of a file holding several episodes:
      its packets are chosen by its recorded start,
      so they sit on its timestamps to within half a frame whatever happened;
    - a camera whose paired frames hold a duplicate or a gap:
      every pair after it is off by a frame period,
      and `pts_defect_pct` and `max_gap_s` report the frames themselves;
    - a camera whose paired frames hold a null or non-finite recorded timestamp:
      that frame has no offset to read.

    Parameters
    ----------
    ctx : EpisodeContext
        The recording whose cameras are compared.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the metadata tier reads no frames
        - numpy is not installed
        - PyAV is not installed
        - fewer than two cameras have an offset that can be read
        `report_only` otherwise, with the largest offset less the smallest.
    """

    if not reads_numeric_payloads():
        return not_applicable("metadata tier does not read frames")
    numpy = load_numpy()
    if numpy is None:
        return not_applicable("spacing frames needs numpy; add kalanos[video]")

    # Step 1: pick the cameras whose frames come from a video file in source order
    candidates = [
        (stream, stream.payload)
        for stream in ctx.streams
        if is_camera_footage(stream.kind.value, stream.taxonomy_type)
        and isinstance(stream.payload, VideoPayload)
        and stream.source_order.preserved
        and stream.source_order.original_index is None
    ]

    # Step 2: offset each camera's packets against its own recorded timestamps
    cameras: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    offsets: list[float] = []
    for stream, payload in candidates:
        name = stream.source_field or stream.taxonomy_type
        if not payload.whole_file:
            skipped.append(
                {
                    "stream": name,
                    "reason": "its file holds several episodes, "
                    "so the segment is placed by its recorded start",
                }
            )
            continue
        try:
            packets = _packets(payload)
        except DecoderUnavailable as exc:
            return not_applicable(str(exc))
        if isinstance(packets, DecodeFailed):
            skipped.append({"stream": name, "reason": str(packets)})
            continue
        n = min(len(packets.pts), len(stream.timestamps))
        if n < 1:
            skipped.append({"stream": name, "reason": "no frame to pair"})
            continue
        pts = numpy.asarray(packets.pts[:n], dtype="float64")
        steps = numpy.diff(pts) * packets.time_base
        if (
            (steps == 0) | (steps > _GAP_PERIODS / _declared_fps(payload, packets))
        ).any():
            skipped.append(
                {
                    "stream": name,
                    "reason": "a repeated or missing frame shifts pairing by index",
                }
            )
            continue
        presented = pts * packets.time_base - payload.start_s
        # A null reads back as NaN, and one NaN turns the median and the spread NaN.
        recorded = stream.timestamps.head(n).cast(pl.Float64).to_numpy()
        if not numpy.isfinite(recorded).all():
            skipped.append(
                {
                    "stream": name,
                    "reason": "a recorded timestamp is null or not finite",
                }
            )
            continue
        offset = float(numpy.median(presented - recorded))
        offsets.append(offset)
        info = stream.clock_info or ClockInfo.from_legacy(stream.clock)
        cameras.append(
            {
                "stream": name,
                "offset_ms": offset * 1000,
                "frames_paired": n,
                "clock_origin": info.origin.value,
            }
        )
    if len(offsets) < 2:
        reasons = sorted({skip["reason"] for skip in skipped})
        return not_applicable(
            "fewer than two cameras have an offset that can be read"
            + "".join(f"; {reason}" for reason in reasons)
        )

    # Step 3: compare the cameras
    return MetricResult(
        value=(max(offsets) - min(offsets)) * 1000,
        unit="ms",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "cameras": cameras,
            "skipped": skipped,
            "timestamp_basis": "packet presentation time",
            "interpretation": (
                "where each video file's frames sit against the recorded "
                "timestamps; capture synchronisation is not measured"
            ),
        },
    )


@metric(
    level=Level.STREAM, family=Family.VISION, label="camera file missing or unreadable"
)
def camera_stream_present(ctx: StreamContext) -> MetricResult:
    """Whether a declared camera's file opens and yields this episode's first frame.

    Reads the container header, then decodes up to the first frame the segment holds,
    discarding the seek preroll before it.
    That frame need not sit exactly at the segment's start,
    so a frame the encoder dropped there does not read as a missing camera.
    A decode failure after that frame is not this metric's to judge:
    the sampled metrics report it, and the episode stays unknown.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to check.

    Returns
    -------
    MetricResult
        `report_only` with 1.0 when the frame decoded and 0.0 when it did not,
        for the policy to regrade,
        or `not_applicable` when the stream is not backed by a video file.
    """

    payload = _video_file(ctx)
    if isinstance(payload, MetricResult):
        return payload
    if load_numpy() is None:
        return not_applicable("sampling frames needs numpy; add kalanos[video]")

    reason: str | None = None
    try:
        header = _header(payload)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    # A failure outside the file, such as on the network, is no evidence it is absent.
    if isinstance(header, ReadInterrupted):
        return _stopped_result((Availability.UNAVAILABLE, str(header)))
    if isinstance(header, DecodeFailed):
        reason = str(header)
    else:
        try:
            # The whole segment as one window, closed after its first frame:
            # the generator decodes nothing past the frame it yields.
            with closing(
                payload.gray_windows(
                    [(0, payload.frame_count)],
                    _SAMPLE_SIZE,
                    max_decode_frames=ctx.vision.max_decode_frames,
                    max_pixels=ctx.vision.max_pixels,
                )
            ) as frames:
                decoded = next(frames, None) is not None
        except (DecodeLimitReached, ReadInterrupted) as exc:
            return _stopped_result((Availability.UNAVAILABLE, str(exc)))
        except DecodeFailed as exc:
            decoded, reason = False, str(exc)
        if not decoded and reason is None:
            reason = "no frame could be decoded"

    return MetricResult(
        value=0.0 if reason is not None else 1.0,
        unit="flag",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "camera": ctx.stream.source_field,
            "path": str(payload.path),
            "reason": reason,
        },
    )


def _fps_differs(found: float, declared: float) -> bool:
    """Whether a container's frame rate differs from the declared one.

    They differ when further apart than `_FPS_REL_TOL` of the declared rate.

    Parameters
    ----------
    found : float
        The rate the container states.
    declared : float
        The rate the dataset declares.

    Returns
    -------
    bool
        Whether the two rates differ.
    """

    return abs(found - declared) > _FPS_REL_TOL * declared


@metric(
    level=Level.STREAM,
    family=Family.VISION,
    label="video format differs from declaration",
)
def video_format_consistency(ctx: StreamContext) -> MetricResult:
    """How many declared format fields the container header contradicts.

    Compares width, height, frame rate and pixel format, decoding nothing.
    A field the dataset does not declare is listed under `undeclared` and skipped,
    and one the container does not state is listed under `unstated` and skipped:
    a missing value contradicts nothing.
    A file that cannot be opened is `camera_stream_present`'s to report.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to check.

    Returns
    -------
    MetricResult
        `report_only` with the count of differing fields,
        or `not_applicable` when the stream is not backed by a readable video file
        or the dataset declares no format for it.
    """

    payload = _video_file(ctx)
    if isinstance(payload, MetricResult):
        return payload
    declared = payload.declared
    if declared is None:
        return not_applicable("the dataset declares no video format")
    try:
        header = _header(payload)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(header, DecodeFailed):
        return not_applicable(str(header))

    differs: list[dict[str, int | float | str | None]] = []
    undeclared: list[str] = []
    unstated: list[str] = []
    for field in ("width", "height", "fps", "pix_fmt"):
        expected = getattr(declared, field)
        found = getattr(header, field)
        if expected is None:
            undeclared.append(field)
            continue
        if found is None:
            unstated.append(field)
            continue
        if field == "fps":
            mismatch = _fps_differs(found, expected)
        else:
            mismatch = found != expected
        if mismatch:
            differs.append({"field": field, "declared": expected, "found": found})

    return MetricResult(
        value=float(len(differs)),
        unit="fields",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "differs": differs,
            "undeclared": undeclared,
            "unstated": unstated,
            "camera": ctx.stream.source_field,
        },
    )


def _action_deltas(ctx: StreamContext) -> "list[tuple[np.ndarray, np.ndarray]]":
    """Return `(step_times, moving_mask)` for each usable action stream beside ctx."""

    numpy = load_numpy()
    assert numpy is not None

    actions = []
    for stream in ctx.episode_streams or []:
        if not stream.taxonomy_type.startswith("action."):
            continue
        if stream.payload is None or not stream.channels:
            continue
        # An action payload may be lazy, so this reads it a second time;
        # grade_stream fetches it again for the stream's own channels.
        frame = stream.payload.fetch()
        if not isinstance(frame, pl.DataFrame):
            continue
        names = [
            channel.name
            for channel in stream.channels
            if channel.name in frame.columns and frame.schema[channel.name].is_numeric()
        ]
        if not names:
            continue
        values = frame.select(names).cast(pl.Float64).to_numpy()
        if len(values) < 2:
            continue

        # The mean over each step's finite readings, NaN where there are none,
        # without nanmean's warning on an all-NaN step.
        steps = numpy.abs(numpy.diff(values, axis=0))
        finite = numpy.isfinite(steps)
        # A stream with no readings cannot say whether the robot moved.
        if not finite.any():
            continue
        readings = finite.sum(axis=1)
        totals = numpy.where(finite, steps, 0.0).sum(axis=1)
        deltas = numpy.where(
            readings > 0, totals / numpy.maximum(readings, 1), numpy.nan
        )
        scale = float(numpy.percentile(deltas[readings > 0], _MOVING_PCT))
        if scale == 0:
            scale = 1.0
        # NaN compares False, so a step with no reading never moves.
        moving = deltas > _MOVING_SHARE * scale
        actions.append((stream.timestamps.to_numpy()[1:], moving))
    return actions


def _moves_over(
    actions: "list[tuple[np.ndarray, np.ndarray]]", t_start: float, t_end: float
) -> bool:
    """Whether any action stream moves over most of `[t_start, t_end]`."""

    numpy = load_numpy()
    assert numpy is not None

    for step_times, moving in actions:
        inside = (step_times >= t_start) & (step_times <= t_end)
        if inside.any():
            share = float(moving[inside].mean())
        else:
            nearest = int(numpy.argmin(numpy.abs(step_times - (t_start + t_end) / 2)))
            share = float(moving[nearest])
        if share > _MOVING_MAJORITY:
            return True
    return False


@metric(level=Level.STREAM, family=Family.VISION, label="repeated frames")
def frozen_frame_pct(ctx: StreamContext) -> MetricResult:
    """Share of frames read that repeat their predecessor while the robot moves.

    A freeze is a run of at least 0.5 s of frame pairs whose 64x64 gray
    mean absolute difference is below max(0.10, 0.25 x the 10th percentile
    of the differences read).
    Each difference is scaled to a contrast of 255 by the lower P95 - P5 of the pair,
    and a pair with a blank frame is never frozen.
    It counts only while an action stream of the same episode moves over it:
    an idle robot in a still scene looks frozen too.
    It reads the windows `_layout` picks,
    in the decode it shares with blur and exposure.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure, with its episode's streams.

    Returns
    -------
    MetricResult
        `report_only` in percent, with each counted run as a support interval,
        or `not_applicable` when the stream cannot be sampled,
        has no action stream beside it, or holds too few frames to show a freeze.
    """

    # Step 1: the camera, and the actions that tell a freeze from an idle robot.
    payload = _readable(ctx)
    if isinstance(payload, MetricResult):
        return payload
    numpy = load_numpy()
    assert numpy is not None
    actions = _action_deltas(ctx)
    if not actions:
        return not_applicable(
            "no action stream to tell a frozen camera from an idle robot"
        )

    # Step 2: the frame rate sets how many repeats make a freeze.
    fps = _frame_rate(ctx)
    if fps is None:
        return not_applicable("the stream has no frame rate to measure a freeze in")
    min_run = _min_run(fps)
    timestamps = ctx.timestamps.to_numpy()
    n = len(timestamps)
    if n < 2 * min_run:
        return not_applicable("too few frames to tell a freeze from a still scene")

    # Step 3: read the windows shared with blur and exposure.
    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.diffs:
        return not_applicable("too few frames decoded to compare neighbours")

    # Step 4: one tolerance, pooled over every window's non-blank differences.
    pooled = numpy.concatenate([window_diffs for _start, window_diffs in frames.diffs])
    n_blank = int(numpy.isnan(pooled).sum())
    pooled = pooled[numpy.isfinite(pooled)]
    if not pooled.size:
        return not_applicable("every frame pair read holds a blank frame")
    eps = max(
        _FREEZE_EPS_FLOOR,
        _FREEZE_EPS_SHARE * float(numpy.percentile(pooled, _FREEZE_NOISE_PCT)),
    )

    # Step 5: find each window's frozen runs, and keep the ones the robot moves over.
    # Transition j of a window starting at s lies between frames s + j and s + j + 1.
    counted: list[tuple[int, int]] = []
    idle = 0
    for start, window_diffs in frames.diffs:
        frozen = numpy.concatenate(([False], window_diffs < eps, [False]))
        edges = numpy.flatnonzero(frozen[1:] != frozen[:-1])
        for j0, j1_end in zip(edges[::2], edges[1::2], strict=True):
            run = int(j1_end - j0)
            if run < min_run:
                continue
            t_start = float(timestamps[start + j0])
            t_end = float(timestamps[min(start + j1_end, n - 1)])
            if _moves_over(actions, t_start, t_end):
                counted.append((start + int(j0) + 1, run))
            else:
                idle += 1

    read = frames.n_read
    repeated = sum(run for _index, run in counted)
    longest = max((run for _index, run in counted), default=0)
    return MetricResult(
        value=float(100 * repeated / read),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_frames": n,
            "n_read": read,
            "n_windows": len(frames.windows),
            "full_scan": bool(frames.windows == [(0, n)]),
            "n_runs": len(counted),
            "n_idle_runs": idle,
            "longest_run_s": float(longest / fps),
            "first_run_index": counted[0][0] if counted else None,
            "eps": float(eps),
            "n_blank_pairs": n_blank,
        },
        support=support_for(
            list(range(n)), [(index, index + run) for index, run in counted]
        ),
    )


def camera_format(stream: Stream, *, reads_payloads: bool) -> CameraFormat | None:
    """Publish the camera's declared and container format, whether or not they agree.

    The container header read is shared with, and cached for, the vision metrics.
    The declared format comes from the adapter and reads no media,
    so the metadata tier states it too.

    Parameters
    ----------
    stream : Stream
        The camera stream to describe, with the payload its adapter built.
    reads_payloads : bool
        Whether the execution tier lets the container be opened.

    Returns
    -------
    CameraFormat or None
        Both formats, with `found` all `None` when `reads_payloads` is false,
        without PyAV, or when the header cannot be read.
        `None` when the stream's frames do not come from a video file.
    """

    payload = stream.payload
    if not isinstance(payload, VideoPayload):
        return None
    header: ContainerHeader | DecodeFailed | None = None
    if reads_payloads:
        # Publishing evidence must not abort the grade, whatever the read raised.
        try:
            header = _header(payload)
        except DecoderUnavailable:
            pass
        except Exception:
            logger.warning(
                "format read of %s failed", stream.source_path, exc_info=True
            )
            return None
    declared = payload.declared or DeclaredFormat()
    found = header if isinstance(header, ContainerHeader) else None
    # Copying every field keeps one added to the dataclasses from being dropped here;
    # their field names must match VideoFormat's.
    return CameraFormat(
        declared=VideoFormat(**asdict(declared)),
        found=VideoFormat() if found is None else VideoFormat(**asdict(found)),
    )


def camera_frames(ctx: StreamContext) -> CameraFrames | None:
    """Publish the stream's shared read as evidence.

    Evidence frames, the ones that carry a hash,
    are those a sampled read would measure and exposure_shift_pct's bad frames.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream the vision metrics read.

    Returns
    -------
    CameraFrames or None
        The read's rows and per-frame evidence,
        or `None` when the stream was not read as camera footage.
    """

    # A read that raised anything but a decode error has already failed each
    # metric through the registry's guard; here it must not abort the grade.
    try:
        frames = _frames(ctx)
    except Exception:
        logger.warning(
            "camera read of %s failed", ctx.stream.source_path, exc_info=True
        )
        return None
    if isinstance(frames, MetricResult):
        return None

    bad = _exposure(frames).bad if frames.sampled else []
    evidence = set(frames.evidence_positions) | set(bad)
    records = [
        CameraFrame(
            source_row=index,
            presentation_time_s=pts,
            shape=list(shape),
            blur_score=laplacian,
            clipped_fraction=dark + white,
            luma_sha256=digest if index in evidence else None,
        )
        for index, pts, shape, laplacian, dark, white, digest in zip(
            frames.sampled,
            frames.pts,
            frames.shape,
            frames.laplacian,
            frames.dark_share,
            frames.white_share,
            frames.luma_sha256,
            strict=True,
        )
    ]
    # Transition j of a window starting at s lies between frames s + j and s + j + 1.
    identical = [
        [start + j, start + j + 1]
        for start, window_diffs in frames.diffs
        for j, difference in enumerate(window_diffs.tolist())
        if difference == 0.0
    ]
    availability, reason = frames.stopped or (Availability.COMPUTED, None)
    return CameraFrames(
        requested_rows=frames.requested,
        missing_rows=sorted(set(frames.requested) - set(frames.sampled)),
        frames=records,
        decoded_frames=frames.n_read,
        adjacent_pairs_examined=sum(len(d) for _start, d in frames.diffs),
        identical_adjacent_pairs=identical,
        sample_plan="full_scan_v1" if ctx.full_frame_scan else "evenly_spaced_v1",
        parameters={
            **ctx.vision.model_dump(),
            "sample_frames": ctx.vision_samples,
            "full_frame_scan": ctx.full_frame_scan,
        },
        availability=availability,
        reason=reason,
    )


def footage_vector(ctx: StreamContext) -> "np.ndarray | None":
    """Return the stream's sampled frames, pooled, as one vector.

    Called after `camera_frames` on the same stream,
    it reads the cache that filled and decodes nothing more.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream the vision metrics read.

    Returns
    -------
    numpy.ndarray or None
        Each frame a sampled read measures, in order,
        as its `_VARIATION_SIZE` block mean on a 0-1 scale,
        flattened to one float32 vector;
        `None` when the stream was not read as camera footage,
        or the read missed any of those frames.
    """

    # `camera_frames` has already logged the same failure.
    try:
        frames = _frames(ctx)
    except Exception:
        return None
    if isinstance(frames, MetricResult):
        return None
    if any(index not in frames.pooled for index in frames.evidence_positions):
        return None

    numpy = load_numpy()
    assert numpy is not None

    return numpy.concatenate(
        [frames.pooled[index].ravel() for index in frames.evidence_positions]
    ).astype(numpy.float32)
