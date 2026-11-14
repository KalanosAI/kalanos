"""The vision family.

Seventeen metrics measure camera footage:

- sharpness_score, exposure_shift_pct and exposure_level measure
  `StreamContext.vision_samples` evenly spaced frames of a camera stream
  (10 by default);
- blank_frame_pct, contrast_score, dark_frame_pct and bright_frame_pct
  read the statistics blur and exposure keep of those same frames,
  decoding nothing more, and the default policy reports them without a band;
- corrupt_frame_pct counts the sampled frames that are blank or decoder noise,
  from each one's native luminance, and blocks through the contract route;
- frozen_frame_pct reads that many windows of about 1 s of consecutive frames,
  and needs an action stream in the same episode;
- still_scene_pct reads the change of every frame against the frame 1 s earlier,
  under a full scan only;
- frame_count_vs_timebase, pts_defect_pct and max_gap_s share one packet probe,
  decoding nothing, under a full scan too;
- camera_sync_skew_ms compares the cameras of one episode through the same probe;
- camera_stream_present, video_format_consistency and delivery_spec_met share
  one read of the container header, and presence also takes the segment's first frame;
  frames_readable judges the segment's packet probe and the shared decode.

Blur and exposure each judge one episode by itself,
and `kalanos.analysis.scoring.cameras` compares one camera across episodes.
Under `StreamContext.full_frame_scan`, every frame is decoded,
and exposure_shift_pct, the per-frame shares and freezes read every one of them.
sharpness_score, contrast_score and exposure_level summarize the episode:
they measure `VisionSpec.summary_ratio` of its frames, evenly spaced,
plus the frames a sampled read would measure.

The three share one decode,
which also reads `kalanos.analysis.metrics.continuity.cut_count`'s windows
and the segment's first frame for presence:

- when frozen_frame_pct runs, the sampled frames are taken from inside its
  windows, and otherwise only the sampled frames are decoded;
- each frame is decoded as a 128x128 gray image for exposure and freezes,
  and each frame the summary metrics measure also at native resolution for blur;
- every frame is measured and dropped, so a full scan runs in flat memory;
  of the rest, only cut_count's thumbnails are kept.

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
from collections import OrderedDict, deque
from contextlib import closing
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any

# External
import polars as pl

# Internal
from kalanos.analysis.adapters.video import (
    ContainerHeader,
    DeclaredFormat,
    DecodeFailed,
    DecodeLimitReached,
    DecoderUnavailable,
    PixelLimitReached,
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
from kalanos.analysis.optional import load_cv2, load_numpy
from kalanos.assets.dictionary import load_dictionary


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from kalanos.analysis.metrics.continuity import _CutPlan

    # The frame cache's key: what was read, and how it was read.
    _Key = tuple[
        int,  # id(payload)
        tuple[tuple[int, int], ...],  # the [start, end) windows read
        tuple[int, ...],  # the sampled frame positions
        tuple[int, ...],  # the evidence frame positions
        tuple[int, ...],  # the summary frame positions
        int,  # VisionSpec.dark_level
        int,  # VisionSpec.bright_level
        int | None,  # the max_decode_frames cap, None under a full scan
        int,  # VisionSpec.max_pixels
        tuple[tuple[int, int], ...],  # cut_count's [start, end) windows
        bool,  # whether the first frame is read in a window of its own
        tuple[int, int] | None,  # the (grid step, lag) stillness is read at
    ]

    _CutPlanEntry = tuple[
        SampledFrames, list[Stream], _CutPlan | MetricResult | Exception
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
# The 5-point Laplacian; `_laplacian_var` builds it as a float32 kernel.
_LAPLACIAN_KERNEL = ((0, 1, 0), (1, -4, 1), (0, 1, 0))

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
# A non-blank frame is decoder noise when its Laplacian variance is at least this
# multiple of its luminance variance, measured at native resolution.
# Candidate: pixel noise reads about 20 in theory and 6-18 on a corrupt simulated
# dataset, while clean footage on the calibration datasets reads 0.003-3.6.
_NOISE_RATIO = 5.0
# Candidate: a native P95 - P5 below this is a dark, noisy real scene and no garbage.
# Dark frames with added noise read 5-12 at a spread of 11-22,
# and every corrupt frame measured had a spread of 41 or more.
_NOISE_MIN_SPREAD = 32
# Candidate: on frames this small, edges alone dominate the noise statistic.
# The smallest real camera in the calibration set is 96 px on its shorter side.
_MIN_NOISE_SIDE = 32
# Candidate: a frame between this and `_NOISE_RATIO` is kept as a candidate,
# for the camera-relative rule in `kalanos.analysis.scoring.cameras`.
_RELATIVE_FLOOR = 2.0
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
# Stillness is read from a small thumbnail of each frame.
# The thumbnail is the 2x2 mean of the 64x64 one, which averages away sensor noise.
_STILL_SIZE = 32
# It is read on a grid of this many frames per second.
_STILL_GRID_HZ = 2.0
# Each grid frame is compared with the frame this long before it.
_STILL_LAG_S = 1.0
# A small moving arm changes a few cells by tens of levels,
# so the change is a high percentile of the difference, not its mean.
_STILL_PCT = 98
# A reading is smoothed by the median of those within this much grid time
# either side, when at least this many are finite.
_STILL_SMOOTH_S = 2.0
_STILL_SMOOTH_MIN = 3
# Candidate: 2.0 splits a low-resolution camera's still stretches into pieces
# (docs/METRICS.md, "Still scenes").
_STILL_CHANGE = 3.0
# The shortest stretch reported.
_STILL_MIN_S = 5.0
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
_CACHE: "OrderedDict[_Key, tuple[SampledFrames, _Read]]" = OrderedDict()

# cut_count's plan, keyed on id(payload) and the ids of the camera stream
# and its episode's streams, whose control signals the plan is drawn from.
# The value holds the payload and those streams so no id is reused while cached.
_CUT_PLANS: "OrderedDict[tuple[int, tuple[int, ...]], _CutPlanEntry]" = OrderedDict()

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
        The positions of the frames exposure measures, among those decoded.
    summary : list of int
        The positions of the sampled frames also measured at native resolution,
        the ones the summary metrics read: every sampled frame,
        unless the read is a full scan.
    summary_slots : list of int
        Each summary frame's index into `sampled` and the lists aligned to it.
    loss : list of float
        Each summary frame's share of Laplacian variance
        lost under a `_REBLUR` box blur.
    laplacian : list of float
        Each summary frame's native Laplacian variance.
    mean, p5, p95 : list of float
        Each sampled frame's mean, 5th and 95th percentile gray level.
    white_share : list of float
        Each sampled frame's share of pixels at or above `VisionSpec.bright_level`.
    dark_share : list of float
        Each sampled frame's share of pixels at or below `VisionSpec.dark_level`.
    noise : list of float
        Each sampled frame's native Laplacian variance over its luminance variance,
        `nan` for a frame of one gray level.
    native_spread : list of float
        Each sampled frame's native P95 - P5 spread.
    min_side : int or None
        The shortest side of any sampled frame at native resolution,
        `None` when none was measured.
    diffs : list of tuple of (int, numpy.ndarray)
        Per range that decoded at least two frames, its start
        and the contrast-scaled mean absolute difference between each pair
        of neighbouring thumbnails, NaN where either frame is blank.
    luma_steps : list of tuple of (int, numpy.ndarray)
        The same ranges' starts, and the unscaled mean absolute difference
        between each pair of neighbouring thumbnails, blank or not.
    pts : list of float or None
        Each summary frame's presentation time, `None` where the payload has none.
    shape : list of tuple of (int, int)
        Each summary frame's native `(height, width)`.
    luma_sha256 : list of str
        SHA-256 of each summary frame's native luminance.
    requested : list of int
        The summary positions the read was asked to measure.
    evidence_positions : list of int
        The positions a sampled read would measure,
        equal to `requested` unless the read is a full scan.
    pooled : dict of int to numpy.ndarray
        Each decoded evidence frame's `_VARIATION_SIZE` block mean
        of its `_SAMPLE_SIZE` gray image, scaled to 0-1, keyed by position.
    stopped : tuple of (Availability, str) or None
        Why the read stopped early, `None` when it ran to the end.
    failure : DecodeFailed or None
        The decode failure that stopped the read, a `ReadInterrupted` included.
    still_changes : list of tuple of (int, float)
        Each grid frame's index with its change against the frame
        `lag` frames earlier, NaN where either is blank;
        empty unless the read was a full scan.
    """

    windows: list[tuple[int, int]]
    n_read: int
    sampled: list[int]
    summary: list[int]
    summary_slots: list[int]
    loss: list[float]
    laplacian: list[float]
    mean: list[float]
    p5: list[float]
    p95: list[float]
    white_share: list[float]
    dark_share: list[float]
    noise: list[float]
    native_spread: list[float]
    min_side: int | None
    diffs: "list[tuple[int, np.ndarray]]"
    luma_steps: "list[tuple[int, np.ndarray]]"
    pts: list[float | None]
    shape: list[tuple[int, int]]
    luma_sha256: list[str]
    requested: list[int]
    evidence_positions: list[int]
    pooled: "dict[int, np.ndarray]"
    stopped: tuple[Availability, str] | None = None
    failure: DecodeFailed | None = None
    still_changes: list[tuple[int, float]] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class _Read:
    """What every consumer of one camera stream's shared decode keeps of it.

    Attributes
    ----------
    frames : _Frames or MetricResult
        What the vision metrics keep, or why they read nothing.
    cut : dict of int to numpy.ndarray
        The `_thumbnail` of every frame read inside a cut_count window,
        keyed by frame index.
    cut_failure : DecodeFailed or DecodeLimitReached or None
        What stopped the read of cut_count's windows, if anything did.
    first : bool or None
        `True` when the range starting at frame 0 yielded a frame,
        `False` when the read failed before it did,
        `None` when presence must read the first frame alone.
    first_failure : DecodeFailed or DecodeLimitReached or None
        What stopped the read before the first frame, when `first` is `False`.
    """

    frames: "_Frames | MetricResult"
    cut: "dict[int, np.ndarray]"
    cut_failure: DecodeFailed | DecodeLimitReached | None
    first: bool | None
    first_failure: DecodeFailed | DecodeLimitReached | None


@dataclass(frozen=True)
class _Exposure:
    """exposure_shift_pct's verdict on each sampled frame."""

    counts: dict[str, int]
    bad: list[int]
    worst_index: int | None
    typical_mean: float
    typical_white: float


class _VisionAccumulator:
    """What the vision metrics keep of a read, gathered frame by frame as it decodes."""

    def __init__(
        self,
        windows: list[tuple[int, int]],
        sampled: list[int],
        evidence: list[int],
        summary: list[int],
        *,
        dark_level: int,
        bright_level: int,
        still: tuple[int, int] | None = None,
    ) -> None:
        """Start an empty accumulation over `windows`.

        Parameters
        ----------
        windows : list of tuple of (int, int)
            The ascending `[start, end)` windows the vision metrics read.
        sampled : list of int
            The positions exposure measures.
        evidence : list of int
            The positions whose frames are pooled for the footage vector.
        summary : list of int
            The positions among `sampled` also measured at native resolution.
        dark_level, bright_level : int
            `VisionSpec.dark_level` and `VisionSpec.bright_level`.
        still : tuple of (int, int) or None
            `(grid_step, lag)` in frames to read stillness at,
            or `None` to skip it.
        """

        numpy = load_numpy()
        assert numpy is not None

        self.windows = windows
        self.sampled = sampled
        self.evidence = evidence
        self.summary = summary
        self.wanted = set(sampled)
        self.native = set(summary)
        self.evidence_set = set(evidence)
        self.dark_level = dark_level
        self.bright_level = bright_level
        self.kept: dict[str, list[float]] = {
            name: []
            for name in (
                "loss",
                "laplacian",
                "mean",
                "p5",
                "p95",
                "white_share",
                "dark_share",
                "noise",
                "native_spread",
            )
        }
        self.min_side: int | None = None
        self.measured: list[int] = []
        self.summary_measured: list[int] = []
        self.pts: list[float | None] = []
        self.shapes: list[tuple[int, int]] = []
        self.hashes: list[str] = []
        self.pooled: dict[int, np.ndarray] = {}
        self.counts = [0] * len(windows)
        self.diffs: list[list[float]] = [[] for _ in windows]
        self.steps: list[list[float]] = [[] for _ in windows]
        self.previous: np.ndarray | None = None
        self.previous_window = -1
        self.previous_spread = 0.0
        self.levels = numpy.arange(256)
        self.still = still
        # Only the last `lag + 1` thumbnails are kept: a full scan stays flat in memory.
        self.still_ring: deque[tuple[int, np.ndarray, float]] | None = (
            deque(maxlen=still[1] + 1) if still is not None else None
        )
        self.still_changes: list[tuple[int, float]] = []

    def add(
        self,
        window: int,
        index: int,
        frame: "np.ndarray",
        full: "np.ndarray | None",
        frame_pts: float | None,
    ) -> "np.ndarray":
        """Measure one decoded frame.

        Parameters
        ----------
        window : int
            The frame's window, as a position in `windows`.
        index : int
            The frame's index in the stream.
        frame : numpy.ndarray
            The frame as `gray_windows` yields it at `_SAMPLE_SIZE`.
        full : numpy.ndarray or None
            Its native luminance, present when `index` is a sampled position.
        frame_pts : float or None
            Its presentation time.

        Returns
        -------
        numpy.ndarray
            The frame's `_thumbnail`.
        """

        numpy = load_numpy()
        assert numpy is not None

        self.counts[window] += 1
        histogram = numpy.bincount(frame.ravel(), minlength=256)
        cumulative = numpy.cumsum(histogram)
        p5, p95 = _percentile(cumulative, 5), _percentile(cumulative, 95)
        spread = p95 - p5
        laplacian: float | None = None
        if index in self.wanted:
            self.measured.append(index)
            self.kept["mean"].append(int(histogram @ self.levels) / frame.size)
            self.kept["p5"].append(p5)
            self.kept["p95"].append(p95)
            # A pixel at or above the bright level is clipped white (rda),
            # one at or below the dark level crushed black.
            self.kept["white_share"].append(
                int(histogram[self.bright_level :].sum()) / frame.size
            )
            self.kept["dark_share"].append(
                int(histogram[: self.dark_level + 1].sum()) / frame.size
            )
            # Every sampled frame arrives at native resolution.
            assert full is not None
            if index in self.native:
                laplacian = _laplacian_var(full)
            noise, native_spread = _noise_score(full, laplacian)
            self.kept["noise"].append(noise)
            self.kept["native_spread"].append(native_spread)
            shortest = min(full.shape)
            self.min_side = (
                shortest if self.min_side is None else min(self.min_side, shortest)
            )
        # Every summary position is also sampled, so its slot in `measured` exists.
        if index in self.native:
            assert full is not None and laplacian is not None
            self.summary_measured.append(index)
            self.pts.append(frame_pts)
            self.shapes.append((int(full.shape[0]), int(full.shape[1])))
            self.hashes.append(hashlib.sha256(full.tobytes()).hexdigest())
            self.kept["loss"].append(_reblur_loss(full, laplacian))
            self.kept["laplacian"].append(laplacian)

        # Only the frames a sample measures are pooled,
        # so a full scan's vector matches a sampled read's.
        if index in self.evidence_set:
            self.pooled[index] = _thumbnail(frame, _VARIATION_SIZE) / 255

        thumbnail = _thumbnail(frame)
        if self.still is not None:
            self._add_still(window, index, thumbnail, spread)
        if window == self.previous_window and self.previous is not None:
            difference = float(numpy.abs(thumbnail - self.previous).mean())
            self.steps[window].append(difference)
            contrast = min(spread, self.previous_spread)
            if contrast < _BLANK_SPREAD:
                self.diffs[window].append(float("nan"))
            else:
                self.diffs[window].append(difference * _FREEZE_CONTRAST / contrast)
        self.previous, self.previous_window = thumbnail, window
        self.previous_spread = spread
        return thumbnail

    def _add_still(
        self, window: int, index: int, thumbnail: "np.ndarray", spread: float
    ) -> None:
        """Compare a grid frame with the frame `lag` frames before it.

        Parameters
        ----------
        window : int
            The frame's window; a new window starts a new comparison,
            since its first frames have no predecessor to compare with.
        index : int
            The frame's index in the stream.
        thumbnail : numpy.ndarray
            The frame's `_thumbnail`.
        spread : float
            The frame's P95 - P5.
        """

        numpy = load_numpy()
        assert numpy is not None
        assert self.still is not None and self.still_ring is not None

        grid_step, lag = self.still
        # `previous_window` is reassigned after this call.
        if window != self.previous_window:
            self.still_ring.clear()
        # The 2x2 means, as strided sums: the thumbnail's values are
        # multiples of 0.25, so the sum is exact in any order.
        small = (
            thumbnail[0::2, 0::2]
            + thumbnail[1::2, 0::2]
            + thumbnail[0::2, 1::2]
            + thumbnail[1::2, 1::2]
        ) * numpy.float32(0.25)
        self.still_ring.append((index, small, spread))
        if index % grid_step or self.still_ring[0][0] != index - lag:
            return
        _then_index, then, then_spread = self.still_ring[0]
        if min(spread, then_spread) < _BLANK_SPREAD:
            self.still_changes.append((index, float("nan")))
            return
        self.still_changes.append((index, _still_change(small - then)))

    def finish(
        self,
        stopped: tuple[Availability, str] | None,
        failure: DecodeFailed | None,
    ) -> "_Frames | MetricResult":
        """Close the accumulation into what the vision metrics read.

        Parameters
        ----------
        stopped : tuple of (Availability, str) or None
            Why the read stopped early, when a decode cap stopped it.
        failure : DecodeFailed or None
            The decode failure that stopped the read, if one did.

        Returns
        -------
        _Frames or MetricResult
            What was kept, or `not_applicable` when no frame was read:
            a failure before any frame decoded, such as a file that cannot be opened,
            and a read that ended without a frame, are not camera footage to grade.
            A failure in the file after a frame decoded is `unavailable`,
            since `frames_readable` grades it,
            and a failure outside the file, such as on the network, is an `error`.
        """

        numpy = load_numpy()
        assert numpy is not None

        n_read = sum(self.counts)
        if failure is not None:
            # Only a file that started decoding was readable and then broke;
            # one that never yields a frame was never camera footage to read.
            if not n_read:
                return not_applicable(str(failure))
            stopped = (
                Availability.ERROR
                if isinstance(failure, ReadInterrupted)
                else Availability.UNAVAILABLE,
                str(failure),
            )
        if not n_read and stopped is None:
            return not_applicable("no frame could be decoded")

        slot = {index: k for k, index in enumerate(self.measured)}
        return _Frames(
            windows=self.windows,
            n_read=n_read,
            sampled=self.measured,
            summary=self.summary_measured,
            summary_slots=[slot[index] for index in self.summary_measured],
            min_side=self.min_side,
            diffs=[
                (start, numpy.asarray(window_diffs, dtype=numpy.float32))
                for (start, _end), window_diffs in zip(
                    self.windows, self.diffs, strict=True
                )
                if window_diffs
            ],
            luma_steps=[
                (start, numpy.asarray(window_steps, dtype=numpy.float32))
                for (start, _end), window_steps in zip(
                    self.windows, self.steps, strict=True
                )
                if window_steps
            ],
            pts=self.pts,
            shape=self.shapes,
            luma_sha256=self.hashes,
            requested=list(self.summary),
            evidence_positions=list(self.evidence),
            pooled=self.pooled,
            stopped=stopped,
            failure=failure,
            still_changes=self.still_changes,
            **self.kept,
        )


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
    if load_numpy() is None or load_cv2() is None:
        return not_applicable(
            "sampling frames needs numpy and OpenCV; add kalanos[video]"
        )
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


def _still_plan(ctx: StreamContext) -> tuple[int, int] | None:
    """Where stillness is read, as `(grid_step, lag)` in frames.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream.

    Returns
    -------
    tuple of (int, int) or None
        `None` unless the read is a full scan of a stream with a frame rate.
    """

    fps = _frame_rate(ctx)
    if not ctx.full_frame_scan or fps is None:
        return None
    return max(1, round(fps / _STILL_GRID_HZ)), max(1, round(fps * _STILL_LAG_S))


def _still_change(difference: "np.ndarray") -> float:
    """The change between two thumbnails, once their common shift is removed.

    Equal to the `_STILL_PCT`th percentile of `|difference - median(difference)|`,
    taken with two partial sorts instead of `numpy.median` and `numpy.percentile`,
    which cost several times as much on a thumbnail this small.

    Parameters
    ----------
    difference : numpy.ndarray
        The `(_STILL_SIZE, _STILL_SIZE)` float32 difference of two thumbnails.

    Returns
    -------
    float
        The change, in gray levels.
    """

    numpy = load_numpy()
    assert numpy is not None

    flat = difference.ravel()
    middle = flat.size // 2
    low, high = numpy.partition(flat, [middle - 1, middle])[middle - 1 : middle + 1]
    # The median of an even count, the mean of its two middle values.
    centred = numpy.abs(flat - (low + high) / 2)
    # Linear interpolation between the neighbouring ranks, as `numpy.percentile`.
    rank = (flat.size - 1) * (_STILL_PCT / 100)
    below = int(rank)
    weight = rank - below
    low, high = numpy.partition(centred, [below, below + 1])[below : below + 2]
    if weight < 0.5:
        return float(low + (high - low) * weight)
    return float(high - (high - low) * (1 - weight))


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
    """Pick frozen frames' windows and the frames exposure measures.

    Frozen frames read `vision_samples` evenly spaced windows,
    each two minimum freeze runs long (about 1 s),
    or the whole stream when the windows would cover it.
    Blur and exposure measure `vision_samples` evenly spaced frames,
    each inside a window, so one decode serves all three.
    Under `full_frame_scan`, exposure and freezes read every frame,
    and `_summary_positions` picks the frames blur measures.

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


def _summary_positions(
    ctx: StreamContext, sampled: list[int], evidence: list[int]
) -> list[int]:
    """Pick the frames the summary metrics measure, each decoded at native resolution.

    A sampled read measures its sampled frames.
    A full scan measures `VisionSpec.summary_ratio` of the stream's frames,
    evenly spaced, plus the frames a sampled read would measure,
    so their hashes and the footage vector match a sampled read's.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to lay out.
    sampled : list of int
        The positions `_layout` picked for exposure.
    evidence : list of int
        The positions a sampled read would measure.

    Returns
    -------
    list of int
        The ascending summary positions.
    """

    if not ctx.full_frame_scan:
        return sampled
    n = ctx.n_samples
    ratio = _spread(n, math.ceil(ctx.vision.summary_ratio * n))
    return sorted(set(evidence) | set(ratio))


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


def _merge(windows: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Clip `[start, end)` windows to `[0, n)`, then sort and merge overlapping ones.

    Parameters
    ----------
    windows : list of tuple of (int, int)
        Windows, in any order.
    n : int
        The frame count to clip to.

    Returns
    -------
    list of tuple of (int, int)
        Ascending, non-overlapping windows, as `gray_windows` takes them.
    """

    merged: list[tuple[int, int]] = []
    for start, end in sorted((max(0, a), min(n, b)) for a, b in windows):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _frames(ctx: StreamContext) -> "_Frames | MetricResult":
    """Return what the vision metrics keep of the stream's frames, or `not_applicable`.

    Whichever vision metric asks first decodes for every one of them:
    the windows when frozen_frame_pct will read them,
    the sampled frames alone otherwise.
    The stream is checked on every call; only the read is cached,
    so two streams sharing one payload are each judged on their own type.
    """

    try:
        read = _shared_read(ctx)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(read, MetricResult):
        return read
    return read.frames


def _cut_plan(ctx: StreamContext, payload: SampledFrames) -> "_CutPlan | MetricResult":
    """Return cut_count's plan for the stream, computing it once per stream and episode.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream, with its episode's streams.
    payload : SampledFrames
        The stream's frames.

    Returns
    -------
    _CutPlan or MetricResult
        The windows cut_count reads, or why it reads none.

    Raises
    ------
    Exception
        Whatever reading the control streams raised;
        it is kept, so a failing control payload is fetched once per stream.
    """

    # continuity imports this module.
    from kalanos.analysis.metrics.continuity import _plan_cuts

    streams = [ctx.stream, *(ctx.episode_streams or [])]
    key = (id(payload), tuple(id(stream) for stream in streams))
    entry = _CUT_PLANS.get(key)
    if entry is not None and entry[0] is payload:
        _CUT_PLANS.move_to_end(key)
        plan = entry[2]
    else:
        try:
            plan = _plan_cuts(ctx)
        except Exception as exc:
            plan = exc
        _CUT_PLANS[key] = (payload, streams, plan)
        while len(_CUT_PLANS) > _CACHE_SIZE:
            _CUT_PLANS.popitem(last=False)
    if isinstance(plan, Exception):
        raise plan
    return plan


def _shared_read(ctx: StreamContext) -> "_Read | MetricResult":
    """Decode the stream once for the vision metrics, cut_count and presence.

    The read covers the vision metrics' windows, cut_count's windows
    and the segment's first frame, cut at every window's bounds by `_split`.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream, with its episode's streams.

    Returns
    -------
    _Read or MetricResult
        What each consumer keeps of the read,
        or `not_applicable` when the stream's frames cannot be read.

    Raises
    ------
    DecoderUnavailable
        If no decoder is installed.
    """

    payload = _readable(ctx)
    if isinstance(payload, MetricResult):
        return payload

    windows, sampled = _layout(ctx, ctx.full_frame_scan)
    evidence = _layout(ctx, False)[1] if ctx.full_frame_scan else sampled
    summary = _summary_positions(ctx, sampled, evidence)
    if not _freeze_reads(ctx):
        windows = _ranges(sampled)
    spec = ctx.vision
    # A full scan is asked to read every frame, so no frame budget stops it.
    max_decode_frames = None if ctx.full_frame_scan else spec.max_decode_frames
    try:
        plan = _cut_plan(ctx, payload)
    except Exception:
        # cut_count reports the failure itself;
        # the other consumers read without its windows.
        plan = None
    cut_windows = [] if plan is None or isinstance(plan, MetricResult) else plan.windows
    first = [] if any(start == 0 for start, _end in windows + cut_windows) else [(0, 1)]
    still = _still_plan(ctx)
    key = (
        id(payload),
        tuple(windows),
        tuple(sampled),
        tuple(evidence),
        tuple(summary),
        spec.dark_level,
        spec.bright_level,
        max_decode_frames,
        spec.max_pixels,
        tuple(cut_windows),
        bool(first),
        still,
    )
    if key in _CACHE and _CACHE[key][0] is payload:
        _CACHE.move_to_end(key)
        return _CACHE[key][1]

    result = _pass(
        payload,
        _split(windows + cut_windows + first, ctx.n_samples),
        windows,
        sampled,
        evidence,
        summary,
        cut_windows,
        dark_level=spec.dark_level,
        bright_level=spec.bright_level,
        max_decode_frames=max_decode_frames,
        max_pixels=spec.max_pixels,
        still=still,
    )
    _CACHE[key] = (payload, result)
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return result


def _split(windows: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Cover `windows`, clipped to `[0, n)`, with pieces cut at every window's bounds.

    Each window then starts a piece,
    so the read seeks to it as the window's own read would,
    and every piece lies wholly inside or wholly outside each window.

    Parameters
    ----------
    windows : list of tuple of (int, int)
        `[start, end)` windows, in any order, overlapping or not.
    n : int
        The frame count to clip to.

    Returns
    -------
    list of tuple of (int, int)
        Ascending, non-overlapping pieces, as `gray_windows` takes them.
    """

    clipped = [(max(0, start), min(n, end)) for start, end in windows]
    clipped = [(start, end) for start, end in clipped if start < end]
    bounds = sorted({bound for window in clipped for bound in window})
    return [
        (start, end)
        for start, end in zip(bounds, bounds[1:], strict=False)
        if any(a <= start and end <= b for a, b in clipped)
    ]


class _Numbering:
    """Number one consumer's frames as its own read of its windows would.

    The shared read seeks at every piece, the consumer's own read once per window.
    Both take the same frames, numbered alike,
    while each earlier piece of a window yielded exactly its length,
    and no piece yields again a frame the one before it did,
    as a seek clipped to the segment's end can.
    Otherwise, on footage missing or reordering frames,
    the consumer is irregular and reads its windows alone.
    """

    def __init__(
        self, windows: list[tuple[int, int]], pieces: list[tuple[int, int]]
    ) -> None:
        """Map each piece to the window holding it.

        Parameters
        ----------
        windows : list of tuple of (int, int)
            The consumer's ascending, non-overlapping windows.
        pieces : list of tuple of (int, int)
            The pieces the shared read decodes, from `_split`.
        """

        self.windows = windows
        self.pieces = pieces
        self.window_of = [
            next(
                (w for w, (a, b) in enumerate(windows) if a <= start and end <= b),
                None,
            )
            for start, end in pieces
        ]
        self.taken = [0] * len(windows)
        self.piece = [-1] * len(windows)
        self.last_pts: list[float | None] = [None] * len(windows)
        self.irregular = False

    def take(self, piece: int, pts: float | None) -> int | None:
        """Return the window a frame of `piece` belongs to, or `None` if it serves none.

        Parameters
        ----------
        piece : int
            The frame's piece, as a position in `pieces`.
        pts : float or None
            The frame's presentation time.

        Returns
        -------
        int or None
            The frame's window, as a position in `windows`;
            `None` when no window holds the piece or the consumer is irregular.
        """

        w = self.window_of[piece]
        if w is None or self.irregular:
            return None
        if self.piece[w] != piece:
            last_pts = self.last_pts[w]
            if self.piece[w] >= 0 and (
                self.taken[w] != self.pieces[piece][0] - self.windows[w][0]
                or (pts is not None and last_pts is not None and pts <= last_pts)
            ):
                self.irregular = True
                return None
            self.piece[w] = piece
        self.taken[w] += 1
        self.last_pts[w] = pts
        return w


def _pass(
    payload: SampledFrames,
    ranges: list[tuple[int, int]],
    windows: list[tuple[int, int]],
    sampled: list[int],
    evidence: list[int],
    summary: list[int],
    cut_windows: list[tuple[int, int]],
    *,
    dark_level: int,
    bright_level: int,
    max_decode_frames: int | None,
    max_pixels: int,
    still: tuple[int, int] | None = None,
) -> _Read:
    """Decode `ranges` of `payload` once, handing each frame to the consumers it serves.

    A stop is charged to the consumers it kept from their frames,
    each as its own read would have met it:

    - the decode cap counts every frame of the pass,
      so it stops every consumer with frames left to read;
    - a decode failure, or a frame over `max_pixels`,
      stops the consumers whose own read would have met it,
      and a consumer whose frames the pass did not reach reads them alone.

    `_charge` places each stop.

    Parameters
    ----------
    payload : SampledFrames
        The camera's frames.
    ranges : list of tuple of (int, int)
        The pieces to decode, from `_split` over every window below.
    windows : list of tuple of (int, int)
        The vision metrics' windows.
    sampled : list of int
        The positions exposure measures.
    evidence : list of int
        The positions pooled for the footage vector.
    summary : list of int
        The positions among `sampled` also decoded at native resolution.
    cut_windows : list of tuple of (int, int)
        cut_count's windows.
    dark_level, bright_level : int
        `VisionSpec.dark_level` and `VisionSpec.bright_level`.
    max_decode_frames : int or None
        The most frames the whole pass may decode.
    max_pixels : int
        The most pixels a summary frame may hold.
    still : tuple of (int, int) or None
        `(grid_step, lag)` in frames to read stillness at, `None` to skip it.

    Returns
    -------
    _Read
        What each consumer keeps of the pass.

    Raises
    ------
    DecoderUnavailable
        If no decoder is installed.
    """

    # continuity imports this module.
    from kalanos.analysis.metrics.continuity import _window_thumbnails

    accumulator = _VisionAccumulator(
        windows,
        sampled,
        evidence,
        summary,
        dark_level=dark_level,
        bright_level=bright_level,
        still=still,
    )
    cut: dict[int, np.ndarray] = {}
    vision_numbering = _Numbering(windows, ranges)
    cut_numbering = _Numbering(cut_windows, ranges)
    counts = [0] * len(ranges)
    last = -1
    first_read = False
    stop: DecodeFailed | DecodeLimitReached | None = None

    # Step 1: one read over every range.
    try:
        with closing(
            payload.gray_windows(
                ranges,
                _SAMPLE_SIZE,
                native=frozenset(sampled),
                max_decode_frames=max_decode_frames,
                max_pixels=max_pixels,
            )
        ) as frames:
            for position, frame, full, frame_pts in frames:
                # Step 2: each frame to the consumers whose windows hold it.
                # A piece is only ever cut short at its end,
                # so the frames it did yield are its first ones.
                index = ranges[position][0] + counts[position]
                counts[position] += 1
                last = position
                if ranges[position][0] == 0:
                    first_read = True
                thumbnail = None
                window = vision_numbering.take(position, frame_pts)
                if window is not None:
                    thumbnail = accumulator.add(window, index, frame, full, frame_pts)
                if cut_numbering.take(position, frame_pts) is not None:
                    cut[index] = (
                        thumbnail if thumbnail is not None else _thumbnail(frame)
                    )
    # Step 3: what stopped the read, if anything did.
    except (DecodeFailed, DecodeLimitReached) as exc:
        stop = exc

    # Step 4: charge the stop to the consumers it kept from their frames.
    stopped: tuple[Availability, str] | None = None
    failure: DecodeFailed | None = None
    frames_read: _Frames | MetricResult | None = None
    cut_failure: DecodeFailed | DecodeLimitReached | None = None
    first: bool | None = True if first_read else None
    first_failure: DecodeFailed | DecodeLimitReached | None = None
    if stop is not None:
        stops = _charge(stop, ranges, counts, last)
        vision_stop, cut_stop = stops(windows), stops(cut_windows)
        if vision_stop == "hit":
            if isinstance(stop, DecodeFailed):
                failure = stop
            else:
                stopped = (Availability.UNAVAILABLE, str(stop))
        elif vision_stop == "alone":
            frames_read = _read(
                payload,
                windows,
                sampled,
                evidence,
                summary,
                dark_level=dark_level,
                bright_level=bright_level,
                max_decode_frames=max_decode_frames,
                max_pixels=max_pixels,
                still=still,
            )
        # cut_count's own read takes no frame at native resolution,
        # so it never meets a frame over max_pixels.
        if cut_stop == "hit" and not isinstance(stop, PixelLimitReached):
            cut_failure = stop
        elif cut_stop is not None:
            cut, cut_failure = _window_thumbnails(payload, cut_windows)
        # Before any frame, only the decode cap is charged to presence;
        # after any other stop it reads its first frame alone.
        if last < 0 and _is_cap(stop):
            first, first_failure = False, stop

    # Step 5: a consumer whose frames the pieces could not number reads alone.
    if vision_numbering.irregular:
        frames_read = _read(
            payload,
            windows,
            sampled,
            evidence,
            summary,
            dark_level=dark_level,
            bright_level=bright_level,
            max_decode_frames=max_decode_frames,
            max_pixels=max_pixels,
            still=still,
        )
    if cut_numbering.irregular:
        cut, cut_failure = _window_thumbnails(payload, cut_windows)

    # Step 6: what each consumer keeps.
    return _Read(
        frames=frames_read
        if frames_read is not None
        else accumulator.finish(stopped, failure),
        cut=cut,
        cut_failure=cut_failure,
        first=first,
        first_failure=first_failure,
    )


def _is_cap(stop: DecodeFailed | DecodeLimitReached) -> bool:
    """Whether `stop` came from `max_decode_frames`."""

    return isinstance(stop, DecodeLimitReached) and not isinstance(
        stop, PixelLimitReached
    )


def _charge(
    stop: DecodeFailed | DecodeLimitReached,
    ranges: list[tuple[int, int]],
    counts: list[int],
    last: int,
) -> "Callable[[list[tuple[int, int]]], str | None]":
    """Decide how a stop of the shared read reaches each consumer's windows.

    A read shows only which frames it yielded.
    A stop inside a range fell on the frame after the last one yielded.
    A stop after a range yielded all its frames fell either on the frame
    the read decodes past that range to find its end, or in the seek to the next range.

    - The decode cap stops every consumer with frames left to read.
    - A decode failure or a frame over `max_pixels` inside a range hits the consumers
      whose windows hold that frame or end on it,
      since their own read would have decoded it too;
      a consumer with frames only after it reads alone.
    - The same stop between ranges cannot be placed,
      so the consumers on either side of it read alone,
      each meeting it, or not, as its own read would.

    Parameters
    ----------
    stop : DecodeFailed or DecodeLimitReached
        What stopped the read.
    ranges : list of tuple of (int, int)
        The ranges read.
    counts : list of int
        How many frames each range yielded.
    last : int
        The last range that yielded a frame, -1 when none did.

    Returns
    -------
    Callable
        Given a consumer's windows, `"hit"` when the stop is the consumer's own,
        `"alone"` when the consumer must read its windows alone,
        `None` when the stop does not reach it.
    """

    inside = last >= 0 and counts[last] < ranges[last][1] - ranges[last][0]
    if inside:
        at = ranges[last][0] + counts[last]
    else:
        at = ranges[last][1] if last >= 0 else 0

    def charge(windows: list[tuple[int, int]]) -> str | None:
        """How the stop reaches a consumer reading `windows`."""

        windows = [(start, end) for start, end in windows if start < end]
        left = any(end > at for _start, end in windows)
        if _is_cap(stop):
            return "hit" if left else None
        if inside:
            if any(start <= at <= end for start, end in windows):
                return "hit"
            return "alone" if left else None
        ends_here = last >= 0 and any(end == at for _start, end in windows)
        return "alone" if left or ends_here else None

    return charge


def _read(
    payload: SampledFrames,
    windows: list[tuple[int, int]],
    sampled: list[int],
    evidence: list[int],
    summary: list[int],
    *,
    dark_level: int,
    bright_level: int,
    max_decode_frames: int | None,
    max_pixels: int,
    still: tuple[int, int] | None = None,
) -> "_Frames | MetricResult":
    """Decode `windows` of `payload` for the vision metrics alone.

    The reader the shared pass falls back on when a failure in another consumer's
    frames stopped it before the vision metrics' own.
    A read stopped by a decode cap, or by a decode failure after its first frame,
    keeps what it read, with the reason in `stopped`:
    `unavailable` for a cap or a failure in the file, which `frames_readable` grades,
    and `error` for a failure outside the file, such as on the network.
    A failure before any frame decoded, such as a file that cannot be opened,
    is `not_applicable`.
    `still` is `(grid_step, lag)` in frames to read stillness at, `None` to skip it.
    """

    accumulator = _VisionAccumulator(
        windows,
        sampled,
        evidence,
        summary,
        dark_level=dark_level,
        bright_level=bright_level,
        still=still,
    )
    stopped: tuple[Availability, str] | None = None
    failure: DecodeFailed | None = None
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
                index = windows[position][0] + accumulator.counts[position]
                accumulator.add(position, index, frame, full, frame_pts)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    except DecodeLimitReached as exc:
        stopped = (Availability.UNAVAILABLE, str(exc))
    except DecodeFailed as exc:
        failure = exc
    return accumulator.finish(stopped, failure)


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
    """Empty the frame, cut plan, packet and header caches."""

    _CACHE.clear()
    _CUT_PLANS.clear()
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


def _noise_score(
    full: "np.ndarray", laplacian: float | None = None
) -> tuple[float, float]:
    """Measure how much of a frame's variance is pixel-to-pixel noise.

    Parameters
    ----------
    full : numpy.ndarray
        The frame's native luminance.
    laplacian : float, optional
        The frame's `_laplacian_var`, when the caller already has it.

    Returns
    -------
    tuple of (float, float)
        The Laplacian variance over the luminance variance, `nan` for a frame
        of one gray level or under 3 pixels on a side,
        and the native P95 - P5 spread.
    """

    numpy = load_numpy()
    assert numpy is not None

    cumulative = numpy.cumsum(numpy.bincount(full.ravel(), minlength=256))
    spread = _percentile(cumulative, 95) - _percentile(cumulative, 5)
    variance = float(full.var())
    if variance == 0:
        return math.nan, spread
    if laplacian is None:
        laplacian = _laplacian_var(full)
    return laplacian / variance, spread


def _laplacian_var(frame: "np.ndarray") -> float:
    """Variance of the 3x3 Laplacian over the whole frame.

    The border row and column on each side are left out,
    since their Laplacian would need pixels outside the frame.

    Parameters
    ----------
    frame : numpy.ndarray
        A 2-D uint8 or float32 luminance frame.

    Returns
    -------
    float
        The variance, or `nan` for a frame under 3 pixels in either dimension.
    """

    height, width = frame.shape
    if height < 3 or width < 3:
        return math.nan
    numpy = load_numpy()
    cv2 = load_cv2()
    assert numpy is not None and cv2 is not None

    image = frame if frame.dtype == numpy.float32 else frame.astype(numpy.float32)
    kernel = numpy.array(_LAPLACIAN_KERNEL, numpy.float32)
    laplacian = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    _, deviation = cv2.meanStdDev(laplacian[1:-1, 1:-1])
    return float(deviation[0, 0]) ** 2


def _reblur_loss(frame: "np.ndarray", laplacian: float) -> float:
    """Share of `laplacian` a `_REBLUR` box blur of the frame removes.

    Parameters
    ----------
    frame : numpy.ndarray
        A 2-D uint8 luminance frame.
    laplacian : float
        `_laplacian_var` of the frame.

    Returns
    -------
    float
        The share, or 0.0 for a frame with no edges.
    """

    if laplacian == 0:
        return 0.0
    cv2 = load_cv2()
    assert cv2 is not None

    # The 3x3 sums of uint8 pixels and their Laplacian are exact integers in float32;
    # dividing each pixel by 9 would round it,
    # which on smooth frames shifts the loss by up to about 4e-6.
    # Dividing the variance by `_REBLUR ** 4` afterwards is exact.
    window_sum = cv2.boxFilter(
        frame,
        cv2.CV_32F,
        (_REBLUR, _REBLUR),
        normalize=False,
        borderType=cv2.BORDER_REPLICATE,
    )
    return 1 - _laplacian_var(window_sum) / _REBLUR**4 / laplacian


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
    """Median share of the summary frames' edge energy a 3x3 box re-blur removes.

    Higher is sharper. The summary frames are the sampled frames,
    or under a full scan `VisionSpec.summary_ratio` of the stream's frames
    plus the sampled read's.

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
    if not frames.summary:
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
            "worst_index": int(frames.summary[worst]),
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
    """Median mean gray of the summary frames, 0 to 255; reported, never graded.

    The summary frames are those sharpness_score measures.

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
    if not frames.summary:
        return not_applicable("no sampled frame could be decoded")
    numpy = load_numpy()
    assert numpy is not None

    slots = frames.summary_slots
    return MetricResult(
        value=float(numpy.median([frames.mean[k] for k in slots])),
        unit="gray level",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_sampled": len(frames.summary),
            "n_frames": len(ctx.timestamps),
            "dark_share": float(numpy.median([frames.dark_share[k] for k in slots])),
            "bright_share": float(numpy.median([frames.white_share[k] for k in slots])),
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


@metric(level=Level.STREAM, family=Family.VISION, label="corrupt camera frames")
def corrupt_frame_pct(ctx: StreamContext) -> MetricResult:
    """Share of sampled frames that are blank or hold decoder noise.

    A frame is blank when its P95 - P5 spread on the `_SAMPLE_SIZE` decode is below
    `_BLANK_SPREAD`, at any gray level.
    Any other frame is noise when its native P95 - P5 spread is at least
    `_NOISE_MIN_SPREAD` and its Laplacian variance is at least `_NOISE_RATIO`
    times its luminance variance, the signature of pixel noise and grey streaks.
    The spread guard keeps dark, noisy real scenes out.

    The frames that read between `_RELATIVE_FLOOR` and `_NOISE_RATIO`
    are published as candidates,
    for a rule that judges them against the same camera across the dataset.
    The share is report-only here; the default policy grades it.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only`, the share in percent with each corrupt frame as a support
        interval, or `not_applicable` when the stream cannot be sampled
        or its frames are under `_MIN_NOISE_SIDE` pixels on their shorter side.
    """

    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.sampled:
        return not_applicable("no sampled frame could be decoded")
    if frames.min_side is None or frames.min_side < _MIN_NOISE_SIDE:
        return not_applicable("frames too small to tell noise from detail")

    bad: list[int] = []
    blank = noise = 0
    scores: list[float] = []
    candidates: list[list[float]] = []
    for index, p5, p95, ratio, spread in zip(
        frames.sampled,
        frames.p5,
        frames.p95,
        frames.noise,
        frames.native_spread,
        strict=True,
    ):
        if p95 - p5 < _BLANK_SPREAD:
            blank += 1
            bad.append(int(index))
            continue
        if not math.isnan(ratio):
            scores.append(ratio)
        if spread < _NOISE_MIN_SPREAD or math.isnan(ratio):
            continue
        if ratio >= _NOISE_RATIO:
            noise += 1
            bad.append(int(index))
        elif ratio >= _RELATIVE_FLOOR:
            candidates.append([int(index), round(ratio, 4)])

    numpy = load_numpy()
    assert numpy is not None
    return _frame_pct(
        ctx,
        frames,
        bad,
        blank=blank,
        noise=noise,
        median_score=float(numpy.median(scores)) if scores else None,
        candidates=candidates,
    )


@metric(level=Level.STREAM, family=Family.VISION, label="low contrast")
def contrast_score(ctx: StreamContext) -> MetricResult:
    """Median P95 - P5 spread of the summary frames, 0 to 255 gray levels.

    Lower is flatter. It follows what is in view as much as the camera,
    so it is reported and not graded.
    The summary frames are those sharpness_score measures.

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
    if not frames.summary:
        return not_applicable("no sampled frame could be decoded")
    numpy = load_numpy()
    assert numpy is not None

    spreads = [frames.p95[k] - frames.p5[k] for k in frames.summary_slots]
    worst = min(range(len(spreads)), key=spreads.__getitem__)
    return MetricResult(
        value=float(numpy.median(spreads)),
        unit="gray level",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_sampled": len(frames.summary),
            "n_frames": len(ctx.timestamps),
            "min_spread": float(spreads[worst]),
            "worst_index": int(frames.summary[worst]),
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

    Reads the container header, then takes the segment's first frame
    from the decode the other vision metrics share.
    When that read cannot tell, presence decodes up to the first frame the segment holds
    by itself, discarding the seek preroll before it.
    That frame need not sit exactly at the segment's start,
    so a frame the encoder dropped there does not read as a missing camera.
    A decode failure after that frame is not this metric's to judge:
    `frames_readable` judges it.

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
    if load_numpy() is None or load_cv2() is None:
        return not_applicable(
            "sampling frames needs numpy and OpenCV; add kalanos[video]"
        )

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
            read = _shared_read(ctx)
        except DecoderUnavailable as exc:
            return not_applicable(str(exc))
        if isinstance(read, MetricResult) or read.first is None:
            first = _first_frame(payload, ctx)
            if isinstance(first, MetricResult):
                return first
            decoded, reason = first
        elif read.first:
            decoded = True
        else:
            failure = read.first_failure
            # A stop at the decode cap is no evidence the file is absent.
            if isinstance(failure, DecodeLimitReached):
                return _stopped_result((Availability.UNAVAILABLE, str(failure)))
            decoded, reason = False, str(failure)
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


def _first_frame(
    payload: VideoPayload, ctx: StreamContext
) -> MetricResult | tuple[bool, str | None]:
    """Decode up to the segment's first frame by itself, for presence.

    Parameters
    ----------
    payload : VideoPayload
        The camera's video file.
    ctx : StreamContext
        The camera stream, for its decode limits.

    Returns
    -------
    MetricResult or tuple of (bool, str or None)
        Whether a frame decoded, with the decode failure that stopped it, if any;
        or an unavailable result when the read stopped for a reason
        that says nothing about the file.
    """

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
            return next(frames, None) is not None, None
    except (DecodeLimitReached, ReadInterrupted) as exc:
        return _stopped_result((Availability.UNAVAILABLE, str(exc)))
    except DecodeFailed as exc:
        return False, str(exc)


@metric(level=Level.STREAM, family=Family.VISION, label="unreadable camera frames")
def frames_readable(ctx: StreamContext) -> MetricResult:
    """Whether the segment's packets demux and its frames decode past the first.

    Judges the packet probe the timing metrics share
    and the decode the other vision metrics share.
    A file that does not open, or yields no frame of the segment,
    is `camera_stream_present`'s to judge, so one cause raises one finding.
    A failure outside the file, such as on the network, is `unavailable`.
    The sampled read finds only failures in the frames it reads,
    and the full tier reads every frame.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to check.

    Returns
    -------
    MetricResult
        `report_only` with 1.0 when every frame the segment's reads asked for
        demuxed and decoded, and 0.0 when the file failed in the segment,
        for the policy to regrade;
        `not_applicable` when the stream is not backed by a video file
        or no frame of it decoded.
    """

    payload = _video_file(ctx)
    if isinstance(payload, MetricResult):
        return payload
    if load_numpy() is None or load_cv2() is None:
        return not_applicable(
            "sampling frames needs numpy and OpenCV; add kalanos[video]"
        )

    # A failure outside the file hides no defect another stage found,
    # so it only decides the result when neither stage found one.
    interrupted: ReadInterrupted | None = None
    try:
        header = _header(payload)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(header, ReadInterrupted):
        interrupted = header
    elif isinstance(header, DecodeFailed):
        return not_applicable("the file does not open; camera_stream_present judges it")

    try:
        read = _shared_read(ctx)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(read, MetricResult) or isinstance(read.frames, MetricResult):
        if interrupted is not None:
            return _stopped_result((Availability.UNAVAILABLE, str(interrupted)))
        return not_applicable(
            "no frame of the segment decoded; camera_stream_present judges it"
        )

    defect: DecodeFailed | None = None
    stage: str | None = None
    failure = read.frames.failure
    if failure is None and isinstance(read.cut_failure, DecodeFailed):
        failure = read.cut_failure
    packets = _packets(payload)
    # ReadInterrupted subclasses DecodeFailed, so it is checked first.
    if isinstance(packets, ReadInterrupted):
        interrupted = interrupted or packets
    elif isinstance(packets, DecodeFailed):
        defect, stage = packets, "demux"
    if isinstance(failure, ReadInterrupted):
        interrupted = interrupted or failure
    elif failure is not None and defect is None:
        defect, stage = failure, "decode"
    if defect is None and interrupted is not None:
        return _stopped_result((Availability.UNAVAILABLE, str(interrupted)))
    if defect is None and read.frames.stopped is not None:
        return _stopped_result(read.frames.stopped)

    return MetricResult(
        value=0.0 if defect is not None else 1.0,
        unit="flag",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "camera": ctx.stream.source_field,
            "path": str(payload.path),
            "stage": stage,
            "reason": str(defect) if defect is not None else None,
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


@metric(
    level=Level.STREAM,
    family=Family.VISION,
    label="footage misses the delivery spec",
)
def delivery_spec_met(ctx: StreamContext) -> MetricResult:
    """Whether the stream's footage meets the declared delivery spec.

    Compares the container header with `VisionSpec.delivery`, decoding nothing:
    width, height and frame rate against their minimums,
    the segment's span against the minimum duration,
    and the codec against the allowed names.
    A frame rate meets its minimum within `_FPS_REL_TOL`.
    A value the container does not state is listed under `unstated` and skipped.
    A file that cannot be opened is `camera_stream_present`'s to report.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to check.

    Returns
    -------
    MetricResult
        A flag, 1 when every declared field is met and 0 otherwise,
        with each miss's field, required and found values in the evidence,
        or `not_applicable` when no spec is declared
        or the stream is not backed by a readable video file.
    """

    delivery = ctx.vision.delivery
    if not delivery.declared():
        return not_applicable("no delivery spec is declared")
    payload = _video_file(ctx)
    if isinstance(payload, MetricResult):
        return payload
    try:
        header = _header(payload)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if isinstance(header, DecodeFailed):
        return not_applicable(str(header))

    # The segment's span: one LeRobot v3 file holds many episodes.
    found: dict[str, Any] = {
        "width": header.width,
        "height": header.height,
        "fps": header.fps,
        "duration_s": payload.end_s - payload.start_s,
        "codec": None if header.codec is None else header.codec.lower(),
    }
    minimums: dict[str, tuple[str, Any]] = {
        "width": ("min_width", delivery.min_width),
        "height": ("min_height", delivery.min_height),
        "fps": ("min_fps", delivery.min_fps),
        "duration_s": ("min_duration_s", delivery.min_duration_s),
    }
    misses: list[dict[str, Any]] = []
    unstated: list[str] = []
    for name, (field, required) in minimums.items():
        if required is None:
            continue
        value = found[name]
        if value is None:
            unstated.append(name)
            continue
        floor = required * (1 - _FPS_REL_TOL) if name == "fps" else required
        if value < floor:
            misses.append({"field": field, "required": required, "found": value})
    if delivery.codecs is not None:
        if found["codec"] is None:
            unstated.append("codec")
        elif found["codec"] not in delivery.codecs:
            misses.append(
                {
                    "field": "codecs",
                    "required": delivery.codecs,
                    "found": found["codec"],
                }
            )

    return MetricResult(
        value=0.0 if misses else 1.0,
        unit="flag",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "misses": misses,
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
    On simulated footage a still scene repeats its frames exactly,
    so the result is marked and scoring keeps it ungraded.
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
    evidence: dict[str, Any] = {
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
    }
    if ctx.simulated:
        evidence["footage"] = "simulated"
    return MetricResult(
        value=float(100 * repeated / read),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence=evidence,
        support=support_for(
            list(range(n)), [(index, index + run) for index, run in counted]
        ),
    )


@metric(level=Level.STREAM, family=Family.VISION, label="still scene")
def still_scene_pct(ctx: StreamContext) -> MetricResult:
    """Share of frames inside stretches where the camera's picture does not move.

    Every second frame is compared with the frame 1 s before it,
    on 32x32 gray thumbnails, and the change is the 98th percentile of the
    difference once its median is removed.
    Each reading is then the median of those within 2 s either side.
    A stretch is a run of frames whose readings stay under 3 grey levels
    for at least 5 s, and sensor noise in a still scene reads under that.
    A reading with a blank frame is never still.
    The footage setting is not read: a still simulated scene reads 0,
    and is as much a stretch as a still real one.
    The metric needs every frame, so it is `not_applicable` outside a full scan.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure.

    Returns
    -------
    MetricResult
        `report_only` in percent, with each stretch's time range in evidence
        and as a support interval, or `not_applicable` outside a full scan,
        without a frame rate, or when the stream is shorter than a stretch.
    """

    # Step 1: the stream must be a camera read in full, long enough to hold a stretch.
    payload = _readable(ctx)
    if isinstance(payload, MetricResult):
        return payload
    if not ctx.full_frame_scan:
        return not_applicable("needs every frame; grade with --full-frame-scan")
    plan = _still_plan(ctx)
    if plan is None:
        return not_applicable("the stream has no frame rate to measure stillness in")
    grid_step, lag = plan
    numpy = load_numpy()
    assert numpy is not None
    timestamps = ctx.timestamps.to_numpy()
    n = len(timestamps)
    if timestamps[-1] - timestamps[0] < _STILL_MIN_S:
        return not_applicable(
            f"shorter than {_STILL_MIN_S:g} s, too short to hold a still stretch"
        )

    # Step 2: the readings from the shared full read.
    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if frames.n_read > n:
        return not_applicable(
            f"the footage decodes to {frames.n_read} frames"
            f" but the stream has {n} timestamps"
        )
    if not frames.still_changes:
        return not_applicable("too few frames decoded to compare over 1 s")

    # Step 3: smooth each reading over +-2 s, and call it still under the threshold.
    positions = numpy.array([index for index, _change in frames.still_changes])
    changes = numpy.array([change for _index, change in frames.still_changes])
    times = timestamps[positions]
    lows = numpy.searchsorted(times, times - _STILL_SMOOTH_S, side="left")
    highs = numpy.searchsorted(times, times + _STILL_SMOOTH_S, side="right")
    still = numpy.zeros(len(changes), dtype=bool)
    for k in range(len(changes)):
        if not numpy.isfinite(changes[k]):
            continue
        near = changes[lows[k] : highs[k]]
        near = near[numpy.isfinite(near)]
        still[k] = near.size >= _STILL_SMOOTH_MIN and numpy.median(near) < _STILL_CHANGE

    # Step 4: a run of still grid frames, with the second each reading looks back over,
    # is a stretch when it lasts long enough.
    stretches: list[tuple[int, int]] = []
    k = 0
    while k < len(still):
        if not still[k]:
            k += 1
            continue
        end = k
        while (
            end + 1 < len(still)
            and still[end + 1]
            and positions[end + 1] - positions[end] == grid_step
        ):
            end += 1
        first, last = int(positions[k]) - lag, int(positions[end])
        # Neighbouring runs can look back over the same frames.
        if stretches:
            first = max(first, stretches[-1][1])
        if timestamps[last] - timestamps[first] >= _STILL_MIN_S:
            stretches.append((first, last + 1))
        k = end + 1

    covered = sum(end - start for start, end in stretches)
    longest = round(
        max(
            (
                float(timestamps[end - 1] - timestamps[start])
                for start, end in stretches
            ),
            default=0.0,
        ),
        3,
    )
    return MetricResult(
        value=float(100 * covered / n),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_frames": n,
            "grid_step": grid_step,
            "lag": lag,
            "threshold": _STILL_CHANGE,
            "min_stretch_s": _STILL_MIN_S,
            "n_readings": len(changes),
            "n_blank_readings": int(numpy.isnan(changes).sum()),
            "stretches_s": [
                [
                    round(float(timestamps[start]), 3),
                    round(float(timestamps[end - 1]), 3),
                ]
                for start, end in stretches
            ],
            "longest_s": longest,
            "first_stretch_index": stretches[0][0] if stretches else None,
        },
        support=support_for(list(range(n)), stretches),
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

    One row per summary frame, the frames decoded at native resolution.
    Evidence frames, the rows that carry a hash,
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
            clipped_fraction=frames.dark_share[slot] + frames.white_share[slot],
            luma_sha256=digest if index in evidence else None,
        )
        for index, slot, pts, shape, laplacian, digest in zip(
            frames.summary,
            frames.summary_slots,
            frames.pts,
            frames.shape,
            frames.laplacian,
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
        missing_rows=sorted(set(frames.requested) - set(frames.summary)),
        frames=records,
        decoded_frames=frames.n_read,
        adjacent_pairs_examined=sum(len(d) for _start, d in frames.diffs),
        identical_adjacent_pairs=identical,
        sample_plan="full_scan_v1" if ctx.full_frame_scan else "evenly_spaced_v1",
        parameters={
            **ctx.vision.model_dump(exclude={"delivery"}),
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
