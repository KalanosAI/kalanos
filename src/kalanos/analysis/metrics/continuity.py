"""Camera footage checked against the robot's own signals.

These vision metrics read a camera stream
beside the state and action streams of its episode:

- cut_count takes the largest jumps in the control signals
  and confirms each in the footage;
  it decodes at most three windows of 14 frames, at any tier;
- dropped_chunk_count looks for a jump in the footage that no state stream shows,
  and reads every frame, so it runs only under a full frame scan,
  sharing the decode of the other vision metrics;
- video_control_lag_ms correlates the footage's motion with the control signals'
  to measure how far one runs ahead of the other,
  also under a full frame scan only.

State is every numeric series in the episode
that is not an action, not a reward and not the camera itself.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

# External
import polars as pl

# Internal
from kalanos.analysis.adapters.video import (
    DecodeFailed,
    DecoderUnavailable,
    SampledFrames,
)
from kalanos.analysis.localization import support_for
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
from kalanos.analysis.metrics.vision import (
    _SAMPLE_SIZE,
    _frame_rate,
    _frames,
    _readable,
    _stopped_result,
    _thumbnail,
)
from kalanos.analysis.models.domain import Kind
from kalanos.analysis.models.metrics import (
    Family,
    Level,
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.optional import load_numpy


if TYPE_CHECKING:
    import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# A step is compared with this many neighbours on each side.
_LOCAL_HALF = 6
# Footage spikes this many steps from either end are not searched:
# the first frames of most aloha wrist-camera episodes jump as the camera starts.
_EDGE = 10

# The control jumps checked in the footage, largest first.
_CUT_CANDIDATES = 3
# A candidate's window holds this many frames before it,
# and its own frames with this many less one after them.
_CUT_HALF = 7
# A candidate is a cut when its step moves the footage
# this many times more than its decoded neighbours.
# Calibrated on lerobot/aloha_static_towel only:
# 1.25 x the largest ratio over 20 clean calibration episodes,
# the larger of its two cameras.
_CUT_RATIO = 3.6

# A footage spike is a dropped chunk when it stands this many times above its neighbours
# more than the state's spike at the same step does.
# Calibrated on lerobot/aloha_static_towel only:
# 1.25 x the largest score over 20 clean calibration episodes,
# the larger of its two cameras (7.92 on cam_high),
# each score taken from the 64x64 thumbnails this module reads.
_DROP_RATIO = 9.9
# The furthest lag, in frames either way, that the footage is matched at.
_MAX_LAG = 10
# Kalanos does not know which arm a camera watches,
# so the lag is read from the control channels that follow the footage best.
_TOP_CHANNELS = 3
# Motion is detrended by its moving average over this many steps.
_DETREND = 25
# A lag needs at least this many steps of overlap to be correlated at.
_MIN_OVERLAP = 10

# Fewer nonzero footage steps than this leave too few neighbours to judge a spike by.
_DROP_MIN_STEPS = 14


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class _Control:
    """The numeric readings of one control stream.

    Attributes
    ----------
    times : numpy.ndarray
        Each row's timestamp.
    values : numpy.ndarray
        `(rows, channels)` float64 readings, NaN where a reading is missing.
    """

    times: "np.ndarray"
    values: "np.ndarray"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _controls(ctx: StreamContext) -> tuple[list[_Control], list[_Control]]:
    """Read the state and action streams beside the camera.

    A stream qualifies when it is a numeric series with at least two rows
    and is not a `reward.*` stream.
    `action.*` streams are action, and every other qualifying stream is state.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream, with its episode's streams.

    Returns
    -------
    tuple of (list of _Control, list of _Control)
        The state streams and the action streams.
    """

    state: list[_Control] = []
    action: list[_Control] = []
    for stream in ctx.episode_streams or []:
        if stream is ctx.stream or stream.kind is not Kind.SERIES:
            continue
        if stream.payload is None or not stream.channels:
            continue
        # A reward or done flag holds still and then jumps
        # with the task rather than with the robot,
        # so its jumps would take cut candidates from the control signals.
        if stream.taxonomy_type.startswith("reward."):
            continue
        # A control payload may be lazy, so this reads it again;
        # grade_stream fetches it for the stream's own channels too.
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
        control = _Control(times=stream.timestamps.to_numpy(), values=values)
        if stream.taxonomy_type.startswith("action."):
            action.append(control)
        else:
            state.append(control)
    return state, action


def _steps(values: "np.ndarray") -> "np.ndarray":
    """Mean |delta| over each step's finite readings, 0 where a step has none.

    Parameters
    ----------
    values : numpy.ndarray
        `(rows, channels)` readings.

    Returns
    -------
    numpy.ndarray
        `(rows - 1,)`; step i runs from row i to row i + 1.
    """

    numpy = load_numpy()
    assert numpy is not None

    steps = numpy.abs(numpy.diff(values, axis=0))
    finite = numpy.isfinite(steps)
    readings = finite.sum(axis=1)
    totals = numpy.where(finite, steps, 0.0).sum(axis=1)
    return numpy.where(readings > 0, totals / numpy.maximum(readings, 1), 0.0)


def _robust_z(x: "np.ndarray") -> "np.ndarray":
    """Robust z-score by median and median absolute deviation.

    When more than half the values equal the median, the MAD is 0,
    and the median of the nonzero deviations stands in for it.

    Parameters
    ----------
    x : numpy.ndarray
        Values.

    Returns
    -------
    numpy.ndarray
        `(x - median) / (1.4826 * MAD)`, all 0 when every value is the median.
    """

    numpy = load_numpy()
    assert numpy is not None

    median = numpy.median(x)
    deviations = numpy.abs(x - median)
    mad = float(numpy.median(deviations))
    # A stream that mostly holds still, such as a gripper,
    # has a MAD of 0, and a floor near 0 would score each of its rare changes
    # far above any other stream's jumps, crowding them out of the cut candidates.
    if mad == 0:
        moved = deviations[deviations > 0]
        if not len(moved):
            return numpy.zeros(len(x))
        mad = float(numpy.median(moved))
    return (x - median) / (1.4826 * mad)


def _local_ratio(
    x: "np.ndarray", half: int = _LOCAL_HALF, skip_zero: bool = False
) -> "np.ndarray":
    """Each value over the median of its neighbours, the value itself left out.

    The denominator is floored at the series' 25th percentile,
    so a near-zero still stretch cannot turn noise into a large ratio.

    Parameters
    ----------
    x : numpy.ndarray
        Values.
    half : int
        Neighbours on each side.
    skip_zero : bool
        Leave exact zeros out of the neighbours and the floor, and score them 0.
        A repeated frame reads as a zero luma step,
        so a camera that repeats every other frame
        would otherwise pull the denominator to zero.

    Returns
    -------
    numpy.ndarray
        Ratios, same length as `x`.
    """

    numpy = load_numpy()
    assert numpy is not None

    n = len(x)
    keep = x != 0 if skip_zero else numpy.ones(n, dtype=bool)
    if not keep.any():
        return numpy.zeros(n)
    floor = max(float(numpy.percentile(x[keep], 25)), 1e-9)
    out = numpy.zeros(n)
    for i in range(n):
        if not keep[i]:
            continue
        lo, hi = max(0, i - half), i + 1 + half
        neighbours = numpy.concatenate(
            [x[lo:i][keep[lo:i]], x[i + 1 : hi][keep[i + 1 : hi]]]
        )
        typical = float(numpy.median(neighbours)) if len(neighbours) else floor
        out[i] = x[i] / max(typical, floor)
    return out


def _inner(score: "np.ndarray") -> "np.ndarray":
    """Zero a per-step score within `_EDGE` steps of either end.

    Parameters
    ----------
    score : numpy.ndarray
        Per-step scores.

    Returns
    -------
    numpy.ndarray
        A float copy with both ends zeroed.
    """

    numpy = load_numpy()
    assert numpy is not None

    out = numpy.array(score, dtype=float)
    out[:_EDGE] = 0
    out[len(out) - _EDGE :] = 0
    return out


def _frame_at(video_times: "np.ndarray", times: "np.ndarray") -> "np.ndarray":
    """The index of the video timestamp nearest each of `times`.

    Control and camera may run at different rates,
    so a control row is placed on the camera's clock by its timestamp.

    Parameters
    ----------
    video_times : numpy.ndarray
        The camera's ascending frame timestamps.
    times : numpy.ndarray
        The timestamps to place.

    Returns
    -------
    numpy.ndarray
        One frame index per timestamp in `times`.
    """

    numpy = load_numpy()
    assert numpy is not None

    after = numpy.clip(numpy.searchsorted(video_times, times), 1, len(video_times) - 1)
    before = after - 1
    nearer_before = times - video_times[before] <= video_times[after] - times
    return numpy.where(nearer_before, before, after)


def _frames_spanned(
    video_times: "np.ndarray", times: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """The frames whose incoming step overlaps each step between `times`.

    A jump between rows r - 1 and r of a control
    happens somewhere in `(times[r - 1], times[r]]`,
    so in the footage it lands on the step into one of the frames
    from the first after `times[r - 1]` to the first at or after `times[r]`.
    The frame nearest `times[r]` can be the one before the jump
    when the two clocks run at different rates.

    Parameters
    ----------
    video_times : numpy.ndarray
        The camera's ascending frame timestamps.
    times : numpy.ndarray
        A control's ascending row timestamps.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The first and last frame of each step's span, `len(times) - 1` of each,
        or -1 for both where the step lies wholly before the second frame
        or after the last.
    """

    numpy = load_numpy()
    assert numpy is not None

    n = len(video_times)
    # Timestamps written at a shared rate differ by float rounding,
    # which must not widen a span by a frame on either side.
    tolerance = 1e-3 * float(numpy.median(numpy.diff(video_times))) if n > 1 else 0.0
    first = numpy.searchsorted(video_times, times[:-1] + tolerance, side="right")
    last = numpy.searchsorted(video_times, times[1:] - tolerance, side="left")
    last = numpy.maximum(last, first)
    # Frame 0 has no step into it.
    outside = (last < 1) | (first > n - 1)
    first = numpy.clip(first, 1, n - 1)
    last = numpy.clip(last, 1, n - 1)
    return numpy.where(outside, -1, first), numpy.where(outside, -1, last)


def _on_video_steps(
    steps: "np.ndarray", times: "np.ndarray", video_times: "np.ndarray"
) -> "np.ndarray":
    """Resample a control's per-step series onto the camera's steps.

    Video step j, from frame j to frame j + 1,
    takes the control step whose end row is nearest frame j + 1.

    Parameters
    ----------
    steps : numpy.ndarray
        One value per control step; step i ends at row i + 1.
    times : numpy.ndarray
        The control's row timestamps.
    video_times : numpy.ndarray
        The camera's frame timestamps.

    Returns
    -------
    numpy.ndarray
        One value per video step, `len(video_times) - 1` of them.
    """

    numpy = load_numpy()
    assert numpy is not None

    rows = _frame_at(times, video_times[1:])
    return steps[numpy.maximum(rows - 1, 0)]


def _full_steps(ctx: StreamContext) -> "np.ndarray | MetricResult":
    """Every step's luma difference from the shared full read, or why there is none.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream.

    Returns
    -------
    numpy.ndarray or MetricResult
        The unscaled mean absolute difference of each pair of neighbouring
        thumbnails, `n - 1` of them, or a result that is not graded.
    """

    if not ctx.full_frame_scan:
        return not_applicable("needs every frame; grade with --full-frame-scan")
    frames = _frames(ctx)
    if isinstance(frames, MetricResult):
        return frames
    if frames.stopped is not None:
        return _stopped_result(frames.stopped)
    if not frames.luma_steps:
        return not_applicable("too few frames decoded to compare neighbours")
    # A full scan reads one window, every frame.
    [(_start, steps)] = frames.luma_steps
    return steps


def _detrended(x: "np.ndarray", width: int = _DETREND) -> "np.ndarray":
    """Remove the moving average of `width` steps from a series.

    Parameters
    ----------
    x : numpy.ndarray
        The series.
    width : int
        The moving average's width in steps.

    Returns
    -------
    numpy.ndarray
        `x` minus its local mean, the mean taken over the steps inside the series.
    """

    numpy = load_numpy()
    assert numpy is not None

    # The centre of the full convolution, which mode="same" returns
    # only for a series at least as long as the kernel.
    kernel = numpy.ones(width)
    centre = slice((width - 1) // 2, (width - 1) // 2 + len(x))
    total = numpy.convolve(x, kernel)[centre]
    count = numpy.convolve(numpy.ones(len(x)), kernel)[centre]
    return x - total / numpy.maximum(count, 1)


def _video_motion(m: "np.ndarray") -> "np.ndarray":
    """The footage's detrended motion per step, robust to repeated frames.

    A repeated frame reads as a zero step,
    so each step takes the largest of itself and its two neighbours before detrending.

    Parameters
    ----------
    m : numpy.ndarray
        The luma difference of each step.

    Returns
    -------
    numpy.ndarray
        The detrended motion, one value per step.
    """

    numpy = load_numpy()
    assert numpy is not None

    padded = numpy.concatenate([[m[0]], m, [m[-1]]])
    largest = numpy.maximum(numpy.maximum(padded[:-2], padded[1:-1]), padded[2:])
    return _detrended(largest)


def _control_motion(
    controls: list[_Control], video_times: "np.ndarray"
) -> "np.ndarray":
    """Every channel's detrended step size, on the camera's steps.

    Parameters
    ----------
    controls : list of _Control
        The control streams of one side, state or action.
    video_times : numpy.ndarray
        The camera's frame timestamps.

    Returns
    -------
    numpy.ndarray
        `(len(video_times) - 1, channels)`, the channels of every stream pooled.
        A missing reading moves nothing.
    """

    numpy = load_numpy()
    assert numpy is not None

    columns = []
    for control in controls:
        steps = numpy.abs(numpy.diff(control.values, axis=0))
        steps = numpy.where(numpy.isfinite(steps), steps, 0.0)
        for channel in range(steps.shape[1]):
            on_video = _on_video_steps(steps[:, channel], control.times, video_times)
            columns.append(_detrended(on_video))
    return numpy.stack(columns, axis=1)


def _curve(v: "np.ndarray", c: "np.ndarray") -> "np.ndarray":
    """Correlation of `v[t]` with `c[t + lag]` at every lag within `_MAX_LAG`.

    Parameters
    ----------
    v, c : numpy.ndarray
        Equal-length series.

    Returns
    -------
    numpy.ndarray
        One correlation per lag from `-_MAX_LAG` to `_MAX_LAG`,
        NaN where the overlap is too short or either side is constant.
    """

    numpy = load_numpy()
    assert numpy is not None

    n = len(v)
    lags = numpy.arange(-_MAX_LAG, _MAX_LAG + 1)
    corr = numpy.full(len(lags), numpy.nan)
    for j, lag in enumerate(lags):
        t = numpy.arange(max(0, -lag), min(n, n - lag))
        if len(t) < _MIN_OVERLAP:
            continue
        a, b = v[t], c[t + lag]
        if a.std() == 0 or b.std() == 0:
            continue
        corr[j] = float(numpy.corrcoef(a, b)[0, 1])
    return corr


def _xcorr_lag(v: "np.ndarray", motion: "np.ndarray") -> dict[str, float]:
    """The lag at which the footage's motion best matches the control's.

    Positive means the footage leads: `v[t]` matches the control at `t + lag`.
    The curve is the mean over the `_TOP_CHANNELS` channels whose peak is highest,
    and the peak is refined to a fraction of a step on a parabola.

    Parameters
    ----------
    v : numpy.ndarray
        The footage's detrended motion per step.
    motion : numpy.ndarray
        `(steps, channels)` detrended control motion on the same steps.

    Returns
    -------
    dict of str to float
        `lag` in frames, the `peak` correlation,
        and its `margin` over the best correlation at least 2 lags away;
        all NaN when no channel correlates at any lag.
    """

    numpy = load_numpy()
    assert numpy is not None

    curves = [_curve(v, motion[:, j]) for j in range(motion.shape[1])]
    peaks = [
        float(numpy.nanmax(curve)) if not numpy.isnan(curve).all() else -numpy.inf
        for curve in curves
    ]
    best = sorted(range(len(curves)), key=lambda j: -peaks[j])[:_TOP_CHANNELS]
    stack = numpy.stack([curves[j] for j in best])
    if numpy.isnan(stack).all():
        return {"lag": numpy.nan, "peak": numpy.nan, "margin": numpy.nan}
    # nanmean of an all-NaN lag warns; -1 is the lowest correlation anyway.
    counts = (~numpy.isnan(stack)).sum(axis=0)
    totals = numpy.where(numpy.isnan(stack), 0.0, stack).sum(axis=0)
    filled = numpy.where(counts > 0, totals / numpy.maximum(counts, 1), -1.0)

    lags = numpy.arange(-_MAX_LAG, _MAX_LAG + 1)
    j = int(numpy.argmax(filled))
    lag = float(lags[j])
    if 0 < j < len(lags) - 1:
        y0, y1, y2 = filled[j - 1], filled[j], filled[j + 1]
        curvature = y0 - 2 * y1 + y2
        if curvature < 0:
            lag += 0.5 * (y0 - y2) / curvature
    away = [filled[i] for i in range(len(lags)) if abs(i - j) >= 2]
    second = max(away) if away else -1.0
    return {
        "lag": float(lag),
        "peak": float(filled[j]),
        "margin": float(filled[j] - second),
    }


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


def _window_thumbnails(
    payload: SampledFrames, windows: list[tuple[int, int]]
) -> "tuple[dict[int, np.ndarray], DecodeFailed | None]":
    """Decode `windows` of `payload` as freeze thumbnails, keyed by frame index.

    Parameters
    ----------
    payload : SampledFrames
        The camera's frames.
    windows : list of tuple of (int, int)
        Ascending, non-overlapping `[start, end)` windows.

    Returns
    -------
    tuple of (dict of int to numpy.ndarray, DecodeFailed or None)
        The thumbnails decoded, and the failure that stopped the read, if any.

    Raises
    ------
    DecoderUnavailable
        If no decoder is installed.
    """

    thumbnails: dict[int, np.ndarray] = {}
    counts = [0] * len(windows)
    try:
        for position, frame, _full, _pts in payload.gray_windows(windows, _SAMPLE_SIZE):
            # A window is only ever cut short at its end,
            # so the frames it did yield are its first ones.
            thumbnails[windows[position][0] + counts[position]] = _thumbnail(frame)
            counts[position] += 1
    except DecodeFailed as exc:
        return thumbnails, exc
    return thumbnails, None


def _window_ratio(
    thumbnails: "dict[int, np.ndarray]", start: int, end: int, at: int
) -> float:
    """How much more the step into frame `at` moves the footage than its window's.

    The step's mean absolute difference
    over the median of the window's other nonzero steps,
    floored at their 25th percentile.

    Parameters
    ----------
    thumbnails : dict of int to numpy.ndarray
        The decoded thumbnails, keyed by frame index.
    start, end : int
        The `[start, end)` window holding `at`.
    at : int
        The frame whose incoming step is scored.

    Returns
    -------
    float
        The ratio, 0 when the window lacks the frames or the step does not move.
    """

    numpy = load_numpy()
    assert numpy is not None

    indices = [i for i in range(start, end) if i in thumbnails and i - 1 in thumbnails]
    if at not in indices:
        return 0.0
    steps = numpy.array(
        [float(numpy.abs(thumbnails[i] - thumbnails[i - 1]).mean()) for i in indices]
    )
    step = steps[indices.index(at)]
    if step == 0:
        return 0.0
    others = numpy.delete(steps, indices.index(at))
    others = others[others != 0]
    floor = max(float(numpy.percentile(steps[steps != 0], 25)), 1e-9)
    typical = float(numpy.median(others)) if len(others) else floor
    return float(step / max(typical, floor))


@metric(level=Level.STREAM, family=Family.VISION, label="cuts in footage")
def cut_count(ctx: StreamContext) -> MetricResult:
    """Count the cuts in camera footage: stretches removed from video and control alike.

    A cut leaves both the footage and the control signals jumping at one step.
    The `_CUT_CANDIDATES` largest robust z-scores of the per-step mean |delta|
    of every state and action stream are placed on the camera's clock,
    each on the frames whose incoming step overlaps it in time:
    one frame when the clocks share a rate, more when they do not.
    A window from `_CUT_HALF` frames before each to `_CUT_HALF - 1` after it is decoded,
    and a candidate is a cut when the 64x64 gray mean absolute difference
    of the step into one of its frames
    exceeds `_CUT_RATIO` times the median of the window's other nonzero steps.
    So at most `_CUT_CANDIDATES` cuts are counted per stream.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure, with its episode's streams.

    Returns
    -------
    MetricResult
        `report_only` with the count of cuts,
        each as a one-frame support interval at the first frame after it,
        or `not_applicable` when the stream cannot be sampled,
        has no state or action stream beside it, holds too few frames,
        or no frame of the windows decodes.
    """

    # Step 1: the camera, and the control signals a cut also shows in.
    payload = _readable(ctx)
    if isinstance(payload, MetricResult):
        return payload
    numpy = load_numpy()
    assert numpy is not None
    state, action = _controls(ctx)
    if not state and not action:
        return not_applicable("no state or action stream to locate a cut against")
    n = ctx.n_samples
    if n < 2 * _CUT_HALF + 1:
        return not_applicable("too few frames to confirm a cut in")

    # Step 2: the largest control jumps, each on the frames its step spans.
    video_times = ctx.timestamps.to_numpy()
    zs, firsts, lasts = [], [], []
    for control in state + action:
        first, last = _frames_spanned(video_times, control.times)
        zs.append(_robust_z(_steps(control.values)))
        firsts.append(first)
        lasts.append(last)
    z, first, last = (numpy.concatenate(parts) for parts in (zs, firsts, lasts))
    spans: dict[int, int] = {}
    # Largest z first, ties to the earlier frame; one candidate per first frame.
    for k in numpy.lexsort((first, -z)):
        if len(spans) == _CUT_CANDIDATES:
            break
        if first[k] >= 0 and numpy.isfinite(z[k]) and int(first[k]) not in spans:
            spans[int(first[k])] = int(last[k])
    candidates = list(spans)

    # Step 3: decode a short window around each candidate.
    windows = _merge([(p - _CUT_HALF, spans[p] + _CUT_HALF) for p in candidates], n)
    try:
        thumbnails, failure = _window_thumbnails(payload, windows)
    except DecoderUnavailable as exc:
        return not_applicable(str(exc))
    if not thumbnails:
        return not_applicable(str(failure) if failure else "no frame could be decoded")

    # Step 4: a candidate is a cut when the footage jumps in its span too.
    # One jump is one cut, so a span scores only its frame that jumps most.
    ratios: dict[int, float] = {}
    for p in candidates:
        start, end = next(w for w in windows if w[0] <= p < w[1])
        spanned = {
            q: _window_ratio(thumbnails, start, end, q) for q in range(p, spans[p] + 1)
        }
        q = max(spanned, key=lambda frame: spanned[frame])
        ratios[q] = max(ratios.get(q, 0.0), spanned[q])
    cuts = sorted(q for q, ratio in ratios.items() if ratio > _CUT_RATIO)
    best_index = max(ratios, key=lambda p: ratios[p]) if ratios else None

    return MetricResult(
        value=float(len(cuts)),
        unit="cuts",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "cut_indices": cuts,
            "best_ratio": float(ratios[best_index]) if best_index is not None else None,
            "best_index": best_index,
            "candidates": candidates,
            "n_decoded": len(thumbnails),
            "threshold": _CUT_RATIO,
            # A read that broke partway leaves its later candidates unscored.
            "decode_error": str(failure) if failure else None,
        },
        support=support_for(list(range(n)), [(p, p + 1) for p in cuts]),
    )


@metric(level=Level.STREAM, family=Family.VISION, label="dropped video chunks")
def dropped_chunk_count(ctx: StreamContext) -> MetricResult:
    """Find a chunk of footage dropped while the timestamps stayed uniform.

    A dropped chunk makes the footage jump at one step while the state does not.
    Each step's 64x64 gray mean absolute difference is divided by the median
    of its 6 nonzero neighbours on each side,
    and that ratio by the largest such ratio of any state stream's
    per-step mean |delta| at the same step, floored at 1.
    The step scoring highest, more than `_EDGE` steps from either end,
    is a drop when its score exceeds `_DROP_RATIO`.
    So at most one drop is reported per episode.
    A cut moves the state too, so it scores low here; `cut_count` finds it.
    A drop that leaves the footage short of its timestamps
    is `frame_count_vs_timebase`'s to find.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure, with its episode's streams.

    Returns
    -------
    MetricResult
        `report_only` with 1 for a drop or 0,
        the drop as a one-frame support interval at the first frame after it,
        or `not_applicable` when the stream cannot be sampled,
        has no state stream beside it, is not read in full,
        has too few moving frames,
        or no step lies more than `_EDGE` steps from either end.
    """

    # Step 1: the camera, and the state that tells a drop from a cut.
    payload = _readable(ctx)
    if isinstance(payload, MetricResult):
        return payload
    numpy = load_numpy()
    assert numpy is not None
    state, _action = _controls(ctx)
    if not state:
        return not_applicable("no state stream to tell a dropped chunk from a cut")

    # Step 2: every step of the footage, from the shared full read.
    m = _full_steps(ctx)
    if isinstance(m, MetricResult):
        return m
    if int((m != 0).sum()) < _DROP_MIN_STEPS:
        return not_applicable("too few moving frames to judge a jump against")
    if len(m) <= 2 * _EDGE:
        return not_applicable("too few frames to search between the episode's ends")

    # Step 3: footage spikes the state does not share.
    video_times = ctx.timestamps.to_numpy()[: len(m) + 1]
    footage = _local_ratio(m, skip_zero=True)
    control = numpy.ones(len(m))
    for stream in state:
        steps = _on_video_steps(_steps(stream.values), stream.times, video_times)
        control = numpy.maximum(control, _local_ratio(steps))
    score = _inner(footage / control)

    # Step 4: the highest score, a drop when it clears the threshold.
    i = int(numpy.argmax(score))
    found = bool(score[i] > _DROP_RATIO)
    return MetricResult(
        value=1.0 if found else 0.0,
        unit="chunks",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "drop_index": i + 1 if found else None,
            "best_ratio": float(score[i]),
            "best_index": i + 1,
            "threshold": _DROP_RATIO,
            "n_steps": len(m),
        },
        support=support_for(
            list(range(ctx.n_samples)), [(i + 1, i + 2)] if found else []
        ),
    )


@metric(level=Level.STREAM, family=Family.VISION, label="video lag behind control")
def video_control_lag_ms(ctx: StreamContext) -> MetricResult:
    """Measure how far camera footage runs ahead of the robot's control signals.

    The footage's motion per step (its 64x64 gray mean absolute difference,
    the largest of each step and its neighbours, detrended over 25 steps)
    is correlated with each control channel's detrended step size
    at every lag within `_MAX_LAG` frames.
    The lag is the peak of the mean curve of the `_TOP_CHANNELS` channels
    that correlate best, refined on a parabola.
    It is read against state when a state channel correlates with the footage,
    otherwise against action.

    Positive means the footage leads the control, negative that it trails.
    Every clean episode of lerobot/aloha_static_towel reads a lag,
    its footage trailing state by about 3 frames,
    so the value describes the rig.
    A single episode's peak stands only a little above the rest of its curve,
    so one value can lock onto a wrong lag.

    Parameters
    ----------
    ctx : StreamContext
        The camera stream to measure, with its episode's streams.

    Returns
    -------
    MetricResult
        `report_only` with the lag in milliseconds,
        or `not_applicable` when the stream cannot be sampled,
        has no state or action stream beside it, is not read in full,
        has no frame rate, or no control channel correlates with it.
    """

    # Step 1: the camera, its controls and every step of its footage.
    payload = _readable(ctx)
    if isinstance(payload, MetricResult):
        return payload
    state, action = _controls(ctx)
    if not state and not action:
        return not_applicable("no state or action stream to time the footage against")
    m = _full_steps(ctx)
    if isinstance(m, MetricResult):
        return m
    fps = _frame_rate(ctx)
    if fps is None:
        return not_applicable("the stream has no frame rate to measure a lag in")

    # Step 2: the lag against each side present.
    v = _video_motion(m)
    video_times = ctx.timestamps.to_numpy()[: len(m) + 1]
    results = {
        side: _xcorr_lag(v, _control_motion(controls, video_times))
        for side, controls in (("state", state), ("action", action))
        if controls
    }

    # Step 3: state's lag when state correlates, action's otherwise.
    finite = [side for side in results if not math.isnan(results[side]["lag"])]
    if not finite:
        return not_applicable("no control channel correlates with the video")
    reference = finite[0]
    chosen = results[reference]

    def lag_ms(side: str) -> float | None:
        """The lag against `side` in ms, `None` where it is absent or undefined."""

        lag = results[side]["lag"] if side in results else math.nan
        return None if math.isnan(lag) else lag * 1000 / fps

    return MetricResult(
        value=chosen["lag"] * 1000 / fps,
        unit="ms",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "reference": reference,
            "lag_frames": chosen["lag"],
            "peak_corr": chosen["peak"],
            "peak_margin": chosen["margin"],
            "lag_state_ms": lag_ms("state"),
            "lag_action_ms": lag_ms("action"),
            "fps": fps,
            "n_steps": len(m),
        },
    )
