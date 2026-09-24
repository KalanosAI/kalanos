"""The motion family: jerk, chatter, saturation, limit proximity, drift, vibration,
torque and energy.

Whether the recorded motion is physically plausible. Every metric here gates
on a `proprio.*` taxonomy type, since values like normalised jerk only mean
something once a signal is known to be a joint position, velocity or torque.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import statistics
from typing import cast

# External
import polars as pl

# Internal
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
from kalanos.analysis.models.domain import FramePayload
from kalanos.analysis.models.metrics import (
    ChannelContext,
    EpisodeContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
    Requires,
    StreamContext,
)
from kalanos.analysis.optional import load_numpy


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Taxonomy types the metrics below gate on.
# fmt: off
_JOINT_POSITION = "proprio.joint_position"
_JOINT_VELOCITY = "proprio.joint_velocity"
_JOINT_TORQUE   = "proprio.joint_torque"
# fmt: om

# still_drift tuning: how much of the recording counts as "the tail",
# and how settled that tail's step size must be, relative to the whole
# recording's, to call the joint static.
_STILL_TAIL_FRACTION = 0.2
_STILL_MIN_TAIL_SAMPLES = 5
_STILL_STEP_RATIO = 0.05

# hf_vibration tuning: above this, spectral energy belongs to the mechanism,
# not to deliberate motion.
_HF_CUTOFF_HZ = 20.0

# Structural requirements, one per metric below.
_REQUIRES_JERK = Requires(
    regular_sampling=True, min_samples=5, taxonomy=[_JOINT_POSITION],
)
_REQUIRES_CHATTER = Requires(
    regular_sampling=True, min_samples=5, taxonomy=[_JOINT_VELOCITY],
)
_REQUIRES_STILL_DRIFT     = Requires(min_samples=8, taxonomy=[_JOINT_POSITION])
_REQUIRES_VEL_SATURATION  = Requires(min_samples=2, taxonomy=[_JOINT_VELOCITY])
_REQUIRES_LIMIT_PROXIMITY = Requires(min_samples=2, taxonomy=[_JOINT_POSITION])
_REQUIRES_HF_VIBRATION = Requires(
    regular_sampling=True,
    min_samples=32,
    taxonomy=[_JOINT_TORQUE],
)
_REQUIRES_TORQUE_STAT = Requires(min_samples=2, taxonomy=[_JOINT_TORQUE])

# No structural requirement: the decorator refuses one at Level.EPISODE,
# so energy_proxy checks its own timebase and payload shape itself, below.
_REQUIRES_ENERGY_PROXY = Requires(taxonomy=[_JOINT_TORQUE, _JOINT_VELOCITY])


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _median_abs_step(values: list[float]) -> float:
    """Median absolute step-to-step change across a list of values."""

    steps = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
    return statistics.median(steps) if steps else 0.0


def _median_gap(timestamps: list[float]) -> float | None:
    """Median gap between consecutive timestamps, or `None` when there are none."""

    gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
    return statistics.median(gaps) if gaps else None


def _normalized_jerk(
    ctx: StreamContext,
) -> tuple[list[float], int, float] | str:
    """Compute every channel's per-sample normalised jerk, or say why none exists.

    Divides each channel by its own std before the third difference, so
    amplitude doesn't matter. A channel with too few values or zero
    variance has no scale to normalise by and is skipped.

    Parameters
    ----------
    ctx : StreamContext
        The joint-position stream to measure.

    Returns
    -------
    tuple[list[float], int, float] or str
        The flattened normalised jerk values, how many channels contributed, and
        the median timestep in seconds; or a reason string when the payload is not a
        `FramePayload`, the median gap is zero, or no channel has a usable scale.
    """

    if not isinstance(ctx.payload, FramePayload):
        return "payload is not a FramePayload; there is no series to differentiate"

    median_dt = _median_gap(ctx.timestamps.to_list())
    if median_dt is None:
        return "fewer than two timestamps to take a gap over"
    if median_dt <= 0:
        return "median gap is zero"

    frame = ctx.payload.frame
    jerks: list[float] = []
    n_channels = 0
    for channel in ctx.stream.channels:
        values = [value for value in frame[channel.name].to_list() if value is not None]
        if len(values) < 4:
            continue
        std = statistics.pstdev(values)
        if std == 0:
            continue
        normalized = [value / std for value in values]
        third_diff = [
            normalized[index + 3]
            - 3 * normalized[index + 2]
            + 3 * normalized[index + 1]
            - normalized[index]
            for index in range(len(normalized) - 3)
        ]
        jerks.extend(value / median_dt**3 for value in third_diff)
        n_channels += 1

    if n_channels == 0:
        return "no channel has non-zero standard deviation to normalise by"

    return jerks, n_channels, median_dt


@metric(level=Level.STREAM, family=Family.MOTION, requires=_REQUIRES_JERK)
def mean_jerk_norm(ctx: StreamContext) -> MetricResult:
    """RMS of the normalised third derivative of joint position.

    Parameters
    ----------
    ctx : StreamContext
        The joint-position stream to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the payload is not a `FramePayload`
        - the median gap is zero
        - no channel has a usable scale
        `report_only` otherwise.
    """

    computed = _normalized_jerk(ctx)
    if isinstance(computed, str):
        return not_applicable(computed)
    jerks, n_channels, median_dt = computed

    rms = math.sqrt(sum(value**2 for value in jerks) / len(jerks))
    return MetricResult(
        value=rms,
        unit="normalised",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_samples": ctx.n_samples,
            "n_channels": n_channels,
            "median_dt_s": median_dt,
        },
    )


@metric(level=Level.STREAM, family=Family.MOTION, requires=_REQUIRES_JERK)
def max_abs_jerk(ctx: StreamContext) -> MetricResult:
    """Maximum absolute normalised third derivative of joint position.

    Parameters
    ----------
    ctx : StreamContext
        The joint-position stream to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the payload is not a `FramePayload`
        - the median gap is zero
        - no channel has a usable scale
        `report_only` otherwise.
    """

    computed = _normalized_jerk(ctx)
    if isinstance(computed, str):
        return not_applicable(computed)
    jerks, n_channels, median_dt = computed

    return MetricResult(
        value=max(abs(value) for value in jerks),
        unit="normalised",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_samples": ctx.n_samples,
            "n_channels": n_channels,
            "median_dt_s": median_dt,
        },
    )


@metric(level=Level.STREAM, family=Family.MOTION, requires=_REQUIRES_CHATTER)
def action_chatter(ctx: StreamContext) -> MetricResult:
    """Mean absolute step-to-step change in velocity, normalised by its own spread.

    Parameters
    ----------
    ctx : StreamContext
        The joint-velocity stream to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the payload is not a `FramePayload`
        - no channel has non-zero standard deviation
        `report_only` otherwise.
    """

    if not isinstance(ctx.payload, FramePayload):
        return not_applicable(
            "payload is not a FramePayload; there is no series to difference"
        )

    frame = ctx.payload.frame
    ratios = []
    for channel in ctx.stream.channels:
        values = [value for value in frame[channel.name].to_list() if value is not None]
        if len(values) < 2:
            continue
        std = statistics.pstdev(values)
        if std == 0:
            continue
        steps = [abs(b - a) for a, b in zip(values, values[1:], strict=False)]
        ratios.append((sum(steps) / len(steps)) / std)

    if not ratios:
        return not_applicable(
            "no channel has non-zero standard deviation to normalise by"
        )

    return MetricResult(
        value=sum(ratios) / len(ratios),
        unit="normalised",
        status=MetricStatus.REPORT_ONLY,
        evidence={"n_channels": len(ratios), "n_samples": ctx.n_samples},
    )


@metric(level=Level.STREAM, family=Family.MOTION, requires=_REQUIRES_STILL_DRIFT)
def still_drift(ctx: StreamContext) -> MetricResult:
    """How far joint readings wander in the recording's own settled tail.

    Parameters
    ----------
    ctx : StreamContext
        The joint-position stream to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the payload is not a `FramePayload`
        - the recording is too short for a tail distinct from the whole
        - no channel has enough non-null values for a tail
        - the tail is not settled relative to the whole
        `report_only` otherwise.
    """

    if not isinstance(ctx.payload, FramePayload):
        return not_applicable(
            "payload is not a FramePayload; there is no series to check"
        )

    n_samples = ctx.n_samples
    tail_samples = max(_STILL_MIN_TAIL_SAMPLES, round(n_samples * _STILL_TAIL_FRACTION))
    if tail_samples >= n_samples:
        return not_applicable(
            "recording is too short for a tail distinct from the whole"
        )

    frame = ctx.payload.frame
    channel_values = {
        name: [value for value in frame[name].to_list() if value is not None]
        for name in (channel.name for channel in ctx.stream.channels)
    }
    channel_values = {
        name: values
        for name, values in channel_values.items()
        if len(values) > tail_samples
    }
    if not channel_values:
        return not_applicable("no channel has enough non-null values for a tail")

    whole_step = statistics.median(
        _median_abs_step(values) for values in channel_values.values()
    )
    tail_step = statistics.median(
        _median_abs_step(values[-tail_samples:]) for values in channel_values.values()
    )
    tail_step_ratio = tail_step / whole_step if whole_step > 0 else 0.0

    if whole_step > 0 and tail_step_ratio > _STILL_STEP_RATIO:
        return not_applicable(
            "recording does not end static; the tail is not settled relative to "
            "the whole"
        )

    drifts = {
        name: abs(values[-1] - values[-tail_samples])
        for name, values in channel_values.items()
    }
    drifting_channel = max(drifts, key=lambda name: drifts[name])

    return MetricResult(
        value=drifts[drifting_channel],
        unit="unit",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "tail_samples": tail_samples,
            "tail_step_ratio": tail_step_ratio,
            "drifting_channel": drifting_channel,
        },
    )


@metric(level=Level.CHANNEL, family=Family.MOTION, requires=_REQUIRES_VEL_SATURATION)
def vel_saturation_pct(ctx: ChannelContext) -> MetricResult:
    """Share of samples at or above 90% of this channel's own observed maximum.

    Parameters
    ----------
    ctx : ChannelContext
        The joint-velocity channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - no non-null sample survives
        - the observed maximum is zero
        `report_only` otherwise.
    """

    abs_values = [abs(value) for value in ctx.values.to_list() if value is not None]
    if not abs_values:
        return not_applicable("no non-null samples to measure against")

    max_abs = max(abs_values)
    if max_abs == 0:
        return not_applicable("observed maximum velocity is zero")

    threshold = 0.9 * max_abs
    n_saturated = sum(1 for value in abs_values if value >= threshold)

    return MetricResult(
        value=100.0 * n_saturated / len(abs_values),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={"max_abs_velocity": max_abs, "limit_source": "observed"},
    )


@metric(level=Level.CHANNEL, family=Family.MOTION, requires=_REQUIRES_LIMIT_PROXIMITY)
def limit_proximity_pct(ctx: ChannelContext) -> MetricResult:
    """Share of samples at or above 95% of this channel's own observed half-range.

    Parameters
    ----------
    ctx : ChannelContext
        The joint-position channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - no non-null sample survives
        - the observed range is zero
        `report_only` otherwise.
    """

    values = [value for value in ctx.values.to_list() if value is not None]
    if not values:
        return not_applicable("no non-null samples to measure against")

    observed_min, observed_max = min(values), max(values)
    observed_range = observed_max - observed_min
    if observed_range == 0:
        return not_applicable("observed range is zero")

    midpoint = (observed_min + observed_max) / 2
    threshold = 0.95 * (observed_range / 2)
    n_near_limit = sum(1 for value in values if abs(value - midpoint) >= threshold)

    return MetricResult(
        value=100.0 * n_near_limit / len(values),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "observed_min": observed_min,
            "observed_max": observed_max,
            "limit_source": "observed",
        },
    )


@metric(level=Level.CHANNEL, family=Family.MOTION, requires=_REQUIRES_HF_VIBRATION)
def hf_vibration_ratio(ctx: ChannelContext) -> MetricResult:
    """Share of a torque channel's spectral energy above `_HF_CUTOFF_HZ`.

    Parameters
    ----------
    ctx : ChannelContext
        The joint-torque channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - numpy is not installed
        - the median gap is zero
        - the sampling rate does not clear twice the cutoff
        - the channel carries null values
        - the total spectral energy is zero
        `report_only` otherwise.
    """

    numpy = load_numpy()
    if numpy is None:
        return not_applicable(
            "numpy is not installed; install kalanos[video] for hf_vibration_ratio"
        )

    median_dt = _median_gap(ctx.stream.timestamps.to_list())
    if median_dt is None:
        return not_applicable("fewer than two timestamps to take a gap over")
    if median_dt <= 0:
        return not_applicable("median gap is zero")

    sample_rate = 1.0 / median_dt
    if sample_rate <= 2 * _HF_CUTOFF_HZ:
        return not_applicable(
            f"sampling rate {sample_rate:.1f} Hz does not clear twice the "
            f"{_HF_CUTOFF_HZ} Hz cutoff; nothing above it would be representable"
        )

    values = ctx.values.to_list()
    if any(value is None for value in values):
        return not_applicable(
            "channel carries null values; there is no full series to transform"
        )

    array = numpy.asarray(values, dtype=float)
    array = array - array.mean()
    spectrum = numpy.abs(numpy.fft.rfft(array)) ** 2
    freqs = numpy.fft.rfftfreq(len(array), d=median_dt)

    total_energy = float(spectrum.sum())
    if total_energy == 0:
        return not_applicable("total spectral energy is zero")

    hf_energy = float(spectrum[freqs > _HF_CUTOFF_HZ].sum())

    return MetricResult(
        value=hf_energy / total_energy,
        unit="fraction",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "cutoff_hz": _HF_CUTOFF_HZ,
            "sample_rate_hz": sample_rate,
            "n_samples": len(array),
        },
    )


def _abs_nonnull(ctx: ChannelContext) -> pl.Series:
    """Absolute value of a channel's non-null samples."""

    return ctx.values.drop_nulls().abs()


@metric(level=Level.CHANNEL, family=Family.MOTION, requires=_REQUIRES_TORQUE_STAT)
def p99_torque(ctx: ChannelContext) -> MetricResult:
    """99th percentile of absolute torque. Report-only by design, not by omission.

    Parameters
    ----------
    ctx : ChannelContext
        The joint-torque channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when fewer than two non-null values survive;
        `report_only` otherwise.
    """

    values = _abs_nonnull(ctx)
    if len(values) < 2:
        return not_applicable("fewer than two non-null values survive")

    return MetricResult(
        value=cast(float, values.quantile(0.99)),
        unit="N·m",
        status=MetricStatus.REPORT_ONLY,
        evidence={"n_samples": len(values)},
    )


@metric(level=Level.CHANNEL, family=Family.MOTION, requires=_REQUIRES_TORQUE_STAT)
def max_torque(ctx: ChannelContext) -> MetricResult:
    """Maximum absolute torque. Report-only by design, not by omission.

    Parameters
    ----------
    ctx : ChannelContext
        The joint-torque channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when fewer than two non-null values survive;
        `report_only` otherwise.
    """

    values = _abs_nonnull(ctx)
    if len(values) < 2:
        return not_applicable("fewer than two non-null values survive")

    return MetricResult(
        value=cast(float, values.max()),
        unit="N·m",
        status=MetricStatus.REPORT_ONLY,
        evidence={"n_samples": len(values)},
    )


@metric(level=Level.CHANNEL, family=Family.MOTION, requires=_REQUIRES_TORQUE_STAT)
def mean_torque(ctx: ChannelContext) -> MetricResult:
    """Mean absolute torque. Report-only by design, not by omission.

    Parameters
    ----------
    ctx : ChannelContext
        The joint-torque channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when fewer than two non-null values survive;
        `report_only` otherwise.
    """

    values = _abs_nonnull(ctx)
    if len(values) < 2:
        return not_applicable("fewer than two non-null values survive")

    return MetricResult(
        value=cast(float, values.mean()),
        unit="N·m",
        status=MetricStatus.REPORT_ONLY,
        evidence={"n_samples": len(values)},
    )


@metric(level=Level.EPISODE, family=Family.MOTION, requires=_REQUIRES_ENERGY_PROXY)
def energy_proxy(ctx: EpisodeContext) -> MetricResult:
    """Sum of |torque x velocity| x dt, a proxy for mechanical work done.

    Parameters
    ----------
    ctx : EpisodeContext
        The recording to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - no instance carries both a torque and a velocity stream
        - either matched stream's payload is not a `FramePayload`
        - the two streams do not share a timebase
        - both declare axes, but share none to pair channels by
        - neither stream declares a channel to pair, when axes are undeclared
        - there are fewer than two samples to take a gap over
        `report_only` otherwise.
    """

    torque_stream = ctx.streams_of(_JOINT_TORQUE)[0]
    velocity_matches = [
        stream
        for stream in ctx.streams_of(_JOINT_VELOCITY)
        if stream.instance == torque_stream.instance
    ]
    if not velocity_matches:
        return not_applicable("no instance carries both a torque and a velocity stream")
    velocity_stream = velocity_matches[0]

    if not isinstance(torque_stream.payload, FramePayload) or not isinstance(
        velocity_stream.payload, FramePayload
    ):
        return not_applicable("one of the matched streams carries no FramePayload")

    torque_times = torque_stream.timestamps.to_list()
    velocity_times = velocity_stream.timestamps.to_list()
    if len(torque_times) != len(velocity_times) or any(
        abs(t - v) > 1e-6 for t, v in zip(torque_times, velocity_times, strict=True)
    ):
        return not_applicable("torque and velocity streams do not share a timebase")

    torque_channels = torque_stream.channels
    velocity_channels = velocity_stream.channels

    axes_declared = (
        bool(torque_channels)
        and bool(velocity_channels)
        and all(channel.axis is not None for channel in torque_channels)
        and all(channel.axis is not None for channel in velocity_channels)
    )
    if axes_declared:
        velocity_by_axis = {channel.axis: channel for channel in velocity_channels}
        axis_pairs = [
            (channel, velocity_by_axis[channel.axis])
            for channel in torque_channels
            if channel.axis in velocity_by_axis
        ]
        if not axis_pairs:
            return not_applicable(
                "both streams declare axes, but share none to pair channels by"
            )
    else:
        axis_pairs = list(zip(torque_channels, velocity_channels, strict=False))
        if not axis_pairs:
            return not_applicable("neither stream declares a channel to pair")

    median_dt = _median_gap(torque_times)
    if median_dt is None or median_dt <= 0:
        return not_applicable("fewer than two samples to take a gap over")
    duration = torque_times[-1] - torque_times[0]

    torque_frame = torque_stream.payload.frame
    velocity_frame = velocity_stream.payload.frame

    total_energy = 0.0
    for torque_channel, velocity_channel in axis_pairs:
        torque_values = torque_frame[torque_channel.name].to_list()
        velocity_values = velocity_frame[velocity_channel.name].to_list()
        total_energy += sum(
            abs(t * v) * median_dt
            for t, v in zip(torque_values, velocity_values, strict=True)
            if t is not None and v is not None
        )

    return MetricResult(
        value=total_energy,
        unit="J (robot units)",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_channels_paired": len(axis_pairs),
            "instance": torque_stream.instance,
            "duration_s": duration,
        },
    )
