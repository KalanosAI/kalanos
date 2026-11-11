"""The integrity family: missing_pct, flatline_pct, spike_pct, drift, snr_db,
rotation_snr_db, dead_taxel_pct and hysteresis.

Five gate on dtype and sample count only, so they run on any numeric channel
regardless of taxonomy. dead_taxel_pct and hysteresis need an array of cells,
so they attach at Level.STREAM and gate on `extero.taxel_pressure` instead.
rotation_snr_db reads a unit quaternion's four channels together,
so it attaches at Level.STREAM too.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import polars as pl

from kalanos.analysis.clocks import samples as clock_samples
from kalanos.analysis.localization import finite, source_values, support_for

# Internal
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
from kalanos.analysis.models.binding import Representation
from kalanos.analysis.models.domain import FramePayload, Stream
from kalanos.analysis.models.metrics import (
    ChannelContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
    Requires,
    StreamContext,
)
from kalanos.analysis.noise import noise_assessment, snr_exclusion


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_TAXEL_PRESSURE = "extero.taxel_pressure"

# One value is the fewest missing_pct needs: a channel of length one still
# has a missing-or-not answer.
_REQUIRES_ANY_VALUE = Requires(min_samples=1)

# Two consecutive values are the fewest flatline_pct can compare.
_REQUIRES_A_PAIR = Requires(min_samples=2)

# Odd, so the centred rolling window has a well-defined middle sample.
_SPIKE_WINDOW = 51
_REQUIRES_SPIKE_WINDOW = Requires(min_samples=_SPIKE_WINDOW)

# Five points are the fewest a least-squares line means anything over.
_REQUIRES_A_TREND = Requires(min_samples=5)

# The window snr_db smooths over before treating what's left as noise, in samples.
_SNR_SMOOTHING_WINDOW = 5

# The longest span that window may cover, in seconds. What the smoothing removes
# is counted as noise, so the window must be short next to the motion itself:
# 5 samples cover 0.1 s at 50 Hz, and less at any faster rate. Sampled slower, the
# same 5 samples span more (1 s at 5 Hz) and smooth away real motion, which then
# reads as noise — found grading real 5-15 Hz LeRobot datasets, whose ratio tracked
# the sampling rate rather than the robot. There the ratio is not reported rather
# than misreported. The 10% allowance keeps a real 50 Hz clock's wobble inside.
_SNR_MAX_SPAN_SECONDS = 0.1
_SNR_SPAN_ALLOWANCE = 1.1
_REQUIRES_REGULAR_AND_SMOOTHABLE = Requires(regular_sampling=True, min_samples=8)

# How far a row's norm may stray from 1 and still be read as a rotation.
_QUATERNION_NORM_TOLERANCE = 1e-3

_REQUIRES_TAXEL_ARRAY = Requires(min_samples=2, taxonomy=[_TAXEL_PRESSURE])
_REQUIRES_TAXEL_CYCLE = Requires(min_samples=8, taxonomy=[_TAXEL_PRESSURE])

# A channel that only ever takes two values is a switch — a gripper open/close
# command, a done flag, a contact bit — not a sampled signal. "Unchanged for most
# of the episode", "noisy" and "spiking" describe a signal, so the three checks
# below stand down on one rather than grade a switch doing its job as a defect.
# A channel stuck on a single value is still checked: that can be a dead sensor.
_SWITCH_DISTINCT_VALUES = 2
_SWITCH_REASON = (
    "channel takes only two values, so it is a switch or flag rather than "
    "a sampled signal; {what} does not apply"
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _is_switch(values: pl.Series) -> bool:
    """Check whether a numeric channel only ever takes exactly two values.

    Parameters
    ----------
    values : pl.Series
        The channel's values; nulls are ignored.

    Returns
    -------
    bool
        `True` when exactly two distinct non-null values occur.
        One value (a constant channel) is not a switch: it may be a stuck sensor.
    """

    return values.drop_nulls().n_unique() == _SWITCH_DISTINCT_VALUES


@metric(
    level=Level.CHANNEL,
    family=Family.INTEGRITY,
    requires=_REQUIRES_ANY_VALUE,
    label="missing values",
)
def missing_pct(ctx: ChannelContext) -> MetricResult:
    """Share of a channel's samples that are null or NaN.

    NaN is counted alongside null: a numeric channel can carry either, and
    a reader asking how much of this signal is missing wants both.

    Parameters
    ----------
    ctx : ChannelContext
        The channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when the channel's dtype is neither numeric nor Boolean;
        `report_only` otherwise.
    """

    if not ctx.values.dtype.is_numeric() and ctx.values.dtype != pl.Boolean:
        return not_applicable(
            "channel is neither numeric nor Boolean; there is no missing value to count"
        )

    n_missing = ctx.values.null_count()
    if ctx.values.dtype.is_float():
        n_missing += int(ctx.values.is_nan().sum())

    n_samples = ctx.n_samples
    return MetricResult(
        value=100.0 * n_missing / n_samples,
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={"n_missing": n_missing, "n_samples": n_samples},
    )


@metric(
    level=Level.CHANNEL,
    family=Family.INTEGRITY,
    requires=_REQUIRES_A_PAIR,
    label="repeated values",
)
def flatline_pct(ctx: ChannelContext) -> MetricResult:
    """Share of adjacent finite source-order samples that did not change.

    Parameters
    ----------
    ctx : ChannelContext
        The channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the dtype is not numeric
        - the channel takes exactly two values (a switch or flag)
        - no adjacent finite pair survives
        `report_only` otherwise, with the longest unchanged run in `evidence`.
    """

    if not ctx.values.dtype.is_numeric():
        return not_applicable("channel is not numeric; there is nothing to flatline")
    if _is_switch(ctx.values):
        return not_applicable(_SWITCH_REASON.format(what="flatline"), inapplicable=True)

    ordered = source_values(ctx)
    if ordered is None:
        return not_applicable("source row order is unavailable")
    series, indices, timestamps = ordered
    values = series.to_list()
    n_pairs = n_unchanged = 0
    ranges = []
    run_start = None
    for i in range(1, len(values)):
        valid = finite(values[i - 1]) and finite(values[i])
        if valid:
            n_pairs += 1
        if valid and values[i] == values[i - 1]:
            n_unchanged += 1
            if run_start is None:
                run_start = i - 1
        elif run_start is not None:
            ranges.append((run_start, i))
            run_start = None
    if run_start is not None:
        ranges.append((run_start, len(values)))
    if not n_pairs:
        return not_applicable("fewer than two adjacent finite values survive")
    longest = max(ranges, key=lambda r: r[1] - r[0], default=(0, 1))
    lo, hi = longest
    ts = timestamps[lo:hi]
    stream = ctx.stream.stream
    info = stream.clock_info
    known_seconds = stream.native_timestamps is None or (
        info is not None
        and (
            info.tick_period_s is not None
            or info.native_unit in ("s", "ms", "us", "ns")
        )
    )
    duration = (
        ts[-1] - ts[0]
        if known_seconds
        and ts
        and all(finite(t) for t in ts)
        and all(b > a for a, b in zip(ts, ts[1:], strict=False))
        else None
    )
    return MetricResult(
        value=100.0 * n_unchanged / n_pairs,
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        support=support_for(indices, ranges),
        evidence={
            "longest_run": hi - lo,
            "longest_run_s": duration,
            "never_changed": n_unchanged == n_pairs,
            "n_pairs": n_pairs,
            "n_unchanged": n_unchanged,
            "n_samples": ctx.n_samples,
            "longest_run_start": indices[lo],
            "longest_run_end_exclusive": indices[hi - 1] + 1,
        },
    )


@metric(
    level=Level.CHANNEL,
    family=Family.INTEGRITY,
    requires=_REQUIRES_SPIKE_WINDOW,
    label="sudden changes",
)
def spike_pct(ctx: ChannelContext) -> MetricResult:
    """Share of samples more than 6 standard deviations from a centred local window.

    Parameters
    ----------
    ctx : ChannelContext
        The channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when the dtype is not numeric, the channel takes
        exactly two values (a switch or flag), or every window's local spread
        is zero or null; `report_only` otherwise.
    """

    if not ctx.values.dtype.is_numeric():
        return not_applicable(
            "channel is not numeric; there is no spread to measure a spike against"
        )
    if _is_switch(ctx.values):
        return not_applicable(
            _SWITCH_REASON.format(what="a spike check"), inapplicable=True
        )

    ordered = source_values(ctx)
    if ordered is None:
        return not_applicable("source row order is unavailable")
    ordered_values, indices, _ = ordered
    values = ordered_values.cast(pl.Float64)
    window_sum = values.rolling_sum(window_size=_SPIKE_WINDOW, center=True)
    window_sq_sum = (values**2).rolling_sum(window_size=_SPIKE_WINDOW, center=True)

    other_samples = _SPIKE_WINDOW - 1
    other_mean = (window_sum - values) / other_samples
    other_mean_sq = (window_sq_sum - values**2) / other_samples
    # Clipped against floating-point round-off pushing a near-zero variance
    # negative just before the square root.
    other_variance = (other_mean_sq - other_mean**2).clip(lower_bound=0.0)
    other_std = other_variance.sqrt()

    scored = other_std.is_not_null() & (other_std > 0)
    n_scored = int(scored.sum())
    if n_scored == 0:
        return not_applicable(
            "local spread is zero throughout; no scale to measure a spike against"
        )

    deviation = (values - other_mean).abs()
    is_spike = scored & (deviation > 6 * other_std)
    positions = [i for i, flag in enumerate(is_spike.to_list()) if flag]

    return MetricResult(
        value=100.0 * len(positions) / n_scored,
        support=support_for(
            indices, [(i, i + 1) for i in positions], window=_SPIKE_WINDOW // 2
        ),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_scored": n_scored,
            "sample_indices": [indices[i] for i in positions],
            "window_samples": _SPIKE_WINDOW,
            "window_is_undecided": True,
        },
    )


@metric(
    level=Level.CHANNEL,
    family=Family.INTEGRITY,
    requires=_REQUIRES_A_TREND,
    label="drift",
)
def drift(ctx: ChannelContext) -> MetricResult:
    """Least-squares slope of a channel's value against time, in units per minute.

    Parameters
    ----------
    ctx : ChannelContext
        The channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the dtype is not numeric
        - the time span is zero
        - the value variance is zero
        `report_only` otherwise.
    """

    if not ctx.values.dtype.is_numeric():
        return not_applicable("channel is not numeric; there is no trend to fit")

    timestamps = ctx.stream.timestamps.to_list()
    values = ctx.values.to_list()
    paired = [
        (t, v)
        for t, v in zip(timestamps, values, strict=False)
        if t is not None and v is not None
    ]
    if len(paired) < 2:
        return not_applicable("fewer than two paired samples to fit a trend over")

    times = [t for t, _ in paired]
    vals = [v for _, v in paired]
    if times[-1] - times[0] <= 0:
        return not_applicable("time span is zero")

    mean_t = sum(times) / len(times)
    mean_v = sum(vals) / len(vals)
    time_variance = sum((t - mean_t) ** 2 for t in times)
    total_variance = sum((v - mean_v) ** 2 for v in vals)
    if total_variance == 0:
        return not_applicable(
            "value variance is zero; r-squared is undefined against a constant"
        )

    covariance = sum((t - mean_t) * (v - mean_v) for t, v in paired)
    slope = covariance / time_variance
    intercept = mean_v - slope * mean_t
    fitted = [slope * t + intercept for t in times]
    residual_variance = sum((v - f) ** 2 for v, f in zip(vals, fitted, strict=True))
    r_squared = 1.0 - residual_variance / total_variance
    slope_per_min = slope * 60.0

    return MetricResult(
        value=slope_per_min,
        unit="unit/min",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "slope_per_min": slope_per_min,
            "r_squared": r_squared,
            "total_change": fitted[-1] - fitted[0],
            "n_samples": len(paired),
        },
    )


def _smoothing_rate(
    stream: Stream, is_regular: bool
) -> tuple[float, float] | MetricResult:
    """Check that a stream's clock lets the smoothing window separate noise from motion.

    Parameters
    ----------
    stream : Stream
        The stream whose clock is read.
    is_regular : bool
        Whether inference classified the stream's sampling as regular.

    Returns
    -------
    tuple[float, float] or MetricResult
        The median sampling rate in Hz and the span the smoothing window covers
        in seconds, or a `not_applicable` result naming the gate that failed.
    """

    ticks = clock_samples(stream)
    if ticks.reason:
        return not_applicable("source row order is unavailable")
    if not ticks.seconds:
        return not_applicable("timestamp units do not establish seconds")
    if ticks.invalid_rows or not ticks.gaps or any(g <= 0 for g in ticks.gaps):
        return not_applicable("SNR needs a complete increasing time axis")
    if not is_regular:
        return not_applicable("sampling is not regular")
    rate_hz = 1 / sorted(ticks.gaps)[len(ticks.gaps) // 2]
    span_s = _SNR_SMOOTHING_WINDOW / rate_hz
    if span_s > _SNR_MAX_SPAN_SECONDS * _SNR_SPAN_ALLOWANCE:
        return not_applicable(
            f"sampled at {rate_hz:.3g} Hz, too slowly to separate sensor noise from "
            f"motion: the {_SNR_SMOOTHING_WINDOW}-sample smoothing window spans "
            f"{span_s:.2g} s, more than {_SNR_MAX_SPAN_SECONDS} s"
        )
    return rate_hz, span_s


@metric(
    level=Level.CHANNEL,
    family=Family.INTEGRITY,
    requires=_REQUIRES_REGULAR_AND_SMOOTHABLE,
    label="low signal-to-noise",
)
def snr_db(ctx: ChannelContext) -> MetricResult:
    """Measure a five-sample smooth/residual ratio, with explicit physical context.

    Returns
    -------
    MetricResult
        The diagnostic ratio and component standard deviations. A validated
        matching reference can establish residual level relative to that
        reference, never sensor health or automatic blocking authority.
    """
    if not ctx.values.dtype.is_numeric():
        return not_applicable("channel is not numeric", inapplicable=True)
    exclusion = snr_exclusion(ctx)
    if exclusion:
        return not_applicable(exclusion, inapplicable=True)
    if _is_switch(ctx.values):
        return not_applicable(
            _SWITCH_REASON.format(what="a signal-to-noise ratio"), inapplicable=True
        )
    binding = ctx.channel.binding
    if binding is not None and binding.representation == Representation.QUATERNION:
        return not_applicable(
            "component of a unit quaternion; its noise is measured on the whole "
            "rotation by rotation_snr_db",
            inapplicable=True,
        )
    ordered = source_values(ctx)
    if ordered is None:
        return not_applicable("source row order is unavailable")
    rate = _smoothing_rate(ctx.stream.stream, ctx.is_regular)
    if isinstance(rate, MetricResult):
        return rate
    rate_hz, span_s = rate
    ordered_values, _, _ = ordered
    values = ordered_values.cast(pl.Float64)
    invalid = values.is_null() | ~values.is_finite()
    values = values.set(invalid, None)
    smoothed = values.rolling_mean(window_size=_SNR_SMOOTHING_WINDOW, center=True)
    residual = values - smoothed
    valid = smoothed.is_not_null() & residual.is_not_null()
    smooth_values, residual_values = smoothed.filter(valid), residual.filter(valid)
    if len(smooth_values) < 2:
        return not_applicable("too few finite contiguous windows survive smoothing")
    raw_signal_variance, raw_noise_variance = smooth_values.var(), residual_values.var()
    # both series stay Float64 throughout, so var() returns a plain float here
    assert isinstance(raw_signal_variance, (int, float))
    assert isinstance(raw_noise_variance, (int, float))
    signal_variance = float(raw_signal_variance)
    noise_variance = float(raw_noise_variance)
    if not all(math.isfinite(v) and v >= 0 for v in (signal_variance, noise_variance)):
        return not_applicable("component variance is not finite")
    signal_std, residual_std = math.sqrt(signal_variance), math.sqrt(noise_variance)
    assessment = noise_assessment(ctx, rate_hz, residual_std, signal_std)
    evidence = {
        "signal_variance": signal_variance,
        "noise_variance": noise_variance,
        "signal_standard_deviation": signal_std,
        "residual_standard_deviation": residual_std,
        "amplitude_unit": ctx.channel.binding.unit if ctx.channel.binding else None,
        "smoothing_window": _SNR_SMOOTHING_WINDOW,
        "smoothing_span_s": span_s,
        "rate_hz": rate_hz,
        "n_samples": len(smooth_values),
        "n_invalid_samples": int(invalid.sum()),
        "estimator": "centered_mean_5_residual_std_v1",
        "noise_assessment": assessment,
        "interpretation": (
            "smooth/residual diagnostic; neither measured sensor SNR "
            "nor proof of health"
        ),
    }
    if signal_variance == 0 or noise_variance == 0:
        result = not_applicable(
            "smoothed signal variance is zero"
            if signal_variance == 0
            else "residual variance is zero; the ratio would be infinite"
        )
        return result.model_copy(update={"evidence": {**evidence, **result.evidence}})
    return MetricResult(
        value=10.0 * (math.log10(signal_variance) - math.log10(noise_variance)),
        unit="dB",
        status=MetricStatus.REPORT_ONLY,
        evidence=evidence,
    )


def _float_mean(values: pl.Series) -> float:
    """Mean of a non-empty Float64 series, as a plain float.

    Parameters
    ----------
    values : pl.Series
        A Float64 series with at least one non-null value.

    Returns
    -------
    float
        The series mean.
    """

    mean = values.mean()
    # a Float64 series with a non-null value has a float mean
    assert isinstance(mean, float)
    return mean


def _rotation_angle(left: list[pl.Expr], right: list[pl.Expr]) -> pl.Expr:
    """Angle in radians between two unit quaternions, row by row.

    Parameters
    ----------
    left, right : list[pl.Expr]
        The four components of each quaternion, in the same order.

    Returns
    -------
    pl.Expr
        `2·acos(|⟨left, right⟩|)`, so `q` and `-q` read as the same rotation.
    """

    dot = pl.sum_horizontal([a * b for a, b in zip(left, right, strict=True)])
    return 2 * dot.abs().clip(upper_bound=1.0).arccos()


@metric(
    level=Level.STREAM,
    family=Family.INTEGRITY,
    label="low rotation signal-to-noise",
)
def rotation_snr_db(ctx: StreamContext) -> MetricResult:
    """Measure a five-sample smooth/residual ratio on a unit quaternion's rotation.

    Each component of a quaternion moves by a different share of the same rotation,
    so the ratio is taken on the angle between rotations.

    Parameters
    ----------
    ctx : StreamContext
        The stream holding the quaternion's four channels.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the stream does not hold exactly four channels bound as quaternion
        - the stream has fewer than 8 samples
        - the clock fails the same gates as `snr_db`
        - the four components do not have unit length
        - too few windows survive smoothing, or either mean square is zero
        `report_only` otherwise, in dB.
    """

    names = [
        channel.name
        for channel in ctx.stream.channels
        if channel.binding is not None
        and channel.binding.representation == Representation.QUATERNION
    ]
    if len(names) != 4:
        return not_applicable(
            "stream holds no single unit quaternion", inapplicable=True
        )
    min_samples = _REQUIRES_REGULAR_AND_SMOOTHABLE.min_samples
    if ctx.n_samples < min_samples:
        return not_applicable(f"fewer than {min_samples} samples")
    rate = _smoothing_rate(ctx.stream, ctx.is_regular)
    if isinstance(rate, MetricResult):
        return rate
    rate_hz, span_s = rate
    if not isinstance(ctx.payload, FramePayload):
        return not_applicable("payload is not a FramePayload; there are no components")

    frame = ctx.payload.frame.select(pl.col(names).cast(pl.Float64))
    order = ctx.stream.source_order
    if not order.preserved and order.original_index is not None:
        index = order.original_index
        frame = frame[sorted(range(frame.height), key=lambda i: index[i])]

    components = [pl.col(name) for name in names]
    invalid = pl.any_horizontal(
        [c.is_null() | ~c.is_finite().fill_null(False) for c in components]
    )
    norm = pl.sum_horizontal([c**2 for c in components]).sqrt()
    frame = frame.with_columns(
        pl.when(~invalid).then(c).otherwise(None) for c in components
    ).with_columns(
        pl.when(~invalid).then(norm).alias("_norm"), invalid.alias("_invalid")
    )
    n_invalid = int(frame["_invalid"].sum())
    norm_error = (frame["_norm"] - 1.0).abs().max()
    max_norm_error = float(norm_error) if isinstance(norm_error, (int, float)) else None
    if max_norm_error is not None and max_norm_error > _QUATERNION_NORM_TOLERANCE:
        result = not_applicable(
            "the four components do not have unit length, so they are not a rotation"
        )
        return result.model_copy(
            update={"evidence": {**result.evidence, "max_norm_error": max_norm_error}}
        )
    frame = frame.with_columns(c / pl.col("_norm") for c in components)

    # q and -q are the same rotation. Flip each row whose dot product with the
    # previous valid row is negative, so consecutive rows sit on one hemisphere.
    previous = [c.shift(1).forward_fill() for c in components]
    dot = pl.sum_horizontal([c * p for c, p in zip(components, previous, strict=True)])
    sign = pl.when(dot < 0).then(-1.0).otherwise(1.0).cum_prod()
    frame = frame.with_columns(c * sign for c in components)

    smoothed_names = [f"_smoothed_{name}" for name in names]
    smoothed = [pl.col(name) for name in smoothed_names]
    frame = frame.with_columns(
        c.rolling_mean(window_size=_SNR_SMOOTHING_WINDOW, center=True).alias(name)
        for c, name in zip(components, smoothed_names, strict=True)
    )
    smoothed_norm = pl.sum_horizontal([c**2 for c in smoothed]).sqrt()
    frame = frame.with_columns(c / smoothed_norm for c in smoothed).filter(
        pl.all_horizontal([c.is_not_null() for c in smoothed])
    )
    if frame.height < 2:
        return not_applicable("too few finite contiguous windows survive smoothing")

    # Rotation SNR. With qᵢ the rows, q̃ᵢ their smoothed rotations, q̄ the episode mean:
    #   θ(a, b) = 2·acos(|⟨a, b⟩|)
    #   SNR = 10·log10( mean θ(q̃ᵢ, q̄)² / mean θ(qᵢ, q̃ᵢ)² )
    # θ is the angle between two rotations: Markley et al. (2007), eq. 16b, gives
    # cos(θ/2) = qᵢᵀq for the error rotation. The numerator is the geodesic variance
    # of the motion around its mean, as defined by Pennec (2006).
    # q̃ᵢ and q̄ are plain means of sign-aligned rows, renormalised. Markley et al.
    # name a plain mean's two flaws, sign dependence and non-unit length, and the
    # alignment and renormalising fix both. Against their eigenvector mean, the SNR
    # differs by at most 0.016 dB on 36 TUM RGB-D segments.
    # The smooth/residual ratio is the heuristic snr_db uses; no paper defines this
    # exact SNR.
    # - Markley, Cheng, Crassidis, Oshman (2007). Averaging Quaternions.
    #   J. Guidance, Control, and Dynamics 30(4), 1193-1197. doi:10.2514/1.28949
    # - Pennec (2006). Intrinsic Statistics on Riemannian Manifolds.
    #   J. Math. Imaging Vis. 25(1), 127-154. doi:10.1007/s10851-006-6228-4

    # The mean must come from the sign-aligned rows above: averaged over rows of
    # mixed sign, it shrinks toward zero and the signal angle is wrong by several dB.
    mean = [_float_mean(frame[name]) for name in smoothed_names]
    mean_norm = math.sqrt(sum(m * m for m in mean))
    centre = [pl.lit(m / mean_norm) for m in mean]
    angles = frame.select(
        _rotation_angle(components, smoothed).alias("residual"),
        _rotation_angle(smoothed, centre).alias("signal"),
    )
    signal_ms = _float_mean(angles["signal"].pow(2))
    residual_ms = _float_mean(angles["residual"].pow(2))

    evidence = {
        "signal_rms_rad": math.sqrt(signal_ms),
        "residual_rms_rad": math.sqrt(residual_ms),
        "max_norm_error": max_norm_error,
        "channels": names,
        "smoothing_window": _SNR_SMOOTHING_WINDOW,
        "smoothing_span_s": span_s,
        "rate_hz": rate_hz,
        "n_samples": frame.height,
        "n_invalid_samples": n_invalid,
        "estimator": "quaternion_geodesic_mean_5_v1",
        "noise_assessment": {
            "status": "unassessed",
            "reason": "no noise floor reference exists for rotations",
        },
        "interpretation": (
            "rotation smooth/residual diagnostic; neither measured sensor SNR "
            "nor proof of health"
        ),
    }
    if signal_ms == 0 or residual_ms == 0:
        result = not_applicable(
            "smoothed signal variance is zero"
            if signal_ms == 0
            else "residual variance is zero; the ratio would be infinite"
        )
        return result.model_copy(update={"evidence": {**evidence, **result.evidence}})
    return MetricResult(
        value=10.0 * (math.log10(signal_ms) - math.log10(residual_ms)),
        unit="dB",
        status=MetricStatus.REPORT_ONLY,
        evidence=evidence,
    )


@metric(
    level=Level.STREAM,
    family=Family.INTEGRITY,
    requires=_REQUIRES_TAXEL_ARRAY,
    label="dead taxels",
)
def dead_taxel_pct(ctx: StreamContext) -> MetricResult:
    """Share of a tactile array's cells whose value never changes across the recording.

    Parameters
    ----------
    ctx : StreamContext
        The tactile stream to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the payload is not a `FramePayload`
        - the stream declares no channels
        `report_only` otherwise.
    """

    if not isinstance(ctx.payload, FramePayload):
        return not_applicable(
            "payload is not a FramePayload; there is no array of cells"
        )

    channel_names = [channel.name for channel in ctx.stream.channels]
    if not channel_names:
        return not_applicable("stream declares no channels")

    frame = ctx.payload.frame
    dead_cells = [name for name in channel_names if frame[name].n_unique() <= 1]

    return MetricResult(
        value=100.0 * len(dead_cells) / len(channel_names),
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_cells": len(channel_names),
            "n_dead": len(dead_cells),
            "dead_cells": dead_cells,
        },
    )


@metric(
    level=Level.STREAM,
    family=Family.INTEGRITY,
    requires=_REQUIRES_TAXEL_CYCLE,
    label="hysteresis",
)
def hysteresis(ctx: StreamContext) -> MetricResult:
    """Mean gap between a tactile array's loading and unloading response, as a fraction.

    Parameters
    ----------
    ctx : StreamContext
        The tactile stream to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the payload is not a `FramePayload`
        - the stream declares no channels
        - the aggregate never rises then falls
        - the aggregate range is zero
        `report_only` otherwise.
    """

    if not isinstance(ctx.payload, FramePayload):
        return not_applicable(
            "payload is not a FramePayload; there is no array of cells"
        )

    channel_names = [channel.name for channel in ctx.stream.channels]
    if not channel_names:
        return not_applicable("stream declares no channels")

    aggregate = ctx.payload.frame.select(channel_names).mean_horizontal().to_list()
    n_samples = len(aggregate)
    peak_index = max(range(n_samples), key=lambda index: aggregate[index])
    if peak_index == 0 or peak_index == n_samples - 1:
        return not_applicable(
            "the aggregate never rises then falls; recording holds no load-unload cycle"
        )

    aggregate_range = max(aggregate) - min(aggregate)
    if aggregate_range == 0:
        return not_applicable(
            "aggregate range is zero; there is no load to compare against"
        )

    # Pair samples by time distance from the peak, not by aggregate level,
    # so the difference reflects the actual gap between the branches.
    differences = [
        abs(aggregate[index] - aggregate[2 * peak_index - index])
        for index in range(peak_index)
        if 2 * peak_index - index < n_samples
    ]
    if not differences:
        return not_applicable("no pre-peak sample has a mirrored post-peak sample")

    return MetricResult(
        value=(sum(differences) / len(differences)) / aggregate_range,
        unit="fraction",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_pairs_compared": len(differences),
            "aggregate_range": aggregate_range,
            "peak_index": peak_index,
        },
    )
