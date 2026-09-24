"""The integrity family: missing_pct, flatline_pct, spike_pct, drift, snr_db,
dead_taxel_pct and hysteresis.

Five gate on dtype and sample count only, so they run on any numeric channel
regardless of taxonomy. dead_taxel_pct and hysteresis need an array of cells,
so they attach at Level.STREAM and gate on `extero.taxel_pressure` instead.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from typing import cast

# External
import polars as pl

# Internal
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
from kalanos.analysis.models.domain import FramePayload
from kalanos.analysis.models.metrics import (
    ChannelContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
    Requires,
    StreamContext,
)


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

# The window snr_db smooths over before treating what's left as noise.
_SNR_SMOOTHING_WINDOW = 5
_REQUIRES_REGULAR_AND_SMOOTHABLE = Requires(regular_sampling=True, min_samples=8)

_REQUIRES_TAXEL_ARRAY = Requires(min_samples=2, taxonomy=[_TAXEL_PRESSURE])
_REQUIRES_TAXEL_CYCLE = Requires(min_samples=8, taxonomy=[_TAXEL_PRESSURE])


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@metric(level=Level.CHANNEL, family=Family.INTEGRITY, requires=_REQUIRES_ANY_VALUE)
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
        `not_applicable` when the channel's dtype is not numeric;
        `report_only` otherwise.
    """

    if not ctx.values.dtype.is_numeric():
        return not_applicable(
            "channel is not numeric; there is no missing value to count"
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


@metric(level=Level.CHANNEL, family=Family.INTEGRITY, requires=_REQUIRES_A_PAIR)
def flatline_pct(ctx: ChannelContext) -> MetricResult:
    """Share of consecutive non-null samples that did not change.

    Parameters
    ----------
    ctx : ChannelContext
        The channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the dtype is not numeric
        - fewer than two non-null values survive
        `report_only` otherwise, with the longest unchanged run in `evidence`.
    """

    if not ctx.values.dtype.is_numeric():
        return not_applicable("channel is not numeric; there is nothing to flatline")

    values = ctx.values.to_list()
    timestamps = ctx.stream.timestamps.to_list()
    valid_indices = [index for index, value in enumerate(values) if value is not None]
    if len(valid_indices) < 2:
        return not_applicable("fewer than two non-null values survive")

    n_pairs = 0
    n_unchanged = 0
    run_length = 1
    run_start = valid_indices[0]
    longest_run = 1
    longest_run_start = valid_indices[0]
    longest_run_end = valid_indices[0]

    for previous_index, index in zip(valid_indices, valid_indices[1:], strict=False):
        n_pairs += 1
        if values[index] == values[previous_index]:
            n_unchanged += 1
            run_length += 1
        else:
            run_length = 1
            run_start = index
        if run_length > longest_run:
            longest_run = run_length
            longest_run_start = run_start
            longest_run_end = index

    return MetricResult(
        value=100.0 * n_unchanged / n_pairs,
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "longest_run": longest_run,
            "longest_run_s": timestamps[longest_run_end]
            - timestamps[longest_run_start],
        },
    )


@metric(level=Level.CHANNEL, family=Family.INTEGRITY, requires=_REQUIRES_SPIKE_WINDOW)
def spike_pct(ctx: ChannelContext) -> MetricResult:
    """Share of samples more than 6 standard deviations from a centred local window.

    Parameters
    ----------
    ctx : ChannelContext
        The channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when the dtype is not numeric or every window's
        local spread is zero or null; `report_only` otherwise.
    """

    if not ctx.values.dtype.is_numeric():
        return not_applicable(
            "channel is not numeric; there is no spread to measure a spike against"
        )

    values = ctx.values.cast(pl.Float64)
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

    return MetricResult(
        value=100.0 * int(is_spike.sum()) / n_scored,
        unit="%",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_scored": n_scored,
            "window_samples": _SPIKE_WINDOW,
            "window_is_undecided": True,
        },
    )


@metric(level=Level.CHANNEL, family=Family.INTEGRITY, requires=_REQUIRES_A_TREND)
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


@metric(
    level=Level.CHANNEL,
    family=Family.INTEGRITY,
    requires=_REQUIRES_REGULAR_AND_SMOOTHABLE,
)
def snr_db(ctx: ChannelContext) -> MetricResult:
    """Ratio of a smoothed component's variance to its residual's, in decibels.

    Parameters
    ----------
    ctx : ChannelContext
        The channel to measure.

    Returns
    -------
    MetricResult
        `not_applicable` when:
        - the dtype is not numeric
        - the smoothed signal variance is zero
        - the residual variance is zero
        `report_only` otherwise.
    """

    if not ctx.values.dtype.is_numeric():
        return not_applicable("channel is not numeric; there is no signal to measure")

    values = ctx.values.cast(pl.Float64)
    smoothed = values.rolling_mean(window_size=_SNR_SMOOTHING_WINDOW, center=True)
    residual = values - smoothed
    valid = smoothed.is_not_null()

    smoothed_values = smoothed.filter(valid)
    residual_values = residual.filter(valid)
    if len(smoothed_values) < 2:
        return not_applicable("too few samples survive smoothing to measure a ratio")

    signal_variance = cast(float, smoothed_values.var())
    noise_variance = cast(float, residual_values.var())
    if signal_variance == 0:
        return not_applicable("smoothed signal variance is zero")
    if noise_variance == 0:
        return not_applicable("residual variance is zero; the ratio would be infinite")

    return MetricResult(
        value=10.0 * math.log10(signal_variance / noise_variance),
        unit="dB",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "signal_variance": signal_variance,
            "noise_variance": noise_variance,
            "smoothing_window": _SNR_SMOOTHING_WINDOW,
            "n_samples": len(smoothed_values),
        },
    )


@metric(level=Level.STREAM, family=Family.INTEGRITY, requires=_REQUIRES_TAXEL_ARRAY)
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


@metric(level=Level.STREAM, family=Family.INTEGRITY, requires=_REQUIRES_TAXEL_CYCLE)
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
