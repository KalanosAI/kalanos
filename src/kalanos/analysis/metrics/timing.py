"""Recorded timeline diagnostics and acquisition timing have separate claims.

Ordering always uses adjacent source rows.
Recorded cadence/spread/drop estimates are descriptive
and do not establish sensor capture timing.
The historical rate/jitter/drop metrics
require explicit producer evidence for capture time.
Uniformity alone neither proves generation nor disproves measured capture time.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import statistics

# Internal
from kalanos.analysis.clocks import clock_evidence, samples
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.models.domain import ClockInfo
from kalanos.analysis.models.metrics import (
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

_UNGRADED_REASON = "grading needs policy thresholds, which this stage does not read"
_RECORDED_REASON = "recorded timeline diagnostic; does not certify acquisition timing"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _cadence(ctx: StreamContext, *, acquisition: bool, kind: str) -> MetricResult:
    data = samples(ctx.stream)
    info = ctx.stream.clock_info or ClockInfo.from_legacy(ctx.stream.clock)
    evidence = clock_evidence(ctx.stream, data)
    evidence["measurement_scope"] = (
        "acquisition" if acquisition else "recorded_timeline"
    )
    reason = data.reason
    if not reason and acquisition and not info.certifies_acquisition:
        reason = (
            f"capture timing is not observable: clock origin {info.origin.value!r} "
            f"has {info.origin_evidence.value!r} evidence; explicit producer "
            "evidence for capture time is required"
        )
    if not reason and (info.native_unit == "unknown" or not data.seconds):
        reason = "timestamp units are unknown; cannot measure cadence in seconds"
    if not reason and data.invalid_rows:
        reason = "invalid timestamps prevent a whole-stream cadence estimate"
    if not reason and len(data.gaps) < (4 if kind == "spread" else 1):
        reason = (
            "fewer than five valid samples"
            if kind == "spread"
            else "fewer than two valid samples"
        )
    if not reason and any(gap <= 0 for gap in data.gaps):
        reason = "repeated/backwards timestamps prevent a whole-stream cadence estimate"
    if not reason and not ctx.is_regular:
        reason = "sampling is not regular"
    if reason:
        return MetricResult(
            value=None,
            unit=None,
            status=MetricStatus.NOT_APPLICABLE,
            evidence={**evidence, "reason": reason},
        )
    median_gap = statistics.median(data.gaps)
    evidence["median_gap_s"] = median_gap
    if kind == "rate":
        value, unit = 1.0 / median_gap, "Hz"
    elif kind == "spread":
        value, unit = statistics.stdev(data.gaps) * 1000.0, "ms"
    else:
        # An estimate against the observed median cadence, never a hardware
        # frame-loss count. Only a complete, strictly increasing axis gets one.
        expected = sum(data.gaps) / median_gap + 1
        value = max(0.0, (expected - data.n_samples) / expected)
        unit = "fraction"
        evidence.update(
            expected_samples=expected,
            observed_samples=data.n_samples,
            denominator="duration / median adjacent gap + 1",
            estimate=True,
        )
    evidence["ungraded_reason"] = _UNGRADED_REASON if acquisition else _RECORDED_REASON
    return MetricResult(
        value=value, unit=unit, status=MetricStatus.REPORT_ONLY, evidence=evidence
    )


@metric(
    level=Level.STREAM,
    family=Family.TIMING,
    requires=Requires(),
    label="low sample rate",
)
def effective_hz(ctx: StreamContext) -> MetricResult:
    """Capture rate from a producer-declared acquisition clock."""
    return _cadence(ctx, acquisition=True, kind="rate")


@metric(
    level=Level.STREAM, family=Family.TIMING, requires=Requires(), label="timing jitter"
)
def dt_jitter_ms(ctx: StreamContext) -> MetricResult:
    """Acquisition interval spread, requiring producer capture-clock evidence."""
    return _cadence(ctx, acquisition=True, kind="spread")


@metric(
    level=Level.STREAM,
    family=Family.TIMING,
    requires=Requires(),
    label="dropped samples",
)
def drop_rate(ctx: StreamContext) -> MetricResult:
    """Missing-sample estimate on a verified capture timebase."""
    return _cadence(ctx, acquisition=True, kind="drop")


@metric(
    level=Level.STREAM, family=Family.TIMING, requires=Requires(), label="recorded rate"
)
def recorded_hz(ctx: StreamContext) -> MetricResult:
    """Median cadence of the recorded timeline, including generated grids."""
    return _cadence(ctx, acquisition=False, kind="rate")


@metric(
    level=Level.STREAM,
    family=Family.TIMING,
    requires=Requires(),
    label="uneven recorded timing",
)
def recorded_dt_spread_ms(ctx: StreamContext) -> MetricResult:
    """Recorded interval spread; zero does not establish perfect capture timing."""
    return _cadence(ctx, acquisition=False, kind="spread")


@metric(
    level=Level.STREAM,
    family=Family.TIMING,
    requires=Requires(),
    label="gaps in recorded timing",
)
def recorded_drop_estimate(ctx: StreamContext) -> MetricResult:
    """Estimated holes in the recorded grid, not verified sensor frame loss."""
    return _cadence(ctx, acquisition=False, kind="drop")


@metric(
    level=Level.STREAM,
    family=Family.TIMING,
    requires=Requires(),
    label="out-of-order timestamps",
)
def monotonic_violations(ctx: StreamContext) -> MetricResult:
    """Repeated/backwards adjacent steps, addressed in original source rows.

    Invalid timestamps break adjacency.
    Missing rows are never removed and then bridged to invent a step.
    Clock origin does not suppress structural evidence.
    """
    data = samples(ctx.stream)
    evidence = clock_evidence(ctx.stream, data)
    evidence["measurement_scope"] = "recorded_timeline"
    reason = data.reason or (
        "fewer than two adjacent valid timestamps to take a gap over"
        if not data.gaps
        else None
    )
    if reason:
        return MetricResult(
            value=None,
            unit=None,
            status=MetricStatus.NOT_APPLICABLE,
            evidence={**evidence, "reason": reason},
        )
    repeated = [
        row for row, gap in zip(data.end_rows, data.gaps, strict=True) if gap == 0
    ]
    backwards = [
        row for row, gap in zip(data.end_rows, data.gaps, strict=True) if gap < 0
    ]
    offending = sorted(repeated + backwards)
    evidence.update(
        fraction=len(offending) / len(data.gaps),
        n_repeated=len(repeated),
        n_backwards=len(backwards),
        first_sample=offending[0] if offending else None,
        sample_indices=offending[:100],
        sample_indices_truncated=len(offending) > 100,
        ungraded_reason=_UNGRADED_REASON,
    )
    return MetricResult(
        value=float(len(offending)),
        unit="count",
        status=MetricStatus.REPORT_ONLY,
        evidence=evidence,
    )
