"""effective_hz, dt_jitter_ms, drop_rate and monotonic_violations — the timing family.

All four read the time column alone. All but monotonic_violations require
regular sampling: a series indexed by event rather than a steady clock has no
meaningful sampling rate, and reporting one is worse than reporting none.
monotonic_violations asks only whether the clock moves forward, which any clocked
series must, however irregular — and a clock broken badly enough to fail the
regularity test is exactly the one it must still see.

None of them is graded here. Grading needs a nominal rate and
threshold bands that live in the policy, which this stage does not read —
every result below carries `report_only` with an `ungraded_reason` in its evidence,
so scoring can tell "not graded yet" apart from a metric like `p99_torque`
that is report-only by design and will never be graded at all.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import statistics

# Internal
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
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


# Two samples are the fewest that yield even one gap, which is what
# effective_hz needs and no more.
_REQUIRES_REGULAR_SAMPLING = Requires(regular_sampling=True, min_samples=2)

# A standard deviation over one or two gaps is not a spread anyone should act on;
# five samples buys four gaps, enough for the number to mean something.
_REQUIRES_ENOUGH_GAPS_FOR_A_SPREAD = Requires(regular_sampling=True, min_samples=5)

# Whether a clock advances needs one gap and nothing about its regularity.
_REQUIRES_A_GAP = Requires(min_samples=2)

# The reason every result below carries report_only —
# factored out so the functions all say the same thing rather than
# slightly different ones.
_UNGRADED_REASON = "grading needs policy thresholds, which this stage does not read"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _ordered_timestamps(ctx: StreamContext) -> list[float]:
    """List the stream's timestamps in row order, with any nulls dropped.

    A row whose timestamp did not resolve carries no gap information on either side
    of it, so it is left out entirely rather than subtracted against `None`.

    Parameters
    ----------
    ctx : StreamContext
        The stream context; only its `timestamps` are read.

    Returns
    -------
    list[float]
        Non-null timestamps, in their original row order.
    """

    return ctx.timestamps.drop_nulls().to_list()


def _consecutive_gaps(ctx: StreamContext) -> list[float]:
    """List the gaps between consecutive non-null timestamps.

    Parameters
    ----------
    ctx : StreamContext
        The stream context; only its `timestamps` are read.

    Returns
    -------
    list[float]
        Every `ordered[i + 1] - ordered[i]`,
        one shorter than the number of non-null timestamps — possibly empty.
    """

    ordered = _ordered_timestamps(ctx)
    return [b - a for a, b in zip(ordered, ordered[1:], strict=False)]


@metric(level=Level.STREAM, family=Family.TIMING, requires=_REQUIRES_REGULAR_SAMPLING)
def effective_hz(ctx: StreamContext) -> MetricResult:
    """Samples per second, from the median gap between timestamps.

    Median rather than mean, so one long gap cannot move it.

    Parameters
    ----------
    ctx : StreamContext
        The stream to measure; only its `timestamps` are read.

    Returns
    -------
    MetricResult
        `not_applicable`, with a reason in `evidence`, when:
        - fewer than two valid timestamps survive to take a gap over
        - the median gap is zero or negative
        `report_only` otherwise.
    """

    gaps = _consecutive_gaps(ctx)
    if not gaps:
        return not_applicable("fewer than two valid timestamps to take a gap over")

    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return not_applicable("median gap is zero or negative")

    return MetricResult(
        value=1.0 / median_gap,
        unit="Hz",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_gaps": len(gaps),
            "median_gap_s": median_gap,
            "ungraded_reason": _UNGRADED_REASON,
        },
    )


@metric(
    level=Level.STREAM,
    family=Family.TIMING,
    requires=_REQUIRES_ENOUGH_GAPS_FOR_A_SPREAD,
)
def dt_jitter_ms(ctx: StreamContext) -> MetricResult:
    """Standard deviation of the gaps between consecutive timestamps.

    Parameters
    ----------
    ctx : StreamContext
        The stream to measure; only its `timestamps` are read.

    Returns
    -------
    MetricResult
        `not_applicable`, with a reason in `evidence`, when:
        - fewer than two gaps survive to take a spread over
        - the median gap is zero or negative
        `report_only` otherwise.
    """

    gaps = _consecutive_gaps(ctx)
    if len(gaps) < 2:
        return not_applicable("fewer than two gaps to take a spread over")
    if statistics.median(gaps) <= 0:
        return not_applicable("median gap is zero or negative")

    jitter_s = statistics.stdev(gaps)
    return MetricResult(
        value=jitter_s * 1000.0,
        unit="ms",
        status=MetricStatus.REPORT_ONLY,
        evidence={"n_gaps": len(gaps), "ungraded_reason": _UNGRADED_REASON},
    )


@metric(level=Level.STREAM, family=Family.TIMING, requires=_REQUIRES_REGULAR_SAMPLING)
def drop_rate(ctx: StreamContext) -> MetricResult:
    """Fraction of expected samples that never arrived.

    Expected count is this series' own duration divided by its own median gap,
    so a stream is judged against the clock it actually kept rather than
    a nominal rate declared elsewhere.

    Parameters
    ----------
    ctx : StreamContext
        The stream to measure; only its `timestamps` are read.

    Returns
    -------
    MetricResult
        `not_applicable`, with a reason in `evidence`, when:
        - fewer than two valid timestamps survive to measure a duration over
        - the duration or the median gap is zero or negative
        `report_only` otherwise.
    """

    ordered = _ordered_timestamps(ctx)
    if len(ordered) < 2:
        return not_applicable(
            "fewer than two valid timestamps to measure a duration over"
        )

    duration = ordered[-1] - ordered[0]
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    median_gap = statistics.median(gaps)

    if duration <= 0 or median_gap <= 0:
        return not_applicable("duration or median gap is zero or negative")

    expected_samples = duration / median_gap + 1
    observed_samples = len(ordered)
    fraction = max(0.0, (expected_samples - observed_samples) / expected_samples)

    return MetricResult(
        value=fraction,
        unit="fraction",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "expected_samples": expected_samples,
            "observed_samples": observed_samples,
            "ungraded_reason": _UNGRADED_REASON,
        },
    )


@metric(level=Level.STREAM, family=Family.TIMING, requires=_REQUIRES_A_GAP)
def monotonic_violations(ctx: StreamContext) -> MetricResult:
    """Count the samples whose timestamp is at or before the one before it.

    A timestamp equal to its predecessor is a repeated step — in simulation,
    usually a dropped physics step; one earlier than its predecessor is a clock
    that ran backwards, a reordered or merged log. Both break velocity estimates
    and time alignment, and neither shows in `drop_rate`, which only sees samples
    that never arrived. Row order is kept, never sorted, so a backwards step
    is visible.

    Parameters
    ----------
    ctx : StreamContext
        The stream to measure; only its `timestamps` are read.

    Returns
    -------
    MetricResult
        `not_applicable` when fewer than two valid timestamps survive;
        `report_only` otherwise, as a count, with the fraction of steps,
        the repeated and backwards counts and the first offending sample
        in `evidence`.
    """

    gaps = _consecutive_gaps(ctx)
    if not gaps:
        return not_applicable("fewer than two valid timestamps to take a gap over")

    repeated = [index for index, gap in enumerate(gaps) if gap == 0]
    backwards = [index for index, gap in enumerate(gaps) if gap < 0]
    offending = sorted(repeated + backwards)
    return MetricResult(
        value=float(len(offending)),
        unit="count",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "n_gaps": len(gaps),
            "fraction": len(offending) / len(gaps),
            "n_repeated": len(repeated),
            "n_backwards": len(backwards),
            "first_sample": offending[0] + 1 if offending else None,
            "ungraded_reason": _UNGRADED_REASON,
        },
    )
