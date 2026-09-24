"""effective_hz, dt_jitter_ms and drop_rate — the timing family.

All three read the time column alone, and all three require regular sampling:
a series indexed by event rather than a steady clock has no meaningful sampling rate,
and reporting one is worse than reporting none.

None of the three is graded here. Grading needs a nominal rate and
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

# The reason every result below carries report_only —
# factored out so the three functions say the same thing rather than
# three slightly different ones.
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
