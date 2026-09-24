"""Classify a time series regular or irregular, and find its nominal interval.

Both are computed from gaps taken within each entity's own series.
Several entities sharing or near-sharing a timestamp
otherwise reads as a burst of zero-length gaps that belongs to no single clock.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import statistics

# Internal
from kalanos.analysis.models.schema import SamplingRegularity


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# How far a gap may sit from the nominal interval and still count as "on
# the grid" — generous enough to absorb clock jitter and an occasional
# drop without masking a genuinely unclocked series.
_TOLERANCE_FRACTION = 0.3
# Share of gaps that must land within tolerance to call the series regular.
_REGULAR_FRACTION_THRESHOLD = 0.8


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def entity_split_gaps(entity_timestamps: list[list[float | None]]) -> list[float]:
    """Pool the within-entity consecutive gaps across every entity.

    Parameters
    ----------
    entity_timestamps : list of list of float or None
        One list of timestamps per entity, each in that entity's own row order.
        Null entries are dropped per entity before sorting, so a missing
        sample reads as one wider gap rather than a break in the series.
        Sorted per entity before differencing, so an out-of-order source still
        yields a physically meaningful gap sequence rather than a run of
        negative gaps from row order alone.

    Returns
    -------
    list[float]
        Every consecutive gap from every entity, pooled.
        Gaps between different entities' rows never appear,
        since each list is differenced only against itself.
    """

    gaps = []
    for timestamps in entity_timestamps:
        ordered = sorted(value for value in timestamps if value is not None)
        gaps.extend(b - a for a, b in zip(ordered, ordered[1:], strict=False))
    return gaps


def regularity(gaps: list[float]) -> SamplingRegularity:
    """Classify pooled within-entity gaps as regular or irregular sampling.

    A series is regular when most of its gaps sit close to a nominal interval
    (the median gap) rather than requiring every gap to match it exactly,
    since jitter and the odd dropped sample are signal-quality problems that
    later metrics grade, not disqualifying evidence that no clock exists at all.

    Parameters
    ----------
    gaps : list[float]
        Pooled within-entity consecutive gaps, from `entity_split_gaps`.

    Returns
    -------
    SamplingRegularity
        `is_regular=True` with the nominal gap as `expected_dt` when
        at least `_REGULAR_FRACTION_THRESHOLD` of gaps fall within
        `_TOLERANCE_FRACTION` of the median; otherwise `is_regular=False`
        with `expected_dt=None`.
        `confidence` is the fraction on the grid either way,
        so a caller can tell a near-miss from an empty series.
    """

    # Step 1: bail out if there are no gaps to classify.
    if not gaps:
        return SamplingRegularity(
            is_regular=False, expected_dt=None, evidence=["no gaps to classify"]
        )

    # Step 2: take the median gap as the nominal interval; bail out if non-positive.
    nominal = statistics.median(gaps)
    if nominal <= 0:
        return SamplingRegularity(
            is_regular=False,
            expected_dt=None,
            evidence=[f"non-positive median gap ({nominal!r}); no interval to trust"],
        )

    # Step 3: measure how many gaps land within tolerance of that nominal interval.
    tolerance = _TOLERANCE_FRACTION * nominal
    on_grid = sum(1 for gap in gaps if abs(gap - nominal) <= tolerance)
    fraction_on_grid = on_grid / len(gaps)
    evidence = [
        f"{on_grid}/{len(gaps)} gaps ({fraction_on_grid:.0%}) within "
        f"{_TOLERANCE_FRACTION:.0%} of the median gap {nominal!r}"
    ]

    # Step 4: regular only if enough gaps cleared the tolerance band.
    if fraction_on_grid >= _REGULAR_FRACTION_THRESHOLD:
        return SamplingRegularity(
            is_regular=True,
            expected_dt=nominal,
            confidence=fraction_on_grid,
            evidence=evidence,
        )
    return SamplingRegularity(
        is_regular=False,
        expected_dt=None,
        confidence=fraction_on_grid,
        evidence=evidence,
    )
