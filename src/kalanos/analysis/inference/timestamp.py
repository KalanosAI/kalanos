"""Identify the time column and its unit from name, monotonicity, and magnitude.

A signal column can share a `_ms`-style suffix with the time column without
being monotonic, and a genuine time column need not be named `t` or `time`.
The unit is read primarily from the sampling interval's magnitude;
a unit suffix in the column name corroborates that reading when present,
and a disagreement between the two lowers confidence instead of picking a
winner silently.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import re
from dataclasses import dataclass

# External
import polars as pl

# Internal
from kalanos.analysis.inference.regularity import entity_split_gaps
from kalanos.analysis.inference.regularity import regularity as _regularity
from kalanos.analysis.models.schema import TimeSpec


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_TIME_NAME_PATTERN = re.compile(r"(^|_)(t|ts|time|timestamp)(_|$)", re.IGNORECASE)

# Below this, a candidate is not trusted as the time axis — kept as a
# module constant so the orchestrator and its tests read the same number.
_ACCEPT_THRESHOLD = 0.6

# Neither weight alone reaches _ACCEPT_THRESHOLD, so neither signal can win
# the time axis by itself — a name match still needs some monotonicity, and
# a monotonic column still needs a time-like name.
_NAME_WEIGHT = 0.45
_MONOTONIC_WEIGHT = 0.55

# A trailing `_ms` / `_sec` / ... suffix, or a bracketed `(ms)` / `[ms]`
# form anywhere in the name.
# Deliberately narrow: only SI time-unit tokens,
# so `arm_x_mm` or a bare trailing `s` (as in `ts`, `times`) cannot match.
_UNIT_TOKEN = r"s|ms|us|ns|sec|msec|usec|nsec"
_UNIT_SUFFIX_PATTERN = re.compile(
    rf"(?:_(?:in_)?({_UNIT_TOKEN})$)|(?:[(\[]\s*({_UNIT_TOKEN})\s*[)\]])",
    re.IGNORECASE,
)
# fmt: off
_UNIT_TOKEN_CANONICAL = {
    "s":    "s",
    "sec":  "s",
    "ms":   "ms",
    "msec": "ms",
    "us":   "us",
    "usec": "us",
    "ns":   "ns",
    "nsec": "ns",
}
# fmt: on

# Applied to the column's confidence when the header
# and the magnitude disagree on the unit.
_UNIT_MISMATCH_PENALTY = 0.8

# Applied when the header is the only signal for the unit, with no magnitude
# to corroborate it — unlike an agreement, which had something to agree with.
_UNCORROBORATED_UNIT_PENALTY = 0.9


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass
class TimeCandidateScore:
    """One column's score as a time-axis candidate, with the evidence behind it.

    Attributes
    ----------
    confidence : float
        Weighted combination of the name-pattern and monotonicity signals,
        in `[0, 1]`.
    evidence : list[str]
        What each signal found, independent of whether the column wins.
    """

    confidence: float
    evidence: list[str]


@dataclass
class UnitResolution:
    """A time column's resolved unit, and how the header and magnitude agreed.

    Attributes
    ----------
    unit : str
        The resolved unit. Magnitude's reading when a nominal interval is available,
        the header's when it is not, `"unknown"` when neither is.
    evidence : list[str]
        What each signal found, independent of which one decided `unit`.
    confidence_factor : float
        Multiplies the column's confidence, in `[0, 1]`.
        `1.0` only when the header and magnitude agree;
        discounted whenever the unit rests on a single, uncorroborated signal,
        and discounted further when the two signals actively disagree.
    """

    unit: str
    evidence: list[str]
    confidence_factor: float


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _monotonic_fraction(values: list[float | None]) -> float:
    """Fraction of consecutive steps that are non-decreasing.

    Parameters
    ----------
    values : list[float or None]
        Column values in row order.

    Returns
    -------
    float
        Nulls are dropped first, then consecutive non-null values are compared:
        the value before a null is compared against the value after it,
        so a step backwards hidden behind a null still counts against the column.
        `0.0` for a column with fewer than two non-null values.
    """

    present = [value for value in values if value is not None]
    if len(present) < 2:
        return 0.0

    steps = len(present) - 1
    non_decreasing = sum(
        1 for a, b in zip(present, present[1:], strict=False) if b >= a
    )
    return non_decreasing / steps


def score_time_candidate(name: str, values: list[float | None]) -> TimeCandidateScore:
    """Score one numeric column as a time-axis candidate.

    Exposed separately from the column that wins so a test can show which
    signal rules a non-time column out, rather than asserting on the winner alone.

    Parameters
    ----------
    name : str
        The column's name.
    values : list[float or None]
        The column's values in row order. A null entry is skipped when
        measuring monotonicity rather than disqualifying the column.

    Returns
    -------
    TimeCandidateScore
        The combined confidence and the evidence behind it.
    """

    # Step 1: gather the name-pattern and monotonicity signals.
    name_matches = bool(_TIME_NAME_PATTERN.search(name))
    monotonic_fraction = _monotonic_fraction(values)

    # Step 2: combine them into a weighted confidence.
    confidence = (
        _NAME_WEIGHT * float(name_matches) + _MONOTONIC_WEIGHT * monotonic_fraction
    )

    # Step 3: record the evidence behind each signal.
    evidence = [
        f"name {name!r} {'matches' if name_matches else 'does not match'} "
        "a time-like pattern",
        f"non-decreasing on {monotonic_fraction:.0%} of consecutive steps",
    ]
    null_count = sum(1 for value in values if value is None)
    if null_count:
        evidence.append(
            f"{null_count} of {len(values)} values are null and were skipped"
        )
    return TimeCandidateScore(confidence=confidence, evidence=evidence)


def infer_unit(nominal_interval: float) -> str:
    """Infer a time unit from the magnitude of a sampling interval alone.

    Never reads the column's name — see `reconcile_unit` for the
    cross-check that catches this when the magnitude alone is wrong.

    Parameters
    ----------
    nominal_interval : float
        The representative gap between samples,
        in whatever scale the raw values are recorded.

    Returns
    -------
    str
        - `"unknown"` for a zero interval — a column that never advances
          has no magnitude to read a unit from
        - `"s"` for a sub-1 interval
        - `"ms"` for 1 up to 1000
        - `"us"` for 1000 up to 1e6
        - `"ns"` beyond that
        Bands are centred on the sampling intervals real robotics loops produce
        (roughly 1 Hz to 1 kHz), not on epoch-timestamp magnitudes.
    """

    magnitude = abs(nominal_interval)
    if magnitude == 0:
        return "unknown"
    if magnitude < 1:
        return "s"
    if magnitude < 1_000:
        return "ms"
    if magnitude < 1_000_000:
        return "us"
    return "ns"


def unit_from_name(name: str) -> str | None:
    """Read a time unit off a column name's suffix, if it carries one.

    Matches a trailing `_ms`-style suffix or a bracketed `(ms)` / `[ms]` form only.
    A bare trailing letter doesn't count, so `ts` or `times` isn't read as seconds,
    and the suffix has to be trailing, so `t_ms_raw` isn't caught.

    Parameters
    ----------
    name : str
        The column's name.

    Returns
    -------
    str or None
        `"s"`, `"ms"`, `"us"` or `"ns"` if the name carries a recognisable
        unit suffix, `None` otherwise.
    """

    match = _UNIT_SUFFIX_PATTERN.search(name)
    if match is None:
        return None
    token = (match.group(1) or match.group(2)).lower()
    return _UNIT_TOKEN_CANONICAL[token]


def reconcile_unit(name: str, nominal_interval: float | None) -> UnitResolution:
    """Resolve a time column's unit from its sampling magnitude and its name.

    Magnitude is the primary signal: it reflects what the data actually does,
    where a header suffix can be missing or wrong.
    The name corroborates that reading when both are available,
    and stands in for it when there's no interval to measure —
    the one case where the header decides on its own, at a discount,
    since nothing else confirms it.

    Parameters
    ----------
    name : str
        The time column's name, as it appears in the source.
    nominal_interval : float or None
        The representative gap between samples,
        or `None` when no regular interval could be established.

    Returns
    -------
    UnitResolution
        The resolved unit, the evidence behind it, and a confidence multiplier
        that drops below `1.0` whenever the unit rests on one uncorroborated
        signal, and drops further when the header and the magnitude disagree.
    """

    # Step 1: read whatever unit the column's name carries, if any.
    header_unit = unit_from_name(name)

    # Step 2: with no magnitude to read, the header is all there is —
    # fall back to it, discounted since nothing corroborates it, or admit
    # the unit is unknown.
    if nominal_interval is None:
        if header_unit is not None:
            return UnitResolution(
                unit=header_unit,
                evidence=[
                    "no regular interval to read a magnitude from; "
                    f"unit taken from name suffix {header_unit!r}, uncorroborated"
                ],
                confidence_factor=_UNCORROBORATED_UNIT_PENALTY,
            )
        return UnitResolution(
            unit="unknown",
            evidence=["no regular interval, and no unit suffix in the name"],
            confidence_factor=1.0,
        )

    # Step 3: read the magnitude — the primary signal, and the fallback answer
    # whenever there is nothing to check it against.
    magnitude_unit = infer_unit(nominal_interval)
    evidence = [f"nominal interval {nominal_interval!r} -> unit {magnitude_unit!r}"]

    if header_unit is None:
        evidence.append(f"name {name!r} carries no unit suffix; no cross-check")
        return UnitResolution(
            unit=magnitude_unit, evidence=evidence, confidence_factor=1.0
        )

    # Step 4: compare the two — agreement costs nothing, disagreement keeps
    # the magnitude's answer but prices the doubt into confidence.
    if header_unit == magnitude_unit:
        evidence.append(
            f"name suffix {header_unit!r} agrees with the interval magnitude"
        )
        return UnitResolution(
            unit=magnitude_unit, evidence=evidence, confidence_factor=1.0
        )

    evidence.append(
        f"name suffix {header_unit!r} disagrees with the interval magnitude "
        f"{magnitude_unit!r}; magnitude wins"
    )
    return UnitResolution(
        unit=magnitude_unit,
        evidence=evidence,
        confidence_factor=_UNIT_MISMATCH_PENALTY,
    )


def _entity_split(
    time_values: list[float | None], frame: pl.DataFrame, entity_column: str | None
) -> list[list[float | None]]:
    """Group a time column's values by entity, for within-entity gap pooling.

    Parameters
    ----------
    time_values : list[float or None]
        The time column's values, in the frame's row order.
    frame : pl.DataFrame
        The frame `time_values` was read from, used to read `entity_column`.
    entity_column : str or None
        The column that splits the frame into entities, or `None`.

    Returns
    -------
    list[list[float or None]]
        One list per entity, in first-seen order — a single list holding
        every value when `entity_column` is `None`.
    """

    if entity_column is None:
        return [time_values]

    entity_values = frame[entity_column].cast(pl.String).to_list()
    groups: dict[str, list[float | None]] = {}
    for value, key in zip(time_values, entity_values, strict=True):
        groups.setdefault(key, []).append(value)
    return list(groups.values())


def time_axis(frame: pl.DataFrame, *, entity_column: str | None = None) -> TimeSpec:
    """Resolve a parsed frame's time axis: which column, what unit, how confident.

    Always returns a `TimeSpec`. `column` is `None` when no candidate cleared
    the acceptance threshold — `confidence` and `evidence` still describe the
    best candidate considered, so a caller can say why it was refused
    instead of just that it was.

    Parameters
    ----------
    frame : pl.DataFrame
        An already-parsed frame.
    entity_column : str or None
        The column that splits the frame into entities, if `entity_key`
        already found one. Gaps are pooled within each entity so that rows
        from different entities sharing a clock don't read as zero-length gaps.

    Returns
    -------
    TimeSpec
        The resolved time axis, its unit, and the regularity of its sampling —
        refused (via `column=None`) rather than raised when nothing qualifies.
    """

    # Step 1: only a numeric column can be a time axis.
    numeric_columns = {
        name: frame[name].cast(pl.Float64).to_list()
        for name, dtype in frame.schema.items()
        if dtype.is_numeric()
    }
    if not numeric_columns:
        logger.debug("no numeric column to consider as a time axis")
        return TimeSpec(
            column=None,
            unit="unknown",
            confidence=0.0,
            evidence=["no numeric column found to consider as a time axis"],
        )

    # Step 2: score every numeric column as a time-axis candidate, keep the best.
    candidates = iter(numeric_columns.items())
    best_column, best_values = next(candidates)
    best_score = score_time_candidate(best_column, best_values)
    for name, values in candidates:
        score = score_time_candidate(name, values)
        if score.confidence > best_score.confidence:
            best_column, best_values, best_score = name, values, score

    # Step 3: derive the winner's sampling regularity, then reconcile its
    # unit against its name — a disagreement here can still pull the
    # combined confidence below the gate checked next.
    entity_groups = _entity_split(best_values, frame, entity_column)
    gaps = entity_split_gaps(entity_groups)
    sampling = _regularity(gaps)
    unit_resolution = reconcile_unit(best_column, sampling.expected_dt)
    confidence = best_score.confidence * unit_resolution.confidence_factor
    evidence = best_score.evidence + unit_resolution.evidence
    logger.debug(
        "time axis candidate %r scored %.2f, unit %r",
        best_column,
        confidence,
        unit_resolution.unit,
    )

    # Step 4: refuse if nothing cleared the confidence bar — column identification
    # and unit agreement combined — while still reporting what was determined about
    # the best candidate.
    if confidence < _ACCEPT_THRESHOLD:
        logger.debug(
            "no time axis: %r scored %.2f, below %.2f",
            best_column,
            confidence,
            _ACCEPT_THRESHOLD,
        )
        return TimeSpec(
            column=None,
            unit=unit_resolution.unit,
            confidence=confidence,
            evidence=[
                *evidence,
                f"{best_column!r} scored {confidence:.2f}, "
                f"below the {_ACCEPT_THRESHOLD} acceptance threshold",
            ],
            regularity=sampling,
        )

    return TimeSpec(
        column=best_column,
        unit=unit_resolution.unit,
        confidence=confidence,
        evidence=evidence,
        regularity=sampling,
    )
