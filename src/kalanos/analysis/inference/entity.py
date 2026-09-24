"""Identify the column that splits a parsed frame into entities, if any."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import re

# External
import polars as pl

# Internal
from kalanos.analysis.models.schema import EntityKey


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_ENTITY_NAME_PATTERN = re.compile(
    r"(^|_)(id|entity|device|source)(_id)?($|_)", re.IGNORECASE
)

# Below this, no column is trusted as an entity key.
# The Source is then treated as one entity, which `mapping` reads as a single subject.
_ACCEPT_THRESHOLD = 0.5

_NAME_WEIGHT = 0.4
_REPETITION_WEIGHT = 0.6


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _repetition_score(values: list[str]) -> float:
    """How strongly a column's values repeat, as a fraction in `[0, 1]`.

    Parameters
    ----------
    values : list[str]
        Column values in row order.

    Returns
    -------
    float
        `1 - unique / total`.
        `0.0` when every value is unique (no grouping) or the column is empty;
        `0.0` also when there is only one distinct value,
        since a constant column cannot split anything.
    """

    if not values:
        return 0.0

    unique = len(set(values))
    if unique <= 1:
        return 0.0

    return 1 - (unique / len(values))


def score_entity_candidate(name: str, values: list[str]) -> tuple[float, list[str]]:
    """Score one column as an entity-key candidate.

    Parameters
    ----------
    name : str
        The column's name.
    values : list[str]
        The column's values in row order.

    Returns
    -------
    tuple[float, list[str]]
        Confidence in `[0, 1]`, and the evidence behind it.
    """

    # Step 1: gather the name-pattern and repetition signals.
    name_matches = bool(_ENTITY_NAME_PATTERN.search(name))
    repetition = _repetition_score(values)

    # Step 2: combine them into a weighted confidence.
    confidence = _NAME_WEIGHT * float(name_matches) + _REPETITION_WEIGHT * repetition

    # Step 3: record the evidence behind each signal.
    evidence = [
        f"name {name!r} {'matches' if name_matches else 'does not match'} "
        "an entity-key pattern",
        f"{len(set(values))} distinct value(s) across {len(values)} row(s)",
    ]
    return confidence, evidence


def sniff_entity_key(columns: dict[str, list[str]]) -> EntityKey | None:
    """Pick the best entity-key candidate among a frame's non-numeric columns.

    Parameters
    ----------
    columns : dict[str, list[str]]
        Candidate column name to its values in row order —
        the caller excludes numeric columns, since an entity key is categorical.

    Returns
    -------
    EntityKey or None
        The winning column, or `None` when nothing clears `_ACCEPT_THRESHOLD`
        — a file with no entity key is one entity, not an error.
    """

    best: EntityKey | None = None
    for name, values in columns.items():
        confidence, evidence = score_entity_candidate(name, values)
        if confidence >= _ACCEPT_THRESHOLD and (
            best is None or confidence > best.confidence
        ):
            best = EntityKey(column=name, confidence=confidence, evidence=evidence)

    return best


def entity_key(frame: pl.DataFrame) -> EntityKey | None:
    """Find the column that splits a parsed frame into entities, if any.

    Restricts the search to non-numeric columns before scoring —
    an entity key is categorical, and a numeric column is `time_axis`'s to consider.

    Parameters
    ----------
    frame : pl.DataFrame
        An already-parsed frame.

    Returns
    -------
    EntityKey or None
        The winning column, or `None` when the frame is one entity.
    """

    columns = {
        name: frame[name].cast(pl.String).to_list()
        for name, dtype in frame.schema.items()
        if not dtype.is_numeric()
    }
    key = sniff_entity_key(columns)
    if key is None:
        logger.debug("no entity key found; frame treated as one entity")
    else:
        logger.debug("entity key %r (confidence %.2f)", key.column, key.confidence)
    return key
