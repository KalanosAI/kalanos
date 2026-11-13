"""The schema family: metrics that check whether Kalanos understood the file.

They compare what a source declares about its data with what its files store.
The adapter makes that comparison, since the vocabulary belongs to the format.
The metrics report what it found.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from typing import Any

# Internal
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
from kalanos.analysis.models.metrics import (
    EpisodeContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@metric(
    level=Level.EPISODE,
    family=Family.SCHEMA,
    label="columns that disagree with the declared features",
)
def feature_mismatch_count(ctx: EpisodeContext) -> MetricResult:
    """Count the declared features the episode's data file disagrees with.

    Parameters
    ----------
    ctx : EpisodeContext
        The episode to check; only its `schema_check` is read.

    Returns
    -------
    MetricResult
        `not_applicable` when the source declares no column schema
        (`schema_check` is `None`). `report_only` otherwise, with the number of
        mismatched features as the value and each mismatch, with the data file's
        path, as evidence.
    """

    check = ctx.episode.schema_check
    if check is None:
        return not_applicable("the source declares no column schema to compare against")

    return MetricResult(
        value=float(len(check.mismatches)),
        unit="features",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "path": str(check.source_path),
            "n_checked": len(check.checked),
            "mismatches": [m.model_dump(mode="json") for m in check.mismatches],
        },
    )


@metric(
    level=Level.EPISODE,
    family=Family.SCHEMA,
    label="episode rows missing from the data file its metadata names",
)
def episode_file_mismatch(ctx: EpisodeContext) -> MetricResult:
    """Report whether the data file the episode's metadata names holds its rows.

    Parameters
    ----------
    ctx : EpisodeContext
        The episode to check; only its `layout_check` is read.

    Returns
    -------
    MetricResult
        `not_applicable` when the source names no data file per episode,
        or when the episode has no metadata row naming one.
        `report_only` otherwise, `1.0` when the named file lacks the episode's rows,
        whether another file holds them or none does, and `0.0` when it holds them,
        with the named file and the files holding the rows as evidence,
        plus the adapter's `pointer_effect` on a mismatch when it states one.
    """

    check = ctx.episode.layout_check
    if check is None:
        return not_applicable("the source names no data file per episode")
    if check.pointed_path is None:
        return not_applicable("the episode has no metadata row naming a data file")

    evidence: dict[str, Any] = {
        "pointed_path": str(check.pointed_path),
        "holding_paths": [str(p) for p in check.holding_paths],
    }
    if not check.pointer_matches and check.pointer_effect is not None:
        evidence["effect"] = check.pointer_effect
    return MetricResult(
        value=0.0 if check.pointer_matches else 1.0,
        unit="episodes",
        status=MetricStatus.REPORT_ONLY,
        evidence=evidence,
    )


@metric(
    level=Level.EPISODE,
    family=Family.SCHEMA,
    label="episodes with rows in the data and no row in the episode metadata",
)
def episode_not_in_metadata(ctx: EpisodeContext) -> MetricResult:
    """Report whether the episode has rows in the data and no episode metadata row.

    Parameters
    ----------
    ctx : EpisodeContext
        The episode to check; only its `layout_check` is read.

    Returns
    -------
    MetricResult
        `not_applicable` when the source names no data file per episode.
        `report_only` otherwise,
        `1.0` when the episode has no metadata row naming a data file,
        with the files holding its rows as evidence, and `0.0` when it has one.
    """

    check = ctx.episode.layout_check
    if check is None:
        return not_applicable("the source names no data file per episode")

    if check.pointed_path is None:
        return MetricResult(
            value=1.0,
            unit="episodes",
            status=MetricStatus.REPORT_ONLY,
            evidence={"holding_paths": [str(p) for p in check.holding_paths]},
        )
    return MetricResult(
        value=0.0, unit="episodes", status=MetricStatus.REPORT_ONLY, evidence={}
    )


@metric(
    level=Level.EPISODE,
    family=Family.SCHEMA,
    label="episode index range fields that disagree with the data",
)
def index_range_mismatch_count(ctx: EpisodeContext) -> MetricResult:
    """Count the episode's index range fields that disagree with the data's `index`.

    Parameters
    ----------
    ctx : EpisodeContext
        The episode to check; only its `layout_check` is read.

    Returns
    -------
    MetricResult
        `not_applicable` when the source names no data file per episode, or when
        the range was not compared. `report_only` otherwise, with the number of
        disagreeing fields as the value and each field's declared and found value,
        with the files holding the rows, as evidence.
    """

    check = ctx.episode.layout_check
    if check is None:
        return not_applicable("the source names no data file per episode")
    if check.range_mismatches is None:
        return not_applicable(
            "the episode index range could not be compared with the data"
        )

    return MetricResult(
        value=float(len(check.range_mismatches)),
        unit="fields",
        status=MetricStatus.REPORT_ONLY,
        evidence={
            "paths": [str(p) for p in check.holding_paths],
            "mismatches": [m.model_dump(mode="json") for m in check.range_mismatches],
        },
    )
