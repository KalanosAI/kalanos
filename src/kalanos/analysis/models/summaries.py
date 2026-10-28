"""What the dataset-level steps read of each graded episode, as plain values.

Every step that combines episodes, from the camera comparison to the gate,
reads an `EpisodeSummary` instead of the `GradedEpisode` it was taken from.
The coverage ledgers are not kept here:
merging them is associative, so they are merged as the episodes are graded.
A run can then let go of each graded episode as soon as it is summarized,
and hold it only as the JSON it will write.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
from pydantic import BaseModel, ConfigDict

# Internal
from kalanos.analysis.models.eligibility import EligibilityReason
from kalanos.analysis.models.metrics import Level
from kalanos.analysis.models.paths import AnyPath
from kalanos.analysis.models.scoring import ScoreResult


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# One graded result in one episode, as the dataset's metric rollups count it:
# the metric as `family.metric`, the level it ran at,
# the channel, stream or camera it ran on (the episode's id at episode level),
# and whether it graded `warning` or `critical`.
MetricSubject = tuple[str, Level, tuple[str | None, ...], bool]


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class CameraSample(BaseModel):
    """One camera stream in one episode, with what the camera comparison reads of it.

    Attributes
    ----------
    episode_id : str
        The episode the stream belongs to.
    camera : str
        The camera, as `camera_key` names it.
    taxonomy_type, instance, source_field, source_path : str or None
        The stream's own, for the findings and the camera's summary.
    laplacian_var : float or None
        The computed sharpness evidence, `None` when sharpness did not compute.
    exposure_level : float or None
        The computed exposure level, `None` when it did not compute.
    dark_share, bright_share : float or None
        The exposure evidence's crushed and clipped pixel shares.
    luma : tuple[str, ...] or None
        The luminance hashes of a complete read, in order,
        `None` when they cannot identify the footage.
    """

    model_config = ConfigDict(frozen=True)

    episode_id: str
    camera: str
    taxonomy_type: str
    instance: str | None
    source_field: str | None
    source_path: str | None
    laplacian_var: float | None
    exposure_level: float | None
    dark_share: float | None
    bright_share: float | None
    luma: tuple[str, ...] | None


class GateCounts(BaseModel):
    """What the gate's coverage reads of one episode's metric results.

    Attributes
    ----------
    graded : int
        The results that graded `good`, `warning` or `critical`.
    total : dict[str, int]
        Per metric name, the results that were applicable.
    not_applicable : dict[str, int]
        Per metric name, the applicable results that could not grade.
    reasons : dict[str, str]
        Per metric name, the first such result's reason.
    """

    model_config = ConfigDict(frozen=True)

    graded: int
    total: dict[str, int]
    not_applicable: dict[str, int]
    reasons: dict[str, str]


class EpisodeSummary(BaseModel):
    """What every dataset-level step reads of one graded episode.

    Attributes
    ----------
    id : str
        The episode's id.
    source_paths : list[UPath]
        The files it was read from.
    first_task : str or None
        Its first non-blank task instruction.
    score : ScoreResult
        Its score, before eligibility sets `train_ready`.
    requirement_reasons : list[EligibilityReason]
        The reasons its eligibility owes to the scope's requirements, not to findings.
    camera_samples : list[CameraSample]
        Its first stream of each camera.
    gate_counts : GateCounts
        What the gate's coverage reads of its metric results.
    metric_subjects : list[MetricSubject]
        Every result it graded, for the dataset's metric rollups.
        Plain tuples, since a large episode grades many.
    operational_errors : list[dict[str, str]]
        One row per coverage metric that errored.
    unmapped_types : frozenset[str]
        The taxonomy types of its streams that no mapping typed.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    source_paths: list[AnyPath]
    first_task: str | None
    score: ScoreResult
    requirement_reasons: list[EligibilityReason]
    camera_samples: list[CameraSample]
    gate_counts: GateCounts
    metric_subjects: list[MetricSubject]
    operational_errors: list[dict[str, str]]
    unmapped_types: frozenset[str]
