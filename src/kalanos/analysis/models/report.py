"""The Report model: what `reporting` assembles and persists.

`schema_version` defaults to the real, current version rather than a
placeholder, because old report JSON on disk will be read by newer code —
a reader needs an actual version string to decide whether it understands the file.

The graded models below mirror `domain.py`'s
`Dataset → Episode → Stream → Channel` tree one level at a time,
each node carrying the `ScoreResult` rolled up to it.
The identity lives in the tree's own shape,
rather than in a string built from names that can themselves contain a separator —
an instance tag comes from an arbitrary `id` column value,
and a path is a real filesystem path.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections import Counter
from enum import Enum
from typing import Any, Literal

# External
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.binding import (
    BindingConflict,
    EvaluationScope,
)
from kalanos.analysis.models.coverage import Availability, CoverageRow
from kalanos.analysis.models.coverage import Coverage as AnalysisCoverage

# Internal
from kalanos.analysis.models.diagnostics import DiagnosticResult, DiagnosticsReport
from kalanos.analysis.models.dictionary import CategoryGroup
from kalanos.analysis.models.discovery import SkippedSource, SourceInfo
from kalanos.analysis.models.domain import (
    Attribution,
    Channel,
    Clock,
    ClockInfo,
    Episode,
    MappingSource,
    SourceOrder,
)
from kalanos.analysis.models.eligibility import (
    EligibilityCounts,
    EligibilityStatus,
    EpisodeEligibility,
    Readiness,
    Sufficiency,
)
from kalanos.analysis.models.mapping import MappingOverride
from kalanos.analysis.models.metrics import MetricResult
from kalanos.analysis.models.paths import AnyPath
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.provenance import Inventory, Producer, RunInfo
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Bumped whenever the graded tree's shape changes: two shapes can carry the same
# field names, so a reader cannot tell them apart by content alone.
CURRENT_SCHEMA_VERSION = "7.0.0"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class AnalysedEpisode(BaseModel):
    """One recording an adapter read, with the adapter's name and its grading policy.

    Attributes
    ----------
    episode : Episode
        The recording an adapter produced.
    adapter : str
        The adapter that read it.
    adapter_confidence : float
        The adapter's winning bid on this episode's source path,
        from `Selection.confidence`.
    policy : Policy
        The policy to grade it against: the run's own policy,
        or a copy with a per-deployment limit `pipeline.run` filled in.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode: Episode
    adapter: str
    adapter_confidence: float
    policy: Policy


class GradedChannel(BaseModel):
    """One Channel with its metrics regraded and rolled up to a score.

    Attributes
    ----------
    channel : Channel
        The channel, exactly as the canonical model carries it.
    score : ScoreResult
        The Channel-level rollup, from `scoring.score_metrics`.
    metrics : dict[str, MetricResult]
        Every metric computed for this channel, keyed by name and regraded
        against the policy, via `scoring.resolve_status`.
    """

    channel: Channel
    score: ScoreResult
    metrics: dict[str, MetricResult] = Field(default_factory=dict)


class PayloadStatus(str, Enum):
    """Whether a stream's channel payload was actually examined.

    A required capability's outcome must be explicit: a stream that was never
    read cannot pass by having another stream in the same family pass.
    """

    # fmt: off
    COMPUTED      = "computed"       # Payload fetched; channels graded
    NOT_REQUIRED  = "not_required"   # No channels to grade (video, text)
    MISSING_INPUT = "missing_input"  # Channels declared but no payload to read
    SKIPPED       = "skipped"        # Execution tier did not read payloads
    ERROR         = "error"          # Fetching or grading the payload raised
    # fmt: on


class StreamEvaluation(BaseModel):
    """What was done with a stream's payload, and why when it was not read.

    Attributes
    ----------
    payload : PayloadStatus
    reason : str or None
    n_channels_declared : int
        Channels the source declares, whether or not they were graded.
    n_channels_graded : int
    """

    payload: PayloadStatus = PayloadStatus.COMPUTED
    reason: str | None = None
    n_channels_declared: int = 0
    n_channels_graded: int = 0


class CameraFrame(BaseModel):
    """One frame of a camera stream's shared read.

    Attributes
    ----------
    source_row : int
        The frame's row in the stream.
    presentation_time_s : float or None
        The container's presentation time, `None` where it has none.
    shape : list of int
        The native luminance's `[height, width]`.
    blur_score : float
        The native luminance's Laplacian variance.
    clipped_fraction : float
        The share of pixels crushed black plus the share clipped white.
    luma_sha256 : str or None
        SHA-256 of the native luminance, set only on an evidence frame.
    """

    source_row: int
    presentation_time_s: float | None = None
    shape: list[int]
    blur_score: float
    clipped_fraction: float
    luma_sha256: str | None = None


class CameraFrames(BaseModel):
    """What one camera stream's shared read examined, as evidence.

    Attributes
    ----------
    requested_rows : list of int
        The rows the sample asked for.
    missing_rows : list of int
        Requested rows the read did not measure.
    frames : list of CameraFrame
        One per examined frame, in order.
    decoded_frames : int
        The frames the read decoded inside its windows, seek preroll excluded.
    adjacent_pairs_examined : int
        Neighbouring frame pairs the freeze read compared.
    identical_adjacent_pairs : list of list of int
        `[row, row + 1]` for each compared pair with no difference at all.
    sample_plan : str
        `evenly_spaced_v1` for a sample, `full_scan_v1` for every frame.
    parameters : dict
        The `VisionSpec` the read ran under.
    availability : Availability
        `computed`, or why the read stopped early.
    reason : str or None
        What stopped the read, when it stopped.
    """

    requested_rows: list[int]
    missing_rows: list[int]
    frames: list[CameraFrame]
    decoded_frames: int
    adjacent_pairs_examined: int
    identical_adjacent_pairs: list[list[int]]
    sample_plan: Literal["evenly_spaced_v1", "full_scan_v1"]
    parameters: dict[str, Any]
    availability: Availability
    reason: str | None = None


class GradedStream(BaseModel):
    """One Stream with its own rolled-up score.

    Attributes
    ----------
    taxonomy_type : str
        What the signal is — mirrors `domain.Stream`.
    instance : str or None
        Which subject it belongs to, or `None` for a single-subject recording.
    attribution : Attribution
        Where `instance` came from — mirrors `domain.Stream.attribution`.
    category : str or None
        The dictionary category of `taxonomy_type`, `None` when the stream is unmapped.
    mapping_source : MappingSource or None
        How `taxonomy_type` was decided — mirrors `domain.Stream.mapping_source`.
    score : ScoreResult
        The Stream-level rollup over the stream's own metrics and `channels`.
    metrics : dict[str, MetricResult]
        Every metric computed on the stream itself keyed by name
        and regraded against the policy, via `scoring.resolve_status`.
    channels : list[GradedChannel]
        The graded channels within this stream.
    evaluation : StreamEvaluation
        Whether the payload was read. Requirements decide eligibility from
        this, not from whether some other stream produced a result.
    frames : CameraFrames or None
        The camera's shared read: the rows examined and their per-frame evidence.
        `None` when the stream was not read as camera footage.
    """

    taxonomy_type: str
    instance: str | None = None
    attribution: Attribution = Attribution.SINGLE
    category: str | None = None
    mapping_source: MappingSource | None = None
    score: ScoreResult
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    channels: list[GradedChannel] = Field(default_factory=list)
    kind: str | None = None
    source_path: str | None = None
    coverage: list[CoverageRow] = Field(default_factory=list)
    source_field: str | None = None
    source_identity: str | None = None
    declared_channels: list[Channel] = Field(default_factory=list)
    clock: Clock = Clock.UNKNOWN
    clock_info: ClockInfo | None = None
    source_order: SourceOrder = Field(default_factory=SourceOrder)
    evaluation: StreamEvaluation = Field(default_factory=StreamEvaluation)
    frames: CameraFrames | None = None


class GradedEpisode(BaseModel):
    """One Episode with its own rolled-up score.

    Attributes
    ----------
    id : str
        The recording's identifier — mirrors `domain.Episode`.
    adapter : str
        The adapter that read this recording.
    adapter_confidence : float
        The adapter's winning bid on this episode's source path,
        from `AnalysedEpisode.adapter_confidence`.
    source_paths : list[UPath]
        The files this episode's streams were read from, so a reader can trace
        a poor score back to something on disk.
    duration_s : float or None
        Seconds from the earliest timestamp in any stream to the latest in any stream.
        `None` when no stream carries a timestamp.
    sample_count : int
        The length of the episode's longest stream, 0 when it has none.
    score : ScoreResult
        The Episode-level rollup over the episode's own metrics and `streams`.
    metrics : dict[str, MetricResult]
        Every metric computed on the episode itself, keyed by name and
        regraded against the policy, via `scoring.resolve_status`.
    streams : list[GradedStream]
        The graded streams within this episode.
    tasks : list[str] or None
        The episode's task instructions as the source stored them — mirrors
        `domain.Episode.tasks`, so a reader sees what each episode was told.
    eligibility : EpisodeEligibility or None
        The authoritative decision. `None` only while the episode is being
        assembled; a finished Report refuses an episode without one.
    """

    id: str
    adapter: str
    adapter_confidence: float
    source_paths: list[AnyPath] = Field(default_factory=list)
    duration_s: float | None = None
    sample_count: int = 0
    score: ScoreResult
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    streams: list[GradedStream] = Field(default_factory=list)
    tasks: list[str] | None = None
    eligibility: EpisodeEligibility | None = None
    coverage: AnalysisCoverage | None = None
    diagnostics: list[DiagnosticResult] = Field(default_factory=list)


class FailingEpisode(BaseModel):
    """An episode whose eligibility is `blocked`, and the reasons why.

    Derived from `GradedEpisode.eligibility`; never decided here.

    Attributes
    ----------
    episode_id : str
        The episode, as it appears in `Report.episodes`.
    reasons : list[str]
        Each critical finding that failed it, as `stream[/channel].metric`.
    """

    episode_id: str
    reasons: list[str]


class TaskTrait(BaseModel):
    """A blocking finding on every episode of one task and on none of any other.

    Descriptive since schema 7: it is reported so a reader can see a task-wide
    pattern, and it exempts nothing. Prevalence cannot excuse a violated
    contract; a scoped policy rule can, explicitly.

    Attributes
    ----------
    task : str
        The task instruction the episodes share.
    finding : str
        The finding, as `stream[/channel].metric`.
    n_episodes : int
        How many episodes the task has, all of which carry the finding.
    """

    task: str
    finding: str
    n_episodes: int


class DatasetTrait(BaseModel):
    """A blocking finding on (nearly) every graded episode of the dataset.

    Descriptive since schema 7: it may describe how the dataset was recorded,
    or it may be corruption in every episode; the report cannot tell, so it
    exempts nothing. Blocked episodes stay blocked.

    Attributes
    ----------
    finding : str
        The finding, as `stream[/channel].metric`.
    n_episodes : int
        How many episodes the dataset has.
    n_with_finding : int
        How many of them carry the finding: at least the gate's share.
    """

    finding: str
    n_episodes: int
    n_with_finding: int


class CameraSummary(BaseModel):
    """One camera compared across the episodes it appears in.

    Attributes
    ----------
    camera : str
        The camera's key: its source field, else its taxonomy type and instance.
    stream : str
        The camera's taxonomy type.
    instance, source_field : str or None
        Mirror the camera's `GradedStream`.
    n_episodes : int
        How many episodes the camera appears in.
    compared : bool
        Whether enough episodes had a blur or an exposure value to compare them.
    reason : str or None
        Which of the two had too few episodes to compare.
    laplacian_var, exposure_level : float or None
        The medians over the camera's episodes, `None` where none computed.
    trait : {"dark", "white"} or None
        How the camera is exposed in every compared episode: descriptive.
    blur_outliers, exposure_outliers : list[str]
        The episodes flagged as blurrier, or exposed unlike, the camera.
    """

    camera: str
    stream: str
    instance: str | None = None
    source_field: str | None = None
    n_episodes: int
    compared: bool
    reason: str | None = None
    laplacian_var: float | None = None
    exposure_level: float | None = None
    trait: Literal["dark", "white"] | None = None
    blur_outliers: list[str] = Field(default_factory=list)
    exposure_outliers: list[str] = Field(default_factory=list)


class NotObservable(BaseModel):
    """A metric that could not grade this dataset, and why.

    Attributes
    ----------
    metric : str
        The metric's bare name.
    reason : str
        Why it was not applicable, as the metric said.
    share : float
        The share of its results, from 0 to 1, that were not applicable.
    """

    metric: str
    reason: str
    share: float


class Coverage(BaseModel):
    """What a dataset's grade rests on, shown beside it on every report.

    Attributes
    ----------
    families_graded : list[str]
        The families that graded at least one metric.
    graded_checks_per_episode : float or None
        The median number of graded results in an episode.
    not_observable : list[NotObservable]
        Metrics that could not grade most of this dataset, with the reason.
    """

    families_graded: list[str]
    graded_checks_per_episode: float | None
    not_observable: list[NotObservable] = Field(default_factory=list)


class Gate(BaseModel):
    """The dataset gate: which episodes are blocked, and what the grade rests on.

    Since schema 7 the gate decides nothing; `failing_episodes` lists the
    episodes whose eligibility is `blocked`. Letter fields remain for
    compatibility and are not a decision surface.

    Attributes
    ----------
    n_episodes : int
        The episodes graded.
    failing_episodes : list[FailingEpisode]
        The episodes whose eligibility is `blocked`.
    failing_share : float
        `len(failing_episodes) / n_episodes`.
    task_traits : list[TaskTrait]
        Blocking findings shared by every episode of one task: descriptive.
    dataset_traits : list[DatasetTrait]
        Blocking findings on (nearly) every episode: descriptive.
    uncapped_grade : Grade or None
        The letter the mean alone would get.
    cap : Grade or None
        The best letter the failing share allows; `None` when it allows any.
    pruned_score : float or None
        Mean quality of the non-blocked episodes: a description of a
        candidate selection, not a claim that it is sufficient training data.
        `None` when none are blocked or none remain.
    coverage : Coverage
        What the grade rests on.
    summary : str
        One plain-language line: the grade, why, and what it rests on.
    """

    n_episodes: int
    failing_episodes: list[FailingEpisode] = Field(default_factory=list)
    failing_share: float
    task_traits: list[TaskTrait] = Field(default_factory=list)
    dataset_traits: list[DatasetTrait] = Field(default_factory=list)
    uncapped_grade: Grade | None
    cap: Grade | None
    pruned_score: float | None = None
    coverage: Coverage
    summary: str


class Report(BaseModel):
    """The graded result of one analysis run.

    Attributes
    ----------
    schema_version : str
        The version of this model the report was written under.
        Defaults to the current version, so a freshly built Report is
        always readable by the code that wrote it.
    root : UPath
        The path the user pointed at — a folder, or a single file.
    score : ScoreResult
        The Dataset-level rollup over `episodes`.
    episodes : list[GradedEpisode]
        Every recording analysed, graded top to bottom.
    findings : list[Finding]
        Every graded metric that came back a defect, flattened across the whole dataset
        and sorted worst first — `findings[0]` is the worst thing this run found,
        if anything was.
    skipped : list[SkippedSource]
        Every file `discovery` declined to carry forward, with its reason.
    unresolved : list[UnresolvedSource]
        Every file that did not make it through to grading, with its evidence —
        `inference` failing to resolve a schema, or a later stage refusing a schema
        it did resolve.
    policy_version : int or None
        The `Policy.schema_version` graded against, so a result stays
        traceable to the policy that produced it. `None` when no policy
        was involved in assembling this report.
    duration_s : float or None
        Wall-clock time the analysis run took, or `None` when not timed.
    source : SourceInfo or None
        What `root` was resolved from.
        `None` when the report was assembled without a resolved source.
    datasets : list[DatasetInfo]
        What the adapter declared about each path it read,
        one entry per path, in walk order.
    categories : dict[str, CategoryGroup]
        Every category slug in the active dictionary mapped onto its group,
        so a reader can group streams by `category` without the dictionary.
    mapping_overrides : list[MappingOverride]
        Every per-run override applied, with where it came from.
        Kalanos records each type as given and does not verify it.

    producer : Producer or None
        The software that wrote this report.
    run : RunInfo or None
        This run's identity: source evidence, configuration identities, tier.
    scope : EvaluationScope or None
        What every decision in this report is relative to.
    inventory : Inventory or None
        Which episodes were expected, loaded and failed.
    eligibility_counts : EligibilityCounts or None
        How the known inventory partitions by eligibility.
    readiness : Readiness or None
        The readiness index, `null` with reasons whenever undefined.
    sufficiency : Sufficiency or None
        Whether the eligible set meets the scope's explicit requirements.
    binding_conflicts : list[BindingConflict]
        Every feature where two mapping sources disagreed, and which won.
    cameras : list[CameraSummary]
        Each camera compared across the episodes it appears in:
        descriptive, and the source of the cross-episode vision findings.

    Letter grades (`score.grade`, `gate.cap`, `gate.uncapped_grade`) are
    compatibility fields only and drive no decision. The headline is
    `eligibility_counts` and `readiness`.
    """

    diagnostics: DiagnosticsReport | None = None
    operational_errors: list[dict[str, str]] = Field(default_factory=list)
    schema_version: str = CURRENT_SCHEMA_VERSION
    coverage: AnalysisCoverage | None = None
    root: AnyPath
    score: ScoreResult
    episodes: list[GradedEpisode] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    skipped: list[SkippedSource] = Field(default_factory=list)
    unresolved: list[UnresolvedSource] = Field(default_factory=list)
    policy_version: int | None = None
    duration_s: float | None = None
    source: SourceInfo | None = None
    datasets: list[DatasetInfo] = Field(default_factory=list)
    categories: dict[str, CategoryGroup] = Field(default_factory=dict)
    mapping_overrides: list[MappingOverride] = Field(default_factory=list)
    gate: Gate | None = None
    readiness: Readiness | None = None
    producer: Producer | None = None
    run: RunInfo | None = None
    scope: EvaluationScope | None = None
    inventory: Inventory | None = None
    eligibility_counts: EligibilityCounts | None = None
    sufficiency: Sufficiency | None = None
    binding_conflicts: list[BindingConflict] = Field(default_factory=list)
    cameras: list[CameraSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def _decisions_agree(self) -> "Report":
        """Refuse a report whose surfaces could disagree about an episode.

        Every episode carries an eligibility; its compatibility `train_ready`
        mirrors it exactly; the gate's failing list is exactly the blocked
        episodes; and the counts partition the episodes plus the failed ones.
        """

        episode_ids = [e.id for e in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("duplicate episode ids")
        if self.coverage is not None:
            from kalanos.analysis.coverage import report_coverage

            if self.inventory is None or any(e.coverage is None for e in self.episodes):
                raise ValueError("coverage requires episode ledgers and inventory")
            if self.coverage != report_coverage(self.episodes, self.inventory):
                raise ValueError("report coverage does not reconcile with episodes")
        if self.diagnostics is not None:
            from kalanos.analysis.models.provenance import content_digest

            if self.diagnostics.plan_digest != content_digest(
                self.diagnostics.plan.model_dump(mode="json")
            ):
                raise ValueError(
                    "diagnostic plan digest disagrees with its configuration"
                )
            recorded = [r for r in self.diagnostics.results if r.episode_id is not None]
            if recorded != [r for e in self.episodes for r in e.diagnostics]:
                raise ValueError("diagnostic report does not reconcile with episodes")
            keys = [(r.episode_id, r.kind, r.id) for r in self.diagnostics.results]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate diagnostic result addresses")
        elif any(e.diagnostics for e in self.episodes):
            raise ValueError(
                "episode diagnostics require an identified diagnostic report"
            )
        ids = [f.id for f in self.findings if f.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate finding ids")
        for episode in self.episodes:
            if episode.eligibility is None:
                raise ValueError(f"episode {episode.id!r} has no eligibility")
            expected = episode.eligibility.compatibility_train_ready
            if episode.score.train_ready != expected:
                raise ValueError(
                    f"episode {episode.id!r}: train_ready={episode.score.train_ready} "
                    f"contradicts eligibility {episode.eligibility.status.value!r}"
                )
        blocked = {
            e.id
            for e in self.episodes
            if e.eligibility is not None
            and e.eligibility.status == EligibilityStatus.BLOCKED
        }
        if self.gate is not None:
            listed = {item.episode_id for item in self.gate.failing_episodes}
            if listed != blocked:
                raise ValueError(
                    "gate.failing_episodes is not the set of blocked episodes"
                )
        if self.eligibility_counts is not None:
            counts = self.eligibility_counts
            failed = len(self.inventory.failed) if self.inventory else 0
            gap = self.inventory.unresolved if self.inventory else 0
            if counts.total != len(self.episodes) + failed + gap:
                raise ValueError(
                    "eligibility_counts.total does not cover the inventory"
                )
            tally = Counter(
                e.eligibility.status.value
                for e in self.episodes
                if e.eligibility is not None
            )
            tally["unknown"] += failed + gap
            for field, status in (
                ("pass_count", "pass"),
                ("blocked", "blocked"),
                ("review", "review"),
                ("unknown", "unknown"),
            ):
                if getattr(counts, field) != tally[status]:
                    raise ValueError(
                        f"eligibility_counts.{field} disagrees with episode "
                        "decisions and failed/unresolved inventory"
                    )
            expected_share = (
                counts.pass_count / counts.total
                if counts.inventory_complete and counts.total
                else None
            )
            if counts.confirmed_eligible_share is not None and (
                expected_share is None
                or not math.isclose(
                    counts.confirmed_eligible_share,
                    expected_share,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("confirmed_eligible_share disagrees with the counts")
            if self.inventory is not None and (
                counts.inventory_complete != self.inventory.complete
            ):
                raise ValueError(
                    "eligibility_counts and inventory disagree on completeness"
                )
            # The dataset compatibility boolean is a function of the counts.
            expected_ready = (
                False
                if counts.blocked
                else True
                if counts.inventory_complete
                and counts.total
                and counts.pass_count == counts.total
                else None
            )
            if self.score.train_ready != expected_ready:
                raise ValueError(
                    f"dataset train_ready={self.score.train_ready} contradicts the "
                    f"eligibility counts (expected {expected_ready})"
                )
        return self
