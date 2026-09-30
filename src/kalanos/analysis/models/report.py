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

# External
from pydantic import BaseModel, ConfigDict, Field

# Internal
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.dictionary import CategoryGroup
from kalanos.analysis.models.discovery import SkippedSource, SourceInfo
from kalanos.analysis.models.domain import Attribution, Channel, Episode
from kalanos.analysis.models.metrics import MetricResult
from kalanos.analysis.models.paths import AnyPath
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Bumped whenever the graded tree's shape changes: two shapes can carry the same
# field names, so a reader cannot tell them apart by content alone.
CURRENT_SCHEMA_VERSION = "6.5.0"


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
    score : ScoreResult
        The Stream-level rollup over the stream's own metrics and `channels`.
    metrics : dict[str, MetricResult]
        Every metric computed on the stream itself keyed by name
        and regraded against the policy, via `scoring.resolve_status`.
    channels : list[GradedChannel]
        The graded channels within this stream.
    """

    taxonomy_type: str
    instance: str | None = None
    attribution: Attribution = Attribution.SINGLE
    category: str | None = None
    score: ScoreResult
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    channels: list[GradedChannel] = Field(default_factory=list)


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


class FailingEpisode(BaseModel):
    """An episode the gate counts as failing, and the critical findings why.

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
    """A critical finding on every episode of one task and on none of any other.

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
    """A critical finding on (nearly) every graded episode of the dataset.

    It describes how the dataset was recorded (a state dimension no episode ever
    moves, say), not a fault in some episodes, so dropping episodes cannot fix it
    and it fails none of them. It is reported so a reader still sees it.

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
    """The dataset gate's verdict: failing episodes cap the dataset's letter.

    `Report.score.score` stays the mean of the episode scores; `Report.score.grade`
    and `train_ready` are this gate's. The mean's own letter is `uncapped_grade`.

    Attributes
    ----------
    n_episodes : int
        The episodes graded.
    failing_episodes : list[FailingEpisode]
        The episodes with a critical finding that is not a task trait.
    failing_share : float
        `len(failing_episodes) / n_episodes`.
    task_traits : list[TaskTrait]
        Critical findings set aside as describing a task rather than a fault.
    dataset_traits : list[DatasetTrait]
        Critical findings on every episode, set aside as describing how the
        dataset was recorded rather than a fault in some of its episodes.
    uncapped_grade : Grade or None
        The letter the mean alone would get.
    cap : Grade or None
        The best letter the failing share allows; `None` when it allows any.
    pruned_score : float or None
        The mean without the failing episodes; `None` when none fail.
    pruned_grade : Grade or None
        `pruned_score`'s letter.
    train_ready_after_pruning : bool or None
        Whether the dataset is train-ready once the failing episodes are removed;
        `None` when none fail.
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
    pruned_grade: Grade | None = None
    train_ready_after_pruning: bool | None = None
    coverage: Coverage
    summary: str


class Readiness(BaseModel):
    """The dataset's headline number: how much of it you can train on, and how clean.

    An episode with a blocking finding (a critical finding that is not a task
    or dataset trait) contributes 0; every other evaluated episode contributes
    its quality score. Readiness is the mean of those contributions over the
    evaluated episodes, equivalently passing share x passing-episode quality.

    Attributes
    ----------
    score : float or None
        Readiness from 0 to 100, full precision; `None` when no episode could
        be evaluated (not graded).
    evaluated_episodes : int
        Episodes with a quality score.
    passing_episodes : int
        Evaluated episodes without a blocking finding.
    blocking_episodes : int
        Evaluated episodes with one.
    passing_quality : float or None
        The mean quality of the passing episodes; `None` when none pass.
    """

    score: float | None
    evaluated_episodes: int
    passing_episodes: int
    blocking_episodes: int
    passing_quality: float | None


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

    Letter grades are deprecated since 0.6.5: `score.grade`, `gate.cap`,
    `gate.uncapped_grade` and `gate.pruned_grade` are still written for
    compatibility and will be removed in 0.7.0. `readiness` is the headline.
    """

    schema_version: str = CURRENT_SCHEMA_VERSION
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
    gate: Gate | None = None
    readiness: Readiness | None = None
