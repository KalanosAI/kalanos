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
from kalanos.analysis.models.discovery import SkippedSource, SourceInfo
from kalanos.analysis.models.domain import Attribution, Channel, Episode
from kalanos.analysis.models.metrics import MetricResult
from kalanos.analysis.models.paths import AnyPath
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, ScoreResult


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Bumped whenever the graded tree's shape changes: two shapes can carry the same
# field names, so a reader cannot tell them apart by content alone.
CURRENT_SCHEMA_VERSION = "6.0.0"


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
