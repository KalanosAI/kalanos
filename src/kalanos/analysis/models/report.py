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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.binding import (
    BindingConflict,
    EvaluationScope,
)
from kalanos.analysis.models.coverage import Availability, CoverageRow
from kalanos.analysis.models.coverage import Coverage as AnalysisCoverage

# Internal
from kalanos.analysis.models.diagnostics import DiagnosticsReport
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
from kalanos.analysis.models.reason_codes import (
    TABLE_KEY,
    decode_reasons,
    encode_reasons,
)
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Bumped whenever the graded tree's shape changes: two shapes can carry the same
# field names, so a reader cannot tell them apart by content alone.
CURRENT_SCHEMA_VERSION = "8.0.0"

# Significant digits a stored near-duplicate distance keeps.
# The near limit, 0.02, is exact at 4 digits,
# so a distance within it never rounds above it.
NRMSE_DIGITS = 4


def round_nrmse(value: float) -> float:
    """`value` rounded to `NRMSE_DIGITS` significant digits.

    Parameters
    ----------
    value : float
        A normalized RMSE.

    Returns
    -------
    float
        The rounded value.
    """

    return float(f"{value:.{NRMSE_DIGITS}g}")


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


class VideoFormat(BaseModel):
    """One statement of a camera's video format.

    A field is `None` where the source states nothing.
    Some fields only one side can state, and stay `None` on the other:
    a dataset declares `channels`, `is_depth_map` and `has_audio`;
    a container states the profile, bitrate, length, colour and tags.
    The container's `bit_rate`, `frame_count` and `duration_s` describe the whole file,
    which in a LeRobot v3.0 dataset holds several episodes.

    Attributes
    ----------
    width : int or None
        The frame width in pixels.
    height : int or None
        The frame height in pixels.
    fps : float or None
        The frame rate.
    pix_fmt : str or None
        The pixel format, as FFmpeg names it (`yuv420p`).
    codec : str or None
        The video codec (`h264`).
    profile : str or None
        The codec profile (`High`). Container only.
    bit_rate : int or None
        The bitrate in bits per second. Container only.
    frame_count : int or None
        How many frames the video file holds. Container only.
    duration_s : float or None
        The video file's duration in seconds. Container only.
    color_range : str or None
        The colour range, as FFmpeg names it (`tv`). Container only.
    color_primaries : str or None
        The colour primaries, as FFmpeg names them (`bt709`). Container only.
    color_transfer : str or None
        The transfer characteristic, as FFmpeg names it (`bt709`). Container only.
    color_space : str or None
        The colour space, as FFmpeg names it (`bt709`). Container only.
    channels : int or None
        How many channels each frame holds. Declared only.
    is_depth_map : bool or None
        Whether the camera records depth rather than colour. Declared only.
    has_audio : bool or None
        Whether the video files carry an audio track. Declared only.
    camera_make : str or None
        The camera maker a container tag names. Container only.
    camera_model : str or None
        The camera model a container tag names. Container only.
    encoder : str or None
        The software that wrote the file, as its `encoder` tag names it. Container only.
    """

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    pix_fmt: str | None = None
    codec: str | None = None
    profile: str | None = None
    bit_rate: int | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    color_range: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_space: str | None = None
    channels: int | None = None
    is_depth_map: bool | None = None
    has_audio: bool | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    encoder: str | None = None


class CameraFormat(BaseModel):
    """A camera's video format as the dataset declares it and as its file states it.

    Both are reported whether or not they agree.

    Attributes
    ----------
    declared : VideoFormat
        What the dataset's own metadata declares; all `None` when it declares nothing.
    found : VideoFormat
        What the video file's container header and tags state
        for its first video stream, covering the whole file.
        All `None` when the container was not opened, as at the metadata tier,
        or could not be read.
    """

    declared: VideoFormat
    found: VideoFormat


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
    video_format : CameraFormat or None
        The camera's declared and container format.
        `None` when the stream is not camera footage backed by a video file.
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
    video_format: CameraFormat | None = None


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
    split : str or None
        The dataset split the source assigns this episode — mirrors
        `domain.Episode.split`. `None` when the source assigns none.
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
    split: str | None = None
    eligibility: EpisodeEligibility | None = None
    coverage: AnalysisCoverage | None = None


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


class Variation(BaseModel):
    """How much a set of episodes differ from each other.

    It is descriptive: it feeds no score, gate or eligibility decision,
    and no threshold marks a value as low.
    A value compares episodes within one dataset, not two datasets.
    The field that holds it says which distance it measures
    and which episodes it covers.

    Attributes
    ----------
    value : float or None
        The root mean square, over every pair of covered episodes, of their distance.
        Rounded to `NRMSE_DIGITS` significant digits.
        `None` when it is unavailable, with the reason in `reason`.
    episodes : int
        The episodes covered.
    reason : str or None
        Why `value` is `None`; `None` when there is a value.
    """

    value: float | None = None
    episodes: int
    reason: str | None = None


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
    repeated_footage : list of list of str
        Each group of episodes whose camera's sampled frames hash identically,
        in episode order.
    visual_variation : Variation or None
        How much the camera's footage changes between episodes:
        the distance is the RMS difference of two episodes' sampled gray frames,
        each average-pooled to 16x16 and scaled to 0-1.
        It covers the largest group of episodes whose read decoded every sampled frame,
        with the same number of frames.
        Motion in view, such as an arm crossing it, counts as change.
        It compares episodes within this camera.
        `None` in a report written before the field existed.
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
    repeated_footage: list[list[str]] = Field(default_factory=list)
    visual_variation: Variation | None = None


class NearPairs(BaseModel):
    """The listed pairs of episodes within the near-duplicate limit.

    A pair is stored once, under its earlier episode,
    and only episodes with a later partner have an entry.
    Episodes are indices into `Report.episodes`.

    Attributes
    ----------
    episode : list[int]
        The earlier episode of each entry's pairs, ascending.
    partners : list[list[int]]
        The later episodes near `episode[k]`, ascending, each greater than `episode[k]`.
    nrmse : list[list[float]]
        The normalized RMSE of each pair, aligned with `partners`,
        to `NRMSE_DIGITS` significant digits.
    """

    episode: list[int] = Field(default_factory=list)
    partners: list[list[int]] = Field(default_factory=list)
    nrmse: list[list[float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _aligned(self) -> "NearPairs":
        """Refuse lists that do not line up, or pairs not stored once in index order.

        Returns
        -------
        NearPairs
            The validated pairs.

        Raises
        ------
        ValueError
            If the lists differ in length, an entry is empty,
            or the indices are not strictly ascending as documented.
        """

        if not len(self.episode) == len(self.partners) == len(self.nrmse):
            raise ValueError("near pairs: episode, partners and nrmse differ in length")
        previous = -1
        for first, partners, distances in zip(
            self.episode, self.partners, self.nrmse, strict=True
        ):
            if not partners or len(partners) != len(distances):
                raise ValueError(
                    f"near pairs of episode {first}: partners and nrmse "
                    "must be non-empty and of equal length"
                )
            if first <= previous:
                raise ValueError("near pairs: episodes are not strictly ascending")
            if any(b <= a for a, b in zip([first, *partners], partners, strict=False)):
                raise ValueError(
                    f"near pairs of episode {first}: partners must be strictly "
                    "ascending and later than the episode"
                )
            previous = first
        return self


class DuplicateEpisodes(BaseModel):
    """The episodes whose numeric payloads are identical or nearly so.

    Episodes are indices into `Report.episodes`.
    Which pairs also repeat their camera footage is read from
    `Report.cameras[].repeated_footage`.

    Attributes
    ----------
    exact : list[list[int]]
        Each group of episodes whose payloads hash identically, earliest first,
        the groups ordered by their earliest episode.
        The earliest episode stands for its group in `near`.
    near : NearPairs
        Every listed near pair, each with its normalized RMSE.
    """

    exact: list[list[int]] = Field(default_factory=list)
    near: NearPairs = Field(default_factory=NearPairs)


class NearDuplicateSummary(BaseModel):
    """How many episode pairs the near-duplicate search found, over the whole dataset.

    Copies that hash identically are compared once, through the earliest of them,
    so the episode and pair counts are over those representatives.

    Attributes
    ----------
    limit : float
        The largest normalized RMSE at which a pair counts as near.
    max_partners : int
        The most near partners each episode keeps in `Report.duplicates`
        when `pairs_listed` is less than `pairs_near`.
        A near finding lists no partner; its pairs are in `Report.duplicates.near`.
    episodes_compared : int
        The representatives the near search compared.
    pairs_compared : int
        The pairs among them, summed over the groups of episodes
        that hold the same streams and channels.
    pairs_near : int
        Of those, the pairs within `limit` whose row counts also agree.
    pairs_listed : int
        The near pairs in `Report.duplicates.near`, each stored once:
        every one of `pairs_near`,
        or, when there are too many to list, those among the nearest
        `max_partners` of either episode.
    episodes_with_partner : int
        The episodes, copies included, with at least one near partner.
    median_nearest_nrmse : float or None
        The median, over representatives with a near partner,
        of the distance to their nearest one.
        `None` when `pairs_near` is 0.
    saturated : bool
        Whether more than half the compared pairs are near,
        and there are more near pairs than compared episodes.
        The limit then does not separate copies from re-runs in this dataset,
        so no near-duplicate finding is emitted;
        the exact tier still applies, and `Report.duplicates` keeps the nearest pairs.
    """

    limit: float
    max_partners: int
    episodes_compared: int
    pairs_compared: int
    pairs_near: int
    pairs_listed: int
    episodes_with_partner: int
    median_nearest_nrmse: float | None
    saturated: bool


class DatasetWarning(BaseModel):
    """A warning about the dataset as a whole, which belongs to no episode.

    It is descriptive: it feeds no score, gate or eligibility decision.

    Attributes
    ----------
    code : {"near_duplicates_saturated"}
        What the warning is about.
        `near_duplicates_saturated`: the near tier is saturated
        (see `NearDuplicateSummary.saturated`),
        so no episode carries a near-duplicate finding;
        the counts are in `Report.near_duplicates`.
    message : str
        The warning, stated for a reader.
    """

    code: Literal["near_duplicates_saturated"]
    message: str


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
    family_weights : dict[str, float]
        The policy's `family_weights` the run graded with, keyed by family id.
        Empty when the policy weights no family.
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
    duplicates : DuplicateEpisodes
        The episodes whose numeric payloads are identical or nearly so,
        as indices into `episodes`.
        Identical episodes form one group, whose first episode
        also stands for them in a near pair.
        Every near pair is listed, unless there are too many:
        `near_duplicates.pairs_listed` then falls short of `pairs_near`.
        A report written before this encoding listed one object per pair,
        and is converted on load.
    near_duplicates : NearDuplicateSummary or None
        How many pairs the near-duplicate search found, and whether that many
        leaves the near tier uninformative for this dataset.
        `None` when the search did not run: numpy is missing,
        or no group of comparable episodes holds two
        that differ in their payloads and could be resampled
        (every value finite, at least 2 rows).
    motion_variation : Variation or None
        How much the episodes' trajectories differ from each other,
        from the same resampled trajectories the near-duplicate search compares.
        The distance is their normalized RMSE, the one the near-duplicate tier measures.
        The channels are scaled by their spread over the covered episodes.
        It covers, exact copies included, the largest group of episodes
        that hold the same numeric streams and channels and could be resampled
        (every value finite, at least 2 rows).
        Episodes whose every channel is constant take no part.
        `episodes` is 0 when numpy is missing.
        `None` in a report written before the field existed.
    dataset_warnings : list[DatasetWarning]
        Warnings about the dataset as a whole, which belong to no episode.
        Descriptive: they have no effect on any score.

    Letter grades (`score.grade`, `gate.cap`, `gate.uncapped_grade`) are
    compatibility fields only and drive no decision. The headline is
    `eligibility_counts` and `readiness`.

    A JSON-mode dump stores each reason text once, in a first key `reason_codes`,
    and every `reason`, `reasons` and `detail` value holds its index there
    (see `reason_codes`). Validation reads the codes back as text.
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
    family_weights: dict[str, float] = Field(default_factory=dict)
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
    duplicates: DuplicateEpisodes = Field(default_factory=DuplicateEpisodes)
    near_duplicates: NearDuplicateSummary | None = None
    motion_variation: Variation | None = None
    dataset_warnings: list[DatasetWarning] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _duplicates_from_pairs(cls, data: Any) -> Any:
        """Convert the one-object-per-pair `duplicates` of reports 7.0.0 to 7.3.0.

        Those reports named both episodes of each pair by id.
        Each pair becomes indices into `episodes`: an exact pair joins
        the group of its first episode, and a near pair is stored under
        its earlier episode with its distance rounded as a fresh report's is.
        The footage fields are dropped, since `cameras` holds the same facts.

        Parameters
        ----------
        data : Any
            The raw input to the model.

        Returns
        -------
        Any
            `data`, with `duplicates` converted when it was a dict holding a list there.

        Raises
        ------
        ValueError
            If a pair is malformed or names an episode that is not in `episodes`.
        """

        if not isinstance(data, dict) or not isinstance(data.get("duplicates"), list):
            return data
        index = {
            episode["id"]: k
            for k, episode in enumerate(data.get("episodes") or [])
            if isinstance(episode, dict) and "id" in episode
        }
        exact: dict[int, list[int]] = {}
        near: dict[int, list[tuple[int, float]]] = {}
        for pair in data["duplicates"]:
            try:
                names, tier = pair["episodes"], pair["tier"]
                nrmse = pair["nrmse"] if tier == "near" else 0.0
            except (KeyError, TypeError):
                raise ValueError(f"duplicates: malformed pair {pair!r}") from None
            if tier not in ("exact", "near") or len(names) != 2:
                raise ValueError(f"duplicates: malformed pair {pair!r}")
            unknown = [name for name in names if name not in index]
            if unknown:
                raise ValueError(
                    f"duplicates: pair names episode {unknown[0]!r}, "
                    "which is not in the report"
                )
            a, b = index[names[0]], index[names[1]]
            if tier == "exact":
                exact.setdefault(a, [a]).append(b)
            else:
                lo, hi = min(a, b), max(a, b)
                near.setdefault(lo, []).append((hi, round_nrmse(nrmse)))
        rows = [(lo, sorted(near[lo])) for lo in sorted(near)]
        return {
            **data,
            "duplicates": {
                "exact": [sorted(exact[first]) for first in sorted(exact)],
                "near": {
                    "episode": [lo for lo, _found in rows],
                    "partners": [[hi for hi, _d in found] for _lo, found in rows],
                    "nrmse": [[d for _hi, d in found] for _lo, found in rows],
                },
            },
        }

    @model_serializer(mode="wrap")
    def _write_reason_codes(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        """Code every reason text when dumping to JSON.

        Parameters
        ----------
        handler : SerializerFunctionWrapHandler
            Pydantic's own serializer for this model.
        info : SerializationInfo
            The dump's settings; only JSON mode is coded,
            and not when the context holds `raw_reasons`.

        Returns
        -------
        dict[str, Any]
            The dumped report, led by `reason_codes` in JSON mode,
            otherwise as Pydantic dumps it.
        """

        data = handler(self)
        # The streaming writer codes the header and the episodes with one table,
        # so it needs the header's texts uncoded.
        raw = isinstance(info.context, dict) and info.context.get("raw_reasons")
        return encode_reasons(data) if info.mode_is_json() and not raw else data

    @model_validator(mode="wrap")
    @classmethod
    def _read_reason_codes(
        cls, data: Any, handler: ModelWrapValidatorHandler["Report"]
    ) -> "Report":
        """Turn reason codes back into text before validating.

        Parameters
        ----------
        data : Any
            The input, coded when it is a dict holding `reason_codes`.
        handler : ModelWrapValidatorHandler[Report]
            Pydantic's own validator for this model.

        Returns
        -------
        Report
            The validated report, holding reason texts.
        """

        if isinstance(data, dict) and TABLE_KEY in data:
            return handler(decode_reasons(data))
        return handler(data)

    @model_validator(mode="after")
    def _decisions_agree(self) -> "Report":
        """Refuse a report whose surfaces could disagree about an episode.

        Every episode carries an eligibility; its compatibility `train_ready`
        mirrors it exactly; the gate's failing list is exactly the blocked
        episodes; the counts partition the episodes plus the failed ones;
        and every index in `duplicates` names an episode.
        """

        episode_ids = [e.id for e in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("duplicate episode ids")
        near = self.duplicates.near
        indices = [
            *(k for group in self.duplicates.exact for k in group),
            *near.episode,
            *(k for partners in near.partners for k in partners),
        ]
        if any(not 0 <= k < len(episode_ids) for k in indices):
            raise ValueError("duplicates names an episode index outside episodes")
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
            if any(
                r.episode_id is not None and r.episode_id not in episode_ids
                for r in self.diagnostics.results
            ):
                raise ValueError("diagnostic result names an episode not in the report")
            keys = [(r.episode_id, r.kind, r.id) for r in self.diagnostics.results]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate diagnostic result addresses")
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
