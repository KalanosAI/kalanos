"""Walk a mapped Episode and grade it into a Report.

Nothing here computes a metric or a score itself —
that is `metrics` and `scoring`'s job.
This module walks the Channel → Stream → Episode → Dataset tree
and hands each level exactly what the stage below expects.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import zlib
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast, overload

# External
from pydantic_core import to_json
from upath import UPath

# Internal
from kalanos.analysis.calibration import apply_calibration
from kalanos.analysis.coverage import (
    episode_coverage,
    report_coverage,
    rows_that_ran,
    stream_rows,
)
from kalanos.analysis.identities import adapter_versions, detector_versions
from kalanos.analysis.metrics.registry import (
    run_channel_metrics,
    run_episode_metrics,
    run_stream_metrics,
)
from kalanos.analysis.metrics.vision import (
    camera_format,
    camera_frames,
    footage_vector,
    is_camera_footage,
)
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.binding import (
    SAMPLED_VIDEO_QUALITY_CAPABILITY,
    VIDEO_QUALITY_CAPABILITY,
    BindingConflict,
    EvaluationScope,
    RequirementsSection,
)
from kalanos.analysis.models.coverage import Coverage
from kalanos.analysis.models.diagnostics import DiagnosticResult, VisionSpec
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.discovery import SkippedSource, SourceInfo
from kalanos.analysis.models.domain import UNMAPPED_TAXONOMY_PREFIX, Episode, Stream
from kalanos.analysis.models.eligibility import EligibilityStatus
from kalanos.analysis.models.mapping import MappingOverride
from kalanos.analysis.models.metrics import (
    DEFAULT_VISION_SAMPLES,
    ChannelContext,
    EpisodeContext,
    Family,
    Level,
    StreamContext,
)
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.provenance import (
    ExecutionTier,
    Inventory,
    Producer,
    RunCompletion,
    RunInfo,
)
from kalanos.analysis.models.report import (
    AgreementRow,
    AnalysedEpisode,
    GradedChannel,
    GradedEpisode,
    GradedStream,
    PayloadStatus,
    Report,
    StreamEvaluation,
    check_agreement,
)
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, FindingLocation
from kalanos.analysis.models.summaries import EpisodeSummary
from kalanos.analysis.reporting.card.sections import CardRow
from kalanos.analysis.reporting.stream import StreamedReport
from kalanos.analysis.scoring.cameras import (
    FootageVariation,
    camera_key,
    camera_samples,
    compare_cameras,
)
from kalanos.analysis.scoring.duplicates import (
    PayloadFingerprint,
    StreamKey,
    find_duplicate_episodes,
    near_duplicate_warning,
    payload_fingerprint,
)
from kalanos.analysis.scoring.eligibility import (
    counts_of,
    dataset_train_ready,
    decide_all,
    readiness_of,
    requirement_reasons,
    sufficiency_of,
)
from kalanos.analysis.scoring.gate import apply_gate, gate_counts
from kalanos.analysis.scoring.score import (
    dataset_family_tails,
    dataset_metric_rollups,
    metric_subjects,
    rollup,
    score_episode,
    score_metrics,
    sort_findings,
)
from kalanos.assets.dictionary import load_default_dictionary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)

# Episode coverage ledgers held before they are merged into one.
# A ledger outweighs the episode's compressed JSON, so a run keeps few of them.
_LEDGERS_HELD = 64


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def grade_stream(
    stream: Stream,
    *,
    policy: Policy,
    is_regular: bool,
    episode_id: str,
    category: str | None,
    vision_samples: int = DEFAULT_VISION_SAMPLES,
    episode_streams: Sequence[Stream] | None = None,
    full_frame_scan: bool = False,
    vision: VisionSpec | None = None,
    tier: ExecutionTier = ExecutionTier.STANDARD,
    fingerprints: dict[StreamKey, PayloadFingerprint] | None = None,
    footage: FootageVariation | None = None,
) -> tuple[GradedStream, list[Finding]]:
    """Grade a Stream's own metrics and every channel within it, then roll both up.

    Parameters
    ----------
    stream : Stream
        The stream to grade, carrying its own timestamps and payload.
    policy : Policy
        The loaded grading policy.
    is_regular : bool
        Whether the sampling behind this stream classified as regular.
    episode_id : str
        The recording this stream belongs to,
        for addressing any finding it or its channels raise.
    category : str or None
        The dictionary category of the stream's taxonomy type, `None` when unmapped.
    vision_samples : int
        How many frames blur and exposure sample, and windows frozen frames read.
    episode_streams : Sequence[Stream] or None
        Every stream of the episode `stream` belongs to, itself included,
        for a metric that reads one stream against another.
    full_frame_scan : bool
        Whether frame metrics read every frame rather than a sample.
    vision : VisionSpec or None
        Decode caps, previews and exposure levels; the defaults when `None`.
    fingerprints : dict or None
        When given, receives the stream's payload fingerprint under its `StreamKey`,
        for the duplicate-episode pass; the key's last element counts
        the episode's earlier streams at the same place.
    footage : FootageVariation or None
        When given and the stream is camera footage,
        receives its pooled sampled frames, for the camera's visual variation.

    Returns
    -------
    tuple[GradedStream, list[Finding]]
        The stream's own metrics and every channel, each graded; a
        Stream-level score rolled up over both; and every finding raised
        anywhere in the stream, unsorted — only `ReportBuilder.finish` sorts,
        once every episode has contributed.
    """

    findings: list[Finding] = []
    location = FindingLocation(
        episode_id=episode_id,
        source_path=str(stream.source_path),
        source_field=stream.source_field,
        stream=stream.taxonomy_type,
        instance=stream.instance,
    )

    # Step 1: grade the stream's own metrics — the ones that read its clock
    # or its payload as a whole. These run whether or not the stream carries a payload.
    stream_ctx = StreamContext(
        stream=stream.model_copy(update={"payload": None})
        if tier == ExecutionTier.METADATA
        else stream,
        is_regular=is_regular,
        vision_samples=vision_samples,
        episode_streams=list(episode_streams) if episode_streams is not None else None,
        full_frame_scan=full_frame_scan,
        reads_payloads=tier != ExecutionTier.METADATA,
        vision=vision or VisionSpec(),
    )
    stream_results = run_stream_metrics(stream_ctx)
    stream_metrics, own_score, stream_findings = score_metrics(
        stream_results,
        level=Level.STREAM,
        taxonomy_type=stream.taxonomy_type,
        policy=policy,
        location=location,
    )
    findings.extend(stream_findings)

    # Step 2: fetch the payload only when there is a channel to grade with it,
    # and only when the execution tier reads payloads at all. Whatever
    # happens is recorded on the stream so a requirement can see it; a
    # warning in a log is not evidence.
    graded_channels = []
    evaluation = StreamEvaluation(payload=PayloadStatus.NOT_REQUIRED)
    if stream.channels:
        if tier == ExecutionTier.METADATA:
            evaluation = StreamEvaluation(
                payload=PayloadStatus.SKIPPED,
                reason="metadata tier does not read numeric payloads",
                n_channels_declared=len(stream.channels),
            )
        elif stream.payload is None:
            logger.warning(
                "%s: stream %r has %d channel(s) but no payload",
                stream.source_path,
                stream.taxonomy_type,
                len(stream.channels),
            )
            evaluation = StreamEvaluation(
                payload=PayloadStatus.MISSING_INPUT,
                reason=f"{len(stream.channels)} channel(s) declared but no payload",
                n_channels_declared=len(stream.channels),
            )
        else:
            try:
                frame = stream.payload.fetch()
                if frame.height != len(stream.timestamps):
                    raise ValueError("payload rows do not align with timestamps")
                absent = [
                    c.name for c in stream.channels if c.name not in frame.columns
                ]
                if absent:
                    raise ValueError(f"payload is missing declared channels: {absent}")
            except Exception as exc:
                evaluation = StreamEvaluation(
                    payload=PayloadStatus.ERROR,
                    reason=f"{type(exc).__name__}: {exc}",
                    n_channels_declared=len(stream.channels),
                )
            else:
                evaluation = StreamEvaluation(
                    payload=PayloadStatus.COMPUTED,
                    n_channels_declared=len(stream.channels),
                    n_channels_graded=len(stream.channels),
                )
                if fingerprints is not None:
                    fingerprint = payload_fingerprint(
                        frame, [channel.name for channel in stream.channels]
                    )
                    if fingerprint is not None:
                        place = (
                            stream.taxonomy_type,
                            stream.instance,
                            stream.source_field,
                        )
                        seen = sum(1 for key in fingerprints if key[:3] == place)
                        fingerprints[(*place, seen)] = fingerprint
                for channel in stream.channels:
                    ctx = ChannelContext(
                        channel=channel, values=frame[channel.name], stream=stream_ctx
                    )
                    results = run_channel_metrics(ctx)
                    graded, score, channel_findings = score_metrics(
                        results,
                        level=Level.CHANNEL,
                        taxonomy_type=ctx.taxonomy_type,
                        policy=policy,
                        location=location.model_copy(
                            update={
                                "channel": channel.name,
                                "stream": ctx.taxonomy_type,
                                "source_index": channel.source_index,
                            }
                        ),
                    )
                    graded_channels.append(
                        GradedChannel(channel=channel, score=score, metrics=graded)
                    )
                    findings.extend(channel_findings)

    # Step 3: fold the stream's own score in alongside its channels',
    # as one more equal-weight contributor. The stream score is for display;
    # the episode score reads the metrics themselves.
    stream_score = rollup(
        Level.STREAM, [*(gc.score for gc in graded_channels), own_score], policy=policy
    )
    is_camera = is_camera_footage(stream.kind.value, stream.taxonomy_type)
    graded_stream = GradedStream(
        taxonomy_type=stream.taxonomy_type,
        kind=stream.kind.value,
        source_path=str(stream.source_path),
        coverage=stream_rows(stream, graded_channels, stream_metrics, evaluation),
        instance=stream.instance,
        attribution=stream.attribution,
        category=category,
        mapping_source=stream.mapping_source,
        score=stream_score,
        metrics=stream_metrics,
        channels=graded_channels,
        source_field=stream.source_field,
        source_identity=stream.source_identity,
        declared_channels=stream.channels if not graded_channels else [],
        clock=stream.clock,
        clock_info=stream.clock_info,
        source_order=stream.source_order,
        evaluation=evaluation,
        frames=camera_frames(stream_ctx) if is_camera else None,
        video_format=camera_format(
            stream, reads_payloads=tier != ExecutionTier.METADATA
        )
        if is_camera
        else None,
    )
    # After `camera_frames`, so the vector comes from the read it cached;
    # a read it could not publish is not tried again.
    if footage is not None and is_camera:
        read = graded_stream.frames is not None
        vector = footage_vector(stream_ctx) if read else None
        footage.add(episode_id, camera_key(graded_stream), vector)
    return graded_stream, findings


def grade_episode(
    episode: Episode,
    *,
    adapter: str,
    adapter_confidence: float,
    policy: Policy,
    dictionary: Dictionary,
    vision_samples: int = DEFAULT_VISION_SAMPLES,
    full_frame_scan: bool = False,
    vision: VisionSpec | None = None,
    tier: ExecutionTier = ExecutionTier.STANDARD,
    fingerprints: dict[str, dict[StreamKey, PayloadFingerprint]] | None = None,
    footage: FootageVariation | None = None,
) -> tuple[GradedEpisode, list[Finding]]:
    """Grade every stream and channel in one Episode, and roll it up.

    Parameters
    ----------
    episode : Episode
        The recording an adapter produced.
    adapter : str
        The adapter that read this episode, recorded on the result.
    adapter_confidence : float
        The adapter's winning bid, recorded on the result.
    policy : Policy
        The loaded grading policy.
    dictionary : Dictionary
        The dictionary each stream's category is looked up in.
    vision_samples : int
        How many frames blur and exposure sample, and windows frozen frames read.
    full_frame_scan : bool
        Whether frame metrics read every frame rather than a sample.
    vision : VisionSpec or None
        Decode caps, previews and exposure levels; the defaults when `None`.
    fingerprints : dict or None
        When given, receives the episode's stream fingerprints under its id,
        for the duplicate-episode pass.
    footage : FootageVariation or None
        When given, receives the pooled sampled frames of each camera stream,
        for the cameras' visual variation.

    Returns
    -------
    tuple[GradedEpisode, list[Finding]]
        Every stream and channel in `episode`, graded, alongside the
        episode's own metrics; a score folded from every metric graded in it;
        the recording's duration and sample count;
        and every finding raised anywhere in it, unsorted —
        only `ReportBuilder.finish` sorts the dataset-wide list.
    """

    # Step 1: grade every stream, and the channels within each.
    graded_streams: list[GradedStream] = []
    findings: list[Finding] = []
    stream_prints: dict[StreamKey, PayloadFingerprint] | None = None
    if fingerprints is not None:
        stream_prints = fingerprints.setdefault(episode.id, {})
    for stream in episode.streams:
        graded_stream, stream_findings = grade_stream(
            stream,
            policy=policy,
            is_regular=stream.is_regular,
            episode_id=episode.id,
            category=dictionary.category_of(stream.taxonomy_type),
            vision_samples=vision_samples,
            episode_streams=episode.streams,
            full_frame_scan=full_frame_scan,
            vision=vision,
            tier=tier,
            fingerprints=stream_prints,
            footage=footage,
        )
        graded_streams.append(graded_stream)
        findings.extend(stream_findings)

    # Step 2: grade the episode's own metrics — the ones that compare streams within it,
    # rather than reading any one of them alone.
    episode_ctx = EpisodeContext(episode=episode)
    episode_results = run_episode_metrics(episode_ctx)
    episode_metrics, _own_score, episode_findings = score_metrics(
        results=episode_results,
        level=Level.EPISODE,
        taxonomy_type=Level.EPISODE.value,
        policy=policy,
        location=FindingLocation(episode_id=episode.id),
    )
    findings.extend(episode_findings)

    # Step 3: score the episode from every metric it graded,
    # each counted once at its worst channel, stream or camera.
    episode_score = score_episode(graded_streams, episode_metrics, policy=policy)

    # Step 4: the recording's extent, from the timestamps its streams already carry.
    timed = [s.timestamps for s in episode.streams if s.timestamps.len()]
    starts = [cast(float, timestamps.min()) for timestamps in timed]
    ends = [cast(float, timestamps.max()) for timestamps in timed]
    duration_s = float(max(ends) - min(starts)) if starts else None
    sample_count = max((s.timestamps.len() for s in episode.streams), default=0)

    return (
        GradedEpisode(
            id=episode.id,
            adapter=adapter,
            adapter_confidence=adapter_confidence,
            source_paths=episode.source_paths,
            duration_s=duration_s,
            sample_count=sample_count,
            score=episode_score,
            metrics=episode_metrics,
            streams=graded_streams,
            tasks=episode.tasks,
            split=episode.split,
        ),
        findings,
    )


def grades_vision(requirements: RequirementsSection) -> bool:
    """Whether the scope requires a video capability, so the vision family grades."""

    required = requirements.required_capabilities
    return (
        VIDEO_QUALITY_CAPABILITY in required
        or SAMPLED_VIDEO_QUALITY_CAPABILITY in required
    )


def scope_policy(policy: Policy, requirements: RequirementsSection) -> Policy:
    """Make the vision metrics report-only unless the scope requires a video capability.

    A scope that asks for neither sampled nor full video quality,
    numeric-core among them, must not let a camera decide or score an episode,
    so its vision results are measured and shown but never graded.
    """

    if grades_vision(requirements):
        return policy
    prefix = f"{Family.VISION.value}."
    metrics = {
        key: entry.model_copy(update={"report_only": True})
        if key.startswith(prefix)
        else entry
        for key, entry in policy.metrics.items()
    }
    return policy.model_copy(update={"metrics": metrics})


def with_coverage(
    episode: GradedEpisode,
    requirements: RequirementsSection,
    diagnostic_results: Sequence[DiagnosticResult] = (),
) -> GradedEpisode:
    """Record an episode's coverage ledger and trim its streams' rows to those that ran.

    Parameters
    ----------
    episode : GradedEpisode
        A graded episode.
    requirements : RequirementsSection
        What a pass needs under this scope.
    diagnostic_results : Sequence[DiagnosticResult]
        The episode's diagnostic results, when a diagnostics plan ran.

    Returns
    -------
    GradedEpisode
        A copy carrying its ledger, its streams' coverage trimmed.
    """

    return episode.model_copy(
        update={
            "coverage": episode_coverage(episode, requirements, diagnostic_results),
            # Trim after episode_coverage, which merges the full rows.
            "streams": [
                s.model_copy(update={"coverage": rows_that_ran(s.coverage)})
                for s in episode.streams
            ],
        }
    )


def summarize(
    episode: GradedEpisode, requirements: RequirementsSection
) -> EpisodeSummary:
    """Take what every dataset-level step reads of one graded episode.

    Parameters
    ----------
    episode : GradedEpisode
        A graded episode carrying its coverage ledger.
    requirements : RequirementsSection
        What a pass needs under this scope.

    Returns
    -------
    EpisodeSummary
        The episode's summary.
    """

    errors = [
        {"episode_id": episode.id, "metric": row.key, "reason": "; ".join(row.reasons)}
        for row in (episode.coverage.metrics if episode.coverage else [])
        if row.error
    ]
    return EpisodeSummary(
        id=episode.id,
        source_paths=episode.source_paths,
        first_task=next((t for t in (episode.tasks or []) if t.strip()), None),
        score=episode.score,
        requirement_reasons=requirement_reasons(episode, requirements),
        camera_samples=camera_samples(episode),
        gate_counts=gate_counts(episode),
        metric_subjects=metric_subjects(episode),
        operational_errors=errors,
        unmapped_types=frozenset(
            stream.taxonomy_type
            for stream in episode.streams
            if stream.taxonomy_type.startswith(f"{UNMAPPED_TAXONOMY_PREFIX}.")
        ),
    )


def _merged(ledgers: list[Coverage | None]) -> list[Coverage | None]:
    """Merge coverage ledgers into one, which merges with others as they would.

    `report_coverage` sums rows by key and frame counts,
    so merging a merge gives what merging every ledger at once gives.
    The inventory's own fields are set again by the final merge.

    Parameters
    ----------
    ledgers : list[Coverage or None]
        Episode ledgers, or merges of them.

    Returns
    -------
    list[Coverage or None]
        One merged ledger.
    """

    return [report_coverage(ledgers, Inventory())]


def _blob(episode: GradedEpisode) -> bytes:
    """Compress one graded episode's JSON, to be read back as the same model.

    Parameters
    ----------
    episode : GradedEpisode
        A graded episode.

    Returns
    -------
    bytes
        Its JSON-mode dump, zlib-compressed at level 1.
    """

    # `model_dump_json` writes NaN as null, which reads back as None,
    # and `omit_empty_defaults` then leaves the field out where NaN was written.
    # Bare NaN and Infinity tokens read back as the floats they were.
    data = to_json(episode.model_dump(mode="json"), inf_nan_mode="constants")
    return zlib.compress(data, 1)


class ReportBuilder:
    """Grade episodes one at a time, then assemble the run's Report from them.

    `add` grades one episode and keeps only what the dataset-level steps read of it,
    so a caller that hands over each recording as it is read
    can let go of it before reading the next.
    `finish` runs the dataset-level steps once, after the last `add`.
    """

    def __init__(
        self,
        *,
        policy: Policy,
        requirements: RequirementsSection | None = None,
        tier: ExecutionTier = ExecutionTier.STANDARD,
        vision_samples: int = DEFAULT_VISION_SAMPLES,
        full_frame_scan: bool = False,
        vision: VisionSpec | None = None,
        diagnostics_plan=None,
        keep_episodes: bool = True,
    ) -> None:
        """Start an empty run.

        Parameters
        ----------
        policy : Policy
            The loaded grading policy.
            Distinct from each `AnalysedEpisode`'s own `policy`:
            this is what the dataset-level rollup and `Report.policy_version`
            grade against.
        requirements : RequirementsSection or None
            What a pass needs. `None` uses the built-in `numeric-core` scope.
        tier : ExecutionTier
            The execution tier the run grades at.
        vision_samples : int
            How many frames blur and exposure sample, and windows frozen frames read.
        full_frame_scan : bool
            Whether frame metrics read every frame rather than a sample.
        vision : VisionSpec or None
            Decode caps, previews and exposure levels; the defaults when `None`.
        diagnostics_plan : DiagnosticPlan or None
            The diagnostics to run. Its diagnostics read across episodes,
            so with a plan every added episode is kept until `finish`.
        keep_episodes : bool
            Make `finish` return a `Report` holding every graded episode.
            Unset, each graded episode is held only as compressed JSON,
            and `finish` returns a `StreamedReport`, unless a diagnostics plan is set.
        """

        self._dictionary = load_default_dictionary()
        self._tier = tier
        self._requirements = requirements or RequirementsSection()
        self._policy = scope_policy(policy, self._requirements)
        self._vision_samples = vision_samples
        self._full_frame_scan = full_frame_scan
        self._vision = vision
        self._diagnostics_plan = diagnostics_plan
        self._keep_episodes = keep_episodes or diagnostics_plan is not None
        self._policies: dict[str, Policy] = {}
        # A dict keeps the order each adapter was first seen in.
        self._adapters: dict[str, None] = {}
        self._analysed: list[AnalysedEpisode] = []
        self._graded_episodes: list[GradedEpisode] = []
        self._blobs: list[bytes] = []
        self._summaries: list[EpisodeSummary] = []
        self._ledgers: list[Coverage | None] = []
        self._findings: list[Finding] = []
        self._fingerprints: dict[str, dict[StreamKey, PayloadFingerprint]] = {}
        self._footage = FootageVariation()

    def __len__(self) -> int:
        """Count the episodes added so far."""

        return len(self._graded_episodes) + len(self._blobs)

    def add(self, item: AnalysedEpisode) -> None:
        """Grade one analysed episode and keep what the dataset-level steps read of it.

        Without a diagnostics plan its coverage is final as soon as it is graded,
        so it is summarized then;
        without `keep_episodes`, it is then held only as compressed JSON.

        Parameters
        ----------
        item : AnalysedEpisode
            The canonical episode, the adapter that read it,
            and the policy to grade it against.

        Raises
        ------
        MappingOverrideError
            If the episode's policy reviews a diagnostic the plan does not configure.
        """

        from kalanos.analysis.diagnostics.runner import validate_review_plan

        validate_review_plan(self._diagnostics_plan, item.policy)
        self._policies[item.episode.id] = item.policy
        self._adapters.setdefault(item.adapter)
        if self._diagnostics_plan is not None:
            self._analysed.append(item)
        graded_episode, episode_findings = grade_episode(
            item.episode,
            adapter=item.adapter,
            adapter_confidence=item.adapter_confidence,
            policy=scope_policy(item.policy, self._requirements),
            dictionary=self._dictionary,
            vision_samples=self._vision_samples,
            full_frame_scan=self._full_frame_scan,
            vision=self._vision,
            tier=self._tier,
            fingerprints=self._fingerprints,
            footage=self._footage,
        )
        self._findings.extend(episode_findings)
        if self._diagnostics_plan is None:
            graded_episode = with_coverage(graded_episode, self._requirements)
            self._summaries.append(summarize(graded_episode, self._requirements))
            self._ledgers.append(graded_episode.coverage)
            if len(self._ledgers) >= _LEDGERS_HELD:
                self._ledgers = _merged(self._ledgers)
        if self._keep_episodes:
            self._graded_episodes.append(graded_episode)
        else:
            self._blobs.append(_blob(graded_episode))

    def finish(
        self,
        *,
        root: UPath,
        skipped: Sequence[SkippedSource] = (),
        unresolved: Sequence[UnresolvedSource] = (),
        duration_s: float | None = None,
        source: SourceInfo | None = None,
        datasets: Sequence[DatasetInfo] = (),
        mapping_overrides: Sequence[MappingOverride] = (),
        scope: EvaluationScope | None = None,
        producer: Producer | None = None,
        run: RunInfo | None = None,
        inventory: Inventory | None = None,
        binding_conflicts: Sequence[BindingConflict] = (),
    ) -> Report | StreamedReport:
        """Run the dataset-level steps over every added episode and assemble the Report.

        Called once, after the last `add`.

        Parameters
        ----------
        root : UPath
            The path the user pointed at.
        skipped : Sequence[SkippedSource]
            Files discovery declined to carry forward, or that no adapter claimed.
        unresolved : Sequence[UnresolvedSource]
            Files an adapter bid on and then could not read.
        duration_s : float or None
            Wall-clock time the run took, if the caller is timing it.
        source : SourceInfo or None
            What `root` was resolved from, if the caller resolved it.
        datasets : Sequence[DatasetInfo]
            What the adapter declared about each path it read, in walk order.
        mapping_overrides : Sequence[MappingOverride]
            The per-run overrides the pipeline applied.
        scope : EvaluationScope or None
            The scope identities to name on the report.
        producer, run : Producer, RunInfo or None
            Provenance, when the caller resolved it.
        inventory : Inventory or None
            Expected/loaded/failed episodes. `None` means every added episode
            is the whole inventory.
        binding_conflicts : Sequence[BindingConflict]
            Mapping disagreements the resolver recorded.

        Returns
        -------
        Report or StreamedReport
            Every added episode, graded and rolled up to a Dataset-level score,
            alongside every skipped and unresolved file with its reason or evidence.
        """

        dictionary = self._dictionary
        tier = self._tier
        requirements = self._requirements
        policy = self._policy
        diagnostics_plan = self._diagnostics_plan
        analysed = self._analysed
        policies = self._policies
        graded_episodes = self._graded_episodes
        blobs = self._blobs
        summaries = self._summaries
        ledgers = self._ledgers
        findings = self._findings
        fingerprints = self._fingerprints
        footage = self._footage
        adapters = adapter_versions(self._adapters)

        # Step 2: decide eligibility once, now that every metric has run. Every
        # count, the gate and the compatibility booleans derive from this.
        scope = scope or EvaluationScope(
            requirements_id=requirements.id, policy_id="default-decisions-v1"
        )
        n_episodes = len(self)
        inventory = inventory or Inventory(loaded=n_episodes)
        producer = producer or Producer(version="unknown")
        producer = producer.model_copy(
            update={"metrics": detector_versions(), "adapters": adapters}
        )
        findings = apply_calibration(findings, policies, run, scope, producer)
        diagnostic_results = {}
        diagnostics = None
        if diagnostics_plan is not None:
            from kalanos.analysis.diagnostics.runner import (
                episode_diagnostics,
                review_findings,
            )

            diagnostic_results = episode_diagnostics(
                [a.episode for a in analysed],
                diagnostics_plan,
                tier,
                findings,
                policies,
                {e.id: e for e in graded_episodes},
            )
            findings.extend(
                review_findings([r for rs in diagnostic_results.values() for r in rs])
            )
            graded_episodes = [
                with_coverage(e, requirements, diagnostic_results.get(e.id, []))
                for e in graded_episodes
            ]
            summaries = [summarize(e, requirements) for e in graded_episodes]
            ledgers = [e.coverage for e in graded_episodes]
        cameras, camera_findings = compare_cameras(
            (sample for summary in summaries for sample in summary.camera_samples),
            grades_vision=grades_vision(requirements),
            footage=footage,
        )
        findings.extend(camera_findings)
        duplicates, duplicate_findings, near_duplicates, motion_variation = (
            find_duplicate_episodes([s.id for s in summaries], fingerprints)
        )
        findings.extend(duplicate_findings)
        dataset_warnings = [
            w
            for w in (near_duplicate_warning(near_duplicates, n_episodes),)
            if w is not None
        ]
        decisions = decide_all(
            summaries, findings, requirements=requirements, policy_id=scope.policy_id
        )
        graded_episodes = [
            episode.model_copy(
                update={
                    "eligibility": decisions[episode.id],
                    "score": episode.score.model_copy(
                        update={
                            "train_ready": decisions[
                                episode.id
                            ].compatibility_train_ready
                        }
                    ),
                }
            )
            for episode in graded_episodes
        ]
        counts = counts_of(decisions.values(), inventory)
        readiness = readiness_of(summaries, decisions, counts)
        sufficiency = sufficiency_of(requirements, counts)
        if diagnostics_plan is not None:
            from kalanos.analysis.diagnostics.runner import finish_diagnostics

            diagnostics = finish_diagnostics(
                [a.episode for a in analysed],
                diagnostics_plan,
                diagnostic_results,
                decisions,
                tier,
            )
        if requirements.min_pass_windows:
            from kalanos.analysis.diagnostics.validation import window_sufficiency

            sufficiency = window_sufficiency(
                sufficiency, diagnostics, decisions, inventory, requirements
            )

        # Step 3: the dataset rollup and the gate, which now only derives.
        episode_scores = [s.score for s in summaries]
        dataset_score = rollup(Level.DATASET, episode_scores, policy=policy).model_copy(
            update={
                "metrics": dataset_metric_rollups(summaries),
                "family_tails": dataset_family_tails(episode_scores, policy=policy),
            }
        )
        dataset_score, gate = apply_gate(
            summaries,
            findings,
            dataset_score,
            policy,
            decisions=decisions,
            readiness_score=readiness.score,
            passing_quality=readiness.passing_quality,
        )
        # The dataset compatibility boolean derives from the counts and the
        # inventory, whether or not the policy has a letter gate.
        dataset_score = dataset_score.model_copy(
            update={"train_ready": dataset_train_ready(counts)}
        )

        errors = [row for summary in summaries for row in summary.operational_errors]
        if diagnostics:
            errors.extend(
                {
                    "episode_id": r.episode_id or "dataset",
                    "metric": f"diagnostics.{r.kind}.{r.id}",
                    "reason": r.reason or "diagnostic failed",
                }
                for r in diagnostics.results
                if r.availability.value == "error"
            )
        if errors and run is not None:
            run = run.model_copy(update={"completion": RunCompletion.PARTIAL})
        fields: dict[str, Any] = dict(
            diagnostics=diagnostics,
            operational_errors=errors,
            root=root,
            coverage=report_coverage(ledgers, inventory),
            score=dataset_score,
            findings=sort_findings(findings),
            skipped=list(skipped),
            unresolved=list(unresolved),
            policy_version=policy.schema_version,
            family_weights=dict(policy.family_weights),
            duration_s=duration_s,
            source=source,
            datasets=list(datasets),
            categories=dictionary.category_groups,
            mapping_overrides=list(mapping_overrides),
            gate=gate,
            readiness=readiness,
            producer=producer,
            run=run,
            scope=scope,
            inventory=inventory,
            eligibility_counts=counts,
            sufficiency=sufficiency,
            binding_conflicts=list(binding_conflicts),
            cameras=cameras,
            duplicates=duplicates,
            near_duplicates=near_duplicates,
            motion_variation=motion_variation,
            dataset_warnings=dataset_warnings,
        )
        if self._keep_episodes:
            return Report(**fields, episodes=graded_episodes)

        # Step 4: a header holding no episodes is not a valid `Report`,
        # so it is built without validation and checked against the summaries instead.
        header = Report.model_construct(**fields, episodes=[])
        check_agreement(
            header,
            [
                AgreementRow(
                    id=s.id,
                    eligibility=decisions[s.id],
                    train_ready=decisions[s.id].compatibility_train_ready,
                )
                for s in summaries
            ],
            ledgers,
        )
        return StreamedReport(
            header=header,
            blobs=blobs,
            decisions=decisions,
            card_rows=[
                CardRow(
                    id=s.id,
                    source_paths=s.source_paths,
                    score=s.score.score,
                    eligibility=decisions[s.id],
                )
                for s in summaries
                if decisions[s.id].status is not EligibilityStatus.PASS
            ],
            unmapped_types=frozenset(t for s in summaries for t in s.unmapped_types),
        )


@overload
def assemble_report(
    *,
    root: UPath,
    analysed: Sequence[AnalysedEpisode],
    policy: Policy,
    skipped: Sequence[SkippedSource] = (),
    unresolved: Sequence[UnresolvedSource] = (),
    duration_s: float | None = None,
    source: SourceInfo | None = None,
    datasets: Sequence[DatasetInfo] = (),
    vision_samples: int = DEFAULT_VISION_SAMPLES,
    full_frame_scan: bool = False,
    vision: VisionSpec | None = None,
    mapping_overrides: Sequence[MappingOverride] = (),
    requirements: RequirementsSection | None = None,
    scope: EvaluationScope | None = None,
    producer: Producer | None = None,
    run: RunInfo | None = None,
    inventory: Inventory | None = None,
    binding_conflicts: Sequence[BindingConflict] = (),
    diagnostics_plan=None,
    keep_episodes: Literal[True] = True,
) -> Report: ...


@overload
def assemble_report(
    *,
    root: UPath,
    analysed: Sequence[AnalysedEpisode],
    policy: Policy,
    skipped: Sequence[SkippedSource] = (),
    unresolved: Sequence[UnresolvedSource] = (),
    duration_s: float | None = None,
    source: SourceInfo | None = None,
    datasets: Sequence[DatasetInfo] = (),
    vision_samples: int = DEFAULT_VISION_SAMPLES,
    full_frame_scan: bool = False,
    vision: VisionSpec | None = None,
    mapping_overrides: Sequence[MappingOverride] = (),
    requirements: RequirementsSection | None = None,
    scope: EvaluationScope | None = None,
    producer: Producer | None = None,
    run: RunInfo | None = None,
    inventory: Inventory | None = None,
    binding_conflicts: Sequence[BindingConflict] = (),
    diagnostics_plan=None,
    keep_episodes: bool,
) -> Report | StreamedReport: ...


def assemble_report(
    *,
    root: UPath,
    analysed: Sequence[AnalysedEpisode],
    policy: Policy,
    skipped: Sequence[SkippedSource] = (),
    unresolved: Sequence[UnresolvedSource] = (),
    duration_s: float | None = None,
    source: SourceInfo | None = None,
    datasets: Sequence[DatasetInfo] = (),
    vision_samples: int = DEFAULT_VISION_SAMPLES,
    full_frame_scan: bool = False,
    vision: VisionSpec | None = None,
    mapping_overrides: Sequence[MappingOverride] = (),
    requirements: RequirementsSection | None = None,
    scope: EvaluationScope | None = None,
    producer: Producer | None = None,
    run: RunInfo | None = None,
    inventory: Inventory | None = None,
    binding_conflicts: Sequence[BindingConflict] = (),
    diagnostics_plan=None,
    keep_episodes: bool = True,
) -> Report | StreamedReport:
    """Grade every analysed Episode and assemble the run's Report.

    Parameters
    ----------
    root : UPath
        The path the user pointed at.
    analysed : Sequence[AnalysedEpisode]
        One entry per recording an adapter read: the canonical episode,
        the name of the adapter that read it, and the policy to grade it against,
        which may carry a per-deployment limit the run's own `policy` does not.
        Without `keep_episodes`, a `deque` is emptied as its episodes are graded,
        so each recording's payloads can be let go of.
    policy : Policy
        The loaded grading policy. Distinct from each `AnalysedEpisode`'s own `policy`:
        this is what the dataset-level rollup and `Report.policy_version` grade against.
    skipped : Sequence[SkippedSource]
        Files discovery declined to carry forward, or that no adapter claimed.
    unresolved : Sequence[UnresolvedSource]
        Files an adapter bid on and then could not read.
    duration_s : float or None
        Wall-clock time the run took, if the caller is timing it.
    source : SourceInfo or None
        What `root` was resolved from, if the caller resolved it.
    datasets : Sequence[DatasetInfo]
        What the adapter declared about each path it read, in walk order.
    vision_samples : int
        How many frames blur and exposure sample, and windows frozen frames read.
    full_frame_scan : bool
        Whether frame metrics read every frame rather than a sample.
    vision : VisionSpec or None
        Decode caps, previews and exposure levels; the defaults when `None`.
    mapping_overrides : Sequence[MappingOverride]
        The per-run overrides the pipeline applied.
    requirements : RequirementsSection or None
        What a pass needs. `None` uses the built-in `numeric-core` scope.
    scope : EvaluationScope or None
        The scope identities to name on the report.
    producer, run : Producer, RunInfo or None
        Provenance, when the caller resolved it.
    inventory : Inventory or None
        Expected/loaded/failed episodes. `None` means every loaded episode
        is the whole inventory.
    binding_conflicts : Sequence[BindingConflict]
        Mapping disagreements the resolver recorded.
    keep_episodes : bool
        Return a `Report` holding every graded episode.
        Unset, each graded episode is held only as compressed JSON,
        and a `StreamedReport` is returned,
        unless a diagnostics plan is set: its diagnostics read across episodes.

    Returns
    -------
    Report or StreamedReport
        Every analysed episode, graded and rolled up to a Dataset-level score,
        alongside every skipped and unresolved file with its reason or evidence.
    """
    builder = ReportBuilder(
        policy=policy,
        requirements=requirements,
        tier=scope.tier if scope is not None else ExecutionTier.STANDARD,
        vision_samples=vision_samples,
        full_frame_scan=full_frame_scan,
        vision=vision,
        diagnostics_plan=diagnostics_plan,
        keep_episodes=keep_episodes,
    )
    if keep_episodes or diagnostics_plan is not None:
        items: Iterable[AnalysedEpisode] = analysed
    else:
        queue = analysed if isinstance(analysed, deque) else deque(analysed)
        items = (queue.popleft() for _ in range(len(queue)))
    for item in items:
        builder.add(item)
    return builder.finish(
        root=root,
        skipped=skipped,
        unresolved=unresolved,
        duration_s=duration_s,
        source=source,
        datasets=datasets,
        mapping_overrides=mapping_overrides,
        scope=scope,
        producer=producer,
        run=run,
        inventory=inventory,
        binding_conflicts=binding_conflicts,
    )
