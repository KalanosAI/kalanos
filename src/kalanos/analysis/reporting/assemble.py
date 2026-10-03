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
from collections.abc import Sequence
from typing import cast

# External
from upath import UPath

# Internal
from kalanos.analysis.adapters.video import SampledFrames
from kalanos.analysis.calibration import apply_calibration
from kalanos.analysis.coverage import episode_coverage, report_coverage, stream_rows
from kalanos.analysis.identities import adapter_versions, detector_versions
from kalanos.analysis.metrics.registry import (
    run_channel_metrics,
    run_episode_metrics,
    run_stream_metrics,
)
from kalanos.analysis.metrics.vision import camera_frames, is_camera_footage, thumbnail
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.binding import (
    SAMPLED_VIDEO_QUALITY_CAPABILITY,
    VIDEO_QUALITY_CAPABILITY,
    BindingConflict,
    EvaluationScope,
    RequirementsSection,
)
from kalanos.analysis.models.diagnostics import VisionSpec
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.discovery import SkippedSource, SourceInfo
from kalanos.analysis.models.domain import Episode, Stream
from kalanos.analysis.models.mapping import MappingOverride
from kalanos.analysis.models.metrics import (
    DEFAULT_VISION_SAMPLES,
    ChannelContext,
    EpisodeContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
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
    AnalysedEpisode,
    GradedChannel,
    GradedEpisode,
    GradedStream,
    PayloadStatus,
    Report,
    StreamEvaluation,
)
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, FindingLocation
from kalanos.analysis.scoring.cameras import compare_cameras
from kalanos.analysis.scoring.eligibility import (
    counts_of,
    dataset_train_ready,
    decide_all,
    readiness_of,
    sufficiency_of,
)
from kalanos.analysis.scoring.gate import apply_gate
from kalanos.analysis.scoring.score import rollup, score_metrics, sort_findings
from kalanos.assets.dictionary import load_default_dictionary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)

# The vision metrics that point at a worst frame, and the evidence key that names it.
_THUMBNAIL_INDEX = {
    "sharpness_score": "worst_index",
    "exposure_shift_pct": "worst_index",
    "frozen_frame_pct": "first_run_index",
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _attach_thumbnails(
    stream_metrics: dict[str, MetricResult], stream_ctx: StreamContext
) -> dict[str, MetricResult]:
    """Add the worst frame's thumbnail to each vision result graded warning or worse."""

    payload = stream_ctx.payload
    if not isinstance(payload, SampledFrames):
        return stream_metrics
    attached = dict(stream_metrics)
    for name, key in _THUMBNAIL_INDEX.items():
        result = attached.get(name)
        if result is None or result.status not in (
            MetricStatus.WARNING,
            MetricStatus.CRITICAL,
        ):
            continue
        index = result.evidence.get(key)
        if index is None:
            continue
        image = thumbnail(payload, index, stream_ctx.vision.preview_size)
        if image is not None:
            attached[name] = result.model_copy(
                update={"evidence": {**result.evidence, "thumbnail_png_base64": image}}
            )
    return attached


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

    Returns
    -------
    tuple[GradedStream, list[Finding]]
        The stream's own metrics and every channel, each graded; a
        Stream-level score rolled up over both; and every finding raised
        anywhere in the stream, unsorted — only `assemble_report` sorts,
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
    stream_metrics = _attach_thumbnails(stream_metrics, stream_ctx)

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

    # Step 3: fold the stream's own score in alongside its channels' —
    # one more equal-weight contributor, the same rule every other level uses.
    stream_score = rollup(
        Level.STREAM, [*(gc.score for gc in graded_channels), own_score], policy=policy
    )
    return (
        GradedStream(
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
            frames=camera_frames(stream_ctx)
            if is_camera_footage(stream.kind.value, stream.taxonomy_type)
            else None,
        ),
        findings,
    )


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

    Returns
    -------
    tuple[GradedEpisode, list[Finding]]
        Every stream and channel in `episode`, graded, alongside the
        episode's own metrics; a score rolled up over both;
        the recording's duration and sample count;
        and every finding raised anywhere in it, unsorted —
        only `assemble_report` sorts the dataset-wide list.
    """

    # Step 1: grade every stream, and the channels within each.
    graded_streams: list[GradedStream] = []
    findings: list[Finding] = []
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
        )
        graded_streams.append(graded_stream)
        findings.extend(stream_findings)

    # Step 2: grade the episode's own metrics — the ones that compare streams within it,
    # rather than reading any one of them alone.
    episode_ctx = EpisodeContext(episode=episode)
    episode_results = run_episode_metrics(episode_ctx)
    episode_metrics, own_score, episode_findings = score_metrics(
        results=episode_results,
        level=Level.EPISODE,
        taxonomy_type=Level.EPISODE.value,
        policy=policy,
        location=FindingLocation(episode_id=episode.id),
    )
    findings.extend(episode_findings)

    # Step 3: fold the episode's own score in alongside its streams' —
    # one more equal-weight contributor, the same rule every other level uses.
    episode_score = rollup(
        Level.EPISODE, [*(gs.score for gs in graded_streams), own_score], policy=policy
    )

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
) -> Report:
    """Grade every analysed Episode and assemble the run's Report.

    Parameters
    ----------
    root : UPath
        The path the user pointed at.
    analysed : Sequence[AnalysedEpisode]
        One entry per recording an adapter read: the canonical episode,
        the name of the adapter that read it, and the policy to grade it against,
        which may carry a per-deployment limit the run's own `policy` does not.
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

    Returns
    -------
    Report
        Every analysed episode, graded and rolled up to a Dataset-level score,
        alongside every skipped and unresolved file with its reason or evidence.
    """

    from kalanos.analysis.diagnostics.runner import validate_review_plan

    for item in analysed:
        validate_review_plan(diagnostics_plan, item.policy)
    dictionary = load_default_dictionary()
    tier = scope.tier if scope is not None else ExecutionTier.STANDARD
    requirements = requirements or RequirementsSection()
    policy = scope_policy(policy, requirements)
    graded_episodes: list[GradedEpisode] = []
    findings: list[Finding] = []
    for item in analysed:
        graded_episode, episode_findings = grade_episode(
            item.episode,
            adapter=item.adapter,
            adapter_confidence=item.adapter_confidence,
            policy=scope_policy(item.policy, requirements),
            dictionary=dictionary,
            vision_samples=vision_samples,
            full_frame_scan=full_frame_scan,
            vision=vision,
            tier=tier,
        )
        graded_episodes.append(graded_episode)
        findings.extend(episode_findings)

    # Step 1: decide eligibility once, now that every metric has run. Every
    # count, the gate and the compatibility booleans derive from this.
    scope = scope or EvaluationScope(
        requirements_id=requirements.id, policy_id="default-decisions-v1"
    )
    inventory = inventory or Inventory(loaded=len(graded_episodes))
    producer = producer or Producer(version="unknown")
    producer = producer.model_copy(
        update={
            "metrics": detector_versions(),
            "adapters": adapter_versions(item.adapter for item in analysed),
        }
    )
    findings = apply_calibration(
        findings,
        {item.episode.id: item.policy for item in analysed},
        run,
        scope,
        producer,
    )
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
            {item.episode.id: item.policy for item in analysed},
            {e.id: e for e in graded_episodes},
        )
        findings.extend(
            review_findings([r for rs in diagnostic_results.values() for r in rs])
        )
        graded_episodes = [
            e.model_copy(update={"diagnostics": diagnostic_results[e.id]})
            for e in graded_episodes
        ]
    cameras, camera_findings = compare_cameras(
        graded_episodes, grades_vision=grades_vision(requirements)
    )
    findings.extend(camera_findings)
    graded_episodes = [
        e.model_copy(update={"coverage": episode_coverage(e, requirements)})
        for e in graded_episodes
    ]
    decisions = decide_all(
        graded_episodes, findings, requirements=requirements, policy_id=scope.policy_id
    )
    graded_episodes = [
        episode.model_copy(
            update={
                "eligibility": decisions[episode.id],
                "score": episode.score.model_copy(
                    update={
                        "train_ready": decisions[episode.id].compatibility_train_ready
                    }
                ),
            }
        )
        for episode in graded_episodes
    ]
    counts = counts_of(decisions.values(), inventory)
    readiness = readiness_of(graded_episodes, decisions, counts)
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

    # Step 2: the dataset rollup and the gate, which now only derives.
    dataset_score = rollup(
        Level.DATASET, [ge.score for ge in graded_episodes], policy=policy
    )
    dataset_score, gate = apply_gate(
        graded_episodes,
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

    errors = []
    for episode in graded_episodes:
        if episode.coverage:
            for row in episode.coverage.metrics:
                if row.error:
                    errors.append(
                        {
                            "episode_id": episode.id,
                            "metric": row.key,
                            "reason": "; ".join(row.reasons),
                        }
                    )
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
    return Report(
        diagnostics=diagnostics,
        operational_errors=errors,
        root=root,
        coverage=report_coverage(graded_episodes, inventory),
        score=dataset_score,
        episodes=graded_episodes,
        findings=sort_findings(findings),
        skipped=list(skipped),
        unresolved=list(unresolved),
        policy_version=policy.schema_version,
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
    )
