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
from kalanos.analysis.metrics.registry import (
    run_channel_metrics,
    run_episode_metrics,
    run_stream_metrics,
)
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.discovery import SkippedSource, SourceInfo
from kalanos.analysis.models.domain import Episode, Stream
from kalanos.analysis.models.metrics import (
    ChannelContext,
    EpisodeContext,
    Level,
    StreamContext,
)
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.report import (
    AnalysedEpisode,
    GradedChannel,
    GradedEpisode,
    GradedStream,
    Report,
)
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, FindingLocation
from kalanos.analysis.scoring.score import rollup, score_metrics, sort_findings


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def grade_stream(
    stream: Stream, *, policy: Policy, is_regular: bool, episode_id: str
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
        stream=stream.taxonomy_type,
        instance=stream.instance,
    )

    # Step 1: grade the stream's own metrics — the ones that read its clock
    # or its payload as a whole. These run whether or not the stream carries a payload.
    stream_ctx = StreamContext(stream=stream, is_regular=is_regular)
    stream_results = run_stream_metrics(stream_ctx)
    stream_metrics, own_score, stream_findings = score_metrics(
        stream_results,
        level=Level.STREAM,
        taxonomy_type=stream.taxonomy_type,
        policy=policy,
        location=location,
    )
    findings.extend(stream_findings)

    # Step 2: fetch the payload only when there is a channel to grade with it.
    # A video stream carries no channels, so it is never fetched here.
    graded_channels = []
    if stream.channels:
        if stream.payload is None:
            logger.warning(
                "%s: stream %r has %d channel(s) but no payload",
                stream.source_path,
                stream.taxonomy_type,
                len(stream.channels),
            )
        else:
            frame = stream.payload.fetch()
            for channel in stream.channels:
                ctx = ChannelContext(
                    channel=channel, values=frame[channel.name], stream=stream_ctx
                )
                results = run_channel_metrics(ctx)
                graded, score, channel_findings = score_metrics(
                    results,
                    level=Level.CHANNEL,
                    taxonomy_type=stream.taxonomy_type,
                    policy=policy,
                    location=location.model_copy(update={"channel": channel.name}),
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
            instance=stream.instance,
            attribution=stream.attribution,
            score=stream_score,
            metrics=stream_metrics,
            channels=graded_channels,
        ),
        findings,
    )


def grade_episode(
    episode: Episode, *, adapter: str, adapter_confidence: float, policy: Policy
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
            stream, policy=policy, is_regular=stream.is_regular, episode_id=episode.id
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
        ),
        findings,
    )


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

    Returns
    -------
    Report
        Every analysed episode, graded and rolled up to a Dataset-level score,
        alongside every skipped and unresolved file with its reason or evidence.
    """

    graded_episodes: list[GradedEpisode] = []
    findings: list[Finding] = []
    for item in analysed:
        graded_episode, episode_findings = grade_episode(
            item.episode,
            adapter=item.adapter,
            adapter_confidence=item.adapter_confidence,
            policy=item.policy,
        )
        graded_episodes.append(graded_episode)
        findings.extend(episode_findings)

    dataset_score = rollup(
        Level.DATASET, [ge.score for ge in graded_episodes], policy=policy
    )

    return Report(
        root=root,
        score=dataset_score,
        episodes=graded_episodes,
        findings=sort_findings(findings),
        skipped=list(skipped),
        unresolved=list(unresolved),
        policy_version=policy.schema_version,
        duration_s=duration_s,
        source=source,
        datasets=list(datasets),
    )
