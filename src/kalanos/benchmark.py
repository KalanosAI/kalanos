"""How often each metric fires on clean data, and how often it catches a defect.

The benign rate grades reference datasets as they are.
The detection rate injects each `kalanos.testing` defect into a sample of their episodes
and counts where a metric that graded good before now grades warning or critical.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib.metadata
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

# External
from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel, Field
from upath import UPath

# Internal
from kalanos.analysis.adapters.discover import discover_adapters
from kalanos.analysis.adapters.select import select_adapter
from kalanos.analysis.adapters.video import (
    DecodeFailed,
    DecoderUnavailable,
    VideoPayload,
    close_remote_handles,
)
from kalanos.analysis.bindings import (
    binding_records,
    check_matched,
    identity_from_records,
    resolve_episodes,
)
from kalanos.analysis.discovery.source import enforce_limits, resolve_source
from kalanos.analysis.execution import use_tier
from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.models.binding import Bundle, EvaluationScope
from kalanos.analysis.models.diagnostics import VisionSpec
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.discovery import SourceLimits
from kalanos.analysis.models.domain import Episode, FramePayload, Stream
from kalanos.analysis.models.errors import MappingOverrideError, NothingToGrade
from kalanos.analysis.models.metrics import (
    DEFAULT_VISION_SAMPLES,
    ChannelContext,
    Level,
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.provenance import ConfigIdentity, ExecutionTier
from kalanos.analysis.models.report import GradedEpisode, GradedStream
from kalanos.analysis.pipeline import with_declared_limits
from kalanos.analysis.reporting.assemble import (
    grade_episode,
    grade_stream,
    grades_vision,
    scope_policy,
)
from kalanos.assets.bundle import prepare_configuration
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.core.settings import get_settings
from kalanos.testing.injectors import (
    Defect,
    SyntheticFrames,
    add_noise,
    apply_defect,
    drift_channel,
    saturate_channel,
    spike_channel,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

REFERENCE_DATASETS = (
    "hf://datasets/lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19",
    "hf://datasets/lerobot/libero_10@551d7d86f25edd0ffeda8b60053c15438cfd1d6a",
)
DEFAULT_SAMPLE = 30

# How many clean episodes pass between progress lines on a long dataset.
_PROGRESS_EVERY = 50

_FIRING = {MetricStatus.WARNING, MetricStatus.CRITICAL}
_GRADED = {MetricStatus.GOOD, MetricStatus.WARNING, MetricStatus.CRITICAL}

# Amplitudes for the value injectors, in units of the channel's own spread.
_SPIKE_STD = 20.0
_NOISE_STD = 0.5
_DRIFT_STD = 3.0
_SATURATION_ABS_QUANTILE = 0.8

# Markdown tables break on a stray indent or blank line,
# so block tags leave neither behind.
_TEMPLATE_ENVIRONMENT = Environment(
    loader=PackageLoader("kalanos", "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)

# Defects that model a tactile array.
# Each is injected only into a stream the named metric would run on.
_TAXEL_DEFECTS = {Defect.DEAD_TAXEL: "dead_taxel_pct", Defect.HYSTERESIS: "hysteresis"}

# Defects that model a camera fault.
# Each is injected only into a stream whose frames are held in memory;
# a sampled video stream is decoded into memory first, at native resolution,
# so an injected camera is measured the way `kalanos grade` measures it.
_FRAME_DEFECTS = {Defect.FROZEN_FRAMES, Defect.BLUR, Defect.CLIPPED}
# The most one camera's decoded native RGB frames may occupy in memory for injection;
# each injected copy is held beside them, so the peak is about twice this.
_MAX_DECODE_BYTES = 1 << 30


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class BenignRate(BaseModel):
    """How often one metric fired on episodes believed clean.

    Attributes
    ----------
    observed_episodes : int
        Episodes with at least one good, warning or critical result for the metric.
    firing_episodes : int
        Of those, episodes with at least one warning or critical result.
    rate : float or None
        `firing_episodes / observed_episodes`, or `None` when nothing was observed,
        so an unobservable metric never reads as 0% firing.
    not_observable : str or None
        Why the metric graded nothing, set exactly when `observed_episodes == 0`.
    """

    metric: str
    family: str
    observed_episodes: int
    firing_episodes: int
    rate: float | None
    not_observable: str | None = None


class DetectionRate(BaseModel):
    """How often one metric caught one injected defect.

    Attributes
    ----------
    eligible : int
        Injections where the metric graded good at the injected location
        on the clean episode.
    detected : int
        Of those, injections after which it graded warning or critical.
    rate : float
        `detected / eligible`. A row exists only when `eligible > 0`.
    """

    defect: Defect
    metric: str
    eligible: int
    detected: int
    rate: float


class DatasetBenchmark(BaseModel):
    """One reference dataset's benign and detection rates.

    Attributes
    ----------
    uri : str
        The source as resolved.
    repo_id : str or None
        The hosted repository as `<org>/<name>`, or `None` for a local source.
    revision : str or None
        The commit read, or `None` when the source is not versioned.
    adapter : str
        The adapter that read the dataset.
    n_episodes : int
        How many episodes were graded clean.
    n_sampled : int
        How many episodes defects were injected into.
    benign : list[BenignRate]
        One row per registered metric, sorted by family then metric.
    detection : list[DetectionRate]
        One row per defect and metric with an eligible injection.
    not_injected : dict[Defect, str]
        Each defect that never ran on this dataset, with the reason.
    """

    uri: str
    repo_id: str | None
    revision: str | None
    adapter: str
    n_episodes: int
    n_sampled: int
    benign: list[BenignRate]
    detection: list[DetectionRate]
    not_injected: dict[Defect, str] = Field(default_factory=dict)
    scope: EvaluationScope | None = None
    configuration: dict[str, ConfigIdentity | None] = Field(default_factory=dict)
    policy_version: int | None = None


class Benchmark(BaseModel):
    """A benchmark run over one or more reference datasets.

    Attributes
    ----------
    kalanos_version : str
        The installed kalanos the run graded with.
    policy_version : int or None
        The grading policy's schema version.
    injection : dict[str, float]
        The amplitudes the value defects were injected at, in channel spread units.
    datasets : list[DatasetBenchmark]
        One entry per dataset, in the order given.
    """

    kalanos_version: str
    policy_version: int | None
    injection: dict[str, float]
    datasets: list[DatasetBenchmark]


@dataclass
class _Rates:
    """What `benchmark_episodes` measured over one stream of episodes."""

    benign: list[BenignRate]
    detection: list[DetectionRate]
    not_injected: dict[Defect, str]
    n_episodes: int
    n_sampled: int


@dataclass
class _MetricTally:
    """One metric's results across the clean episodes."""

    observed: int = 0
    firing: int = 0
    reason: str | None = None
    report_only: bool = False


@dataclass
class _Cell:
    eligible: int = 0
    detected: int = 0


@dataclass
class _Injections:
    """Detection counts, and which defects ran, across the sampled episodes."""

    cells: dict[tuple[Defect, str], _Cell] = field(
        default_factory=lambda: defaultdict(_Cell)
    )
    ran: set[Defect] = field(default_factory=set)
    errors: dict[Defect, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Matrix:
    """A detection matrix flattened to the strings its table shows."""

    metrics: list[str]
    rows: list[tuple[str, list[str]]]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _results(episode: GradedEpisode) -> Iterable[tuple[str, MetricResult]]:
    """Every metric result in one episode, at every level, with its metric name."""

    yield from episode.metrics.items()
    for stream in episode.streams:
        yield from stream.metrics.items()
        for channel in stream.channels:
            yield from channel.metrics.items()


def _tally_clean(graded: GradedEpisode, tallies: dict[str, _MetricTally]) -> None:
    """Count one clean episode into each metric's benign tally."""

    by_metric: dict[str, list[MetricResult]] = defaultdict(list)
    for name, result in _results(graded):
        by_metric[name].append(result)

    for name, results in by_metric.items():
        tally = tallies[name]
        statuses = {result.status for result in results}
        if statuses & _GRADED:
            tally.observed += 1
        if statuses & _FIRING:
            tally.firing += 1
        if MetricStatus.REPORT_ONLY in statuses:
            tally.report_only = True
        if tally.reason is None:
            for result in results:
                reason = result.evidence.get("reason")
                if result.status == MetricStatus.NOT_APPLICABLE and reason:
                    tally.reason = str(reason)
                    break


def _inject(stream: Stream, channel: str, defect: Defect) -> Stream:
    """Inject `defect` into `channel`, scaling a value defect to the channel's spread.

    Raises
    ------
    ValueError
        If the channel has no spread to scale by, or the injector refuses the stream.
    """

    payload = stream.payload
    if defect in _FRAME_DEFECTS and not stream.channels:
        ctx = StreamContext(stream=stream, is_regular=stream.is_regular)
        return cast(StreamContext, apply_defect(ctx, defect)).stream

    if not isinstance(payload, FramePayload):
        raise ValueError("stream carries no in-memory frame to inject into")
    values = payload.frame[channel].drop_nulls()

    if defect in {Defect.SPIKE, Defect.NOISE, Defect.DRIFT, Defect.SATURATION}:
        if not values.dtype.is_numeric():
            raise ValueError("channel has no spread to scale the defect by")
        std = cast(float | None, values.std())
        if not std:
            raise ValueError("channel has no spread to scale the defect by")
        match defect:
            case Defect.SPIKE:
                return spike_channel(stream, channel, magnitude=_SPIKE_STD)
            case Defect.NOISE:
                return add_noise(stream, channel, amplitude=_NOISE_STD * std)
            case Defect.DRIFT:
                return drift_channel(stream, channel, total=_DRIFT_STD * std)
            case _:
                quantile = values.abs().quantile(_SATURATION_ABS_QUANTILE)
                if not quantile:
                    raise ValueError("channel has no spread to scale the defect by")
                return saturate_channel(stream, channel, limit=quantile)

    [target] = [c for c in stream.channels if c.name == channel]
    ctx = ChannelContext(
        channel=target,
        values=payload.frame[channel],
        stream=StreamContext(stream=stream, is_regular=stream.is_regular),
    )
    return cast(ChannelContext, apply_defect(ctx, defect)).stream.stream


def _applies(defect: Defect, stream: Stream) -> bool:
    """Whether `defect` models a fault this stream could carry."""

    if defect in _FRAME_DEFECTS:
        return isinstance(stream.payload, SyntheticFrames)
    if defect in _TAXEL_DEFECTS:
        [entry] = [
            m
            for m in registered_metrics(Level.STREAM)
            if m.name == _TAXEL_DEFECTS[defect]
        ]
        return stream.taxonomy_type in entry.requires.taxonomy
    return isinstance(stream.payload, FramePayload) and bool(stream.channels)


def _decoded(stream: Stream) -> Stream:
    """Return a copy of a video stream with its frames decoded into memory.

    Raises
    ------
    ValueError
        If the decoded frames would pass the decode cap, the video cannot be decoded,
        or it decodes to a different frame count than the stream has timestamps.
    """

    payload = cast(VideoPayload, stream.payload)
    try:
        height, width, _ = payload.rgb_frame(0).shape
        needed = payload.frame_count * height * width * 3
        if needed > _MAX_DECODE_BYTES:
            raise ValueError(
                f"video needs {needed} bytes decoded, "
                f"over the {_MAX_DECODE_BYTES}-byte decode cap"
            )
        frames = payload.rgb_frames(None)
    except (DecoderUnavailable, DecodeFailed) as exc:
        raise ValueError(str(exc)) from exc
    if len(frames) != len(stream.timestamps):
        raise ValueError(
            f"video decoded {len(frames)} frame(s) "
            f"for {len(stream.timestamps)} timestamp(s)"
        )
    return stream.model_copy(update={"payload": SyntheticFrames(frames=frames)})


def _located(graded: GradedStream, channel: str | None) -> dict[str, MetricResult]:
    """The stream's own results and those of `channel`, keyed by metric name."""

    located = dict(graded.metrics)
    for graded_channel in graded.channels:
        if graded_channel.channel.name == channel:
            located.update(graded_channel.metrics)
    return located


def _inject_episode(
    episode: Episode,
    clean: GradedEpisode,
    position: int,
    *,
    policy: Policy,
    injections: _Injections,
    vision_samples: int = DEFAULT_VISION_SAMPLES,
    full_frame_scan: bool = False,
    vision: VisionSpec | None = None,
    inject_cameras: bool = True,
) -> None:
    """Inject every applicable defect into each stream of one sampled episode.

    A camera defect goes into a video stream decoded into memory,
    and is measured against that decoded copy graded clean,
    so before and after read the same frames.
    Without `inject_cameras` no video stream is decoded.
    """

    def grade(stream: Stream, target: Stream, category: str | None) -> GradedStream:
        """Grade `target` in place of `stream` among the episode's streams."""

        graded, _ = grade_stream(
            stream=target,
            policy=policy,
            is_regular=stream.is_regular,
            episode_id=episode.id,
            category=category,
            vision_samples=vision_samples,
            episode_streams=[
                target if other is stream else other for other in episode.streams
            ],
            full_frame_scan=full_frame_scan,
            vision=vision,
        )
        return graded

    for stream, clean_stream in zip(episode.streams, clean.streams, strict=True):
        channel = (
            stream.channels[position % len(stream.channels)].name
            if stream.channels
            else None
        )
        before = _located(clean_stream, channel)
        decoded: tuple[Stream, dict[str, MetricResult]] | None = None
        if (
            inject_cameras
            and isinstance(stream.payload, VideoPayload)
            and not stream.channels
        ):
            try:
                frames = _decoded(stream)
            except ValueError as exc:
                logger.debug(
                    "%s: %s not decoded: %s", episode.id, stream.taxonomy_type, exc
                )
                for defect in _FRAME_DEFECTS:
                    injections.errors[defect] = str(exc)
            else:
                decoded = (
                    frames,
                    _located(grade(stream, frames, clean_stream.category), channel),
                )
        for defect in Defect:
            source, baseline = (
                decoded
                if defect in _FRAME_DEFECTS and decoded is not None
                else (stream, before)
            )
            if not _applies(defect, source):
                continue
            try:
                injected = _inject(source, channel or "", defect)
            except ValueError as exc:
                logger.debug(
                    "%s: %s not injected into %s: %s",
                    episode.id,
                    defect.value,
                    stream.taxonomy_type,
                    exc,
                )
                injections.errors[defect] = str(exc)
                continue
            injections.ran.add(defect)

            after = _located(grade(stream, injected, clean_stream.category), channel)
            for metric, result in baseline.items():
                if result.status != MetricStatus.GOOD:
                    continue
                cell = injections.cells[(defect, metric)]
                cell.eligible += 1
                if metric in after and after[metric].status in _FIRING:
                    cell.detected += 1


def _benign_rows(tallies: dict[str, _MetricTally]) -> list[BenignRate]:
    """One row per registered metric, observed or not."""

    rows = []
    for entry in registered_metrics():
        tally = tallies.get(entry.name, _MetricTally())
        if tally.observed:
            reason = None
        elif tally.reason:
            reason = tally.reason
        elif tally.report_only:
            reason = "report_only on every result"
        else:
            reason = "produced no result"
        rows.append(
            BenignRate(
                metric=entry.name,
                family=entry.family.value,
                observed_episodes=tally.observed,
                firing_episodes=tally.firing,
                rate=tally.firing / tally.observed if tally.observed else None,
                not_observable=reason,
            )
        )
    return sorted(rows, key=lambda row: (row.family, row.metric))


def benchmark_episodes(
    episodes: Iterable[Episode],
    *,
    policy: Policy,
    n_episodes: int | None,
    sample: int,
    tier: ExecutionTier = ExecutionTier.STANDARD,
    vision_samples: int = DEFAULT_VISION_SAMPLES,
    full_frame_scan: bool = False,
    vision: VisionSpec | None = None,
    inject_cameras: bool = True,
) -> _Rates:
    """Grade every episode clean, and inject defects into an evenly spaced sample.

    Parameters
    ----------
    n_episodes : int or None
        How many episodes to expect, to spread the sample across them.
        `None` samples the first `sample` episodes.
    sample : int
        The most episodes to inject defects into.
    vision_samples : int
        How many frames blur and exposure sample, and windows frozen frames read.
    full_frame_scan : bool
        Whether frame metrics read every frame rather than a sample.
    vision : VisionSpec or None
        Decode caps, previews and exposure levels; the defaults when `None`.
    inject_cameras : bool
        Whether to decode video streams for camera defects.
        A scope that does not grade vision could never count one,
        so decoding up to a gigabyte a camera would buy nothing.
    """

    stride = max(1, n_episodes // sample) if n_episodes and sample else 1
    tallies: dict[str, _MetricTally] = defaultdict(_MetricTally)
    injections = _Injections()
    seen = sampled = 0
    expected = n_episodes if n_episodes is not None else "?"
    planned = min(sample, n_episodes) if n_episodes is not None else sample
    dictionary = load_default_dictionary()
    for position, episode in enumerate(episodes):
        seen += 1
        clean, _ = grade_episode(
            episode,
            adapter="benchmark",
            adapter_confidence=1.0,
            policy=policy,
            dictionary=dictionary,
            vision_samples=vision_samples,
            full_frame_scan=full_frame_scan,
            vision=vision,
            tier=tier,
        )
        _tally_clean(clean, tallies)
        if seen % _PROGRESS_EVERY == 0:
            logger.info("graded %d/%s episode(s) clean", seen, expected)
        if (
            position % stride == 0
            and sampled < sample
            and tier != ExecutionTier.METADATA
        ):
            sampled += 1
            logger.info(
                "%s: injecting defects (sample %d/%d)", episode.id, sampled, planned
            )
            _inject_episode(
                episode,
                clean,
                sampled - 1,
                policy=policy,
                injections=injections,
                vision_samples=vision_samples,
                full_frame_scan=full_frame_scan,
                vision=vision,
                inject_cameras=inject_cameras,
            )
    logger.info("graded %d episode(s) clean, injected defects into %d", seen, sampled)

    detection = [
        DetectionRate(
            defect=defect,
            metric=metric,
            eligible=cell.eligible,
            detected=cell.detected,
            rate=cell.detected / cell.eligible,
        )
        for (defect, metric), cell in injections.cells.items()
        if cell.eligible
    ]
    not_injected = {
        defect: (
            "metadata tier does not inject payload defects"
            if tier == ExecutionTier.METADATA
            else "the scope does not grade vision"
            if defect in _FRAME_DEFECTS and not inject_cameras
            else injections.errors.get(defect, "no stream it applies to")
        )
        for defect in Defect
        if defect not in injections.ran
    }
    return _Rates(
        benign=_benign_rows(tallies),
        detection=sorted(detection, key=lambda row: (row.defect.value, row.metric)),
        not_injected=not_injected,
        n_episodes=seen,
        n_sampled=sampled,
    )


def benchmark_dataset(
    path: str,
    *,
    policy: Policy | None = None,
    sample: int,
    dictionary: Dictionary | None = None,
    bundle: Bundle | UPath | str | os.PathLike[str] | None = None,
    mapping: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    mapping_file: UPath | None = None,
    sidecar: bool = True,
    tier: ExecutionTier | None = None,
) -> DatasetBenchmark:
    """Benchmark one dataset, read whole by the adapter that bids highest on it.

    Raises
    ------
    SourceUnavailable
        If `path` does not exist, or a Hugging Face dataset cannot be read.
    SourceTooLarge
        If `path` is remote and over the configured limits.
    NothingToGrade
        If no adapter reads `path` as one dataset.
    AdapterTie
        If two adapters bid the same top confidence on it.
    """

    # Step 1: resolve the source, and refuse an oversized remote one before reading.
    root, source = resolve_source(path)
    logger.info("benchmarking %s", source.uri)
    settings = get_settings()
    enforce_limits(
        source,
        SourceLimits(
            max_bytes=settings.remote_max_bytes, max_files=settings.remote_max_files
        ),
    )

    policy, dictionary, config = prepare_configuration(
        root,
        policy=policy,
        dictionary=dictionary,
        bundle=bundle,
        mapping=mapping,
        mapping_file=mapping_file,
        sidecar=sidecar,
        tier=tier,
    )
    enforce_limits(source, config.limits)
    # Step 2: one adapter reads the whole root.
    selection = select_adapter(root, discover_adapters().adapters)
    if selection is None:
        raise NothingToGrade(
            f"{root}: no adapter reads it as one dataset, "
            "and the benchmark grades whole datasets"
        )

    # Step 3: fill the limits the adapter declares, as `kalanos grade` does.
    info = selection.adapter.describe(root)
    logger.info(
        "%s: read by %s, %s episode(s) declared",
        source.repo_id or source.uri,
        selection.name,
        info.episode_count if info.episode_count is not None else "?",
    )
    # Scoped as `kalanos grade` scopes it, so vision grades only when required.
    episode_policy = scope_policy(
        with_declared_limits(policy, info), config.requirements
    )

    # Step 4: resolve the same bindings as grade, without buffering payloads.
    channels = config.bundle.binding.channels if config.bundle.binding else []
    matched_channels: set[tuple[str | None, str, int | None]] = set()
    seen_features: set[str] = set()
    records: dict[str, dict] = {}

    def bound_episodes():
        for raw in selection.adapter.episodes(root):
            seen_features.update(
                s.source_field for s in raw.streams if s.source_field is not None
            )
            [bound] = resolve_episodes(
                [raw],
                dictionary=dictionary,
                channels=channels,
                overrides=config.overrides,
                source_identity=str(root),
                matched=matched_channels,
            )
            records.update(binding_records([bound]))
            yield bound

    vision = config.vision
    try:
        with use_tier(config.scope.tier):
            rates = benchmark_episodes(
                episodes=bound_episodes(),
                policy=episode_policy,
                n_episodes=info.episode_count,
                sample=sample,
                tier=config.scope.tier,
                vision_samples=vision.sample_frames,
                full_frame_scan=vision.full_frame_scan,
                vision=vision,
                inject_cameras=grades_vision(config.requirements),
            )
    finally:
        # Remote video stays open across episodes, as in `kalanos grade`.
        close_remote_handles()
    check_matched(channels, matched_channels)
    unmatched = [o.feature for o in config.overrides if o.feature not in seen_features]
    if unmatched:
        raise MappingOverrideError(f"mapping override matched no stream: {unmatched}")
    return DatasetBenchmark(
        uri=source.uri,
        repo_id=source.repo_id,
        revision=source.revision,
        adapter=selection.name,
        n_episodes=rates.n_episodes,
        n_sampled=rates.n_sampled,
        benign=rates.benign,
        detection=rates.detection,
        not_injected=rates.not_injected,
        scope=config.scope.model_copy(
            update={"binding_id": identity_from_records(records, config.binding_id).id}
        ),
        configuration={
            "binding": identity_from_records(records, config.binding_id),
            "requirements": config.requirements_id,
            "policy": config.policy_id,
            "dictionary": config.dictionary_id,
            "execution": config.execution_id,
            "bundle": config.bundle_id,
        },
        policy_version=policy.schema_version,
    )


def run_benchmark(
    paths: Sequence[str] = REFERENCE_DATASETS,
    *,
    sample: int = DEFAULT_SAMPLE,
    policy: Policy | None = None,
    dictionary: Dictionary | None = None,
    bundle: Bundle | UPath | str | os.PathLike[str] | None = None,
    mapping: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    mapping_file: UPath | None = None,
    sidecar: bool = True,
    tier: ExecutionTier | None = None,
) -> Benchmark:
    """Benchmark each dataset under the configured policy and dictionary.

    Parameters
    ----------
    sample : int
        How many episodes per dataset to inject defects into.
    """

    datasets = [
        benchmark_dataset(
            path,
            policy=policy,
            sample=sample,
            dictionary=dictionary,
            bundle=bundle,
            mapping=mapping,
            mapping_file=mapping_file,
            sidecar=sidecar,
            tier=tier,
        )
        for path in paths
    ]
    versions = {d.policy_version for d in datasets}
    return Benchmark(
        kalanos_version=importlib.metadata.version("kalanos"),
        policy_version=next(iter(versions)) if len(versions) == 1 else None,
        injection={
            "spike_std": _SPIKE_STD,
            "noise_std": _NOISE_STD,
            "drift_std": _DRIFT_STD,
            "saturation_abs_quantile": _SATURATION_ABS_QUANTILE,
        },
        datasets=datasets,
    )


def _detection_matrix(dataset: DatasetBenchmark) -> _Matrix:
    """Lay out the detection cells with defects down and metrics across.

    A defect that never ran has no row, and a cell with no eligible injection is blank.
    """

    metrics = sorted({row.metric for row in dataset.detection})
    cells = {(row.defect, row.metric): row for row in dataset.detection}
    rows = [
        (
            defect.value,
            [
                f"{cells[defect, m].detected}/{cells[defect, m].eligible}"
                if (defect, m) in cells
                else ""
                for m in metrics
            ],
        )
        for defect in Defect
        if defect not in dataset.not_injected
    ]
    return _Matrix(metrics=metrics, rows=rows)


def render_markdown(benchmark: Benchmark) -> str:
    """Render a benchmark as a self-contained Markdown document."""

    template = _TEMPLATE_ENVIRONMENT.get_template("benchmark.md.j2")
    return template.render(
        benchmark=benchmark,
        sections=[
            (dataset, _detection_matrix(dataset)) for dataset in benchmark.datasets
        ],
    )
