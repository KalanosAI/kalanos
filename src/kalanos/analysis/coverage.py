"""Build and merge coverage from actual result states, not policy grades."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections import Counter, defaultdict
from collections.abc import Iterable

# Internal
from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.metrics.vision import CORE_METRICS, is_camera_footage
from kalanos.analysis.models.binding import (
    SAMPLED_VIDEO_QUALITY_CAPABILITY,
    VIDEO_QUALITY_CAPABILITY,
    RequirementsSection,
)
from kalanos.analysis.models.coverage import Availability, Coverage, CoverageRow
from kalanos.analysis.models.domain import Stream
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.provenance import Inventory
from kalanos.analysis.models.report import (
    GradedChannel,
    GradedEpisode,
    GradedStream,
    StreamEvaluation,
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def state_of(result: MetricResult) -> Availability:
    if result.availability is not None and result.availability != Availability.COMPUTED:
        return result.availability
    if (
        result.value is not None
        and math.isfinite(result.value)
        and result.status != MetricStatus.NOT_APPLICABLE
    ):
        return Availability.COMPUTED
    return Availability.UNAVAILABLE


def row(
    key: str, unit: str, states: Iterable[tuple[Availability, str | None]]
) -> CoverageRow:
    states = list(states)
    counts = Counter(s.value for s, _ in states)
    reasons = Counter(reason for _, reason in states if reason)
    eligible = sum(
        counts[k.value]
        for k in (
            Availability.COMPUTED,
            Availability.UNAVAILABLE,
            Availability.SKIPPED,
            Availability.ERROR,
        )
    )
    return CoverageRow(
        key=key, unit=unit, eligible=eligible, reasons=dict(reasons), **counts
    )


def metric_rows(
    level: Level,
    subjects: Iterable[tuple[str, dict[str, MetricResult]]],
    *,
    skipped: Availability | None = None,
    skip_reason: str | None = None,
) -> list[CoverageRow]:
    """Each subject is (taxonomy, result mapping); omitted payloads remain counted."""
    rows = []
    for entry in registered_metrics(level):
        states = []
        for taxonomy, results in subjects:
            result = results.get(entry.name)
            if result is not None:
                states.append(
                    (
                        state_of(result),
                        result.evidence.get("reason")
                        or result.evidence.get("ungraded_reason")
                        if state_of(result) != Availability.COMPUTED
                        else None,
                    )
                )
            elif (
                entry.requires.taxonomy
                and taxonomy not in entry.requires.taxonomy
                and not taxonomy.startswith("unmapped")
            ):
                states.append((Availability.NOT_APPLICABLE, "outside metric taxonomy"))
            else:
                states.append(
                    (
                        skipped or Availability.UNAVAILABLE,
                        skip_reason or "no result recorded",
                    )
                )
        rows.append(
            row(f"{entry.family.value}.{entry.name}", f"{level.value}_episode", states)
        )
    return rows


def stream_rows(
    stream: Stream,
    graded: list[GradedChannel],
    metrics: dict[str, MetricResult],
    evaluation: StreamEvaluation,
) -> list[CoverageRow]:
    subjects = [
        (
            g.channel.binding.taxonomy_type
            if g.channel.binding
            else stream.taxonomy_type,
            g.metrics,
        )
        for g in graded
    ]
    if not graded:
        subjects = [
            (c.binding.taxonomy_type if c.binding else stream.taxonomy_type, {})
            for c in stream.channels
        ]
    state = {"skipped": Availability.SKIPPED, "error": Availability.ERROR}.get(
        evaluation.payload.value, Availability.UNAVAILABLE
    )
    return metric_rows(Level.STREAM, [(stream.taxonomy_type, metrics)]) + metric_rows(
        Level.CHANNEL, subjects, skipped=state, skip_reason=evaluation.reason
    )


def merge_rows(rows: Iterable[CoverageRow]) -> list[CoverageRow]:
    grouped = defaultdict(list)
    for item in rows:
        grouped[(item.key, item.unit)].append(item)
    merged = []
    fields = [s.value for s in Availability] + ["eligible"]
    for (key, unit), items in sorted(grouped.items()):
        reasons = Counter()
        for item in items:
            reasons.update(item.reasons)
        merged.append(
            CoverageRow(
                key=key,
                unit=unit,
                reasons=dict(reasons),
                **{k: sum(getattr(i, k) for i in items) for k in fields},
            )
        )
    return merged


def _sampled(stream):
    """A camera stream's `CORE_METRICS` results when all computed, else `None`.

    A read that decoded fewer frames than it requested has not finished its sample,
    however many the metrics computed on.
    """

    if not is_camera_footage(stream.kind, stream.taxonomy_type):
        return None
    if stream.frames is not None and stream.frames.missing_rows:
        return None
    results = [stream.metrics.get(name) for name in CORE_METRICS]
    if all(r is not None and state_of(r) == Availability.COMPUTED for r in results):
        return results
    return None


def _unsampled_reason(stream):
    """Why the vision metrics did not sample a stream, from their own evidence."""

    for name in CORE_METRICS:
        result = stream.metrics.get(name)
        if result is not None and state_of(result) != Availability.COMPUTED:
            reason = result.evidence.get("reason")
            if reason:
                return reason
    if stream.frames is not None and stream.frames.missing_rows:
        return f"{len(stream.frames.missing_rows)} requested frame(s) were not decoded"
    return "vision metrics did not sample the stream"


def vision_metric_coverage(streams: list[GradedStream]) -> tuple[bool, bool]:
    """Whether the vision metrics sampled every camera stream, and read every frame.

    Parameters
    ----------
    streams : list of GradedStream

    Returns
    -------
    tuple of (bool, bool)
        `(sampled, full)`: sampled when every camera stream computed each of
        `CORE_METRICS`, full when each of them also measured every declared frame.
        No camera stream is `(False, False)`: nothing was evaluated.
    """

    cameras = [s for s in streams if is_camera_footage(s.kind or "", s.taxonomy_type)]
    if not cameras:
        return False, False
    results = []
    for s in cameras:
        sampled = _sampled(s)
        if sampled is None:
            return False, False
        results += sampled
    full = all(
        r.evidence.get("n_sampled") == r.evidence.get("n_frames") for r in results
    )
    return True, full


def vision_metric_frames(streams: list[GradedStream]) -> tuple[int, int] | None:
    """`(examined, declared)` frames over the cameras the vision metrics sampled.

    A camera's examined count is the most frames any of `CORE_METRICS` measured on it.

    Parameters
    ----------
    streams : list of GradedStream

    Returns
    -------
    tuple of (int, int) or None
        `None` when the metrics sampled no camera.
    """

    counts = [
        (
            max(int(r.evidence.get("n_sampled", 0)) for r in results),
            int(results[0].evidence.get("n_frames", 0)),
        )
        for results in map(_sampled, streams)
        if results is not None
    ]
    if not counts:
        return None
    return sum(e for e, _ in counts), sum(d for _, d in counts)


def episode_coverage(
    episode: GradedEpisode, requirements: RequirementsSection
) -> Coverage:
    metrics = merge_rows(
        [r for s in episode.streams for r in s.coverage]
        + metric_rows(Level.EPISODE, [("episode", episode.metrics)])
    )
    channels = [
        c
        for s in episode.streams
        for c in ([g.channel for g in s.channels] or s.declared_channels)
    ]
    numeric_streams = [s for s in episode.streams if s.channels or s.declared_channels]
    numeric_ok = bool(numeric_streams) and all(
        s.evaluation.payload.value == "computed"
        and s.channels
        and all(
            "missing_pct" in g.metrics
            and state_of(g.metrics["missing_pct"]) == Availability.COMPUTED
            for g in s.channels
        )
        for s in numeric_streams
    )
    acquisition_ok = bool(episode.streams) and all(
        all(
            k in s.metrics and state_of(s.metrics[k]) == Availability.COMPUTED
            for k in ("effective_hz", "dt_jitter_ms", "drop_rate")
        )
        for s in episode.streams
    )
    diagnostic_results = episode.diagnostics
    metrics += [
        row(f"diagnostics.{r.kind}.{r.id}", "episode", [(r.availability, r.reason)])
        for r in diagnostic_results
    ]
    by_kind = {
        kind: [r for r in diagnostic_results if r.kind == kind]
        for kind in ("timing", "tracking", "motion", "vision", "windows")
    }

    def done(kind: str) -> bool:
        return bool(by_kind[kind]) and all(
            r.availability == Availability.COMPUTED for r in by_kind[kind]
        )

    camera_count = sum(
        is_camera_footage(s.kind or "", s.taxonomy_type) for s in episode.streams
    )
    sampled_vision = done("vision") and len(by_kind["vision"]) == camera_count
    full_vision = sampled_vision and all(
        r.measurements.get("examined_frames") == r.measurements.get("declared_frames")
        for r in by_kind["vision"]
    )
    metric_sampled, metric_full = vision_metric_coverage(episode.streams)
    diagnostic_capabilities = {
        "cross_stream_timing": done("timing"),
        "action_consistency": done("tracking"),
        "motion_shape": done("motion"),
        "sampled_video_quality": sampled_vision or metric_sampled,
        "video_quality": full_vision or metric_full,
        "training_windows": done("windows"),
    }
    keys = set(requirements.required_capabilities) | {
        "numeric",
        "acquisition_timing",
        "video_quality",
        "behavioral_diversity",
    }
    capabilities = []
    for key in sorted(keys):
        required = key in requirements.required_capabilities or (
            key == "numeric" and requirements.require_numeric_payloads
        )
        computed = {
            "numeric": numeric_ok,
            "acquisition_timing": acquisition_ok,
            **diagnostic_capabilities,
        }.get(key, False)
        state = (
            Availability.COMPUTED
            if computed and required
            else Availability.UNAVAILABLE
            if required
            else Availability.NOT_REQUIRED
        )
        capabilities.append(
            row(
                key,
                "episode",
                [
                    (
                        state,
                        None
                        if state == Availability.COMPUTED
                        else "outside required scope"
                        if not required
                        else "required capability was not fully evaluated",
                    )
                ],
            )
        )
    mapping = [
        (
            Availability.COMPUTED
            if c.binding and not c.binding.taxonomy_type.startswith("unmapped")
            else Availability.UNAVAILABLE,
            None,
        )
        for c in channels
    ]
    clocks = [
        (
            Availability.COMPUTED
            if s.clock_info and s.clock_info.certifies_acquisition
            else Availability.UNAVAILABLE,
            None
            if s.clock_info and s.clock_info.certifies_acquisition
            else "no producer capture evidence",
        )
        for s in episode.streams
    ]
    cameras = [
        s for s in episode.streams if is_camera_footage(s.kind or "", s.taxonomy_type)
    ]
    visual_state = (
        Availability.UNAVAILABLE
        if "video_quality" in requirements.required_capabilities
        else Availability.NOT_REQUIRED
    )
    dimensions = [
        row("semantic_mapping", "channel_episode", mapping),
        row("capture_origin_evidence", "stream_episode", clocks),
        row(
            "visual_quality",
            "camera_stream_episode",
            [
                (
                    visual_state,
                    "visual quality not evaluated; decoded frame count unknown",
                )
                for s in cameras
            ],
        ),
        row(
            "behavioral_diversity",
            "episode",
            [
                (
                    Availability.UNAVAILABLE
                    if "behavioral_diversity" in requirements.required_capabilities
                    else Availability.NOT_REQUIRED,
                    "behavioral diversity not evaluated",
                )
            ],
        ),
    ]
    if by_kind["vision"]:
        dimensions = [r for r in dimensions if r.key != "visual_quality"] + [
            row(
                "visual_quality",
                "camera_stream_episode",
                [(r.availability, r.reason) for r in by_kind["vision"]],
            )
        ]
    metric_frames = vision_metric_frames(episode.streams)
    # The vision metrics run under every scope; only one that needs video
    # counts them against the camera streams, the rest keep them not required.
    video_required = bool(
        {VIDEO_QUALITY_CAPABILITY, SAMPLED_VIDEO_QUALITY_CAPABILITY}
        & set(requirements.required_capabilities)
    )
    if not by_kind["vision"] and metric_frames is not None and video_required:
        dimensions = [r for r in dimensions if r.key != "visual_quality"] + [
            row(
                "visual_quality",
                "camera_stream_episode",
                [
                    (Availability.COMPUTED, None)
                    if _sampled(s) is not None
                    else (Availability.UNAVAILABLE, _unsampled_reason(s))
                    for s in cameras
                ],
            )
        ]
    return Coverage(
        metrics=metrics,
        capabilities=capabilities,
        dimensions=dimensions,
        decoded_frames_examined=sum(
            r.measurements.get("examined_frames", 0) for r in by_kind["vision"]
        )
        if by_kind["vision"]
        else (metric_frames[0] if metric_frames else 0),
        eligible_visual_frames=(
            sum(r.measurements["declared_frames"] for r in by_kind["vision"])
            if all("declared_frames" in r.measurements for r in by_kind["vision"])
            else None
        )
        if by_kind["vision"]
        else (metric_frames[1] if metric_frames else None),
        training_windows_examined=sum(
            r.measurements.get("examined_windows", 0) for r in by_kind["windows"]
        )
        if by_kind["windows"]
        else None,
    )


def report_coverage(episodes: list[GradedEpisode], inventory: Inventory) -> Coverage:
    coverages = [e.coverage for e in episodes if e.coverage is not None]
    return Coverage(
        metrics=merge_rows(r for c in coverages for r in c.metrics),
        capabilities=merge_rows(r for c in coverages for r in c.capabilities),
        dimensions=merge_rows(r for c in coverages for r in c.dimensions),
        inventory_complete=inventory.complete,
        unassessed_episodes=len(inventory.failed) + inventory.unresolved,
        refused_sources=len(inventory.refused_sources),
        decoded_frames_examined=sum(c.decoded_frames_examined for c in coverages),
        eligible_visual_frames=sum(
            c.eligible_visual_frames
            for c in coverages
            if c.eligible_visual_frames is not None
        )
        if any(c.eligible_visual_frames is not None for c in coverages)
        and all(
            c.eligible_visual_frames is not None
            or not any(
                r.key == "visual_quality"
                and r.eligible + r.not_required + r.not_applicable
                for r in c.dimensions
            )
            for c in coverages
        )
        else None,
        training_windows_examined=sum(
            c.training_windows_examined or 0 for c in coverages
        )
        if any(c.training_windows_examined is not None for c in coverages)
        else None,
    )


def coverage_lines(coverage: Coverage | None) -> list[str]:
    """The same strings feed terminal, inspect and HTML; never a blended score."""
    if coverage is None:
        return ["Coverage was not recorded in this report."]
    lines = []
    for r in coverage.capabilities:
        if r.eligible:
            lines.append(
                f"{r.key}: {r.computed}/{r.eligible} required episode checks computed; "
                f"{r.unavailable} unavailable, {r.skipped} skipped, {r.error} errors"
            )
    for r in coverage.dimensions:
        if r.key == "visual_quality" and not coverage.decoded_frames_examined:
            n = r.eligible + r.not_required + r.not_applicable
            lines.append(
                f"Visual quality not evaluated: 0 of {n} discovered "
                "camera stream-episodes "
                "examined; decoded frame coverage unknown."
            )
        else:
            lines.append(
                f"{r.key}: {r.computed}/{r.eligible} {r.unit} computed; "
                f"{r.not_required} not required"
            )
    if coverage.decoded_frames_examined:
        visual_total = coverage.eligible_visual_frames
        lines.append(
            f"Visual frames examined: {coverage.decoded_frames_examined}; "
            "declared eligible frames: "
            f"{visual_total if visual_total is not None else 'unknown'}; "
            "sampled evidence only unless every frame was evaluated."
        )
    if coverage.training_windows_examined is not None:
        lines.append(
            f"Training windows examined: {coverage.training_windows_examined}; "
            "overlapping windows are not independent demonstrations."
        )
    lines.append(
        f"Coverage counts loaded subjects only; {coverage.unassessed_episodes} "
        f"known episodes unassessed, {coverage.refused_sources} refused sources; "
        f"inventory {'complete' if coverage.inventory_complete else 'incomplete'}."
    )
    return lines
