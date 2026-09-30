"""Build and merge coverage from actual result states, not policy grades."""

import math
from collections import Counter, defaultdict

from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.models.coverage import Availability, Coverage, CoverageRow
from kalanos.analysis.models.metrics import Level, MetricStatus


def state_of(result):
    if result.availability is not None and result.availability != Availability.COMPUTED:
        return result.availability
    if (
        result.value is not None
        and math.isfinite(result.value)
        and result.status != MetricStatus.NOT_APPLICABLE
    ):
        return Availability.COMPUTED
    return Availability.UNAVAILABLE


def row(key, unit, states):
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


def metric_rows(level, subjects, *, skipped=None, skip_reason=None):
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


def stream_rows(stream, graded, metrics, evaluation):
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


def merge_rows(rows):
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


def episode_coverage(episode, requirements):
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
        computed = {"numeric": numeric_ok, "acquisition_timing": acquisition_ok}.get(
            key, False
        )
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
    cameras = [s for s in episode.streams if s.kind in ("video", "image")]
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
    return Coverage(metrics=metrics, capabilities=capabilities, dimensions=dimensions)


def report_coverage(episodes, inventory):
    coverages = [e.coverage for e in episodes if e.coverage is not None]
    return Coverage(
        metrics=merge_rows(r for c in coverages for r in c.metrics),
        capabilities=merge_rows(r for c in coverages for r in c.capabilities),
        dimensions=merge_rows(r for c in coverages for r in c.dimensions),
        inventory_complete=inventory.complete,
        unassessed_episodes=len(inventory.failed) + inventory.unresolved,
        refused_sources=len(inventory.refused_sources),
    )


def coverage_lines(coverage):
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
        if r.key == "visual_quality":
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
    lines.append(
        f"Coverage counts loaded subjects only; {coverage.unassessed_episodes} "
        f"known episodes unassessed, {coverage.refused_sources} refused sources; "
        f"inventory {'complete' if coverage.inventory_complete else 'incomplete'}."
    )
    return lines
