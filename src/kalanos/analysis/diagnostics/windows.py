"""Explicit training-window evidence without editing or exporting source data."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import bisect
import math
from collections import Counter
from collections.abc import Collection, Sequence
from fractions import Fraction

# Internal
from kalanos.analysis.diagnostics.common import (
    Unavailable,
    axis,
    finite,
    nearest,
    paired_axes,
    result,
    select,
)
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.diagnostics import DiagnosticResult, WindowSpec
from kalanos.analysis.models.domain import Channel, Episode, FramePayload, Stream
from kalanos.analysis.models.scoring import Finding


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_PRIORITY = {"pass": 0, "review": 1, "unknown": 2, "blocked": 3}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def overlap_finding(
    finding: Finding, stream: Stream, rows: Sequence[int], channels: Sequence[Channel]
) -> bool:
    """Whether a consequential finding touches the consumed source rows."""
    if finding.source_path is not None and finding.source_path != str(
        stream.source_path
    ):
        return False
    if finding.source_field is not None and finding.source_field != stream.source_field:
        return False
    if (
        finding.stream is not None
        and finding.source_field is None
        and finding.stream != stream.taxonomy_type
    ):
        return False
    if finding.instance is not None and finding.instance != stream.instance:
        return False
    if finding.source_index is not None:
        channels = [c for c in channels if c.source_index == finding.source_index]
        if not channels:
            return False
    if finding.channel is not None and not any(
        c.name == finding.channel for c in channels
    ):
        return False
    if finding.support.kind.value == "whole_episode":
        return True
    return any(
        interval.start <= row < interval.end_exclusive
        for row in rows
        for interval in finding.support.intervals
    )


def windows(
    episode: Episode,
    spec: WindowSpec,
    findings: Sequence[Finding] = (),
    visual: Sequence[DiagnosticResult] = (),
    pending: Collection[str] = (),
) -> DiagnosticResult:
    """Count complete history/future windows on a declared grid, with bounded probes.

    A finding whose id is in `pending` does not set a window's status yet:
    its id is listed under the window's `pending` key,
    for `resolve_windows` to apply once the finding is calibrated.
    """
    anchor, _ = select(episode, spec.anchor)
    _, _, at = axis(anchor, exact=True)
    rate = Fraction(str(spec.sample_rate_hz))
    max_gap = Fraction(str(spec.max_gap_s))
    n = math.floor(at[-1] * rate) + 1
    length = spec.history_steps + spec.prediction_steps
    total = max(0, (n - length) // spec.stride + 1)
    prepared = []
    for modality in spec.modalities:
        stream, channel = select(episode, modality.selector)
        channels = [channel] if channel else stream.channels
        if stream is anchor:
            rows, positions, times = axis(stream, exact=True)
        else:
            if modality.relation is None:
                raise Unavailable(
                    "each non-anchor modality requires a scoped clock relation"
                )
            _, (rows, positions, times) = paired_axes(
                anchor, stream, modality.relation, exact=True
            )
        if isinstance(stream.payload, FramePayload):
            columns = [channel.name] if channel else [c.name for c in stream.channels]
            if not columns:
                raise Unavailable("required numeric modality has no channels")
            frame = stream.payload.frame.select(columns)
            good = [
                all(
                    v is not None
                    and (finite(v) if isinstance(v, (float, int)) else True)
                    for v in frame.row(i)
                )
                for i in positions
            ]
        elif modality.require_decoded_frames:
            matches = [
                r
                for r in visual
                if r.subject.get("feature") == stream.source_field
                and r.subject.get("instance") == stream.instance
                and r.subject.get("source_path") == str(stream.source_path)
            ]
            sampled = {
                f["source_row"] for r in matches for f in r.evidence.get("frames", [])
            }
            good = [True if row in sampled else None for row in rows]
        else:
            raise Unavailable(
                "non-numeric modalities require_decoded_frames; "
                "timestamps alone cannot validate payloads"
            )
        prepared.append(
            (
                modality,
                stream,
                rows,
                times,
                good,
                channels,
                Fraction(str(modality.max_age_s)),
            )
        )
    budget = min(spec.max_windows, spec.max_probes // max(1, length * len(prepared)))
    counts = Counter(
        {"pass": 0, "blocked": 0, "review": 0, "unknown": max(0, total - budget)}
    )
    records = []
    for index in range(min(total, budget)):
        start = index * spec.stride
        grid = [(start + i) / rate for i in range(length)]
        states, reasons, consumed, waiting = ["pass"], [], [], set()
        for modality, stream, rows, times, good, channels, max_age in prepared:
            chosen = []
            for time in grid:
                j = nearest(times, time, modality.matching)
                if j is None or abs(time - times[j]) > max_age:
                    states.append("blocked")
                    reasons.append("missing sample within declared max_age_s")
                    continue
                chosen.append(j)
                if good[j] is None:
                    states.append("unknown")
                    reasons.append("required frame was not decoded")
                elif not good[j]:
                    states.append("blocked")
                    reasons.append("nonfinite/null required payload")
            a = max(0, bisect.bisect_right(times, grid[0]) - 1)
            b = min(len(times) - 1, bisect.bisect_left(times, grid[-1]))
            if any(
                times[i + 1] - times[i] > max_gap or rows[i + 1] != rows[i] + 1
                for i in range(a, b)
            ):
                states.append("blocked")
                reasons.append("window crosses a source gap")
            used = sorted({rows[j] for j in chosen})
            consumed.append(
                {"selector": modality.selector.model_dump(), "source_rows": used}
            )
            for finding in findings:
                if (
                    finding.episode_id == episode.id
                    and finding.consequence.value in ("block", "review")
                    and overlap_finding(finding, stream, used, channels)
                ):
                    if finding.id in pending:
                        waiting.add(finding.id)
                        continue
                    states.append(
                        "blocked" if finding.consequence.value == "block" else "review"
                    )
                    reasons.append(f"finding:{finding.id or finding.metric_id}")
        status = max(states, key=_PRIORITY.__getitem__)
        counts[status] += 1
        record = {
            "grid_start": start,
            "anchor_grid_index": start + spec.history_steps - 1,
            "grid_end_exclusive": start + length,
            "start_s_relative": float(grid[0]),
            "end_s_relative": float(grid[-1]),
            "status": status,
            "reasons": sorted(set(reasons)),
            "consumed": consumed,
        }
        if waiting:
            record["pending"] = sorted(waiting)
        records.append(record)
    out = result(
        "windows",
        spec,
        episode,
        subject=spec.anchor.model_dump(),
        measurements={
            "candidate_windows": total,
            "examined_windows": len(records),
            "counts": dict(counts),
            "budget_unexamined": max(0, total - budget),
        },
        evidence={
            "specification": spec.model_dump(),
            "windows": records,
            "interpretation": (
                "input-contract windows; overlapping windows are not "
                "independent episodes or an export selection"
            ),
        },
    )
    if total > budget:
        out.availability = Availability.SKIPPED
        out.reason = (
            "window/probe budget reached; unevaluated candidates remain unknown"
        )
    return out


def resolve_windows(
    result: DiagnosticResult, findings: Sequence[Finding]
) -> DiagnosticResult:
    """Apply calibrated findings to the windows that listed them as pending.

    Parameters
    ----------
    result : DiagnosticResult
        A windows result, possibly holding windows with a `pending` key.
    findings : Sequence[Finding]
        The run's calibrated findings.

    Returns
    -------
    DiagnosticResult
        A copy without `pending` keys,
        each such window's status, reasons and count updated as `windows` would have,
        or `result` itself when no window is pending.
    """

    records = result.evidence.get("windows", [])
    if not any("pending" in r for r in records):
        return result
    calibrated = {
        f.id: f
        for f in findings
        if f.episode_id == result.episode_id
        and f.consequence.value in ("block", "review")
    }
    counts = dict(result.measurements["counts"])
    resolved = []
    for record in records:
        if "pending" not in record:
            resolved.append(record)
            continue
        matched = [calibrated[i] for i in record["pending"] if i in calibrated]
        states = [record["status"]] + [
            "blocked" if f.consequence.value == "block" else "review" for f in matched
        ]
        status = max(states, key=_PRIORITY.__getitem__)
        counts[record["status"]] -= 1
        counts[status] += 1
        resolved.append(
            {
                **{k: v for k, v in record.items() if k != "pending"},
                "status": status,
                "reasons": sorted(
                    {*record["reasons"], *(f"finding:{f.id}" for f in matched)}
                ),
            }
        )
    return result.model_copy(
        update={
            "measurements": {**result.measurements, "counts": counts},
            "evidence": {**result.evidence, "windows": resolved},
        }
    )
