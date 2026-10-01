"""Execute optional diagnostics once and feed the existing decision pipeline."""

import json
from collections import Counter
from functools import partial
from types import SimpleNamespace

from kalanos.analysis.diagnostics.common import Unavailable
from kalanos.analysis.diagnostics.dataset import run_dataset_diagnostics
from kalanos.analysis.diagnostics.signals import motion, timing, tracking
from kalanos.analysis.diagnostics.vision import vision
from kalanos.analysis.diagnostics.windows import windows
from kalanos.analysis.identities import analysis_digest
from kalanos.analysis.metrics.vision import is_camera_footage
from kalanos.analysis.models.diagnostics import DiagnosticResult, DiagnosticsReport
from kalanos.analysis.models.provenance import content_digest
from kalanos.analysis.models.scoring import Finding


def execute(kind, spec, episode, operation, *, skipped=False, subject=None):
    """Retain prerequisite/budget/error distinctions and reject nonfinite output."""
    base = dict(
        id=spec.id,
        kind=kind,
        episode_id=episode.id if episode else None,
        subject=subject or {},
    )
    if skipped:
        return DiagnosticResult(
            **base,
            availability="skipped",
            reason="metadata tier does not execute payload diagnostics",
        )
    try:
        measured = operation()
        json.dumps(measured.model_dump(), allow_nan=False)
        return measured
    except Unavailable as exc:
        return DiagnosticResult(**base, availability="unavailable", reason=str(exc))
    except Exception as exc:
        return DiagnosticResult(
            **base, availability="error", reason=f"{type(exc).__name__}: {exc}"
        )


def review_findings(results):
    """Convert explicit configured review triggers into ordinary review findings.

    These first-version diagnostics cannot authorize a statistical block. They
    enter the same one-time eligibility calculation as existing findings.
    """
    findings = []
    for r in results:
        if r.consequence != "review" or r.episode_id is None:
            continue
        evidence = {"diagnostic_id": r.id, "measurements": r.measurements, **r.evidence}
        findings.append(
            Finding(
                id=content_digest(
                    {"diagnostic": r.id, "episode": r.episode_id, "subject": r.subject}
                ),
                metric_id=f"diagnostics.{r.kind}.{r.id}",
                family={"tracking": "consistency"}.get(r.kind, r.kind),
                severity="critical",
                value=1,
                unit="review trigger",
                points=0,
                episode_id=r.episode_id,
                subject_level="episode",
                source_path=r.subject.get("source_path"),
                source_field=r.subject.get("feature"),
                source_index=r.subject.get("index"),
                instance=r.subject.get("instance"),
                evidence=evidence,
                support=r.support,
                consequence="review",
                route=None,
                calibration={
                    "accepted": False,
                    "reason": (
                        "diagnostic v1 supports review only; production "
                        "promotion is not enabled"
                    ),
                },
            )
        )
    return findings


def _graded_camera(graded_episode, stream):
    """The graded stream of one episode that `stream` was graded as, or `None`."""
    if graded_episode is None:
        return None
    for graded in graded_episode.streams:
        if (graded.source_field, graded.instance, graded.source_path) == (
            stream.source_field,
            stream.instance,
            str(stream.source_path),
        ):
            return graded
    return None


def episode_diagnostics(episodes, plan, tier, findings, policies, graded):
    """Run signal and visual work before windows, without changing source arrays.

    Parameters
    ----------
    graded : Mapping[str, GradedEpisode]
        Each episode graded, by id; the vision diagnostic reads its camera reads.
    """
    results = {}
    for episode in episodes:
        items = []
        review = policies[episode.id].diagnostic_reviews
        skip = tier.value == "metadata"
        for kind, specs, operation in (
            ("timing", plan.timing, timing),
            ("tracking", plan.tracking, tracking),
            ("motion", plan.motion, motion),
        ):
            for spec in specs:
                selector = spec.channel if kind == "motion" else spec.left
                items.append(
                    execute(
                        kind,
                        spec,
                        episode,
                        partial(operation, episode, spec, review),
                        skipped=skip,
                        subject=selector.model_dump(),
                    )
                )
        if plan.vision:
            for stream in episode.streams:
                if not is_camera_footage(stream.kind.value, stream.taxonomy_type):
                    continue
                subject = {
                    "feature": stream.source_field,
                    "instance": stream.instance,
                    "source_path": str(stream.source_path),
                }
                identifier = "camera_" + content_digest(subject)[:16]
                spec = SimpleNamespace(id=identifier)
                camera = _graded_camera(graded.get(episode.id), stream)
                frames = camera.frames if camera is not None else None
                count = (
                    camera.metrics.get("frame_count_vs_timebase") if camera else None
                )
                sharpness = camera.metrics.get("sharpness_score") if camera else None
                items.append(
                    execute(
                        "vision",
                        spec,
                        episode,
                        partial(
                            vision,
                            episode,
                            stream,
                            frames,
                            identifier,
                            review,
                            segment_frames=count.evidence.get("present")
                            if count is not None
                            else None,
                            reason=sharpness.evidence.get("reason")
                            if frames is None and sharpness is not None
                            else None,
                        ),
                        skipped=skip,
                        subject=subject,
                    )
                )
        extra = review_findings(items)
        for spec in plan.windows:
            items.append(
                execute(
                    "windows",
                    spec,
                    episode,
                    lambda spec=spec, episode=episode, extra=extra, items=items: (
                        windows(
                            episode,
                            spec,
                            [*findings, *extra],
                            [r for r in items if r.kind == "vision"],
                        )
                    ),
                    skipped=skip,
                    subject=spec.anchor.model_dump(),
                )
            )
        results[episode.id] = items
    return results


def finish_diagnostics(episodes, plan, results, decisions, tier):
    """Run dataset-level diagnostics after eligibility, preserving both populations."""

    def execute_dataset(kind, spec, episode, operation):
        """Respect the execution tier at dataset level too."""
        return execute(kind, spec, episode, operation, skipped=tier.value == "metadata")

    dataset = run_dataset_diagnostics(
        episodes,
        plan.cohorts,
        {k: v.status.value for k, v in decisions.items()},
        execute_dataset,
    )
    return DiagnosticsReport(
        plan_digest=content_digest(plan.model_dump(mode="json")),
        implementation_digest=analysis_digest(),
        plan=plan,
        results=[r for rs in results.values() for r in rs] + dataset,
    )


def diagnostic_lines(report):
    """Produce shared, concise terminal/HTML/inspect wording from recorded results."""
    if report is None:
        return []
    states = Counter(r.availability.value for r in report.results)
    return [
        "Diagnostics: "
        + ", ".join(f"{n} {state}" for state, n in sorted(states.items())),
        (
            "Diagnostic thresholds request review only; descriptive "
            "results do not certify training success."
        ),
    ]


def validate_review_plan(plan, policy):
    """Reject review rules that would silently target no configured diagnostic."""
    from kalanos.analysis.models.errors import MappingOverrideError

    review = policy.diagnostic_reviews
    timing_ids = {s.id for s in plan.timing} if plan else set()
    tracking_ids = {s.id for s in plan.tracking} if plan else set()
    unmatched = (set(review.timing_max_unmatched_fraction) - timing_ids) | (
        set(review.tracking_max_abs_error) - tracking_ids
    )
    if unmatched:
        raise MappingOverrideError(
            "diagnostic review policy targets unknown check ids: "
            + ", ".join(sorted(unmatched))
        )
    if (
        review.vision_max_clipped_fraction is not None
        or review.vision_min_blur_score is not None
        or review.review_video_integrity
    ) and (plan is None or not plan.vision):
        raise MappingOverrideError(
            "visual review policy requires an explicit vision plan"
        )
    if review.review_motion_limits and (plan is None or not plan.motion):
        raise MappingOverrideError(
            "motion review policy requires an explicit motion plan"
        )
