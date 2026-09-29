"""Decide eligibility once, after every metric has run, and derive the rest from it.

The gate, the dataset counts, the readiness index, the CI exit code and the
compatibility `train_ready` boolean all read what this module writes. No other
code path may answer "is this episode usable?".
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence

# Internal
from kalanos.analysis.models.binding import RequirementsSection
from kalanos.analysis.models.domain import UNMAPPED_TAXONOMY_PREFIX
from kalanos.analysis.models.eligibility import (
    Consequence,
    EligibilityCounts,
    EligibilityReason,
    EligibilityStatus,
    EpisodeEligibility,
    Readiness,
    ReasonKind,
    Sufficiency,
    SufficiencyCheck,
    SufficiencyStatus,
    worst_status,
)
from kalanos.analysis.models.provenance import Inventory
from kalanos.analysis.models.report import GradedEpisode
from kalanos.analysis.models.scoring import Finding


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def finding_key(finding: Finding) -> str:
    """Name a finding by where it sits and what fired: `stream[/channel].metric`."""

    where = finding.stream or "episode"
    if finding.instance:
        where = f"{where}[{finding.instance}]"
    if finding.channel:
        where = f"{where}/{finding.channel}"
    return f"{where}.{finding.metric_id}"


def _finding_reasons(findings: Iterable[Finding]) -> list[EligibilityReason]:
    """One reason per finding with a consequence; report-only findings say nothing."""

    reasons = []
    for finding in findings:
        match finding.consequence:
            case Consequence.BLOCK:
                status = EligibilityStatus.BLOCKED
            case Consequence.REVIEW:
                status = EligibilityStatus.REVIEW
            case _:
                continue
        reasons.append(
            EligibilityReason(
                id=finding_key(finding),
                kind=ReasonKind.FINDING,
                status=status,
                consequence=finding.consequence,
                route=finding.route,
                detail=(
                    f"{finding.metric_id} {finding.severity.value}: "
                    f"{finding.value}{' ' + finding.unit if finding.unit else ''}"
                ),
            )
        )
    return reasons


def _requirement_reasons(
    episode: GradedEpisode, requirements: RequirementsSection
) -> list[EligibilityReason]:
    """Unknown reasons: a required family graded nothing, or a binding is unresolved."""

    reasons = []
    if episode.score.score is None:
        reasons.append(
            EligibilityReason(
                id="graded_result",
                kind=ReasonKind.REQUIREMENT,
                status=EligibilityStatus.UNKNOWN,
                detail="no metric produced a graded result for this episode",
            )
        )
    for family in requirements.required_families:
        if family not in episode.score.families:
            reasons.append(
                EligibilityReason(
                    id=f"family:{family}",
                    kind=ReasonKind.REQUIREMENT,
                    status=EligibilityStatus.UNKNOWN,
                    detail=f"required family {family!r} graded no result",
                )
            )
    if requirements.require_resolved_bindings:
        for stream in episode.streams:
            if stream.taxonomy_type.startswith(UNMAPPED_TAXONOMY_PREFIX):
                reasons.append(
                    EligibilityReason(
                        id=f"binding:{stream.taxonomy_type}",
                        kind=ReasonKind.BINDING,
                        status=EligibilityStatus.UNKNOWN,
                        detail=(
                            f"stream {stream.taxonomy_type!r} has no resolved binding"
                        ),
                    )
                )
    return reasons


def eligibility_of(
    episode: GradedEpisode,
    findings: Iterable[Finding],
    *,
    requirements: RequirementsSection,
    policy_id: str,
) -> EpisodeEligibility:
    """Decide one episode's eligibility from its findings and the scope's requirements.

    Parameters
    ----------
    episode : GradedEpisode
        The episode, fully graded.
    findings : Iterable[Finding]
        This episode's findings only.
    requirements : RequirementsSection
        What a pass needs under this scope.
    policy_id : str
        The decision policy that assigned the findings' consequences.

    Returns
    -------
    EpisodeEligibility
        The status by precedence over every reason, all reasons kept.
    """

    reasons = _finding_reasons(findings) + _requirement_reasons(episode, requirements)
    return EpisodeEligibility(
        status=worst_status([r.status for r in reasons]),
        scope_id=requirements.id,
        policy_id=policy_id,
        reasons=reasons,
    )


def decide_all(
    episodes: Sequence[GradedEpisode],
    findings: Sequence[Finding],
    *,
    requirements: RequirementsSection,
    policy_id: str,
) -> dict[str, EpisodeEligibility]:
    """Decide every graded episode, keyed by episode id."""

    by_episode: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_episode[finding.episode_id].append(finding)
    return {
        episode.id: eligibility_of(
            episode,
            by_episode.get(episode.id, []),
            requirements=requirements,
            policy_id=policy_id,
        )
        for episode in episodes
    }


def counts_of(
    decisions: Iterable[EpisodeEligibility], inventory: Inventory
) -> EligibilityCounts:
    """Partition the known inventory: decided episodes plus failed ones as unknown."""

    tally = {status: 0 for status in EligibilityStatus}
    for decision in decisions:
        tally[decision.status] += 1
    tally[EligibilityStatus.UNKNOWN] += len(inventory.failed)
    total = sum(tally.values())
    share = (
        tally[EligibilityStatus.PASS] / total if total and inventory.complete else None
    )
    return EligibilityCounts(
        total=total,
        pass_count=tally[EligibilityStatus.PASS],
        blocked=tally[EligibilityStatus.BLOCKED],
        review=tally[EligibilityStatus.REVIEW],
        unknown=tally[EligibilityStatus.UNKNOWN],
        inventory_complete=inventory.complete,
        confirmed_eligible_share=share,
    )


def readiness_of(
    episodes: Sequence[GradedEpisode],
    decisions: dict[str, EpisodeEligibility],
    counts: EligibilityCounts,
) -> Readiness:
    """The readiness index, or `None` with the reasons it is undefined.

    Defined only when the inventory is complete and non-empty, every episode
    is pass or blocked, and every pass has a quality score. A known all-blocked
    inventory scores 0; a fully passing one scores its mean quality.
    """

    reasons = []
    if not counts.inventory_complete:
        reasons.append(
            "inventory incomplete: whole-dataset fractions are not published"
        )
    if counts.total == 0:
        reasons.append("no episodes in the inventory")
    if counts.review:
        reasons.append(f"{counts.review} episode(s) require review")
    if counts.unknown:
        reasons.append(f"{counts.unknown} episode(s) have unresolved required evidence")
    passing = [
        episode.score.score
        for episode in episodes
        if decisions[episode.id].status == EligibilityStatus.PASS
    ]
    if any(score is None for score in passing):
        reasons.append("a passing episode has no quality score")
    scored = [score for score in passing if score is not None]
    passing_quality = statistics.fmean(scored) if scored else None
    if reasons:
        return Readiness(score=None, reasons=reasons, passing_quality=passing_quality)
    return Readiness(
        score=sum(scored) / counts.total, reasons=[], passing_quality=passing_quality
    )


def sufficiency_of(
    requirements: RequirementsSection, counts: EligibilityCounts
) -> Sufficiency:
    """Evaluate the explicit dataset-level requirements the scope declares.

    Only count-based requirements are evaluable in 0.7. A window or diversity
    requirement, once declared, will report unknown until its runner exists;
    it never reports sufficient by omission.
    """

    checks: list[SufficiencyCheck] = []
    if requirements.min_pass_episodes is not None:
        required = requirements.min_pass_episodes
        observed = counts.pass_count
        if not counts.inventory_complete and observed < required:
            status = SufficiencyStatus.UNKNOWN
            detail = "inventory incomplete; more episodes may still pass"
        elif observed >= required:
            status = SufficiencyStatus.SUFFICIENT
            detail = (
                f"{observed} passing episodes meet the declared minimum of {required}"
            )
        else:
            status = SufficiencyStatus.INSUFFICIENT
            detail = (
                f"{observed} passing episodes fall short of the declared "
                f"minimum of {required}"
            )
        checks.append(
            SufficiencyCheck(
                requirement="min_pass_episodes",
                required=float(required),
                observed=float(observed),
                status=status,
                detail=detail,
            )
        )
    if not checks:
        return Sufficiency(status=SufficiencyStatus.UNKNOWN, checks=[])
    statuses = {check.status for check in checks}
    if SufficiencyStatus.INSUFFICIENT in statuses:
        overall = SufficiencyStatus.INSUFFICIENT
    elif SufficiencyStatus.UNKNOWN in statuses:
        overall = SufficiencyStatus.UNKNOWN
    else:
        overall = SufficiencyStatus.SUFFICIENT
    return Sufficiency(status=overall, checks=checks)


__all__ = [
    "counts_of",
    "decide_all",
    "eligibility_of",
    "finding_key",
    "readiness_of",
    "sufficiency_of",
]
