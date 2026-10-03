"""Explicit window sufficiency and unaccepted calibration-study summaries."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Mapping, Sequence
from typing import Any, Literal

# External
from pydantic import Field

# Internal
from kalanos.analysis.models.binding import RequirementsSection
from kalanos.analysis.models.diagnostics import DiagnosticsReport, StrictModel
from kalanos.analysis.models.eligibility import (
    EpisodeEligibility,
    Sufficiency,
    SufficiencyCheck,
    SufficiencyStatus,
)
from kalanos.analysis.models.provenance import Inventory, content_digest


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class StudyObservation(StrictModel):
    """One externally labelled episode and the complete policy's observed decision."""

    episode_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    split: Literal["tuning", "validation"]
    truth: Literal["valid", "fault"]
    decision: Literal["pass", "blocked", "review", "unknown"]
    evidence: str = Field(min_length=1)
    real_fault: bool = False


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def window_sufficiency(
    base: Sufficiency,
    diagnostics: DiagnosticsReport | None,
    decisions: Mapping[str, EpisodeEligibility],
    inventory: Inventory,
    requirements: RequirementsSection,
) -> Sufficiency:
    """Evaluate minimum window counts using passing episodes and known lower bounds."""
    checks = list(base.checks)
    for identifier, minimum in requirements.min_pass_windows.items():
        passing = {k for k, v in decisions.items() if v.status.value == "pass"}
        rows = (
            [
                r
                for r in diagnostics.results
                if r.kind == "windows"
                and r.id == identifier
                and r.episode_id in passing
            ]
            if diagnostics
            else []
        )
        observed = sum(r.measurements.get("counts", {}).get("pass", 0) for r in rows)
        incomplete = (
            not inventory.complete
            or len(rows) != len(passing)
            or any(
                r.availability.value != "computed"
                or r.measurements.get("counts", {}).get("unknown", 0)
                for r in rows
            )
        )
        status = (
            SufficiencyStatus.SUFFICIENT
            if observed >= minimum
            else SufficiencyStatus.UNKNOWN
            if incomplete
            else SufficiencyStatus.INSUFFICIENT
        )
        checks.append(
            SufficiencyCheck(
                requirement=f"min_pass_windows:{identifier}",
                required=minimum,
                observed=observed,
                status=status,
                detail=(
                    "Counts include only contract-passing windows in passing"
                    " episodes; unevaluated windows cannot establish a "
                    "shortfall."
                ),
            )
        )
    states = {c.status.value for c in checks}
    status = (
        SufficiencyStatus.INSUFFICIENT
        if "insufficient" in states
        else SufficiencyStatus.UNKNOWN
        if "unknown" in states or not states
        else SufficiencyStatus.SUFFICIENT
    )
    return Sufficiency(status=status, checks=checks)


def summarize_study(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize held-out outcomes without asserting independence or acceptance."""
    rows = [StudyObservation.model_validate(x) for x in observations]
    if not rows or len({r.episode_id for r in rows}) != len(rows):
        raise ValueError("study must contain unique episode observations")
    tuning = {r.session_id for r in rows if r.split == "tuning"}
    validation = {r.session_id for r in rows if r.split == "validation"}
    if tuning & validation:
        raise ValueError("tuning and validation sessions overlap")
    held = [r for r in rows if r.split == "validation"]
    valid = [r for r in held if r.truth == "valid"]
    fault = [r for r in held if r.truth == "fault"]
    return {
        "status": "draft",
        "can_authorize_blocking": False,
        "input_digest": content_digest([r.model_dump(mode="json") for r in rows]),
        "tuning_sessions": sorted(tuning),
        "validation_sessions": sorted(validation),
        "valid_episodes": len(valid),
        "false_blocks": sum(r.decision == "blocked" for r in valid),
        "fault_episodes": len(fault),
        "detected_faults": sum(r.decision == "blocked" for r in fault),
        "real_fault_episodes": sum(r.real_fault for r in fault),
        "review_episodes": sum(r.decision == "review" for r in held),
        "unknown_episodes": sum(r.decision == "unknown" for r in held),
        "remaining_review": [
            "Verify labels and independent sessions",
            "Validate the complete policy in its physical operating scope",
            "Record final detector, thresholds, binding and scope identities",
            "Review confidence bounds and explicitly accept a separate manifest",
        ],
        "interpretation": (
            "Detection counts blocking decisions only. Review "
            "candidates are not accepted blocks; counts alone cannot"
            " establish detector validity."
        ),
    }
