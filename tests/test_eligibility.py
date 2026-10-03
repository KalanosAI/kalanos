"""Verifies eligibility: precedence, readiness, counts, sufficiency and exit codes."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import shutil
from pathlib import Path

# External
import pytest
import yaml
from calibration_helpers import grade_with_test_calibration
from diagnostics_helpers import audit, episode, signal, two_channels
from pydantic import ValidationError
from typer.testing import CliRunner

# Internal
from kalanos.analysis.models.binding import RequirementsSection
from kalanos.analysis.models.eligibility import (
    Consequence,
    EligibilityCounts,
    EligibilityReason,
    EligibilityStatus,
    EpisodeEligibility,
    Readiness,
    ReasonKind,
    SufficiencyStatus,
    worst_status,
)
from kalanos.analysis.models.provenance import FailedEpisode, Inventory
from kalanos.analysis.models.report import Report
from kalanos.analysis.scoring.eligibility import counts_of, readiness_of, sufficiency_of
from kalanos.api import grade
from kalanos.assets.policy import load_policy
from kalanos.cli import app

# Local
from helpers import write_spiked_arm as _write_arm


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _reason(status: EligibilityStatus, ident: str = "r") -> EligibilityReason:
    kind = (
        ReasonKind.FINDING
        if status != EligibilityStatus.UNKNOWN
        else ReasonKind.REQUIREMENT
    )
    consequence = (
        Consequence.BLOCK
        if status == EligibilityStatus.BLOCKED
        else Consequence.REVIEW
        if status == EligibilityStatus.REVIEW
        else None
    )
    return EligibilityReason(
        id=ident, kind=kind, status=status, consequence=consequence
    )


def _counts(**kw) -> EligibilityCounts:
    base = dict(total=0, pass_count=0, blocked=0, review=0, unknown=0)
    base.update(kw)
    return EligibilityCounts.model_validate(base)


def mixed_report():
    """Two passing episodes and one review, all produced by normal assembly."""
    return audit(
        [
            episode(signal(), identifier="a"),
            episode(signal(), identifier="b"),
            two_channels(missing=True),
        ]
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


# Precedence keeps every reason, and the compatibility fields cannot disagree


def test_precedence_is_blocked_over_unknown_over_review_over_pass():
    """Whatever the mix, the worst state wins and nothing is dropped."""

    assert worst_status([]) == EligibilityStatus.PASS
    assert (
        worst_status([EligibilityStatus.REVIEW, EligibilityStatus.UNKNOWN])
        == EligibilityStatus.UNKNOWN
    )
    assert (
        worst_status(
            [
                EligibilityStatus.REVIEW,
                EligibilityStatus.UNKNOWN,
                EligibilityStatus.BLOCKED,
            ]
        )
        == EligibilityStatus.BLOCKED
    )
    decision = EpisodeEligibility(
        status=EligibilityStatus.BLOCKED,
        scope_id="s",
        policy_id="p",
        reasons=[
            _reason(EligibilityStatus.REVIEW, "a"),
            _reason(EligibilityStatus.BLOCKED, "b"),
        ],
    )
    assert {r.id for r in decision.reasons} == {"a", "b"}


def test_a_status_its_reasons_do_not_justify_is_rejected():
    """`pass` with a blocking reason, or `blocked` with none, cannot be written."""

    with pytest.raises(ValueError, match="does not follow"):
        EpisodeEligibility(
            status=EligibilityStatus.PASS,
            scope_id="s",
            policy_id="p",
            reasons=[_reason(EligibilityStatus.BLOCKED)],
        )
    with pytest.raises(ValueError, match="does not follow"):
        EpisodeEligibility(
            status=EligibilityStatus.BLOCKED, scope_id="s", policy_id="p"
        )


def test_compatibility_train_ready_mirrors_eligibility_exactly():
    """true for pass, false for blocked, null for review and unknown."""

    def decide(status: EligibilityStatus) -> bool | None:
        reasons = [] if status == EligibilityStatus.PASS else [_reason(status)]
        return EpisodeEligibility(
            status=status, scope_id="s", policy_id="p", reasons=reasons
        ).compatibility_train_ready

    assert decide(EligibilityStatus.PASS) is True
    assert decide(EligibilityStatus.BLOCKED) is False
    assert decide(EligibilityStatus.REVIEW) is None
    assert decide(EligibilityStatus.UNKNOWN) is None


def test_a_report_refuses_a_blocked_episode_marked_train_ready(tmp_path):
    """The contradiction the legacy reports carry cannot be authored under schema 7."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 4, glitched={1})
    report = grade_with_test_calibration(path)
    blocked = next(e for e in report.episodes if e.score.train_ready is False)
    assert blocked.eligibility is not None
    assert blocked.eligibility.status == EligibilityStatus.BLOCKED

    forged = blocked.model_copy(
        update={"score": blocked.score.model_copy(update={"train_ready": True})}
    )
    payload = report.model_dump(mode="json")
    payload["episodes"] = [
        forged.model_dump(mode="json") if e["id"] == blocked.id else e
        for e in payload["episodes"]
    ]
    with pytest.raises(ValueError, match="contradicts eligibility"):
        Report.model_validate(payload)


def test_every_surface_reads_the_same_decision(tmp_path):
    """JSON, gate and counts agree, because all derive from one eligibility."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 10, glitched={2, 5})
    report = grade_with_test_calibration(path)

    blocked = {
        e.id
        for e in report.episodes
        if e.eligibility and e.eligibility.status == EligibilityStatus.BLOCKED
    }
    assert report.gate is not None
    assert {f.episode_id for f in report.gate.failing_episodes} == blocked
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.blocked == len(blocked) == 2
    assert report.eligibility_counts.pass_count == 8
    assert all(
        e.score.train_ready == e.eligibility.compatibility_train_ready
        for e in report.episodes
        if e.eligibility
    )
    # And it survives a round trip.
    again = Report.model_validate_json(report.model_dump_json())
    assert again.eligibility_counts == report.eligibility_counts


# Prevalence exempts nothing


@pytest.mark.parametrize("share", [0.01, 0.5, 1.0])
def test_a_blocking_finding_never_disappears_as_its_prevalence_grows(tmp_path, share):
    """Inject a spike into 1%, 50% and 100% of episodes: each stays blocked."""

    n = 100
    glitched = set(range(int(n * share)))
    path = tmp_path / "arm.hdf5"
    _write_arm(path, n, glitched=glitched)
    report = grade_with_test_calibration(path)
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.blocked == len(glitched)
    if share == 1.0:
        assert report.readiness is not None and report.readiness.score == 0.0
        assert report.gate is not None and report.gate.dataset_traits


# Readiness is null whenever it cannot be stated


def test_readiness_is_null_with_reasons_when_review_or_unknown_remain():
    """A review or unknown episode makes the index undefined, and says so."""

    counts = _counts(total=3, pass_count=1, review=1, unknown=1)
    r = readiness_of([], {}, counts)
    assert r.score is None
    assert any("review" in reason for reason in r.reasons)
    assert any("unresolved" in reason for reason in r.reasons)


def test_readiness_is_null_for_an_empty_or_incomplete_inventory():
    assert readiness_of([], {}, _counts()).score is None
    incomplete = EligibilityCounts(
        total=2, pass_count=2, blocked=0, review=0, unknown=0, inventory_complete=False
    )
    r = readiness_of([], {}, incomplete)
    assert r.score is None and any("incomplete" in x for x in r.reasons)


def test_readiness_model_refuses_null_without_reasons_and_reasons_with_a_score():
    with pytest.raises(ValueError):
        Readiness(score=None, reasons=[])
    with pytest.raises(ValueError):
        Readiness(score=50.0, reasons=["x"])


def test_counts_must_partition_the_inventory():
    with pytest.raises(ValueError, match="partition"):
        _counts(total=3, pass_count=1)


def test_failed_episodes_count_as_unknown_and_incomplete_inventory_has_no_share():
    decisions = [
        EpisodeEligibility(status=EligibilityStatus.PASS, scope_id="s", policy_id="p")
    ]
    inventory = Inventory(
        loaded=1, failed=[FailedEpisode(id="x", reason="unreadable")], complete=False
    )
    counts = counts_of(decisions, inventory)
    assert (counts.total, counts.pass_count, counts.unknown) == (2, 1, 1)
    assert counts.confirmed_eligible_share is None


# Report counts reconcile with the episodes they summarise


@pytest.mark.parametrize(
    "source,target",
    [("pass_count", "review"), ("review", "pass_count"), ("review", "unknown")],
)
def test_report_rejects_balanced_but_false_episode_status_counts(source, target):
    report = mixed_report()
    data = report.model_dump(mode="json")
    counts = data["eligibility_counts"]
    counts[source] -= 1
    counts[target] += 1
    counts["confirmed_eligible_share"] = counts["pass_count"] / counts["total"]
    with pytest.raises(ValidationError, match="eligibility_counts"):
        Report.model_validate(data)


def test_report_rejects_duplicate_episode_identity():
    data = mixed_report().model_dump(mode="json")
    data["episodes"][1]["id"] = data["episodes"][0]["id"]
    with pytest.raises(ValidationError, match="duplicate episode"):
        Report.model_validate(data)


def test_report_count_reconciliation_keeps_failed_and_unresolved_unknown():
    r = mixed_report()
    inventory = Inventory(
        loaded=3,
        failed=[FailedEpisode(id="failed", reason="read error")],
        unresolved=2,
        complete=False,
    )
    decisions: list[EpisodeEligibility] = []
    for e in r.episodes:
        assert e.eligibility is not None
        decisions.append(e.eligibility)
    counts = counts_of(decisions, inventory)
    data = r.model_dump(mode="json")
    # The fixture exercises the status boundary independently of optional ledgers.
    data["coverage"] = None
    data["inventory"] = inventory.model_dump(mode="json")
    data["eligibility_counts"] = counts.model_dump(mode="json")
    parsed = Report.model_validate(data)
    assert parsed.eligibility_counts is not None
    assert parsed.eligibility_counts.unknown == 3
    assert parsed.eligibility_counts.confirmed_eligible_share is None
    assert Report.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize("share", [0.99, float("nan")])
def test_report_rejects_false_eligible_share(share):
    data = mixed_report().model_dump(mode="json")
    data["eligibility_counts"]["confirmed_eligible_share"] = share
    with pytest.raises(ValidationError, match="share"):
        Report.model_validate(data)


# Sufficiency


def test_a_known_unmet_count_is_insufficient_and_no_requirement_is_unknown():
    req = RequirementsSection(min_pass_episodes=20)
    assert (
        sufficiency_of(req, _counts(total=49, pass_count=1, blocked=48)).status
        == SufficiencyStatus.INSUFFICIENT
    )
    assert (
        sufficiency_of(req, _counts(total=49, pass_count=39, blocked=10)).status
        == SufficiencyStatus.SUFFICIENT
    )
    assert (
        sufficiency_of(RequirementsSection(), _counts(total=1, pass_count=1)).status
        == SufficiencyStatus.UNKNOWN
    )


def test_a_required_family_that_graded_nothing_makes_the_episode_unknown(tmp_path):
    """An unmet requirement is unknown, never a pass."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 3, glitched=set())
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "requirements": {"id": "vision-v1", "required_families": ["vision"]},
            }
        )
    )
    report = grade(path, bundle=bundle)
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.unknown == 3
    assert report.readiness is not None and report.readiness.score is None
    assert all(
        any(r.id == "family:vision" for r in e.eligibility.reasons)
        for e in report.episodes
        if e.eligibility
    )


# Dataset train_ready follows the counts, gate or no gate


def test_a_failed_input_beside_passing_episodes_leaves_dataset_train_ready_null(
    tmp_path,
):
    folder = tmp_path / "mixed"
    shutil.copytree(TINY_V3, folder / "lerobot_v3_tiny")
    (folder / "no_time.csv").write_text("a,b\n1,2\n3,4\n")
    report = grade(folder)
    c = report.eligibility_counts
    assert c is not None and c.pass_count == 2
    # A refused source is not one failed episode: it holds an unknown number
    # of them, so the inventory is incomplete rather than short by one.
    assert report.inventory is not None
    assert report.inventory.refused_sources and not report.inventory.complete
    assert c.inventory_complete is False
    assert report.score.train_ready is None


def test_all_pass_without_a_letter_gate_is_train_ready(tmp_path):
    path = tmp_path / "clean.hdf5"
    _write_arm(path, 3, glitched=set())
    report = grade(path, policy=load_policy(Path("legacy_0_5")))
    assert report.gate is None
    assert report.score.train_ready is True


def test_a_forged_dataset_train_ready_is_refused(tmp_path):
    path = tmp_path / "dirty.hdf5"
    _write_arm(path, 3, glitched={1})
    report = grade_with_test_calibration(path)
    assert report.score.train_ready is False
    payload = report.model_dump(mode="json")
    payload["score"]["train_ready"] = True
    with pytest.raises(ValueError, match="dataset train_ready"):
        Report.model_validate(payload)


# CLI exit codes


def test_exit_codes_follow_the_truth_table(tmp_path):
    runner = CliRunner()
    clean = tmp_path / "clean.hdf5"
    _write_arm(clean, 4, glitched=set())
    dirty = tmp_path / "dirty.hdf5"
    _write_arm(dirty, 4, glitched={1})

    assert runner.invoke(app, ["grade", str(clean)]).exit_code == 0
    assert runner.invoke(app, ["grade", str(dirty)]).exit_code == 0
    # Uncalibrated statistical faults require review; opt in to that gate.
    assert (
        runner.invoke(app, ["grade", str(dirty), "--fail-on", "review"]).exit_code == 1
    )
    # ...and an unknown status is a configuration error, before any grading.
    result = runner.invoke(app, ["grade", str(dirty), "--fail-on", "pass"])
    assert result.exit_code == 2
    # A missing path is operational, not a gate failure.
    assert runner.invoke(app, ["grade", str(tmp_path / "missing.hdf5")]).exit_code == 2
    # A malformed bundle is operational too.
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 99\n")
    assert (
        runner.invoke(app, ["grade", str(clean), "--profile", str(bad)]).exit_code == 2
    )


def test_a_tier_never_changes_the_requirements(tmp_path):
    """metadata tier with the vision scope: unknown, and the gate fails on it."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 2, glitched=set())
    bundle = tmp_path / "b.yaml"
    bundle.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "requirements": {"id": "v", "required_families": ["vision"]},
            }
        )
    )
    runner = CliRunner()
    assert (
        runner.invoke(
            app, ["grade", str(path), "--profile", str(bundle), "--tier", "metadata"]
        ).exit_code
        == 1
    )
    assert (
        runner.invoke(
            app, ["grade", str(path), "--profile", str(bundle), "--fail-on", "blocked"]
        ).exit_code
        == 0
    )
