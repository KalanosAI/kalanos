"""The schema-7 decision contract, asserted directly.

Each test names the release-plan scenario it covers (T04, T05, ...). They
exercise the models and evaluator on hand-built inputs and the real fixtures,
so a change that lets two surfaces disagree about an episode fails here
before it reaches a user.
"""

# Built-in
import hashlib
from pathlib import Path
from typing import cast

# External
import h5py
import numpy as np
import pytest
import yaml
from calibration_helpers import grade_with_test_calibration
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis.models.binding import (
    BindingOrigin,
    FeatureAssertion,
    RequirementsSection,
    SamePriorityConflict,
    resolve_feature_types,
)
from kalanos.analysis.models.domain import Clock, ClockInfo, ClockOrigin, OriginEvidence
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
from kalanos.analysis.models.legacy import LegacyReport, UnsupportedSchema, load_any
from kalanos.analysis.models.provenance import Inventory
from kalanos.analysis.models.report import Report
from kalanos.analysis.models.scoring import (
    SampleInterval,
    SupportKind,
    TemporalSupport,
)
from kalanos.analysis.scoring.eligibility import (
    counts_of,
    readiness_of,
    sufficiency_of,
)
from kalanos.api import grade
from kalanos.cli import app


LEGACY = Path(__file__).parent / "legacy_reports"
TOWEL_6_4 = LEGACY / "aloha_static_towel-048fef2-6.4.0-trimmed.json"
TINY_6_5 = LEGACY / "lerobot_v3_tiny-6.5.0.json"
TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"


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


def _write_arm(path: Path, n_episodes: int, glitched: set[int]) -> None:
    """The same glitched-arm HDF5 `test_gate.py` writes: spikes in `glitched`."""

    rng = np.random.default_rng(0)
    with h5py.File(str(path), "w") as store:
        for index in range(n_episodes):
            group = store.create_group(f"data/demo_{index}")
            t = np.arange(200) / 50.0
            signal = np.sin(t) + rng.normal(0, 0.01, 200)
            if index in glitched:
                signal[50:60] += 40.0
            group.create_dataset("joint_pos", data=signal)
            group.create_dataset("timestamp", data=t)


# ── T04: precedence, every reason kept, compatibility fields cannot disagree ──


def test_t04_precedence_is_blocked_over_unknown_over_review_over_pass():
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


def test_t04_a_status_its_reasons_do_not_justify_is_rejected():
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


def test_t04_compatibility_train_ready_mirrors_eligibility_exactly():
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


def test_t04_a_report_refuses_a_blocked_episode_marked_train_ready(tmp_path):
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


def test_t04_every_surface_reads_the_same_decision(tmp_path):
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


# ── T05: prevalence exempts nothing ──


@pytest.mark.parametrize("share", [0.01, 0.5, 1.0])
def test_t05_a_blocking_finding_never_disappears_as_its_prevalence_grows(
    tmp_path, share
):
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


# ── Readiness null rules ──


def _counts(**kw) -> EligibilityCounts:
    base = dict(total=0, pass_count=0, blocked=0, review=0, unknown=0)
    base.update(kw)
    return EligibilityCounts(**base)


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
        loaded=1, failed=[{"id": "x", "reason": "unreadable"}], complete=False
    )
    counts = counts_of(decisions, inventory)
    assert (counts.total, counts.pass_count, counts.unknown) == (2, 1, 1)
    assert counts.confirmed_eligible_share is None


# ── T13: sufficiency ──


def test_t13_a_known_unmet_count_is_insufficient_and_no_requirement_is_unknown():
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


# ── Binding precedence and conflicts ──


def test_binding_precedence_is_argument_file_bundle_sidecar_and_conflicts_are_kept():
    resolved = resolve_feature_types(
        [
            FeatureAssertion(
                feature="f", taxonomy_type="sidecar", origin=BindingOrigin.SIDECAR
            ),
            FeatureAssertion(
                feature="f", taxonomy_type="bundle", origin=BindingOrigin.BUNDLE
            ),
            FeatureAssertion(
                feature="f", taxonomy_type="file", origin=BindingOrigin.FILE
            ),
            FeatureAssertion(
                feature="f", taxonomy_type="arg", origin=BindingOrigin.ARGUMENT
            ),
            FeatureAssertion(
                feature="g", taxonomy_type="same", origin=BindingOrigin.BUNDLE
            ),
            FeatureAssertion(
                feature="g", taxonomy_type="same", origin=BindingOrigin.SIDECAR
            ),
        ]
    )
    assert resolved.types["f"].taxonomy_type == "arg"
    [conflict] = resolved.conflicts
    assert conflict.feature == "f"
    assert [d.origin for d in conflict.displaced] == [
        BindingOrigin.FILE,
        BindingOrigin.BUNDLE,
        BindingOrigin.SIDECAR,
    ]
    # Agreement is not a conflict.
    assert resolved.types["g"].taxonomy_type == "same"


def test_two_assertions_at_one_priority_that_disagree_are_a_configuration_error():
    with pytest.raises(SamePriorityConflict):
        resolve_feature_types(
            [
                FeatureAssertion(
                    feature="f", taxonomy_type="a", origin=BindingOrigin.FILE
                ),
                FeatureAssertion(
                    feature="f", taxonomy_type="b", origin=BindingOrigin.FILE
                ),
            ]
        )


def test_a_bundle_sets_scope_and_requirements_and_a_sidecar_cannot(tmp_path):
    """`--profile` decides the scope; a sidecar only asserts mappings."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 5, glitched=set())
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "requirements": {"id": "strict-v1", "min_pass_episodes": 10},
                "policy": {"id": "default-decisions-v1"},
                "execution": {"tier": "metadata"},
            }
        )
    )
    report = grade(path, bundle=bundle)
    assert report.scope is not None
    assert report.scope.requirements_id == "strict-v1"
    assert report.scope.tier.value == "metadata"
    assert report.sufficiency is not None
    assert report.sufficiency.status == SufficiencyStatus.INSUFFICIENT
    assert report.run is not None and report.run.requirements is not None
    assert report.run.requirements.id == "strict-v1"
    assert report.run.requirements.digest


def test_a_required_family_that_graded_nothing_makes_the_episode_unknown(tmp_path):
    """T07 in miniature: an unmet requirement is unknown, never a pass."""

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


# ── Clocks and temporal support ──


def test_a_legacy_reconstructed_clock_migrates_as_inferred_generation():
    info = ClockInfo.from_legacy(Clock.RECONSTRUCTED)
    assert info.origin == ClockOrigin.GENERATED
    assert info.origin_evidence == OriginEvidence.INFERRED
    assert not info.certifies_acquisition
    assert ClockInfo(
        origin=ClockOrigin.CAPTURE, origin_evidence=OriginEvidence.PRODUCER
    ).certifies_acquisition
    assert not ClockInfo(
        origin=ClockOrigin.CAPTURE, origin_evidence=OriginEvidence.INFERRED
    ).certifies_acquisition


def test_publication_and_log_origins_are_distinct_and_never_capture():
    assert ClockOrigin.PUBLISH.compatibility_clock == Clock.LOG
    assert ClockOrigin.PRESENTATION.compatibility_clock == Clock.UNKNOWN
    assert ClockOrigin.GENERATED.compatibility_clock == Clock.RECONSTRUCTED


def test_whole_episode_support_carries_no_intervals_and_intervals_need_an_index_space():
    assert TemporalSupport().kind == SupportKind.WHOLE_EPISODE
    with pytest.raises(ValueError):
        TemporalSupport(
            kind=SupportKind.WHOLE_EPISODE,
            intervals=[SampleInterval(start=0, end_exclusive=1)],
        )
    with pytest.raises(ValueError):
        TemporalSupport(
            kind=SupportKind.INTERVALS,
            intervals=[SampleInterval(start=0, end_exclusive=1)],
        )
    ok = TemporalSupport(
        kind=SupportKind.INTERVALS,
        index_space="joint_pos",
        intervals=[
            SampleInterval(
                start=50, end_exclusive=60, support_start=45, support_end_exclusive=65
            )
        ],
    )
    assert ok.intervals[0].support_start == 45
    with pytest.raises(ValueError):
        SampleInterval(start=5, end_exclusive=3)
    with pytest.raises(ValueError):
        SampleInterval(start=5, end_exclusive=8, support_start=6)


# ── T12: legacy reports load unchanged ──


def test_t12_a_schema_6_4_report_loads_losslessly_and_names_its_contradictions():
    loaded = load_any(UPath(TOWEL_6_4))
    assert isinstance(loaded, LegacyReport)
    assert loaded.schema_version == "6.4.0"
    assert loaded.sha256 == hashlib.sha256(TOWEL_6_4.read_bytes()).hexdigest()
    # Values untouched: the original 50-episode readiness under its own formula.
    assert loaded.summary.legacy_readiness == pytest.approx(75.6127, abs=1e-3)
    assert loaded.summary.n_episodes == 4
    assert loaded.summary.n_gate_failing == 2
    assert loaded.summary.n_score_train_ready == 4
    assert {c.episode_id.rsplit("_", 1)[-1] for c in loaded.contradictions} == {
        "000000",
        "000003",
    }
    assert "episodes[].eligibility" in loaded.unknown
    assert (
        loaded.data["gate"]["train_ready_after_pruning"] is True
    )  # preserved, not repaired


def test_t12_a_schema_6_5_branch_report_loads_with_its_mapping_fields():
    loaded = load_any(UPath(TINY_6_5))
    assert isinstance(loaded, LegacyReport)
    assert loaded.schema_version == "6.5.0"
    assert "mapping_overrides" in loaded.data
    assert loaded.contradictions == []
    assert loaded.summary.n_score_train_ready == 2


def test_t12_a_current_report_is_not_legacy(tmp_path):
    report = grade(TINY_V3)
    out = tmp_path / "r.json"
    out.write_text(report.model_dump_json())
    assert isinstance(load_any(UPath(out)), Report)
    with pytest.raises(UnsupportedSchema):
        from kalanos.analysis.models.legacy import load_legacy

        load_legacy(out.read_bytes())


# ── T14: CLI exit codes ──


def test_t14_exit_codes_follow_the_truth_table(tmp_path):
    runner = CliRunner()
    clean = tmp_path / "clean.hdf5"
    _write_arm(clean, 4, glitched=set())
    dirty = tmp_path / "dirty.hdf5"
    _write_arm(dirty, 4, glitched={1})

    assert runner.invoke(app, ["grade", str(clean)]).exit_code == 0
    assert runner.invoke(app, ["grade", str(dirty)]).exit_code == 0
    # Uncalibrated statistical faults now require review; opt in to that gate.
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


def test_t14_a_tier_never_changes_the_requirements(tmp_path):
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


def test_inspect_reads_current_and_legacy_reports(tmp_path):
    runner = CliRunner()
    current = tmp_path / "r.json"
    current.write_text(grade(TINY_V3).model_dump_json())
    out = runner.invoke(app, ["inspect", str(current)])
    assert out.exit_code == 0 and "2/2 pass" in out.stdout
    out = runner.invoke(app, ["inspect", str(TOWEL_6_4)])
    assert out.exit_code == 0 and "2 contradiction" in out.stdout


def test_producer_and_run_are_recorded_and_config_digests_are_stable(tmp_path):
    a = grade(TINY_V3)
    b = grade(TINY_V3)
    assert a.producer is not None and a.producer.version
    assert a.run is not None and b.run is not None
    assert a.run.id != b.run.id
    assert a.run.policy == b.run.policy
    assert a.run.requirements == b.run.requirements
    assert (
        a.run.source.complete is False and a.run.source.digest is None
    )  # not hashed yet: honest
    assert cast(str, a.run.tier.value) == "standard"
