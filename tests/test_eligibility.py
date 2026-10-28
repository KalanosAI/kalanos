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
from diagnostics_helpers import audit, episode, signal, two_channels, window_spec
from pydantic import ValidationError
from typer.testing import CliRunner

# Internal
from kalanos.analysis.coverage import report_coverage
from kalanos.analysis.models.binding import RequirementsSection
from kalanos.analysis.models.diagnostics import DiagnosticPlan
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
from kalanos.analysis.models.metrics import Level
from kalanos.analysis.models.provenance import FailedEpisode, Inventory
from kalanos.analysis.models.report import (
    AgreementRow,
    DuplicateEpisodes,
    GradedEpisode,
    Report,
    check_agreement,
)
from kalanos.analysis.models.scoring import MetricRollup, ScoreResult
from kalanos.analysis.scoring.eligibility import (
    counts_of,
    readiness_of,
    requirement_reasons,
    sufficiency_of,
)
from kalanos.api import grade
from kalanos.cli import app

# Local
from helpers import no_gate_policy
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


def test_a_required_family_that_graded_only_a_non_core_metric_is_not_unknown():
    """A family with a graded result counts as graded, even with no family score."""

    episode = GradedEpisode(
        id="e0",
        adapter="csv",
        adapter_confidence=1.0,
        score=ScoreResult(
            level=Level.EPISODE,
            score=None,
            grade=None,
            train_ready=None,
            n_contributing=0,
            metrics={
                "integrity.snr_db": MetricRollup(
                    level=Level.CHANNEL, points=40.0, n_graded=1, n_flagged=1
                )
            },
        ),
    )
    requirements = RequirementsSection(required_families=["integrity", "vision"])

    reasons = {r.id for r in requirement_reasons(episode, requirements)}

    assert "family:integrity" not in reasons
    assert "family:vision" in reasons


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
    report = grade(path, policy=no_gate_policy(tmp_path))
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


# One agreement check for both paths


def _rows(report: Report) -> list[AgreementRow]:
    """What `check_agreement` reads of a report's episodes.

    Parameters
    ----------
    report : Report
        A graded report.

    Returns
    -------
    list[AgreementRow]
        One row per episode, in report order.
    """

    return [
        AgreementRow(
            id=e.id, eligibility=e.eligibility, train_ready=e.score.train_ready
        )
        for e in report.episodes
    ]


def _ledgers(report: Report) -> list:
    """Each episode's coverage ledger, in report order."""

    return [e.coverage for e in report.episodes]


@pytest.fixture(scope="module")
def decided(tmp_path_factory) -> Report:
    """A graded report with a blocked episode, a gate and finding ids.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Where the recording is written.

    Returns
    -------
    Report
        The graded report.
    """

    path = tmp_path_factory.mktemp("agreement") / "arm.hdf5"
    _write_arm(path, 6, glitched={1})
    report = grade_with_test_calibration(path)
    assert report.gate is not None and report.gate.failing_episodes
    assert report.eligibility_counts is not None and report.inventory is not None
    assert report.coverage is not None and report.findings
    assert (
        report.inventory.complete and report.episodes[0].score.train_ready is not None
    )
    return report


@pytest.fixture(scope="module")
def diagnosed() -> Report:
    """A report holding diagnostic results.

    Returns
    -------
    Report
        The assembled report.
    """

    report = audit(
        [episode(signal("command"), signal())],
        DiagnosticPlan(windows=[window_spec()]),
    )
    assert report.diagnostics is not None and report.diagnostics.results
    return report


def _with_counts(report: Report, **update) -> Report:
    """A copy of `report` with its eligibility counts updated."""

    assert report.eligibility_counts is not None
    counts = report.eligibility_counts.model_copy(update=update)
    return report.model_copy(update={"eligibility_counts": counts})


def _with(report: Report, **update) -> tuple[Report, list[AgreementRow], list]:
    """A copy of `report` with fields updated, and its unchanged rows and ledgers."""

    return report.model_copy(update=update), _rows(report), _ledgers(report)


def _first_row(report: Report, **update) -> tuple[Report, list[AgreementRow], list]:
    """`report`, with its first row updated."""

    rows = _rows(report)
    return report, [rows[0].model_copy(update=update), *rows[1:]], _ledgers(report)


def _blocked(report: Report) -> tuple[Report, list[AgreementRow], list]:
    """`report`, with its first passing episode's row blocked."""

    rows = _rows(report)
    k = next(k for k, r in enumerate(rows) if r.train_ready is True)
    blocked = EpisodeEligibility(
        status=EligibilityStatus.BLOCKED,
        scope_id="s",
        policy_id="p",
        reasons=[_reason(EligibilityStatus.BLOCKED)],
    )
    rows[k] = rows[k].model_copy(update={"eligibility": blocked, "train_ready": False})
    return report, rows, _ledgers(report)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda r: (r, [*_rows(r), _rows(r)[0]], _ledgers(r)),
            "duplicate episode ids",
        ),
        (
            lambda r: _with(r, duplicates=DuplicateEpisodes(exact=[[0, 99]])),
            "index outside episodes",
        ),
        (
            lambda r: (r, _rows(r), [None, *_ledgers(r)[1:]]),
            "requires episode ledgers",
        ),
        (
            lambda r: _with(
                r, coverage=r.coverage.model_copy(update={"unassessed_episodes": 7})
            ),
            "does not reconcile",
        ),
        (
            lambda r: _with(r, findings=[r.findings[0], r.findings[0]]),
            "duplicate finding ids",
        ),
        (lambda r: _first_row(r, eligibility=None), "has no eligibility"),
        (
            lambda r: _first_row(r, train_ready=not _rows(r)[0].train_ready),
            "contradicts eligibility",
        ),
        (_blocked, "not the set of blocked"),
        (
            lambda r: (
                _with_counts(r, total=r.eligibility_counts.total + 1),
                _rows(r),
                _ledgers(r),
            ),
            "does not cover the inventory",
        ),
        (
            lambda r: (
                _with_counts(
                    r,
                    pass_count=r.eligibility_counts.pass_count + 1,
                    blocked=r.eligibility_counts.blocked - 1,
                ),
                _rows(r),
                _ledgers(r),
            ),
            "disagrees with episode decisions",
        ),
        (
            lambda r: (
                _with_counts(r, confirmed_eligible_share=0.123),
                _rows(r),
                _ledgers(r),
            ),
            "confirmed_eligible_share disagrees",
        ),
        (
            lambda r: (
                _with_counts(
                    r, inventory_complete=False, confirmed_eligible_share=None
                ),
                _rows(r),
                _ledgers(r),
            ),
            "disagree on completeness",
        ),
        (
            lambda r: _with(
                r,
                score=r.score.model_copy(
                    update={"train_ready": not r.score.train_ready}
                ),
            ),
            "contradicts the eligibility counts",
        ),
    ],
    ids=[
        "ids",
        "duplicates",
        "ledger",
        "coverage",
        "findings",
        "eligibility",
        "train-ready",
        "gate",
        "total",
        "tally",
        "share",
        "completeness",
        "dataset-train-ready",
    ],
)
def test_check_agreement_refuses_each_disagreement(decided, mutate, message):
    report, rows, ledgers = mutate(decided)

    check_agreement(decided, _rows(decided), _ledgers(decided))
    with pytest.raises(ValueError, match=message):
        check_agreement(report, rows, ledgers)


def test_check_agreement_reads_merged_ledgers_as_the_episodes_ones(decided):
    merged = report_coverage(_ledgers(decided), Inventory())

    check_agreement(decided, _rows(decided), [merged])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d.model_copy(update={"plan_digest": "bad"}),
            "plan digest disagrees",
        ),
        (
            lambda d: d.model_copy(
                update={
                    "results": [
                        d.results[0].model_copy(update={"episode_id": "elsewhere"})
                    ]
                }
            ),
            "names an episode not in the report",
        ),
        (
            lambda d: d.model_copy(update={"results": [d.results[0], d.results[0]]}),
            "duplicate diagnostic result addresses",
        ),
    ],
    ids=["digest", "episode", "addresses"],
)
def test_check_agreement_refuses_disagreeing_diagnostics(diagnosed, mutate, message):
    assert diagnosed.diagnostics is not None
    report = diagnosed.model_copy(update={"diagnostics": mutate(diagnosed.diagnostics)})

    check_agreement(diagnosed, _rows(diagnosed), _ledgers(diagnosed))
    with pytest.raises(ValueError, match=message):
        check_agreement(report, _rows(diagnosed), _ledgers(diagnosed))
