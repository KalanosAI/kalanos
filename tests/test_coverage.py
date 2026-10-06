"""Verifies the coverage ledger: what each check examined, the same on every surface."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import pytest
from calibration_helpers import grade_with_test_calibration
from pydantic import ValidationError
from typer.testing import CliRunner

# Internal
from kalanos.analysis.coverage import coverage_lines, metric_rows
from kalanos.analysis.models.binding import Bundle, RequirementsSection
from kalanos.analysis.models.coverage import CoverageRow
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.card import render_terminal
from kalanos.analysis.reporting.render import render_html
from kalanos.analysis.reporting.summary import finding_groups
from kalanos.api import grade, load_report
from kalanos.assets.policy import load_default_policy
from kalanos.cli import app

# Local
from helpers import LEROBOT_FIXTURE
from helpers import channel_ctx as _channel_ctx
from helpers import write_arm as _write_arm


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_torque_coverage_uses_torque_channels_and_counts_report_only():
    grade(LEROBOT_FIXTURE)  # discover the shipped metrics
    result = MetricResult(value=1, unit="N·m", status=MetricStatus.REPORT_ONLY)
    subjects = [("proprio.joint_torque", {"p99_torque": result})] * 700 + [
        ("proprio.joint_position", {})
    ] * 1400
    rows = metric_rows(Level.CHANNEL, subjects)
    r = next(r for r in rows if r.key == "motion.p99_torque")
    assert (r.computed, r.eligible, r.not_applicable) == (700, 700, 1400)


def test_unknown_taxonomy_stays_in_potential_coverage_denominator():
    grade(LEROBOT_FIXTURE)
    r = next(
        r
        for r in metric_rows(Level.CHANNEL, [("unmapped.motor", {})])
        if r.key == "motion.p99_torque"
    )
    assert r.eligible == r.unavailable == 1


def test_coverage_rejects_inconsistent_arithmetic():
    with pytest.raises(ValidationError):
        CoverageRow(key="x", unit="channel_episode", computed=3, eligible=2)


def test_metadata_preserves_skipped_numeric_and_required_vision_counts():
    report = grade(
        LEROBOT_FIXTURE,
        tier=ExecutionTier.METADATA,
        bundle=Bundle(requirements=RequirementsSection(id="vision-imitation-v1")),
    )
    assert report.coverage is not None
    r = next(r for r in report.coverage.metrics if r.key == "integrity.missing_pct")
    assert r.eligible == r.skipped > 0 and r.computed == 0
    vision = next(r for r in report.coverage.capabilities if r.key == "video_quality")
    assert vision.unavailable == vision.eligible == len(report.episodes)
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.unknown == len(report.episodes)
    assert report.readiness is not None
    assert report.readiness.score is None


def test_optional_video_is_explicitly_outside_numeric_scope():
    report = grade(LEROBOT_FIXTURE)
    assert report.coverage is not None
    r = next(r for r in report.coverage.dimensions if r.key == "visual_quality")
    assert r.eligible == 0 and r.not_required > 0 and r.computed == 0


def test_a_depth_stream_is_no_camera_in_the_visual_quality_row():
    """Depth video is not camera footage, so it is neither covered nor missing."""

    report = grade(
        LEROBOT_FIXTURE,
        mapping={"observation.images.up": "extero.depth"},
        bundle=Bundle(
            requirements=RequirementsSection(required_capabilities=["video_quality"])
        ),
    )

    for episode in report.episodes:
        assert episode.coverage is not None
        [row] = [r for r in episode.coverage.dimensions if r.key == "visual_quality"]
        assert (row.eligible, row.not_required) == (0, 0)


def test_missing_required_metric_cannot_be_covered_by_another_check():
    report = grade(
        LEROBOT_FIXTURE,
        bundle=Bundle(
            requirements=RequirementsSection(required_metrics=["vision.not_installed"])
        ),
    )
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.unknown == len(report.episodes)
    assert all(
        e.eligibility is not None
        and any(r.id == "metric:vision.not_installed" for r in e.eligibility.reasons)
        for e in report.episodes
    )


def test_default_review_and_explicit_calibration_propagate_to_every_decision(tmp_path):
    path = tmp_path / "arm.hdf5"
    _write_arm(path, 4, {1})
    report = grade(path)
    assert report.eligibility_counts is not None
    assert (report.eligibility_counts.review, report.eligibility_counts.blocked) == (
        1,
        0,
    )
    assert report.readiness is not None
    assert report.readiness.score is None and report.score.train_ready is None
    assert any(
        f.calibration.get("reason") == "SNR physical noise assessment is unavailable"
        for f in report.findings
    )
    accepted = grade_with_test_calibration(path)
    assert accepted.eligibility_counts is not None
    assert (
        accepted.eligibility_counts.review,
        accepted.eligibility_counts.blocked,
    ) == (0, 1)
    assert accepted.readiness is not None
    assert accepted.readiness.score is not None
    assert all(
        f.calibration.get("accepted")
        for f in accepted.findings
        if f.consequence.value == "block"
    )
    assert (
        Report.model_validate_json(accepted.model_dump_json()).coverage
        == accepted.coverage
    )
    assert {f.id for f in accepted.findings} == {f.id for f in report.findings}
    groups = finding_groups(report)
    assert sum(g["count"] for g in groups) == len(report.findings)
    assert {i for g in groups for i in g["member_ids"]} == {
        f.id for f in report.findings
    }


def test_coverage_summary_is_shared_by_html_and_inspect_and_counted_on_the_card(
    tmp_path,
):
    report = grade(LEROBOT_FIXTURE)
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json())
    terminal = render_terminal(report, width=300)
    html = render_html(report)
    inspected = CliRunner().invoke(app, ["inspect", str(path)])
    assert inspected.exit_code == 0
    for line in coverage_lines(report.coverage):
        assert line in html
        # CLI can wrap; compare collapsed whitespace.
        assert " ".join(line.split()) in " ".join(inspected.stdout.split())
    assert report.coverage is not None
    for row in report.coverage.capabilities:
        if row.eligible:
            assert f"{row.computed}/{row.eligible}" in terminal
    assert (
        CliRunner()
        .invoke(app, ["inspect", str(path), "--episode", "missing"])
        .exit_code
        == 2
    )


def test_metric_error_has_coverage_unknown_and_operational_exit_precedence(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from kalanos.analysis.metrics import registry

    grade(LEROBOT_FIXTURE)

    def broken(ctx):
        raise RuntimeError("synthetic computation failure")

    entries = [
        replace(e, func=broken) if e.name == "missing_pct" else e
        for e in registry._REGISTRY
    ]
    monkeypatch.setattr(registry, "_REGISTRY", entries)
    report = grade(LEROBOT_FIXTURE)
    assert report.run is not None
    assert report.operational_errors and report.run.completion.value == "partial"
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.unknown == len(report.episodes)
    assert report.coverage is not None
    metric = next(
        r for r in report.coverage.metrics if r.key == "integrity.missing_pct"
    )
    assert metric.error == metric.eligible > 0
    output = tmp_path / "failed.json"
    result = CliRunner().invoke(
        app,
        [
            "grade",
            str(LEROBOT_FIXTURE),
            "--fail-on",
            "blocked",
            "--report",
            str(output),
        ],
    )
    assert result.exit_code == 2 and output.exists()
    loaded = load_report(output)
    assert isinstance(loaded, Report)
    assert loaded.operational_errors


def test_payload_error_is_counted_and_not_misreported_as_an_empty_stream():
    from kalanos.analysis.models.report import PayloadStatus
    from kalanos.analysis.reporting.assemble import grade_stream

    class Broken:
        def __len__(self):
            return 100

        def fetch(self):
            raise OSError("synthetic payload read failure")

    stream = _channel_ctx([math.sin(i) for i in range(100)]).stream.stream.model_copy(
        update={"payload": Broken()}
    )
    result, _ = grade_stream(
        stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="test",
        category=None,
    )
    assert result.evaluation.payload == PayloadStatus.ERROR
    assert result.declared_channels and not result.channels
    row = next(r for r in result.coverage if r.key == "integrity.missing_pct")
    assert row.error == row.eligible == len(stream.channels)


def test_coverage_cannot_be_forged_on_report_loading():
    report = grade(LEROBOT_FIXTURE)
    data = report.model_dump(mode="json")
    data["coverage"]["metrics"][0]["eligible"] += 1
    data["coverage"]["metrics"][0]["computed"] += 1
    with pytest.raises(ValidationError, match="reconcile"):
        Report.model_validate(data)


def test_whole_episode_spectral_results_do_not_invent_intervals():
    from kalanos.analysis.metrics.integrity import snr_db

    result = snr_db(
        _channel_ctx([math.sin(i * 0.08) + 0.01 * math.sin(i * 2) for i in range(100)])
    )
    assert result.value is not None
    assert result.support.kind.value == "whole_episode" and not result.support.intervals
