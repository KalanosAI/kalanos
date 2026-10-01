"""R07-05–07 acceptance: coverage, evidence, surfaces and fail-closed promotion."""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from calibration_helpers import grade_with_test_calibration, synthetic_manifest
from pydantic import ValidationError
from test_gate import _write_arm
from test_integrity import _channel_ctx
from typer.testing import CliRunner

from kalanos.analysis.calibration import binomial_upper, evaluate
from kalanos.analysis.coverage import coverage_lines, metric_rows
from kalanos.analysis.metrics.integrity import flatline_pct, spike_pct
from kalanos.analysis.models.binding import Bundle, RequirementsSection
from kalanos.analysis.models.coverage import CoverageRow
from kalanos.analysis.models.domain import SourceOrder
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.provenance import ExecutionTier, HashScope
from kalanos.analysis.models.report import Report
from kalanos.analysis.models.support import SampleInterval
from kalanos.analysis.reporting.card import render_terminal
from kalanos.analysis.reporting.render import render_html
from kalanos.analysis.reporting.summary import finding_groups
from kalanos.api import compare, grade, load_report
from kalanos.assets.policy import _parse_policy, load_default_policy
from kalanos.cli import app

from helpers import LEROBOT_FIXTURE


def _context():
    return dict(
        metric_id="integrity.spike_pct",
        taxonomy_type="proprio.joint_position",
        detector_digest="1" * 64,
        thresholds_digest="2" * 64,
        binding_digest="3" * 64,
        scope_digest="4" * 64,
    )


def _policy(manifest):
    return load_default_policy().model_copy(
        update={"calibration_manifests": [manifest]}
    )


@pytest.mark.parametrize(
    "key",
    [
        "metric_id",
        "taxonomy_type",
        "detector_digest",
        "thresholds_digest",
        "binding_digest",
        "scope_digest",
    ],
)
def test_every_manifest_identity_must_match(key):
    context = _context()
    manifest = synthetic_manifest(context)
    changed = dict(context)
    changed[key] = (
        "motion.p99_torque"
        if key == "metric_id"
        else "proprio.joint_torque"
        if key == "taxonomy_type"
        else "b" * 64
    )
    assert not evaluate(_policy(manifest), changed)["accepted"]
    assert evaluate(_policy(manifest), context)["accepted"]


@pytest.mark.parametrize(
    "change",
    [
        {"status": "draft"},
        {"status": "revoked"},
        {"accepted_by": None},
        {"accepted_at": None},
        {"accepted_at": datetime.now(timezone.utc) + timedelta(days=1)},
        {"valid_until": datetime(2021, 1, 1, tzinfo=timezone.utc)},
        {"independent_episodes": False},
        {"real_fault_validation": False},
        {"combined_policy_validated": False},
        {"validation_sessions": []},
        {"valid_episodes": 599},
        {"false_blocks": 1},
        {"fault_episodes": 99},
        {"detected_faults": 95},
    ],
)
def test_insufficient_or_unaccepted_validation_never_promotes(change):
    context = _context()
    assert not evaluate(_policy(synthetic_manifest(context, **change)), context)[
        "accepted"
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"false_blocks": 601},
        {"detected_faults": 101},
        {"accepted_at": datetime(2020, 1, 1)},
        {"validation_sessions": ["same"], "tuning_sessions": ["same"]},
        {"validation_sessions": ["same", "same"]},
        {"detector_digest": "not-a-digest"},
        {"valid_until": datetime(2019, 1, 1, tzinfo=timezone.utc)},
    ],
)
def test_malformed_manifests_are_rejected(change):
    with pytest.raises(ValidationError):
        synthetic_manifest(_context(), **change)


def test_confidence_bounds_match_the_declared_acceptance_gate():
    assert binomial_upper(0, 600) == pytest.approx(1 - 0.05 ** (1 / 600))
    assert binomial_upper(0, 500) > 0.005
    assert 1 - binomial_upper(4, 100) > 0.90
    assert 1 - binomial_upper(5, 100) < 0.90


def test_legacy_name_list_is_not_calibration_and_false_is_rejected():
    policy = load_default_policy().model_copy(
        update={"calibrated_metrics": {"integrity.spike_pct": "approved"}}
    )
    assert not evaluate(policy, _context())["accepted"]
    with pytest.raises(ValidationError):
        type(policy).model_validate(
            {**policy.model_dump(), "enforce_calibration": False}
        )


def test_missing_runtime_identity_never_authorizes_blocking():
    context = _context()
    manifest = synthetic_manifest(context)
    context["binding_digest"] = None
    assert not evaluate(_policy(manifest), context)["accepted"]


def test_duplicate_manifest_ids_and_yaml_keys_are_configuration_errors():
    manifest = synthetic_manifest(_context())
    data = load_default_policy().model_dump()
    with pytest.raises(ValidationError, match="duplicate"):
        type(load_default_policy()).model_validate(
            {**data, "calibration_manifests": [manifest, manifest]}
        )
    with pytest.raises(ValueError, match="duplicate"):
        _parse_policy(
            "extends: default\nenforce_calibration: true\nenforce_calibration: true",
            source="test",
        )


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
    r = next(r for r in report.coverage.metrics if r.key == "integrity.missing_pct")
    assert r.eligible == r.skipped > 0 and r.computed == 0
    vision = next(r for r in report.coverage.capabilities if r.key == "video_quality")
    assert vision.unavailable == vision.eligible == len(report.episodes)
    assert report.eligibility_counts.unknown == len(report.episodes)
    assert report.readiness.score is None


def test_optional_video_is_explicitly_outside_numeric_scope():
    report = grade(LEROBOT_FIXTURE)
    r = next(r for r in report.coverage.dimensions if r.key == "visual_quality")
    assert r.eligible == 0 and r.not_required > 0 and r.computed == 0


def test_missing_required_metric_cannot_be_covered_by_another_check():
    report = grade(
        LEROBOT_FIXTURE,
        bundle=Bundle(
            requirements=RequirementsSection(required_metrics=["vision.not_installed"])
        ),
    )
    assert report.eligibility_counts.unknown == len(report.episodes)
    assert all(
        any(r.id == "metric:vision.not_installed" for r in e.eligibility.reasons)
        for e in report.episodes
    )


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_flatline_intervals_break_at_invalid_rows(invalid):
    ctx = _channel_ctx([1.0, 1.0, invalid, 1.0, 1.0, 2.0, 3.0])
    result = flatline_pct(ctx)
    assert result.support.kind.value == "intervals"
    assert [(i.start, i.end_exclusive) for i in result.support.intervals] == [
        (0, 2),
        (3, 5),
    ]
    assert result.evidence["n_pairs"] == 4
    assert result.value == 50


def test_flatline_addresses_follow_source_map_and_split_gaps():
    ctx = _channel_ctx([1.0, 1.0, 1.0, 2.0, 3.0])
    ctx.stream.stream.source_order = SourceOrder(original_index=[0, 2, 4, 6, 8])
    result = flatline_pct(ctx)
    assert [(i.start, i.end_exclusive) for i in result.support.intervals] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


def test_transformed_flatline_without_row_map_abstains():
    ctx = _channel_ctx([1.0, 1.0, 1.0, 2.0, 3.0])
    ctx.stream.stream.source_order = SourceOrder(preserved=False, transform="sorted")
    assert flatline_pct(ctx).value is None


def test_spike_marks_sample_and_wider_filter_support():
    values = [math.sin(i * 0.13) for i in range(101)]
    values[50] = 100
    result = spike_pct(_channel_ctx(values))
    match = next(i for i in result.support.intervals if i.start == 50)
    assert match.end_exclusive == 51
    assert (match.support_start, match.support_end_exclusive) == (25, 76)


def test_empty_or_half_specified_intervals_are_rejected():
    with pytest.raises(ValidationError):
        SampleInterval(start=2, end_exclusive=2)
    with pytest.raises(ValidationError):
        SampleInterval(start=2, end_exclusive=3, support_start=0)


def test_default_review_and_explicit_calibration_propagate_to_every_decision(tmp_path):
    path = tmp_path / "arm.hdf5"
    _write_arm(path, 4, {1})
    report = grade(path)
    assert (report.eligibility_counts.review, report.eligibility_counts.blocked) == (
        1,
        0,
    )
    assert report.readiness.score is None and report.score.train_ready is None
    assert any(
        f.calibration.get("reason") == "SNR physical noise assessment is unavailable"
        for f in report.findings
    )
    accepted = grade_with_test_calibration(path)
    assert (
        accepted.eligibility_counts.review,
        accepted.eligibility_counts.blocked,
    ) == (0, 1)
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


def test_coverage_summary_is_shared_by_terminal_html_and_inspect(tmp_path):
    report = grade(LEROBOT_FIXTURE)
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json())
    terminal = render_terminal(report, width=300)
    html = render_html(report)
    inspected = CliRunner().invoke(app, ["inspect", str(path)])
    assert inspected.exit_code == 0
    for line in coverage_lines(report.coverage):
        assert line in terminal
        assert line in html
        # CLI can wrap; compare collapsed whitespace.
        assert " ".join(line.split()) in " ".join(inspected.stdout.split())
    assert (
        CliRunner()
        .invoke(app, ["inspect", str(path), "--episode", "missing"])
        .exit_code
        == 2
    )


def _identified_report():
    report = grade(LEROBOT_FIXTURE)
    report.run.source = report.run.source.model_copy(
        update={
            "scope": HashScope.COMPLETE,
            "canonicalization": "sorted-relative-path-content-sha256-v1",
            "digest": "a" * 64,
            "complete": True,
            "covered_inputs": 4,
        }
    )
    return report


def test_compare_requires_complete_source_identity():
    old = grade(LEROBOT_FIXTURE)
    assert not compare(old, old).comparable
    identified = _identified_report()
    result = compare(identified, identified)
    assert result.comparable
    assert (
        result.readiness_delta == 0
        if identified.readiness.score is not None
        else result.readiness_delta is None
    )


@pytest.mark.parametrize(
    "identity",
    [
        "binding",
        "dictionary",
        "policy",
        "requirements",
        "execution",
        "source",
        "metrics",
        "adapters",
    ],
)
def test_compare_refuses_changed_identities(identity):
    old = _identified_report()
    new = old.model_copy(deep=True)
    if identity in ("metrics", "adapters"):
        setattr(new.producer, identity, {"changed": "b" * 64})
    else:
        item = getattr(new.run, identity)
        setattr(new.run, identity, item.model_copy(update={"digest": "b" * 64}))
    result = compare(old, new)
    assert not result.comparable and identity in result.identity_changes
    assert result.readiness_delta is None


def test_compare_records_raw_metric_changes_and_does_not_guess_causality():
    old = _identified_report()
    new = old.model_copy(deep=True)
    metric = new.episodes[0].streams[0].metrics["recorded_hz"]
    metric.value = 123
    result = compare(old, new)
    assert result.metric_changes and "not proven" in result.attribution


def test_comparison_cli_writes_json_and_does_not_overwrite_inputs(tmp_path):
    report = _identified_report()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(report.model_dump_json())
    b.write_text(report.model_dump_json())
    out = tmp_path / "comparison.json"
    runner = CliRunner()
    assert (
        runner.invoke(app, ["compare", str(a), str(b), "--report", str(out)]).exit_code
        == 0
    )
    assert json.loads(out.read_text())["comparable"]
    assert (
        runner.invoke(app, ["compare", str(a), str(b), "--report", str(a)]).exit_code
        == 2
    )
    assert isinstance(load_report(a), Report)


def test_legacy_compare_refuses_synthetic_current_decisions():
    path = Path(__file__).parent / "legacy_reports/lerobot_v3_tiny-6.5.0.json"
    result = compare(path, path)
    assert not result.comparable and result.readiness_delta is None


def test_profiles_commands_validate_structure_and_resolve_policy(tmp_path):
    runner = CliRunner()
    assert "numeric-core-v1" in runner.invoke(app, ["profiles", "list"]).stdout
    shown = runner.invoke(app, ["profiles", "show", "vision-imitation-v1"])
    assert shown.exit_code == 0 and "video_quality" in shown.stdout
    assert runner.invoke(app, ["profiles", "show", "no-such-profile"]).exit_code == 2
    bundle = tmp_path / "profile.yaml"
    bundle.write_text("requirements:\n  id: numeric-core-v1\n")
    assert runner.invoke(app, ["profiles", "validate", str(bundle)]).exit_code == 0
    bundle.write_text("policy:\n  path: missing.yaml\n")
    assert runner.invoke(app, ["profiles", "validate", str(bundle)]).exit_code == 2


def test_real_schema63_fixture_loads_without_fabricating_coverage():
    path = (
        Path(__file__).parent
        / "legacy_reports/aloha_static_towel-43e0cb1-6.3.0-trimmed.json"
    )
    report = load_report(path)
    assert report.schema_version == "6.3.0" and report.summary.n_episodes == 1
    assert not compare(path, path).comparable


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
    assert report.operational_errors and report.run.completion.value == "partial"
    assert report.eligibility_counts.unknown == len(report.episodes)
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
    assert load_report(output).operational_errors


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


def test_new_cli_commands_are_registered_in_module_entrypoint():
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "kalanos.cli", "profiles", "list"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0 and "vision-imitation-v1" in result.stdout


def test_opt_in_byte_hash_makes_real_reports_comparable():
    old = grade(LEROBOT_FIXTURE, hash_source=True)
    new = grade(LEROBOT_FIXTURE, hash_source=True)
    assert old.run.source.complete and old.run.source.covered_inputs > 0
    assert compare(old, new).comparable


def test_source_hash_refuses_symlinks(tmp_path):
    from kalanos.analysis.models.errors import SourceUnavailable
    from kalanos.analysis.source_identity import hash_local_source

    p = tmp_path / "link"
    p.symlink_to(str(LEROBOT_FIXTURE), target_is_directory=True)
    with pytest.raises(SourceUnavailable, match="symlink"):
        hash_local_source(p)


def test_change_during_analysis_withholds_report(monkeypatch):
    from kalanos import api
    from kalanos.analysis.models.errors import SourceUnavailable
    from kalanos.analysis.source_identity import hash_local_source

    original = hash_local_source(LEROBOT_FIXTURE)
    values = iter([original, original.model_copy(update={"digest": "b" * 64})])
    monkeypatch.setattr(api, "hash_local_source", lambda root: next(values))
    with pytest.raises(SourceUnavailable, match="changed during"):
        grade(LEROBOT_FIXTURE, hash_source=True)


@pytest.mark.parametrize("field", ["adapters", "metrics"])
def test_comparison_refuses_unknown_plugin_implementation(field):
    old = grade(LEROBOT_FIXTURE, hash_source=True)
    new = old.model_copy(deep=True)
    getattr(old.producer, field)["custom_plugin"] = "unknown"
    getattr(new.producer, field)["custom_plugin"] = "unknown"
    result = compare(old, new)
    assert not result.comparable
    assert f"{field} identity is missing or insufficient" in result.reasons


def test_flatline_duration_requires_known_timestamp_units():
    import polars as pl

    from kalanos.analysis.models.domain import ClockInfo

    ctx = _channel_ctx([1.0, 1.0, 1.0, 2.0, 3.0])
    ctx.stream.stream.native_timestamps = pl.Series([0, 1, 2, 3, 4])
    ctx.stream.stream.clock_info = ClockInfo(native_unit="unknown")
    result = flatline_pct(ctx)
    assert result.evidence["longest_run"] == 3
    assert result.evidence["longest_run_s"] is None
    assert result.support.intervals[0].start == 0
    assert result.support.intervals[0].end_exclusive == 3
