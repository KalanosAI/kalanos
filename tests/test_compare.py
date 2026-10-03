"""Verifies that two reports compare only when every identity behind them matches."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from pathlib import Path

# External
import pytest
from typer.testing import CliRunner

# Internal
from kalanos.analysis.models.provenance import HashScope
from kalanos.analysis.models.report import Report
from kalanos.api import compare, grade, load_report
from kalanos.cli import app

# Local
from helpers import LEROBOT_FIXTURE


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _identified_report():
    report = grade(LEROBOT_FIXTURE)
    assert report.run is not None and report.run.source is not None
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


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_compare_requires_complete_source_identity():
    old = grade(LEROBOT_FIXTURE)
    assert not compare(old, old).comparable
    identified = _identified_report()
    result = compare(identified, identified)
    assert result.comparable
    assert identified.readiness is not None
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


@pytest.mark.parametrize("field", ["adapters", "metrics"])
def test_comparison_refuses_unknown_plugin_implementation(field):
    old = grade(LEROBOT_FIXTURE, hash_source=True)
    new = old.model_copy(deep=True)
    getattr(old.producer, field)["custom_plugin"] = "unknown"
    getattr(new.producer, field)["custom_plugin"] = "unknown"
    result = compare(old, new)
    assert not result.comparable
    assert f"{field} identity is missing or insufficient" in result.reasons
