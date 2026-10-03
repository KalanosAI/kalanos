"""Verifies older-schema reports load unchanged and never pass for current ones."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
from pathlib import Path

# External
import pytest
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis.models.legacy import LegacyReport, UnsupportedSchema, load_any
from kalanos.analysis.models.report import Report
from kalanos.api import compare, grade, load_report
from kalanos.cli import app


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

LEGACY = Path(__file__).parent / "legacy_reports"
TOWEL_6_4 = LEGACY / "aloha_static_towel-048fef2-6.4.0-trimmed.json"
TINY_6_5 = LEGACY / "lerobot_v3_tiny-6.5.0.json"
TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_schema_6_4_report_loads_losslessly_and_names_its_contradictions():
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


def test_a_schema_6_5_branch_report_loads_with_its_mapping_fields():
    loaded = load_any(UPath(TINY_6_5))
    assert isinstance(loaded, LegacyReport)
    assert loaded.schema_version == "6.5.0"
    assert "mapping_overrides" in loaded.data
    assert loaded.contradictions == []
    assert loaded.summary.n_score_train_ready == 2


def test_real_schema63_fixture_loads_without_fabricating_coverage():
    path = (
        Path(__file__).parent
        / "legacy_reports/aloha_static_towel-43e0cb1-6.3.0-trimmed.json"
    )
    report = load_report(UPath(path))
    assert isinstance(report, LegacyReport)
    assert report.schema_version == "6.3.0" and report.summary.n_episodes == 1
    assert not compare(path, path).comparable


def test_a_current_report_is_not_legacy(tmp_path):
    report = grade(TINY_V3)
    out = tmp_path / "r.json"
    out.write_text(report.model_dump_json())
    assert isinstance(load_any(UPath(out)), Report)
    with pytest.raises(UnsupportedSchema):
        from kalanos.analysis.models.legacy import load_legacy

        load_legacy(out.read_bytes())


def test_inspect_reads_current_and_legacy_reports(tmp_path):
    runner = CliRunner()
    current = tmp_path / "r.json"
    current.write_text(grade(TINY_V3).model_dump_json())
    out = runner.invoke(app, ["inspect", str(current)])
    assert out.exit_code == 0 and "2/2 pass" in out.stdout
    out = runner.invoke(app, ["inspect", str(TOWEL_6_4)])
    assert out.exit_code == 0 and "2 contradiction" in out.stdout
