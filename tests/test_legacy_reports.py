"""Verifies older-schema reports load unchanged and never pass for current ones."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import json
import re
from pathlib import Path

# External
import pytest
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis.models.diagnostics import DiagnosticPlan
from kalanos.analysis.models.legacy import (
    LegacyReport,
    UnsupportedSchema,
    load_any,
    load_legacy,
)
from kalanos.analysis.models.reason_codes import decode_reasons
from kalanos.analysis.models.report import Report, Variation
from kalanos.analysis.scoring.duplicates import _TOO_FEW
from kalanos.api import compare, grade, load_report
from kalanos.assets.bundle import Bundle
from kalanos.cli import app

# Local
from helpers import CSV_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

LEGACY = Path(__file__).parent / "legacy_reports"
TOWEL_6_4 = LEGACY / "aloha_static_towel-048fef2-6.4.0-trimmed.json"
TINY_6_5 = LEGACY / "lerobot_v3_tiny-6.5.0.json"
V7_1 = LEGACY / "lerobot_v3_tiny-7.1.0.json"
V7_2 = LEGACY / "lerobot_v3_tiny-7.2.0.json"
TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"

# The two conversions `docs/SCHEMA.md` documents for a 7.x report:
# pair-object `duplicates` become `DuplicateEpisodes`,
# and per-episode `diagnostics` are ignored.
CONVERTED_ON_LOAD = re.compile(r"/duplicates|/episodes\[\d+\]/diagnostics")


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


def _assert_contains(written, loaded, path: str) -> None:
    """Assert every value in `written` is in `loaded`, through nested dicts and lists.

    Parameters
    ----------
    written : Any
        A value as the file stores it.
    loaded : Any
        The same value from the loaded report's decoded JSON dump.
    path : str
        Where `written` sits in the file, as `/key/list[i]`, for the failure message.

    Raises
    ------
    AssertionError
        If a key is missing from `loaded`, a list changes length, or a value differs,
        outside the paths in `CONVERTED_ON_LOAD`.
    """

    if isinstance(written, dict):
        assert isinstance(loaded, dict), path
        for key, value in written.items():
            if CONVERTED_ON_LOAD.fullmatch(f"{path}/{key}"):
                continue
            assert key in loaded, f"{path}/{key} missing after load"
            _assert_contains(value, loaded[key], f"{path}/{key}")
    elif isinstance(written, list):
        assert isinstance(loaded, list) and len(loaded) == len(written), path
        for i, (w, ld) in enumerate(zip(written, loaded, strict=True)):
            _assert_contains(w, ld, f"{path}[{i}]")
    else:
        assert written == loaded, f"{path}: {written!r} became {loaded!r}"


def test_a_new_report_declares_8_0_0(tmp_path):
    out = tmp_path / "r.json"
    out.write_text(grade(TINY_V3).model_dump_json())
    assert json.loads(out.read_text())["schema_version"] == "8.0.0"


@pytest.mark.parametrize(("path", "version"), [(V7_1, "7.1.0"), (V7_2, "7.2.0")])
def test_a_0_8_x_report_loads_with_its_values(tmp_path, path, version):
    loaded = load_report(path)

    assert isinstance(loaded, Report)
    assert loaded.schema_version == version
    _assert_contains(
        json.loads(path.read_text()),
        decode_reasons(loaded.model_dump(mode="json")),
        "",
    )
    assert loaded.duplicates.exact == []
    assert loaded.duplicates.near.episode == []
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert f"schema {version}" in result.output and "2/2 pass" in result.output
    fresh = tmp_path / "fresh.json"
    fresh.write_text(grade(TINY_V3).model_dump_json())
    result = runner.invoke(app, ["compare", str(path), str(fresh)])
    # Code 1 says two builds' reports are not comparable; code 2 is a load failure.
    # A crash would also exit 1, so the verdict must be printed and agree with the code.
    assert result.exit_code in (0, 1), result.output
    assert json.loads(result.stdout)["comparable"] is (result.exit_code == 0)
    diff = compare(path, path)
    assert diff.added_episodes == diff.removed_episodes == diff.episode_changes == []


def test_a_current_report_is_not_legacy(tmp_path):
    report = grade(TINY_V3)
    out = tmp_path / "r.json"
    out.write_text(report.model_dump_json())
    assert isinstance(load_any(UPath(out)), Report)
    with pytest.raises(UnsupportedSchema):
        from kalanos.analysis.models.legacy import load_legacy

        load_legacy(out.read_bytes())


@pytest.mark.parametrize("version", ["7.1.0", "7.0.0"])
def test_an_older_7_report_loads_through_load_any(tmp_path, version):
    payload = grade(TINY_V3).model_dump(mode="json")
    for episode in payload["episodes"]:
        del episode["split"]
        for stream in episode["streams"]:
            del stream["video_format"]
            for channel in stream["channels"]:
                del channel["channel"]["source_dtype"]
            for channel in stream["declared_channels"]:
                del channel["source_dtype"]
    payload["schema_version"] = version
    out = tmp_path / "r.json"
    out.write_text(json.dumps(payload))

    assert isinstance(load_any(UPath(out)), Report)
    with pytest.raises(UnsupportedSchema):
        load_legacy(out.read_bytes())


def test_a_7_2_report_with_episode_diagnostics_and_thumbnails_loads(tmp_path):
    report = grade(TINY_V3, bundle=Bundle(diagnostics=DiagnosticPlan(vision=True)))
    assert report.diagnostics is not None
    payload = report.model_dump(mode="json")
    results = payload["diagnostics"]["results"]
    results[0]["evidence"]["frames"][0]["thumbnail_png_base64"] = "iVBORw0KGgo="
    for e in payload["episodes"]:
        e["diagnostics"] = [r for r in results if r["episode_id"] == e["id"]]
    payload["schema_version"] = "7.2.0"
    out = tmp_path / "r.json"
    out.write_text(json.dumps(payload))

    loaded = load_any(UPath(out))
    assert isinstance(loaded, Report)
    assert loaded.diagnostics is not None
    assert loaded.diagnostics.model_dump(mode="json")["results"] == results


def _with_pair_objects(version: str) -> tuple[dict, list[str]]:
    """A graded report as `version` wrote `duplicates`: one object per pair, by id.

    Parameters
    ----------
    version : str
        The schema version to declare.

    Returns
    -------
    tuple of (dict, list of str)
        The payload, and its episode ids in order.
    """

    payload = grade(TINY_V3).model_dump(mode="json")
    payload["schema_version"] = version
    ids = [e["id"] for e in payload["episodes"]]
    payload["duplicates"] = [
        {
            "episodes": [ids[1], ids[0]],
            "tier": "near",
            "nrmse": 0.0123456789,
            "footage_identical": True,
            "cameras": ["observation.images.top"],
        }
    ]
    return payload, ids


@pytest.mark.parametrize("version", ["7.2.0", "7.3.0"])
def test_a_7_x_report_with_pair_objects_loads(tmp_path, version):
    payload, ids = _with_pair_objects(version)
    assert len(ids) == 2
    out = tmp_path / "r.json"
    out.write_text(json.dumps(payload))

    loaded = load_any(UPath(out))

    assert isinstance(loaded, Report)
    assert loaded.duplicates.exact == []
    near = loaded.duplicates.near
    assert (near.episode, near.partners, near.nrmse) == ([0], [[1]], [[0.01235]])
    assert isinstance(load_report(out), Report)
    diff = compare(out, out)
    assert diff.added_episodes == diff.removed_episodes == diff.episode_changes == []
    result = CliRunner().invoke(app, ["inspect", str(out)])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda pair: pair["episodes"].__setitem__(0, "not_an_episode"),
            "not_an_episode",
        ),
        (lambda pair: pair.pop("tier"), "malformed pair"),
        (lambda pair: pair.__setitem__("tier", "nearby"), "malformed pair"),
    ],
)
def test_a_malformed_or_unknown_pair_is_refused(tmp_path, change, message):
    payload, _ids = _with_pair_objects("7.3.0")
    change(payload["duplicates"][0])
    out = tmp_path / "r.json"
    out.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_any(UPath(out))


@pytest.mark.parametrize("version", ["7.4.0", "8.0.1", "8.1.0", "9.0.0", "7.x.0"])
def test_a_newer_or_malformed_report_is_refused(tmp_path, version):
    payload = grade(TINY_V3).model_dump(mode="json")
    payload["schema_version"] = version
    out = tmp_path / "r.json"
    out.write_text(json.dumps(payload))

    with pytest.raises(UnsupportedSchema):
        load_any(UPath(out))


def test_inspect_reads_current_and_legacy_reports(tmp_path):
    runner = CliRunner()
    current = tmp_path / "r.json"
    current.write_text(grade(TINY_V3).model_dump_json())
    out = runner.invoke(app, ["inspect", str(current)])
    assert out.exit_code == 0 and "2/2 pass" in out.stdout
    out = runner.invoke(app, ["inspect", str(TOWEL_6_4)])
    assert out.exit_code == 0 and "2 contradiction" in out.stdout


def test_a_report_without_dataset_warnings_loads(tmp_path):
    """A report written before `dataset_warnings` existed loads with none."""

    payload = grade(TINY_V3).model_dump(mode="json")
    payload.pop("dataset_warnings", None)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload))

    loaded = load_any(UPath(path))

    assert isinstance(loaded, Report)
    assert loaded.dataset_warnings == []


def test_a_report_without_motion_variation_loads(tmp_path):
    """A report written before `motion_variation` existed loads with none."""

    payload = grade(TINY_V3).model_dump(mode="json")
    payload.pop("motion_variation", None)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload))

    loaded = load_any(UPath(path))

    assert isinstance(loaded, Report)
    assert loaded.motion_variation is None


def test_a_report_without_visual_variation_loads(tmp_path):
    """A report written before `visual_variation` existed loads with none."""

    payload = grade(TINY_V3).model_dump(mode="json")
    assert payload["cameras"]
    for camera in payload["cameras"]:
        camera.pop("visual_variation", None)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload))

    loaded = load_any(UPath(path))

    assert isinstance(loaded, Report)
    assert [camera.visual_variation for camera in loaded.cameras] == [None]


def test_an_unavailable_motion_variation_keeps_its_count_when_written(tmp_path):
    """A written report leaves out the null value but keeps `episodes`,
    and reads the reason back from its code."""

    out = tmp_path / "r.json"
    result = CliRunner().invoke(app, ["grade", str(CSV_FIXTURE), "--report", str(out)])
    assert result.exit_code == 0, result.output

    written = json.loads(out.read_text())["motion_variation"]
    loaded = load_any(UPath(out))

    assert "value" not in written and written["episodes"] == 1
    assert isinstance(loaded, Report)
    assert loaded.motion_variation == Variation(episodes=1, reason=_TOO_FEW)
