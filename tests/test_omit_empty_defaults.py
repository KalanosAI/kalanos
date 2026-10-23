"""Verifies a written report leaves out only the empty values its defaults restore."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import copy
import json
from pathlib import Path
from typing import Any

# External
import pytest
import yaml
from pydantic import BaseModel, Field

# Local
from test_reporting import _sample_report

# Internal
from kalanos.analysis.models.diagnostics import DiagnosticPlan
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.omit import omit_empty_defaults
from kalanos.analysis.reporting.render import render_json, render_yaml
from kalanos.api import grade, load_report
from kalanos.assets.bundle import Bundle


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"

EMPTY_TOP_LEVEL_LISTS = (
    "findings",
    "skipped",
    "unresolved",
    "operational_errors",
    "mapping_overrides",
    "binding_conflicts",
    "duplicates",
)


class Child(BaseModel):
    note: str | None = None
    tags: list[str] = Field(default_factory=list)


class Parent(BaseModel):
    required_nullable: int | None
    opt: str | None = None
    items: list[Child] = Field(default_factory=list)
    by_name: dict[str, Child] = Field(default_factory=dict)
    listed_or_none: list[int] | None = None
    level: str = "unknown"
    evidence: dict[str, Any] = Field(default_factory=dict)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_omits_only_empty_values_equal_to_the_default():
    parent = Parent(
        required_nullable=None,
        items=[Child(), Child(note="x", tags=["a"])],
        by_name={"k": Child()},
        listed_or_none=[],
        evidence={"gap": None, "empty": {}},
    )
    data = parent.model_dump(mode="json")
    before = copy.deepcopy(data)

    result = omit_empty_defaults(parent, data)

    assert result == {
        "required_nullable": None,
        "items": [{}, {"note": "x", "tags": ["a"]}],
        "by_name": {"k": {}},
        "listed_or_none": [],
        "level": "unknown",
        "evidence": {"gap": None, "empty": {}},
    }
    assert Parent.model_validate(result) == parent
    assert data == before


@pytest.mark.parametrize("vision", [False, True], ids=["plain", "vision"])
def test_a_written_report_loads_back_equal(tmp_path, vision):
    bundle = Bundle(diagnostics=DiagnosticPlan(vision=True)) if vision else None
    report = grade(TINY_V3, bundle=bundle)
    path = tmp_path / "r.json"
    path.write_text(render_json(report))

    assert load_report(path) == report
    assert Report.model_validate(yaml.safe_load(render_yaml(report))) == report


def test_a_written_report_leaves_out_empty_defaults():
    report = grade(TINY_V3)

    text = render_json(report)
    data = json.loads(text)

    assert "schema_version" in data
    assert not set(EMPTY_TOP_LEVEL_LISTS) & data.keys()
    bindings = [
        graded["channel"]["binding"]
        for episode in data["episodes"]
        for stream in episode.get("streams", [])
        for graded in stream.get("channels", [])
        if "binding" in graded["channel"]
    ]
    assert bindings
    for binding in bindings:
        assert not [k for k, v in binding.items() if v is None or v == [] or v == {}]
    assert len(text) < len(report.model_dump_json())


def test_render_json_writes_nan_as_null():
    def _raise(token):
        raise ValueError(f"bare {token} in JSON")

    report = _sample_report()
    report.findings[0].value = float("nan")
    report.findings[0].evidence = {"v": float("nan")}

    text = render_json(report)

    assert '"v":null' in text
    assert json.loads(text)["findings"][0]["value"] is None
    assert "NaN" not in text
    json.loads(text, parse_constant=_raise)
