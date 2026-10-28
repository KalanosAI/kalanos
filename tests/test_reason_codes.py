"""Verifies reason texts are written once as codes and read back as the same text."""

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
from upath import UPath

# Internal
from kalanos.analysis.models.legacy import load_any
from kalanos.analysis.models.metrics import MetricResult
from kalanos.analysis.models.reason_codes import (
    REASON_KEYS,
    TABLE_KEY,
    ReasonCoder,
    decode_reasons,
    encode_reasons,
)
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.render import render_json
from kalanos.api import grade


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"
VALUE_REASON = "needs one of ['proprio."


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(scope="module")
def report() -> Report:
    """Grade the tiny LeRobot v3 fixture once for the whole module.

    Returns
    -------
    Report
        The graded report.
    """

    return grade(TINY_V3)


def _reason_values(data: Any) -> list[Any]:
    """Collect every value held under a reason key, at any depth.

    Parameters
    ----------
    data : Any
        A parsed report, or any part of one.

    Returns
    -------
    list[Any]
        Each value under a key in `REASON_KEYS`, with the items of a list
        and the keys of a map listed one by one. A map key that is a decimal string
        is listed as its integer.
    """

    found: list[Any] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in REASON_KEYS:
                if isinstance(value, list):
                    found.extend(v for v in value if not isinstance(v, dict))
                elif isinstance(value, dict):
                    keys = value if key == "reasons" else []
                    found.extend(int(k) if k.isdecimal() else k for k in keys)
                else:
                    found.append(value)
            found.extend(_reason_values(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(_reason_values(item))
    return found


def _metric_reasons(report: Report) -> dict[tuple[Any, ...], str]:
    """Map each metric result's place in the report to its evidence reason.

    Parameters
    ----------
    report : Report
        The report to read.

    Returns
    -------
    dict[tuple[Any, ...], str]
        The text under `evidence["reason"]`, keyed by episode id,
        stream and channel position, and metric name.
    """

    found: dict[tuple[Any, ...], str] = {}
    for episode in report.episodes:
        places: list[tuple[tuple[Any, ...], dict[str, MetricResult]]] = [
            ((episode.id,), episode.metrics)
        ]
        for s, stream in enumerate(episode.streams):
            places.append(((episode.id, s), stream.metrics))
            for c, channel in enumerate(stream.channels):
                places.append(((episode.id, s, c), channel.metrics))
        for address, metrics in places:
            for name, result in metrics.items():
                reason = result.evidence.get("reason")
                if isinstance(reason, str):
                    found[(*address, name)] = reason
    return found


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_encode_then_decode_restores_every_shape():
    original = {
        "id": "needs a camera",
        "reason": "needs a camera",
        "skipped": {"reason": None},
        "readiness": {"reasons": ["needs a camera", "too short"]},
        "coverage": {"reasons": {"too short": 2, "no timestamps": 1}},
        "eligibility": {
            "reasons": [{"code": "x", "detail": "sampled at 10 Hz"}],
        },
        "evidence": {"windows": [{"reasons": ["too short", "gap at 3.2 s"]}]},
    }
    before = copy.deepcopy(original)

    encoded = encode_reasons(original)

    assert next(iter(encoded)) == TABLE_KEY
    assert encoded[TABLE_KEY] == [
        "needs a camera",
        "too short",
        "no timestamps",
        "sampled at 10 Hz",
        "gap at 3.2 s",
    ]
    assert encoded["id"] == "needs a camera"
    assert encoded["reason"] == 0
    assert encoded["skipped"] == {"reason": None}
    assert encoded["readiness"]["reasons"] == [0, 1]
    assert encoded["coverage"]["reasons"] == {"1": 2, "2": 1}
    assert encoded["eligibility"]["reasons"] == [{"code": "x", "detail": 3}]
    assert encoded["evidence"]["windows"][0]["reasons"] == [1, 4]
    assert decode_reasons(encoded) == original
    assert TABLE_KEY not in decode_reasons(encoded)
    assert original == before


def test_a_coder_over_the_parts_codes_as_the_whole_document_does():
    parts = [
        {"reason": "b"},
        {"coverage": {"reasons": {"a": 1, "b": 2}}},
        {"episodes": [{"detail": "c"}, {"reasons": ["a", "d"]}]},
    ]
    whole = encode_reasons({k: v for part in parts for k, v in part.items()})
    coder = ReasonCoder()

    coded = [coder.walk(part) for part in parts]

    assert coder.table() == whole[TABLE_KEY] == ["b", "a", "c", "d"]
    assert {k: v for part in coded for k, v in part.items()} == {
        k: v for k, v in whole.items() if k != TABLE_KEY
    }


@pytest.mark.parametrize("data", [{"reason": 3}, {"reasons": [3, "a"]}])
def test_encode_refuses_a_number_under_a_reason_key(data):
    with pytest.raises(TypeError):
        encode_reasons(data)


def test_decode_refuses_an_unknown_code():
    with pytest.raises(ValueError, match="reason code 1"):
        decode_reasons({TABLE_KEY: ["only one"], "reason": 1})


def test_a_graded_report_round_trips_through_its_codes(report, tmp_path):
    text = render_json(report)
    parsed = json.loads(text)

    assert next(iter(parsed)) == TABLE_KEY
    values = _reason_values(parsed)
    assert values
    assert all(v is None or type(v) is int for v in values)
    assert Report.model_validate_json(text) == report

    path = tmp_path / "r.json"
    path.write_text(text)
    assert load_any(UPath(path)) == report


def test_a_reason_with_a_value_keeps_it(report, tmp_path):
    graded = {
        address: text
        for address, text in _metric_reasons(report).items()
        if VALUE_REASON in text
    }
    assert graded

    path = tmp_path / "r.json"
    path.write_text(render_json(report))
    loaded = load_any(UPath(path))
    table = json.loads(path.read_text())[TABLE_KEY]

    assert isinstance(loaded, Report)
    reread = _metric_reasons(loaded)
    for address, text in graded.items():
        assert reread[address] == text
        assert text in table


def test_a_report_without_reason_codes_loads_its_text(report):
    assert Report.model_validate(report.model_dump()) == report


def test_python_mode_dump_keeps_text(report):
    assert TABLE_KEY not in report.model_dump()
    assert TABLE_KEY in report.model_dump(mode="json")
