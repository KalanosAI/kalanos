"""Verifies report keys are written as short aliases and read back as the same keys."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import copy
import json
import tracemalloc
from pathlib import Path
from typing import Any

# External
import pytest
from pydantic_core import to_json
from upath import UPath

# Internal
from kalanos.analysis.models.compact_keys import (
    TABLE_KEY,
    UnsupportedEncoding,
    alias,
    decode_keys,
    encode_keys,
    is_compact,
    key_aliases,
)
from kalanos.analysis.models.legacy import load_any
from kalanos.analysis.models.report import CURRENT_SCHEMA_VERSION, Report
from kalanos.analysis.reporting.render import render_json
from kalanos.api import grade


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"

NESTED: dict[str, Any] = {
    "schema_version": "8.0.0",
    "episodes": [
        {"id": "e0", "metrics": {"motion.jerk": {"value": 1.5, "status": "ok"}}},
        {"id": "e1", "metrics": {"vision.blur": {"value": None, "status": "error"}}},
    ],
    "source": {"repo_id": "org/name", "files": [[{"path": "a"}], []]},
    "empty": {},
}


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


def _key_orders(data: Any) -> list[list[str]]:
    """Collect the key order of every object, depth first.

    Parameters
    ----------
    data : Any
        A parsed document, or any part of one.

    Returns
    -------
    list[list[str]]
        The keys of each object in `data`, in document order.
    """

    match data:
        case dict():
            return [list(data)] + [o for v in data.values() for o in _key_orders(v)]
        case list():
            return [o for v in data for o in _key_orders(v)]
        case _:
            return []


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "0"), (61, "Z"), (62, "00"), (3905, "ZZ"), (3906, "000")],
)
def test_alias_is_bijective_base_62(index, expected):
    assert alias(index) == expected


def test_encode_then_decode_restores_keys_and_their_order():
    original = copy.deepcopy(NESTED)
    encoded = encode_keys(NESTED)

    assert NESTED == original
    assert decode_keys(encoded) == NESTED
    assert _key_orders(decode_keys(encoded)) == _key_orders(NESTED)


def test_dynamic_keys_are_aliased_and_restored():
    encoded = encode_keys(NESTED)
    aliases = key_aliases(encoded)
    metrics = encoded[aliases["episodes"]][0][aliases["metrics"]]

    assert list(metrics) == [aliases["motion.jerk"]]
    assert "motion.jerk" not in json.dumps(encoded[aliases["episodes"]])
    assert decode_keys(encoded)["episodes"][1]["metrics"] == {
        "vision.blur": {"value": None, "status": "error"}
    }


def test_the_table_comes_first_and_orders_keys_by_count_then_key():
    data = {"b": [{"x": 1}, {"x": 2}, {"x": 3}], "a": {"x": 0, "y": 1}, "y": 2}
    encoded = encode_keys(data)

    assert next(iter(encoded)) == TABLE_KEY
    assert is_compact(encoded) and not is_compact(data)
    # x appears 4 times, y twice, a and b once each.
    assert encoded[TABLE_KEY] == {"version": 1, "keys": ["x", "y", "a", "b"]}
    assert encoded["3"] == [{"0": 1}, {"0": 2}, {"0": 3}]


def test_aliases_are_unique_and_case_sensitive():
    data = {f"k{i:03}": i for i in range(100)}
    aliases = key_aliases(encode_keys(data))

    assert len(set(aliases.values())) == 100
    assert aliases["k010"] == "a" and aliases["k036"] == "A"


def test_two_encodings_of_one_report_are_identical():
    assert to_json(encode_keys(NESTED)) == to_json(encode_keys(copy.deepcopy(NESTED)))


def test_keys_that_look_like_aliases_round_trip():
    data = {"0": {"a": 1, "ZZ": [{"_compact": "nested"}]}, "a": "0", "x": "0"}
    encoded = encode_keys(data)

    assert decode_keys(encoded) == data
    # Values equal to an alias are never translated.
    assert encoded[key_aliases(encoded)["x"]] == "0"


def test_encode_refuses_a_top_level_table_key():
    with pytest.raises(ValueError, match="_compact"):
        encode_keys({TABLE_KEY: 1})


def test_decode_refuses_another_version():
    encoded = encode_keys(NESTED)
    encoded[TABLE_KEY]["version"] = 2

    with pytest.raises(UnsupportedEncoding, match="version 2 is not supported"):
        decode_keys(encoded)
    with pytest.raises(UnsupportedEncoding):
        key_aliases(encoded)


@pytest.mark.parametrize(
    "table",
    [{"version": 1, "keys": ["a", "a"]}, {"version": 1, "keys": "a"}, ["a"]],
)
def test_decode_refuses_a_malformed_table(table):
    with pytest.raises(ValueError, match="malformed compact key table"):
        decode_keys({TABLE_KEY: table, "0": 1})


def test_decode_refuses_an_alias_missing_from_the_table():
    with pytest.raises(ValueError, match="'1' is not in the key table"):
        decode_keys({TABLE_KEY: {"version": 1, "keys": ["a"]}, "0": {"1": 2}})


def test_a_graded_report_round_trips_through_its_aliases(report, tmp_path):
    full = json.loads(render_json(report))
    encoded = encode_keys(full)
    full_path = tmp_path / "full.json"
    full_path.write_text(render_json(report))
    compact_path = tmp_path / "compact.json"
    compact_path.write_bytes(to_json(encoded, inf_nan_mode="null"))

    assert decode_keys(json.loads(compact_path.read_text())) == full
    assert load_any(UPath(compact_path)) == load_any(UPath(full_path)) == report


def test_load_any_names_an_unsupported_encoding(tmp_path):
    encoded = encode_keys(NESTED)
    encoded[TABLE_KEY]["version"] = 2
    path = tmp_path / "r.json"
    path.write_bytes(to_json(encoded))

    with pytest.raises(UnsupportedEncoding, match="version 2"):
        load_any(UPath(path))


def test_load_any_keeps_a_repeated_long_key_shared(tmp_path):
    """A small file repeating one long key loads without writing the key at each use."""

    key = "k" * 10_000
    document = {
        TABLE_KEY: {"version": 1, "keys": ["schema_version", key]},
        "0": CURRENT_SCHEMA_VERSION,
        "1": [{"1": 0} for _ in range(2_000)],
    }
    path = tmp_path / "r.json"
    path.write_bytes(to_json(document))

    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            load_any(UPath(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Writing the key out at every use would take 2,000 x 10 kB = 20 MB.
    assert peak < 5_000_000
