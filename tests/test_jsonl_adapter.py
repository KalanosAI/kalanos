"""Verifies JsonlAdapter: content-based detection and array-to-channel expansion."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from pathlib import Path

# External
import polars as pl

# Internal
from kalanos.analysis.adapters.jsonl import JsonlAdapter
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, FIXTURES_DIR, IMU_FIXTURE


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _jsonl(tmp_path: Path, name: str, text: str) -> Path:
    """Write `text` to `tmp_path / name` and return the path."""

    path = tmp_path / name
    path.write_text(text)
    return path


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_passes_the_adapter_contract_suite():
    """Verify JsonlAdapter satisfies the Adapter protocol on the real fixture."""

    check_adapter(
        JsonlAdapter(), IMU_FIXTURE, [CSV_FIXTURE, FIXTURES_DIR / "capture_index.json"]
    )


def test_detection_is_by_content_not_extension(tmp_path):
    """Verify a .txt of JSONL bids, a bare-first-line or one-object file declines."""

    adapter = JsonlAdapter()

    txt_jsonl = _jsonl(tmp_path, "records.txt", '{"a": 1}\n{"a": 2}\n')
    assert adapter.detect(txt_jsonl) == 0.3

    pretty_printed = _jsonl(tmp_path, "pretty.jsonl", '{\n "a": 1\n}\n{\n "a": 2\n}\n')
    assert adapter.detect(pretty_printed) == 0.0

    single_object = _jsonl(tmp_path, "single.jsonl", '{"a": 1}\n')
    assert adapter.detect(single_object) == 0.0


def test_acc_expands_to_indexed_channels_under_its_own_stem():
    """Verify acc, not a dictionary alias, keeps index axes under unmapped.acc."""

    episode = next(JsonlAdapter().episodes(IMU_FIXTURE))

    stream = next(s for s in episode.streams if s.source_field == "acc")
    assert stream.taxonomy_type == "unmapped.acc"
    assert [channel.name for channel in stream.channels] == ["acc_0", "acc_1", "acc_2"]
    assert [channel.axis for channel in stream.channels] == ["0", "1", "2"]


def test_gyro_expands_to_axis_lettered_channels_via_the_dictionary():
    """Verify gyro, a dictionary alias with group_hint axes, is promoted to x/y/z."""

    episode = next(JsonlAdapter().episodes(IMU_FIXTURE))

    stream = next(s for s in episode.streams if s.source_field == "gyro")
    assert stream.taxonomy_type == "extero.imu"
    assert [channel.name for channel in stream.channels] == [
        "gyro_0",
        "gyro_1",
        "gyro_2",
    ]
    assert [channel.axis for channel in stream.channels] == ["x", "y", "z"]


def test_t_us_reaches_canonical_seconds():
    """Verify the microsecond-epoch time column converts to canonical seconds."""

    episode = next(JsonlAdapter().episodes(IMU_FIXTURE))

    stream = next(s for s in episode.streams if s.source_field == "gyro")
    assert stream.timestamps.to_list()[0] == 1700000000.0


def test_a_key_missing_from_some_records_becomes_null(tmp_path):
    """Verify a record missing a key present in others fills as null, not an error."""

    path = _jsonl(
        tmp_path,
        "sparse.txt",
        json.dumps({"t_us": 0, "acc": [1.0, 2.0, 3.0], "extra": 1.0})
        + "\n"
        + json.dumps({"t_us": 1000, "acc": [1.1, 2.1, 3.1]})
        + "\n",
    )

    episode = next(JsonlAdapter().episodes(path))

    extra_stream = next(s for s in episode.streams if s.source_field == "extra")
    assert extra_stream.payload is not None
    assert extra_stream.payload.fetch()["extra"].to_list() == [1.0, None]


def test_a_nested_list_value_survives_as_a_json_string_column(tmp_path):
    """Verify a nested list — not a list of scalars — becomes JSON text, not dropped."""

    path = _jsonl(
        tmp_path,
        "nested.txt",
        json.dumps({"t_us": 0, "x": 1.0, "matrix": [[1, 2], [3, 4]]})
        + "\n"
        + json.dumps({"t_us": 1000, "x": 1.1, "matrix": [[5, 6], [7, 8]]})
        + "\n",
    )

    parsed = JsonlAdapter().parse(path)

    assert parsed.frame["matrix"].dtype == pl.String
    assert parsed.frame["matrix"].to_list() == ["[[1, 2], [3, 4]]", "[[5, 6], [7, 8]]"]


def test_an_array_nested_under_another_key_still_promotes_to_axis_letters(tmp_path):
    """Verify a dotted prefix's stem, not the prefix itself, is what array_stems holds.

    `imu.gyro_0..2`'s grouping stem drops the `imu.` namespace the same way
    a bare `gyro_0..2` would, so the array-derived promotion has to key off
    that same stem or it silently never fires on nested input.
    """

    path = _jsonl(
        tmp_path,
        "nested_array.txt",
        json.dumps({"t_us": 0, "imu": {"gyro": [0.0, 0.1, 0.2]}})
        + "\n"
        + json.dumps({"t_us": 1000, "imu": {"gyro": [0.1, 0.2, 0.3]}})
        + "\n",
    )

    episode = next(JsonlAdapter().episodes(path))

    stream = next(s for s in episode.streams if s.source_field == "gyro")
    assert stream.taxonomy_type == "extero.imu"
    assert [channel.axis for channel in stream.channels] == ["x", "y", "z"]
