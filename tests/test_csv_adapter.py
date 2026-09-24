"""Verifies CsvAdapter: bidding, discovery, and the two defects the old
CSV path carried.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from pathlib import Path

# External
import pytest

# Internal
from kalanos.analysis.adapters.csv import CsvAdapter
from kalanos.analysis.adapters.discover import discover_adapters
from kalanos.analysis.models.adapters import AdapterRefusal
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.domain import Attribution
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, NO_TIMESERIES_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# docs/ADAPTERS.md: 0.6-0.8 is the band for "the extension matches and the
# header parses as expected" — a specialised adapter's floor. The generic
# CSV reader must always bid strictly below it.
_SPECIALISED_ADAPTER_FLOOR = 0.6

_BARE_ENTRY = {
    "label": "Anything",
    "category": "proprioceptive_state",
    "modality": "numeric",
    "kind": "series",
    "shape": "scalar",
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _csv(tmp_path: Path, name: str, text: str) -> Path:
    """Write `text` to `tmp_path / name` and return the path.

    Parameters
    ----------
    tmp_path : Path
        Pytest's per-test temp directory.
    name : str
        File name to write.
    text : str
        File contents.

    Returns
    -------
    Path
        The written file's path.
    """

    path = tmp_path / name
    path.write_text(text)
    return path


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀

# --- Contract and bidding -------------------------------------------------


def test_passes_the_adapter_contract_suite():
    """Verify CsvAdapter satisfies the Adapter protocol, unmodified."""

    check_adapter(CsvAdapter(), CSV_FIXTURE, NO_TIMESERIES_FIXTURE)


def test_detect_bids_the_generic_table_band_below_any_specialised_adapter(tmp_path):
    """Verify detect() bids 0.3 on a .csv path and 0.0 otherwise."""

    adapter = CsvAdapter()

    bid = adapter.detect(CSV_FIXTURE)
    assert bid == 0.3
    assert bid < _SPECIALISED_ADAPTER_FLOOR
    assert adapter.detect(NO_TIMESERIES_FIXTURE) == 0.0
    assert adapter.detect(tmp_path / "no_extension") == 0.0


def test_csv_named_file_with_non_csv_content_fails_the_sniff(tmp_path):
    """Verify the content sniff catches a .csv extension that lies about its content."""

    path = _csv(tmp_path, "not_really.csv", '{"fps": 30, "codec": "h264"}\n')

    assert CsvAdapter().detect(path) == 0.0


def test_field_over_the_csv_limit_bids_zero_rather_than_raising(tmp_path):
    """Verify an over-limit first-line field bids 0.0, not raises.

    `csv.reader` raises `csv.Error` once a single field exceeds
    `csv.field_size_limit()`, 131072 bytes by default.
    """

    path = _csv(tmp_path, "huge_field.csv", "a" * 200_000 + ",b\n1,2\n")

    assert CsvAdapter().detect(path) == 0.0


def test_a_non_utf8_encoded_csv_bids_zero(tmp_path):
    """Verify a non-UTF-8-encoded CSV fails the sniff and bids 0.0, not raises.

    A latin-1-encoded header with `°C` is ordinary for a sensor CSV but is
    not valid UTF-8, so decoding it under the platform's default text
    encoding raises `UnicodeDecodeError`.
    """

    path = tmp_path / "sensor.csv"
    path.write_bytes("temp_°C,time_µs\n1,2\n".encode("latin-1"))

    assert CsvAdapter().detect(path) == 0.0


def test_a_detect_that_cannot_open_the_file_bids_zero(monkeypatch, tmp_path):
    """Verify detect() bids 0.0 rather than raising when opening the file fails."""

    path = _csv(tmp_path, "mystery.csv", "a,b\n1,2\n")
    original_open = Path.open

    def _boom(self, *args, **kwargs):
        if self == path:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _boom)

    assert CsvAdapter().detect(path) == 0.0


def test_discover_adapters_finds_csv_through_the_entry_point():
    """Verify a real discovery run surfaces an adapter named `csv`."""

    result = discover_adapters()
    assert "csv" in {loaded.name for loaded in result.adapters}


# --- Ported from tests/test_loading.py ------------------------------------


def test_the_fixtures_four_ids_each_become_their_own_instance_id_sorted():
    """Verify the fixture's four ids each become their own instance, id-sorted."""

    episode = next(CsvAdapter().episodes(CSV_FIXTURE))

    seen: list[str | None] = []
    for stream in episode.streams:
        if stream.instance not in seen:
            seen.append(stream.instance)
    assert seen == ["armA", "armB", "armC", "armD"]


def test_each_instances_stream_holds_only_its_own_rows_in_increasing_time_order():
    """Verify no cross-instance leakage, and every stream's timestamps increase."""

    episode = next(CsvAdapter().episodes(CSV_FIXTURE))
    counts = {"armA": 50, "armB": 50, "armC": 45, "armD": 50}

    for stream in episode.streams:
        assert stream.payload is not None
        assert len(stream.payload) == counts[stream.instance]
        timestamps = stream.timestamps.to_list()
        assert timestamps == sorted(timestamps)


def test_an_out_of_order_source_is_sorted_per_instance(tmp_path):
    """Verify rows out of time order in the source still come out sorted.

    A fully descending series never clears inference's monotonicity gate,
    so only the last two rows are swapped, which still forces the sort.
    """

    times = list(range(0, 100, 10))
    times[-1], times[-2] = times[-2], times[-1]
    lines = ["t_ms,id,x"] + [f"{t},armA,{t}" for t in times]
    path = _csv(tmp_path, "reordered.csv", "\n".join(lines) + "\n")

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    timestamps = stream.timestamps.to_list()
    assert timestamps == sorted(timestamps)


def test_t_ms_becomes_canonical_seconds(tmp_path):
    """Verify the numeric conversion on a hand-written millisecond grid."""

    path = _csv(tmp_path, "ms_grid.csv", "t_ms,x\n0,1.0\n10,2.0\n20,3.0\n")

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.timestamps.to_list() == [0.0, 0.01, 0.02]


def test_a_mystery_column_reaches_the_payload_intact(tmp_path):
    """Verify a column the adapter has no reason to touch reaches the payload intact."""

    path = _csv(tmp_path, "mystery.csv", "t_ms,id,mystery\n0,a,7\n10,a,9\n")

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.payload is not None
    frame = stream.payload.fetch()
    assert frame.columns == ["mystery"]
    assert frame["mystery"].to_list() == [7, 9]


def test_a_keyless_source_yields_one_single_group_with_no_instance(tmp_path):
    """Verify a keyless source yields Attribution.SINGLE, no file-stem placeholder."""

    path = _csv(tmp_path, "keyless.csv", "t_ms,x\n0,1.0\n10,2.0\n20,3.0\n")

    episode = next(CsvAdapter().episodes(path))

    assert episode.streams
    for stream in episode.streams:
        assert stream.instance is None
        assert stream.attribution is Attribution.SINGLE


def test_an_unknown_time_unit_leaves_the_time_column_unconverted(tmp_path):
    """Verify no conversion happens when inference cannot resolve a unit."""

    path = _csv(tmp_path, "unknown_unit.csv", "timestamp,x\n5,1\n3,2\n9,3\n")

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.timestamps.to_list() == [3, 5, 9]


def test_a_preexisting_unrelated_time_s_column_raises_adapter_refusal(tmp_path):
    """Verify normalising into an existing unrelated `time_s` column raises."""

    path = _csv(
        tmp_path,
        "collision.csv",
        "t_ms,time_s,x\n0,5,1.0\n10,3,2.0\n20,9,3.0\n30,1,4.0\n",
    )

    with pytest.raises(AdapterRefusal, match="time_s"):
        next(CsvAdapter().episodes(path))


def test_a_time_columns_own_time_s_name_is_rescaled_in_place(tmp_path):
    """Verify a source's own `time_s` column is rescaled, not treated as a collision."""

    path = _csv(tmp_path, "already_named.csv", "time_s,x\n0,1.0\n10,2.0\n20,3.0\n")

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.timestamps.to_list() == [0.0, 0.01, 0.02]


def test_a_file_with_no_resolvable_time_axis_raises_adapter_refusal(tmp_path):
    """Verify no candidate time column raises AdapterRefusal with the schema so far."""

    path = _csv(tmp_path, "no_time.csv", "a,b\n1.0,2.0\n1.1,2.2\n1.2,2.3\n")

    with pytest.raises(AdapterRefusal) as excinfo:
        next(CsvAdapter().episodes(path))

    unresolved = excinfo.value.as_unresolved()
    assert unresolved.schema_so_far.columns
    assert "time axis" in unresolved.reason


# --- Ported from tests/test_mapping.py -------------------------------------


def test_every_numeric_column_in_the_fixture_becomes_exactly_one_channel_per_instance():
    """Verify each axis appears once per instance, not zero, not twice."""

    episode = next(CsvAdapter().episodes(CSV_FIXTURE))

    by_instance: dict[str | None, list[str]] = {}
    for stream in episode.streams:
        names = by_instance.setdefault(stream.instance, [])
        names.extend(channel.name for channel in stream.channels)
    assert by_instance
    for names in by_instance.values():
        assert sorted(names) == ["tcp_pose_x_mm", "tcp_pose_y_mm", "tcp_pose_z_mm"]


def test_the_three_pose_axes_land_in_one_stream_keyed_by_the_shared_stem():
    """Verify tcp_pose_x_mm, tcp_pose_y_mm and tcp_pose_z_mm form one stream."""

    episode = next(CsvAdapter().episodes(CSV_FIXTURE))

    assert len(episode.streams) == 4
    stream = episode.streams[0]
    assert stream.taxonomy_type == "proprio.ee_pose"
    assert stream.source_field == "tcp_pose"
    assert {channel.name for channel in stream.channels} == {
        "tcp_pose_x_mm",
        "tcp_pose_y_mm",
        "tcp_pose_z_mm",
    }


def test_two_unrelated_unsuffixed_columns_stay_in_separate_streams(tmp_path):
    """Verify ungrouped columns don't merge just for sharing a lack of axis suffix."""

    path = _csv(tmp_path, "flat.csv", "t_ms,voltage,counter\n0,1,3\n10,2,4\n")

    episode = next(CsvAdapter().episodes(path))

    assert [stream.taxonomy_type for stream in episode.streams] == [
        "unmapped.voltage",
        "unmapped.counter",
    ]
    assert [
        channel.name for stream in episode.streams for channel in stream.channels
    ] == ["voltage", "counter"]


def test_an_ungrouped_column_interleaved_with_a_stem_still_groups_correctly(tmp_path):
    """Verify a mystery column between two axis members doesn't split the stem."""

    path = _csv(
        tmp_path,
        "mixed.csv",
        "t_ms,tcp_pose_x_mm,mystery,tcp_pose_y_mm,tcp_pose_z_mm\n"
        "0,1.0,7,3.0,5.0\n10,2.0,9,4.0,6.0\n",
    )

    episode = next(CsvAdapter().episodes(path))

    assert [
        (stream.source_field, [channel.name for channel in stream.channels])
        for stream in episode.streams
    ] == [
        ("tcp_pose", ["tcp_pose_x_mm", "tcp_pose_y_mm", "tcp_pose_z_mm"]),
        ("mystery", ["mystery"]),
    ]


def test_a_stem_the_dictionary_claims_carries_its_taxonomy_type():
    """Verify every instance's stream carries the dictionary's taxonomy type."""

    episode = next(CsvAdapter().episodes(CSV_FIXTURE))

    assert episode.streams
    for stream in episode.streams:
        assert stream.taxonomy_type == "proprio.ee_pose"


def test_a_non_numeric_column_ending_in_an_axis_letter_is_ignored(tmp_path):
    """Verify the numeric-dtype gate runs before the axis-suffix match, not after."""

    path = _csv(
        tmp_path,
        "stringy.csv",
        "t_ms,label_x,tcp_pose_x_mm\n0,a,1.0\n10,b,2.0\n20,c,3.0\n",
    )

    episode = next(CsvAdapter().episodes(path))

    channel_names = {
        channel.name for stream in episode.streams for channel in stream.channels
    }
    assert channel_names == {"tcp_pose_x_mm"}


def test_a_streams_payload_holds_only_its_own_channels(tmp_path):
    """Verify each stream's payload is narrowed to the columns it declares."""

    path = _csv(tmp_path, "flat.csv", "t_ms,voltage,counter\n0,1,3\n10,2,4\n")

    episode = next(CsvAdapter().episodes(path))

    columns = []
    for stream in episode.streams:
        assert stream.payload is not None
        columns.append(stream.payload.fetch().columns)
    assert columns == [["voltage"], ["counter"]]


def test_an_ambiguous_stem_stays_unmapped_rather_than_picking_a_claimant(
    tmp_path, caplog
):
    """Verify two entries claiming one alias leave the stream unmapped and logged."""

    dictionary = Dictionary.model_validate(
        {
            "schema_version": 1,
            "entries": {
                "proprio.joint_position": {**_BARE_ENTRY, "aliases": ["pos"]},
                "derived.tcp_position": {**_BARE_ENTRY, "aliases": ["pos"]},
            },
        }
    )
    path = _csv(tmp_path, "ambiguous.csv", "t_ms,pos\n0,1.0\n10,2.0\n")

    with caplog.at_level(logging.WARNING):
        episode = next(CsvAdapter(dictionary=dictionary).episodes(path))

    [stream] = episode.streams
    assert stream.taxonomy_type == "unmapped.pos"
    assert any(
        "pos" in record.getMessage() and "ambiguous" in record.getMessage()
        for record in caplog.records
    )


def test_numerically_indexed_columns_group_into_one_stream(tmp_path):
    """Verify q_0, q_1, q_2 group under the shared stem `q`."""

    path = _csv(
        tmp_path, "indexed.csv", "t_ms,q_0,q_1,q_2\n0,1.0,3.0,5.0\n10,2.0,4.0,6.0\n"
    )

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.taxonomy_type == "proprio.joint_position"
    assert {channel.name for channel in stream.channels} == {"q_0", "q_1", "q_2"}


def test_axes_carrying_a_single_letter_unit_group_into_one_stream(tmp_path):
    """Verify accel_x_g, accel_y_g, accel_z_g group under the shared stem `accel`."""

    path = _csv(
        tmp_path,
        "accel.csv",
        "t_ms,accel_x_g,accel_y_g,accel_z_g\n0,0.0,0.0,1.0\n10,0.0,0.0,1.0\n",
    )

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert {channel.name for channel in stream.channels} == {
        "accel_x_g",
        "accel_y_g",
        "accel_z_g",
    }


def test_a_column_that_normalises_to_nothing_keeps_its_own_name(tmp_path):
    """Verify an all-punctuation column stays its own stream rather than collapsing."""

    path = _csv(tmp_path, "punct.csv", "t_ms,__,voltage\n0,1,3\n10,2,4\n")

    episode = next(CsvAdapter().episodes(path))

    assert len(episode.streams) == 2
    assert {stream.taxonomy_type for stream in episode.streams} == {
        "unmapped.__",
        "unmapped.voltage",
    }


# --- New: the defects this issue fixes -------------------------------------


def test_a_null_instance_key_becomes_its_own_unattributed_group(tmp_path, caplog):
    """Verify a null instance key surfaces as its own Attribution.UNATTRIBUTED group.

    No stream reads `instance == "None"`; the real ids are unaffected;
    a WARNING names the row count.
    """

    rows = [
        "t_ms,id,x",
        "0,armA,1.0",
        "10,armA,2.0",
        "20,,3.0",
        "30,armB,4.0",
        "40,,5.0",
        "50,armB,6.0",
    ]
    path = _csv(tmp_path, "null_key.csv", "\n".join(rows) + "\n")

    with caplog.at_level(logging.WARNING):
        episode = next(CsvAdapter().episodes(path))

    for stream in episode.streams:
        assert stream.instance != "None"

    unattributed = [
        stream
        for stream in episode.streams
        if stream.attribution is Attribution.UNATTRIBUTED
    ]
    assert unattributed
    assert all(stream.instance is None for stream in unattributed)

    keyed_ids = {
        stream.instance
        for stream in episode.streams
        if stream.attribution is Attribution.KEYED
    }
    assert keyed_ids == {"armA", "armB"}

    assert any(
        "2 row(s) had no value for the instance key" in record.getMessage()
        for record in caplog.records
    )


def test_bare_axis_columns_group_under_the_file_stem(tmp_path):
    """Verify bare x, y, z columns group under the file's own stem.

    A filename is not a signal name — the group types as `unmapped.<file stem>`.
    """

    path = _csv(
        tmp_path,
        "bare_axes.csv",
        "t_ms,x,y,z\n0,1.0,2.0,3.0\n10,1.1,2.1,3.1\n20,1.2,2.2,3.2\n",
    )

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.source_field == "bare_axes"
    assert stream.taxonomy_type == "unmapped.bare_axes"
    assert [channel.name for channel in stream.channels] == ["x", "y", "z"]
    assert [channel.axis for channel in stream.channels] == ["x", "y", "z"]


def test_uppercase_axis_suffixes_group_correctly(tmp_path):
    """Verify Arm_X_mm/Arm_Y_mm/Arm_Z_mm group under `arm` with axes x, y, z."""

    path = _csv(
        tmp_path,
        "upper_axes.csv",
        "t_ms,Arm_X_mm,Arm_Y_mm,Arm_Z_mm\n0,1.0,2.0,3.0\n10,1.1,2.1,3.1\n",
    )

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.source_field == "arm"
    assert [channel.axis for channel in stream.channels] == ["x", "y", "z"]


def test_a_lone_w_column_is_not_mistaken_for_an_axis(tmp_path):
    """Verify power_w alone in its group carries axis=None and stays stem power."""

    path = _csv(tmp_path, "power_w.csv", "t_ms,power_w\n0,1.0\n10,2.0\n")

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.source_field == "power"
    assert [channel.axis for channel in stream.channels] == [None]


def test_a_w_axis_with_xyz_siblings_keeps_its_axis(tmp_path):
    """Verify a rot_x/y/z/w quaternion keeps `w` as a real axis."""

    path = _csv(
        tmp_path,
        "quat_w.csv",
        "t_ms,rot_x,rot_y,rot_z,rot_w\n0,0.0,0.0,0.0,1.0\n10,0.0,0.0,0.0,1.0\n",
    )

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    axes = {channel.name: channel.axis for channel in stream.channels}
    assert axes == {"rot_x": "x", "rot_y": "y", "rot_z": "z", "rot_w": "w"}


def test_duplicate_axis_claimants_both_survive_and_are_logged(tmp_path, caplog):
    """Verify arm_x_mm and arm_x_deg both survive as channels, with a WARNING."""

    path = _csv(
        tmp_path, "dup_axis.csv", "t_ms,arm_x_mm,arm_x_deg\n0,1.0,10.0\n10,1.1,10.1\n"
    )

    with caplog.at_level(logging.WARNING):
        episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert {channel.name for channel in stream.channels} == {"arm_x_mm", "arm_x_deg"}
    assert any(
        "arm" in record.getMessage()
        and "'x'" in record.getMessage()
        and "arm_x_mm" in record.getMessage()
        and "arm_x_deg" in record.getMessage()
        for record in caplog.records
    )


def test_a_hand_written_indexed_group_keeps_its_indices_even_when_dictionary_claimed(
    tmp_path,
):
    """Verify gyro_0/1/2 in a CSV stays indexed rather than being promoted to x/y/z.

    `CsvAdapter` declares no `array_stems`, so the promotion `TabularAdapter`
    applies to a JSON-array-derived group never triggers here — even though
    `gyro` is a dictionary alias for `extero.imu` with `group_hint: axes`.
    A hand-written index is not necessarily an axis order.
    """

    path = _csv(
        tmp_path,
        "gyro_indexed.csv",
        "t_ms,gyro_0,gyro_1,gyro_2\n0,0.0,0.1,0.2\n10,0.1,0.2,0.3\n",
    )

    episode = next(CsvAdapter().episodes(path))

    [stream] = episode.streams
    assert stream.taxonomy_type == "extero.imu"
    assert [channel.axis for channel in stream.channels] == ["0", "1", "2"]
