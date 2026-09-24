"""Verifies TabularAdapter: parse-then-infer, written once for table-shaped formats."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from pathlib import Path

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters.tabular import ParsedTable, TabularAdapter
from kalanos.analysis.models.adapters import AdapterRefusal
from kalanos.analysis.models.domain import Attribution

# Local
from helpers import CSV_FIXTURE


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _CountingAdapter(TabularAdapter):
    """A tabular adapter writing only `name`, `detect` and `parse`, counting reads."""

    name = "counting"
    reads = 0

    def detect(self, path: UPath) -> float:
        """Bid confidently on any path, since detection isn't under test."""

        return 1.0

    def parse(self, path: UPath) -> ParsedTable:
        """Read the CSV, counting the read."""

        self.reads += 1
        with path.open("rb") as handle:
            return ParsedTable(frame=pl.read_csv(handle))


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _csv(tmp_path: Path, name: str, text: str) -> UPath:
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
    UPath
        The written file's path.
    """

    path = tmp_path / name
    path.write_text(text)
    return UPath(path)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_writing_only_parse_yields_a_valid_episode():
    """Verify a subclass writing only `parse` produces valid episodes."""

    adapter = _CountingAdapter()
    episodes = list(adapter.episodes(CSV_FIXTURE))

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.id == CSV_FIXTURE.stem
    assert episode.streams
    for stream in episode.streams:
        assert stream.payload is not None
        assert len(stream.payload) == len(stream.timestamps)


def test_the_file_is_parsed_once_per_run():
    """Verify draining `episodes` reads the file exactly once."""

    adapter = _CountingAdapter()
    list(adapter.episodes(CSV_FIXTURE))
    assert adapter.reads == 1


def test_describe_does_not_parse():
    """Verify `describe` answers without reading the file."""

    adapter = _CountingAdapter()
    info = adapter.describe(CSV_FIXTURE)
    assert adapter.reads == 0
    assert info.adapter == adapter.name


def test_sample_zero_reads_nothing():
    """Verify `sample=0` yields no episodes and reads nothing."""

    adapter = _CountingAdapter()
    episodes = list(adapter.episodes(CSV_FIXTURE, sample=0))
    assert episodes == []
    assert adapter.reads == 0


def test_iterating_twice_yields_the_same_episodes():
    """Verify calling `episodes` twice gives equal episodes both times."""

    adapter = _CountingAdapter()
    first = list(adapter.episodes(CSV_FIXTURE))
    second = list(adapter.episodes(CSV_FIXTURE))

    assert [episode.id for episode in first] == [episode.id for episode in second]

    def _triples(episodes):
        return [
            (stream.taxonomy_type, stream.instance, stream.attribution)
            for episode in episodes
            for stream in episode.streams
        ]

    assert _triples(first) == _triples(second)


def test_time_converts_to_canonical_seconds(tmp_path):
    """Verify a millisecond time grid comes back as canonical seconds."""

    path = _csv(tmp_path, "ms_grid.csv", "t_ms,x\n0,1.0\n10,2.0\n20,3.0\n")
    adapter = _CountingAdapter()
    episode = next(adapter.episodes(path))

    assert episode.streams
    assert episode.streams[0].timestamps.to_list() == [0.0, 0.01, 0.02]


def test_a_keyed_source_tags_each_stream_with_instance_and_keyed(tmp_path):
    """Verify a keyed source's streams carry their subject and Attribution.KEYED."""

    rows = ["t_ms,id,x"]
    for t in range(0, 200, 10):
        arm = "armA" if (t // 10) % 2 == 0 else "armB"
        rows.append(f"{t},{arm},{float(t)}")
    path = _csv(tmp_path, "keyed.csv", "\n".join(rows) + "\n")

    adapter = _CountingAdapter()
    episode = next(adapter.episodes(path))

    assert episode.streams
    for stream in episode.streams:
        assert stream.instance in {"armA", "armB"}
        assert stream.attribution is Attribution.KEYED


def test_a_keyless_source_yields_none_instance_and_single(tmp_path):
    """Verify a keyless source's streams have no instance and Attribution.SINGLE.

    No file-stem placeholder anywhere: the episode already carries the path.
    """

    path = _csv(tmp_path, "keyless.csv", "t_ms,x\n0,1.0\n10,2.0\n20,3.0\n")
    adapter = _CountingAdapter()
    episode = next(adapter.episodes(path))

    assert episode.streams
    for stream in episode.streams:
        assert stream.instance is None
        assert stream.attribution is Attribution.SINGLE


def test_a_null_instance_key_becomes_its_own_unattributed_group(tmp_path, caplog):
    """Verify a null instance key surfaces as its own Attribution.UNATTRIBUTED group.

    No stream reads `instance == "None"`; the real ids are unaffected;
    a WARNING names the file and the row count.
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
    adapter = _CountingAdapter()

    with caplog.at_level(logging.WARNING):
        episode = next(adapter.episodes(path))

    for stream in episode.streams:
        assert stream.instance != "None"

    unattributed = [
        stream
        for stream in episode.streams
        if stream.attribution is Attribution.UNATTRIBUTED
    ]
    assert unattributed
    assert all(stream.instance is None for stream in unattributed)
    payload = unattributed[0].payload
    assert payload is not None
    assert len(payload) == 2

    keyed_ids = {
        stream.instance
        for stream in episode.streams
        if stream.attribution is Attribution.KEYED
    }
    assert keyed_ids == {"armA", "armB"}

    assert any(
        "2 row(s) had no value for the instance key" in record.message
        for record in caplog.records
    )


def test_channels_order_by_axis(tmp_path):
    """Verify axis-suffixed columns come back in x, y, z order with axis set."""

    path = _csv(
        tmp_path,
        "axes.csv",
        "t_ms,pose_z,pose_x,pose_y\n0,3.0,1.0,2.0\n10,3.1,1.1,2.1\n20,3.2,1.2,2.2\n",
    )
    adapter = _CountingAdapter()
    episode = next(adapter.episodes(path))

    stream = next(s for s in episode.streams if s.source_field == "pose")
    names = [channel.name for channel in stream.channels]
    assert names == ["pose_x", "pose_y", "pose_z"]
    assert [channel.axis for channel in stream.channels] == ["x", "y", "z"]
    assert stream.payload is not None
    assert stream.payload.fetch().columns == ["pose_x", "pose_y", "pose_z"]


def test_numerically_indexed_columns_order_ascending(tmp_path):
    """Verify numeric indices order ascending, including past ten."""

    path = _csv(
        tmp_path,
        "indexed.csv",
        "t_ms,q_10,q_2\n0,10.0,2.0\n10,10.1,2.1\n20,10.2,2.2\n",
    )
    adapter = _CountingAdapter()
    episode = next(adapter.episodes(path))

    stream = next(s for s in episode.streams if s.source_field == "q")
    assert [channel.name for channel in stream.channels] == ["q_2", "q_10"]


def test_an_unsuffixed_column_keeps_its_source_position(tmp_path):
    """Verify an unsuffixed column stays where it was among other unsuffixed members."""

    path = _csv(
        tmp_path,
        "mixed.csv",
        "t_ms,torque,observation.torque,torque_x\n"
        "0,1.0,2.0,3.0\n10,1.1,2.1,3.1\n20,1.2,2.2,3.2\n",
    )
    adapter = _CountingAdapter()
    episode = next(adapter.episodes(path))

    stream = next(s for s in episode.streams if s.source_field == "torque")
    assert [channel.name for channel in stream.channels] == [
        "torque_x",
        "torque",
        "observation.torque",
    ]


def test_an_unresolved_column_becomes_an_unmapped_stream(tmp_path):
    """Verify a column the dictionary does not claim yields an unmapped.* stream."""

    path = _csv(tmp_path, "unmapped.csv", "t_ms,widget_flux\n0,1.0\n10,2.0\n20,3.0\n")
    adapter = _CountingAdapter()
    episode = next(adapter.episodes(path))

    stream = next(s for s in episode.streams if s.source_field == "widget_flux")
    assert stream.taxonomy_type == "unmapped.widget_flux"
    assert [channel.name for channel in stream.channels] == ["widget_flux"]


def test_a_file_with_no_resolvable_time_axis_raises_adapter_refusal(tmp_path):
    """Verify no candidate time column raises AdapterRefusal with the schema so far."""

    path = _csv(tmp_path, "no_time.csv", "a,b\n1.0,2.0\n1.1,2.2\n1.2,2.3\n")
    adapter = _CountingAdapter()

    with pytest.raises(AdapterRefusal) as excinfo:
        next(adapter.episodes(path))

    refusal = excinfo.value
    unresolved = refusal.as_unresolved()
    assert unresolved.schema_so_far.columns
    assert "time axis" in unresolved.reason


def test_a_preexisting_unrelated_time_s_column_raises_adapter_refusal(tmp_path):
    """Verify normalising into an existing unrelated `time_s` column raises."""

    path = _csv(
        tmp_path,
        "collision.csv",
        "t_ms,time_s,x\n0,5,1.0\n10,3,2.0\n20,9,3.0\n30,1,4.0\n",
    )
    adapter = _CountingAdapter()

    with pytest.raises(AdapterRefusal, match="time_s"):
        next(adapter.episodes(path))
