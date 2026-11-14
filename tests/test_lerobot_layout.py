"""Verifies how a LeRobot episode's rows are located across data files, how that is
compared with the episode's metadata, and the two schema metrics that report it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

# External
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.layout import (
    layout_check,
    read_episode_rows,
    scan_episode_rows,
)
from kalanos.analysis.metrics.schema import (
    episode_file_mismatch,
    episode_not_in_metadata,
    index_range_mismatch_count,
)
from kalanos.analysis.models.domain import Episode, LayoutCheck, RangeMismatch
from kalanos.analysis.models.metrics import EpisodeContext, MetricStatus


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_FILE_0 = UPath("/data/chunk-000/file-000.parquet")
_FILE_1 = UPath("/data/chunk-000/file-001.parquet")

# A metadata row whose range matches rows [10, 20)
_ROW = {"dataset_from_index": 10, "dataset_to_index": 20, "length": 10}


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _RangeServer(ThreadingHTTPServer):
    """An HTTP server over one in-memory payload that counts the bytes it sends."""

    def __init__(self, payload: bytes):
        super().__init__(("127.0.0.1", 0), _RangeHandler)
        self.payload = payload
        self.sent = 0
        self.lock = threading.Lock()


class _RangeHandler(BaseHTTPRequestHandler):
    """Serve the payload, honouring single `Range` requests."""

    def do_HEAD(self):
        """Answer with the headers a GET would carry."""

        self._serve(send_body=False)

    def do_GET(self):
        """Answer with the requested bytes."""

        self._serve(send_body=True)

    def _serve(self, *, send_body):
        """Send the whole payload, or the byte range the request asks for.

        Parameters
        ----------
        send_body : bool
            Whether to write the body after the headers.
        """

        server = cast(_RangeServer, self.server)
        data = server.payload
        requested = self.headers.get("Range")
        if requested:
            start, end = requested.removeprefix("bytes=").split("-")
            if start:
                first = int(start)
                last = min(int(end) if end else len(data) - 1, len(data) - 1)
            else:
                first, last = len(data) - int(end), len(data) - 1
            body = data[first : last + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if send_body:
            self.wfile.write(body)
            with server.lock:
                server.sent += len(body)

    def log_message(self, format, *args):
        """Keep the test output quiet."""


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _rows(source, episode, index_from=None, index_to=None):
    """Build one row of `read_episode_rows` output.

    Parameters
    ----------
    source : int
        The data file's position.
    episode : int
        The episode index.
    index_from, index_to : int or None
        The index range, left out when `None`.

    Returns
    -------
    pl.DataFrame
        The row.
    """

    columns = {
        "source": pl.Series([source], dtype=pl.UInt32),
        "episode_index": pl.Series([episode], dtype=pl.Int64),
        "rows": pl.Series([1], dtype=pl.UInt32),
    }
    if index_from is not None:
        columns["index_from"] = pl.Series([index_from], dtype=pl.Int64)
        columns["index_to"] = pl.Series([index_to], dtype=pl.Int64)
    return pl.DataFrame(columns)


def _data(episodes):
    """Build a data parquet's two located columns.

    Parameters
    ----------
    episodes : list[tuple[int, int, int]]
        `(episode_index, index_from, index_to)` per episode.

    Returns
    -------
    pl.DataFrame
        `episode_index` and `index`, one row per index.
    """

    return pl.DataFrame(
        {
            "episode_index": [e for e, lo, hi in episodes for _ in range(lo, hi)],
            "index": [i for _, lo, hi in episodes for i in range(lo, hi)],
        }
    )


def _context(check):
    """Wrap a layout check in the context a metric reads.

    Parameters
    ----------
    check : LayoutCheck or None
        The episode's layout check.

    Returns
    -------
    EpisodeContext
        The context.
    """

    return EpisodeContext(episode=Episode(id="e", layout_check=check))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_scan_reads_only_the_two_columns(tmp_path, monkeypatch):
    """Verify a scan over HTTP fetches a small fraction of a large data file."""

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    n = 200_000
    # Hashed values compress poorly, so `action` dominates the file size.
    action = [
        pl.int_range(n).hash(seed).cast(pl.Float64).alias(f"a{seed}")
        for seed in range(16)
    ]
    frame = pl.select(
        episode_index=pl.int_range(n) % 100,
        index=pl.int_range(n),
        action=pl.concat_list(action),
    )
    file_path = tmp_path / "file-000.parquet"
    frame.write_parquet(file_path, row_group_size=50_000)
    payload = file_path.read_bytes()

    server = _RangeServer(payload)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/file-000.parquet"
        located = scan_episode_rows([url], None, with_index=True)
    finally:
        server.shutdown()
        server.server_close()

    assert located.height == 100
    first = located.row(0, named=True)
    last = located.row(99, named=True)
    assert (first["episode_index"], first["rows"]) == (0, 2_000)
    assert (first["index_from"], first["index_to"]) == (0, 199_901)
    assert (last["episode_index"], last["index_from"], last["index_to"]) == (
        99,
        99,
        200_000,
    )
    assert server.sent < 0.10 * len(payload)


@pytest.mark.parametrize("with_index", [True, False])
def test_read_episode_rows_falls_back_to_handles(tmp_path, with_index):
    """Verify a file with no direct location gives the same frame as a local one."""

    data = _data([(0, 0, 5), (1, 5, 8)])
    local = UPath(tmp_path) / "file-000.parquet"
    data.write_parquet(local.path)
    memory = UPath("memory://layout/file-000.parquet")
    with memory.open("wb") as handle:
        data.write_parquet(handle)

    native = read_episode_rows([local], with_index=with_index)
    fallback = read_episode_rows([memory], with_index=with_index)

    assert_frame_equal(native, fallback)
    assert native["rows"].to_list() == [5, 3]
    if with_index:
        assert native["index_from"].to_list() == [0, 5]
        assert native["index_to"].to_list() == [5, 8]


def test_read_episode_rows_tags_each_file_by_position(tmp_path):
    """Verify `source` indexes the paths, so a row maps back to its file."""

    paths = [UPath(tmp_path) / f"file-00{i}.parquet" for i in range(2)]
    _data([(0, 0, 3)]).write_parquet(paths[0].path)
    _data([(1, 3, 7), (2, 7, 9)]).write_parquet(paths[1].path)

    located = read_episode_rows(paths, with_index=True)

    assert located.select("source", "episode_index").rows() == [(0, 0), (1, 1), (1, 2)]


def test_read_episode_rows_accepts_mixed_integer_widths(tmp_path):
    """Verify files storing the located columns as Int32 and Int64 read together."""

    paths = [UPath(tmp_path) / f"file-00{i}.parquet" for i in range(2)]
    _data([(0, 0, 3)]).cast(pl.Int32).write_parquet(paths[0].path)
    _data([(1, 3, 7)]).write_parquet(paths[1].path)

    located = read_episode_rows(paths, with_index=True)

    assert located.select("episode_index", "index_from", "index_to").rows() == [
        (0, 0, 3),
        (1, 3, 7),
    ]


def test_layout_check_with_a_matching_pointer():
    """Verify rows in the pointed file match it, and a matching range raises nothing."""

    check = layout_check(_ROW, _FILE_0, _rows(0, 7, 10, 20), [_FILE_0])

    assert check.pointer_matches
    assert check.holding_paths == [_FILE_0]
    assert check.range_mismatches == []
    assert check.pointer_effect is not None
    assert "KeyError" in check.pointer_effect


def test_layout_check_with_rows_in_another_file():
    """Verify rows found only in another file break the pointer."""

    check = layout_check(_ROW, _FILE_0, _rows(1, 7, 10, 20), [_FILE_0, _FILE_1])

    assert not check.pointer_matches
    assert check.holding_paths == [_FILE_1]
    assert check.range_mismatches == []


def test_layout_check_with_rows_in_no_file():
    """Verify an episode found nowhere has no holding file and no range compared."""

    check = layout_check(_ROW, _FILE_0, _rows(0, 7, 10, 20).clear(), [_FILE_0])

    assert check.holding_paths == []
    assert check.range_mismatches is None


@pytest.mark.parametrize(
    "row",
    [
        {"dataset_from_index": 10, "dataset_to_index": 20},
        {"dataset_from_index": 10, "dataset_to_index": 20, "length": None},
    ],
)
def test_layout_check_without_every_range_field(row):
    """Verify the range is not compared when the metadata lacks one of its fields."""

    check = layout_check(row, _FILE_0, _rows(0, 7, 10, 20), [_FILE_0])

    assert check.range_mismatches is None


def test_layout_check_without_the_index_column():
    """Verify the range is not compared when the data has no `index` column."""

    check = layout_check(_ROW, _FILE_0, _rows(0, 7), [_FILE_0])

    assert check.pointer_matches
    assert check.range_mismatches is None


def test_layout_check_compares_length_with_the_rows_found():
    """Verify a self-consistent but wrong range reports the two indices only."""

    row = {"dataset_from_index": 605, "dataset_to_index": 1224, "length": 619}

    check = layout_check(row, _FILE_1, _rows(0, 397, 336051, 336670), [_FILE_1])

    assert check.range_mismatches == [
        RangeMismatch(field="dataset_from_index", declared=605, found=336051),
        RangeMismatch(field="dataset_to_index", declared=1224, found=336670),
    ]


def test_layout_check_reports_a_wrong_length():
    """Verify a `length` that disagrees with the range found is reported alone."""

    row = {"dataset_from_index": 10, "dataset_to_index": 20, "length": 12}

    check = layout_check(row, _FILE_0, _rows(0, 7, 10, 20), [_FILE_0])

    assert check.range_mismatches == [
        RangeMismatch(field="length", declared=12, found=10)
    ]


@pytest.mark.parametrize(
    "metric",
    [episode_file_mismatch, episode_not_in_metadata, index_range_mismatch_count],
)
def test_the_metrics_are_not_applicable_without_a_layout_check(metric):
    """Verify an episode no adapter located reads as not applicable."""

    result = metric(_context(None))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert result.evidence["reason"] == "the source names no data file per episode"


def test_the_pointer_metric_reads_a_mismatch_when_no_file_holds_the_episode():
    """Verify an episode found in no data file reads 1, naming the file it points at."""

    check = LayoutCheck(pointed_path=_FILE_0, pointer_effect="loads nothing")

    result = episode_file_mismatch(_context(check))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 1.0
    assert result.evidence == {
        "pointed_path": str(_FILE_0),
        "holding_paths": [],
        "effect": "loads nothing",
    }


def test_the_pointer_metric_is_not_applicable_without_a_metadata_row():
    """Verify an episode found in the data alone is left to the metadata check."""

    check = LayoutCheck(pointed_path=None, holding_paths=[_FILE_0])

    result = episode_file_mismatch(_context(check))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert (
        result.evidence["reason"]
        == "the episode has no metadata row naming a data file"
    )


def test_the_metadata_metric_flags_an_episode_found_in_the_data_alone():
    """Verify an episode without a metadata row reads 1 and names its data files."""

    check = LayoutCheck(pointed_path=None, holding_paths=[_FILE_0, _FILE_1])

    result = episode_not_in_metadata(_context(check))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 1.0
    assert result.evidence == {"holding_paths": [str(_FILE_0), str(_FILE_1)]}


def test_the_metadata_metric_passes_an_episode_with_a_metadata_row():
    """Verify an episode with a metadata row reads 0, found in its file or not."""

    for check in (
        LayoutCheck(pointed_path=_FILE_0, holding_paths=[_FILE_0]),
        LayoutCheck(pointed_path=_FILE_0),
    ):
        result = episode_not_in_metadata(_context(check))

        assert result.status == MetricStatus.REPORT_ONLY
        assert result.value == 0.0


def test_the_pointer_metric_on_a_match():
    """Verify a matching pointer reads 0 and names the file, with no effect."""

    check = LayoutCheck(
        pointed_path=_FILE_0, holding_paths=[_FILE_0], pointer_effect="loads nothing"
    )

    result = episode_file_mismatch(_context(check))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 0.0
    assert result.evidence == {
        "pointed_path": str(_FILE_0),
        "holding_paths": [str(_FILE_0)],
    }


def test_the_pointer_metric_on_a_mismatch():
    """Verify a wrong pointer reads 1 and names both files and the adapter's effect."""

    check = LayoutCheck(
        pointed_path=_FILE_0, holding_paths=[_FILE_1], pointer_effect="loads nothing"
    )

    result = episode_file_mismatch(_context(check))

    assert result.value == 1.0
    assert result.evidence == {
        "pointed_path": str(_FILE_0),
        "holding_paths": [str(_FILE_1)],
        "effect": "loads nothing",
    }


def test_the_pointer_metric_on_a_mismatch_without_an_effect():
    """Verify a wrong pointer from an adapter that states no effect carries none."""

    check = LayoutCheck(pointed_path=_FILE_0, holding_paths=[_FILE_1])

    result = episode_file_mismatch(_context(check))

    assert result.value == 1.0
    assert "effect" not in result.evidence


def test_the_range_metric_is_not_applicable_when_the_range_was_not_compared():
    """Verify a layout check without a range comparison reads as not applicable."""

    check = LayoutCheck(pointed_path=_FILE_0, holding_paths=[_FILE_0])

    result = index_range_mismatch_count(_context(check))

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_the_range_metric_counts_and_names_each_field():
    """Verify the value counts disagreeing fields and the evidence gives both values."""

    check = LayoutCheck(
        pointed_path=_FILE_1,
        holding_paths=[_FILE_1],
        range_mismatches=[
            RangeMismatch(field="dataset_from_index", declared=605, found=336051),
            RangeMismatch(field="dataset_to_index", declared=1224, found=336670),
        ],
    )

    result = index_range_mismatch_count(_context(check))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 2.0
    assert result.evidence == {
        "paths": [str(_FILE_1)],
        "mismatches": [
            {"field": "dataset_from_index", "declared": 605, "found": 336051},
            {"field": "dataset_to_index", "declared": 1224, "found": 336670},
        ],
    }
