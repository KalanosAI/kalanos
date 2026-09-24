"""Verifies DelimitedTextAdapter: comment-header parsing and CSV disjointness."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# External
import polars as pl

# Internal
from kalanos.analysis.adapters.delimited import DelimitedTextAdapter
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, FIXTURES_DIR, IMU_FIXTURE, POSE_FIXTURE


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _delimited(tmp_path: Path, name: str, text: str) -> Path:
    """Write `text` to `tmp_path / name` and return the path."""

    path = tmp_path / name
    path.write_text(text)
    return path


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_passes_the_adapter_contract_suite():
    """Verify DelimitedTextAdapter satisfies the Adapter protocol on the fixture."""

    check_adapter(DelimitedTextAdapter(), POSE_FIXTURE, [CSV_FIXTURE, IMU_FIXTURE])


def test_detect_bids_zero_on_a_csv_path():
    """Verify a .csv path always bids 0.0, so CsvAdapter and this can never tie."""

    assert DelimitedTextAdapter().detect(CSV_FIXTURE) == 0.0


def test_detect_declines_prose_that_also_splits_into_several_tokens():
    """Verify a numberless, ragged-width text file — the fixtures README — bids 0.0.

    A sentence splits into more than one whitespace token same as a data
    row does; carrying no number and varying in width line to line is
    what tells the two apart.
    """

    assert DelimitedTextAdapter().detect(FIXTURES_DIR / "README.md") == 0.0


def test_the_comment_header_becomes_column_names():
    """Verify `# timestamp ax ay az` supplies the column names."""

    episode = next(DelimitedTextAdapter().episodes(POSE_FIXTURE))

    names = {channel.name for stream in episode.streams for channel in stream.channels}
    assert names == {"ax", "ay", "az"}


def test_a_mismatched_comment_header_falls_back_to_generic_names(tmp_path):
    """Verify a header whose token count disagrees with the data falls back."""

    path = _delimited(
        tmp_path,
        "mismatched.txt",
        "# t x\n0.0 1.0 2.0\n0.1 1.1 2.1\n0.2 1.2 2.2\n",
    )
    parsed = DelimitedTextAdapter().parse(path)

    assert parsed.frame.columns == ["column_0", "column_1", "column_2"]


def test_the_float_seconds_unit_comes_from_the_median_gap_magnitude():
    """Verify the unit resolves to seconds by magnitude, timestamps unchanged."""

    episode = next(DelimitedTextAdapter().episodes(POSE_FIXTURE))

    timestamps = episode.streams[0].timestamps.to_list()
    assert timestamps[:3] == [0.0, 0.02, 0.04]


def test_a_trailing_non_numeric_column_stays_a_string_and_is_excluded(tmp_path):
    """Verify a trailing text column parses as pl.String and forms no stream."""

    path = _delimited(
        tmp_path,
        "labelled.txt",
        "# t x label\n0.0 1.0 ok\n0.1 1.1 ok\n0.2 1.2 bad\n",
    )
    assert DelimitedTextAdapter().detect(path) == 0.3

    parsed = DelimitedTextAdapter().parse(path)
    assert parsed.frame["label"].dtype == pl.String

    episode = next(DelimitedTextAdapter().episodes(path))
    assert not any(
        channel.name == "label"
        for stream in episode.streams
        for channel in stream.channels
    )


def test_a_row_shorter_than_the_first_fills_missing_fields_with_null(tmp_path):
    """Verify a truncated trailing row nulls its missing fields rather than crashing.

    A log cut off mid-write is the ordinary way this happens; one short
    row must not refuse the whole file.
    """

    path = _delimited(
        tmp_path,
        "truncated.txt",
        "# t x y\n0.0 1.0 2.0\n0.1 1.1 2.1\n0.2 1.2\n",
    )

    parsed = DelimitedTextAdapter().parse(path)

    assert parsed.frame["y"].to_list() == [2.0, 2.1, None]
