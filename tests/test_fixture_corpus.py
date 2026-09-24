"""Enforces the invariants the fixture corpus is supposed to carry."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import json
from collections import Counter

# External
from upath import UPath

# Local
from helpers import CSV_FIXTURE, FIXTURES_DIR, NO_TIMESERIES_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


MAX_FIXTURE_BYTES = 10 * 1024

# A real h264 mp4 needs a larger cap.
MAX_VIDEO_FIXTURE_BYTES = 200 * 1024


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


def _fixture_files() -> list[UPath]:
    """List every fixture file, recursively, excluding the README that documents them.

    Returns
    -------
    list of UPath
        Every file under `tests/fixtures/`, at any depth, except the README.
    """

    return sorted(
        p for p in FIXTURES_DIR.rglob("*") if p.is_file() and p.name != "README.md"
    )


def _top_level_fixtures() -> list[UPath]:
    """List every fixture directly under `tests/fixtures/`, files and directories alike.

    This is what the README documents one row per: a directory such as
    `lerobot_v3_tiny/` gets a single row naming the whole tree, not one
    row per file inside it.

    Returns
    -------
    list of UPath
        Every entry directly under `tests/fixtures/` except the README.
    """

    return sorted(p for p in FIXTURES_DIR.glob("*") if p.name != "README.md")


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_corpus_is_not_empty():
    """Verify the fixture directory actually holds fixtures."""

    assert _fixture_files(), "expected at least one fixture under tests/fixtures/"


def test_every_fixture_is_under_the_size_cap():
    """Verify every fixture stays a few KB, an mp4 apart, as the corpus is meant to."""

    oversized = {}
    for p in _fixture_files():
        cap = MAX_VIDEO_FIXTURE_BYTES if p.suffix == ".mp4" else MAX_FIXTURE_BYTES
        size = p.stat().st_size
        if size > cap:
            oversized[str(p.relative_to(FIXTURES_DIR))] = size
    assert not oversized, f"fixtures over their size cap: {oversized}"


def test_every_fixture_is_named_in_the_readme():
    """Verify the README documents which pathology each top-level fixture exercises."""

    readme = (FIXTURES_DIR / "README.md").read_text()
    undocumented = [p.name for p in _top_level_fixtures() if p.name not in readme]
    assert not undocumented, f"fixtures missing from README.md: {undocumented}"


def test_the_no_timeseries_fixture_is_part_of_the_corpus():
    """Verify `NO_TIMESERIES_FIXTURE` names a real, committed fixture."""

    fixture_names = {p.name for p in _fixture_files()}
    assert NO_TIMESERIES_FIXTURE.name in fixture_names


def test_video_meta_has_no_time_index():
    """Verify the no-timeseries fixture is actually what it claims to be."""

    data = json.loads(NO_TIMESERIES_FIXTURE.read_text())
    assert isinstance(data, dict)
    time_like_keys = {
        k for k in data if "time" in k.lower() or k.lower() in {"t", "ts"}
    }
    assert not time_like_keys, f"video_meta.json has a time-like key: {time_like_keys}"


def test_csv_fixture_interleaves_entities_at_shared_timestamps():
    """Verify the CSV fixture exercises entity splitting rather than assuming it.

    Several `id` values must share at least one identical `t_ms`, so a
    naive loader that ignores `id` would silently merge distinct entities.
    """

    counts: Counter[str] = Counter()
    with CSV_FIXTURE.open() as f:
        for row in csv.DictReader(f):
            counts[row["t_ms"]] += 1

    shared_timestamps = [t for t, n in counts.items() if n > 1]
    assert shared_timestamps, "no CSV timestamp is shared by more than one id"
    assert max(counts.values()) > 1


def test_csv_fixture_has_more_than_one_entity():
    """Verify the CSV fixture actually carries several distinct ids."""

    ids: set[str] = set()
    with CSV_FIXTURE.open() as f:
        for row in csv.DictReader(f):
            ids.add(row["id"])

    assert len(ids) > 1, f"expected more than one id in the CSV fixture, found {ids}"
