"""Verifies discovery: walking a folder and classifying the files it finds.

Retrofitting these later is expensive, so they are pinned as tests
alongside the walk they describe.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# External
from upath import UPath

# Internal
from kalanos.analysis.discovery.walk import walk_folder
from kalanos.analysis.models.discovery import SkippedSource, SkipReason, SourceCandidate


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FIXTURES_DIR = UPath(__file__).parent / "fixtures"


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


def _fixture_files() -> list[UPath]:
    """List every file under the tracked fixture folder, recursively.

    Returns
    -------
    list of UPath
        Every file under `tests/fixtures/`, README and anything nested included.
    """

    return sorted(p for p in FIXTURES_DIR.rglob("*") if p.is_file())


def _fixture_dirs() -> list[UPath]:
    """List every directory under the tracked fixture folder, root included.

    Returns
    -------
    list of UPath
        `tests/fixtures/` itself, plus every directory nested under it.
    """

    return [FIXTURES_DIR, *sorted(p for p in FIXTURES_DIR.rglob("*") if p.is_dir())]


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_every_result_is_a_candidate_or_a_skip():
    """Verify walk_folder yields only SourceCandidate or SkippedSource, nothing else."""

    results = walk_folder(FIXTURES_DIR)
    assert results
    for result in results:
        assert isinstance(result, (SourceCandidate, SkippedSource))


def test_candidate_and_skip_counts_cover_every_fixture_file():
    """Verify nothing is dropped: every file becomes a candidate or a skip,
    and every directory under the root, root included, becomes a candidate.
    """

    results = walk_folder(FIXTURES_DIR)
    candidate_paths = {r.path for r in results if isinstance(r, SourceCandidate)}
    skipped_paths = {r.path for r in results if isinstance(r, SkippedSource)}

    assert set(_fixture_dirs()) <= candidate_paths
    assert set(_fixture_files()) <= candidate_paths | skipped_paths
    assert len(results) == len(_fixture_files()) + len(_fixture_dirs())


def test_every_fixture_is_classified_a_candidate():
    """Verify every readable, non-empty fixture reaches selection as a candidate.

    Modality is no longer discovery's decision: what an adapter claims is
    selection's job, so a file discovery cannot read or that is empty is the
    only thing left for `_classify` to skip.
    """

    results = walk_folder(FIXTURES_DIR)
    candidate_paths = {r.path for r in results if isinstance(r, SourceCandidate)}
    assert set(_fixture_files()) <= candidate_paths


def test_the_walk_offers_the_root_itself_as_a_candidate():
    """Verify the walked root reaches selection alongside its contents.

    A directory-level adapter, such as LeRobot's, must be able to bid on the
    folder itself rather than only on the files underneath it.
    """

    results = walk_folder(FIXTURES_DIR)
    candidate_paths = {r.path for r in results if isinstance(r, SourceCandidate)}
    assert FIXTURES_DIR in candidate_paths


def test_the_walk_is_not_bounded_by_a_file_count(tmp_path):
    """Verify every file is classified however many there are: the walk has no cap."""

    for i in range(50):
        (tmp_path / f"file_{i}.csv").write_text("a,b\n1,2\n")

    results = walk_folder(tmp_path)
    # 50 files, plus the walked root itself.
    assert len(results) == 51


def test_zero_byte_file_is_skipped_as_empty(tmp_path):
    """Verify a zero-byte file is skipped with the empty reason."""

    empty = tmp_path / "empty.csv"
    empty.write_text("")

    [skip] = [r for r in walk_folder(tmp_path) if isinstance(r, SkippedSource)]
    assert skip.reason == SkipReason.EMPTY


def test_unreadable_file_is_skipped_when_stat_fails(monkeypatch, tmp_path):
    """Verify a file that errors on stat is skipped with the unreadable reason.

    Monkeypatches Path.stat rather than chmod, since a permission-denied
    stat is not reliably reproducible across sandboxes (e.g. running as root).
    """

    mystery = tmp_path / "mystery.csv"
    mystery.write_text("a,b\n1,2\n")
    original_stat = Path.stat

    def _boom(self, *args, **kwargs):
        if self == mystery:
            raise PermissionError("denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _boom)

    [skip] = [r for r in walk_folder(tmp_path) if isinstance(r, SkippedSource)]
    assert skip.reason == SkipReason.UNREADABLE


def test_empty_takes_precedence_over_no_adapter_for_one_file(tmp_path):
    """Verify a zero-byte file gets `empty` at discovery, before selection ever runs.

    `video.mp4` would also reach selection with no adapter to claim it once
    discovery lets it through — pins that emptiness is decided first, so an
    empty file never reaches selection at all.
    """

    empty_video = tmp_path / "sensor.mp4"
    empty_video.write_bytes(b"")

    [skip] = [r for r in walk_folder(tmp_path) if isinstance(r, SkippedSource)]
    assert skip.reason == SkipReason.EMPTY


def test_a_single_file_path_yields_exactly_one_result(tmp_path):
    """Verify pointing discovery at a single file goes through walk_folder too.

    The headline command passes a file, not a folder, so this
    must resolve through the same function as the folder case rather than
    a separate single-file entry point.
    """

    recording = tmp_path / "readings.csv"
    recording.write_text("a,b\n1,2\n")

    results = walk_folder(recording)

    assert len(results) == 1
    assert isinstance(results[0], SourceCandidate)
    assert results[0].path == recording


def test_walk_recurses_into_subdirectories(tmp_path):
    """Verify files nested under a subdirectory are still walked and classified,
    and the subdirectory itself is offered as a candidate alongside them.
    """

    nested = tmp_path / "session_1"
    nested.mkdir()
    (nested / "readings.csv").write_text("a,b\n1,2\n")

    candidate_paths = {
        r.path for r in walk_folder(tmp_path) if isinstance(r, SourceCandidate)
    }
    assert nested / "readings.csv" in candidate_paths
    assert nested in candidate_paths


def test_walk_folder_classifies_the_same_on_a_non_local_filesystem():
    """Verify walk_folder classifies a memory:// folder the same as it does on disk.

    The memory filesystem is process-global and persists between tests in one
    run, so the root is unique to this test.
    """

    root = UPath("memory://test_walk_folder_classifies_the_same")
    root.mkdir(parents=True, exist_ok=True)
    (root / "good.csv").write_text("a,b\n1,2\n")
    (root / "notes.txt").write_text("not a csv")

    results = walk_folder(root)

    by_name = {r.path.name: r for r in results}
    # 2 files, plus the walked root itself.
    assert len(results) == 3
    assert isinstance(by_name["good.csv"], SourceCandidate)
    assert isinstance(by_name["notes.txt"], SourceCandidate)


def test_a_hidden_directory_is_one_skip_and_nothing_inside_it_is_offered(tmp_path):
    """A local Hugging Face snapshot leaves .cache/ beside the data; it is not data."""

    (tmp_path / "data.csv").write_text("t,x\n0,1\n")
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "file.lock").write_text("x")
    (tmp_path / ".gitattributes").write_text("*.mp4 filter=lfs")

    found = walk_folder(UPath(tmp_path))

    skipped = {i.path.name: i.reason for i in found if isinstance(i, SkippedSource)}
    assert skipped == {".cache": SkipReason.HIDDEN, ".gitattributes": SkipReason.HIDDEN}
    assert not any("huggingface" in str(item.path) for item in found)


def test_a_dot_named_root_is_still_walked(tmp_path):
    """Verify a root that is itself dot-named is walked, not skipped as hidden."""

    root = tmp_path / ".dataset"
    root.mkdir()
    (root / "data.csv").write_text("t,x\n0,1\n")

    found = walk_folder(UPath(root))

    assert any(
        isinstance(i, SourceCandidate) and i.path.name == "data.csv" for i in found
    )
