"""Verifies that an incomplete audit is reported as one and fails the default gate."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

# External
import polars as pl
import pytest
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis import pipeline
from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter
from kalanos.analysis.adapters.select import Selection, select_adapter
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.domain import Episode
from kalanos.analysis.models.provenance import Inventory
from kalanos.api import grade
from kalanos.assets.policy import load_policy
from kalanos.cli import app


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"
NO_TIME_CSV = "a,b\n1,2\n3,4\n"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _Refusing:
    """An adapter that declares `declared` episodes, yields `yield_n`, then refuses."""

    name = "refusing"

    def __init__(self, declared: int | None, yield_n: int):
        self.declared = declared
        self.yield_n = yield_n

    def describe(self, path: UPath) -> DatasetInfo:
        return DatasetInfo(adapter=self.name, path=path, episode_count=self.declared)

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        real = LeRobotV3Adapter().episodes(path)
        for _ in range(self.yield_n):
            yield next(real)
        raise AdapterRefusal(path, "simulated refusal after some episodes")


class _PartialAdapter:
    """Yields the first `yield_n` real episodes of the tiny fixture, then refuses."""

    name = "partial"

    def __init__(self, declared: int | None, yield_n: int):
        self.declared, self.yield_n = declared, yield_n

    def describe(self, path: UPath) -> DatasetInfo:
        return DatasetInfo(adapter=self.name, path=path, episode_count=self.declared)

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        real = LeRobotV3Adapter().episodes(path)
        for _ in range(self.yield_n):
            yield next(real)
        if self.yield_n < 2:
            raise AdapterRefusal(path, "simulated refusal")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _truncated_tiny(tmp_path: Path) -> Path:
    """The tiny LeRobot fixture with one episode-index row removed, count kept."""

    copy = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(TINY_V3, copy)
    index = copy / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    frame = pl.read_parquet(index)
    assert frame.height == 2
    frame.head(1).write_parquet(index)
    return copy


def _run_with(adapter, path: Path):
    """Run the pipeline with `adapter` as the only bidder on `path`."""

    def select(candidate, adapters):
        return Selection(
            path=candidate, name=adapter.name, adapter=adapter, confidence=1.0
        )

    with patch.object(pipeline, "select_adapter", side_effect=select):
        return pipeline.run(UPath(path), policy=load_policy(None))


def _run(adapter, path: Path):
    def select(candidate, adapters):
        return Selection(
            path=candidate, name=adapter.name, adapter=adapter, confidence=1.0
        )

    with patch.object(pipeline, "select_adapter", side_effect=select):
        return pipeline.run(UPath(path), policy=load_policy(None))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


# A declared episode that never loads is an inventory gap


def test_a_missing_index_row_is_an_unresolved_episode_not_a_smaller_denominator(
    tmp_path,
):
    report = grade(_truncated_tiny(tmp_path))
    inv = report.inventory
    assert inv is not None
    assert (inv.expected, inv.loaded, inv.unresolved, inv.complete) == (2, 1, 1, False)
    assert inv.notes and "declared 2" in inv.notes[0]
    counts = report.eligibility_counts
    assert counts is not None
    assert (counts.total, counts.pass_count, counts.unknown) == (2, 1, 1)
    assert counts.inventory_complete is False
    assert counts.confirmed_eligible_share is None
    assert report.readiness is not None and report.readiness.score is None
    assert any("incomplete" in r for r in report.readiness.reasons)
    assert report.score.train_ready is None
    assert report.run is not None and report.run.completion.value == "partial"


def test_an_inventory_cannot_claim_a_gap_and_completeness():
    with pytest.raises(ValueError, match="is not complete"):
        Inventory(loaded=1, unresolved=1, complete=True)


# The default gate fails an incomplete audit


def test_the_default_gate_refuses_an_incomplete_audit(tmp_path):
    result = CliRunner().invoke(app, ["grade", str(_truncated_tiny(tmp_path))])
    assert result.exit_code == 1


def test_a_refused_source_alone_fails_the_default_gate(tmp_path):
    folder = tmp_path / "only_refused"
    folder.mkdir()
    (folder / "no_time.csv").write_text(NO_TIME_CSV)
    report = grade(folder)
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.total == 0, "no episode count is invented"
    assert report.inventory is not None and report.inventory.refused_sources
    assert report.inventory.complete is False
    assert report.run is not None and report.run.completion.value == "partial"

    result = CliRunner().invoke(app, ["grade", str(folder)])
    assert result.exit_code == 1


def test_passing_episodes_beside_a_refused_source_fail_the_default_gate(tmp_path):
    folder = tmp_path / "mixed"
    shutil.copytree(TINY_V3, folder / "lerobot_v3_tiny")
    (folder / "no_time.csv").write_text(NO_TIME_CSV)
    report = grade(folder)
    c = report.eligibility_counts
    assert c is not None
    assert (c.pass_count, c.blocked, c.review, c.unknown) == (2, 0, 0, 0)
    assert c.inventory_complete is False

    runner = CliRunner()
    assert runner.invoke(app, ["grade", str(folder)]).exit_code == 1
    assert (
        runner.invoke(
            app, ["grade", str(folder), "--fail-on", "blocked,unknown"]
        ).exit_code
        == 1
    )


def test_exploratory_blocked_only_permits_a_partial_audit_but_says_so(tmp_path):
    folder = tmp_path / "mixed"
    shutil.copytree(TINY_V3, folder / "lerobot_v3_tiny")
    (folder / "no_time.csv").write_text(NO_TIME_CSV)
    result = CliRunner().invoke(app, ["grade", str(folder), "--fail-on", "blocked"])
    assert result.exit_code == 0
    assert "audit incomplete and not gated" in (result.stderr or result.output)
    assert "1 source(s) refused" in (result.stderr or result.output)


def test_a_complete_clean_audit_still_exits_zero():
    assert CliRunner().invoke(app, ["grade", str(TINY_V3)]).exit_code == 0


def test_a_configuration_error_stays_exit_two(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 99\n")
    assert (
        CliRunner()
        .invoke(app, ["grade", str(TINY_V3), "--profile", str(bad)])
        .exit_code
        == 2
    )


# A source refused part-way keeps what it yielded and what it declared


@pytest.mark.parametrize(
    ("declared", "yield_n", "expect_loaded", "expect_unresolved"),
    [
        (3, 1, 1, 2),  # declared three, yielded one, refused
        (3, 0, 0, 3),  # refused before the first yield
        (None, 1, 1, 0),  # undeclared count: gap unknowable, still incomplete
    ],
)
def test_a_refusal_mid_read_keeps_the_yield_and_the_declaration(
    declared, yield_n, expect_loaded, expect_unresolved
):
    report = _run_with(_Refusing(declared, yield_n), TINY_V3)
    inv = report.inventory
    assert inv is not None
    assert inv.expected == declared
    assert inv.loaded == expect_loaded == len(report.episodes)
    assert inv.unresolved == expect_unresolved
    assert inv.failed == [], "a refused source is not one failed episode"
    assert len(inv.refused_sources) == 1
    assert inv.complete is False
    assert report.run is None or report.run.completion.value != "complete"
    counts = report.eligibility_counts
    assert counts is not None
    assert counts.total == expect_loaded + expect_unresolved
    assert counts.inventory_complete is False
    assert report.readiness is not None and report.readiness.score is None
    assert report.score.train_ready is None
    # What was yielded is graded, not discarded.
    if expect_loaded:
        assert report.episodes[0].eligibility is not None
    # The refusal itself is still reported as an unresolved source.
    assert len(report.unresolved) == 1


def test_a_refused_directory_is_not_offered_again_file_by_file(tmp_path):
    """The refused source's contents are accounted for once, as that source."""

    import shutil

    folder = tmp_path / "root"
    shutil.copytree(TINY_V3, folder / "lerobot_v3_tiny")
    report = _run_with(_Refusing(2, 1), folder / "lerobot_v3_tiny")
    # Only the refused source and its one yielded episode; no child parquet or
    # mp4 graded on its own under a second adapter.
    assert len(report.episodes) == 1
    assert len(report.datasets) == 1


@pytest.mark.parametrize("declared", [2, None])
def test_a_retained_episode_keeps_its_id_when_a_later_sibling_fails(declared):
    complete = _run(_PartialAdapter(declared, 2), TINY_V3)
    partial = _run(_PartialAdapter(declared, 1), TINY_V3)
    assert len(complete.episodes) == 2 and len(partial.episodes) == 1
    assert partial.episodes[0].id == complete.episodes[0].id
    assert partial.episodes[0].id.endswith("::episode_000000")
    assert partial.inventory is not None and partial.inventory.complete is False


def test_single_recording_files_keep_their_short_id(tmp_path):
    """A file whose adapter names the episode after the file stays unqualified."""

    csv = Path(__file__).parent / "fixtures" / "arm_multi_device.csv"
    shutil.copy(csv, tmp_path / csv.name)
    report = grade(tmp_path)
    assert [e.id for e in report.episodes] == ["arm_multi_device.csv"]


def test_a_container_that_yields_one_episode_is_still_qualified(tmp_path):
    """A one-episode LeRobot export keeps its episode key in the id."""

    copy = tmp_path / "one_episode"
    shutil.copytree(TINY_V3, copy)
    index = copy / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pl.read_parquet(index).head(1).write_parquet(index)
    report = grade(copy)
    assert [e.id for e in report.episodes] == ["one_episode::episode_000000"]


# A directory that yields nothing is still owned by its adapter


def test_a_directory_with_an_empty_index_is_one_source_not_two(tmp_path):
    root = tmp_path / "root"
    copy = root / "lerobot_v3_tiny"
    shutil.copytree(TINY_V3, copy)
    index = copy / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pl.read_parquet(index).head(0).write_parquet(index)

    offered: list[str] = []
    real = select_adapter

    def spy(candidate, adapters):
        offered.append(str(candidate))
        return real(candidate, adapters)

    with patch.object(pipeline, "select_adapter", side_effect=spy):
        report = pipeline.run(UPath(root), policy=load_policy(None))

    assert [d.adapter for d in report.datasets] == ["lerobot_v3"]
    assert report.inventory is not None
    assert (
        report.inventory.expected,
        report.inventory.loaded,
        report.inventory.unresolved,
    ) == (2, 0, 2)
    assert not report.inventory.refused_sources, (
        "an empty index is a gap, not a refusal"
    )
    assert report.unresolved == [], "its manifest is not refused as a second dataset"
    assert not any("/meta/" in p or p.endswith(".parquet") for p in offered), (
        f"files inside the owned directory were offered: {offered}"
    )
    assert (
        report.eligibility_counts is not None and report.eligibility_counts.unknown == 2
    )
