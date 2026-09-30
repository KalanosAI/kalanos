"""Controls for the round-3 review findings.

1 — an incomplete audit (refused source, undelivered declared count) fails the
    default CLI gate, without inventing episode counts.
2 — a retained episode keeps its id whether or not later siblings loaded.
3 — a recognised directory that yields nothing is still owned by its adapter;
    its manifest is not graded again as a separate dataset.
"""

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
from kalanos.api import grade
from kalanos.assets.policy import load_policy
from kalanos.cli import app


TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"
NO_TIME_CSV = "a,b\n1,2\n3,4\n"


# ── 1: the default gate fails an incomplete audit ──


def test_1_a_refused_source_alone_fails_the_default_gate(tmp_path):
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


def test_1_passing_episodes_beside_a_refused_source_fail_the_default_gate(tmp_path):
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


def test_1_exploratory_blocked_only_permits_a_partial_audit_but_says_so(tmp_path):
    folder = tmp_path / "mixed"
    shutil.copytree(TINY_V3, folder / "lerobot_v3_tiny")
    (folder / "no_time.csv").write_text(NO_TIME_CSV)
    result = CliRunner().invoke(app, ["grade", str(folder), "--fail-on", "blocked"])
    assert result.exit_code == 0
    assert "audit incomplete and not gated" in (result.stderr or result.output)
    assert "1 source(s) refused" in (result.stderr or result.output)


def test_1_a_complete_clean_audit_still_exits_zero():
    assert CliRunner().invoke(app, ["grade", str(TINY_V3)]).exit_code == 0


def test_1_a_configuration_error_stays_exit_two(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 99\n")
    assert (
        CliRunner()
        .invoke(app, ["grade", str(TINY_V3), "--profile", str(bad)])
        .exit_code
        == 2
    )


# ── 1 and 2: a partially yielded source, declared and undeclared ──


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


def _run(adapter, path: Path):
    def select(candidate, adapters):
        return Selection(
            path=candidate, name=adapter.name, adapter=adapter, confidence=1.0
        )

    with patch.object(pipeline, "select_adapter", side_effect=select):
        return pipeline.run(UPath(path), policy=load_policy(None))


@pytest.mark.parametrize("declared", [2, None])
def test_2_a_retained_episode_keeps_its_id_when_a_later_sibling_fails(declared):
    complete = _run(_PartialAdapter(declared, 2), TINY_V3)
    partial = _run(_PartialAdapter(declared, 1), TINY_V3)
    assert len(complete.episodes) == 2 and len(partial.episodes) == 1
    assert partial.episodes[0].id == complete.episodes[0].id
    assert partial.episodes[0].id.endswith("::episode_000000")
    assert partial.inventory is not None and partial.inventory.complete is False


def test_2_single_recording_files_keep_their_short_id(tmp_path):
    """A file whose adapter names the episode after the file stays unqualified."""

    csv = Path(__file__).parent / "fixtures" / "arm_multi_device.csv"
    shutil.copy(csv, tmp_path / csv.name)
    report = grade(tmp_path)
    assert [e.id for e in report.episodes] == ["arm_multi_device.csv"]


def test_2_a_container_that_yields_one_episode_is_still_qualified(tmp_path):
    """Previously a one-episode LeRobot export lost its episode key."""

    copy = tmp_path / "one_episode"
    shutil.copytree(TINY_V3, copy)
    index = copy / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pl.read_parquet(index).head(1).write_parquet(index)
    report = grade(copy)
    assert [e.id for e in report.episodes] == ["one_episode::episode_000000"]


# ── 3: a zero-yield directory is owned, not re-offered ──


def test_3_a_directory_with_an_empty_index_is_one_source_not_two(tmp_path):
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
