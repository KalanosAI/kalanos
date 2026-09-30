"""Controls for the round-2 review corrections.

A — an adapter that refuses part-way through keeps what it yielded and what
    it declared; a refused source is not one failed episode.
B — the metadata tier is enforced at the storage boundary: the LeRobot
    adapters never materialise numeric feature columns.
"""

# Built-in
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis import pipeline
from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter
from kalanos.analysis.adapters.select import Selection
from kalanos.analysis.execution import current_tier, use_tier
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.domain import Episode
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import PayloadStatus
from kalanos.api import grade
from kalanos.assets.policy import load_policy


TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"


# ── A: refusal mid-read ──


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


def _run_with(adapter, path: Path):
    """Run the pipeline with `adapter` as the only bidder on `path`."""

    def select(candidate, adapters):
        return Selection(
            path=candidate, name=adapter.name, adapter=adapter, confidence=1.0
        )

    with patch.object(pipeline, "select_adapter", side_effect=select):
        return pipeline.run(UPath(path), policy=load_policy(None))


@pytest.mark.parametrize(
    ("declared", "yield_n", "expect_loaded", "expect_unresolved"),
    [
        (3, 1, 1, 2),  # declared three, yielded one, refused
        (3, 0, 0, 3),  # refused before the first yield
        (None, 1, 1, 0),  # undeclared count: gap unknowable, still incomplete
    ],
)
def test_a_a_refusal_mid_read_keeps_the_yield_and_the_declaration(
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


def test_a_a_refused_directory_is_not_offered_again_file_by_file(tmp_path):
    """The refused source's contents are accounted for once, as that source."""

    import shutil

    folder = tmp_path / "root"
    shutil.copytree(TINY_V3, folder / "lerobot_v3_tiny")
    report = _run_with(_Refusing(2, 1), folder / "lerobot_v3_tiny")
    # Only the refused source and its one yielded episode; no child parquet or
    # mp4 graded on its own under a second adapter.
    assert len(report.episodes) == 1
    assert len(report.datasets) == 1


# ── B: metadata tier at the storage boundary ──


def _numeric_features() -> set[str]:
    import json

    info = json.loads((TINY_V3 / "meta" / "info.json").read_text())
    return {
        k
        for k, v in info["features"].items()
        if v.get("dtype") != "video"
        and (k.startswith("observation.") or k.startswith("action"))
    }


def test_b_metadata_tier_never_materialises_numeric_columns():
    """Every parquet read at metadata tier projects away the feature vectors."""

    reads: list[list[str] | None] = []
    real = pl.read_parquet

    def spy(source, *args, **kwargs):
        frame = real(source, *args, **kwargs)
        reads.append(list(frame.columns))
        return frame

    with (
        patch.object(pl, "read_parquet", side_effect=spy),
        use_tier(ExecutionTier.METADATA),
    ):
        episodes = list(LeRobotV3Adapter().episodes(UPath(TINY_V3)))

    assert reads, "the adapter still reads the index and clock columns"
    numeric = _numeric_features()
    for columns in reads:
        assert not (set(columns) & numeric), f"numeric columns materialised: {columns}"
    # Streams are declared with their channels, and carry no payload.
    for episode in episodes:
        for stream in episode.streams:
            if stream.channels:
                assert stream.payload is None
                assert stream.channels, "channel declarations survive"


def test_b_standard_tier_still_reads_and_grades_numeric_columns():
    reads: list[list[str]] = []
    real = pl.read_parquet

    def spy(source, *args, **kwargs):
        frame = real(source, *args, **kwargs)
        reads.append(list(frame.columns))
        return frame

    with patch.object(pl, "read_parquet", side_effect=spy):
        report = grade(TINY_V3)
    assert any(set(c) & _numeric_features() for c in reads)
    assert all(
        s.evaluation.payload == PayloadStatus.COMPUTED
        for e in report.episodes
        for s in e.streams
        if s.evaluation.payload != PayloadStatus.NOT_REQUIRED
    )


def test_b_end_to_end_metadata_tier_is_skipped_not_missing_input(tmp_path):
    """Through the CLI/API path: the tier reaches the adapter, and grading
    records `skipped`, not `missing_input`, for payload-less streams."""

    import yaml

    bundle = tmp_path / "meta.yaml"
    bundle.write_text(
        yaml.safe_dump({"schema_version": 1, "execution": {"tier": "metadata"}})
    )
    report = grade(TINY_V3, bundle=bundle)
    statuses = {
        s.evaluation.payload
        for e in report.episodes
        for s in e.streams
        if s.evaluation.payload != PayloadStatus.NOT_REQUIRED
    }
    assert statuses == {PayloadStatus.SKIPPED}
    assert all(
        s.evaluation.n_channels_declared > 0 and s.evaluation.n_channels_graded == 0
        for e in report.episodes
        for s in e.streams
        if s.evaluation.payload == PayloadStatus.SKIPPED
    )
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.unknown == len(report.episodes)


def test_b_the_tier_is_restored_after_a_run():
    assert current_tier() == ExecutionTier.STANDARD
    with use_tier(ExecutionTier.METADATA):
        assert current_tier() == ExecutionTier.METADATA
    assert current_tier() == ExecutionTier.STANDARD
