"""Verifies the metadata tier reads no numeric payload and leaves its checks unknown."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path
from unittest.mock import patch

# External
import polars as pl
import yaml
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter
from kalanos.analysis.execution import current_tier, use_tier
from kalanos.analysis.models.domain import FramePayload
from kalanos.analysis.models.eligibility import EligibilityStatus
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import PayloadStatus
from kalanos.analysis.reporting.assemble import grade_episode
from kalanos.api import grade
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.assets.policy import load_policy
from kalanos.cli import app


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _tiny_episode():
    from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter

    return next(iter(LeRobotV3Adapter().episodes(UPath(TINY_V3))))


def _numeric_features() -> set[str]:
    import json

    info = json.loads((TINY_V3 / "meta" / "info.json").read_text())
    return {
        k
        for k, v in info["features"].items()
        if v.get("dtype") != "video"
        and (k.startswith("observation.") or k.startswith("action"))
    }


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


# A stream with no payload cannot pass on another stream's evidence


def test_a_required_stream_without_a_payload_makes_the_episode_unknown():
    episode = _tiny_episode()
    with_channels = [s for s in episode.streams if s.channels]
    assert len(with_channels) >= 2, "fixture needs two numeric streams"
    starved = with_channels[0].model_copy(update={"payload": None})
    episode = episode.model_copy(
        update={
            "streams": [
                starved if s is with_channels[0] else s for s in episode.streams
            ]
        }
    )
    graded, _ = grade_episode(
        episode,
        adapter="lerobot_v3",
        adapter_confidence=1.0,
        policy=load_policy(None),
        dictionary=load_default_dictionary(),
    )
    statuses = {s.taxonomy_type: s.evaluation.payload for s in graded.streams}
    assert statuses[starved.taxonomy_type] == PayloadStatus.MISSING_INPUT
    assert statuses[with_channels[1].taxonomy_type] == PayloadStatus.COMPUTED

    from kalanos.analysis.models.binding import RequirementsSection
    from kalanos.analysis.scoring.eligibility import eligibility_of

    decision = eligibility_of(
        graded, [], requirements=RequirementsSection(), policy_id="p"
    )
    assert decision.status == EligibilityStatus.UNKNOWN
    assert any(r.id == f"payload:{starved.taxonomy_type}" for r in decision.reasons)


def test_a_stream_without_channels_is_not_required():
    graded, _ = grade_episode(
        _tiny_episode(),
        adapter="lerobot_v3",
        adapter_confidence=1.0,
        policy=load_policy(None),
        dictionary=load_default_dictionary(),
    )
    for stream in graded.streams:
        if not stream.channels:
            assert stream.evaluation.payload == PayloadStatus.NOT_REQUIRED
        else:
            assert stream.evaluation.payload == PayloadStatus.COMPUTED
            assert stream.evaluation.n_channels_graded == len(stream.channels)


# The metadata tier is an execution boundary


def test_metadata_tier_fetches_no_numeric_payload_and_leaves_required_checks_unknown(
    tmp_path,
):
    bundle = tmp_path / "meta.yaml"
    bundle.write_text(
        yaml.safe_dump({"schema_version": 1, "execution": {"tier": "metadata"}})
    )
    with patch.object(FramePayload, "fetch", autospec=True) as fetch:
        report = grade(TINY_V3, bundle=bundle)
    assert fetch.call_count == 0
    evaluated = [
        s.evaluation.payload
        for e in report.episodes
        for s in e.streams
        if s.evaluation.payload != PayloadStatus.NOT_REQUIRED
    ]
    assert evaluated and set(evaluated) == {PayloadStatus.SKIPPED}
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.unknown == len(report.episodes)
    assert report.readiness is not None and report.readiness.score is None
    assert (
        CliRunner().invoke(app, ["grade", str(TINY_V3), "--tier", "metadata"]).exit_code
        == 1
    )


def test_end_to_end_metadata_tier_is_skipped_not_missing_input(tmp_path):
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


def test_metadata_tier_never_materialises_numeric_columns():
    """Every parquet read at metadata tier projects away the feature vectors."""

    reads: list[list[str]] = []
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


def test_standard_tier_still_reads_payloads():
    with patch.object(
        FramePayload, "fetch", autospec=True, side_effect=FramePayload.fetch
    ) as fetch:
        report = grade(TINY_V3)
    assert fetch.call_count > 0
    assert (
        report.eligibility_counts is not None
        and report.eligibility_counts.pass_count == 2
    )


def test_standard_tier_still_reads_and_grades_numeric_columns():
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


def test_the_tier_is_restored_after_a_run():
    assert current_tier() == ExecutionTier.STANDARD
    with use_tier(ExecutionTier.METADATA):
        assert current_tier() == ExecutionTier.METADATA
    assert current_tier() == ExecutionTier.STANDARD
