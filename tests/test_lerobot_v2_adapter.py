"""Verifies the LeRobot v2.0/v2.1 adapter: one parquet and one mp4 per episode,
chunk rollover at a small `chunks_size`, and a camera-less tree.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json

# External
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.lerobot.v2 import LeRobotV2Adapter
from kalanos.analysis.models.domain import Kind
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, LEROBOT_V2_0_FIXTURE, LEROBOT_V2_1_FIXTURE


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _episodes(fixture):
    """Drain every episode `LeRobotV2Adapter` reads from `fixture`.

    Parameters
    ----------
    fixture : UPath
        The dataset root to read.

    Returns
    -------
    list[Episode]
        Every episode, in episode-index order.
    """

    return list(LeRobotV2Adapter().episodes(fixture))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_contract_v2_0(tmp_path):
    """Verify the adapter holds every property `check_adapter` requires, on v2.0."""

    not_lerobot = tmp_path / "not_lerobot"
    not_lerobot.mkdir()
    check_adapter(LeRobotV2Adapter(), LEROBOT_V2_0_FIXTURE, [CSV_FIXTURE, not_lerobot])


def test_contract_v2_1(tmp_path):
    """Verify the adapter holds every property `check_adapter` requires, on v2.1."""

    not_lerobot = tmp_path / "not_lerobot"
    not_lerobot.mkdir()
    check_adapter(LeRobotV2Adapter(), LEROBOT_V2_1_FIXTURE, [CSV_FIXTURE, not_lerobot])


def test_each_episode_is_its_own_parquet():
    """Verify the two episodes' series streams come from different parquet files.

    In this layout one parquet holds exactly one episode, so two episodes
    never share a source path.
    """

    first, second = _episodes(LEROBOT_V2_0_FIXTURE)
    first_state = next(
        s for s in first.streams if s.source_field == "observation.state"
    )
    second_state = next(
        s for s in second.streams if s.source_field == "observation.state"
    )

    assert first_state.source_path != second_state.source_path
    assert str(first_state.source_path).endswith("episode_000000.parquet")
    assert str(second_state.source_path).endswith("episode_000001.parquet")
    assert len(first_state.timestamps) == 8
    assert len(second_state.timestamps) == 6


def test_a_camera_mp4_holds_exactly_its_own_episode(monkeypatch):
    """Verify each episode's video payload spans only its own frames."""

    monkeypatch.setattr(video, "_load_av", lambda: pytest.fail("_load_av() was called"))

    first, second = _episodes(LEROBOT_V2_0_FIXTURE)
    [first_video] = [s for s in first.streams if s.kind == Kind.VIDEO]
    [second_video] = [s for s in second.streams if s.kind == Kind.VIDEO]

    assert first_video.source_path != second_video.source_path
    for stream in (first_video, second_video):
        assert stream.payload is not None
        assert stream.payload.start_s == 0.0
        assert stream.payload.end_s == len(stream.timestamps) / 30
        assert len(stream.payload) == len(stream.timestamps)


def test_an_episode_past_the_chunk_size_is_read_from_the_next_chunk():
    """Verify chunk rollover: at `chunks_size: 1`, each episode gets its own chunk."""

    first, second = _episodes(LEROBOT_V2_1_FIXTURE)
    first_state = next(
        s for s in first.streams if s.source_field == "observation.state"
    )
    second_state = next(
        s for s in second.streams if s.source_field == "observation.state"
    )

    assert "chunk-000" in str(first_state.source_path)
    assert "chunk-001" in str(second_state.source_path)


def test_a_tree_with_no_camera_yields_only_series_streams():
    """Verify a camera-less tree names no video stream and one source path each."""

    for episode in _episodes(LEROBOT_V2_1_FIXTURE):
        assert all(stream.kind != Kind.VIDEO for stream in episode.streams)
        assert len(set(episode.source_paths)) == 1


def test_features_resolve_the_same_way_they_do_in_v3():
    """Verify the shared `common.py` feature resolution is actually shared."""

    [first, _] = _episodes(LEROBOT_V2_0_FIXTURE)
    velocity = next(
        s for s in first.streams if s.source_field == "observation.velocity"
    )
    state = next(s for s in first.streams if s.source_field == "observation.state")

    assert velocity.taxonomy_type == "proprio.joint_velocity"
    assert state.taxonomy_type == "unmapped.observation.state"
    assert len(state.channels) == 6


def test_a_v3_dataset_is_declined(tmp_path):
    """Verify a v3.0 codebase_version bids zero rather than being misread."""

    root = tmp_path / "v3_dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))

    assert LeRobotV2Adapter().detect(UPath(root)) == 0.0
