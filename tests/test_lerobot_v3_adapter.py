"""Verifies the LeRobot v3.0 adapter: several files per episode, one parquet
chunk split by episode index, the declared rate read rather than inferred,
and camera payloads that never decode while grading.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import shutil

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.lerobot.common import (
    CONFIDENCE,
    resolve_taxonomy,
    series_stream,
)
from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter
from kalanos.analysis.models.domain import Channel, FramePayload, Kind
from kalanos.analysis.models.report import AnalysedEpisode
from kalanos.analysis.reporting.assemble import assemble_report
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.assets.policy import load_default_policy
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, LEROBOT_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The names scripts/fixture_gen/corpus/lerobot_v3_tiny.py
# declared for its two vector features
_MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _episodes():
    """Drain both of the fixture's episodes.

    Returns
    -------
    list[Episode]
        Both episodes, in episode-index order.
    """

    return list(LeRobotV3Adapter().episodes(LEROBOT_FIXTURE))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_contract(tmp_path):
    """Verify the adapter holds every property `check_adapter` requires."""

    not_lerobot = tmp_path / "not_lerobot"
    not_lerobot.mkdir()
    check_adapter(LeRobotV3Adapter(), LEROBOT_FIXTURE, [CSV_FIXTURE, not_lerobot])


def test_one_episode_draws_on_several_files():
    """Verify one episode's streams reference the data parquet and its mp4."""

    [first, _] = _episodes()

    assert len(first.source_paths) >= 2
    for path in first.source_paths:
        assert path.exists()


def test_two_episodes_come_from_one_parquet():
    """Verify both episodes' series streams share one data parquet, split by index.

    LeRobot's `timestamp` is `frame_index / fps`, and `frame_index` restarts
    at every episode, so the two episodes' timestamps overlap by design.
    `episode_index` is what actually splits the shared chunk, confirmed
    below by the differing lengths.
    """

    first, second = _episodes()
    first_state = next(
        s for s in first.streams if s.source_field == "observation.state"
    )
    second_state = next(
        s for s in second.streams if s.source_field == "observation.state"
    )

    assert first_state.source_path == second_state.source_path
    assert len(first_state.timestamps) == 8
    assert len(second_state.timestamps) == 6


def test_the_declared_rate_is_read_not_inferred(tmp_path):
    """Verify describe() answers from meta/info.json alone, no episode read.

    The data parquet is made unreadable before `describe` runs, so a
    passing `nominal_rate_hz`/`robot_type`/`episode_count` proves none of
    them came from actually reading an episode.
    """

    copy_root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), copy_root)
    data_path = copy_root / "data" / "chunk-000" / "file-000.parquet"
    data_path.write_bytes(b"not a parquet file")

    info = LeRobotV3Adapter().describe(UPath(copy_root))

    assert info.nominal_rate_hz == pytest.approx(30.0)
    assert info.robot_type == "so100_follower"
    assert info.episode_count == 2


def test_a_camera_stream_is_never_decoded_while_grading(monkeypatch):
    """Verify grading the whole fixture never touches the video decoder."""

    monkeypatch.setattr(
        video, "_load_av", lambda: pytest.fail("_load_av() was called while grading")
    )

    policy = load_default_policy()
    analysed = [
        AnalysedEpisode(
            episode=episode,
            adapter="lerobot_v3",
            adapter_confidence=CONFIDENCE,
            policy=policy,
        )
        for episode in _episodes()
    ]

    report = assemble_report(root=LEROBOT_FIXTURE, analysed=analysed, policy=policy)

    assert len(report.episodes) == 2


def test_a_video_stream_carries_frames_it_has_not_read(monkeypatch):
    """Verify a video payload's length costs nothing, with the decoder patched out."""

    monkeypatch.setattr(video, "_load_av", lambda: pytest.fail("_load_av() was called"))

    [first, _] = _episodes()
    [video_stream] = [s for s in first.streams if s.kind == Kind.VIDEO]

    assert video_stream.payload is not None
    assert len(video_stream.payload) == len(video_stream.timestamps)


def test_a_feature_key_resolves_through_the_dictionary():
    """Verify observation.velocity resolves to proprio.joint_velocity by its key."""

    [first, _] = _episodes()
    velocity = next(
        s for s in first.streams if s.source_field == "observation.velocity"
    )

    assert velocity.taxonomy_type == "proprio.joint_velocity"


def test_an_unresolvable_feature_reaches_the_report_unmapped():
    """Verify observation.state is unmapped, its channels named from info.json."""

    [first, _] = _episodes()
    state = next(s for s in first.streams if s.source_field == "observation.state")

    assert state.taxonomy_type == "unmapped.observation.state"
    assert [c.name for c in state.channels] == _MOTOR_NAMES


def test_a_video_key_resolves_through_the_dictionary_by_its_own_key():
    """Verify a camera feature key that the dictionary does claim resolves.

    The fixture's own camera (`observation.images.up`) deliberately does
    not resolve, since LeRobot's real key names an unlisted alias. So this
    test exercises the success path directly rather than through the fixture.
    """

    taxonomy_type = resolve_taxonomy(
        "observation.images.wrist", {"dtype": "video"}, load_default_dictionary()
    )

    assert taxonomy_type == "extero.wrist_rgb"


def test_a_v2_dataset_is_declined(tmp_path):
    """Verify a v2.x codebase_version bids zero rather than being misread."""

    root = tmp_path / "v2_dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))

    assert LeRobotV3Adapter().detect(UPath(root)) == 0.0


def test_a_fixed_size_array_feature_unnests_into_one_column_per_channel():
    """Hugging Face writes a vector feature as a fixed-size Array, not a List."""

    frame = pl.DataFrame(
        {"observation.state": [[0.0, 1.0], [2.0, 3.0]]},
        schema={"observation.state": pl.Array(pl.Float32, 2)},
    )
    channels = [Channel(name="x"), Channel(name="y")]

    stream = series_stream(
        frame,
        "observation.state",
        channels,
        "unmapped.observation.state",
        pl.Series([0.0, 0.1]),
        UPath("data/chunk-000/file-000.parquet"),
        is_regular=True,
    )

    assert isinstance(stream.payload, FramePayload)
    assert stream.payload.frame.columns == ["x", "y"]
    assert stream.payload.frame["y"].to_list() == [1.0, 3.0]
