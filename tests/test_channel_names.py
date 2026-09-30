"""Verifies LeRobot channel names are read in both spellings, and that a channel
named as a gripper grades as one: holding and snapping are its normal behaviour,
while a glitching gripper still fails its episode.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# External
import h5py
import numpy as np
import polars as pl
from calibration_helpers import grade_with_test_calibration
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.common import channels_for, declared_names
from kalanos.analysis.bindings import resolve_stream
from kalanos.analysis.models.domain import Channel, Kind, Stream, TimestampDtype
from kalanos.api import grade
from kalanos.assets.dictionary import load_default_dictionary


ALOHA_MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def channel_taxonomy(stream_type: str, name: str) -> str:
    """Exercise the resolver, rather than a hidden reporting-stage heuristic."""
    stream = Stream(
        taxonomy_type=stream_type,
        kind=Kind.SERIES,
        timestamps=pl.Series("t", [0.0, 0.1]),
        source_path=UPath("fixture.csv"),
        timestamp_dtype=TimestampDtype.FLOAT64,
        channels=[Channel(name=name)],
    )
    resolved = resolve_stream(stream, dictionary=load_default_dictionary())
    return resolved.channels[0].binding.taxonomy_type


def test_names_declared_as_a_list_are_read():
    assert declared_names({"names": ["x", "y", "z"]}) == ["x", "y", "z"]


def test_names_nested_under_a_label_are_read_as_aloha_declares_them():
    """ALOHA: `"names": {"motors": [...]}` — read as 14 channel names, in order."""

    assert declared_names({"names": {"motors": ALOHA_MOTORS}}) == ALOHA_MOTORS


def test_anything_else_declares_no_names():
    assert declared_names({}) == []
    assert declared_names({"names": None}) == []
    assert declared_names({"names": {"motors": "left_waist"}}) == []


def test_nested_names_name_the_channels():
    """Channels come out as `left_gripper`, not `observation.state_6`."""

    spec = {"dtype": "float32", "shape": [14], "names": {"motors": ALOHA_MOTORS}}

    channels = channels_for("observation.state", spec, load_default_dictionary())

    assert [channel.name for channel in channels] == ALOHA_MOTORS


def test_a_channel_named_as_a_gripper_grades_as_one():
    assert channel_taxonomy("unmapped.observation.state", "left_gripper") == (
        "proprio.gripper_width"
    )
    assert channel_taxonomy("action.action_vector", "right_gripper") == (
        "action.gripper_command"
    )
    assert channel_taxonomy("unmapped.observation.state", "left_waist") == (
        "unmapped.observation.state"
    )
    assert channel_taxonomy("proprio.gripper_width", "gripper") == (
        "proprio.gripper_width"
    )
    # A gripper's motor current stays a torque channel.
    assert channel_taxonomy("proprio.joint_torque", "left_gripper") == (
        "proprio.joint_torque"
    )


def _write_with_gripper(path: Path, n_episodes: int, glitched: set[int]) -> None:
    """Smooth 50 Hz arm motion plus a gripper that holds, snaps shut, holds."""

    rng = np.random.default_rng(0)
    t = np.arange(200) * 0.02
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        data.attrs["fps"] = 50.0
        for index in range(n_episodes):
            group = data.create_group(f"demo_{index}")
            actions = np.stack(
                [0.2 * np.sin(2 * np.pi * 0.4 * t + phase) for phase in range(6)], 1
            )
            group.create_dataset("actions", data=actions)
            group.create_dataset("timestamps", data=t)
            gripper = np.where(np.arange(200) < 60 + index, 0.08, 0.01)
            if index in glitched:
                rows = rng.choice(np.arange(5, 195), size=10, replace=False)
                gripper = gripper.copy()
                gripper[rows] += rng.choice([-1, 1], 10) * 0.5
            group.create_dataset("left_gripper", data=gripper)


def test_a_gripper_that_holds_and_snaps_fails_no_episode(tmp_path):
    """Held for most of the episode, one snap: ALOHA's grippers, not a fault."""

    path = tmp_path / "arm.hdf5"
    _write_with_gripper(path, 20, glitched=set())

    report = grade(path)

    assert report.gate is not None
    assert report.gate.failing_episodes == []


def test_a_glitching_gripper_still_fails_its_episode(tmp_path):
    """The spike check still grades a gripper: a glitch fails the episode."""

    path = tmp_path / "arm.hdf5"
    _write_with_gripper(path, 20, glitched={4})

    report = grade_with_test_calibration(path)

    assert report.gate is not None
    assert [f.episode_id.rsplit("_", 1)[-1] for f in report.gate.failing_episodes] == [
        "4"
    ]
    assert any(
        reason.endswith("spike_pct")
        for failing in report.gate.failing_episodes
        for reason in failing.reasons
    )
