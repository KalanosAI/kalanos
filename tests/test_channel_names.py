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
import pytest
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


def channel_taxonomy(stream_type: str, name: str, *, feature: str | None = None) -> str:
    """Exercise the resolver, rather than a hidden reporting-stage heuristic."""
    stream = Stream(
        taxonomy_type=stream_type,
        source_field=feature,
        kind=Kind.SERIES,
        timestamps=pl.Series("t", [0.0, 0.1]),
        source_path=UPath("fixture.csv"),
        timestamp_dtype=TimestampDtype.FLOAT64,
        channels=[Channel(name=name)],
    )
    resolved = resolve_stream(stream, dictionary=load_default_dictionary())
    binding = resolved.channels[0].binding
    assert binding is not None
    return binding.taxonomy_type


def _state_stream(channels: list[Channel]) -> Stream:
    """An unmapped `observation.state` stream holding the given channels."""
    return Stream(
        taxonomy_type="unmapped.observation.state",
        source_field="observation.state",
        kind=Kind.SERIES,
        timestamps=pl.Series("t", [0.0, 0.1]),
        source_path=UPath("fixture.csv"),
        timestamp_dtype=TimestampDtype.FLOAT64,
        channels=channels,
    )


def test_names_declared_as_a_list_are_read():
    assert declared_names({"names": ["x", "y", "z"]}) == ["x", "y", "z"]


def test_names_nested_under_a_label_are_read_as_aloha_declares_them():
    """ALOHA: `"names": {"motors": [...]}` — read as 14 channel names, in order."""

    assert declared_names({"names": {"motors": ALOHA_MOTORS}}) == ALOHA_MOTORS


def test_names_declared_as_a_list_of_lists_are_read():
    """Unitree declares `"names": [["kLeftHandThumb", ...]]`."""

    assert declared_names({"names": [["a", "b"], ["c"]]}) == ["a", "b", "c"]
    assert declared_names({"names": [["a", 1]]}) == []


def test_anything_else_declares_no_names():
    assert declared_names({}) == []
    assert declared_names({"names": None}) == []
    assert declared_names({"names": {"motors": "left_waist"}}) == []
    assert declared_names({"names": [["a"], "b"]}) == []


def test_nested_names_name_the_channels():
    """Channels come out as `left_gripper`, not `observation.state_6`."""

    spec = {"dtype": "float32", "shape": [14], "names": {"motors": ALOHA_MOTORS}}

    channels = channels_for("observation.state", spec, load_default_dictionary())

    assert [channel.name for channel in channels] == ALOHA_MOTORS


def test_a_multi_dimensional_feature_has_one_channel_per_value():
    """A `[1, 3]` end-effector position reads as three channels, in flat order."""

    spec = {"dtype": "float32", "shape": [1, 3], "names": None}

    channels = channels_for("f", spec, load_default_dictionary())

    assert [channel.name for channel in channels] == ["f_0", "f_1", "f_2"]
    assert [channel.source_index for channel in channels] == [0, 1, 2]


def test_a_multi_dimensional_feature_takes_its_nested_names():
    """Names nested as one list per row name the channels of a `[1, 3]` feature."""

    spec = {"dtype": "float32", "shape": [1, 3], "names": [["x", "y", "z"]]}

    channels = channels_for("f", spec, load_default_dictionary())

    assert [channel.name for channel in channels] == ["x", "y", "z"]


def test_a_square_grid_has_one_channel_per_cell():
    """A `[2, 2]` grid yields four channels, not one per row."""

    spec = {"dtype": "int16", "shape": [2, 2], "names": None}

    channels = channels_for("f", spec, load_default_dictionary())

    assert [channel.name for channel in channels] == ["f_0", "f_1", "f_2", "f_3"]


def test_repeated_names_are_indexed():
    """LIBERO's state names indices 6 and 7 `gripper`; each gets its own name."""

    names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]
    spec = {"dtype": "float32", "shape": [8], "names": names}

    channels = channels_for("observation.state", spec, load_default_dictionary())

    assert [channel.name for channel in channels] == [
        *names[:6],
        "observation.state[6]",
        "observation.state[7]",
    ]
    assert [channel.declared_name for channel in channels] == names
    assert [channel.source_index for channel in channels[6:]] == [6, 7]


def test_a_repeated_gripper_name_types_no_gripper():
    """A name two channels share can't tell them apart, so it types neither."""

    dictionary = load_default_dictionary()
    repeated = _state_stream(
        [
            Channel(
                name=f"observation.state[{i}]", declared_name="gripper", source_index=i
            )
            for i in (6, 7)
        ]
    )
    unique = _state_stream(
        [Channel(name="gripper", declared_name="gripper", source_index=0)]
    )

    resolved = resolve_stream(repeated, dictionary=dictionary)
    bindings = [channel.binding for channel in resolved.channels]
    [alone] = resolve_stream(unique, dictionary=dictionary).channels

    assert [b.taxonomy_type for b in bindings if b] == [
        "unmapped.observation.state",
        "unmapped.observation.state",
    ]
    assert [b.name for b in bindings if b] == ["gripper", "gripper"]
    assert alone.binding is not None
    assert alone.binding.taxonomy_type == "proprio.gripper_width"


def test_a_channel_named_as_a_gripper_grades_as_one():
    assert channel_taxonomy("unmapped.observation.state", "left_gripper") == (
        "proprio.gripper_width"
    )
    assert channel_taxonomy("action.action_vector", "right_gripper") == (
        "action.gripper_command"
    )
    assert channel_taxonomy(
        "unmapped.observation.state", "left_waist", feature="observation.state"
    ) == ("proprio.joint_position")
    assert channel_taxonomy("proprio.gripper_width", "gripper") == (
        "proprio.gripper_width"
    )
    # A gripper's motor current stays a torque channel.
    assert channel_taxonomy("proprio.joint_torque", "left_gripper") == (
        "proprio.joint_torque"
    )


def test_a_finger_joint_grades_as_a_gripper():
    assert channel_taxonomy("unmapped.observation.state", "finger_joint.pos") == (
        "proprio.gripper_width"
    )
    assert channel_taxonomy("action.action_vector", "finger_joint.pos") == (
        "action.gripper_command"
    )


def test_a_named_finger_is_not_a_gripper():
    for name in ("index_finger_joint_0", "thumb_finger.pos"):
        assert channel_taxonomy("unmapped.observation.state", name) == (
            "unmapped.observation.state"
        )


def test_a_hand_digit_grades_as_a_finger_joint():
    for name in ("kLeftHandThumb", "kRightHandPinky", "left_hand_index"):
        assert channel_taxonomy("unmapped.observation.state", name) == (
            "proprio.finger_joint_state"
        )
    assert channel_taxonomy("action.action_vector", "kLeftHandThumb") == (
        "action.action_vector"
    )


def test_a_hand_without_a_digit_is_not_a_finger_joint():
    for name in ("panda_hand_joint", "left_hand_pos_x", "hand_force"):
        assert channel_taxonomy("unmapped.observation.state", name) == (
            "unmapped.observation.state"
        )


@pytest.mark.parametrize(
    "name",
    [
        "shoulder_pan.pos",
        "shoulder_pan_joint.pos",
        "joint_1.pos",
        "main_shoulder_pan",
        "kLeftShoulderPitch",
        "left_arm_joint_1_rad",
        "left_waist",
    ],
)
def test_a_state_channel_named_as_an_arm_joint_grades_as_a_joint_position(name):
    assert channel_taxonomy(
        "unmapped.observation.state", name, feature="observation.state"
    ) == ("proprio.joint_position")


def test_a_joint_in_a_second_state_feature_grades_as_a_joint_position():
    feature = "observation.state.radian_urdf0"
    assert channel_taxonomy(
        f"unmapped.{feature}", "elbow_flex.pos", feature=feature
    ) == ("proprio.joint_position")


def test_an_end_effector_channel_in_the_state_grades_as_its_pose():
    assert channel_taxonomy(
        "unmapped.observation.state", "left_eef_pos_x_m", feature="observation.state"
    ) == ("proprio.ee_pose")


def test_a_patterned_gripper_or_finger_joint_still_grades_as_a_gripper():
    for name in ("gripper.pos", "finger_joint.pos"):
        assert channel_taxonomy(
            "unmapped.observation.state", name, feature="observation.state"
        ) == ("proprio.gripper_width")
    assert channel_taxonomy(
        "unmapped.observation.state", "kLeftHandThumb0", feature="observation.state"
    ) == ("proprio.finger_joint_state")


@pytest.mark.parametrize(
    "name",
    [
        "thumb_finger.pos",
        "panda_hand_joint",
        "left_hand_pos_x",
        "shoulder_pan.vel",
        "shoulder_pan.load",
        "shoulder_pan.acc",
        "kLeftShoulderPitch_dq",
        "kLeftElbow_tau",
        "elbow_speed",
        "wrist_cam_exposure",
        "left_eef_pos_x_vel",
    ],
)
def test_a_hand_or_other_quantity_channel_is_no_joint_position(name):
    assert channel_taxonomy(
        "unmapped.observation.state", name, feature="observation.state"
    ) == ("unmapped.observation.state")


def test_an_unmapped_action_feature_types_its_joints_and_gripper():
    feature = "action.radian_urdf0"
    assert channel_taxonomy(
        f"unmapped.{feature}", "wrist_roll.pos", feature=feature
    ) == ("action.joint_position_command")
    assert channel_taxonomy(f"unmapped.{feature}", "gripper.pos", feature=feature) == (
        "action.gripper_command"
    )
    assert channel_taxonomy(f"unmapped.{feature}", "gripper.vel", feature=feature) == (
        f"unmapped.{feature}"
    )


def test_the_main_action_feature_keeps_its_type():
    assert channel_taxonomy(
        "action.action_vector", "wrist_roll.pos", feature="action"
    ) == ("action.action_vector")


def test_a_joint_name_outside_the_state_features_stays_unmapped():
    assert channel_taxonomy("unmapped.foo", "shoulder_pan.pos", feature="foo") == (
        "unmapped.foo"
    )
    feature = "observation.state_velocity"
    assert channel_taxonomy(f"unmapped.{feature}", "shoulder_pan", feature=feature) == (
        f"unmapped.{feature}"
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
