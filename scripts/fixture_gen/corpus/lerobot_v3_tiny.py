"""`lerobot_v3_tiny/`: a directory-shaped LeRobot v3.0 dataset.

A tree of five files in three formats, so it carries its own spec shape and its
own writer instead of fitting `fixture_gen`'s flat-time-grid machinery.
Two episodes share one data parquet and one mp4, exercising the split-by-episode-index
and several-files-per-episode pathologies this format exists to prove.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class LeRobotV3Feature:
    """One vector feature: its `info.json` key and its per-channel motor names."""

    key: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class LeRobotV3TinySpec:
    """Everything `fixture_gen.writers._write_lerobot_v3` needs to build the tree.

    Attributes
    ----------
    dirname : str
        The dataset directory's name under `tests/fixtures/`.
    fps : int
        The declared frame rate, read by `describe` rather than inferred.
    robot_type : str
        The declared robot platform.
    episode_lengths : tuple[int, ...]
        Frame count per episode, in episode order.
    state_feature : LeRobotV3Feature
        A vector feature whose names resolve to nothing in `dictionary.yaml`.
    velocity_feature : LeRobotV3Feature
        A vector feature whose key resolves to `proprio.joint_velocity`.
    video_key : str
        The single camera feature's key.
    video_size : tuple[int, int]
        The camera's `(width, height)` in pixels.
    """

    dirname: str
    fps: int
    robot_type: str
    episode_lengths: tuple[int, ...]
    state_feature: LeRobotV3Feature
    velocity_feature: LeRobotV3Feature
    video_key: str
    video_size: tuple[int, int]


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀

_MOTOR_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

LEROBOT_V3_TINY = LeRobotV3TinySpec(
    dirname="lerobot_v3_tiny",
    fps=30,
    robot_type="so100_follower",
    episode_lengths=(8, 6),
    state_feature=LeRobotV3Feature(key="observation.state", names=_MOTOR_NAMES),
    velocity_feature=LeRobotV3Feature(key="observation.velocity", names=_MOTOR_NAMES),
    video_key="observation.images.up",
    video_size=(32, 32),
)
