"""`lerobot_v2_0_tiny/` and `lerobot_v2_1_tiny/`: LeRobot v2.x datasets.

Each carries its own spec shape and its own writer.
The two specs prove different things:
- v2.0 exercises one parquet and one mp4 per episode in a single chunk;
- v2.1 exercises chunk rollover, at `chunks_size: 1`, and a camera-less tree.
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
class LeRobotV2Feature:
    """One vector feature: its `info.json` key and its per-channel motor names."""

    key: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class LeRobotV2TinySpec:
    """Everything `fixture_gen.writers._write_lerobot_v2` needs to build the tree.

    Attributes
    ----------
    dirname : str
        The dataset directory's name under `tests/fixtures/`.
    codebase_version : str
        `"v2.0"` or `"v2.1"`, which stats file `meta/` gets follows from it.
    fps : int
        The declared frame rate, read by `describe` rather than inferred.
    robot_type : str
        The declared robot platform.
    chunks_size : int
        How many episodes share a chunk directory.
    episode_lengths : tuple[int, ...]
        Frame count per episode, in episode order.
    state_feature : LeRobotV2Feature
        A vector feature whose names resolve to nothing in `dictionary.yaml`.
    velocity_feature : LeRobotV2Feature
        A vector feature whose key resolves to `proprio.joint_velocity`.
    video_key : str or None
        The single camera feature's key, or `None` to write no camera
        feature and no `videos/` tree at all.
    video_size : tuple[int, int]
        The camera's `(width, height)` in pixels.
    """

    dirname: str
    codebase_version: str
    fps: int
    robot_type: str
    chunks_size: int
    episode_lengths: tuple[int, ...]
    state_feature: LeRobotV2Feature
    velocity_feature: LeRobotV2Feature
    video_key: str | None
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

LEROBOT_V2_0_TINY = LeRobotV2TinySpec(
    dirname="lerobot_v2_0_tiny",
    codebase_version="v2.0",
    fps=30,
    robot_type="so100_follower",
    chunks_size=1000,
    episode_lengths=(8, 6),
    state_feature=LeRobotV2Feature(key="observation.state", names=_MOTOR_NAMES),
    velocity_feature=LeRobotV2Feature(key="observation.velocity", names=_MOTOR_NAMES),
    video_key="observation.images.up",
    video_size=(32, 32),
)

LEROBOT_V2_1_TINY = LeRobotV2TinySpec(
    dirname="lerobot_v2_1_tiny",
    codebase_version="v2.1",
    fps=30,
    robot_type="so100_follower",
    chunks_size=1,
    episode_lengths=(5, 4),
    state_feature=LeRobotV2Feature(key="observation.state", names=_MOTOR_NAMES),
    velocity_feature=LeRobotV2Feature(key="observation.velocity", names=_MOTOR_NAMES),
    video_key=None,
    video_size=(32, 32),
)
