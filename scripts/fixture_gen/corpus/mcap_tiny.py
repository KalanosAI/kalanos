"""`mcap_tiny.mcap`: three ROS 2 topics, each exercising a different adapter path.

- `/joint_states` carries a `std_msgs/Header`, so its clock is capture rather than log;
- `/cmd_vel` carries none, so its clock is log;
- `/camera/wrist/image_raw` is a blob-payload topic with no channels at all.
One resolves through its message type, one through its topic name, one through neither.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

from dataclasses import dataclass


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The ROS 2 concatenated-msgdef separator: a line of 80 `=`.
_SEPARATOR = "=" * 80

_HEADER_SUFFIX = (
    f"{_SEPARATOR}\n"
    "MSG: std_msgs/Header\n"
    "builtin_interfaces/Time stamp\n"
    "string frame_id\n"
    f"{_SEPARATOR}\n"
    "MSG: builtin_interfaces/Time\n"
    "int32 sec\n"
    "uint32 nanosec\n"
)

_JOINT_STATE_MSGDEF = (
    "std_msgs/Header header\n"
    "string[] name\n"
    "float64[] position\n"
    "float64[] velocity\n"
    "float64[] effort\n"
) + _HEADER_SUFFIX

_TWIST_MSGDEF = (
    "geometry_msgs/Vector3 linear\n"
    "geometry_msgs/Vector3 angular\n"
    f"{_SEPARATOR}\n"
    "MSG: geometry_msgs/Vector3\n"
    "float64 x\n"
    "float64 y\n"
    "float64 z\n"
)

_IMAGE_MSGDEF = (
    "std_msgs/Header header\n"
    "uint32 height\n"
    "uint32 width\n"
    "string encoding\n"
    "uint8 is_bigendian\n"
    "uint32 step\n"
    "uint8[] data\n"
) + _HEADER_SUFFIX


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class McapTopic:
    """One topic to write: its name, message definition, and message count."""

    topic: str
    msgdef: str
    count: int
    rate_hz: float


@dataclass(frozen=True)
class McapTinySpec:
    """Everything `fixture_gen.writers._write_mcap_tiny` needs to build the file.

    Attributes
    ----------
    filename : str
        The file's name under `tests/fixtures/`.
    joint : McapTopic
        The `sensor_msgs/msg/JointState` topic, header-stamped.
    joint_log_offset_ns : int
        How far each joint-state message's `log_time` sits after its
        `header.stamp`, so the capture-versus-log split is real in the file
        rather than assumed.
    twist : McapTopic
        The `geometry_msgs/msg/Twist` topic, carrying no header.
    image : McapTopic
        The `sensor_msgs/msg/Image` topic, a blob-payload message.
    image_size : tuple[int, int]
        The image's `(width, height)` in pixels.
    """

    filename: str
    joint: McapTopic
    joint_log_offset_ns: int
    twist: McapTopic
    image: McapTopic
    image_size: tuple[int, int]


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀

MCAP_TINY = McapTinySpec(
    filename="mcap_tiny.mcap",
    joint=McapTopic(
        topic="/joint_states", msgdef=_JOINT_STATE_MSGDEF, count=6, rate_hz=100.0
    ),
    joint_log_offset_ns=500_000,
    twist=McapTopic(topic="/cmd_vel", msgdef=_TWIST_MSGDEF, count=3, rate_hz=50.0),
    image=McapTopic(
        topic="/camera/wrist/image_raw", msgdef=_IMAGE_MSGDEF, count=2, rate_hz=2.0
    ),
    image_size=(2, 2),
)
