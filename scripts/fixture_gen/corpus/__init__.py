"""Every constant the fixture corpus is built from.

One module per file under `tests/fixtures/`, each exporting one
`FixtureSpec`. Adding a fixture in an existing output format is a new
module plus an entry in `FIXTURES`, not a change to `fixture_gen.sampling`
or `fixture_gen.writers`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from fixture_gen.corpus.arm_multi_device import ARM_MULTI_DEVICE
from fixture_gen.corpus.capture_index import CAPTURE_INDEX
from fixture_gen.corpus.hdf5_tiny import HDF5_TINY, Hdf5TinySpec
from fixture_gen.corpus.imu_stream import IMU_STREAM
from fixture_gen.corpus.lerobot_v2_tiny import (
    LEROBOT_V2_0_TINY,
    LEROBOT_V2_1_TINY,
    LeRobotV2TinySpec,
)
from fixture_gen.corpus.lerobot_v3_tiny import LEROBOT_V3_TINY, LeRobotV3TinySpec
from fixture_gen.corpus.mcap_tiny import MCAP_TINY, McapTinySpec
from fixture_gen.corpus.pose_log import POSE_LOG
from fixture_gen.corpus.video_meta import VIDEO_META
from fixture_gen.models import FlatFixtureSpec


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FixtureSpec = (
    FlatFixtureSpec
    | LeRobotV3TinySpec
    | LeRobotV2TinySpec
    | Hdf5TinySpec
    | McapTinySpec
)

FIXTURES: list[FixtureSpec] = [
    ARM_MULTI_DEVICE,
    IMU_STREAM,
    POSE_LOG,
    CAPTURE_INDEX,
    VIDEO_META,
    LEROBOT_V3_TINY,
    LEROBOT_V2_0_TINY,
    LEROBOT_V2_1_TINY,
    HDF5_TINY,
    MCAP_TINY,
]

__all__ = ["FIXTURES", "FixtureSpec"]
