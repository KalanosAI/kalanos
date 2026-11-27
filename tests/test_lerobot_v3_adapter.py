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
from fractions import Fraction

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.lerobot import v3
from kalanos.analysis.adapters.lerobot.common import (
    CONFIDENCE,
    declared_format,
    episode_clock,
    holds_numbers,
    resolve_taxonomy,
    series_stream,
    storage_columns,
)
from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter
from kalanos.analysis.execution import use_tier
from kalanos.analysis.models.adapters import AdapterRefusal
from kalanos.analysis.models.binding import Representation
from kalanos.analysis.models.domain import (
    Channel,
    Clock,
    FramePayload,
    Kind,
    MappingSource,
    TimestampDtype,
)
from kalanos.analysis.models.eligibility import Consequence, EligibilityStatus
from kalanos.analysis.models.metrics import MetricStatus
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import AnalysedEpisode
from kalanos.analysis.reporting.assemble import assemble_report
from kalanos.api import grade
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.assets.policy import load_default_policy
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, LEROBOT_FIXTURE, lerobot_stats, with_repeated_gripper


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

# How polars 2 loads a Hugging Face `datasets` Array2D column.
# polars 1 loads the same column as its storage.
_ARRAY2D = pl.Extension(
    "datasets.features.features.Array2DExtensionType",
    pl.List(pl.List(pl.Int16)),
    '[[2, 2], "int16"]',
)

_POLARS_2 = pytest.mark.skipif(
    int(pl.__version__.split(".")[0]) < 2,
    reason="polars 1 loads extension columns as their storage",
)

_MISMATCH_METRIC = "feature_mismatch_count"
_MISMATCH_FINDING = f"schema.{_MISMATCH_METRIC}"

_MISSING_FEATURE = {
    "observation.effort": {"dtype": "float32", "shape": [6], "names": None}
}

_POINTER_METRIC = "episode_file_mismatch"
_POINTER_FINDING = f"schema.{_POINTER_METRIC}"
_ORPHAN_FINDING = "schema.episode_not_in_metadata"
_RANGE_METRIC = "index_range_mismatch_count"
_RANGE_FINDING = f"schema.{_RANGE_METRIC}"

_FEATURE_DTYPES = {"float32": pl.Float32, "int64": pl.Int64}

# Real metadata rows and data `index` ranges, read 2026-11-10.
# Pointer and declared ranges per episode: (episode, file index, from, to, length).
_G1_META = [
    (394, 0, 333583, 334664, 1081),
    (395, 0, 334664, 335446, 782),
    (396, 0, 335446, 336051, 605),
    (397, 1, 605, 1224, 619),
    (398, 1, 1224, 1888, 664),
    (399, 1, 1888, 2773, 885),
]
_G1_DATA = {
    0: [(394, 333583, 334664), (395, 334664, 335446)],
    1: [
        (396, 335446, 336051),
        (397, 336051, 336670),
        (398, 336670, 337334),
        (399, 337334, 338219),
    ],
}
_LIBERO_DATA = {
    0: [(0, 0, 214), (1, 214, 498), (2, 498, 843)],
    1: [(3, 843, 1128), (4, 1128, 1406), (5, 1406, 1646)],
    2: [(6, 1646, 1925), (7, 1925, 2197)],
}

# Pointer and declared ranges for libero episodes 0-7, in the shape of `_G1_META`.
# Both repos declare the data's own ranges. HuggingFaceVLA/libero points every episode
# at file-000, lerobot/libero at the file that holds it.
_STALE_LIBERO_META = [
    (0, 0, 0, 214, 214),
    (1, 0, 214, 498, 284),
    (2, 0, 498, 843, 345),
    (3, 0, 843, 1128, 285),
    (4, 0, 1128, 1406, 278),
    (5, 0, 1406, 1646, 240),
    (6, 0, 1646, 1925, 279),
    (7, 0, 1925, 2197, 272),
]
_CLEAN_LIBERO_META = [
    (0, 0, 0, 214, 214),
    (1, 0, 214, 498, 284),
    (2, 0, 498, 843, 345),
    (3, 1, 843, 1128, 285),
    (4, 1, 1128, 1406, 278),
    (5, 1, 1406, 1646, 240),
    (6, 2, 1646, 1925, 279),
    (7, 2, 1925, 2197, 272),
]

# Real metadata rows and data `index` ranges, read 2026-11-17, shaped like `_G1_META`.
# The metadata names file-001 for episodes 231 and 232, and no file-001 exists.
_MISSING_FILE_META = [
    (229, 0, 77038, 77357, 319),
    (230, 0, 77357, 77665, 308),
    (231, 1, 77665, 78001, 336),
    (232, 1, 78001, 78327, 326),
    (700, 0, 217062, 217552, 490),
]
_MISSING_FILE_DATA = {
    0: [
        (229, 77038, 77357),
        (230, 77357, 77665),
        (231, 77665, 78001),
        (232, 78001, 78327),
        (700, 217062, 217552),
    ]
}


# The first two frames of `obs.state.manipulator.eef_pos` in
# `sleepyshaman123/stack_ur5_allegro`, a `[1, 3]` feature stored as one row per frame.
_EEF_POS = [[[0.490671, -0.134666, 0.587828]], [[0.490398, -0.134538, 0.586533]]]

_EEF_FEATURES = {
    "obs.state.manipulator.eef_pos": {
        "dtype": "float32",
        "shape": [1, 3],
        "names": None,
    },
    "obs.state.manipulator.eef_quat": {
        "dtype": "float32",
        "shape": [1, 4],
        "names": None,
    },
}


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


def _layout_dataset(root, meta_rows, data_files, fps=None):
    """Write a LeRobot v3 tree with the given episode pointers and data layout.

    Parameters
    ----------
    root : Path
        Where to write it.
    meta_rows : list[tuple[int, int, int, int, int]]
        One metadata row per episode: episode index, data file index,
        `dataset_from_index`, `dataset_to_index` and `length`.
    data_files : dict[int, list[tuple[int, int, int]]]
        The episodes each data file holds, by file index:
        episode index and its `index` range, end exclusive.
    fps : int or None
        The declared rate, the fixture's when `None`.

    Returns
    -------
    UPath
        The dataset root.
    """

    info = json.loads((LEROBOT_FIXTURE / "meta" / "info.json").read_text())
    del info["features"]["observation.images.up"]
    if fps is not None:
        info["fps"] = fps
    ranges = [r for rows in data_files.values() for r in rows]
    info["total_episodes"] = len(meta_rows)
    info["total_frames"] = sum(hi - lo for _, lo, hi in ranges)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    pl.DataFrame(
        meta_rows,
        schema=[
            "episode_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
            "length",
        ],
        orient="row",
    ).with_columns(
        pl.lit(0, pl.Int64).alias("data/chunk_index"),
        pl.lit(["pick up the cube"], pl.List(pl.String)).alias("tasks"),
    ).write_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    for file_index, episodes in data_files.items():
        episode_index = [e for e, lo, hi in episodes for _ in range(lo, hi)]
        index = [i for _, lo, hi in episodes for i in range(lo, hi)]
        frame_index = [i - lo for _, lo, hi in episodes for i in range(lo, hi)]
        columns = {
            "episode_index": pl.Series(episode_index, dtype=pl.Int64),
            "index": pl.Series(index, dtype=pl.Int64),
            "frame_index": pl.Series(frame_index, dtype=pl.Int64),
            "timestamp": pl.Series(frame_index, dtype=pl.Float32) / info["fps"],
        }
        for key, feature in info["features"].items():
            if key in columns:
                continue
            dtype = _FEATURE_DTYPES[feature["dtype"]]
            if feature["shape"] == [1]:
                columns[key] = pl.Series([0] * len(index), dtype=dtype)
            else:
                width = feature["shape"][0]
                columns[key] = pl.Series(
                    [[0] * width] * len(index), dtype=pl.Array(dtype, width)
                )
        pl.DataFrame(columns).write_parquet(
            root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
        )
    return UPath(root)


def _write_camera_file(path, frames):
    """Encode `frames` 16x16 frames at 15 fps, each a level of its own.

    Parameters
    ----------
    path : Path
        Where to write the mp4.
    frames : int
        How many frames it holds, from pts 0.
    """

    av = pytest.importorskip("av")
    numpy = pytest.importorskip("numpy")

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=15, options={"bf": "0"})
    stream.width = stream.height = 16
    stream.pix_fmt = "yuv420p"
    stream.codec_context.time_base = Fraction(1, 15)
    for k in range(frames):
        ramp = (20 + (k * 7) % 150 + numpy.arange(16)).astype(numpy.uint8)
        pixels = numpy.broadcast_to(ramp[None, :, None], (16, 16, 3)).copy()
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = k
        frame.time_base = Fraction(1, 15)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _taco_like_dataset(root, *, fixed_pointer):
    """Write three 66-frame episodes, the middle one's fixed camera past its file.

    It copies the layout of `lerobot/taco_play` episodes 2141 to 2143:
    `rgb_static/file-000.mp4` holds episode 2141 alone,
    while episode 2142's pointer names it at the 4.4 s that follow,
    and `file-001.mp4` holds 2142's frames and then 2143's.
    `rgb_gripper/file-000.mp4` holds all three, each pointer inside it.

    Parameters
    ----------
    root : Path
        Where to write it.
    fixed_pointer : bool
        Whether episode 2142's fixed camera points at `file-001.mp4` from 0.0 s,
        where its frames are.

    Returns
    -------
    UPath
        The dataset root.
    """

    episodes = (2141, 2142, 2143)
    root = _layout_dataset(
        root,
        [(e, 0, 66 * k, 66 * (k + 1), 66) for k, e in enumerate(episodes)],
        {0: [(e, 66 * k, 66 * (k + 1)) for k, e in enumerate(episodes)]},
        fps=15,
    )
    static, gripper = "observation.images.rgb_static", "observation.images.rgb_gripper"
    info = json.loads((root / "meta" / "info.json").read_text())
    for key in (static, gripper):
        info["features"][key] = {
            "dtype": "video",
            "shape": [16, 16, 3],
            "info": {"video.fps": 15},
        }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    videos = root / "videos"
    _write_camera_file(videos / static / "chunk-000" / "file-000.mp4", 66)
    _write_camera_file(videos / static / "chunk-000" / "file-001.mp4", 132)
    _write_camera_file(videos / gripper / "chunk-000" / "file-000.mp4", 198)

    # Per episode: (file index, from, to) for each camera.
    misplaced = (0, 4.4, 8.8) if not fixed_pointer else (1, 0.0, 4.4)
    pointers = {
        static: [(0, 0.0, 4.4), misplaced, (1, 4.4, 8.8)],
        gripper: [(0, 0.0, 4.4), (0, 4.4, 8.8), (0, 8.8, 13.2)],
    }
    index_file = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    index = pl.read_parquet(str(index_file))
    for key, rows in pointers.items():
        file_index, start, end = zip(*rows, strict=True)
        index = index.with_columns(
            pl.lit(0, pl.Int64).alias(f"videos/{key}/chunk_index"),
            pl.Series(f"videos/{key}/file_index", file_index, dtype=pl.Int64),
            pl.Series(f"videos/{key}/from_timestamp", start, dtype=pl.Float64),
            pl.Series(f"videos/{key}/to_timestamp", end, dtype=pl.Float64),
        )
    index.write_parquet(str(index_file))
    return root


def _camera_presence(report):
    """Read each camera's presence result by episode and camera key.

    Parameters
    ----------
    report : Report
        The graded report.

    Returns
    -------
    dict[tuple[int, str], MetricResult]
        `camera_stream_present` per episode index and camera key.
    """

    presence = {}
    for episode in report.episodes:
        for stream in episode.streams:
            result = stream.metrics.get("camera_stream_present")
            if result is not None:
                camera = result.evidence.get("camera")
                presence[(_episode_number(episode.id), camera)] = result
    return presence


def _episode_findings(report, metric_id):
    """Group a report's findings of one metric by episode.

    Parameters
    ----------
    report : Report
        The graded report.
    metric_id : str
        The finding's metric id.

    Returns
    -------
    dict[str, list[Finding]]
        The findings, by episode id.
    """

    grouped = {}
    for finding in report.findings:
        if finding.metric_id == metric_id:
            grouped.setdefault(finding.episode_id, []).append(finding)
    return grouped


def _episode_number(episode_id):
    """Read the episode index back from an episode id.

    Parameters
    ----------
    episode_id : str
        An id ending in `episode_<index>`.

    Returns
    -------
    int
        The episode index.
    """

    return int(episode_id.rsplit("episode_", 1)[1])


def _with_repeated_index(root, kept=None):
    """Copy the fixture and add a second episode index file repeating its rows.

    The copy's rows carry their own task, so a test can tell which row an episode
    was read from.

    Parameters
    ----------
    root : Path
        Where to write the copy.
    kept : list[int] or None
        The episode indices the second file repeats. `None` repeats them all.

    Returns
    -------
    UPath
        The dataset root.
    """

    shutil.copytree(str(LEROBOT_FIXTURE), root)
    index_dir = root / "meta" / "episodes" / "chunk-000"
    index = pl.read_parquet(index_dir / "file-000.parquet")
    if kept is not None:
        index = index.filter(pl.col("episode_index").is_in(kept))
    index.with_columns(
        pl.lit(["fold the towel"], pl.List(pl.String)).alias("tasks")
    ).write_parquet(index_dir / "file-001.parquet")
    return UPath(root)


def _split_episode_index(root, edit_second):
    """Copy the fixture to `root` with its episode index split across two files.

    Parameters
    ----------
    root : Path
        Where to copy the fixture.
    edit_second : Callable[[pl.DataFrame], pl.DataFrame]
        Applied to the second file's single row before it is written.

    Returns
    -------
    UPath
        The copy's root.
    """

    shutil.copytree(str(LEROBOT_FIXTURE), root)
    index_dir = root / "meta" / "episodes" / "chunk-000"
    index = pl.read_parquet(index_dir / "file-000.parquet")
    index.head(1).write_parquet(index_dir / "file-000.parquet")
    edit_second(index.tail(1)).write_parquet(index_dir / "file-001.parquet")
    return UPath(root)


def _vector_stream(frame, channels):
    """Build the stream of `frame`'s column `f` with `channels`.

    Parameters
    ----------
    frame : pl.DataFrame
        One column, `f`, one row per frame.
    channels : list[Channel]
        The channels the feature declares.

    Returns
    -------
    pl.DataFrame
        The stream's payload, one column per channel.
    """

    stream = series_stream(
        frame,
        "f",
        channels,
        "unmapped.f",
        pl.Series([0.1 * i for i in range(frame.height)]),
        UPath("data/chunk-000/file-000.parquet"),
        mapping_source=None,
        clock=Clock.UNKNOWN,
        timestamp_dtype=TimestampDtype.FLOAT64,
        is_regular=True,
    )
    assert isinstance(stream.payload, FramePayload)
    return stream.payload.frame


def _eef_columns(height):
    """Build `[1, 3]` position and `[1, 4]` quaternion columns, as lists of lists.

    Parameters
    ----------
    height : int
        The data parquet's row count.

    Returns
    -------
    list[pl.Series]
        The two columns.
    """

    nested = pl.List(pl.List(pl.Float32))
    return [
        pl.Series(
            "obs.state.manipulator.eef_pos",
            [[[0.49 + 0.001 * i, -0.13, 0.58 - 0.001 * i]] for i in range(height)],
            dtype=nested,
        ),
        pl.Series(
            "obs.state.manipulator.eef_quat",
            [[[1.0, 0.0, 0.0, 0.0]] for _ in range(height)],
            dtype=nested,
        ),
    ]


def _short_row_dataset(root):
    """Copy the fixture with a `[3]` list feature whose second row holds 2 values.

    Parameters
    ----------
    root : Path
        Where to copy the fixture.

    Returns
    -------
    UPath
        The copied dataset root.
    """

    def columns(height):
        rows = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(height)]
        rows[1] = rows[1][:2]
        return [pl.Series("ee.position", rows, dtype=pl.List(pl.Float32))]

    return _with_features(
        root,
        {"ee.position": {"dtype": "float32", "shape": [3], "names": None}},
        columns,
    )


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


@pytest.mark.parametrize(
    ("fps", "expected"),
    [
        (30, 30.0),
        ("30", 30.0),
        (0, None),
        (-30, None),
        ("thirty", None),
        (True, None),
        (None, None),
    ],
)
def test_an_unusable_declared_rate_is_reported_as_none(tmp_path, fps, expected):
    """Verify describe() keeps a usable `fps` and drops anything else to None.

    A `None` case removes the key, so an absent rate is covered too.
    """

    copy_root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), copy_root)
    info_path = copy_root / "meta" / "info.json"
    info_json = json.loads(info_path.read_text())
    info_json.pop("fps")
    if fps is not None:
        info_json["fps"] = fps
    info_path.write_text(json.dumps(info_json))

    info = LeRobotV3Adapter().describe(UPath(copy_root))

    assert info.nominal_rate_hz == expected


def test_a_camera_stream_is_never_fully_decoded_while_grading(monkeypatch):
    """Verify grading the whole fixture samples frames and never decodes them all."""

    monkeypatch.setattr(
        video.VideoPayload,
        "fetch",
        lambda self: pytest.fail("VideoPayload.fetch() was called while grading"),
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

    monkeypatch.setattr(video, "av", None)

    [first, _] = _episodes()
    [video_stream] = [s for s in first.streams if s.kind == Kind.VIDEO]

    assert video_stream.payload is not None
    assert len(video_stream.payload) == len(video_stream.timestamps)
    # The fixture's shape has no `names`, so only the rate is declared.
    assert video_stream.payload.declared == video.DeclaredFormat(fps=30.0)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            {
                "shape": [96, 128, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": 10.0, "video.pix_fmt": "yuv420p"},
            },
            video.DeclaredFormat(width=128, height=96, fps=10.0, pix_fmt="yuv420p"),
        ),
        (
            {
                "shape": [3, 96, 128],
                "info": {"video.height": 480, "video.width": 640},
            },
            video.DeclaredFormat(width=640, height=480),
        ),
        ({"shape": [32, 32, 3]}, None),
        (
            {"info": {"video.fps": "fast", "video.height": 48}},
            video.DeclaredFormat(height=48),
        ),
        (
            {
                "video_info": {
                    "video.codec": "av1",
                    "video.channels": 3,
                    "video.is_depth_map": True,
                    "has_audio": False,
                }
            },
            video.DeclaredFormat(
                codec="av1", channels=3, is_depth_map=True, has_audio=False
            ),
        ),
        (
            {
                "info": {
                    "video.channels": True,
                    "video.is_depth_map": "false",
                    "has_audio": 0,
                    "video.fps": 30,
                }
            },
            video.DeclaredFormat(fps=30.0),
        ),
    ],
    ids=[
        "v3_video_info",
        "info_size",
        "nothing",
        "uncoercible",
        "codec_and_flags",
        "flags_not_booleans",
    ],
)
def test_declared_format_reads_only_what_the_feature_states(spec, expected):
    """Verify the frame size comes from labelled axes or `info`, and bad values drop."""

    assert declared_format(spec) == expected


def test_a_feature_key_resolves_through_the_dictionary():
    """Verify observation.velocity resolves to proprio.joint_velocity by its key."""

    [first, _] = _episodes()
    velocity = next(
        s for s in first.streams if s.source_field == "observation.velocity"
    )

    assert velocity.taxonomy_type == "proprio.joint_velocity"
    assert velocity.mapping_source is MappingSource.DICTIONARY


def test_an_unresolvable_feature_reaches_the_report_unmapped():
    """Verify observation.state is unmapped, its channels named from info.json."""

    [first, _] = _episodes()
    state = next(s for s in first.streams if s.source_field == "observation.state")

    assert state.taxonomy_type == "unmapped.observation.state"
    assert state.mapping_source is None
    assert [c.name for c in state.channels] == _MOTOR_NAMES


def test_a_video_key_resolves_through_the_dictionary_by_its_own_key():
    """Verify a camera feature key that the dictionary does claim resolves.

    The fixture's own camera (`observation.images.up`) deliberately does
    not resolve, since LeRobot's real key names an unlisted alias. So this
    test exercises the success path directly rather than through the fixture.
    """

    resolved = resolve_taxonomy(
        "observation.images.wrist", {"dtype": "video"}, load_default_dictionary()
    )

    assert resolved == ("extero.wrist_rgb", MappingSource.DICTIONARY)


def test_a_feature_resolves_through_its_declared_names_when_its_key_does_not():
    """Verify an unlisted key whose channel names agree on one type takes it."""

    spec = {"dtype": "float32", "shape": [2], "names": ["joint_vel_0", "joint_vel_1"]}

    resolved = resolve_taxonomy("observation.foo", spec, load_default_dictionary())

    assert resolved == ("proprio.joint_velocity", MappingSource.DECLARED_NAMES)


def test_a_goal_position_feature_takes_the_goal_type():
    """A goal's parts nest under `goal_position`, past the last-segment match."""

    dictionary = load_default_dictionary()
    arm = {"dtype": "float32", "shape": [2], "names": ["shoulder_pan.pos", "elbow.pos"]}
    gripper = {"dtype": "float32", "shape": [1], "names": ["gripper.pos"]}

    expected = ("task.goal_position", MappingSource.DICTIONARY)
    assert resolve_taxonomy("skill.goal_position.joint", arm, dictionary) == expected
    assert (
        resolve_taxonomy("skill.goal_position.gripper", gripper, dictionary) == expected
    )


def test_a_goal_segment_must_be_whole():
    dictionary = load_default_dictionary()
    spec = {"dtype": "float32", "shape": [2], "names": ["a", "b"]}

    assert resolve_taxonomy("skill.goal_positions.joint", spec, dictionary) == (
        "unmapped.skill.goal_positions.joint",
        None,
    )
    assert resolve_taxonomy("subtask.target_position", spec, dictionary) == (
        "unmapped.subtask.target_position",
        None,
    )


def test_a_goal_key_beats_names_that_agree_on_another_type():
    """The key names a held target, whatever its channels are called."""

    spec = {"dtype": "float32", "shape": [2], "names": ["joint_vel_0", "joint_vel_1"]}

    resolved = resolve_taxonomy(
        "goal_position.unlisted", spec, load_default_dictionary()
    )

    assert resolved == ("task.goal_position", MappingSource.DICTIONARY)


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
        mapping_source=None,
        clock=Clock.UNKNOWN,
        timestamp_dtype=TimestampDtype.FLOAT64,
        is_regular=True,
    )

    assert isinstance(stream.payload, FramePayload)
    assert stream.payload.frame.columns == ["x", "y"]
    assert stream.payload.frame["y"].to_list() == [1.0, 3.0]


@_POLARS_2
def test_an_extension_column_is_read_as_its_storage():
    """Verify an extension column reads as its storage, other columns as they were."""

    grids = pl.Series(
        "observation.tactile", [[[1, 2], [3, 4]]], dtype=pl.List(pl.List(pl.Int16))
    )
    frame = pl.DataFrame(
        [grids.ext.to(_ARRAY2D), pl.Series("x", [0.5], dtype=pl.Float32)]
    )

    read = storage_columns(frame)

    assert read.schema == pl.Schema(
        {"observation.tactile": pl.List(pl.List(pl.Int16)), "x": pl.Float32}
    )
    assert read["observation.tactile"].to_list() == [[[1, 2], [3, 4]]]


@_POLARS_2
def test_a_2d_feature_grades_under_polars_2(tmp_path):
    """Verify a 2-D extension column reads as one channel per value of its grid."""

    copy_root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), copy_root)
    info_path = copy_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.tactile"] = {
        "dtype": "int16",
        "shape": [2, 2],
        "names": None,
    }
    info_path.write_text(json.dumps(info))
    data_path = copy_root / "data" / "chunk-000" / "file-000.parquet"
    frame = pl.read_parquet(data_path)
    grids = pl.Series(
        "observation.tactile",
        [[[i, i + 1], [i + 2, i + 3]] for i in range(frame.height)],
        dtype=pl.List(pl.List(pl.Int16)),
    )
    frame.with_columns(grids.ext.to(_ARRAY2D)).write_parquet(data_path)

    episodes = list(LeRobotV3Adapter().episodes(UPath(copy_root)))

    assert len(episodes) == 2
    for episode in episodes:
        tactile = next(
            s for s in episode.streams if s.source_field == "observation.tactile"
        )
        assert len(tactile.channels) == 4


def test_storage_columns_leaves_a_frame_without_extensions_unchanged():
    """Verify a frame with no extension column passes through as it was."""

    frame = pl.DataFrame(
        {
            "observation.tactile": pl.Series(
                [[[1, 2], [3, 4]]], dtype=pl.List(pl.List(pl.Int16))
            ),
            "x": pl.Series([0.5], dtype=pl.Float32),
        }
    )

    assert storage_columns(frame).schema == frame.schema


def test_frame_number_stamps_are_labelled_a_reconstructed_float32_clock():
    """LeRobot stamps frames as float32 `frame_index / fps`; so does the fixture."""

    for episode in _episodes():
        for stream in episode.streams:
            assert stream.timestamp_dtype is TimestampDtype.FLOAT32
            assert stream.clock is Clock.RECONSTRUCTED


def test_a_skipped_frame_does_not_erase_inferred_generation():
    """Generation inference survives a missing row, but needs a declared rate."""

    def frame(indices):
        return pl.DataFrame(
            {
                "frame_index": indices,
                "timestamp": pl.Series([i / 30 for i in indices], dtype=pl.Float32),
            }
        )

    whole, skipped = frame(list(range(10))), frame([0, 1, 2, 4, 5, 6])
    dtype = TimestampDtype.FLOAT32

    assert episode_clock(whole, 30.0, dtype) is Clock.RECONSTRUCTED
    assert episode_clock(skipped, 30.0, dtype) is Clock.RECONSTRUCTED
    assert episode_clock(whole, None, dtype) is Clock.UNKNOWN


def test_an_episode_carries_no_split_when_info_declares_none():
    """Verify a dataset whose `info.json` has no `splits` assigns none."""

    episodes = _episodes()

    assert episodes
    assert all(e.split is None for e in episodes)


def test_an_episode_index_without_data_columns_is_refused(tmp_path):
    """Verify an episode index missing the data file columns is refused, naming them."""

    root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), root)
    index_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    index = pl.read_parquet(index_path)
    index.drop("data/chunk_index", "data/file_index").write_parquet(index_path)

    with pytest.raises(
        AdapterRefusal, match=r"lacks .*data/chunk_index, data/file_index"
    ):
        list(LeRobotV3Adapter().episodes(UPath(root)))


def test_an_episode_index_without_episode_index_is_refused(tmp_path):
    """Verify an episode index missing `episode_index` is refused, naming it."""

    root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), root)
    index_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    index = pl.read_parquet(index_path)
    index.drop("episode_index").write_parquet(index_path)

    with pytest.raises(
        AdapterRefusal, match=r"lacks the LeRobot v3 columns episode_index$"
    ):
        list(LeRobotV3Adapter().episodes(UPath(root)))


def test_an_episode_index_out_of_order_is_read_in_episode_order(tmp_path):
    """Verify episodes come out in `episode_index` order whatever the row order."""

    root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), root)
    index_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    index = pl.read_parquet(index_path)
    index.reverse().write_parquet(index_path)

    episodes = list(LeRobotV3Adapter().episodes(UPath(root)))

    assert [e.id for e in episodes] == ["episode_000000", "episode_000001"]


def test_list_typed_index_columns_are_refused(tmp_path):
    """Verify a `frame_index` holding one list per frame is refused with its type."""

    root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), root)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    frame = pl.read_parquet(data_path)
    frame.with_columns(pl.concat_list("frame_index")).write_parquet(data_path)

    with pytest.raises(
        AdapterRefusal, match=r"'frame_index' is stored as List\(Int64\)"
    ):
        list(LeRobotV3Adapter().episodes(UPath(root)))


def test_a_video_path_templated_per_episode_is_refused(tmp_path):
    """Verify a `video_path` naming a field v3 never fills is refused by name."""

    root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = (
        "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4"
    )
    info_path.write_text(json.dumps(info))

    with pytest.raises(AdapterRefusal, match=r"video_path .* uses \{episode_index\}"):
        list(LeRobotV3Adapter().episodes(UPath(root)))


def test_episode_index_files_that_disagree_on_an_unread_column_still_read(tmp_path):
    """Verify a stats column typed differently across episode index files is ignored."""

    root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), root)
    index_dir = root / "meta" / "episodes" / "chunk-000"
    index = pl.read_parquet(index_dir / "file-000.parquet")
    first, second = index.head(1), index.tail(1)
    first.with_columns(
        pl.Series("stats/frame_index/min", [[0]], dtype=pl.List(pl.Int64))
    ).write_parquet(index_dir / "file-000.parquet")
    second.with_columns(
        pl.Series("stats/frame_index/min", [[0.0]], dtype=pl.List(pl.Float64))
    ).write_parquet(index_dir / "file-001.parquet")

    episodes = list(LeRobotV3Adapter().episodes(UPath(root)))

    assert [e.id for e in episodes] == ["episode_000000", "episode_000001"]


def test_episode_index_files_with_columns_in_different_orders_still_read(tmp_path):
    """Verify index files with the same columns in another order are read by name."""

    def checks(edit_second):
        # Both copies share one root, since the checks hold absolute paths.
        root = _split_episode_index(tmp_path / "root", edit_second)
        found = [
            (e.id, e.schema_check, e.layout_check, e.stats_check)
            for e in LeRobotV3Adapter().episodes(root)
        ]
        shutil.rmtree(tmp_path / "root")
        return found

    in_order = checks(lambda second: second)
    reordered = checks(lambda second: second.select(reversed(second.columns)))

    assert [row[0] for row in reordered] == ["episode_000000", "episode_000001"]
    assert reordered == in_order


def test_an_episode_index_file_missing_a_column_is_refused(tmp_path):
    """Verify an episode index file lacking a column the first file has is refused."""

    root = _split_episode_index(tmp_path / "root", lambda second: second.drop("tasks"))

    with pytest.raises(
        AdapterRefusal,
        match=r"file-001\.parquet lacks tasks, which .*file-000\.parquet has$",
    ):
        list(LeRobotV3Adapter().episodes(root))


def test_an_episode_index_file_with_an_extra_column_is_refused(tmp_path):
    """Verify an episode index file holding a column the first file lacks is refused."""

    root = _split_episode_index(
        tmp_path / "root",
        lambda second: second.with_columns(pl.lit(1).alias("data/extra")),
    )

    with pytest.raises(
        AdapterRefusal,
        match=r"file-001\.parquet has data/extra, which .*file-000\.parquet lacks$",
    ):
        list(LeRobotV3Adapter().episodes(root))


def test_an_episode_index_file_typed_differently_is_refused(tmp_path):
    """Verify an episode index file storing a column as another type is refused."""

    root = _split_episode_index(
        tmp_path / "root",
        lambda second: second.with_columns(pl.col("length").cast(pl.Float64)),
    )

    with pytest.raises(
        AdapterRefusal,
        match=(
            r"file-001\.parquet stores length as Float64"
            r" where .*file-000\.parquet stores Int64$"
        ),
    ):
        list(LeRobotV3Adapter().episodes(root))


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Float32, True),
        (pl.Int64, True),
        (pl.Boolean, True),
        (pl.List(pl.Float32), True),
        (pl.Array(pl.Float32, 6), True),
        (pl.List(pl.List(pl.Int16)), True),
        pytest.param(_ARRAY2D, True, marks=_POLARS_2),
        (pl.List(pl.Null), False),
        (pl.Struct({"role": pl.String}), False),
        (pl.String, False),
        (pl.List(pl.Struct({"role": pl.String, "content": pl.String})), False),
        (pl.Datetime(), False),
        (pl.Null, False),
    ],
)
def test_holds_numbers(dtype, expected):
    """Verify a column holds numbers when its innermost type is numeric or Boolean."""

    assert holds_numbers(dtype) is expected


def _with_features(root, features, columns):
    """Copy the fixture to `root` with extra features declared and stored.

    Parameters
    ----------
    root : Path
        Where to copy the fixture.
    features : dict[str, dict]
        The `info.json` entries to add, by feature key.
    columns : Callable[[int], list[pl.Series]]
        Builds the columns to add, given the data parquet's row count.

    Returns
    -------
    UPath
        The copied dataset root.
    """

    shutil.copytree(str(LEROBOT_FIXTURE), root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].update(features)
    info_path.write_text(json.dumps(info))
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    frame = pl.read_parquet(data_path)
    frame.with_columns(columns(frame.height)).write_parquet(data_path)
    return UPath(root)


_LANGUAGE_FEATURES = {
    "language_persistent": {"dtype": "language", "shape": [1], "names": None},
    "language_events": {"dtype": "language", "shape": [1], "names": None},
}


def _language_columns(height):
    """Build the two language columns of a LeRobot dataset with annotations.

    Parameters
    ----------
    height : int
        The number of rows to build.

    Returns
    -------
    list[pl.Series]
        `language_persistent` as one role/content item per row, and
        `language_events` as an empty list per row.
    """

    return [
        pl.Series(
            "language_persistent",
            [[{"role": "user", "content": "hang the keys"}]] * height,
            dtype=pl.List(pl.Struct({"role": pl.String, "content": pl.String})),
        ),
        pl.Series("language_events", [[]] * height, dtype=pl.List(pl.Null)),
    ]


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_language_features_are_skipped_and_the_rest_reads(tmp_path, tier):
    """Verify language features are dropped and every numeric feature still reads."""

    root = _with_features(
        tmp_path / "lerobot_v3_tiny", _LANGUAGE_FEATURES, _language_columns
    )

    with use_tier(tier):
        plain = list(LeRobotV3Adapter().episodes(LEROBOT_FIXTURE))
        episodes = list(LeRobotV3Adapter().episodes(root))

    assert len(episodes) == 2
    for episode, expected in zip(episodes, plain, strict=True):
        assert {s.source_field for s in episode.streams} == {
            s.source_field for s in expected.streams
        }


def test_a_column_that_holds_no_numbers_is_skipped_whatever_info_declares(tmp_path):
    """Verify the parquet type decides, even when `info.json` declares a number."""

    root = _with_features(
        tmp_path / "lerobot_v3_tiny",
        {"language_events": {"dtype": "float32", "shape": [1], "names": None}},
        lambda height: [
            pl.Series("language_events", [[]] * height, dtype=pl.List(pl.Null))
        ],
    )

    episodes = list(LeRobotV3Adapter().episodes(root))

    assert len(episodes) == 2
    for episode in episodes:
        assert all(s.source_field != "language_events" for s in episode.streams)


def test_a_text_feature_leaves_the_episodes_gradable(tmp_path):
    """Verify a scalar text feature is dropped instead of marking episodes unknown."""

    root = _with_features(
        tmp_path / "lerobot_v3_tiny",
        {"note": {"dtype": "string", "shape": [1], "names": None}},
        lambda height: [pl.Series("note", ["calm"] * height, dtype=pl.String)],
    )

    report = grade(root)
    episodes = list(LeRobotV3Adapter().episodes(root))

    assert (
        report.eligibility_counts is not None
        and report.eligibility_counts.pass_count == 2
    )
    for episode in episodes:
        assert all(s.source_field != "note" for s in episode.streams)


def test_a_boolean_feature_is_still_read(tmp_path):
    """Verify a Boolean feature still reads as a stream."""

    root = _with_features(
        tmp_path / "lerobot_v3_tiny",
        {"gripper.closed": {"dtype": "bool", "shape": [1], "names": None}},
        lambda height: [pl.Series("gripper.closed", [True] * height, dtype=pl.Boolean)],
    )

    episodes = list(LeRobotV3Adapter().episodes(root))

    assert len(episodes) == 2
    for episode in episodes:
        assert any(s.source_field == "gripper.closed" for s in episode.streams)


@_POLARS_2
def test_a_2d_feature_is_declared_at_the_metadata_tier(tmp_path):
    """Verify a 2-D extension column declares one channel per value of its grid."""

    root = _with_features(
        tmp_path / "lerobot_v3_tiny",
        {"observation.tactile": {"dtype": "int16", "shape": [2, 2], "names": None}},
        lambda height: [
            pl.Series(
                "observation.tactile",
                [[[i, i + 1], [i + 2, i + 3]] for i in range(height)],
                dtype=pl.List(pl.List(pl.Int16)),
            ).ext.to(_ARRAY2D)
        ],
    )

    with use_tier(ExecutionTier.METADATA):
        episodes = list(LeRobotV3Adapter().episodes(root))

    assert len(episodes) == 2
    for episode in episodes:
        tactile = next(
            s for s in episode.streams if s.source_field == "observation.tactile"
        )
        assert len(tactile.channels) == 4


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_every_fixture_episode_agrees_with_its_declared_features(tier):
    """Verify each episode carries a schema check of its own data file, all clean."""

    with use_tier(tier):
        episodes = list(LeRobotV3Adapter().episodes(LEROBOT_FIXTURE))

    chunk = LEROBOT_FIXTURE / "data" / "chunk-000" / "file-000.parquet"
    assert len(episodes) == 2
    for episode in episodes:
        assert episode.schema_check is not None
        assert episode.schema_check.mismatches == []
        assert "observation.images.up" not in episode.schema_check.checked
        assert str(episode.schema_check.source_path) == str(chunk)


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_a_declared_feature_with_no_column_blocks_every_episode(tmp_path, tier):
    """Verify a missing declared column blocks each episode, naming column and file."""

    root = _with_features(tmp_path / "lerobot_v3_tiny", _MISSING_FEATURE, lambda _: [])

    plain = grade(LEROBOT_FIXTURE, tier=tier)
    report = grade(root, tier=tier)

    assert len(report.episodes) == 2
    for episode, expected in zip(report.episodes, plain.episodes, strict=True):
        assert episode.metrics[_MISMATCH_METRIC].value == 1.0
        assert episode.score.score == expected.score.score
    findings = [f for f in report.findings if f.metric_id == _MISMATCH_FINDING]
    assert len(findings) == 2
    for finding in findings:
        assert finding.consequence == Consequence.BLOCK
        assert finding.evidence["path"].endswith("data/chunk-000/file-000.parquet")
        assert [m["column"] for m in finding.evidence["mismatches"]] == [
            "observation.effort"
        ]
        assert finding.evidence["mismatches"][0]["problem"] == "missing"
    assert (
        report.eligibility_counts is not None
        and report.eligibility_counts.pass_count == 0
    )


def test_language_features_stored_as_lists_raise_no_mismatch(tmp_path):
    """Verify language columns, including empty `List(Null)` ones, keep decisions."""

    root = _with_features(
        tmp_path / "lerobot_v3_tiny", _LANGUAGE_FEATURES, _language_columns
    )

    plain = grade(LEROBOT_FIXTURE)
    report = grade(root)

    assert not [f for f in report.findings if f.metric_id == _MISMATCH_FINDING]
    assert all(e.metrics[_MISMATCH_METRIC].value == 0.0 for e in report.episodes)
    assert (
        report.eligibility_counts is not None and plain.eligibility_counts is not None
    )
    assert report.eligibility_counts.pass_count == plain.eligibility_counts.pass_count


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_g1_episode_pointed_at_the_wrong_file_is_blocked(tmp_path, tier):
    """Verify an episode whose metadata names a file without its rows is blocked."""

    root = _layout_dataset(tmp_path / "g1", _G1_META, _G1_DATA)

    report = grade(root, tier=tier)

    findings = _episode_findings(report, _POINTER_FINDING)
    assert [_episode_number(e) for e in findings] == [396]
    [finding] = next(iter(findings.values()))
    assert finding.consequence == Consequence.BLOCK
    assert finding.evidence["pointed_path"].endswith("data/chunk-000/file-000.parquet")
    [holding] = finding.evidence["holding_paths"]
    assert holding.endswith("data/chunk-000/file-001.parquet")
    assert "KeyError" in finding.evidence["effect"]
    blocked = {
        _episode_number(e.id)
        for e in report.episodes
        if e.eligibility is not None
        and e.eligibility.status == EligibilityStatus.BLOCKED
    }
    assert 396 in blocked


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_g1_episodes_with_a_wrong_index_range_are_blocked(tmp_path, tier):
    """Verify each episode whose declared range disagrees with the data is blocked."""

    root = _layout_dataset(tmp_path / "g1", _G1_META, _G1_DATA)

    report = grade(root, tier=tier)

    findings = _episode_findings(report, _RANGE_FINDING)
    assert sorted(_episode_number(e) for e in findings) == [397, 398, 399]
    for episode_findings in findings.values():
        [finding] = episode_findings
        assert finding.consequence == Consequence.BLOCK
        assert finding.value == 2
        assert [m["field"] for m in finding.evidence["mismatches"]] == [
            "dataset_from_index",
            "dataset_to_index",
        ]
    [ep_397] = next(f for e, f in findings.items() if _episode_number(e) == 397)
    assert ep_397.evidence["mismatches"][0] == {
        "field": "dataset_from_index",
        "declared": 605,
        "found": 336051,
    }
    values = {
        _episode_number(e.id): e.metrics[_RANGE_METRIC].value for e in report.episodes
    }
    assert values == {394: 0, 395: 0, 396: 0, 397: 2, 398: 2, 399: 2}


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_libero_stale_pointers_are_blocked(tmp_path, tier):
    """Verify episodes pointed at a file that no longer holds them are blocked."""

    root = _layout_dataset(tmp_path / "libero", _STALE_LIBERO_META, _LIBERO_DATA)

    report = grade(root, tier=tier)

    findings = _episode_findings(report, _POINTER_FINDING)
    assert sorted(_episode_number(e) for e in findings) == [3, 4, 5, 6, 7]
    assert all(
        f.consequence == Consequence.BLOCK for fs in findings.values() for f in fs
    )


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_clean_libero_layout_raises_nothing(tmp_path, tier):
    """Verify metadata that points each episode at its own file raises nothing."""

    root = _layout_dataset(tmp_path / "libero", _CLEAN_LIBERO_META, _LIBERO_DATA)

    report = grade(root, tier=tier)

    assert not _episode_findings(report, _POINTER_FINDING)
    assert not _episode_findings(report, _RANGE_FINDING)
    assert len(report.episodes) == 8
    for episode in report.episodes:
        assert episode.metrics[_POINTER_METRIC].value == 0.0
        assert episode.metrics[_RANGE_METRIC].value == 0.0


@pytest.mark.parametrize(
    ("meta_rows", "calls"), [(_CLEAN_LIBERO_META, 0), (_STALE_LIBERO_META, 1)]
)
def test_the_scan_runs_only_on_a_miss(tmp_path, monkeypatch, meta_rows, calls):
    """Verify the data files are scanned once, and only when an episode is missing."""

    root = _layout_dataset(tmp_path / "libero", meta_rows, _LIBERO_DATA)
    counted = []
    original = v3.read_episode_rows

    def counting(*args, **kwargs):
        counted.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(v3, "read_episode_rows", counting)

    episodes = list(LeRobotV3Adapter().episodes(root))

    assert len(episodes) == 8
    assert len(counted) == calls


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_an_episode_pointed_at_a_missing_file_is_blocked(tmp_path, tier):
    """Verify an episode whose metadata names a file that does not exist is blocked."""

    root = _layout_dataset(tmp_path / "missing", _MISSING_FILE_META, _MISSING_FILE_DATA)

    report = grade(root, tier=tier)

    assert len(report.episodes) == 5
    assert report.inventory is not None
    assert report.inventory.loaded == 5
    assert report.inventory.unresolved == 0
    assert report.inventory.refused_sources == []
    assert report.inventory.complete is True
    assert report.unresolved == []
    findings = _episode_findings(report, _POINTER_FINDING)
    assert sorted(_episode_number(e) for e in findings) == [231, 232]
    for episode_findings in findings.values():
        [finding] = episode_findings
        assert finding.consequence == Consequence.BLOCK
        assert finding.evidence["pointed_path"].endswith(
            "data/chunk-000/file-001.parquet"
        )
        assert finding.evidence["pointed_path_exists"] is False
        [holding] = finding.evidence["holding_paths"]
        assert holding.endswith("data/chunk-000/file-000.parquet")
        assert "effect" not in finding.evidence
    assert {231, 232} <= _blocked(report)
    values = {
        _episode_number(e.id): e.metrics[_POINTER_METRIC].value for e in report.episodes
    }
    assert values == {229: 0.0, 230: 0.0, 231: 1.0, 232: 1.0, 700: 0.0}


def test_a_missing_file_is_checked_once(tmp_path, monkeypatch):
    """Verify a missing data file is checked once and the data files scanned once."""

    root = _layout_dataset(tmp_path / "missing", _MISSING_FILE_META, _MISSING_FILE_DATA)
    checked = []
    scanned = []
    original_exists = type(root).exists
    original_scan = v3.read_episode_rows

    def counting_exists(self, *args, **kwargs):
        if str(self).endswith("file-001.parquet"):
            checked.append(self)
        return original_exists(self, *args, **kwargs)

    def counting_scan(*args, **kwargs):
        scanned.append(args)
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(type(root), "exists", counting_exists)
    monkeypatch.setattr(v3, "read_episode_rows", counting_scan)

    episodes = list(LeRobotV3Adapter().episodes(root))

    assert len(episodes) == 5
    assert len(checked) == 1
    assert len(scanned) == 1
    missing = {
        _episode_number(e.id): e
        for e in episodes
        if _episode_number(e.id) in (231, 232)
    }
    for episode in missing.values():
        assert episode.streams == []
        assert episode.schema_check is None
        assert episode.layout_check is not None
        assert episode.layout_check.pointed_exists is False
    assert all(
        e.schema_check is not None for e in episodes if e not in missing.values()
    )


def _blocked(report):
    """List the episode numbers a report blocks.

    Parameters
    ----------
    report : Report
        The graded report.

    Returns
    -------
    set[int]
        The blocked episodes' indices.
    """

    return {
        _episode_number(e.id)
        for e in report.episodes
        if e.eligibility is not None
        and e.eligibility.status == EligibilityStatus.BLOCKED
    }


def _count_warnings(report):
    """List the messages of a report's `episode_count_mismatch` warnings.

    Parameters
    ----------
    report : Report
        The graded report.

    Returns
    -------
    list[str]
        The warnings' messages.
    """

    return [
        w.message for w in report.dataset_warnings if w.code == "episode_count_mismatch"
    ]


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_an_episode_missing_from_the_metadata_is_blocked(tmp_path, tier):
    """Verify an episode with rows in a named file and no metadata row is blocked."""

    root = _layout_dataset(tmp_path / "libero", _CLEAN_LIBERO_META[:7], _LIBERO_DATA)

    report = grade(root, tier=tier)

    assert len(report.episodes) == 8
    findings = _episode_findings(report, _ORPHAN_FINDING)
    assert [_episode_number(e) for e in findings] == [7]
    [finding] = next(iter(findings.values()))
    assert finding.consequence == Consequence.BLOCK
    [holding] = finding.evidence["holding_paths"]
    assert holding.endswith("data/chunk-000/file-002.parquet")
    assert 7 in _blocked(report)
    pointer = _episode_findings(report, _POINTER_FINDING)
    assert not [e for e in pointer if _episode_number(e) == 7]


def test_a_data_only_episode_is_found_after_a_full_scan(tmp_path):
    """Verify the full scan a pointer miss runs also finds a data-only episode."""

    meta = [*_CLEAN_LIBERO_META[:5], (5, 2, 1406, 1646, 240), _CLEAN_LIBERO_META[6]]
    root = _layout_dataset(tmp_path / "libero", meta, _LIBERO_DATA)

    episodes = list(LeRobotV3Adapter().episodes(root))

    [orphan] = [
        e
        for e in episodes
        if e.layout_check is not None and e.layout_check.pointed_path is None
    ]
    assert _episode_number(orphan.id) == 7
    assert orphan.layout_check is not None
    [holding] = orphan.layout_check.holding_paths
    assert str(holding).endswith("data/chunk-000/file-002.parquet")


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_an_episode_in_an_unnamed_data_file_is_found(tmp_path, tier):
    """Verify episodes in a data file no metadata row names are reported and blocked."""

    root = _layout_dataset(tmp_path / "libero", _CLEAN_LIBERO_META[:6], _LIBERO_DATA)

    report = grade(root, tier=tier)

    findings = _episode_findings(report, _ORPHAN_FINDING)
    assert sorted(_episode_number(e) for e in findings) == [6, 7]
    for episode_findings in findings.values():
        [finding] = episode_findings
        [holding] = finding.evidence["holding_paths"]
        assert holding.endswith("data/chunk-000/file-002.parquet")
    assert {6, 7} <= _blocked(report)


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_an_episode_with_rows_in_no_file_is_blocked(tmp_path, tier):
    """Verify a metadata row for an episode with no rows anywhere is blocked."""

    meta = [*_CLEAN_LIBERO_META, (8, 2, 2197, 2300, 103)]
    root = _layout_dataset(tmp_path / "libero", meta, _LIBERO_DATA)

    report = grade(root, tier=tier)

    findings = _episode_findings(report, _POINTER_FINDING)
    assert [_episode_number(e) for e in findings] == [8]
    [finding] = next(iter(findings.values()))
    assert finding.consequence == Consequence.BLOCK
    assert finding.evidence["pointed_path"].endswith("data/chunk-000/file-002.parquet")
    assert finding.evidence["holding_paths"] == []
    assert 8 in _blocked(report)
    [message] = _count_warnings(report)
    assert "metadata lists 9" in message
    assert "data holds 8" in message


@pytest.mark.parametrize("declared", [0, 9])
@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_a_declared_count_that_disagrees_is_a_warning(tmp_path, tier, declared):
    """Verify a declared episode count that disagrees gives the counts found."""

    root = _layout_dataset(tmp_path / "libero", _CLEAN_LIBERO_META, _LIBERO_DATA)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = declared
    info_path.write_text(json.dumps(info))

    report = grade(root, tier=tier)

    [message] = _count_warnings(report)
    assert f"declares {declared} episodes" in message
    assert "metadata lists 8" in message
    assert "data holds 8" in message


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_a_matching_declared_count_raises_no_warning(tmp_path, tier):
    """Verify a clean layout with a matching count raises no inventory finding."""

    root = _layout_dataset(tmp_path / "libero", _CLEAN_LIBERO_META, _LIBERO_DATA)

    report = grade(root, tier=tier)

    assert not _count_warnings(report)
    assert not _episode_findings(report, _ORPHAN_FINDING)


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_a_sampled_read_yields_no_data_only_episode(tmp_path, tier):
    """Verify a sampled read stops at the sample, with no data-only episode."""

    root = _layout_dataset(tmp_path / "libero", _CLEAN_LIBERO_META[:7], _LIBERO_DATA)

    with use_tier(tier):
        episodes = list(LeRobotV3Adapter().episodes(root, sample=3))

    assert len(episodes) == 3


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_every_fixture_episode_is_where_its_metadata_says(tier):
    """Verify each fixture episode's pointer and index range agree with its data."""

    with use_tier(tier):
        episodes = list(LeRobotV3Adapter().episodes(LEROBOT_FIXTURE))

    assert len(episodes) == 2
    for episode in episodes:
        assert episode.layout_check is not None
        assert episode.layout_check.pointer_matches
        assert episode.layout_check.range_mismatches == []


def _card_copy(tmp_path, card):
    """Copy the fixture, giving it `card` as `README.md` when it is not `None`."""

    copy_root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), copy_root)
    if card is not None:
        (copy_root / "README.md").write_text(card)
    return UPath(copy_root)


@pytest.mark.parametrize(
    ("card", "expected"),
    [
        ("---\ntags:\n- LeRobot\n- robosuite\n---\n# Card\n", "robosuite"),
        ("---\ntags: [LeRobot, Isaac-Sim]\n---\n", "isaac-sim"),
        ("---\ntags: [LeRobot, Mujoco_Menagerie]\n---\n", "mujoco_menagerie"),
    ],
)
def test_a_dataset_card_tag_names_the_simulator(tmp_path, card, expected):
    """A tag starting with a simulator name, in any case, is read from the card."""

    info = LeRobotV3Adapter().describe(_card_copy(tmp_path, card))

    assert info.simulator == expected
    assert info.episode_count == 2


@pytest.mark.parametrize(
    "card",
    [
        None,
        "# A card without front matter\n",
        "---\ntags: [unclosed\n---\n",
        "---\ntags: robosuite\n---\n",
        "---\n- robosuite\n---\n",
        "---\ntags: [LeRobot, 3, null]\n---\n",
        "---\ntags: [LeRobot, robosuite]\n",
        "---\ntags: [LeRobot]\ntags: [robosuite]\n---\n",
    ],
    ids=[
        "no_card",
        "no_front_matter",
        "invalid_yaml",
        "tags_not_a_list",
        "front_matter_not_a_mapping",
        "no_simulator_tag",
        "front_matter_never_closed",
        "duplicate_tags_key",
    ],
)
def test_a_card_that_names_no_simulator_leaves_the_declared_facts(tmp_path, card):
    """A missing or malformed card gives no simulator and costs nothing else."""

    info = LeRobotV3Adapter().describe(_card_copy(tmp_path, card))

    assert info.simulator is None
    assert info.episode_count == 2
    assert info.robot_type == "so100_follower"


def test_an_undecodable_card_leaves_the_declared_facts(tmp_path):
    """A card that is not UTF-8 text gives no simulator."""

    root = _card_copy(tmp_path, None)
    (root / "README.md").write_bytes(b"---\n\xff\xfe\n---\n")

    info = LeRobotV3Adapter().describe(root)

    assert info.simulator is None
    assert info.episode_count == 2


def _with_episode_stats(root, edit=None):
    """Copy the v3 fixture to `root`, adding `stats/*` columns computed from its data.

    Parameters
    ----------
    root : Path
        Where to write the copy.
    edit : Callable[[dict[int, dict]], None] or None
        Called with each episode's stats, by `episode_index`, before they are written.
    """

    shutil.copytree(str(LEROBOT_FIXTURE), root)
    data = pl.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta = pl.read_parquet(meta_path)
    per_episode = {
        index: lerobot_stats(
            data.filter(pl.col("episode_index") == index), data.columns
        )
        for index in meta["episode_index"].to_list()
    }
    if edit is not None:
        edit(per_episode)
    columns = [
        pl.Series(
            f"stats/{feature}/{stat}",
            [per_episode[index][feature][stat] for index in meta["episode_index"]],
        )
        for feature in data.columns
        for stat in ("min", "max", "mean", "std", "count")
    ]
    meta.with_columns(columns).write_parquet(meta_path)


def test_v3_stats_written_from_the_data_match_it(tmp_path):
    """Verify every v3 episode's stats columns are compared, with no mismatch."""

    root = tmp_path / "lerobot_v3_tiny"
    _with_episode_stats(root)

    episodes = list(LeRobotV3Adapter().episodes(UPath(root)))

    assert len(episodes) == 2
    for episode in episodes:
        assert episode.stats_check is not None
        assert episode.stats_check.mismatches == []
        assert "observation.state" in episode.stats_check.checked
        assert UPath(episode.stats_check.source_path).name == "file-000.parquet"


def test_an_edited_v3_stored_mean_is_reported_on_its_episode_only(tmp_path):
    """Verify a stale mean in one episode's stats columns flags that episode alone."""

    def stale_mean(per_episode):
        per_episode[1]["observation.state"]["mean"][0] += 10.0

    root = tmp_path / "lerobot_v3_tiny"
    _with_episode_stats(root, stale_mean)

    first, second = LeRobotV3Adapter().episodes(UPath(root))

    assert first.stats_check is not None and first.stats_check.mismatches == []
    assert second.stats_check is not None
    assert [(m.feature, m.stat) for m in second.stats_check.mismatches] == [
        ("observation.state", "mean")
    ]


def test_v3_stats_are_not_read_at_the_metadata_tier(tmp_path):
    """Verify a metadata-tier read leaves `stats_check` unset."""

    root = tmp_path / "lerobot_v3_tiny"
    _with_episode_stats(root)

    with use_tier(ExecutionTier.METADATA):
        episodes = list(LeRobotV3Adapter().episodes(UPath(root)))

    assert [e.stats_check for e in episodes] == [None, None]


def test_v3_without_stats_columns_has_no_stats_check():
    """Verify an episode index with no `stats/*` columns leaves `stats_check` unset."""

    assert [e.stats_check for e in _episodes()] == [None, None]


def test_a_repeated_channel_name_is_indexed_and_warned(tmp_path):
    """Verify a v3 feature that repeats `gripper` indexes each member and warns."""

    root = UPath(with_repeated_gripper(LEROBOT_FIXTURE, tmp_path / "repeated"))

    # The metadata tier grades no channel, so the names are read off the adapter.
    with use_tier(ExecutionTier.METADATA):
        episodes = list(LeRobotV3Adapter().episodes(root))
    report = grade(root, tier=ExecutionTier.METADATA)

    assert len(episodes) == 2
    for episode in episodes:
        [state] = [s for s in episode.streams if s.source_field == "observation.state"]
        assert [(c.name, c.declared_name) for c in state.channels[4:]] == [
            ("observation.state[4]", "gripper"),
            ("observation.state[5]", "gripper"),
        ]
    assert len(report.episodes) == 2
    assert not report.unresolved
    [warning] = [
        w for w in report.dataset_warnings if w.code == "duplicate_channel_label"
    ]
    assert 'observation.state uses "gripper" at indices 4 and 5' in warning.message


def test_macos_sidecar_files_are_ignored(tmp_path):
    """Verify `._*` AppleDouble files beside the index and data parquets are skipped."""

    root = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(str(LEROBOT_FIXTURE), root)
    apple_double = b"\x00\x05\x16\x07" + b"\x00" * 60
    for directory in ("meta/episodes/chunk-000", "data/chunk-000"):
        (root / directory / "._file-000.parquet").write_bytes(apple_double)

    plain = _episodes()
    episodes = list(LeRobotV3Adapter().episodes(UPath(root)))

    assert [e.id for e in episodes] == [e.id for e in plain]
    assert [len(e.streams) for e in episodes] == [len(e.streams) for e in plain]


def test_a_repeated_episode_index_loads_each_row_under_its_own_id(tmp_path):
    """Verify each row sharing an `episode_index` gets an id naming its row."""

    root = _with_repeated_index(tmp_path / "lerobot_v3_tiny")

    episodes = list(LeRobotV3Adapter().episodes(root))

    assert [e.id for e in episodes] == [
        "row000000_episode_000000",
        "row000002_episode_000000",
        "row000001_episode_000001",
        "row000003_episode_000001",
    ]
    assert [e.tasks for e in episodes] == [
        ["pick up the cube"],
        ["fold the towel"],
        ["pick up the cube"],
        ["fold the towel"],
    ]
    for episode in episodes:
        assert episode.layout_check is not None
        assert episode.layout_check.index_rows == 2


def test_a_repeated_episode_index_grades_and_blocks_every_repeated_row(tmp_path):
    """Verify grading a repeated index finishes and blocks each repeated row."""

    root = _with_repeated_index(tmp_path / "lerobot_v3_tiny")

    report = grade(root)

    findings = _episode_findings(report, "schema.episode_index_repeated")
    assert sorted(findings) == sorted(e.id for e in report.episodes)
    assert len(findings) == 4
    for episode_findings in findings.values():
        [finding] = episode_findings
        assert finding.consequence == Consequence.BLOCK
        assert finding.evidence["index_rows"] == 2
    assert all(
        e.eligibility is not None and e.eligibility.status == EligibilityStatus.BLOCKED
        for e in report.episodes
    )


def test_a_repeated_episode_index_compares_no_stored_stats(tmp_path):
    """Verify repeated rows skip the stats check, since stats are keyed by index."""

    root = tmp_path / "lerobot_v3_tiny"
    _with_episode_stats(root)
    index_dir = root / "meta" / "episodes" / "chunk-000"
    shutil.copy(index_dir / "file-000.parquet", index_dir / "file-001.parquet")

    episodes = list(LeRobotV3Adapter().episodes(UPath(root)))

    assert len(episodes) == 4
    assert [e.stats_check for e in episodes] == [None] * 4


def test_a_unique_episode_index_keeps_its_plain_id(tmp_path):
    """Verify only the repeated index takes row ids; the unique one keeps its own."""

    root = _with_repeated_index(tmp_path / "lerobot_v3_tiny", kept=[1])

    report = grade(root)

    assert [e.id.rsplit("::", 1)[1] for e in report.episodes] == [
        "episode_000000",
        "row000001_episode_000001",
        "row000002_episode_000001",
    ]
    plain = report.episodes[0]
    assert plain.metrics["episode_index_repeated"].value == 0.0
    findings = _episode_findings(report, "schema.episode_index_repeated")
    assert sorted(findings) == sorted(e.id for e in report.episodes[1:])


_STATIC = "observation.images.rgb_static"
_GRIPPER = "observation.images.rgb_gripper"


def test_a_window_past_its_file_names_the_pointer_mismatch(tmp_path):
    """Verify a camera window past its file's end blocks, naming the window and file."""

    report = grade(_taco_like_dataset(tmp_path / "taco", fixed_pointer=False))

    result = _camera_presence(report)[(2142, _STATIC)]
    assert result.value == 0.0
    evidence = result.evidence
    assert evidence["reason"] == "Video window outside referenced file"
    assert evidence["path"].endswith(f"videos/{_STATIC}/chunk-000/file-000.mp4")
    assert evidence["declared_window_s"] == [4.4, 8.8]
    assert evidence["file_bounds_s"] == [0.0, pytest.approx(4.4)]
    assert (evidence["observed_frames"], evidence["expected_frames"]) == (0, 66)
    [finding] = [
        f
        for e, found in _episode_findings(
            report, "vision.camera_stream_present"
        ).items()
        if _episode_number(e) == 2142
        for f in found
        if f.source_field == _STATIC
    ]
    assert finding.consequence == Consequence.BLOCK
    assert 2142 in _blocked(report)


def test_neighbours_and_the_other_camera_of_a_misplaced_window_pass(tmp_path):
    """Verify the episodes either side, and the misplaced one's other camera, pass."""

    report = grade(_taco_like_dataset(tmp_path / "taco", fixed_pointer=False))

    presence = _camera_presence(report)
    for key in [
        (2141, _STATIC),
        (2141, _GRIPPER),
        (2143, _STATIC),
        (2143, _GRIPPER),
        (2142, _GRIPPER),
    ]:
        assert presence[key].value == 1.0, key
        assert "file_bounds_s" not in presence[key].evidence


def test_a_corrected_pointer_reads_every_frame(tmp_path):
    """Verify pointing the episode at the file holding its frames clears the finding."""

    report = grade(_taco_like_dataset(tmp_path / "taco", fixed_pointer=True))

    assert _camera_presence(report)[(2142, _STATIC)].value == 1.0
    [episode] = [e for e in report.episodes if _episode_number(e.id) == 2142]
    [stream] = [
        s
        for s in episode.streams
        if s.metrics["camera_stream_present"].evidence.get("camera") == _STATIC
    ]
    counted = stream.metrics["frame_count_vs_timebase"].evidence
    assert (counted["present"], counted["implied"]) == (66, 66)
    assert not [
        f
        for f in report.findings
        if _episode_number(f.episode_id) == 2142
        and f.source_field == _STATIC
        and f.consequence == Consequence.BLOCK
    ]


def test_a_nested_list_feature_reads_one_channel_per_value():
    """Verify a `[1, 3]` row stored as a list of lists splits into three channels."""

    frame = pl.DataFrame({"f": _EEF_POS}, schema={"f": pl.List(pl.List(pl.Float32))})
    channels = [Channel(name=f"f_{i}", source_index=i) for i in range(3)]

    payload = _vector_stream(frame, channels)

    assert payload.columns == ["f_0", "f_1", "f_2"]
    assert all(dtype == pl.Float32 for dtype in payload.dtypes)
    assert payload["f_0"].to_list() == pytest.approx([0.490671, 0.490398])
    assert payload["f_2"].to_list() == pytest.approx([0.587828, 0.586533])


def test_a_grid_feature_reads_its_values_row_major():
    """Verify a `[2, 2]` grid's channels read its cells row by row."""

    frame = pl.DataFrame(
        {"f": [[[1, 2], [3, 4]]]}, schema={"f": pl.Array(pl.Int16, (2, 2))}
    )
    channels = [Channel(name=f"f_{i}", source_index=i) for i in range(4)]

    payload = _vector_stream(frame, channels)

    assert payload.row(0) == (1, 2, 3, 4)


def test_a_short_row_nulls_every_channel_of_its_frame():
    """Verify a list row holding 2 values of 3 reads as null, the others as stored."""

    frame = pl.DataFrame(
        {"f": [[1.0, 2.0, 3.0], [4.0, 5.0], [7.0, 8.0, 9.0]]},
        schema={"f": pl.List(pl.Float32)},
    )
    channels = [Channel(name=f"f_{i}", source_index=i) for i in range(3)]

    payload = _vector_stream(frame, channels)

    assert payload.rows() == [(1.0, 2.0, 3.0), (None, None, None), (7.0, 8.0, 9.0)]


def test_a_multi_dimensional_feature_grades_one_numeric_channel_per_value(tmp_path):
    """Verify `[1, 3]` and `[1, 4]` features grade as 3 and 4 numeric channels."""

    root = _with_features(tmp_path / "lerobot_v3_tiny", _EEF_FEATURES, _eef_columns)

    report = grade(root)

    for episode in report.episodes:
        by_field = {s.source_field: s for s in episode.streams}
        position = by_field["obs.state.manipulator.eef_pos"]
        quaternion = by_field["obs.state.manipulator.eef_quat"]
        assert [g.channel.name for g in position.channels] == [
            f"obs.state.manipulator.eef_pos_{i}" for i in range(3)
        ]
        assert len(quaternion.channels) == 4
        for graded in position.channels + quaternion.channels:
            missing = graded.metrics["missing_pct"]
            assert missing.status != MetricStatus.NOT_APPLICABLE
            assert missing.value == 0.0
        assert [
            g.channel.binding and g.channel.binding.representation
            for g in quaternion.channels
        ] == [Representation.UNKNOWN] * 4
        rotation = quaternion.metrics["rotation_snr_db"]
        assert rotation.status == MetricStatus.NOT_APPLICABLE
        assert rotation.evidence["reason"] == "stream holds no single unit quaternion"
        assert episode.metrics["element_count_mismatch_count"].value == 0.0
    assert report.coverage is not None
    numeric = next(r for r in report.coverage.capabilities if r.key == "numeric")
    assert numeric.unavailable == 0
    assert numeric.computed == len(report.episodes)


def test_a_short_row_is_reported_and_every_episode_still_loads(tmp_path):
    """Verify a short row is reported per frame and the dataset still loads."""

    root = _short_row_dataset(tmp_path / "lerobot_v3_tiny")
    frame = pl.read_parquet(str(root / "data" / "chunk-000" / "file-000.parquet"))
    short = frame.row(1, named=True)
    short_episode, short_frame = short["episode_index"], short["frame_index"]

    episodes = list(LeRobotV3Adapter().episodes(root))

    assert len(episodes) == 2
    checks = {e.id: e.element_count_check for e in episodes}
    assert all(c is not None and c.checked for c in checks.values())
    [mismatch] = [m for c in checks.values() if c is not None for m in c.mismatches]
    assert (mismatch.field, mismatch.expected, mismatch.found, mismatch.frame) == (
        "ee.position",
        3,
        2,
        short_frame,
    )
    assert mismatch.message == f"ee.position frame {short_frame}: expected 3, found 2"
    assert checks[f"episode_{short_episode:06d}"].mismatches == [mismatch]


def test_a_short_row_blocks_its_episode(tmp_path):
    """Verify the episode holding the short row is blocked by the contract route."""

    root = _short_row_dataset(tmp_path / "lerobot_v3_tiny")

    report = grade(root)

    findings = _episode_findings(report, "schema.element_count_mismatch_count")
    assert len(findings) == 1
    [finding] = next(iter(findings.values()))
    assert finding.consequence == Consequence.BLOCK
    [field] = finding.evidence["fields"]
    assert field["frames"][0]["message"].endswith("expected 3, found 2")


def test_the_element_count_check_is_skipped_at_the_metadata_tier(tmp_path):
    """Verify a metadata-tier read leaves `element_count_check` unset."""

    root = _short_row_dataset(tmp_path / "lerobot_v3_tiny")

    with use_tier(ExecutionTier.METADATA):
        episodes = list(LeRobotV3Adapter().episodes(root))

    assert [e.element_count_check for e in episodes] == [None, None]
