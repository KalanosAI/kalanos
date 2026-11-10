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
from kalanos.analysis.models.domain import (
    Channel,
    Clock,
    FramePayload,
    Kind,
    MappingSource,
    TimestampDtype,
)
from kalanos.analysis.models.eligibility import Consequence, EligibilityStatus
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import AnalysedEpisode
from kalanos.analysis.reporting.assemble import assemble_report
from kalanos.api import grade
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


def _layout_dataset(root, meta_rows, data_files):
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

    Returns
    -------
    UPath
        The dataset root.
    """

    info = json.loads((LEROBOT_FIXTURE / "meta" / "info.json").read_text())
    del info["features"]["observation.images.up"]
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
    """Verify a 2-D extension column reads as one channel per row of its grid."""

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
        assert len(tactile.channels) == 2


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
    """Verify a 2-D extension column counts as numbers in the raw parquet schema."""

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
        assert len(tactile.channels) == 2


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
def test_every_fixture_episode_is_where_its_metadata_says(tier):
    """Verify each fixture episode's pointer and index range agree with its data."""

    with use_tier(tier):
        episodes = list(LeRobotV3Adapter().episodes(LEROBOT_FIXTURE))

    assert len(episodes) == 2
    for episode in episodes:
        assert episode.layout_check is not None
        assert episode.layout_check.pointer_matches
        assert episode.layout_check.range_mismatches == []
