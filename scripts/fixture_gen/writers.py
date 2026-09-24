"""Turn a spec's samples into bytes on disk."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import json
import logging
from pathlib import Path

# External
import av
import h5py
import numpy as np
import polars as pl
from mcap.writer import CompressionType
from mcap_ros2.writer import Writer as McapRos2Writer

# Internal
from fixture_gen.corpus import (
    FixtureSpec,
    Hdf5TinySpec,
    LeRobotV2TinySpec,
    LeRobotV3TinySpec,
    McapTinySpec,
)
from fixture_gen.models import (
    CsvOutput,
    DelimitedTextOutput,
    Fields,
    JsonLinesOutput,
    KeyedJsonOutput,
    Sample,
    StaticSpec,
    TimeSeriesSpec,
)
from fixture_gen.sampling import build_samples


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

# An arbitrary but realistic epoch, chosen so the fixture actually exercises
# the origin-subtraction the adapter does: at this magnitude float64's ULP is
# already ~256 ns, enough to blur the fixture's own half-millisecond offset
# if the adapter ever converted a raw epoch to seconds without subtracting it.
_MCAP_EPOCH_NS = 1_700_000_000_000_000_000
_NS_PER_S = 1_000_000_000


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _write_csv(
    spec: TimeSeriesSpec, samples: list[Sample], output: CsvOutput, path: Path
) -> None:
    """Write a CSV fixture: one row per sample, sorted by (time, entity).

    Parameters
    ----------
    spec : TimeSeriesSpec
        The fixture spec, for its channel names and ordering.
    samples : list of Sample
        The samples to write.
    output : CsvOutput
        The time and entity column names.
    path : Path
        Destination file.
    """

    channel_names = [c.name for g in spec.entities[0].groups for c in g.channels]
    has_entities = any(s.entity_id is not None for s in samples)
    fieldnames = [output.time_field]
    if has_entities:
        fieldnames.append(output.entity_field)
    fieldnames += channel_names

    # Group by time than entity ID
    ordered = sorted(samples, key=lambda s: (s.t, s.entity_id or ""))

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in ordered:
            row: dict[str, object] = {output.time_field: s.t}
            if has_entities:
                row[output.entity_field] = s.entity_id
            row.update(s.fields)
            writer.writerow(row)

    logger.info("Wrote %d rows to %s", len(ordered), path)


def _write_jsonlines(
    samples: list[Sample], output: JsonLinesOutput, path: Path
) -> None:
    """Write a JSONL fixture: one JSON object per sample, in grid order.

    Parameters
    ----------
    samples : list of Sample
        The samples to write.
    output : JsonLinesOutput
        The time field name.
    path : Path
        Destination file.
    """

    with path.open("w") as f:
        for s in samples:
            record = {output.time_field: s.t, **s.fields}
            f.write(json.dumps(record) + "\n")

    logger.info("Wrote %d records to %s", len(samples), path)


def _write_delimited_text(
    samples: list[Sample], output: DelimitedTextOutput, path: Path
) -> None:
    """Write a whitespace-delimited fixture with a comment-line header.

    Parameters
    ----------
    samples : list of Sample
        The samples to write.
    output : DelimitedTextOutput
        The comment line to write before the data.
    path : Path
        Destination file.
    """

    with path.open("w") as f:
        f.write(output.comment_header + "\n")
        for s in samples:
            columns = [str(s.t), *(str(v) for v in s.fields.values())]
            f.write(" ".join(columns) + "\n")

    logger.info("Wrote %d rows to %s", len(samples), path)


def _write_keyed_json(
    samples: list[Sample], output: KeyedJsonOutput, path: Path
) -> None:
    """Write a fixture keyed by a per-record template, nested by entity.

    Parameters
    ----------
    samples : list of Sample
        The samples to write. Samples sharing a `t` become one record.
    output : KeyedJsonOutput
        The key format string (`{minutes}`/`{seconds}` fields, derived
        from `t`) and the JSON indent.
    path : Path
        Destination file.

    Raises
    ------
    ValueError
        If a sample has no entity id. Keyed JSON nests every record
        under an entity key, so there is nowhere to put an anonymous
        entity — that is a spec error, not a case to fall back from.
    """

    by_time: dict[int, dict[str, Fields]] = {}
    for s in samples:
        if s.entity_id is None:
            raise ValueError(f"{output.kind} output needs an entity id on every sample")
        # Every keyed-JSON grid used today is integer-valued;
        # the {minutes:02d}/{seconds:02d} template would raise on a float t.
        if not isinstance(s.t, int):
            raise ValueError(f"{output.kind} output needs an integer-valued time grid")
        by_time.setdefault(s.t, {})[s.entity_id] = s.fields

    data = {
        output.key_template.format(minutes=t // 60, seconds=t % 60): entities
        for t, entities in by_time.items()
    }
    path.write_text(json.dumps(data, indent=output.indent) + "\n")

    logger.info("Wrote %d timestamps to %s", len(data), path)


def write_fixture(spec: FixtureSpec) -> None:
    """Dispatch a fixture spec to the writer for its kind.

    Parameters
    ----------
    spec : FixtureSpec
        The fixture spec to write.
    """

    match spec:
        case TimeSeriesSpec():
            _write_timeseries(spec, build_samples(spec))
        case StaticSpec():
            _write_static(spec)
        case LeRobotV3TinySpec():
            _write_lerobot_v3(spec)
        case LeRobotV2TinySpec():
            _write_lerobot_v2(spec)
        case Hdf5TinySpec():
            _write_hdf5_tiny(spec)
        case McapTinySpec():
            _write_mcap_tiny(spec)


def _write_timeseries(spec: TimeSeriesSpec, samples: list[Sample]) -> None:
    """Dispatch a time-series spec's samples to the writer for its output kind.

    Parameters
    ----------
    spec : TimeSeriesSpec
        The fixture spec, for its filename and output config.
    samples : list of Sample
        The samples built by `build_samples`.
    """

    path = FIXTURES_DIR / spec.filename
    match spec.output:
        case CsvOutput() as output:
            _write_csv(spec, samples, output, path)
        case JsonLinesOutput() as output:
            _write_jsonlines(samples, output, path)
        case DelimitedTextOutput() as output:
            _write_delimited_text(samples, output, path)
        case KeyedJsonOutput() as output:
            _write_keyed_json(samples, output, path)


def _write_lerobot_video(
    path: Path, *, frame_count: int, fps: int, size: tuple[int, int]
) -> None:
    """Encode `frame_count` solid-colour frames as an h264 mp4.

    Parameters
    ----------
    path : Path
        Destination file.
    frame_count : int
        How many frames to encode.
    fps : int
        The stream's frame rate.
    size : tuple[int, int]
        `(width, height)` in pixels.
    """

    width, height = size
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(frame_count):
        level = round(255 * i / max(frame_count - 1, 1))
        array = np.full((height, width, 3), level, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    logger.info("Wrote %d frames to %s", frame_count, path)


def _write_lerobot_v3(spec: LeRobotV3TinySpec) -> None:
    """Write a tiny LeRobot v3.0 dataset tree under `tests/fixtures/<dirname>/`.

    Both episodes share one data parquet chunk and one mp4 per the spec's single camera,
    so the split-by-episode-index and several-files-per-episode pathologies are
    exercised by the fixture itself rather than only by a unit test.

    Parameters
    ----------
    spec : LeRobotV3TinySpec
        The dataset to build.
    """

    root = FIXTURES_DIR / spec.dirname
    fps = spec.fps
    total_frames = sum(spec.episode_lengths)

    features = {
        spec.state_feature.key: {
            "dtype": "float32",
            "shape": [len(spec.state_feature.names)],
            "names": list(spec.state_feature.names),
        },
        spec.velocity_feature.key: {
            "dtype": "float32",
            "shape": [len(spec.velocity_feature.names)],
            "names": list(spec.velocity_feature.names),
        },
        spec.video_key: {
            "dtype": "video",
            "shape": [spec.video_size[1], spec.video_size[0], 3],
            "info": {"video.fps": fps},
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": spec.robot_type,
        "total_episodes": len(spec.episode_lengths),
        "total_frames": total_frames,
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": features,
    }
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    # Both episodes' rows land in one data parquet chunk.
    rows = []
    global_index = 0
    for episode_index, length in enumerate(spec.episode_lengths):
        for frame_index in range(length):
            rows.append(
                {
                    spec.state_feature.key: [
                        float(frame_index + i)
                        for i in range(len(spec.state_feature.names))
                    ],
                    spec.velocity_feature.key: [
                        float(-(frame_index + i))
                        for i in range(len(spec.velocity_feature.names))
                    ],
                    "timestamp": frame_index / fps,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1
    # Hugging Face writes a fixed-width vector as Array, so state mirrors real data;
    # velocity stays List so the fixture carries both shapes.
    data_frame = pl.DataFrame(rows).select(
        pl.col(spec.state_feature.key).cast(
            pl.Array(pl.Float32, len(spec.state_feature.names))
        ),
        pl.col(spec.velocity_feature.key).cast(pl.List(pl.Float32)),
        pl.col("timestamp").cast(pl.Float32),
        pl.col("frame_index").cast(pl.Int64),
        pl.col("episode_index").cast(pl.Int64),
        pl.col("index").cast(pl.Int64),
        pl.col("task_index").cast(pl.Int64),
    )
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_frame.write_parquet(data_dir / "file-000.parquet")
    logger.info("Wrote %d rows to %s", data_frame.height, data_dir / "file-000.parquet")

    video_dir = root / "videos" / spec.video_key / "chunk-000"
    video_dir.mkdir(parents=True, exist_ok=True)
    _write_lerobot_video(
        video_dir / "file-000.mp4",
        frame_count=total_frames,
        fps=fps,
        size=spec.video_size,
    )

    # One row per episode, both pointing at the chunk written above.
    episode_rows = []
    cursor = 0
    for episode_index, length in enumerate(spec.episode_lengths):
        episode_rows.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
                f"videos/{spec.video_key}/chunk_index": 0,
                f"videos/{spec.video_key}/file_index": 0,
                f"videos/{spec.video_key}/from_timestamp": cursor / fps,
                f"videos/{spec.video_key}/to_timestamp": (cursor + length) / fps,
                "tasks": ["pick up the cube"],
                "length": length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        cursor += length
    episode_index_frame = pl.DataFrame(episode_rows)
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_index_frame.write_parquet(episodes_dir / "file-000.parquet")
    logger.info("Wrote %d episode(s) to %s", episode_index_frame.height, root)


def _write_lerobot_v2(spec: LeRobotV2TinySpec) -> None:
    """Write a tiny LeRobot v2.0/v2.1 dataset tree under `tests/fixtures/<dirname>/`.

    One parquet and, when the spec declares a camera, one mp4 per episode,
    so the per-episode-file and chunk-rollover pathologies are exercised by
    the fixture itself rather than only by a unit test.

    Parameters
    ----------
    spec : LeRobotV2TinySpec
        The dataset to build.
    """

    root = FIXTURES_DIR / spec.dirname
    fps = spec.fps
    total_episodes = len(spec.episode_lengths)
    total_frames = sum(spec.episode_lengths)
    task = "pick up the cube"

    features = {
        spec.state_feature.key: {
            "dtype": "float32",
            "shape": [len(spec.state_feature.names)],
            "names": list(spec.state_feature.names),
        },
        spec.velocity_feature.key: {
            "dtype": "float32",
            "shape": [len(spec.velocity_feature.names)],
            "names": list(spec.velocity_feature.names),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if spec.video_key is not None:
        features[spec.video_key] = {
            "dtype": "video",
            "shape": [spec.video_size[1], spec.video_size[0], 3],
            "info": {"video.fps": fps},
        }

    chunks = {
        episode_index // spec.chunks_size for episode_index in range(total_episodes)
    }
    info = {
        "codebase_version": spec.codebase_version,
        "robot_type": spec.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes if spec.video_key is not None else 0,
        "total_chunks": len(chunks),
        "chunks_size": spec.chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": features,
    }
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    global_index = 0
    for episode_index, length in enumerate(spec.episode_lengths):
        episode_chunk = episode_index // spec.chunks_size
        rows = []
        for frame_index in range(length):
            rows.append(
                {
                    spec.state_feature.key: [
                        float(frame_index + i)
                        for i in range(len(spec.state_feature.names))
                    ],
                    spec.velocity_feature.key: [
                        float(-(frame_index + i))
                        for i in range(len(spec.velocity_feature.names))
                    ],
                    "timestamp": frame_index / fps,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                }
            )
            global_index += 1
        # Hugging Face writes a fixed-width vector as Array, so state mirrors real data;
        # velocity stays List so the fixture carries both shapes.
        data_frame = pl.DataFrame(rows).select(
            pl.col(spec.state_feature.key).cast(
                pl.Array(pl.Float32, len(spec.state_feature.names))
            ),
            pl.col(spec.velocity_feature.key).cast(pl.List(pl.Float32)),
            pl.col("timestamp").cast(pl.Float32),
            pl.col("frame_index").cast(pl.Int64),
            pl.col("episode_index").cast(pl.Int64),
            pl.col("index").cast(pl.Int64),
            pl.col("task_index").cast(pl.Int64),
        )
        data_dir = root / "data" / f"chunk-{episode_chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"episode_{episode_index:06d}.parquet"
        data_frame.write_parquet(data_path)
        logger.info("Wrote %d rows to %s", data_frame.height, data_path)

        if spec.video_key is not None:
            video_dir = root / "videos" / f"chunk-{episode_chunk:03d}" / spec.video_key
            video_dir.mkdir(parents=True, exist_ok=True)
            _write_lerobot_video(
                video_dir / f"episode_{episode_index:06d}.mp4",
                frame_count=length,
                fps=fps,
                size=spec.video_size,
            )

    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n"
    )

    episodes_lines = [
        json.dumps({"episode_index": i, "tasks": [task], "length": length})
        for i, length in enumerate(spec.episode_lengths)
    ]
    (meta_dir / "episodes.jsonl").write_text("\n".join(episodes_lines) + "\n")

    # The adapter reads neither stats file;
    # their presence is what distinguishes a v2.0 tree from a v2.1 tree on disk.
    if spec.codebase_version == "v2.0":
        (meta_dir / "stats.json").write_text(json.dumps({}) + "\n")
    else:
        stats_lines = [
            json.dumps({"episode_index": i, "stats": {}}) for i in range(total_episodes)
        ]
        (meta_dir / "episodes_stats.jsonl").write_text("\n".join(stats_lines) + "\n")

    logger.info("Wrote %d episode(s) to %s", total_episodes, root)


def _write_hdf5_tiny(spec: Hdf5TinySpec) -> None:
    """Write a tiny two-episode HDF5 file under `tests/fixtures/<filename>`.

    Two sibling episode groups of different lengths under one container group,
    so the sibling-group split and the declared-rate pathologies
    are exercised by the fixture itself rather than only by a unit test.

    Parameters
    ----------
    spec : Hdf5TinySpec
        The dataset to build.
    """

    path = FIXTURES_DIR / spec.filename
    # h5py's default superblock costs about 5 KB of object-header padding
    # here, which puts the file over tests/test_fixture_corpus.py's 10 KB cap.
    with h5py.File(str(path), "w", libver="latest") as store:
        container = store.create_group(spec.container)
        container.attrs["fps"] = spec.fps
        for name, length in zip(spec.episode_names, spec.episode_lengths, strict=True):
            episode = container.create_group(name)
            episode.create_dataset(
                "actions",
                data=np.array(
                    [
                        [float(t + i) for i in range(spec.action_dim)]
                        for t in range(length)
                    ],
                    dtype=np.float32,
                ),
            )
            episode.create_dataset(
                "rewards",
                data=np.array([float(t) for t in range(length)], dtype=np.float32),
            )
            subgroup = episode.create_group(spec.subgroup)
            for feature in spec.subgroup_features:
                subgroup.create_dataset(
                    feature.key,
                    data=np.array(
                        [
                            [float(t - i) for i in range(feature.width)]
                            for t in range(length)
                        ],
                        dtype=np.float32,
                    ),
                )
    logger.info("Wrote %d episode(s) to %s", len(spec.episode_lengths), path)


def _write_mcap_tiny(spec: McapTinySpec) -> None:
    """Write a tiny three-topic MCAP file under `tests/fixtures/<filename>`.

    One topic carries a header and is offset from its own `log_time`, one
    carries none, and one is a blob-payload topic with no channels — so the
    capture-vs-log split and the blob-kind path are exercised by the fixture
    itself rather than only by a unit test.

    Parameters
    ----------
    spec : McapTinySpec
        The dataset to build.
    """

    path = FIXTURES_DIR / spec.filename
    with path.open("wb") as handle:
        writer = McapRos2Writer(handle, compression=CompressionType.NONE)
        joint_schema = writer.register_msgdef(
            "sensor_msgs/msg/JointState", spec.joint.msgdef
        )
        twist_schema = writer.register_msgdef(
            "geometry_msgs/msg/Twist", spec.twist.msgdef
        )
        image_schema = writer.register_msgdef(
            "sensor_msgs/msg/Image", spec.image.msgdef
        )

        joint_interval_ns = round(_NS_PER_S / spec.joint.rate_hz)
        for i in range(spec.joint.count):
            stamp_ns = _MCAP_EPOCH_NS + i * joint_interval_ns
            writer.write_message(
                topic=spec.joint.topic,
                schema=joint_schema,
                message={
                    "header": {
                        "stamp": {
                            "sec": stamp_ns // _NS_PER_S,
                            "nanosec": stamp_ns % _NS_PER_S,
                        },
                        "frame_id": "base_link",
                    },
                    "name": ["shoulder", "elbow"],
                    "position": [float(i), float(i) + 1.0],
                    "velocity": [0.1 * i, 0.1 * i + 0.1],
                    "effort": [0.01 * i, 0.01 * i + 0.01],
                },
                log_time=stamp_ns + spec.joint_log_offset_ns,
                publish_time=stamp_ns + spec.joint_log_offset_ns,
            )

        twist_interval_ns = round(_NS_PER_S / spec.twist.rate_hz)
        for i in range(spec.twist.count):
            log_time = _MCAP_EPOCH_NS + i * twist_interval_ns
            writer.write_message(
                topic=spec.twist.topic,
                schema=twist_schema,
                message={
                    "linear": {"x": 0.1 * i, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.05 * i},
                },
                log_time=log_time,
                publish_time=log_time,
            )

        width, height = spec.image_size
        image_interval_ns = round(_NS_PER_S / spec.image.rate_hz)
        frame_bytes = width * height * 3
        for i in range(spec.image.count):
            stamp_ns = _MCAP_EPOCH_NS + i * image_interval_ns
            writer.write_message(
                topic=spec.image.topic,
                schema=image_schema,
                message={
                    "header": {
                        "stamp": {
                            "sec": stamp_ns // _NS_PER_S,
                            "nanosec": stamp_ns % _NS_PER_S,
                        },
                        "frame_id": "wrist_cam",
                    },
                    "height": height,
                    "width": width,
                    "encoding": "rgb8",
                    "is_bigendian": 0,
                    "step": width * 3,
                    "data": bytes(range(frame_bytes)),
                },
                log_time=stamp_ns,
                publish_time=stamp_ns,
            )

        writer.finish()
    logger.info("Wrote 3 topic(s) to %s", path)


def _write_static(spec: StaticSpec) -> None:
    """Write a fixture with no time axis: its payload, as flat JSON.

    Parameters
    ----------
    spec : StaticSpec
        The fixture spec, for its filename and payload.
    """

    path = FIXTURES_DIR / spec.filename
    path.write_text(json.dumps(spec.payload, indent=spec.output.indent) + "\n")

    logger.info("Wrote static fixture to %s", path)
