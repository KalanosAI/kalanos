"""LeRobot v2.0/v2.1: one parquet and one mp4 per episode, no episode table.

v2.0 and v2.1 share one on-disk layout: one parquet per episode under
`data/chunk-{episode_chunk:03d}/`, one mp4 per episode per camera under
`videos/chunk-{episode_chunk:03d}/{video_key}/`, no episode table, boundaries
carried by the filenames themselves. The two versions differ only in which
stats file `meta/` holds (`meta/stats.json` for v2.0, `meta/episodes_stats.jsonl`
for v2.1). v2.1's per-episode stats are compared with each episode's data
when numeric payloads are read; v2.0's dataset-wide `meta/stats.json` is not read.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from collections.abc import Iterator
from typing import Any

# External
import polars as pl
from upath import UPath

from kalanos.analysis.adapters.lerobot.common import (
    CONFIDENCE,
    TIME_COLUMN,
    LeRobotAdapter,
    check_element_counts,
    check_path_template,
    check_scalar_index,
    check_schema,
    check_stats,
    declared_format,
    declared_series_streams,
    describe_from_info,
    episode_clock,
    feature_plan,
    read_info,
    sampling_is_regular,
    series_streams,
    split_of,
    storage_columns,
    timestamp_dtype_of,
    with_episode_clock,
)
from kalanos.analysis.adapters.prefetch import prefetching
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.adapters.video import (
    VideoPayload,
    close_video_files,
    current_remote_reads,
    decoder_available,
)

# Internal
from kalanos.analysis.execution import reads_numeric_payloads
from kalanos.analysis.hosts import LOCAL_PROTOCOLS
from kalanos.analysis.inference.tasks import as_task_list, dataset_tasks
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.domain import (
    Clock,
    Episode,
    Kind,
    MappingSource,
    Stream,
    TimestampDtype,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

_SUPPORTED_CODEBASE_VERSIONS = frozenset({"v2.0", "v2.1"})

# LeRobot's own default when info.json omits it.
_DEFAULT_CHUNKS_SIZE = 1000


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _video_path(
    info: dict[str, Any],
    dataset_root: UPath,
    video_key: str,
    episode_index: int,
    episode_chunk: int,
) -> UPath:
    """Return the mp4 `info["video_path"]` names for one camera of one episode.

    Parameters
    ----------
    info : dict
        The parsed `info.json`.
    dataset_root : UPath
        The dataset root, `video_path` is resolved relative to it.
    video_key : str
        The video feature key.
    episode_index : int
        The episode's own index.
    episode_chunk : int
        The chunk `episode_index` falls into.

    Returns
    -------
    UPath
        The video file.
    """

    return dataset_root / info["video_path"].format(
        video_key=video_key,
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )


def _video_stream(
    info: dict[str, Any],
    video_key: str,
    taxonomy_type: str,
    timestamps: pl.Series,
    dataset_root: UPath,
    *,
    mapping_source: MappingSource | None,
    episode_index: int,
    episode_chunk: int,
    fps: float,
    clock: Clock,
    timestamp_dtype: TimestampDtype,
    is_regular: bool,
) -> Stream:
    """Build one video Stream for `video_key`, its payload lazy and undecoded.

    One mp4 holds exactly one episode in this layout, so the payload spans
    the whole file rather than a from/to slice of a shared one,
    and its packet count reads to the file's end, past the timestamps.

    Parameters
    ----------
    info : dict
        The parsed `info.json`,
        for `video_path`'s template and the feature's declared format.
    video_key : str
        The video feature key.
    taxonomy_type : str
        The taxonomy type `feature_plan` resolved for this key.
    timestamps : pl.Series
        The episode's own timestamps, shared across every one of its streams.
    dataset_root : UPath
        The dataset root, `video_path` is resolved relative to it.
    mapping_source : MappingSource or None
        How `taxonomy_type` was decided.
    episode_index : int
        This episode's own index.
    episode_chunk : int
        The chunk `episode_index` falls into.
    fps : float
        The declared frame rate, for the payload's `end_s`.
    is_regular : bool
        Whether the episode's own sampling classified as regular.

    Returns
    -------
    Stream
        One channel-less video stream; `VideoPayload.fetch` decodes nothing
        until a metric actually calls it.
    """

    video_path = _video_path(
        info, dataset_root, video_key, episode_index, episode_chunk
    )
    return Stream(
        taxonomy_type=taxonomy_type,
        kind=Kind.VIDEO,
        timestamps=timestamps,
        payload=VideoPayload(
            path=video_path,
            frame_count=len(timestamps),
            start_s=0.0,
            end_s=len(timestamps) / fps,
            whole_file=True,
            declared=declared_format(info["features"][video_key]),
        ),
        source_path=video_path,
        source_field=video_key,
        mapping_source=mapping_source,
        clock=clock,
        timestamp_dtype=timestamp_dtype,
        is_regular=is_regular,
        channels=[],
    )


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


def _read_episode_tasks(path: UPath) -> dict[int, list[str]] | None:
    """Read each episode's task instructions from `meta/episodes.jsonl`.

    Parameters
    ----------
    path : UPath
        The dataset root.

    Returns
    -------
    dict[int, list[str]] or None
        Each episode's instructions keyed by `episode_index`; `None` when the file
        is absent, names no `tasks` at all, or holds no non-blank instruction for
        any episode — a dataset that carries none is not missing any.
    """

    episodes_path = path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return None
    per_episode: dict[int, list[str]] = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "tasks" in record:
                per_episode[int(record["episode_index"])] = as_task_list(
                    record["tasks"]
                )
    return dataset_tasks(per_episode) if per_episode else None


def _read_episode_stats(path: UPath) -> dict[int, dict[str, Any]] | None:
    """Read each episode's per-feature statistics from `meta/episodes_stats.jsonl`.

    Parameters
    ----------
    path : UPath
        The dataset root.

    Returns
    -------
    dict[int, dict[str, Any]] or None
        Each episode's `stats` object keyed by `episode_index`;
        `None` when the file is absent, as in a v2.0 dataset.

    Raises
    ------
    AdapterRefusal
        If a line is not a JSON object with an `episode_index` and a `stats` object.
    """

    stats_path = path / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return None
    per_episode: dict[int, dict[str, Any]] = {}
    with stats_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            stats = record.get("stats") if isinstance(record, dict) else None
            if not isinstance(stats, dict) or "episode_index" not in record:
                raise AdapterRefusal(
                    path,
                    f"{stats_path} line {number} is not an object "
                    "with an episode_index and a stats object",
                )
            per_episode[int(record["episode_index"])] = stats
    return per_episode


@adapter
class LeRobotV2Adapter(LeRobotAdapter):
    """Reads a LeRobot v2.0/v2.1 dataset directory: one parquet and mp4 per episode."""

    name = "lerobot_v2"

    def detect(self, path: UPath) -> float:
        """Bid on a directory whose `meta/info.json` declares codebase_version v2.x.

        Never raises: a v3.0 tree, or anything else that fails to parse,
        bids zero and is reported unclaimed rather than misread.

        Parameters
        ----------
        path : UPath
            The path to consider.

        Returns
        -------
        float
            `CONFIDENCE`, or `0.0`.
        """

        try:
            if not path.is_dir():
                return 0.0
            info = read_info(path)
            if info.get("codebase_version") not in _SUPPORTED_CODEBASE_VERSIONS:
                return 0.0
            return CONFIDENCE
        except Exception:
            return 0.0

    def describe(self, path: UPath) -> DatasetInfo:
        """Report `meta/info.json`'s own facts, without reading any episode.

        Parameters
        ----------
        path : UPath
            The dataset root.

        Returns
        -------
        DatasetInfo
            `episode_count`, `nominal_rate_hz` and `robot_type`, as declared,
            and the `simulator` the dataset card's tags name.

        Raises
        ------
        AdapterRefusal
            If `meta/info.json` cannot be read.
        """

        return describe_from_info(self.name, path, read_info(path))

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield one Episode per file pair, resolved from `total_episodes` alone.

        Parameters
        ----------
        path : UPath
            The dataset root.
        sample : int or None
            When given, stop after this many episodes.

        Yields
        ------
        Episode
            One recording, its streams drawn from its own data parquet
            and, for each declared camera, its own mp4.

        Raises
        ------
        AdapterRefusal
            If anything about the input could not be read.
        """

        if sample is not None and sample <= 0:
            return

        try:
            info = read_info(path)
            check_path_template(
                path, info, "data_path", frozenset({"episode_chunk", "episode_index"})
            )
            check_path_template(
                path,
                info,
                "video_path",
                frozenset({"video_key", "episode_chunk", "episode_index"}),
            )
            plan = feature_plan(info, self._resolve_dictionary())
            total = int(info["total_episodes"])
            fps = float(info["fps"])
            chunks_size = int(info.get("chunks_size", _DEFAULT_CHUNKS_SIZE))
            task_map = _read_episode_tasks(path)
            stats_map = _read_episode_stats(path) if reads_numeric_payloads() else None
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(path, f"could not read {path}: {exc}") from exc

        episode_count = total if sample is None else min(total, sample)

        # Each episode's video files, fetched whole ahead of the reader when remote.
        # A metadata tier, or a run without the decoder, opens no video,
        # so prefetching there would only download bytes nobody reads.
        cap = (
            current_remote_reads().prefetch_bytes
            if path.protocol not in LOCAL_PROTOCOLS
            and plan.video
            and reads_numeric_payloads()
            and decoder_available()
            else 0
        )
        video_paths = (
            [
                [
                    _video_path(info, path, video_key, index, index // chunks_size)
                    for video_key in plan.video
                ]
                for index in range(episode_count)
            ]
            if cap
            else []
        )

        with prefetching(video_paths, cap) as prefetcher:
            for episode_index in range(episode_count):
                try:
                    episode_chunk = episode_index // chunks_size
                    data_path = path / info["data_path"].format(
                        episode_chunk=episode_chunk, episode_index=episode_index
                    )
                    # One parquet holds exactly one episode, unlike v3's shared chunks:
                    # no episode_index filter and no chunk cache are needed here.
                    materialise = reads_numeric_payloads()
                    with data_path.open("rb") as handle:
                        parquet_schema = pl.read_parquet_schema(handle)
                    check_scalar_index(path, parquet_schema)
                    schema_check = check_schema(info, parquet_schema, data_path)
                    with data_path.open("rb") as handle:
                        # Metadata tier: index and clock columns only; the
                        # numeric feature vectors never leave the file.
                        frame = storage_columns(
                            pl.read_parquet(
                                handle,
                                columns=None
                                if materialise
                                else [
                                    c
                                    for c in (
                                        "episode_index",
                                        TIME_COLUMN,
                                        "frame_index",
                                    )
                                    if c in parquet_schema
                                ],
                            )
                        )

                    episode_frame = frame
                    stored_stats = (
                        stats_map.get(episode_index) if stats_map is not None else None
                    )
                    stats_check = (
                        check_stats(
                            info,
                            stored_stats,
                            episode_frame,
                            path / "meta" / "episodes_stats.jsonl",
                        )
                        if materialise and stored_stats is not None
                        else None
                    )
                    element_count_check = (
                        check_element_counts(plan, episode_frame, data_path)
                        if materialise
                        else None
                    )
                    timestamp_dtype = timestamp_dtype_of(episode_frame[TIME_COLUMN])
                    clock = episode_clock(episode_frame, fps, timestamp_dtype)
                    timestamps = episode_frame[TIME_COLUMN].cast(pl.Float64)
                    is_regular = sampling_is_regular(timestamps)

                    streams = (
                        series_streams(
                            episode_frame,
                            plan,
                            timestamps,
                            data_path,
                            path=path,
                            episode_label=episode_index,
                            clock=clock,
                            timestamp_dtype=timestamp_dtype,
                            is_regular=is_regular,
                        )
                        if materialise
                        else declared_series_streams(
                            plan,
                            parquet_schema,
                            timestamps,
                            data_path,
                            clock=clock,
                            timestamp_dtype=timestamp_dtype,
                            is_regular=is_regular,
                        )
                    )
                    streams.extend(
                        _video_stream(
                            info,
                            video_key,
                            taxonomy_type,
                            timestamps,
                            path,
                            mapping_source=mapping_source,
                            episode_index=episode_index,
                            episode_chunk=episode_chunk,
                            fps=fps,
                            clock=clock,
                            timestamp_dtype=timestamp_dtype,
                            is_regular=is_regular,
                        )
                        for video_key, (
                            taxonomy_type,
                            mapping_source,
                        ) in plan.video.items()
                    )
                except AdapterRefusal:
                    raise
                except Exception as exc:
                    raise AdapterRefusal(
                        path, f"episode {episode_index}: {exc}"
                    ) from exc

                streams = with_episode_clock(
                    streams,
                    episode_frame,
                    clock=clock,
                    domain=f"{path}#episode:{episode_index}",
                )
                yield Episode(
                    id=f"episode_{episode_index:06d}",
                    streams=streams,
                    tasks=None if task_map is None else task_map.get(episode_index, []),
                    split=split_of(info, episode_index),
                    schema_check=schema_check,
                    stats_check=stats_check,
                    element_count_check=element_count_check,
                )
                if prefetcher is not None:
                    # The pipeline grades an episode before asking for the next,
                    # so its files are done with. Pooled containers are closed first,
                    # since one left idle would keep the released bytes alive.
                    close_video_files(video_paths[episode_index])
                    prefetcher.release(video_paths[episode_index])
