"""LeRobot v3.0: one directory, several parquet chunks, an mp4 per camera per chunk.

One episode spans a data parquet and one mp4 per camera;
one parquet chunk holds many episodes, split by `episode_index`.
`meta/info.json` declares the frame rate and the robot type,
and `meta/episodes/**/*.parquet` maps each episode to the files that hold it.
An episode whose rows are in the data with no row in `meta/episodes` is yielded
after the others, with no streams, so the report can block it.
The per-episode stats in the `stats/*` columns of `meta/episodes` are compared
with each episode's data when numeric payloads are read.
Each row whose `episode_index` repeats in `meta/episodes` is yielded under an id
naming its row, `row<k>_episode_<index>`, so the report can block it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import itertools
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Any

# External
import polars as pl
from upath import UPath

from kalanos.analysis.adapters.lerobot.common import (
    CONFIDENCE,
    TIME_COLUMN,
    LeRobotAdapter,
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
from kalanos.analysis.adapters.lerobot.layout import (
    aggregate_episode_rows,
    layout_check,
    read_episode_rows,
)
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.adapters.video import VideoPayload

# Internal
from kalanos.analysis.execution import reads_numeric_payloads
from kalanos.analysis.inference.tasks import as_task_list, dataset_tasks
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.domain import (
    Clock,
    Episode,
    Kind,
    LayoutCheck,
    MappingSource,
    SchemaCheck,
    Stream,
    TimestampDtype,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

_SUPPORTED_CODEBASE_VERSION = "v3.0"
# Each episode index row's 0-based position across `meta/episodes` in path order.
_INDEX_ROW = "__kalanos_index_row"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _parquet_files(root: UPath) -> list[UPath]:
    """List every parquet under `root`, at any depth, sorted, leaving hidden ones out.

    A file or directory below `root` whose name starts with `.` is left out,
    such as the `._*` AppleDouble sidecars macOS writes beside each file it copies.

    Parameters
    ----------
    root : UPath
        The directory to search.

    Returns
    -------
    list[UPath]
        Every `*.parquet` under `root` with no dot-named part below `root`, sorted.
    """

    return sorted(
        candidate
        for candidate in root.rglob("*.parquet")
        if not any(part.startswith(".") for part in candidate.relative_to(root).parts)
    )


def _check_same_columns(
    path: UPath,
    first: UPath,
    first_schema: pl.Schema,
    other: UPath,
    other_schema: pl.Schema,
) -> None:
    """Refuse an episode index file whose columns or types differ from the first file's.

    Parameters
    ----------
    path : UPath
        The dataset root, named in the refusal.
    first : UPath
        The first episode index file, relative to `path`.
    first_schema : pl.Schema
        The columns read from `first`.
    other : UPath
        A later episode index file, relative to `path`.
    other_schema : pl.Schema
        The columns read from `other`.

    Raises
    ------
    AdapterRefusal
        At the first disagreement, checked in this order:
        `other` lacks columns `first` holds,
        `other` holds columns `first` lacks,
        or a column's type differs.
    """

    lacking = ", ".join(sorted(set(first_schema) - set(other_schema)))
    if lacking:
        raise AdapterRefusal(
            path, f"episode index file {other} lacks {lacking}, which {first} has"
        )
    extra = ", ".join(sorted(set(other_schema) - set(first_schema)))
    if extra:
        raise AdapterRefusal(
            path, f"episode index file {other} has {extra}, which {first} lacks"
        )
    for column in sorted(first_schema):
        if first_schema[column] != other_schema[column]:
            raise AdapterRefusal(
                path,
                f"episode index file {other} stores {column} as "
                f"{other_schema[column]} where {first} stores {first_schema[column]}",
            )


def _read_episode_index(path: UPath) -> pl.DataFrame:
    """Read and concatenate every `meta/episodes/**/*.parquet`, sorted by episode.

    Parameters
    ----------
    path : UPath
        The dataset root.

    Returns
    -------
    pl.DataFrame
        Every episode's row, one per episode, sorted by `episode_index`
        when the index has that column; the caller checks that it does.
        Rows that share an `episode_index` keep their file-path order.
        The `_INDEX_ROW` column holds each row's 0-based position before sorting.
        Only `episode_index`, `tasks`, `dataset_from_index`, `dataset_to_index`,
        `length` and the `data/*` and `videos/*` columns are read; any other
        column has to be added here before a caller reads it.
        Columns are matched across files by name, whatever their order in each file.

    Raises
    ------
    AdapterRefusal
        If `meta/episodes/` holds no parquet, or any of them fail to read.
        Files that disagree with the first file on which of those columns they hold,
        or on a column's type, are refused, naming both files and the column.
        A v2.x tree has no such directory, so this is the seam it falls
        through on.
    """

    episodes_dir = path / "meta" / "episodes"
    try:
        parquet_paths = _parquet_files(episodes_dir)
        if not parquet_paths:
            raise AdapterRefusal(path, f"no episode index parquet under {episodes_dir}")
        frames = []
        for parquet_path in parquet_paths:
            with parquet_path.open("rb") as handle:
                names = list(pl.read_parquet_schema(handle))
            # Every other column, `stats/*` among them, is left unread: its type can
            # differ from one file to the next, which would fail the concat below.
            # The order is sorted so files that list the same columns in different
            # orders stack by name.
            columns = sorted(
                name
                for name in names
                if name
                in (
                    "episode_index",
                    "tasks",
                    "dataset_from_index",
                    "dataset_to_index",
                    "length",
                )
                or name.startswith(("data/", "videos/"))
            )
            with parquet_path.open("rb") as handle:
                frame = pl.read_parquet(handle, columns=columns)
            if frames:
                _check_same_columns(
                    path,
                    parquet_paths[0].relative_to(path),
                    frames[0].schema,
                    parquet_path.relative_to(path),
                    frame.schema,
                )
            frames.append(frame)
        index = pl.concat(frames).with_row_index(_INDEX_ROW)
    except AdapterRefusal:
        raise
    except Exception as exc:
        raise AdapterRefusal(
            path, f"could not read episode index under {episodes_dir}: {exc}"
        ) from exc
    # A missing `episode_index` is refused by name in `episodes`.
    if "episode_index" not in index.columns:
        return index
    return index.sort("episode_index", maintain_order=True)


def _read_episode_stats(
    path: UPath,
) -> dict[int, tuple[UPath, dict[str, dict[str, Any]]]]:
    """Read each episode's per-feature statistics from the `stats/*` columns.

    Each `meta/episodes/**/*.parquet` is read and converted on its own,
    since a `stats/*` column's type can differ from one file to the next.

    Parameters
    ----------
    path : UPath
        The dataset root.

    Returns
    -------
    dict[int, tuple[UPath, dict[str, dict[str, Any]]]]
        For each `episode_index`, the parquet its stats were read from and its stats
        as `{feature: {stat: value}}`, a column `stats/<feature>/<stat>` split on its
        last `/`. Empty when no file has a `stats/*` column.
    """

    per_episode: dict[int, tuple[UPath, dict[str, dict[str, Any]]]] = {}
    for parquet_path in _parquet_files(path / "meta" / "episodes"):
        with parquet_path.open("rb") as handle:
            names = list(pl.read_parquet_schema(handle))
        stats_columns = [name for name in names if name.startswith("stats/")]
        if "episode_index" not in names or not stats_columns:
            continue
        with parquet_path.open("rb") as handle:
            frame = pl.read_parquet(handle, columns=["episode_index", *stats_columns])
        for row in frame.iter_rows(named=True):
            stats: dict[str, dict[str, Any]] = {}
            for column in stats_columns:
                feature, _, stat = column.removeprefix("stats/").rpartition("/")
                if feature:
                    stats.setdefault(feature, {})[stat] = row[column]
            per_episode[int(row["episode_index"])] = (parquet_path, stats)
    return per_episode


def _data_files(path: UPath, info: dict[str, Any]) -> list[UPath]:
    """List every data parquet under the dataset's data root, sorted.

    Parameters
    ----------
    path : UPath
        The dataset root.
    info : dict
        The parsed `info.json`;
        the first part of its `data_path` template names the data root.

    Returns
    -------
    list[UPath]
        Every `*.parquet` under the data root, at any depth, sorted,
        with hidden files and directories left out.
    """

    data_root = path / PurePosixPath(info["data_path"]).parts[0]
    return _parquet_files(data_root)


def _video_stream(
    info: dict[str, Any],
    row: dict[str, Any],
    video_key: str,
    taxonomy_type: str,
    timestamps: pl.Series,
    dataset_root: UPath,
    *,
    mapping_source: MappingSource | None,
    clock: Clock,
    timestamp_dtype: TimestampDtype,
    is_regular: bool,
) -> Stream:
    """Build one video Stream for `video_key`, its payload lazy and undecoded.

    Parameters
    ----------
    info : dict
        The parsed `info.json`,
        for `video_path`'s template and the feature's declared format.
    row : dict
        This episode's own row from the episode index, for the chunk/file
        indices and the from/to timestamps `video_path` and `VideoPayload` need.
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
    is_regular : bool
        Whether the episode's own sampling classified as regular.

    Returns
    -------
    Stream
        One channel-less video stream; `VideoPayload.fetch` decodes nothing
        until a metric actually calls it.
    """

    video_path = dataset_root / info["video_path"].format(
        video_key=video_key,
        chunk_index=row[f"videos/{video_key}/chunk_index"],
        file_index=row[f"videos/{video_key}/file_index"],
    )
    return Stream(
        taxonomy_type=taxonomy_type,
        kind=Kind.VIDEO,
        timestamps=timestamps,
        payload=VideoPayload(
            path=video_path,
            frame_count=len(timestamps),
            start_s=row[f"videos/{video_key}/from_timestamp"],
            end_s=row[f"videos/{video_key}/to_timestamp"],
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


@adapter
class LeRobotV3Adapter(LeRobotAdapter):
    """Reads a LeRobot v3.0 dataset directory: many episodes, several files each."""

    name = "lerobot_v3"

    def detect(self, path: UPath) -> float:
        """Bid on a directory whose `meta/info.json` declares codebase_version v3.0.

        Never raises: a v2.x tree, or anything else that fails to parse,
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
            if info.get("codebase_version") != _SUPPORTED_CODEBASE_VERSION:
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
        """Yield one Episode per row of the episode index, then the data-only ones.

        Parameters
        ----------
        path : UPath
            The dataset root.
        sample : int or None
            When given, stop after this many episodes.
            A sampled read yields no data-only episode,
            since it cannot see the whole inventory.

        Yields
        ------
        Episode
            One recording, its streams drawn from the data parquet chunk
            its row names and, for each declared camera, its own mp4.
            When `sample` is `None`, then one Episode per `episode_index`
            found in the data and absent from the episode index, in index order:
            no streams, `tasks` `None`, and a `layout_check` whose `pointed_path`
            is `None` and whose `holding_paths` are the data files holding its rows.
            An episode whose row names a data file that does not exist has no
            streams, no `schema_check`, and a `layout_check` with `pointed_exists`
            `False`.
            A row whose `episode_index` repeats in the episode index is yielded
            with id `row<k>_episode_<index>`, k its `_INDEX_ROW`, no `stats_check`,
            and a `layout_check` whose `index_rows` is the number of such rows.

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
                path, info, "data_path", frozenset({"chunk_index", "file_index"})
            )
            check_path_template(
                path,
                info,
                "video_path",
                frozenset({"video_key", "chunk_index", "file_index"}),
            )
            episode_index = _read_episode_index(path)
            stats_map = _read_episode_stats(path) if reads_numeric_payloads() else {}
            plan = feature_plan(info, self._resolve_dictionary())
            required = {"episode_index", "data/chunk_index", "data/file_index"}
            for video_key in plan.video:
                required |= {
                    f"videos/{video_key}/chunk_index",
                    f"videos/{video_key}/file_index",
                    f"videos/{video_key}/from_timestamp",
                    f"videos/{video_key}/to_timestamp",
                }
            missing = ", ".join(sorted(required - set(episode_index.columns)))
            if missing:
                raise AdapterRefusal(
                    path, f"the episode index lacks the LeRobot v3 columns {missing}"
                )
            repeated: dict[int, int] = dict(
                episode_index.group_by("episode_index")
                .len()
                .filter(pl.col("len") > 1)
                .iter_rows()
            )
            fps = float(info["fps"]) if "fps" in info else None
            # Each row of the episode index lists its task instructions, when the
            # dataset records any; read for every episode, not just the sampled ones,
            # since whether the dataset carries instructions at all is a dataset fact.
            task_map = (
                dataset_tasks(
                    {
                        row[_INDEX_ROW]: as_task_list(row["tasks"])
                        for row in episode_index.select(_INDEX_ROW, "tasks").iter_rows(
                            named=True
                        )
                    }
                )
                if "tasks" in episode_index.columns
                else None
            )
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(path, f"could not read {path}: {exc}") from exc

        # One parquet chunk holds several episodes, so each is read once and shared.
        # The episode index is sorted by episode_index and LeRobot lays chunks out
        # contiguously in that order.
        # So only the current chunk is ever needed, and a one-entry cache is enough.
        cached_path: UPath | None = None
        cached_frame: pl.DataFrame | None = None
        cached_schema: dict[str, pl.DataType] = {}
        cached_check: SchemaCheck | None = None
        # Where every episode's rows are, across all data files. Built on the first
        # episode missing from the file its metadata names, then reused.
        located: tuple[list[UPath], pl.DataFrame] | None = None
        # The data files read so far and the episodes each holds, collected from the
        # frames read anyway, so the inventory check after the loop only has to scan
        # the files no metadata row names.
        read_paths: set[UPath] = set()
        held: dict[int, list[UPath]] = {}
        # Each named data file is checked for existence once, since on a remote
        # host every check is a request and many episodes can name the same file.
        missing_paths: set[UPath] = set()
        present_paths: set[UPath] = set()

        def locate(with_index: bool) -> tuple[list[UPath], pl.DataFrame]:
            """Find every episode's rows across all data files, scanning them once.

            Parameters
            ----------
            with_index : bool
                Whether to read the `index` column too.

            Returns
            -------
            tuple[list[UPath], pl.DataFrame]
                The data files, and the `read_episode_rows` output that indexes them.
            """

            nonlocal located
            if located is None:
                all_paths = _data_files(path, info)
                located = (
                    all_paths,
                    read_episode_rows(all_paths, with_index=with_index),
                )
            return located

        def episode_id(row: dict[str, Any]) -> str:
            """Name the episode a row of the episode index describes.

            Parameters
            ----------
            row : dict
                The row, with its `_INDEX_ROW` position.

            Returns
            -------
            str
                `episode_<index>`, or `row<k>_episode_<index>` when the row's
                `episode_index` repeats, since the index alone names several rows.
            """

            if row["episode_index"] in repeated:
                return f"row{row[_INDEX_ROW]:06d}_episode_{row['episode_index']:06d}"
            return f"episode_{row['episode_index']:06d}"

        row_iter = episode_index.iter_rows(named=True)
        if sample is not None:
            row_iter = itertools.islice(row_iter, sample)

        for row in row_iter:
            try:
                data_path = path / info["data_path"].format(
                    chunk_index=row["data/chunk_index"],
                    file_index=row["data/file_index"],
                )
                if data_path not in missing_paths | present_paths:
                    if data_path.exists():
                        present_paths.add(data_path)
                    else:
                        missing_paths.add(data_path)
                if data_path in missing_paths:
                    source_paths, all_rows = locate("index" in cached_schema)
                    holding = all_rows.filter(
                        pl.col("episode_index") == row["episode_index"]
                    )
                    missing_layout = layout_check(
                        row,
                        data_path,
                        holding,
                        source_paths,
                        pointed_exists=False,
                        index_rows=repeated.get(row["episode_index"], 1),
                    )
                else:
                    missing_layout = None
            except AdapterRefusal:
                raise
            except Exception as exc:
                raise AdapterRefusal(
                    path, f"episode {row.get('episode_index')}: {exc}"
                ) from exc

            if missing_layout is not None:
                # No file to read a frame or schema from, and the pointer finding
                # blocks the episode anyway. The cache is left alone, so a later
                # episode naming the cached file does not read it again.
                yield Episode(
                    id=episode_id(row),
                    streams=[],
                    tasks=None
                    if task_map is None
                    else task_map.get(row[_INDEX_ROW], []),
                    split=split_of(info, row["episode_index"]),
                    schema_check=None,
                    layout_check=missing_layout,
                )
                continue

            try:
                materialise = reads_numeric_payloads()
                if data_path != cached_path:
                    with data_path.open("rb") as handle:
                        # A metadata-tier run projects the index and clock
                        # columns only: the numeric feature vectors never leave
                        # the file. The schema still says which columns exist.
                        parquet_schema = pl.read_parquet_schema(handle)
                    check_scalar_index(path, parquet_schema)
                    cached_schema = dict(parquet_schema)
                    cached_check = check_schema(info, cached_schema, data_path)
                    with data_path.open("rb") as handle:
                        cached_frame = storage_columns(
                            pl.read_parquet(
                                handle,
                                columns=None
                                if materialise
                                else [
                                    c
                                    for c in (
                                        "episode_index",
                                        "index",
                                        TIME_COLUMN,
                                        "frame_index",
                                    )
                                    if c in cached_schema
                                ],
                            )
                        )
                    cached_path = data_path
                    if sample is None and data_path not in read_paths:
                        read_paths.add(data_path)
                        for idx in cached_frame["episode_index"].unique().to_list():
                            held.setdefault(idx, []).append(data_path)
                assert cached_frame is not None
                frame = cached_frame

                episode_frame = frame.filter(
                    pl.col("episode_index") == row["episode_index"]
                )
                with_index = "index" in cached_schema
                if episode_frame.height:
                    source_paths = [data_path]
                    holding = aggregate_episode_rows(
                        episode_frame, with_index=with_index
                    )
                else:
                    source_paths, all_rows = locate(with_index)
                    holding = all_rows.filter(
                        pl.col("episode_index") == row["episode_index"]
                    )
                episode_layout = layout_check(
                    row,
                    data_path,
                    holding,
                    source_paths,
                    index_rows=repeated.get(row["episode_index"], 1),
                )
                # Stats are keyed by `episode_index`, so a repeated index's rows
                # can't be told apart and none of them is compared.
                stored_stats = (
                    None
                    if row["episode_index"] in repeated
                    else stats_map.get(row["episode_index"])
                )
                stats_check = (
                    check_stats(info, stored_stats[1], episode_frame, stored_stats[0])
                    if materialise and stored_stats is not None
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
                        episode_label=row["episode_index"],
                        clock=clock,
                        timestamp_dtype=timestamp_dtype,
                        is_regular=is_regular,
                    )
                    if materialise
                    else declared_series_streams(
                        plan,
                        cached_schema,
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
                        row,
                        video_key,
                        taxonomy_type,
                        timestamps,
                        path,
                        mapping_source=mapping_source,
                        clock=clock,
                        timestamp_dtype=timestamp_dtype,
                        is_regular=is_regular,
                    )
                    for video_key, (taxonomy_type, mapping_source) in plan.video.items()
                )
            except AdapterRefusal:
                raise
            except Exception as exc:
                raise AdapterRefusal(
                    path, f"episode {row.get('episode_index')}: {exc}"
                ) from exc

            streams = with_episode_clock(
                streams,
                episode_frame,
                clock=clock,
                domain=f"{path}#episode:{row['episode_index']}",
            )
            yield Episode(
                id=episode_id(row),
                streams=streams,
                tasks=None if task_map is None else task_map.get(row[_INDEX_ROW], []),
                split=split_of(info, row["episode_index"]),
                schema_check=cached_check,
                layout_check=episode_layout,
                stats_check=stats_check,
            )

        if sample is not None:
            return
        try:
            if located is not None:
                # The miss scan already read every data file.
                located_paths, located_rows = located
                held = {}
                for source, idx in located_rows.select(
                    "source", "episode_index"
                ).iter_rows():
                    held.setdefault(idx, []).append(located_paths[source])
            else:
                orphans = [p for p in _data_files(path, info) if p not in read_paths]
                if orphans:
                    orphan_rows = read_episode_rows(orphans, with_index=False)
                    for source, idx in orphan_rows.select(
                        "source", "episode_index"
                    ).iter_rows():
                        held.setdefault(idx, []).append(orphans[source])
            metadata = set(episode_index["episode_index"].to_list())
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(
                path, f"could not reconcile the episode inventory: {exc}"
            ) from exc

        for idx in sorted(held.keys() - metadata):
            yield Episode(
                id=f"episode_{idx:06d}",
                streams=[],
                # Its instructions are unknown, and an empty list would read as
                # an episode recorded without one.
                tasks=None,
                split=split_of(info, idx),
                layout_check=LayoutCheck(
                    pointed_path=None,
                    holding_paths=sorted(set(held[idx]), key=str),
                ),
            )
