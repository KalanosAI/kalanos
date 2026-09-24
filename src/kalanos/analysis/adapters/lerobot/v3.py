"""LeRobot v3.0: one directory, several parquet chunks, an mp4 per camera per chunk.

One episode spans a data parquet and one mp4 per camera;
one parquet chunk holds many episodes, split by `episode_index`.
`meta/info.json` declares the frame rate and the robot type,
and `meta/episodes/**/*.parquet` maps each episode to the files that hold it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import itertools
from collections.abc import Iterator
from typing import Any

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.common import (
    CONFIDENCE,
    TIME_COLUMN,
    LeRobotAdapter,
    describe_from_info,
    feature_plan,
    read_info,
    sampling_is_regular,
    series_streams,
)
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.adapters.video import VideoPayload
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.domain import Clock, Episode, Kind, Stream


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

_SUPPORTED_CODEBASE_VERSION = "v3.0"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _read_episode_index(path: UPath) -> pl.DataFrame:
    """Read and concatenate every `meta/episodes/**/*.parquet`, sorted by episode.

    Parameters
    ----------
    path : UPath
        The dataset root.

    Returns
    -------
    pl.DataFrame
        Every episode's row, one per episode, sorted by `episode_index`.

    Raises
    ------
    AdapterRefusal
        If `meta/episodes/` holds no parquet, or any of them fail to read.
        A v2.x tree has no such directory, so this is the seam it falls
        through on.
    """

    episodes_dir = path / "meta" / "episodes"
    try:
        parquet_paths = sorted(episodes_dir.rglob("*.parquet"))
        if not parquet_paths:
            raise AdapterRefusal(path, f"no episode index parquet under {episodes_dir}")
        frames = []
        for parquet_path in parquet_paths:
            with parquet_path.open("rb") as handle:
                frames.append(pl.read_parquet(handle))
    except AdapterRefusal:
        raise
    except Exception as exc:
        raise AdapterRefusal(
            path, f"could not read episode index under {episodes_dir}: {exc}"
        ) from exc
    return pl.concat(frames).sort("episode_index")


def _video_stream(
    info: dict[str, Any],
    row: dict[str, Any],
    video_key: str,
    taxonomy_type: str,
    timestamps: pl.Series,
    dataset_root: UPath,
    *,
    is_regular: bool,
) -> Stream:
    """Build one video Stream for `video_key`, its payload lazy and undecoded.

    Parameters
    ----------
    info : dict
        The parsed `info.json`, for `video_path`'s template.
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
        ),
        source_path=video_path,
        source_field=video_key,
        clock=Clock.UNKNOWN,
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
            `episode_count`, `nominal_rate_hz` and `robot_type`, as declared.

        Raises
        ------
        AdapterRefusal
            If `meta/info.json` cannot be read.
        """

        return describe_from_info(self.name, path, read_info(path))

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield one Episode per row of the episode index, lazily.

        Parameters
        ----------
        path : UPath
            The dataset root.
        sample : int or None
            When given, stop after this many episodes.

        Yields
        ------
        Episode
            One recording, its streams drawn from the data parquet chunk
            its row names and, for each declared camera, its own mp4.

        Raises
        ------
        AdapterRefusal
            If anything about the input could not be read.
        """

        if sample is not None and sample <= 0:
            return

        try:
            info = read_info(path)
            episode_index = _read_episode_index(path)
            plan = feature_plan(info, self._resolve_dictionary())
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

        row_iter = episode_index.iter_rows(named=True)
        if sample is not None:
            row_iter = itertools.islice(row_iter, sample)

        for row in row_iter:
            try:
                data_path = path / info["data_path"].format(
                    chunk_index=row["data/chunk_index"],
                    file_index=row["data/file_index"],
                )
                if data_path != cached_path:
                    with data_path.open("rb") as handle:
                        cached_frame = pl.read_parquet(handle)
                    cached_path = data_path
                assert cached_frame is not None
                frame = cached_frame

                episode_frame = frame.filter(
                    pl.col("episode_index") == row["episode_index"]
                ).sort(TIME_COLUMN)
                timestamps = episode_frame[TIME_COLUMN].cast(pl.Float64)
                is_regular = sampling_is_regular(timestamps)

                streams = series_streams(
                    episode_frame,
                    plan,
                    timestamps,
                    data_path,
                    path=path,
                    episode_label=row["episode_index"],
                    is_regular=is_regular,
                )
                streams.extend(
                    _video_stream(
                        info,
                        row,
                        video_key,
                        taxonomy_type,
                        timestamps,
                        path,
                        is_regular=is_regular,
                    )
                    for video_key, taxonomy_type in plan.video.items()
                )
            except AdapterRefusal:
                raise
            except Exception as exc:
                raise AdapterRefusal(
                    path, f"episode {row.get('episode_index')}: {exc}"
                ) from exc

            yield Episode(id=f"episode_{row['episode_index']:06d}", streams=streams)
