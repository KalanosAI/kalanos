"""LeRobot v2.0/v2.1: one parquet and one mp4 per episode, no episode table.

v2.0 and v2.1 share one on-disk layout: one parquet per episode under
`data/chunk-{episode_chunk:03d}/`, one mp4 per episode per camera under
`videos/chunk-{episode_chunk:03d}/{video_key}/`, no episode table, boundaries
carried by the filenames themselves. The two versions differ only in which
stats file `meta/` holds (`meta/stats.json` for v2.0, `meta/episodes_stats.jsonl`
for v2.1), which this reader does not read.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
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

_SUPPORTED_CODEBASE_VERSIONS = frozenset({"v2.0", "v2.1"})

# LeRobot's own default when info.json omits it.
_DEFAULT_CHUNKS_SIZE = 1000


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _video_stream(
    info: dict[str, Any],
    video_key: str,
    taxonomy_type: str,
    timestamps: pl.Series,
    dataset_root: UPath,
    *,
    episode_index: int,
    episode_chunk: int,
    fps: float,
    is_regular: bool,
) -> Stream:
    """Build one video Stream for `video_key`, its payload lazy and undecoded.

    One mp4 holds exactly one episode in this layout, so the payload spans
    the whole file rather than a from/to slice of a shared one.

    Parameters
    ----------
    info : dict
        The parsed `info.json`, for `video_path`'s template.
    video_key : str
        The video feature key.
    taxonomy_type : str
        The taxonomy type `feature_plan` resolved for this key.
    timestamps : pl.Series
        The episode's own timestamps, shared across every one of its streams.
    dataset_root : UPath
        The dataset root, `video_path` is resolved relative to it.
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

    video_path = dataset_root / info["video_path"].format(
        video_key=video_key,
        episode_chunk=episode_chunk,
        episode_index=episode_index,
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
            `episode_count`, `nominal_rate_hz` and `robot_type`, as declared.

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
            plan = feature_plan(info, self._resolve_dictionary())
            total = int(info["total_episodes"])
            fps = float(info["fps"])
            chunks_size = int(info.get("chunks_size", _DEFAULT_CHUNKS_SIZE))
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(path, f"could not read {path}: {exc}") from exc

        episode_count = total if sample is None else min(total, sample)

        for episode_index in range(episode_count):
            try:
                episode_chunk = episode_index // chunks_size
                data_path = path / info["data_path"].format(
                    episode_chunk=episode_chunk, episode_index=episode_index
                )
                # One parquet holds exactly one episode, unlike v3's shared chunks:
                # no episode_index filter and no chunk cache are needed here.
                with data_path.open("rb") as handle:
                    frame = pl.read_parquet(handle)

                episode_frame = frame.sort(TIME_COLUMN)
                timestamps = episode_frame[TIME_COLUMN].cast(pl.Float64)
                is_regular = sampling_is_regular(timestamps)

                streams = series_streams(
                    episode_frame,
                    plan,
                    timestamps,
                    data_path,
                    path=path,
                    episode_label=episode_index,
                    is_regular=is_regular,
                )
                streams.extend(
                    _video_stream(
                        info,
                        video_key,
                        taxonomy_type,
                        timestamps,
                        path,
                        episode_index=episode_index,
                        episode_chunk=episode_chunk,
                        fps=fps,
                        is_regular=is_regular,
                    )
                    for video_key, taxonomy_type in plan.video.items()
                )
            except AdapterRefusal:
                raise
            except Exception as exc:
                raise AdapterRefusal(path, f"episode {episode_index}: {exc}") from exc

            yield Episode(id=f"episode_{episode_index:06d}", streams=streams)
