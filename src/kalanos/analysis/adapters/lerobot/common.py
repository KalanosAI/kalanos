"""What every LeRobot generation shares, whichever layout is on disk."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import logging
from dataclasses import dataclass
from typing import Any

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.inference.regularity import regularity
from kalanos.analysis.inference.roles import roles
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.domain import (
    UNMAPPED_TAXONOMY_PREFIX,
    Channel,
    Clock,
    FramePayload,
    Kind,
    Stream,
)
from kalanos.assets.dictionary import load_default_dictionary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)

# The format announces itself unambiguously through meta/info.json.
CONFIDENCE = 0.95

TIME_COLUMN = "timestamp"

# These index the dataset rather than measure anything captured in it.
INDEX_COLUMNS = frozenset(
    {"timestamp", "episode_index", "frame_index", "index", "task_index"}
)

VIDEO_DTYPE = "video"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class LeRobotAdapter:
    """Everything the v2 and v3 readers share: how the dictionary is resolved."""

    def __init__(self, *, dictionary: Dictionary | None = None) -> None:
        self._dictionary = dictionary

    def _resolve_dictionary(self) -> Dictionary:
        """Return the dictionary to resolve feature names against.

        Loads the packaged default lazily, on first use, so constructing
        this adapter during plugin discovery never reads the packaged YAML.

        Returns
        -------
        Dictionary
            The dictionary passed to `__init__`, or the loaded default.
        """

        if self._dictionary is None:
            self._dictionary = load_default_dictionary()
        return self._dictionary


@dataclass(frozen=True)
class FeaturePlan:
    """What `episodes()` needs to build every stream, resolved once per dataset.

    Attributes
    ----------
    series : dict[str, tuple[str, list[Channel]]]
        Each series feature key, mapped to its taxonomy type and channels.
    video : dict[str, str]
        Each video feature key, mapped to its taxonomy type.
    """

    series: dict[str, tuple[str, list[Channel]]]
    video: dict[str, str]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_info(path: UPath) -> dict[str, Any]:
    """Read and parse `path/meta/info.json`.

    Parameters
    ----------
    path : UPath
        The dataset root.

    Returns
    -------
    dict
        The parsed `info.json`.

    Raises
    ------
    AdapterRefusal
        If the file is missing, unreadable, or not valid JSON.
    """

    info_path = path / "meta" / "info.json"
    try:
        with info_path.open("r") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterRefusal(path, f"could not read {info_path}: {exc}") from exc


def video_keys(info: dict[str, Any]) -> list[str]:
    """List every feature key whose declared dtype is video, in features order.

    Parameters
    ----------
    info : dict
        The parsed `info.json`.

    Returns
    -------
    list[str]
        The video feature keys.
    """

    return [
        key
        for key, spec in info.get("features", {}).items()
        if spec.get("dtype") == VIDEO_DTYPE
    ]


def resolve_taxonomy(feature: str, spec: dict[str, Any], dictionary: Dictionary) -> str:
    """Resolve one feature's taxonomy type: its key first, then its declared `names`.

    Tries the feature key itself first.
    When that yields nothing, falls back to the feature's own declared `names`.
    A feature matching neither stays `unmapped.<feature>`.

    Parameters
    ----------
    feature : str
        The feature key as `info.json` spells it.
    spec : dict
        That feature's own entry under `info.json`'s `"features"`.
    dictionary : Dictionary
        The taxonomy to resolve names against.

    Returns
    -------
    str
        The stream's taxonomy type.
    """

    [key_role] = roles([feature], dictionary)
    taxonomy_type = key_role.taxonomy_type

    names: list[str] = spec.get("names") or []
    if taxonomy_type is None and names:
        name_roles = roles(names, dictionary)
        resolved = {role.taxonomy_type for role in name_roles if role.taxonomy_type}
        if len(resolved) == 1:
            [taxonomy_type] = resolved

    if taxonomy_type is None:
        taxonomy_type = f"{UNMAPPED_TAXONOMY_PREFIX}.{feature}"
    return taxonomy_type


def channels_for(
    feature: str, spec: dict[str, Any], dictionary: Dictionary
) -> list[Channel]:
    """Resolve one series feature's channels.

    Only meaningful for a series feature: a video feature's `shape` names
    its frame dimensions, not a channel count, so this is never called for one.

    Parameters
    ----------
    feature : str
        The feature key as `info.json` spells it.
    spec : dict
        That feature's own entry under `info.json`'s `"features"`.
    dictionary : Dictionary
        The taxonomy to resolve each channel's axis against.

    Returns
    -------
    list[Channel]
        One Channel per member. Named from `names` when its length matches
        the feature's width, indexed otherwise. A scalar feature yields
        exactly one channel, named after the feature itself.
    """

    names: list[str] = spec.get("names") or []
    shape = spec.get("shape") or [1]
    width = shape[0] if shape else 1

    if width == 1:
        channel_names = [feature]
    elif len(names) == width:
        channel_names = names
    else:
        channel_names = [f"{feature}_{i}" for i in range(width)]

    axis_by_name = {role.column: role.axis for role in roles(channel_names, dictionary)}
    return [Channel(name=name, axis=axis_by_name[name]) for name in channel_names]


def taxonomy_and_channels(
    feature: str, spec: dict[str, Any], dictionary: Dictionary
) -> tuple[str, list[Channel]]:
    """Resolve one series feature's taxonomy type and its channels together.

    Parameters
    ----------
    feature : str
        The feature key as `info.json` spells it.
    spec : dict
        That feature's own entry under `info.json`'s `"features"`.
    dictionary : Dictionary
        The taxonomy to resolve names against.

    Returns
    -------
    tuple[str, list[Channel]]
        `resolve_taxonomy`'s and `channels_for`'s results, paired.
    """

    return resolve_taxonomy(feature, spec, dictionary), channels_for(
        feature, spec, dictionary
    )


def feature_plan(info: dict[str, Any], dictionary: Dictionary) -> FeaturePlan:
    """Resolve every declared feature's taxonomy once, ahead of any episode read.

    Parameters
    ----------
    info : dict
        The parsed `info.json`.
    dictionary : Dictionary
        The taxonomy to resolve feature names against.

    Returns
    -------
    FeaturePlan
        The series and video feature plans.
    """

    features = info.get("features", {})
    videos = video_keys(info)
    series_keys = [
        key for key in features if key not in INDEX_COLUMNS and key not in videos
    ]
    return FeaturePlan(
        series={
            key: taxonomy_and_channels(key, features[key], dictionary)
            for key in series_keys
        },
        video={key: resolve_taxonomy(key, features[key], dictionary) for key in videos},
    )


def describe_from_info(name: str, path: UPath, info: dict[str, Any]) -> DatasetInfo:
    """Report `info.json`'s own facts, without reading any episode.

    Parameters
    ----------
    name : str
        The adapter's own name, carried onto `DatasetInfo.adapter`.
    path : UPath
        The dataset root.
    info : dict
        The parsed `info.json`.

    Returns
    -------
    DatasetInfo
        `episode_count`, `nominal_rate_hz` and `robot_type`, as declared.
    """

    return DatasetInfo(
        adapter=name,
        path=path,
        episode_count=info.get("total_episodes"),
        nominal_rate_hz=float(info["fps"]) if "fps" in info else None,
        robot_type=info.get("robot_type"),
    )


def sampling_is_regular(timestamps: pl.Series) -> bool:
    """Classify an episode's own timestamps as regular or not.

    The declared fps is never trusted to assert regularity from:
    these timestamps are frame_index / fps, so that would be circular.
    A dropped row still shows up as a doubled gap.

    Parameters
    ----------
    timestamps : pl.Series
        The episode's own timestamps, already time-sorted.

    Returns
    -------
    bool
        Whether the gaps between consecutive timestamps classify as regular.
    """

    ts_values = timestamps.to_list()
    gaps = [b - a for a, b in zip(ts_values, ts_values[1:], strict=False)]
    return regularity(gaps).is_regular


def series_stream(
    frame: pl.DataFrame,
    feature: str,
    channels: list[Channel],
    taxonomy_type: str,
    timestamps: pl.Series,
    source_path: UPath,
    *,
    is_regular: bool,
) -> Stream:
    """Build one series Stream for `feature`, one column per channel.

    Parameters
    ----------
    frame : pl.DataFrame
        The episode's own rows, already time-sorted.
    feature : str
        The feature key as `info.json` spells it.
    channels : list[Channel]
        The channels `taxonomy_and_channels` resolved for this feature.
    taxonomy_type : str
        The taxonomy type `taxonomy_and_channels` resolved for this feature.
    timestamps : pl.Series
        The episode's own timestamps, shared across every one of its streams.
    source_path : UPath
        The data parquet this stream's rows were read from.
    is_regular : bool
        Whether the episode's own sampling classified as regular.

    Returns
    -------
    Stream
        One series stream, its List or Array column unnested into one channel each,
        or its single scalar column carried through unchanged.
    """

    dtype = frame.schema[feature]
    if isinstance(dtype, (pl.List, pl.Array)):
        members = (
            pl.col(feature).arr if isinstance(dtype, pl.Array) else pl.col(feature).list
        )
        payload_frame = frame.select(
            [members.get(i).alias(c.name) for i, c in enumerate(channels)]
        )
    else:
        [channel] = channels
        payload_frame = frame.select(pl.col(feature).alias(channel.name))

    return Stream(
        taxonomy_type=taxonomy_type,
        kind=Kind.SERIES,
        timestamps=timestamps,
        payload=FramePayload(frame=payload_frame),
        source_path=source_path,
        source_field=feature,
        # LeRobot's timestamp is frame_index / fps:
        # a timebase synthesised from the declared rate, not a recorded capture clock.
        clock=Clock.UNKNOWN,
        is_regular=is_regular,
        channels=channels,
    )


def series_streams(
    episode_frame: pl.DataFrame,
    plan: FeaturePlan,
    timestamps: pl.Series,
    source_path: UPath,
    *,
    path: UPath,
    episode_label: object,
    is_regular: bool,
) -> list[Stream]:
    """Build one series Stream per `plan.series` feature present in `episode_frame`.

    A feature `plan.series` declares but `episode_frame` does not carry is
    skipped with a debug log rather than raised.

    Parameters
    ----------
    episode_frame : pl.DataFrame
        The episode's own rows, already time-sorted.
    plan : FeaturePlan
        The dataset's resolved feature plan.
    timestamps : pl.Series
        The episode's own timestamps, shared across every one of its streams.
    source_path : UPath
        The data parquet these rows were read from.
    path : UPath
        The dataset root, named in the debug log.
    episode_label : object
        The episode's own index, named in the debug log.
    is_regular : bool
        Whether the episode's own sampling classified as regular.

    Returns
    -------
    list[Stream]
        One stream per `plan.series` feature found in `episode_frame`'s columns.
    """

    streams = []
    for feature, (taxonomy_type, channels) in plan.series.items():
        if feature not in episode_frame.columns:
            logger.debug(
                "%s: episode %s: declared feature %r has no column",
                path,
                episode_label,
                feature,
            )
            continue
        streams.append(
            series_stream(
                episode_frame,
                feature,
                channels,
                taxonomy_type,
                timestamps,
                source_path,
                is_regular=is_regular,
            )
        )
    return streams
