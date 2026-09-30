"""What every LeRobot generation shares, whichever layout is on disk."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import logging
import math
from collections.abc import Iterable
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
    ClockInfo,
    ClockOrigin,
    FramePayload,
    Kind,
    MappingSource,
    OriginEvidence,
    Stream,
    TimestampDtype,
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
    series : dict[str, tuple[str, MappingSource | None, list[Channel]]]
        Each series feature key, mapped to its taxonomy type, how that was decided,
        and its channels.
    video : dict[str, tuple[str, MappingSource | None]]
        Each video feature key, mapped to its taxonomy type and how that was decided.
    """

    series: dict[str, tuple[str, MappingSource | None, list[Channel]]]
    video: dict[str, tuple[str, MappingSource | None]]


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


def resolve_taxonomy(
    feature: str, spec: dict[str, Any], dictionary: Dictionary
) -> tuple[str, MappingSource | None]:
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
    tuple[str, MappingSource or None]
        The stream's taxonomy type, and `DICTIONARY` when the key matched,
        `DECLARED_NAMES` when the names did, or `None` when it stayed unmapped.
    """

    [key_role] = roles([feature], dictionary)
    if key_role.taxonomy_type is not None:
        return key_role.taxonomy_type, MappingSource.DICTIONARY

    names: list[str] = declared_names(spec)
    if names:
        name_roles = roles(names, dictionary)
        resolved = {role.taxonomy_type for role in name_roles if role.taxonomy_type}
        if len(resolved) == 1 and all(r.taxonomy_type for r in name_roles):
            [taxonomy_type] = resolved
            return taxonomy_type, MappingSource.DECLARED_NAMES

    return f"{UNMAPPED_TAXONOMY_PREFIX}.{feature}", None


def declared_names(spec: dict[str, Any]) -> list[str]:
    """A feature's declared channel names, as one flat list.

    LeRobot spells them either as a list (`"names": ["x", "y"]`) or nested
    under a label, as ALOHA's `"names": {"motors": ["left_waist", ...]}`; the
    nested lists are joined in order. Anything else declares no names.

    Parameters
    ----------
    spec : dict
        One feature's entry under `info.json`'s `"features"`.

    Returns
    -------
    list[str]
        The names, or an empty list when none are declared.
    """

    names = spec.get("names")
    if isinstance(names, list):
        return [str(name) for name in names if isinstance(name, str)]
    if isinstance(names, dict):
        flat: list[str] = []
        for value in names.values():
            if not isinstance(value, list):
                return []
            flat.extend(str(name) for name in value if isinstance(name, str))
        return flat
    return []


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

    names: list[str] = declared_names(spec)
    shape = spec.get("shape") or [1]
    width = shape[0] if shape else 1

    if width == 1:
        channel_names = [feature]
    elif len(names) == width:
        channel_names = names
    else:
        channel_names = [f"{feature}_{i}" for i in range(width)]

    axis_by_name = {role.column: role.axis for role in roles(channel_names, dictionary)}
    return [
        Channel(
            name=name,
            axis=axis_by_name[name],
            source_index=i,
            declared_name=names[i] if len(names) == width else None,
        )
        for i, name in enumerate(channel_names)
    ]


def taxonomy_and_channels(
    feature: str, spec: dict[str, Any], dictionary: Dictionary
) -> tuple[str, MappingSource | None, list[Channel]]:
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
    tuple[str, MappingSource or None, list[Channel]]
        `resolve_taxonomy`'s and `channels_for`'s results, together.
    """

    taxonomy_type, mapping_source = resolve_taxonomy(feature, spec, dictionary)
    return taxonomy_type, mapping_source, channels_for(feature, spec, dictionary)


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
        The episode's own timestamps, in source row order.

    Returns
    -------
    bool
        Whether the gaps between consecutive timestamps classify as regular.
    """

    ts_values = timestamps.to_list()
    if any(value is None or not math.isfinite(value) for value in ts_values):
        return False
    gaps = [b - a for a, b in zip(ts_values, ts_values[1:], strict=False)]
    return regularity(gaps).is_regular


def timestamp_dtype_of(column: pl.Series) -> TimestampDtype:
    """Name the float format an episode's timestamp column was stored in."""

    return (
        TimestampDtype.FLOAT32 if column.dtype == pl.Float32 else TimestampDtype.FLOAT64
    )


def episode_clock(
    episode_frame: pl.DataFrame, fps: float | None, timestamp_dtype: TimestampDtype
) -> Clock:
    """Label an episode's clock `RECONSTRUCTED` when its stamps are `frame_index / fps`.

    Every timestamp has to match to within the rounding its source format allows,
    including when rows are missing, repeated or out of order. Those defects
    remain visible to structural timing checks.

    Parameters
    ----------
    episode_frame : pl.DataFrame
        The episode's own rows, in source row order.
    fps : float or None
        The declared frame rate, or `None` when `info.json` declares none.
    timestamp_dtype : TimestampDtype
        The format the timestamp column was stored in.

    Returns
    -------
    Clock
        `RECONSTRUCTED` when every condition holds, `UNKNOWN` otherwise.
    """

    if (
        fps is None
        or not math.isfinite(fps)
        or fps <= 0
        or "frame_index" not in episode_frame.columns
    ):
        return Clock.UNKNOWN
    frame_index = episode_frame["frame_index"]
    has_nulls = frame_index.null_count() or episode_frame[TIME_COLUMN].null_count()
    if episode_frame.height == 0 or has_nulls:
        return Clock.UNKNOWN
    if (
        not frame_index.is_finite().all()
        or not episode_frame[TIME_COLUMN].is_finite().all()
    ):
        return Clock.UNKNOWN

    timestamp = pl.col(TIME_COLUMN).cast(pl.Float64)
    deviation, largest = episode_frame.select(
        (timestamp - pl.col("frame_index") / fps).abs().max().alias("deviation"),
        timestamp.abs().max().alias("largest"),
    ).row(0)
    within = (
        deviation is not None
        and largest is not None
        and math.isfinite(deviation)
        and math.isfinite(largest)
        and deviation <= timestamp_dtype.epsilon * max(largest, 1.0)
    )
    return Clock.RECONSTRUCTED if within else Clock.UNKNOWN


def series_stream(
    frame: pl.DataFrame,
    feature: str,
    channels: list[Channel],
    taxonomy_type: str,
    timestamps: pl.Series,
    source_path: UPath,
    *,
    mapping_source: MappingSource | None,
    clock: Clock,
    timestamp_dtype: TimestampDtype,
    is_regular: bool,
) -> Stream:
    """Build one series Stream for `feature`, one column per channel.

    Parameters
    ----------
    frame : pl.DataFrame
        The episode's own rows, in source row order.
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
    mapping_source : MappingSource or None
        How `taxonomy_type` was decided.
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
        mapping_source=mapping_source,
        clock=clock,
        timestamp_dtype=timestamp_dtype,
        is_regular=is_regular,
        channels=channels,
    )


def declared_series_streams(
    plan: FeaturePlan,
    present_columns: Iterable[str],
    timestamps: pl.Series,
    source_path: UPath,
    *,
    clock: Clock,
    timestamp_dtype: TimestampDtype,
    is_regular: bool,
) -> list[Stream]:
    """Series streams with their channels declared but no payload read.

    What a metadata-tier run yields: the stream exists, its channels are
    known from the manifest and the parquet schema says the column is
    there, but no numeric value has been materialised. Grading records such
    a stream as `skipped`, never as computed.

    Parameters
    ----------
    plan : FeaturePlan
        The dataset's resolved feature plan.
    present_columns : Iterable[str]
        The columns the data parquet's schema declares, from
        `pl.read_parquet_schema`, so a missing column is still noticed.
    """

    present = set(present_columns)
    return [
        Stream(
            taxonomy_type=taxonomy_type,
            kind=Kind.SERIES,
            timestamps=timestamps,
            payload=None,
            source_path=source_path,
            source_field=feature,
            mapping_source=mapping_source,
            clock=clock,
            timestamp_dtype=timestamp_dtype,
            is_regular=is_regular,
            channels=channels,
        )
        for feature, (taxonomy_type, mapping_source, channels) in plan.series.items()
        if feature in present
    ]


def series_streams(
    episode_frame: pl.DataFrame,
    plan: FeaturePlan,
    timestamps: pl.Series,
    source_path: UPath,
    *,
    path: UPath,
    episode_label: object,
    clock: Clock,
    timestamp_dtype: TimestampDtype,
    is_regular: bool,
) -> list[Stream]:
    """Build one series Stream per `plan.series` feature present in `episode_frame`.

    A feature `plan.series` declares but `episode_frame` does not carry is
    skipped with a debug log rather than raised.

    Parameters
    ----------
    episode_frame : pl.DataFrame
        The episode's own rows, in source row order.
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
    for feature, (taxonomy_type, mapping_source, channels) in plan.series.items():
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
                mapping_source=mapping_source,
                clock=clock,
                timestamp_dtype=timestamp_dtype,
                is_regular=is_regular,
            )
        )
    return streams


def with_episode_clock(
    streams: list[Stream], frame: pl.DataFrame, *, clock: Clock, domain: str
) -> list[Stream]:
    """Attach the actual recorded time column, without claiming decoded video PTS."""
    native = frame[TIME_COLUMN]
    info = ClockInfo(
        origin=ClockOrigin.GENERATED
        if clock == Clock.RECONSTRUCTED
        else ClockOrigin.UNKNOWN,
        origin_evidence=OriginEvidence.INFERRED
        if clock == Clock.RECONSTRUCTED
        else OriginEvidence.NONE,
        source_field=TIME_COLUMN,
        native_unit="s",
        native_dtype=str(native.dtype),
        epoch="episode-relative",
        domain=domain,
        transforms=[f"{native.dtype}->Float64 seconds"],
    )
    return [
        s.model_copy(
            update={
                "clock_info": info.model_copy(deep=True),
                "clock": info.origin.compatibility_clock,
                "native_timestamps": native,
            }
        )
        for s in streams
    ]
