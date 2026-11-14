"""What every LeRobot generation shares, whichever layout is on disk."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import logging
import math
import string
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

# External
import polars as pl
from polars.datatypes import DataTypeClass
from upath import UPath

# Internal
from kalanos.analysis.adapters.video import DeclaredFormat
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
    SchemaCheck,
    SchemaMismatch,
    SchemaProblem,
    Stream,
    TimestampDtype,
)
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.assets.yaml_strict import YAMLError, safe_load_strict


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)

# The format announces itself unambiguously through meta/info.json.
CONFIDENCE = 0.95

TIME_COLUMN = "timestamp"

# A dataset card tag names a simulator when it starts with one of these,
# lower-cased, so `Isaac-Sim` and `isaaclab` match `isaac`.
SIMULATOR_TAGS = ("robosuite", "mujoco", "isaac")

# A card's front matter is a short block of metadata; a file that has not
# closed it by then is not read further.
_CARD_FRONT_MATTER_LINES = 500

# These index the dataset rather than measure anything captured in it.
INDEX_COLUMNS = frozenset(
    {"timestamp", "episode_index", "frame_index", "index", "task_index"}
)

VIDEO_DTYPE = "video"
IMAGE_DTYPE = "image"

# Each declared dtype LeRobot writes, mapped to a test on the innermost stored type.
# Width is not compared: a declared float64 stored as Float32 matches.
_FAMILIES: dict[str, Callable[[pl.DataType | DataTypeClass], bool]] = {
    **dict.fromkeys(("float16", "float32", "float64"), lambda t: t.is_float()),
    **dict.fromkeys(
        ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"),
        lambda t: t.is_integer(),
    ),
    "bool": lambda t: t == pl.Boolean,
    "string": lambda t: t == pl.String,
    # A dataset with no annotations stores empty lists, whose innermost type is Null.
    "language": lambda t: isinstance(t, pl.Struct) or t in (pl.String, pl.Null),
}


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


def check_path_template(
    path: UPath, info: dict[str, Any], key: str, fields: frozenset[str]
) -> None:
    """Refuse a path template that names a field the reader never fills.

    A template that is absent or null passes,
    since a dataset without cameras declares no `video_path`.

    Parameters
    ----------
    path : UPath
        The dataset root.
    info : dict
        The parsed `info.json`.
    key : str
        The template's key in `info.json`, such as `data_path`.
    fields : frozenset[str]
        The fields this reader fills into that template.

    Raises
    ------
    AdapterRefusal
        If the template names a field outside `fields`.
    """

    template = info.get(key)
    if not isinstance(template, str):
        return
    version = info.get("codebase_version", "?")
    filled = ", ".join(sorted(fields))
    for _, name, _, _ in string.Formatter().parse(template):
        if name is not None and name not in fields:
            raise AdapterRefusal(
                path,
                f"info.json {key} {template!r} uses {{{name}}}, which this LeRobot "
                f"{version} reader does not fill (it fills {filled})",
            )


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


def _coerced(value: Any, kind: type) -> Any:
    """Return `value` as `kind`, or `None` when it is absent or does not convert.

    Parameters
    ----------
    value : Any
        The value the dataset's metadata holds, `None` when it holds none.
    kind : type
        The type to convert it to, such as `int` or `float`.

    Returns
    -------
    Any
        `kind(value)`, or `None` when `value` is `None` or the conversion fails.
    """

    if value is None:
        return None
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def split_of(info: dict[str, Any], episode_index: int) -> str | None:
    """Name the split `info.json` assigns an episode.

    `splits` maps each name to a `"start:end"` range of episode indices,
    end exclusive. A range in any other form is skipped rather than guessed at.

    Parameters
    ----------
    info : dict
        The parsed `info.json`.
    episode_index : int
        The episode's index in the dataset.

    Returns
    -------
    str or None
        The one split whose range holds `episode_index`,
        `None` when none does or more than one does.
    """

    splits = info.get("splits")
    if not isinstance(splits, dict):
        return None
    found: list[str] = []
    for name, bounds in splits.items():
        if not isinstance(bounds, str):
            continue
        try:
            start, end = (int(part) for part in bounds.split(":"))
        except ValueError:
            continue
        if start <= episode_index < end:
            found.append(str(name))
    return found[0] if len(found) == 1 else None


def declared_format(spec: dict[str, Any]) -> DeclaredFormat | None:
    """Read the video format one video feature declares.

    The rate, pixel format, codec and channel count come from the feature's `info`,
    which LeRobot v3.0 datasets spell `video_info`,
    as `video.fps`, `video.pix_fmt`, `video.codec` and `video.channels`,
    along with the `video.is_depth_map` and `has_audio` flags.
    The frame size comes from there as `video.height`/`video.width` when present,
    and otherwise from `shape`, only when `names` labels its `height` and `width`.
    A value that does not convert to its field's type counts as undeclared,
    and so does a flag that is not a JSON boolean.

    Parameters
    ----------
    spec : dict
        One video feature's own entry under `info.json`'s `"features"`.

    Returns
    -------
    DeclaredFormat or None
        The declared fields, or `None` when the feature declares none of them.
    """

    info = spec.get("info")
    if not isinstance(info, dict):
        info = spec.get("video_info")
    if not isinstance(info, dict):
        info = {}
    height = info.get("video.height")
    width = info.get("video.width")
    shape, names = spec.get("shape"), spec.get("names")
    if (
        isinstance(shape, list)
        and isinstance(names, list)
        and "height" in names
        and "width" in names
    ):
        if height is None and names.index("height") < len(shape):
            height = shape[names.index("height")]
        if width is None and names.index("width") < len(shape):
            width = shape[names.index("width")]
    channels = info.get("video.channels")
    is_depth_map = info.get("video.is_depth_map")
    has_audio = info.get("has_audio")
    declared = DeclaredFormat(
        width=_coerced(width, int),
        height=_coerced(height, int),
        fps=_coerced(info.get("video.fps"), float),
        pix_fmt=_coerced(info.get("video.pix_fmt"), str),
        codec=_coerced(info.get("video.codec"), str),
        # bool is an int subclass, and bool("false") is True: neither may pass.
        channels=None if isinstance(channels, bool) else _coerced(channels, int),
        is_depth_map=is_depth_map if isinstance(is_depth_map, bool) else None,
        has_audio=has_audio if isinstance(has_audio, bool) else None,
    )
    return None if declared == DeclaredFormat() else declared


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

    LeRobot spells them as a list (`"names": ["x", "y"]`), nested under a
    label, as ALOHA's `"names": {"motors": ["left_waist", ...]}`, or as a list
    holding lists, as Unitree's `"names": [["kLeftShoulderPitch", ...]]`.
    The nested lists are joined in order. Anything else declares no names.

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
        if any(isinstance(item, list) for item in names):
            if not all(isinstance(item, list) for item in names):
                return []
            flat_inner: list[str] = []
            for inner in names:
                if not all(isinstance(name, str) for name in inner):
                    return []
                flat_inner.extend(inner)
            return flat_inner
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
        Each carries the feature's declared `dtype`, `None` when it declares none.
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

    dtype = spec.get("dtype")
    axis_by_name = {role.column: role.axis for role in roles(channel_names, dictionary)}
    return [
        Channel(
            name=name,
            axis=axis_by_name[name],
            source_index=i,
            declared_name=names[i] if len(names) == width else None,
            source_dtype=dtype if isinstance(dtype, str) else None,
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
        `episode_count`, `nominal_rate_hz` and `robot_type`, as declared,
        and the `simulator` the dataset card's tags name.
    """

    return DatasetInfo(
        adapter=name,
        path=path,
        episode_count=info.get("total_episodes"),
        nominal_rate_hz=_declared_rate(info.get("fps")),
        robot_type=info.get("robot_type"),
        simulator=card_simulator(path),
    )


def card_simulator(path: UPath) -> str | None:
    """Read the simulator a dataset card's tags name, when it names one.

    The card is `README.md` at the dataset root, whose YAML front matter
    carries a `tags` list. A card is optional metadata, so any problem reading
    it gives `None` rather than costing the dataset its other declared facts.

    Parameters
    ----------
    path : UPath
        The dataset root.

    Returns
    -------
    str or None
        The first tag that starts with a name in `SIMULATOR_TAGS`, lower-cased,
        or `None` when the card is missing, unreadable or malformed,
        or its tags name no simulator.
    """

    try:
        with (path / "README.md").open("r", encoding="utf-8") as handle:
            lines = [
                line.rstrip("\r\n") for line in islice(handle, _CARD_FRONT_MATTER_LINES)
            ]
    except (OSError, UnicodeDecodeError):
        return None
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return None
    try:
        front = safe_load_strict("\n".join(lines[1:end]))
    except YAMLError:
        return None
    tags = front.get("tags") if isinstance(front, dict) else None
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.lower().startswith(SIMULATOR_TAGS):
            return tag.lower()
    return None


def _declared_rate(value: Any) -> float | None:
    """Read a declared rate, or `None` when it is not a usable rate.

    Parameters
    ----------
    value : Any
        The `fps` value `info.json` holds, `None` when it holds none.

    Returns
    -------
    float or None
        The rate in Hz, or `None` when it is absent, unparseable,
        non-finite, or not positive.
    """

    # bool is an int subclass, so float(True) would declare a 1 Hz rate.
    if isinstance(value, bool):
        return None
    rate = _coerced(value, float)
    if rate is None or not math.isfinite(rate) or rate <= 0:
        return None
    return rate


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


def storage_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Replace every extension-typed column with its storage.

    Parameters
    ----------
    frame : pl.DataFrame
        A data parquet as `pl.read_parquet` loaded it.

    Returns
    -------
    pl.DataFrame
        `frame` with each `pl.Extension` column cast to its storage type,
        or `frame` itself when it holds none.
    """

    # polars 2 loads Hugging Face `datasets` Array2D/3D columns as extension types,
    # where polars 1 loads them as their storage, which every reader here expects.
    extension_columns = [
        name for name, dtype in frame.schema.items() if isinstance(dtype, pl.Extension)
    ]
    if not extension_columns:
        return frame
    return frame.with_columns(pl.col(name).ext.storage() for name in extension_columns)


def holds_numbers(dtype: pl.DataType) -> bool:
    """Whether a column of `dtype` holds values grading can read as channels.

    Parameters
    ----------
    dtype : pl.DataType
        A data parquet column's type, as stored or as `storage_columns` left it.

    Returns
    -------
    bool
        True when the innermost type, past any extension, List and Array
        wrappers, is numeric or Boolean.
    """

    innermost = _innermost(dtype)
    # polars counts Boolean as non-numeric, yet Booleans grade as channels.
    return innermost.is_numeric() or innermost == pl.Boolean


def _innermost(dtype: pl.DataType | DataTypeClass) -> pl.DataType | DataTypeClass:
    """Unwrap every extension, List and Array layer around a column's value type.

    Parameters
    ----------
    dtype : pl.DataType | DataTypeClass
        A data parquet column's type.

    Returns
    -------
    pl.DataType | DataTypeClass
        The type of the values the column ultimately holds.
    """

    innermost = dtype
    while True:
        if isinstance(innermost, pl.Extension):
            innermost = innermost.ext_storage()
        elif isinstance(innermost, (pl.List, pl.Array)):
            innermost = innermost.inner
        else:
            return innermost


def _declared_shape(spec: Mapping[str, Any]) -> list[int] | None:
    """Read a feature's declared shape, or `None` when it is not a list of ints.

    Parameters
    ----------
    spec : Mapping[str, Any]
        One `info.json` feature entry.

    Returns
    -------
    list[int] or None
        The declared shape.
    """

    shape = spec.get("shape")
    if isinstance(shape, list) and all(
        isinstance(n, int) and not isinstance(n, bool) for n in shape
    ):
        return shape
    return None


def check_schema(
    info: dict[str, Any], schema: Mapping[str, pl.DataType], source_path: UPath
) -> SchemaCheck:
    """Compare a data parquet's schema with every feature `info.json` declares.

    Video and image features live outside the data parquet and are not
    compared. A feature whose declared dtype this reader does not know is
    checked for presence only, so an unfamiliar vocabulary never counts as a
    mismatch.

    Parameters
    ----------
    info : dict
        The parsed `info.json`.
    schema : Mapping[str, pl.DataType]
        The data parquet's schema, as `pl.read_parquet_schema` reads it.
    source_path : UPath
        The data parquet `schema` was read from.

    Returns
    -------
    SchemaCheck
        The features compared and, for each that disagrees, how.
    """

    checked = []
    mismatches = []
    for key, spec in info.get("features", {}).items():
        dtype = spec.get("dtype") if isinstance(spec, dict) else None
        if dtype in (VIDEO_DTYPE, IMAGE_DTYPE):
            continue
        checked.append(key)
        declared_shape = _declared_shape(spec) if isinstance(spec, dict) else None
        family = _FAMILIES.get(dtype) if isinstance(dtype, str) else None
        found = schema.get(key)
        problem = None
        if found is None:
            problem = SchemaProblem.MISSING
        elif family is not None and not family(_innermost(found)):
            problem = SchemaProblem.DTYPE
        elif family is not None and declared_shape:
            outer = found.ext_storage() if isinstance(found, pl.Extension) else found
            if isinstance(outer, pl.Array) and list(outer.shape) != declared_shape:
                problem = SchemaProblem.SHAPE
        if problem is not None:
            mismatches.append(
                SchemaMismatch(
                    column=key,
                    problem=problem,
                    declared_dtype=str(dtype),
                    declared_shape=declared_shape,
                    found_dtype=None if found is None else str(found),
                )
            )
    return SchemaCheck(source_path=source_path, checked=checked, mismatches=mismatches)


def check_scalar_index(path: UPath, schema: Mapping[str, pl.DataType]) -> None:
    """Refuse a data parquet whose index columns hold lists instead of scalars.

    Parameters
    ----------
    path : UPath
        The dataset root.
    schema : Mapping[str, pl.DataType]
        The data parquet's schema, as `pl.read_parquet_schema` reads it.

    Raises
    ------
    AdapterRefusal
        If `episode_index`, `frame_index` or the timestamp column is a List or Array.
    """

    for name in ("episode_index", "frame_index", TIME_COLUMN):
        dtype = schema.get(name)
        if isinstance(dtype, (pl.List, pl.Array)):
            raise AdapterRefusal(
                path,
                f"column {name!r} is stored as {dtype}; "
                "LeRobot stores one scalar per frame",
            )


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
    schema: Mapping[str, pl.DataType],
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
    schema : Mapping[str, pl.DataType]
        The data parquet's schema, as `pl.read_parquet_schema` reads it, so a
        missing column is still noticed and a column holding no numbers is skipped.
    """

    streams = []
    for feature, (taxonomy_type, mapping_source, channels) in plan.series.items():
        if feature not in schema:
            continue
        if not holds_numbers(schema[feature]):
            logger.debug(
                "%s: feature %r is stored as %s, not numbers; not graded",
                source_path,
                feature,
                schema[feature],
            )
            continue
        streams.append(
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
        )
    return streams


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
    skipped with a debug log rather than raised. A feature whose column holds
    no numbers is skipped the same way.

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
        if not holds_numbers(episode_frame.schema[feature]):
            logger.debug(
                "%s: episode %s: feature %r is stored as %s, not numbers; not graded",
                path,
                episode_label,
                feature,
                episode_frame.schema[feature],
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
