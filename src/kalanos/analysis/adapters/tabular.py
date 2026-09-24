"""Parse-then-infer for every table-shaped format, written once.

A table-shaped adapter writes only its `parse` step; everything after
parsing — resolving the schema, converting time to seconds, splitting rows
into instances, and typing columns into streams — is format-agnostic, which
is why `inference` is a library rather than something each adapter repeats.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.inference import infer_schema, roles
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.dictionary import Dictionary, GroupHintKind, normalise_name
from kalanos.analysis.models.domain import (
    CANONICAL_TIME_COLUMN,
    CANONICAL_TIME_UNIT,
    UNMAPPED_TAXONOMY_PREFIX,
    Attribution,
    Channel,
    Clock,
    Episode,
    FramePayload,
    Kind,
    Stream,
)
from kalanos.analysis.models.schema import ColumnRole, RefusalCode
from kalanos.assets.dictionary import load_default_dictionary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Every unit `inference` can resolve a time column to, expressed as the
# factor that converts it to seconds.
_UNIT_TO_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}

# Canonical channel order within a stem group: axis letters first, in this rank order,
# then numeric indices ascending, then anything unsuffixed.
_AXIS_ORDER = {"x": 0, "y": 1, "z": 2, "w": 3}

# What a positional array's consecutive index tokens promote to,
# when the stem's entry declares group_hint: axes.
_INDEX_TO_AXIS_LETTER = {"0": "x", "1": "y", "2": "z", "3": "w"}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _normalise_time(
    frame: pl.DataFrame, source_column: str, unit: str
) -> tuple[pl.DataFrame, str, str, float | None]:
    """Convert a time column to seconds and rename it, when a unit is known.

    An `"unknown"` unit has nothing to convert by, so the column is left
    exactly as the source had it rather than mislabelled as seconds.

    Parameters
    ----------
    frame : pl.DataFrame
        The source frame, time column not yet touched.
    source_column : str
        The time column's name as `inference` found it.
    unit : str
        The unit `inference` resolved for `source_column`, or `"unknown"`.

    Returns
    -------
    tuple[pl.DataFrame, str, str, float or None]
        1. The frame with the time column converted
        2. The time column's name in that frame
        3. The unit it is now expressed in
        4. The factor applied to reach seconds,
           or `None` when `unit` was `"unknown"` and nothing was converted.

    Raises
    ------
    ValueError
        If the frame already carries an unrelated column named `CANONICAL_TIME_COLUMN` —
        converting `source_column` into it would silently overwrite real data.
    """

    if unit == "unknown":
        return frame, source_column, unit, None

    if (
        CANONICAL_TIME_COLUMN in frame.columns
        and CANONICAL_TIME_COLUMN != source_column
    ):
        raise ValueError(
            f"cannot normalise {source_column!r} to {CANONICAL_TIME_COLUMN!r}: "
            f"the source already has an unrelated column with that name"
        )

    factor = _UNIT_TO_SECONDS[unit]
    frame = frame.with_columns(
        (pl.col(source_column) * factor).alias(CANONICAL_TIME_COLUMN)
    )
    if source_column != CANONICAL_TIME_COLUMN:
        frame = frame.drop(source_column)
    return frame, CANONICAL_TIME_COLUMN, CANONICAL_TIME_UNIT, factor


def _split_by_instance(
    frame: pl.DataFrame, key_column: str | None
) -> list[tuple[str | None, Attribution, pl.DataFrame]]:
    """Split a frame into per-instance frames, keeping a null key's rows apart.

    Splits on nullness before partitioning, which is what stops a null key
    bucketing under the string `"None"`. Only a true null counts here; an
    empty-string key is left alone.

    Parameters
    ----------
    frame : pl.DataFrame
        The frame to split, time already normalised.
    key_column : str or None
        The column that splits the frame into instances,
        or `None` when the source declared no such column.

    Returns
    -------
    list[tuple[str or None, Attribution, pl.DataFrame]]
        `(instance, attribution, sub_frame)` triples:
        - `SINGLE`: a single group when there is no key.
        - `KEYED`: the keyed groups, id-sorted.
        - `UNATTRIBUTED`: when any row's key was null, one more group
          last, with `instance=None`, so it reads as an appendix rather
          than sorting into the middle of the real subjects.
    """

    if key_column is None:
        return [(None, Attribution.SINGLE, frame)]

    is_null = pl.col(key_column).is_null()
    null_frame = frame.filter(is_null)
    keyed_frame = frame.filter(~is_null)

    partitions = keyed_frame.partition_by(key_column, as_dict=True, maintain_order=True)
    keyed: list[tuple[str | None, Attribution, pl.DataFrame]] = [
        (str(key[0]), Attribution.KEYED, part) for key, part in partitions.items()
    ]
    groups = sorted(keyed, key=lambda group: group[0] or "")

    if null_frame.height:
        groups.append((None, Attribution.UNATTRIBUTED, null_frame))

    return groups


def _bare_axis(column: str) -> str | None:
    """Return the axis a column named after nothing but its axis carries."""

    normalised = normalise_name(column)
    if normalised.axis is None and normalised.stem in _AXIS_ORDER:
        return normalised.stem
    return None


def _group_by_stem(
    columns: list[str], *, file_stem: str
) -> list[tuple[str, list[str]]]:
    """Group columns by shared normalised stem, in first-seen order.

    A bare axis letter (`x`, `y`, `z`) has no signal name of its own,
    so it groups under `file_stem`, merging with any real stem that matches.
    `w` joins this only when the file also has a bare `x`, `y` or `z`;
    alone, it is watts as often as an axis, so it keeps its own name.

    Parameters
    ----------
    columns : list[str]
        Numeric column names, in source order.
    file_stem : str
        The source file's own normalised stem, used to group bare-axis columns.

    Returns
    -------
    list[tuple[str, list[str]]]
        `(stem, member columns)` pairs, in the order each stem first appeared.
    """

    bare = {column: _bare_axis(column) for column in columns}
    if "w" in bare.values() and not ({"x", "y", "z"} & set(bare.values())):
        bare = {
            column: (None if axis == "w" else axis) for column, axis in bare.items()
        }

    groups: list[tuple[str, list[str]]] = []
    index_by_stem: dict[str, int] = {}

    for column in columns:
        stem = file_stem if bare[column] else (normalise_name(column).stem or column)
        if stem not in index_by_stem:
            index_by_stem[stem] = len(groups)
            groups.append((stem, []))
        groups[index_by_stem[stem]][1].append(column)

    return groups


def _axes_for_group(
    columns: list[str],
    roles_by_column: dict[str, ColumnRole],
    *,
    array_derived: bool,
) -> dict[str, str | None]:
    """Resolve one stem group's per-column axes, demoting a lone `w`.

    `w` is watts as often as it is a quaternion axis, and a real rotation always ships
    x/y/z alongside it — so `w` counts as an axis only when the group also carries one
    of them; otherwise it is dropped back to `None` and the stem is left alone.

    Parameters
    ----------
    columns : list[str]
        The group's member columns.
    roles_by_column : dict[str, ColumnRole]
        Every column's dictionary resolution.
    array_derived : bool
        Whether this stem's columns were expanded from a positional array,
        rather than hand-written with a numeric suffix.

    Returns
    -------
    dict[str, str or None]
        Each column's resolved axis.
    """

    axes = {
        column: roles_by_column[column].axis or _bare_axis(column) for column in columns
    }

    # A hand-written `foo_0, foo_1, foo_2` keeps its index axes.
    # Only a group the parse step expanded from a positional array,
    # and whose entry expects axis letters, is eligible — and
    # only when its axes are exactly the consecutive tokens a promotion has letters for;
    if array_derived and roles_by_column[columns[0]].group_hint is GroupHintKind.AXES:
        axis_values = set(axes.values())
        if axis_values in ({"0", "1", "2"}, {"0", "1", "2", "3"}):
            axes = {
                column: _INDEX_TO_AXIS_LETTER[axis] if axis is not None else None
                for column, axis in axes.items()
            }

    if "w" in axes.values() and not ({"x", "y", "z"} & set(axes.values())):
        demoted = [column for column, axis in axes.items() if axis == "w"]
        logger.debug(
            "stem with columns %s: demoting axis 'w' with no x/y/z sibling",
            ", ".join(demoted),
        )
        axes = {
            column: (None if axis == "w" else axis) for column, axis in axes.items()
        }

    return axes


def _warn_on_duplicate_axes(
    stem: str, axes: dict[str, str | None], *, path: UPath
) -> None:
    """Warn when more than one column in a stem group claims the same axis.

    Both columns still survive as channels; the warning is the only effect.

    Parameters
    ----------
    stem : str
        The group's shared stem, named in the warning.
    axes : dict[str, str or None]
        Each column's resolved axis.
    path : UPath
        The source file, named in the warning.
    """

    columns_by_axis: dict[str, list[str]] = {}
    for column, axis in axes.items():
        if axis is not None:
            columns_by_axis.setdefault(axis, []).append(column)

    for axis, claimants in columns_by_axis.items():
        if len(claimants) > 1:
            logger.warning(
                "%s: stem %r has %d columns claiming axis %r: %s",
                path,
                stem,
                len(claimants),
                axis,
                ", ".join(claimants),
            )


def _order_by_axis(
    columns: list[str], axis_by_column: dict[str, str | None]
) -> list[str]:
    """Order a stem group's columns canonically.

    Parameters
    ----------
    columns : list[str]
        The group's member columns, in source order.
    axis_by_column : dict[str, str or None]
        Each column's axis or index token, as `roles` resolved it.

    Returns
    -------
    list[str]
        `columns`, stably reordered so known axis tokens come first in
        `_AXIS_ORDER` order, numeric indices come next ascending, and
        anything unsuffixed keeps its source position at the end.
    """

    def _sort_key(item: tuple[int, str]) -> tuple[int, int, int]:
        index, column = item
        axis = axis_by_column.get(column)
        if axis in _AXIS_ORDER:
            return (0, _AXIS_ORDER[axis], index)
        if axis is not None and axis.isdigit():
            return (1, int(axis), index)
        return (2, 0, index)

    ordered = sorted(enumerate(columns), key=_sort_key)
    return [column for _, column in ordered]


def _stream_type(stem: str, role: ColumnRole, *, path: UPath) -> str:
    """Decide the taxonomy type for one stream.

    Parameters
    ----------
    stem : str
        The normalised stem the stream's columns were grouped under.
    role : ColumnRole
        The dictionary's resolution for the group's representative column.
    path : UPath
        The source file, named in the warning when the stem is ambiguous.

    Returns
    -------
    str
        `role.taxonomy_type` when the dictionary resolved it,
        otherwise `unmapped.<stem>`.
    """

    if role.taxonomy_type is not None:
        return role.taxonomy_type

    if role.candidates:
        logger.warning(
            "%s: stem %r is ambiguous between %s",
            path,
            stem,
            ", ".join(role.candidates),
        )

    return f"{UNMAPPED_TAXONOMY_PREFIX}.{stem}"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class ParsedTable:
    """One file's rows, and which of its stems came from a positional array.

    Attributes
    ----------
    frame : pl.DataFrame
        The file's data, parsed but otherwise untouched.
    array_stems : frozenset[str]
        Stems whose member columns were expanded from an array,
        so their numeric index suffixes carry the source's own ordering.
        A stem outside this set has index suffixes a human wrote,
        which say nothing about which axis each index is.
    """

    frame: pl.DataFrame
    array_stems: frozenset[str] = frozenset()


class TabularAdapter(ABC):
    """Parse-then-infer for every table-shaped format.

    A subclass writes `name`, `detect` and `parse`. Everything after the parse —
    resolving the time axis, converting it to seconds, splitting the rows into
    instances, and grouping the columns into typed streams — is format-agnostic.
    """

    name: str

    def __init__(self, *, dictionary: Dictionary | None = None) -> None:
        self._dictionary = dictionary

    def _resolve_dictionary(self) -> Dictionary:
        """Return the dictionary to resolve column names against.

        Loads the packaged default lazily, on first use, so constructing an
        adapter during plugin discovery never reads the packaged YAML.

        Returns
        -------
        Dictionary
            The dictionary passed to `__init__`, or the loaded default.
        """

        if self._dictionary is None:
            self._dictionary = load_default_dictionary()
        return self._dictionary

    @abstractmethod
    def parse(self, path: UPath) -> ParsedTable:
        """Read the file into a frame. The only format-specific step.

        Parameters
        ----------
        path : UPath
            The file to read.

        Returns
        -------
        ParsedTable
            The file's data, parsed but otherwise untouched,
            plus which stems (if any) came from a positional array.
        """

    @abstractmethod
    def detect(self, path: UPath) -> float:
        """Say how confident this adapter is that it can read the path.

        Parameters
        ----------
        path : UPath
            The path to consider.

        Returns
        -------
        float
            A confidence clamped to 0.0-1.0 by contract.
        """

    def describe(self, path: UPath) -> DatasetInfo:
        """Report what is known about the path without reading its episodes.

        Parameters
        ----------
        path : UPath
            The path to describe.

        Returns
        -------
        DatasetInfo
            Just the adapter's name and the path — a tabular format usually
            declares nothing before it is read.
        """

        return DatasetInfo(adapter=self.name, path=path)

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield the path's one episode, lazily.

        Parameters
        ----------
        path : UPath
            The path to read.
        sample : int or None
            When `0` or negative, stop before reading anything.

        Yields
        ------
        Episode
            The recording `path` holds.

        Raises
        ------
        AdapterRefusal
            If no column resolved as a time axis, if `parse` raised for any
            reason, or if the resolved time column would collide with an
            existing, unrelated `time_s` column.
        """

        # Step 1: a tabular file is one episode, so nothing is read when
        # nothing was asked for.
        if sample is not None and sample <= 0:
            return

        # Step 2: parse — the only read. Kept to AdapterRefusal-or-nothing
        # regardless of what parse() raises.
        try:
            parsed = self.parse(path)
        except AdapterRefusal:
            raise
        except Exception as exc:
            logger.debug("%s: parse() raised", path, exc_info=True)
            raise AdapterRefusal(
                path, f"parse failed: {type(exc).__name__}: {exc}"
            ) from exc
        frame = parsed.frame

        # Step 3: resolve the schema over the already-parsed frame.
        schema = infer_schema(frame)
        time_spec = schema.time
        if time_spec is None or time_spec.column is None:
            detail = (
                "; ".join(time_spec.evidence)
                if time_spec
                else "no schema could be resolved"
            )
            raise AdapterRefusal(
                path,
                f"no candidate column resolved as a time axis ({detail})",
                schema_so_far=schema,
                code=RefusalCode.NO_TIME_INDEX,
            )

        # TODO: time_spec.regularity carries the nominal sampling interval that
        # timing metrics need, and Episode/Stream have no field to carry it yet.
        # It is resolved here and discarded.

        # Step 4: normalise time to canonical seconds, then sort the whole
        # frame once so every instance's rows come out time-ordered.
        # A time_s collision is a fact about this input file, not a bug.
        try:
            frame, time_column, _, _ = _normalise_time(
                frame, time_spec.column, time_spec.unit
            )
        except ValueError as exc:
            raise AdapterRefusal(path, reason=str(exc), schema_so_far=schema) from exc
        frame = frame.sort(time_column, nulls_last=True, maintain_order=True)

        # Step 5: split into instances, logging when a null key produced an
        # unattributed group.
        key_column = schema.entity_key.column if schema.entity_key else None
        groups = _split_by_instance(frame, key_column)
        for _, attribution, part in groups:
            if attribution is Attribution.UNATTRIBUTED:
                logger.warning(
                    "%s: %d row(s) had no value for the instance key %r",
                    path,
                    part.height,
                    key_column,
                )

        # Step 6: resolve every column's stem, type and channel order once —
        # every group shares the same columns, only the rows differ.
        dictionary = self._resolve_dictionary()
        excluded = {time_column}
        if key_column is not None:
            excluded.add(key_column)

        columns = [
            name
            for name, dtype in frame.schema.items()
            if name not in excluded and dtype.is_numeric()
        ]
        roles_by_column = {role.column: role for role in roles(columns, dictionary)}
        file_stem = normalise_name(path.stem).stem or path.stem
        stems = []
        for stem, group_columns in _group_by_stem(columns, file_stem=file_stem):
            taxonomy_type = _stream_type(
                stem, roles_by_column[group_columns[0]], path=path
            )
            axis_by_column = _axes_for_group(
                group_columns,
                roles_by_column,
                array_derived=stem in parsed.array_stems,
            )
            _warn_on_duplicate_axes(stem, axis_by_column, path=path)
            ordered = _order_by_axis(group_columns, axis_by_column)
            stems.append((stem, taxonomy_type, ordered, axis_by_column))

        # Step 7: build one Stream per stem group, within every instance group.
        streams: list[Stream] = []
        for instance, attribution, part in groups:
            timestamps = part[time_column]
            for stem, taxonomy_type, ordered, axis_by_column in stems:
                streams.append(
                    Stream(
                        taxonomy_type=taxonomy_type,
                        instance=instance,
                        attribution=attribution,
                        kind=Kind.SERIES,
                        timestamps=timestamps,
                        payload=FramePayload(frame=part.select(ordered)),
                        source_path=path,
                        source_field=stem,
                        clock=Clock.UNKNOWN,
                        is_regular=time_spec.regularity.is_regular,
                        channels=[
                            Channel(name=column, axis=axis_by_column[column])
                            for column in ordered
                        ],
                    )
                )

        unmapped_prefix = f"{UNMAPPED_TAXONOMY_PREFIX}."
        unmapped_count = sum(
            1 for stream in streams if stream.taxonomy_type.startswith(unmapped_prefix)
        )
        logger.debug(
            "%s: %d stream(s) across %d instance(s)", path, len(streams), len(groups)
        )
        if unmapped_count:
            logger.warning("%s: %d stream(s) left unmapped", path, unmapped_count)

        # Step 8: a tabular file is one recording.
        yield Episode(id=path.stem, streams=streams)
