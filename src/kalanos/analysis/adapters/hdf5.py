"""HDF5 has no episode convention, so boundaries are inferred from group structure.

Treating `shape[0]` as the time axis is this adapter's assumption, not a format
guarantee.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from collections import Counter, deque
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import NamedTuple

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.entry_points import MissingDependency
from kalanos.analysis.inference.roles import roles
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.domain import (
    UNMAPPED_TAXONOMY_PREFIX,
    Channel,
    Clock,
    Episode,
    FramePayload,
    Kind,
    Stream,
)
from kalanos.assets.dictionary import load_default_dictionary


# Dependencies added by the hdf5 adapter
# Optional better error handling
try:
    import h5py
except ImportError as exc:
    raise MissingDependency("kalanos", "hdf5") from exc


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


class _Layout(NamedTuple):
    """Where a file's episode boundaries are, and how deep to read inside one.

    Attributes
    ----------
    container : h5py.Group
        The group holding the episodes, checked for a declared rate.
    episodes : list[h5py.Group]
        One group per episode, name-sorted.
    whole_file : bool
        Whether the file yielded no sibling split and is being read as a
        single episode, which is read to any depth rather than one level.
    """

    container: h5py.Group
    episodes: list[h5py.Group]
    whole_file: bool


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# A structural guess other HDF5-based formats could coincidentally satisfy,
# so this must never outrank a format-specific reader.
# docs/ADAPTERS.md puts that band at 0.3.
_CONFIDENCE = 0.3

# A group needs at least this many dataset-bearing sibling groups
# before its children are read as episodes, rather than as one more level to search.
_MIN_SIBLING_GROUPS = 2

_RATE_ATTRS = ("fps", "control_freq", "rate_hz", "frequency_hz")

# Assumed when no rate is declared, so every stream still gets a timebase.
# This is the timebase itself, not a fact learned from the file.
_FALLBACK_RATE_HZ = 1.0

# Dataset dtypes read as a channel: float, signed and unsigned int, bool.
# Excludes strings and opaque objects, such as a stashed simulator model.
_NUMERIC_KINDS = frozenset("fiub")

# Beyond its leading (assumed-time) axis, a dataset ranked above this is
# image-shaped rather than a scalar or vector channel.
_MAX_CHANNEL_RANK = 2


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@contextmanager
def _open_store(path: UPath) -> Generator[h5py.File, None, None]:
    """Open `path` as an HDF5 store, from a UPath-generic binary handle.

    Yields
    ------
    h5py.File
        The open store, closed along with the underlying handle on exit.
    """

    with path.open("rb") as handle, h5py.File(handle, "r") as store:
        yield store


def _direct_datasets(group: h5py.Group) -> list[tuple[str, h5py.Dataset]]:
    """List the datasets directly inside `group`, as `(name, dataset)`."""

    return [
        (name, item) for name, item in group.items() if isinstance(item, h5py.Dataset)
    ]


def _shallow_datasets(group: h5py.Group) -> list[tuple[str, h5py.Dataset]]:
    """List `group`'s own datasets, plus those directly under its child groups.

    Keyed by leaf name, like `_deep_datasets`: two datasets sharing a leaf name
    under different subgroups arrive as two entries with the same key.
    Rare, harmless — each still becomes its own Stream — and deliberately
    not deduplicated.
    """

    found = list(_direct_datasets(group))
    for item in group.values():
        if isinstance(item, h5py.Group):
            found.extend(_direct_datasets(item))
    return found


def _deep_datasets(group: h5py.Group) -> list[tuple[str, h5py.Dataset]]:
    """List every dataset reachable from `group`, at any depth.

    Breadth-first and cycle-guarded: the HDF5 group hierarchy is a directed graph
    that may contain a hard link back to an ancestor,
    so an unguarded walk would not terminate.
    """

    found: list[tuple[str, h5py.Dataset]] = []
    queue: deque[h5py.Group] = deque([group])
    visited = {group.id}
    while queue:
        current = queue.popleft()
        for name, item in current.items():
            if isinstance(item, h5py.Dataset):
                found.append((name, item))
            elif item.id not in visited:
                visited.add(item.id)
                queue.append(item)
    return found


def _find_episodes(root: h5py.Group) -> _Layout | None:
    """Find where in `root`'s tree the episode boundaries are.

    Breadth-first from `root`: the shallowest group with at least
    `_MIN_SIBLING_GROUPS` child groups that each hold a dataset directly
    inside becomes the episode container, and those children are the
    episodes. When no such split exists anywhere, the file is one episode
    if it holds a dataset at any depth, and unreadable if it holds none.

    Returns
    -------
    _Layout or None
        The layout found, or `None` when there is nothing to read.
    """

    queue: deque[h5py.Group] = deque([root])
    visited = {root.id}
    while queue:
        group = queue.popleft()
        children = [item for item in group.values() if isinstance(item, h5py.Group)]
        qualifying = sorted(
            (child for child in children if _direct_datasets(child)),
            key=lambda candidate: str(candidate.name),
        )
        if len(qualifying) >= _MIN_SIBLING_GROUPS:
            return _Layout(group, qualifying, whole_file=False)
        for child in children:
            if child.id not in visited:
                visited.add(child.id)
                queue.append(child)

    if not _deep_datasets(root):
        return None
    return _Layout(root, [root], whole_file=True)


def _declared_rate(container: h5py.Group, root: h5py.Group) -> float | None:
    """Read the first rate attribute declared on `container`, else on `root`."""

    for group in (container, root):
        for attr_name in _RATE_ATTRS:
            value = group.attrs.get(attr_name)
            if value is not None:
                return float(value)
    return None


def _is_channel_shaped(dataset: h5py.Dataset) -> bool:
    """Say whether a dataset is a numeric series this adapter reads as channels."""

    return (
        dataset.dtype.kind in _NUMERIC_KINDS and 1 <= dataset.ndim <= _MAX_CHANNEL_RANK
    )


def _channels_for(key: str, width: int, dictionary: Dictionary) -> list[Channel]:
    """Resolve one dataset's channels, named after the key and indexed by column."""

    names = [key] if width == 1 else [f"{key}_{index}" for index in range(width)]
    axis_by_name = {role.column: role.axis for role in roles(names, dictionary)}
    return [Channel(name=name, axis=axis_by_name[name]) for name in names]


def _taxonomy_type(key: str, dictionary: Dictionary) -> str:
    """Resolve one dataset key's taxonomy type, or type it `unmapped.<key>`."""

    [role] = roles([key], dictionary)
    return role.taxonomy_type or f"{UNMAPPED_TAXONOMY_PREFIX}.{key}"


def _series_stream(
    key: str,
    dataset: h5py.Dataset,
    timestamps: pl.Series,
    dictionary: Dictionary,
    source_path: UPath,
) -> Stream:
    """Build one series Stream for `key`, one column per channel of `dataset`.

    Parameters
    ----------
    key : str
        The dataset's own name.
    dataset : h5py.Dataset
        The dataset to read, eagerly.
    timestamps : pl.Series
        The episode's own timestamps, shared across every one of its streams.
    dictionary : Dictionary
        The taxonomy to resolve `key` and its channel names against.
    source_path : UPath
        The file this stream was read from.

    Returns
    -------
    Stream
        One series stream, its columns cast to `pl.Float64`.
    """

    values = dataset[()]
    width = 1 if values.ndim == 1 else values.shape[1]
    channels = _channels_for(key, width, dictionary)

    if values.ndim == 1:
        payload_frame = pl.DataFrame({channels[0].name: values}).cast(pl.Float64)
    else:
        payload_frame = pl.DataFrame(
            {channel.name: values[:, index] for index, channel in enumerate(channels)}
        ).cast(pl.Float64)

    return Stream(
        taxonomy_type=_taxonomy_type(key, dictionary),
        kind=Kind.SERIES,
        timestamps=timestamps,
        payload=FramePayload(frame=payload_frame),
        source_path=source_path,
        source_field=key,
        clock=Clock.UNKNOWN,
        # The timestamps are synthesised from a rate, so declaring regularity
        # would restate the rate rather than describe the capture.
        is_regular=False,
        channels=channels,
    )


def _episode_streams(
    group: h5py.Group,
    dictionary: Dictionary,
    source_path: UPath,
    rate_hz: float,
    *,
    deep: bool,
) -> list[Stream]:
    """Build every Stream in one episode group, timed from a rate-synthesised timebase.

    Raises
    ------
    ValueError
        If the group holds no channel-shaped dataset at all.
    """

    reachable = _deep_datasets(group) if deep else _shallow_datasets(group)
    entries = []
    for key, dataset in reachable:
        if _is_channel_shaped(dataset):
            entries.append((key, dataset))
        else:
            logger.debug("%s: %r is not channel-shaped, skipping", source_path, key)
    if not entries:
        raise ValueError("no channel-shaped dataset found")

    # The modal length, not the alphabetically-first entry's: a non-per-step
    # array (a calibration matrix, a joint-limit vector) can sort before the
    # real per-step datasets and must not hijack the episode's timebase.
    [(length, _)] = Counter(dataset.shape[0] for _, dataset in entries).most_common(1)
    timestamps = pl.Series(
        [index / rate_hz for index in range(length)], dtype=pl.Float64
    )

    streams = []
    for key, dataset in entries:
        if dataset.shape[0] != length:
            logger.warning(
                "%s: %r has %d rows, expected %d, skipping",
                source_path,
                key,
                dataset.shape[0],
                length,
            )
            continue
        streams.append(
            _series_stream(key, dataset, timestamps, dictionary, source_path)
        )
    return streams


def _episode_id(group: h5py.Group) -> str:
    """Name an episode after its own HDF5 path, with the root group called `root`."""

    stripped = str(group.name).strip("/")
    return stripped.replace("/", "_") if stripped else "root"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@adapter
class Hdf5Adapter:
    """Reads any HDF5 file, discovering episode boundaries from its group structure."""

    name = "hdf5"

    def __init__(self, *, dictionary: Dictionary | None = None) -> None:
        self._dictionary = dictionary

    def _resolve_dictionary(self) -> Dictionary:
        """Return the dictionary to resolve dataset names against.

        Loads the packaged default lazily, on first use, so constructing this
        adapter during plugin discovery never reads the packaged YAML.
        """

        if self._dictionary is None:
            self._dictionary = load_default_dictionary()
        return self._dictionary

    def detect(self, path: UPath) -> float:
        """Bid low on any file whose group structure yields an episode layout.

        Never raises and never hangs: anything that fails to open as HDF5, or
        holds no readable structure, bids zero, and the walk behind
        `_find_episodes` is cycle-guarded.

        Returns
        -------
        float
            `_CONFIDENCE`, or zero.
        """

        try:
            if path.is_dir():
                return 0.0
            with _open_store(path) as store:
                return _CONFIDENCE if _find_episodes(store) is not None else 0.0
        except Exception:
            return 0.0

    def describe(self, path: UPath) -> DatasetInfo:
        """Report the episode count and declared rate, without reading any episode.

        Raises
        ------
        AdapterRefusal
            If `path` cannot be opened as HDF5, or holds no episode structure.
        """

        try:
            with _open_store(path) as store:
                layout = _find_episodes(store)
                if layout is None:
                    raise AdapterRefusal(path, "no episode-shaped group found")
                return DatasetInfo(
                    adapter=self.name,
                    path=path,
                    episode_count=len(layout.episodes),
                    nominal_rate_hz=_declared_rate(layout.container, store),
                    robot_type=None,
                )
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(path, f"could not read {path}: {exc}") from exc

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield one Episode per discovered episode group, lazily.

        Parameters
        ----------
        path : UPath
            The file to read.
        sample : int or None
            When given, stop after this many episodes.

        Yields
        ------
        Episode
            One recording, its streams drawn from that group's own datasets.

        Raises
        ------
        AdapterRefusal
            If the file could not be read,
            or an episode group holds no channel-shaped dataset.
        """

        if sample is not None and sample <= 0:
            return

        try:
            with _open_store(path) as store:
                layout = _find_episodes(store)
                if layout is None:
                    raise AdapterRefusal(path, "no episode-shaped group found")
                dictionary = self._resolve_dictionary()
                rate_hz = _declared_rate(layout.container, store) or _FALLBACK_RATE_HZ
                groups = layout.episodes if sample is None else layout.episodes[:sample]

                for group in groups:
                    try:
                        streams = _episode_streams(
                            group,
                            dictionary,
                            path,
                            rate_hz,
                            deep=layout.whole_file,
                        )
                    except Exception as exc:
                        raise AdapterRefusal(path, f"{group.name}: {exc}") from exc
                    yield Episode(id=_episode_id(group), streams=streams)
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(path, f"could not read {path}: {exc}") from exc
