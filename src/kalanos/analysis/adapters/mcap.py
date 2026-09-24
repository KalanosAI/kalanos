"""MCAP: a container for timestamped messages from many topics, in one file.

Specification: https://mcap.dev/spec
Each file is read as one episode, and each topic as one stream.

TODO: decode encodings beyond CDR-encoded ROS 2 messages.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.entry_points import MissingDependency
from kalanos.analysis.inference.regularity import entity_split_gaps, regularity
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


# Dependencies added by the mcap adapter
# Optional better error handling
try:
    from mcap.reader import make_reader
    from mcap.summary import Summary
    from mcap_ros2.decoder import DecoderFactory
except ImportError as exc:
    raise MissingDependency("kalanos", "mcap") from exc


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# MCAP announces itself with a magic number, high confidence.
_MAGIC = b"\x89MCAP0\r\n"
_CONFIDENCE = 0.9

# Nanoseconds to the canonical second.
_NS_PER_S = 1_000_000_000

# Message types whose payload is a blob rather than a series of numbers.
# Their scalar fields — height, width, step — describe the blob, not a signal,
# so the stream carries timestamps and no channels at all.
_BLOB_KINDS = {
    "sensor_msgs/msg/Image": Kind.IMAGE,
    "sensor_msgs/msg/CompressedImage": Kind.IMAGE,
    "sensor_msgs/msg/PointCloud2": Kind.POINTS,
}

# What a std_msgs/Header looks like once decoded. The stamp is the stream's clock,
# not one of its channels, so a field of this shape is consumed rather than flattened.
_HEADER_SLOTS = ("stamp", "frame_id")

# The only encoding pair this adapter can decode:
# - CDR is the wire format ROS 2 messages are serialized with;
# - ros2msg is the IDL dialect describing their layout.
_CDR = "cdr"
_ROS2MSG = "ros2msg"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _readable_channels(summary: Summary) -> dict[str, str]:
    """Map every topic this adapter can decode to its message type.

    Everything else — a channel not encoded as CDR, or backed by a schema not
    encoded as ros2msg — is dropped and logged at debug:
    a protobuf topic sitting beside a ROS 2 one is ordinary, not a fault.
    """

    readable: dict[str, str] = {}
    for channel in summary.channels.values():
        schema = summary.schemas.get(channel.schema_id)
        if (
            channel.message_encoding != _CDR
            or schema is None
            or schema.encoding != _ROS2MSG
        ):
            logger.debug(
                "%r: message_encoding=%r schema_encoding=%r not decodable, skipping",
                channel.topic,
                channel.message_encoding,
                schema.encoding if schema else None,
            )
            continue
        readable[channel.topic] = schema.name
    return readable


def _header_stamp_ns(decoded: Any) -> int | None:
    """Read a decoded message's `header.stamp`, in nanoseconds.

    Returns
    -------
    int or None
        `sec * 1e9 + nanosec`, or `None` when the message carries no
        `std_msgs/Header` as its first field, or that stamp is exactly
        zero — ROS's way of leaving a stamp unset.
    """

    if not decoded.__slots__ or decoded.__slots__[0] != "header":
        return None
    header = decoded.header
    if tuple(getattr(header, "__slots__", ())) != _HEADER_SLOTS:
        return None
    nanoseconds = header.stamp.sec * _NS_PER_S + header.stamp.nanosec
    return nanoseconds or None


def _flatten(decoded: Any, prefix: str = "") -> dict[str, float] | None:
    """Flatten a decoded ROS 2 message's numeric fields into named columns.

    Recurses into a nested field, naming each leaf by its dotted path;
    a `header` field is consumed as the stream's clock rather than flattened,
    and a string, byte or non-numeric array field carries nothing to flatten.

    Parameters
    ----------
    decoded : Any
        A decoded ROS 2 message, or one of its nested fields.
    prefix : str
        The dotted path from the message root to `decoded`.

    Returns
    -------
    dict[str, float] or None
        Every numeric field found, in declaration order;
        `None` when nothing numeric survived.
    """

    row: dict[str, float] = {}
    for name in decoded.__slots__:
        value = getattr(decoded, name)
        path = f"{prefix}{name}"
        nested_slots = getattr(value, "__slots__", None)

        if name == "header" and tuple(nested_slots or ()) == _HEADER_SLOTS:
            continue
        if isinstance(value, bool | int | float):
            row[path] = float(value)
        elif isinstance(value, list) and all(
            isinstance(item, bool | int | float) for item in value
        ):
            row.update(
                {f"{path}_{index}": float(item) for index, item in enumerate(value)}
            )
        elif nested_slots is not None:
            nested = _flatten(value, prefix=f"{path}.")
            if nested is not None:
                row.update(nested)

    return row or None


def _channels_for(names: list[str], dictionary: Dictionary) -> list[Channel]:
    """Resolve one topic's flattened row keys into channels, indexed by column."""

    axis_by_name = {role.column: role.axis for role in roles(names, dictionary)}
    return [Channel(name=name, axis=axis_by_name[name]) for name in names]


def _taxonomy_type(schema_name: str, topic: str, dictionary: Dictionary) -> str:
    """Resolve a topic's type from its message type, falling back to its topic name."""

    by_schema, by_topic = roles([schema_name, topic], dictionary)
    return (
        by_schema.taxonomy_type
        or by_topic.taxonomy_type
        or f"{UNMAPPED_TAXONOMY_PREFIX}.{topic}"
    )


def _topic_stream(
    topic: str,
    schema_name: str,
    stamps_ns: list[int],
    from_header: bool,
    rows: list[dict[str, float]],
    origin_ns: int,
    dictionary: Dictionary,
    source_path: UPath,
) -> Stream:
    """Build one topic's Stream, on its own clock rather than a shared axis.

    Parameters
    ----------
    topic : str
        The topic this stream was read from.
    schema_name : str
        The topic's message type.
    stamps_ns : list[int]
        Each message's chosen timestamp, in nanoseconds, sorted ascending.
    from_header : bool
        Whether `stamps_ns` came from `header.stamp` rather than `log_time`.
    rows : list[dict[str, float]]
        Each message's flattened row, aligned index-for-index with
        `stamps_ns`; empty for a blob-payload topic, which carries
        timestamps but no channels.
    origin_ns : int
        The file's earliest message time, subtracted so the timestamps stay
        within float64 precision while every topic keeps a common zero.
    dictionary : Dictionary
        The taxonomy to resolve `schema_name` and `topic` against.
    source_path : UPath
        The file this stream was read from.

    Returns
    -------
    Stream
        One stream, its `kind` a blob kind when `schema_name` names one.
    """

    timestamps = pl.Series(
        [(ns - origin_ns) / _NS_PER_S for ns in stamps_ns], dtype=pl.Float64
    )
    is_regular = regularity(entity_split_gaps([timestamps.to_list()])).is_regular
    clock = Clock.CAPTURE if from_header else Clock.LOG

    blob_kind = _BLOB_KINDS.get(schema_name)
    if blob_kind is not None:
        kind, payload, channels = blob_kind, None, []
    else:
        names = list(rows[0].keys())
        payload = FramePayload(frame=pl.DataFrame(rows).cast(pl.Float64))
        channels = _channels_for(names, dictionary)
        kind = Kind.SERIES

    return Stream(
        taxonomy_type=_taxonomy_type(schema_name, topic, dictionary),
        kind=kind,
        timestamps=timestamps,
        payload=payload,
        source_path=source_path,
        source_field=topic,
        clock=clock,
        is_regular=is_regular,
        channels=channels,
    )


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@adapter
class McapAdapter:
    """Reads one MCAP file as one episode, one stream per topic on its own clock."""

    name = "mcap"

    def __init__(self, *, dictionary: Dictionary | None = None) -> None:
        self._dictionary = dictionary

    def _resolve_dictionary(self) -> Dictionary:
        """Return the dictionary to resolve topic and message-type names against.

        Loads the packaged default lazily, on first use, so constructing this
        adapter during plugin discovery never reads the packaged YAML.
        """

        if self._dictionary is None:
            self._dictionary = load_default_dictionary()
        return self._dictionary

    def detect(self, path: UPath) -> float:
        """Bid on any file starting with the MCAP magic number.

        Deliberately does not check whether any channel is decodable:
        a protobuf-only MCAP is still an MCAP this adapter should claim
        and then refuse by name, rather than leave unclaimed.

        Returns
        -------
        float
            Confidence in `[0.0, 1.0]` that `path` is an MCAP file.
        """

        try:
            if path.is_dir():
                return 0.0
            with path.open("rb") as handle:
                return _CONFIDENCE if handle.read(len(_MAGIC)) == _MAGIC else 0.0
        except Exception:
            return 0.0

    def describe(self, path: UPath) -> DatasetInfo:
        """Report that the file holds one episode, without decoding a message.

        MCAP declares no nominal rate; anything derived from the message
        statistics would be inference, not a declared fact.

        Returns
        -------
        DatasetInfo
            The file's one episode, with no nominal rate or robot type.

        Raises
        ------
        AdapterRefusal
            If `path` cannot be opened as MCAP,
            or holds no channel this adapter can decode.
        """

        try:
            with path.open("rb") as handle:
                summary = make_reader(handle).get_summary()
                if summary is None or not _readable_channels(summary):
                    raise AdapterRefusal(path, "no channel this adapter can decode")
                return DatasetInfo(
                    adapter=self.name,
                    path=path,
                    episode_count=1,
                    nominal_rate_hz=None,
                    robot_type=None,
                )
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(path, f"could not read {path}: {exc}") from exc

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield the file's one episode, its streams built from readable topics.

        Parameters
        ----------
        path : UPath
            The file to read.
        sample : int or None
            When `0` or less, yields nothing; otherwise this still yields the
            file's one episode, since one MCAP file is always one episode.

        Yields
        ------
        Episode
            The file's one episode.

        Raises
        ------
        AdapterRefusal
            If the file could not be read, held no channel this adapter can decode,
            or no topic yielded a stream.
        """

        if sample is not None and sample <= 0:
            return

        try:
            with path.open("rb") as handle:
                reader = make_reader(handle, decoder_factories=[DecoderFactory()])
                summary = reader.get_summary()
                if summary is None:
                    raise AdapterRefusal(path, "no summary section found")

                readable = _readable_channels(summary)
                if not readable:
                    raise AdapterRefusal(path, "no channel this adapter can decode")
                if summary.statistics is None:
                    raise AdapterRefusal(path, "no statistics section found")

                dictionary = self._resolve_dictionary()
                origin_ns = summary.statistics.message_start_time

                header_ns_by_topic: dict[str, list[int | None]] = defaultdict(list)
                log_time_by_topic: dict[str, list[int]] = defaultdict(list)
                rows_by_topic: dict[str, list[dict[str, float] | None]] = defaultdict(
                    list
                )

                for _, channel, message, decoded in reader.iter_decoded_messages(
                    topics=list(readable)
                ):
                    topic = channel.topic
                    header_ns_by_topic[topic].append(_header_stamp_ns(decoded))
                    log_time_by_topic[topic].append(message.log_time)
                    if readable[topic] not in _BLOB_KINDS:
                        rows_by_topic[topic].append(_flatten(decoded))

                streams = []
                for topic in sorted(header_ns_by_topic):
                    schema_name = readable[topic]
                    header_ns_list = header_ns_by_topic[topic]
                    # A stream is on one clock or the other, never a mix:
                    # a single unset header stamp — common while a driver comes up —
                    # must not leave the rest of the topic on the other clock silently.
                    from_header = all(ns is not None for ns in header_ns_list)
                    raw_stamps = (
                        [ns for ns in header_ns_list if ns is not None]
                        if from_header
                        else log_time_by_topic[topic]
                    )
                    rows: list[dict[str, float]]

                    if schema_name in _BLOB_KINDS:
                        stamps_ns = sorted(raw_stamps)
                        rows = []
                    else:
                        paired = sorted(
                            zip(raw_stamps, rows_by_topic[topic], strict=True),
                            key=lambda pair: pair[0],
                        )
                        stamps_ns = [stamp for stamp, _ in paired]
                        raw_rows = [row for _, row in paired]

                        first_row = raw_rows[0]
                        if not first_row:
                            logger.warning(
                                "%s: topic %r flattened to nothing, skipping",
                                path,
                                topic,
                            )
                            continue
                        # Channels stable across every message, in first-message order:
                        # a message-to-message key mismatch — say, `effort` arriving
                        # empty before a controller warms up — drops just that channel
                        # rather than corrupting the frame with nulls `check_adapter`
                        # cannot see, or the whole topic with it.
                        stable_keys = [
                            key
                            for key in first_row
                            if all(row is not None and key in row for row in raw_rows)
                        ]
                        if not stable_keys:
                            logger.warning(
                                "%s: topic %r flattened to nothing, skipping",
                                path,
                                topic,
                            )
                            continue
                        if len(stable_keys) != len(first_row):
                            logger.warning(
                                "%s: topic %r has a channel whose width changed "
                                "across messages; keeping only the channels every "
                                "message carries",
                                path,
                                topic,
                            )
                        rows = [
                            {key: row[key] for key in stable_keys}
                            for row in raw_rows
                            if row is not None
                        ]

                    streams.append(
                        _topic_stream(
                            topic,
                            schema_name,
                            stamps_ns,
                            from_header,
                            rows,
                            origin_ns,
                            dictionary,
                            path,
                        )
                    )

                if not streams:
                    raise AdapterRefusal(path, "no topic yielded a stream")
                yield Episode(id=path.stem, streams=streams)
        except AdapterRefusal:
            raise
        except Exception as exc:
            raise AdapterRefusal(path, f"could not read {path}: {exc}") from exc
