"""Verifies the MCAP adapter: one stream per topic on its own clock, the
message-type-then-topic-name taxonomy lookup, the blob-payload path for an
image topic, and that an undecodable channel is skipped rather than fatal.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import pytest
from mcap.reader import SeekingReader, make_reader
from mcap.writer import CompressionType
from mcap.writer import Writer as McapWriter
from mcap_ros2.writer import Writer as Ros2Writer
from upath import UPath

# Internal
from kalanos.analysis.adapters.mcap import McapAdapter
from kalanos.analysis.models.adapters import AdapterRefusal
from kalanos.analysis.models.domain import Clock, Kind, Stream
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, MCAP_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_TWIST_MSGDEF = """geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""

_STAMPED_MSGDEF = """std_msgs/Header header
float64 value
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


def _episode_streams() -> dict[str, Stream]:
    """Read the fixture's one episode, keyed by topic.

    Returns
    -------
    dict[str, Stream]
        Every stream, keyed by `source_field`.
    """

    [episode] = list(McapAdapter().episodes(MCAP_FIXTURE))
    return {stream.source_field: stream for stream in episode.streams}


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_contract(tmp_path):
    """Verify the adapter holds every property `check_adapter` requires."""

    decoy_path = tmp_path / "not_mcap.bin"
    decoy_path.write_bytes(b"not an mcap file")

    check_adapter(McapAdapter(), MCAP_FIXTURE, [CSV_FIXTURE, decoy_path])


def test_each_topic_becomes_one_stream():
    """Verify the fixture's three topics each become their own stream."""

    streams = _episode_streams()

    assert sorted(streams) == ["/camera/wrist/image_raw", "/cmd_vel", "/joint_states"]


def test_the_message_type_resolves_the_taxonomy_type():
    """Verify `/joint_states` resolves through its message type, not its topic."""

    streams = _episode_streams()

    assert streams["/joint_states"].taxonomy_type == "proprio.joint_position"


def test_a_topic_name_resolves_when_the_message_type_does_not():
    """Verify `/cmd_vel` resolves through its topic name's own dictionary alias."""

    streams = _episode_streams()

    assert streams["/cmd_vel"].taxonomy_type == "action.base_velocity_command"


def test_a_header_stamped_topic_keeps_the_capture_clock():
    """Verify a header-stamped topic reads `Clock.CAPTURE`, a headerless one `LOG`."""

    streams = _episode_streams()

    assert streams["/joint_states"].clock is Clock.CAPTURE
    assert streams["/cmd_vel"].clock is Clock.LOG


def test_the_capture_and_log_clocks_are_not_collapsed():
    """Verify `/joint_states`' timestamps come from `header.stamp`, not `log_time`.

    A future edit that resampled every topic onto a shared axis would collapse
    this gap back to zero.
    """

    streams = _episode_streams()
    capture_ts = streams["/joint_states"].timestamps[0]

    with MCAP_FIXTURE.open("rb") as handle:
        reader = make_reader(handle)
        summary = reader.get_summary()
        assert summary is not None and summary.statistics is not None
        origin_ns = summary.statistics.message_start_time
        _, _, first_message = next(
            entry
            for entry in reader.iter_messages()
            if entry[1].topic == "/joint_states"
        )
    log_derived_ts = (first_message.log_time - origin_ns) / 1_000_000_000

    assert log_derived_ts - capture_ts == pytest.approx(0.0005)


def test_a_partially_stamped_topic_falls_back_to_the_log_clock_entirely(tmp_path):
    """Verify one unstamped message pulls the whole topic onto `Clock.LOG`.

    A stream is on one clock or the other, never a mix: leaving the rest of
    the topic on `header.stamp` just because an earlier message happened to
    carry one would put unrelated clocks on the same axis.
    """

    path = tmp_path / "partial.mcap"
    with path.open("wb") as handle:
        writer = Ros2Writer(handle, compression=CompressionType.NONE)
        schema = writer.register_msgdef("test_msgs/msg/Stamped", _STAMPED_MSGDEF)
        stamps = [
            {"sec": 5, "nanosec": 0},
            {"sec": 0, "nanosec": 0},
            {"sec": 7, "nanosec": 0},
        ]
        for i, stamp in enumerate(stamps):
            log_time = 2_000_000_000 + i * 10_000_000
            writer.write_message(
                topic="/stamped",
                schema=schema,
                message={
                    "header": {"stamp": stamp, "frame_id": "x"},
                    "value": float(i),
                },
                log_time=log_time,
                publish_time=log_time,
            )
        writer.finish()

    [episode] = list(McapAdapter().episodes(UPath(path)))
    [stream] = episode.streams

    assert stream.clock is Clock.LOG
    assert stream.timestamps.to_list() == pytest.approx([0.0, 0.01, 0.02])


def test_an_image_topic_carries_timestamps_and_no_payload():
    """Verify the image topic is a real Stream with timestamps but no channels.

    Its taxonomy type is `unmapped.*`, since no message-type or topic-name
    alias claims it — proving the fallback lands it in the report rather
    than dropping it.
    """

    stream = _episode_streams()["/camera/wrist/image_raw"]

    assert stream.kind is Kind.IMAGE
    assert stream.payload is None
    assert stream.channels == []
    assert len(stream.timestamps) == 2
    assert stream.taxonomy_type.startswith("unmapped.")


def test_a_joint_state_topic_keeps_velocity_and_effort():
    """Verify all three JointState quantities become channels, and `name` does not."""

    channels = [c.name for c in _episode_streams()["/joint_states"].channels]

    assert channels == [
        "position_0",
        "position_1",
        "velocity_0",
        "velocity_1",
        "effort_0",
        "effort_1",
    ]


def test_a_nested_message_flattens_to_axis_channels():
    """Verify Twist's nested Vector3 fields flatten to six dotted axis channels."""

    channels = _episode_streams()["/cmd_vel"].channels

    assert [c.name for c in channels] == [
        "linear.x",
        "linear.y",
        "linear.z",
        "angular.x",
        "angular.y",
        "angular.z",
    ]
    assert next(c for c in channels if c.name == "linear.x").axis == "x"


def test_a_non_ros2_channel_is_skipped_and_the_rest_still_read(tmp_path):
    """Verify a channel no factory can decode is skipped rather than fatal."""

    path = tmp_path / "mixed.mcap"
    with path.open("wb") as handle:
        writer = Ros2Writer(handle, compression=CompressionType.NONE)
        schema = writer.register_msgdef("geometry_msgs/msg/Twist", _TWIST_MSGDEF)
        writer.write_message(
            topic="/cmd_vel",
            schema=schema,
            message={
                "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            log_time=1_000_000_000,
            publish_time=1_000_000_000,
        )
        raw_schema_id = writer._writer.register_schema(
            "my/JsonThing", "jsonschema", b'{"type": "object"}'
        )
        raw_channel_id = writer._writer.register_channel(
            topic="/diagnostics", message_encoding="json", schema_id=raw_schema_id
        )
        writer._writer.add_message(
            channel_id=raw_channel_id,
            log_time=1_000_000_000,
            publish_time=1_000_000_000,
            sequence=0,
            data=b'{"ok": true}',
        )
        writer.finish()

    [episode] = list(McapAdapter().episodes(UPath(path)))

    assert [s.source_field for s in episode.streams] == ["/cmd_vel"]


def test_a_file_with_no_readable_channel_is_refused(tmp_path):
    """Verify a file holding only an undecodable channel is refused, not misread."""

    path = tmp_path / "undecodable.mcap"
    with path.open("wb") as handle:
        writer = McapWriter(handle, compression=CompressionType.NONE)
        writer.start()
        schema_id = writer.register_schema(
            "my/JsonThing", "jsonschema", b'{"type": "object"}'
        )
        channel_id = writer.register_channel(
            topic="/diagnostics", message_encoding="json", schema_id=schema_id
        )
        writer.add_message(
            channel_id=channel_id, log_time=0, publish_time=0, sequence=0, data=b"{}"
        )
        writer.finish()

    with pytest.raises(AdapterRefusal):
        list(McapAdapter().episodes(UPath(path)))


def test_describe_does_not_decode_a_message(monkeypatch):
    """Verify describe() never decodes a message, only reads the summary tables."""

    monkeypatch.setattr(
        SeekingReader,
        "iter_decoded_messages",
        lambda self, *args, **kwargs: pytest.fail(
            "iter_decoded_messages was called by describe()"
        ),
    )

    info = McapAdapter().describe(MCAP_FIXTURE)

    assert info.episode_count == 1
