"""Verifies VideoPayload: the missing-decoder path degrades, and decoding works."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import random

# External
import polars as pl
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from upath import UPath

# Internal
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.video import (
    DecodeFailed,
    DecodeLimitReached,
    DecoderUnavailable,
    VideoPayload,
)
from kalanos.analysis.metrics import registry
from kalanos.analysis.models.domain import Clock, Kind, Stream, TimestampDtype
from kalanos.analysis.models.metrics import (
    Family,
    Level,
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.reporting.assemble import grade_stream
from kalanos.assets.policy import load_default_policy

# Local
from helpers import LEROBOT_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


LEROBOT_VIDEO = (
    LEROBOT_FIXTURE / "videos" / "observation.images.up" / "chunk-000" / "file-000.mp4"
)

# The ways a vision metric reads a video payload.
READS = [
    lambda payload: payload.count_frames(),
    lambda payload: list(payload.gray_windows([(0, 2)], 64)),
]
READ_IDS = ["count_frames", "gray_windows"]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _windows(payload: VideoPayload, windows, size: int) -> list:
    """Group what `gray_windows` yields into one list of frames per window."""

    grouped: list[list] = [[] for _ in windows]
    for position, frame, _native, _pts in payload.gray_windows(windows, size):
        grouped[position].append(frame)
    return grouped


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_fetch_without_the_decoder_raises_naming_the_extra(monkeypatch):
    """Verify a missing decoder raises DecoderUnavailable naming the install extra."""

    monkeypatch.setattr(video, "_load_av", lambda: None)
    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=0.0, end_s=8 / 30)

    with pytest.raises(DecoderUnavailable, match=r"kalanos\[video\]"):
        payload.fetch()


def test_a_frame_metric_degrades_to_not_applicable(monkeypatch):
    """Verify a frame metric degrades rather than fails when the decoder is missing.

    The stream's own timing metrics must still run.
    """

    monkeypatch.setattr(video, "_load_av", lambda: None)
    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.STREAM, family=Family.VISION)
    def stub_frame_metric(ctx: StreamContext) -> MetricResult:
        assert ctx.payload is not None
        try:
            ctx.payload.fetch()
        except DecoderUnavailable as exc:
            return MetricResult(
                value=None,
                unit=None,
                status=MetricStatus.NOT_APPLICABLE,
                evidence={"reason": str(exc)},
            )
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    stream = Stream(
        taxonomy_type="unmapped.observation.images.up",
        kind=Kind.VIDEO,
        timestamps=pl.Series("time_s", [0.0, 1 / 30]),
        payload=VideoPayload(
            path=LEROBOT_VIDEO, frame_count=2, start_s=0.0, end_s=2 / 30
        ),
        source_path=LEROBOT_VIDEO,
        timestamp_dtype=TimestampDtype.FLOAT64,
        clock=Clock.UNKNOWN,
        is_regular=True,
        channels=[],
    )

    graded, _findings = grade_stream(
        stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
    )

    assert graded.metrics["stub_frame_metric"].status == MetricStatus.NOT_APPLICABLE
    assert graded.metrics["effective_hz"].value is None
    assert graded.metrics["recorded_hz"].value is not None


def test_fetch_decodes_the_fixture():
    """Verify fetch() decodes the fixture's mp4, real PyAV installed."""

    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=0.0, end_s=8 / 30)

    frames = payload.fetch()

    assert len(frames) == payload.frame_count


def test_fetch_seeks_past_the_first_segment_for_a_later_one():
    """Verify a payload starting mid-file seeks correctly rather than reading from 0.

    The fixture's second episode is the one real case that exercises this:
    its segment starts at frame 8 of a 14-frame mp4 shared with episode 1.
    """

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=6, start_s=8 / 30, end_s=14 / 30
    )

    frames = payload.fetch()

    assert len(frames) == payload.frame_count


@pytest.mark.parametrize(
    ("start_s", "end_s", "frame_count"), [(0.0, 8 / 30, 8), (8 / 30, 14 / 30, 6)]
)
def test_sampling_and_counting_read_each_fixture_segment(start_s, end_s, frame_count):
    """Verify both segments sample as small gray frames and count their own packets."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=frame_count, start_s=start_s, end_s=end_s
    )

    frames = [
        frame
        for window in _windows(payload, [(0, 1), (frame_count - 1, frame_count)], 128)
        for frame in window
    ]

    assert [(frame.shape, str(frame.dtype)) for frame in frames] == [
        ((128, 128), "uint8")
    ] * 2
    assert payload.count_frames() == frame_count


@pytest.mark.parametrize(
    ("start_s", "end_s", "frame_count"), [(0.0, 8 / 30, 8), (8 / 30, 14 / 30, 6)]
)
def test_gray_windows_read_consecutive_frames_of_each_segment(
    start_s, end_s, frame_count
):
    """Verify each window holds its own frames, and one window spans the segment."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=frame_count, start_s=start_s, end_s=end_s
    )

    head, tail = _windows(payload, [(0, 3), (frame_count - 3, frame_count)], 64)
    (whole,) = _windows(payload, [(0, frame_count)], 64)

    assert [(frame.shape, str(frame.dtype)) for frame in head + tail] == [
        ((64, 64), "uint8")
    ] * 6
    assert len(whole) == frame_count


def test_gray_windows_return_native_luminance_only_for_listed_frames():
    """Verify only the frames named in `native` come back at full size."""

    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=0.0, end_s=8 / 30)

    natives = [
        full
        for _position, _frame, full, _pts in payload.gray_windows(
            [(0, 8)], 64, native=frozenset({0, 5})
        )
    ]

    assert [
        None if full is None else (full.shape, str(full.dtype)) for full in natives
    ] == [((32, 32), "uint8") if index in (0, 5) else None for index in range(8)]


def test_gray_windows_report_presentation_times():
    """Verify each frame comes back with its container presentation time."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=6, start_s=8 / 30, end_s=14 / 30
    )

    times = [pts for *_frame, pts in payload.gray_windows([(0, 6)], 16)]

    assert times == pytest.approx([(8 + k) / 30 for k in range(6)])


@pytest.mark.parametrize(
    "caps",
    [{"max_decode_frames": 1}, {"max_pixels": 32 * 32 - 1}],
    ids=["max_decode_frames", "max_pixels"],
)
def test_gray_windows_stop_at_the_decode_and_pixel_caps(caps):
    """Verify a read past either cap raises DecodeLimitReached instead of reading on."""

    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=0.0, end_s=8 / 30)

    with pytest.raises(DecodeLimitReached):
        list(payload.gray_windows([(0, 3)], 16, native=frozenset({0}), **caps))


def test_rgb_frame_reads_one_native_frame():
    """Verify rgb_frame decodes the one frame fetch would return at that position."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=6, start_s=8 / 30, end_s=14 / 30
    )

    frame = payload.rgb_frame(2)

    assert frame.shape == (32, 32, 3)
    assert (frame == payload.fetch()[2]).all()


def test_rgb_frames_reads_native_frames_without_a_size():
    """Verify rgb_frames(None) decodes the frames fetch returns, unresized."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=6, start_s=8 / 30, end_s=14 / 30
    )

    frames = payload.rgb_frames(None)

    assert [frame.shape for frame in frames] == [(32, 32, 3)] * 6
    for frame, fetched in zip(frames, payload.fetch(), strict=True):
        assert (frame == fetched).all()


def test_native_luminance_drops_row_padding_and_reads_deep_formats_as_gray():
    """Verify a padded 8-bit Y plane comes back exact, a 10-bit frame as 8-bit gray."""

    av = pytest.importorskip("av")
    numpy = pytest.importorskip("numpy")
    # 40 rows of Y over 20 of chroma; a width of 50 pads each Y row to 64 bytes.
    planes = numpy.random.default_rng(42).integers(0, 256, (60, 50), numpy.uint8)
    frame = av.VideoFrame.from_ndarray(planes, format="yuv420p")

    deep = video._luminance(frame.reformat(format="yuv420p10le"))

    assert (video._luminance(frame) == planes[:40]).all()
    assert (deep.shape, str(deep.dtype)) == ((40, 50), "uint8")


def test_counting_a_segment_past_the_end_of_the_file_finds_fewer_frames():
    """Verify a segment running past the end of the mp4 counts only what it holds."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=8, start_s=8 / 30, end_s=16 / 30
    )

    assert payload.count_frames() == 6


@pytest.fixture
def remote_video():
    """Copy the fixture mp4 into memory, closing cached remote handles afterwards."""

    remote = UPath("memory://videos/file-000.mp4")
    remote.write_bytes(LEROBOT_VIDEO.read_bytes())
    yield remote
    video.close_remote_handles()
    remote.unlink()


def test_sampling_and_counting_read_a_remote_path(remote_video):
    """Verify a non-local path is opened through its own file object."""

    payload = VideoPayload(path=remote_video, frame_count=8, start_s=0.0, end_s=8 / 30)

    assert payload.count_frames() == 8
    assert [len(window) for window in _windows(payload, [(0, 8)], 16)] == [8]


@pytest.mark.parametrize("read", READS, ids=READ_IDS)
def test_reading_without_the_decoder_raises_naming_the_extra(monkeypatch, read):
    """Verify every read raises DecoderUnavailable without PyAV."""

    monkeypatch.setattr(video, "_load_av", lambda: None)
    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=0.0, end_s=8 / 30)

    with pytest.raises(DecoderUnavailable, match=r"kalanos\[video\]"):
        read(payload)


@pytest.mark.parametrize("read", READS, ids=READ_IDS)
def test_reading_a_corrupt_file_raises_decode_failed(tmp_path, read):
    """Verify a file PyAV cannot parse raises DecodeFailed rather than an av error."""

    corrupt = UPath(tmp_path / "corrupt.mp4")
    corrupt.write_bytes(random.Random(42).randbytes(4096))
    payload = VideoPayload(path=corrupt, frame_count=8, start_s=0.0, end_s=8 / 30)

    with pytest.raises(DecodeFailed):
        read(payload)


def _write_b_frame_video(path: UPath, frames: int) -> None:
    """Encode `frames` flat gray frames of level `4 * k`, reordered by B-frames.

    x264 only reorders frames of 64 pixels or more.
    """

    av = pytest.importorskip("av")
    numpy = pytest.importorskip("numpy")

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30, options={"bf": "3"})
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    for k in range(frames):
        pixels = numpy.full((64, 64, 3), 4 * k, dtype=numpy.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.mark.parametrize("start", [0, 20, 40])
def test_b_frame_segments_count_and_sample_their_own_frames(tmp_path, start):
    """Verify packets out of pts order still count, and each index finds its frame.

    A window over the whole segment decodes exactly its frames.
    """

    path = UPath(tmp_path / "b_frames.mp4")
    _write_b_frame_video(path, 60)
    payload = VideoPayload(
        path=path, frame_count=20, start_s=start / 30, end_s=(start + 20) / 30
    )

    (first,), (last,) = _windows(payload, [(0, 1), (19, 20)], 16)

    assert payload.count_frames() == 20
    assert float(first.mean()) == pytest.approx(4 * start, abs=2)
    assert float(last.mean()) == pytest.approx(4 * (start + 19), abs=2)
    (window,) = _windows(payload, [(0, 20)], 16)
    assert len(window) == 20
    assert float(window[0].mean()) == pytest.approx(4 * start, abs=2)
    assert float(window[-1].mean()) == pytest.approx(4 * (start + 19), abs=2)


def test_episodes_sharing_a_remote_file_open_it_once(monkeypatch, remote_video):
    """Verify every payload on one remote mp4 reads through a single file object."""

    opens = []
    original = MemoryFileSystem._open

    def counting(self, path, *args, **kwargs):
        opens.append(path)
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(MemoryFileSystem, "_open", counting)
    segments = [(0.0, 8 / 30, 8), (8 / 30, 14 / 30, 6)]

    for start_s, end_s, frame_count in segments:
        payload = VideoPayload(
            path=remote_video, frame_count=frame_count, start_s=start_s, end_s=end_s
        )
        assert payload.count_frames() == frame_count
        assert [len(w) for w in _windows(payload, [(0, frame_count)], 16)] == [
            frame_count
        ]

    assert len(opens) == 1


def test_a_remote_read_that_fails_drops_its_handle(remote_video):
    """Verify a failed read does not leave a broken handle for the next payload."""

    remote_video.write_bytes(random.Random(42).randbytes(4096))
    payload = VideoPayload(path=remote_video, frame_count=8, start_s=0.0, end_s=8 / 30)

    with pytest.raises(DecodeFailed):
        payload.count_frames()

    assert str(remote_video) not in video._REMOTE_HANDLES
