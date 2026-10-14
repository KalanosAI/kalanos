"""Verifies the vision metrics skip what they would misread, and share one sample."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import random
from fractions import Fraction

# External
import polars as pl
import pytest
from upath import UPath

# Internal
import kalanos.api
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.video import DecodeFailed, SegmentPackets, VideoPayload
from kalanos.analysis.coverage import coverage_lines, vision_metric_coverage
from kalanos.analysis.execution import use_tier
from kalanos.analysis.metrics import vision
from kalanos.analysis.metrics.registry import run_episode_metrics
from kalanos.analysis.metrics.vision import (
    camera_frames,
    exposure_level,
    exposure_shift_pct,
    frame_count_vs_timebase,
    frozen_frame_pct,
    max_gap_s,
    pts_defect_pct,
    sharpness_score,
)
from kalanos.analysis.models.binding import RequirementsSection
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.diagnostics import VisionSpec
from kalanos.analysis.models.domain import (
    Clock,
    Episode,
    FramePayload,
    Kind,
    MappingSource,
    SourceOrder,
    Stream,
    TimestampDtype,
)
from kalanos.analysis.models.eligibility import BlockingRoute, Consequence
from kalanos.analysis.models.metrics import (
    EpisodeContext,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.reporting.assemble import (
    grade_episode,
    grade_stream,
    scope_policy,
)
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.assets.policy import load_default_policy
from kalanos.testing import (
    check_metric,
    clean_frames,
    clean_recording,
    clip_frames,
    freeze_frames,
    stream_context,
)
from kalanos.testing.injectors import SyntheticFrames, blur_frames

# Local
from helpers import LEROBOT_FIXTURE


numpy = pytest.importorskip("numpy")


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

LEROBOT_VIDEO = (
    LEROBOT_FIXTURE / "videos" / "observation.images.up" / "chunk-000" / "file-000.mp4"
)

VISION_METRICS = [
    sharpness_score,
    exposure_shift_pct,
    frame_count_vs_timebase,
    pts_defect_pct,
    max_gap_s,
]

ACTION_TYPE = "action.joint_position_command"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _video_stream(payload: VideoPayload, frames: int) -> Stream:
    """Build a `Kind.VIDEO` stream with `frames` timestamps over `payload`."""

    return Stream(
        taxonomy_type="unmapped.observation.images.up",
        kind=Kind.VIDEO,
        timestamps=pl.Series("time_s", [index / 30 for index in range(frames)]),
        timestamp_dtype=TimestampDtype.FLOAT64,
        payload=payload,
        source_path=payload.path,
        clock=Clock.UNKNOWN,
        is_regular=True,
        channels=[],
    )


def _moving_actions(samples: int) -> Stream:
    """A 30 Hz action stream whose sine channels move throughout."""

    return clean_recording(hz=30.0, samples=samples, taxonomy_type=ACTION_TYPE)


def _with_frames(stream: Stream, frames: list, start: int, end: int) -> Stream:
    """Copy `stream` with its frames in `[start, end)` replaced by those of `frames`."""

    assert isinstance(stream.payload, SyntheticFrames)
    spliced = list(stream.payload.frames)
    spliced[start:end] = frames[start:end]
    return stream.model_copy(update={"payload": SyntheticFrames(frames=spliced)})


def _flickering(scale: float, freeze: tuple[int, int] | None = None) -> Stream:
    """100 frames of fixed noise scaled by `scale`, a corner flickering by one level.

    With `freeze=(start, end)`, frames `start` to `end - 1` all repeat frame `start`.
    """

    base = (
        (numpy.random.default_rng(42).integers(0, 256, (64, 64, 3)) * scale)
        .round()
        .astype(numpy.uint8)
    )
    frames = []
    for index in range(100):
        frame = base.copy()
        frame[:16, :16] = numpy.minimum(
            frame[:16, :16].astype(numpy.int16) + index % 2, 255
        ).astype(numpy.uint8)
        frames.append(frame)
    if freeze is not None:
        start, end = freeze
        frames[start:end] = [frames[start]] * (end - start)
    return clean_frames(frames=100, height=64, width=64).model_copy(
        update={"payload": SyntheticFrames(frames=frames)}
    )


def _full_scan_freeze(stream: Stream):
    """Grade `frozen_frame_pct` over every frame of `stream`, with moving actions."""

    return frozen_frame_pct(
        stream_context(
            stream, episode_streams=[_moving_actions(100)], full_frame_scan=True
        )
    )


def _write_paced_video(path: UPath, pts: list[int]) -> None:
    """Encode one 64x64 gray frame at each of `pts`, in ticks of 1/30 s, at 30 fps."""

    av = pytest.importorskip("av")

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30, options={"bf": "0"})
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    stream.codec_context.time_base = Fraction(1, 30)
    pixels = numpy.full((64, 64, 3), 128, dtype=numpy.uint8)
    for tick in pts:
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = tick
        frame.time_base = Fraction(1, 30)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _paced_camera(
    path: UPath,
    pts: list[int],
    source_field: str,
    frames: int | None = None,
    start: int = 0,
    whole_file: bool = True,
) -> Stream:
    """Encode `pts` to `path` as a camera of `frames` timestamps `index / 30`.

    `frames` defaults to one per pts; the segment starts at tick `start`.
    """

    _write_paced_video(path, pts)
    frames = len(pts) if frames is None else frames
    payload = VideoPayload(
        path=path,
        frame_count=frames,
        start_s=start / 30,
        end_s=(start + frames) / 30,
        whole_file=whole_file,
    )
    return _video_stream(payload, frames).model_copy(
        update={
            "taxonomy_type": f"unmapped.observation.images.{source_field}",
            "source_field": source_field,
        }
    )


def _skew(*cameras: Stream):
    """Run `camera_sync_skew_ms` over one episode holding `cameras`."""

    return run_episode_metrics(
        EpisodeContext(episode=Episode(id="episode_0", streams=list(cameras)))
    )["camera_sync_skew_ms"]


def _grade(stream: Stream, policy):
    """Grade `stream` under `policy`, returning its findings by metric id."""

    _graded, findings = grade_stream(
        stream, policy=policy, is_regular=True, episode_id="episode_0", category=None
    )
    return {finding.metric_id: finding for finding in findings}


def _assert_not_applicable(result) -> str:
    """Assert `result` is `not_applicable` with a reason, and return that reason."""

    assert result.status == MetricStatus.NOT_APPLICABLE
    reason = result.evidence["reason"]
    assert isinstance(reason, str) and reason
    return reason


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(autouse=True)
def _empty_vision_cache():
    """Start every test with no frames sampled, so no test reads another's."""

    vision.clear_cache()


@pytest.fixture
def _fixture_stream() -> Stream:
    """The second episode of `LEROBOT_VIDEO`, six frames starting mid-file."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=6, start_s=8 / 30, end_s=14 / 30
    )
    return _video_stream(payload, 6)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_corrupt_video_is_not_applicable_and_the_stream_still_grades(tmp_path):
    """A file PyAV cannot parse degrades both readers rather than aborting the grade."""

    corrupt = UPath(tmp_path / "corrupt.mp4")
    corrupt.write_bytes(random.Random(42).randbytes(4096))
    stream = _video_stream(
        VideoPayload(path=corrupt, frame_count=6, start_s=0.0, end_s=6 / 30), 6
    )

    graded, _findings = grade_stream(
        stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
    )

    _assert_not_applicable(graded.metrics["sharpness_score"])
    _assert_not_applicable(graded.metrics["frame_count_vs_timebase"])
    assert graded.metrics["sharpness_score"].availability == Availability.UNAVAILABLE


def test_a_short_video_blocks_only_under_a_video_scope():
    """A frame-count mismatch blocks through its contract under a video scope."""

    truncated = _video_stream(
        VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=8 / 30, end_s=16 / 30),
        8,
    )
    video_scope = scope_policy(
        load_default_policy(),
        RequirementsSection(required_capabilities=["sampled_video_quality"]),
    )
    default_scope = scope_policy(load_default_policy(), RequirementsSection())
    metric_id = "vision.frame_count_vs_timebase"

    finding = _grade(truncated, video_scope)[metric_id]
    assert (finding.consequence, finding.route) == (
        Consequence.BLOCK,
        BlockingRoute.CONTRACT,
    )
    reported = _grade(truncated, default_scope).get(metric_id)
    assert reported is None or reported.consequence != Consequence.BLOCK


@pytest.mark.parametrize(
    ("dropped", "declared", "gaps"),
    [([10, 11, 12], 23, 1), (list(range(1, 40, 2)), 40, 19)],
    ids=["one_gap", "every_other_frame"],
)
def test_pts_defect_pct_catches_dropped_frames(tmp_path, dropped, declared, gaps):
    """Frames missing from a 30 fps timeline read as gaps, a steady drop included."""

    clean_path, gapped_path = UPath(tmp_path / "clean.mp4"), UPath(tmp_path / "gap.mp4")
    _write_paced_video(clean_path, list(range(declared)))
    _write_paced_video(gapped_path, [t for t in range(declared) if t not in dropped])
    clean, gapped = (
        _video_stream(
            VideoPayload(
                path=path, frame_count=declared, start_s=0.0, end_s=declared / 30
            ),
            declared,
        )
        for path in (clean_path, gapped_path)
    )

    check_metric(
        pts_defect_pct,
        clean=stream_context(clean, is_regular=True),
        defective=stream_context(gapped, is_regular=True),
        policy=load_default_policy(),
    )
    result = pts_defect_pct(stream_context(gapped, is_regular=True))
    assert (result.evidence["duplicates"], result.evidence["gaps"]) == (0, gaps)


def test_pts_defect_pct_counts_duplicates(monkeypatch, _fixture_stream):
    """Two packets presented at one instant count as a duplicate, not a gap."""

    monkeypatch.setattr(
        VideoPayload,
        "packet_times",
        lambda self: SegmentPackets(pts=(0, 1, 1, 2), time_base=1 / 30, fps=30.0),
    )

    result = pts_defect_pct(stream_context(_fixture_stream, is_regular=True))

    assert (result.evidence["duplicates"], result.evidence["gaps"]) == (1, 0)
    assert result.value == pytest.approx(100 / 3)


def test_max_gap_reports_the_longest_gap_and_where(tmp_path):
    """Three frames missing after row 9 read as one gap of four periods, located."""

    clean_path, gapped_path = UPath(tmp_path / "clean.mp4"), UPath(tmp_path / "gap.mp4")
    _write_paced_video(clean_path, list(range(30)))
    _write_paced_video(gapped_path, [t for t in range(30) if t not in (10, 11, 12)])
    clean, gapped = (
        _video_stream(
            VideoPayload(path=path, frame_count=30, start_s=0.0, end_s=1.0), 30
        )
        for path in (clean_path, gapped_path)
    )

    check_metric(
        max_gap_s,
        clean=stream_context(clean, is_regular=True),
        defective=stream_context(gapped, is_regular=True),
        policy=load_default_policy(),
    )
    result = max_gap_s(stream_context(gapped, is_regular=True))
    assert result.value == pytest.approx(4 / 30)
    assert result.evidence["after_row"] == 9
    assert result.evidence["gap_start_s"] == pytest.approx(9 / 30)
    assert result.evidence["gap_end_s"] == pytest.approx(13 / 30)
    assert result.evidence["gap_periods"] == pytest.approx(4)
    assert max_gap_s(stream_context(clean, is_regular=True)).value == pytest.approx(
        1 / 30
    )


def test_camera_sync_skew_reports_a_known_offset(tmp_path):
    """A camera whose frames sit 100 ms late against its timestamps reads 100 ms."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(3, 33)), "side")

    result = _skew(up, side)

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == pytest.approx(100)
    offsets = {
        camera["stream"]: camera["offset_ms"] for camera in result.evidence["cameras"]
    }
    assert offsets == {"up": pytest.approx(0), "side": pytest.approx(100)}


def test_camera_sync_skew_reads_zero_on_aligned_cameras(tmp_path):
    """Two cameras whose frames sit on their timestamps read no skew."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(30)), "side")

    assert _skew(up, side).value == pytest.approx(0)


def test_camera_sync_skew_needs_two_cameras(tmp_path):
    """One camera has nothing to be skewed against."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")

    _assert_not_applicable(_skew(up, _moving_actions(30)))


def test_camera_sync_skew_skips_an_unreadable_camera(tmp_path):
    """A camera whose file cannot be read is set aside, and the others still compare."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(3, 33)), "side")
    corrupt = UPath(tmp_path / "corrupt.mp4")
    corrupt.write_bytes(random.Random(42).randbytes(4096))
    broken = up.model_copy(
        update={
            "payload": VideoPayload(
                path=corrupt, frame_count=30, start_s=0.0, end_s=1.0, whole_file=True
            ),
            "source_field": "broken",
        }
    )

    result = _skew(up, broken, side)

    assert result.value == pytest.approx(100)
    assert [skip["stream"] for skip in result.evidence["skipped"]] == ["broken"]


def test_camera_sync_skew_is_the_spread_between_cameras(tmp_path):
    """Cameras late by 100, 300 and 200 ms read 200 ms between the extremes."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(3, 33)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(9, 39)), "side")
    low = _paced_camera(UPath(tmp_path / "low.mp4"), list(range(6, 36)), "low")

    assert _skew(up, side, low).value == pytest.approx(200)


def test_camera_sync_skew_measures_from_the_segment_start(tmp_path):
    """A segment starting 8 frames into its file, on time there, reads no offset."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(
        UPath(tmp_path / "side.mp4"), list(range(38)), "side", frames=30, start=8
    )

    assert _skew(up, side).value == pytest.approx(0)


def test_camera_sync_skew_pairs_from_the_first_frame(tmp_path):
    """Frames past the last timestamp, as a whole file holds, shift no offset."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(33)), "side", 30)

    assert _skew(up, side).value == pytest.approx(0)


def test_camera_sync_skew_reads_only_camera_footage_in_source_order(tmp_path):
    """A depth map and a reordered camera take no part, however late."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(3, 33)), "side")
    depth = _paced_camera(UPath(tmp_path / "d.mp4"), list(range(9, 39)), "d")
    depth = depth.model_copy(
        update={
            "taxonomy_type": "extero.depth",
            "mapping_source": MappingSource.DICTIONARY,
        }
    )
    reordered = _paced_camera(UPath(tmp_path / "r.mp4"), list(range(9, 39)), "r")
    reordered = reordered.model_copy(
        update={"source_order": SourceOrder(preserved=False)}
    )

    result = _skew(up, side, depth, reordered)

    assert result.value == pytest.approx(100)
    assert {camera["stream"] for camera in result.evidence["cameras"]} == {
        "up",
        "side",
    }


def test_camera_sync_skew_skips_a_camera_sharing_its_file(tmp_path):
    """A segment of a file holding several episodes is placed by its recorded start.

    Its frames then sit on its timestamps whatever happened,
    so the camera is set aside rather than read as on time.
    """

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(3, 33)), "side")
    shared = _paced_camera(
        UPath(tmp_path / "shared.mp4"),
        list(range(93)),
        "shared",
        frames=30,
        start=33,
        whole_file=False,
    )

    result = _skew(up, side, shared)

    assert result.value == pytest.approx(100)
    assert [skip["stream"] for skip in result.evidence["skipped"]] == ["shared"]
    assert "several episodes" in result.evidence["skipped"][0]["reason"]
    assert "several episodes" in _assert_not_applicable(_skew(up, shared))


@pytest.mark.parametrize(
    "pts",
    [
        [t for t in range(31) if t != 5],
        [0, 1, 2, 2, *range(3, 29)],
    ],
    ids=["dropped", "repeated"],
)
def test_camera_sync_skew_skips_a_camera_with_a_missing_or_repeated_frame(
    tmp_path, pts
):
    """A lost or repeated frame shifts pairing by index, so that camera is set aside."""

    up = _paced_camera(UPath(tmp_path / "up.mkv"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mkv"), list(range(3, 33)), "side")
    broken = _paced_camera(UPath(tmp_path / "broken.mkv"), pts, "broken", 30)

    result = _skew(up, side, broken)

    assert result.value == pytest.approx(100)
    assert [skip["stream"] for skip in result.evidence["skipped"]] == ["broken"]


@pytest.mark.parametrize("stamp", [None, float("nan")], ids=["null", "nan"])
def test_camera_sync_skew_skips_a_camera_with_an_unreadable_timestamp(tmp_path, stamp):
    """A null or NaN recorded timestamp gives no offset, so that camera is set aside."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(3, 33)), "side")
    broken = _paced_camera(UPath(tmp_path / "broken.mp4"), list(range(30)), "broken")
    stamps = [index / 30 for index in range(30)]
    stamps[7] = stamp
    broken = broken.model_copy(
        update={"timestamps": pl.Series("time_s", stamps, dtype=pl.Float64)}
    )

    result = _skew(broken, up, side)

    assert result.value == pytest.approx(100)
    assert [skip["stream"] for skip in result.evidence["skipped"]] == ["broken"]


def test_camera_sync_skew_without_the_decoder_names_the_extra(monkeypatch, tmp_path):
    """With PyAV missing, the skew says which extra to install."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(30)), "side")
    monkeypatch.setattr(video, "av", None)

    assert "kalanos[video]" in _assert_not_applicable(_skew(up, side))


def test_max_gap_locates_its_gap_in_segment_seconds(_fixture_stream):
    """A segment starting mid-file reports its gap from the segment's own start."""

    result = max_gap_s(stream_context(_fixture_stream, is_regular=True))

    assert 0 <= result.evidence["gap_start_s"] < 6 / 30


def test_max_gap_never_starts_before_its_segment(tmp_path):
    """A first frame a little before the recorded start places its gap at 0, not below.

    Bounds sit half a frame early, so a segment's first packet can precede `start_s`,
    and on a clean stream the longest step is the first one.
    """

    path = UPath(tmp_path / "shared.mp4")
    _write_paced_video(path, list(range(93)))
    payload = VideoPayload(
        path=path, frame_count=30, start_s=33 / 30 + 0.001, end_s=63 / 30 + 0.001
    )

    result = max_gap_s(stream_context(_video_stream(payload, 30), is_regular=True))

    assert result.evidence["after_row"] == 0
    assert result.evidence["gap_start_s"] == 0


def test_gap_and_skew_decode_no_frame(monkeypatch, tmp_path):
    """Both metrics read packet timestamps alone, never a frame."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(3, 33)), "side")
    for method in ("fetch", "gray_windows", "rgb_frame", "rgb_frames", "_windows"):
        monkeypatch.setattr(
            VideoPayload,
            method,
            lambda self, *args, **kwargs: pytest.fail("a frame was decoded"),
        )

    assert max_gap_s(stream_context(up, is_regular=True)).value == pytest.approx(1 / 30)
    assert _skew(up, side).value == pytest.approx(100)


def test_gap_and_skew_are_report_only(tmp_path):
    """Graded under the default policy, neither metric raises a finding."""

    up = _paced_camera(UPath(tmp_path / "up.mp4"), list(range(30)), "up")
    side = _paced_camera(UPath(tmp_path / "side.mp4"), list(range(3, 33)), "side")

    graded, findings = grade_episode(
        Episode(id="episode_0", streams=[up, side]),
        adapter="test",
        adapter_confidence=1.0,
        policy=load_default_policy(),
        dictionary=load_default_dictionary(),
    )

    assert graded.metrics["camera_sync_skew_ms"].status == MetricStatus.REPORT_ONLY
    for stream in graded.streams:
        assert stream.metrics["max_gap_s"].status == MetricStatus.REPORT_ONLY
    assert not {finding.metric_id for finding in findings} & {
        "vision.max_gap_s",
        "vision.camera_sync_skew_ms",
    }


def test_frame_count_vs_timebase_catches_a_truncated_video(_fixture_stream):
    """A segment whose mp4 ends two frames early reads as a deviation."""

    truncated = _video_stream(
        VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=8 / 30, end_s=16 / 30),
        8,
    )

    check_metric(
        frame_count_vs_timebase,
        clean=stream_context(_fixture_stream, is_regular=True),
        defective=stream_context(truncated, is_regular=True),
        policy=load_default_policy(),
    )


def test_frame_count_vs_timebase_skips_in_memory_frames():
    """A frame list has no container whose count could disagree with it."""

    ctx = stream_context(clean_frames(frames=30), is_regular=True)

    _assert_not_applicable(frame_count_vs_timebase(ctx))


@pytest.mark.parametrize(
    "func", [*VISION_METRICS, frozen_frame_pct], ids=lambda func: func.__name__
)
@pytest.mark.parametrize(
    "stream",
    [
        clean_recording(),
        clean_frames(frames=30).model_copy(update={"payload": None}),
        clean_frames(frames=30, taxonomy_type="extero.depth"),
    ],
    ids=["series", "no_payload", "depth"],
)
def test_vision_metrics_skip_what_is_not_camera_footage(func, stream):
    """A series, an image stream with no frames, and a depth map are not graded."""

    ctx = stream_context(stream, episode_streams=[_moving_actions(100)])

    _assert_not_applicable(func(ctx))


@pytest.mark.parametrize("func", VISION_METRICS, ids=lambda func: func.__name__)
def test_vision_metrics_without_the_decoder_name_the_extra(
    monkeypatch, func, _fixture_stream
):
    """With PyAV missing, each metric says which extra to install."""

    monkeypatch.setattr(video, "av", None)

    reason = _assert_not_applicable(
        func(stream_context(_fixture_stream, is_regular=True))
    )
    assert "kalanos[video]" in reason


def test_frozen_frame_pct_without_the_decoder_names_the_extra(monkeypatch):
    """With PyAV missing, a stream long enough to window says which extra to install."""

    monkeypatch.setattr(video, "av", None)
    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=60, start_s=0.0, end_s=2.0)
    ctx = stream_context(
        _video_stream(payload, 60), episode_streams=[_moving_actions(60)]
    )

    assert "kalanos[video]" in _assert_not_applicable(frozen_frame_pct(ctx))


def test_a_freeze_while_the_robot_idles_is_not_counted():
    """A still camera over constant actions is an idle robot, not a frozen camera."""

    actions = _moving_actions(100)
    assert isinstance(actions.payload, FramePayload)
    constant = actions.payload.frame.with_columns(
        pl.lit(1.0).alias(channel.name) for channel in actions.channels
    )
    idle = actions.model_copy(update={"payload": FramePayload(frame=constant)})
    frames = freeze_frames(clean_frames(frames=100), start=25, length=25)

    result = frozen_frame_pct(stream_context(frames, episode_streams=[idle]))

    assert result.value == 0
    assert result.evidence["n_idle_runs"] >= 1


@pytest.mark.parametrize(
    "episode_streams",
    [None, [clean_recording(hz=30.0, samples=100)]],
    ids=["no_episode", "no_action_stream"],
)
def test_frozen_frame_pct_without_an_action_stream_is_not_applicable(episode_streams):
    """Without actions, a frozen camera cannot be told from an idle robot."""

    ctx = stream_context(clean_frames(frames=100), episode_streams=episode_streams)

    _assert_not_applicable(frozen_frame_pct(ctx))


def test_frozen_frame_pct_on_under_a_second_of_frames_is_not_applicable():
    """20 frames at 30 Hz cannot hold two minimum runs."""

    ctx = stream_context(clean_frames(frames=20), episode_streams=[_moving_actions(20)])

    _assert_not_applicable(frozen_frame_pct(ctx))


def test_a_sampled_window_finds_a_freeze_inside_it():
    """A freeze inside a later window is found where it starts."""

    frames = freeze_frames(clean_frames(frames=1000), start=220, length=20)

    result = frozen_frame_pct(
        stream_context(frames, episode_streams=[_moving_actions(1000)])
    )

    assert result.evidence["n_runs"] == 1
    assert result.evidence["first_run_index"] == 220
    assert result.value == pytest.approx(100 * 20 / 300)


def test_a_stream_with_no_action_readings_is_not_applicable():
    """An action stream that is all NaN cannot tell a freeze from an idle robot."""

    actions = _moving_actions(100)
    assert isinstance(actions.payload, FramePayload)
    empty = actions.payload.frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(channel.name)
        for channel in actions.channels
    )
    blank = actions.model_copy(update={"payload": FramePayload(frame=empty)})
    frames = freeze_frames(clean_frames(frames=100), start=25, length=25)

    _assert_not_applicable(
        frozen_frame_pct(stream_context(frames, episode_streams=[blank]))
    )


def test_sampled_windows_miss_a_freeze_the_full_scan_finds():
    """A freeze between two windows is invisible until every frame is read."""

    frames = freeze_frames(clean_frames(frames=1000), start=40, length=30)
    actions = [_moving_actions(1000)]

    sampled = frozen_frame_pct(stream_context(frames, episode_streams=actions))
    full = frozen_frame_pct(
        stream_context(frames, episode_streams=actions, full_frame_scan=True)
    )

    assert sampled.evidence["n_read"] == 300
    assert sampled.value == 0
    assert full.evidence["n_read"] == 1000
    assert full.value == pytest.approx(3.0)


def test_isolated_duplicates_do_not_count():
    """Every fourth frame repeating, as a rate conversion leaves it, is no freeze."""

    stream = clean_frames(frames=100)
    assert isinstance(stream.payload, SyntheticFrames)
    frames = list(stream.payload.frames)
    for index in range(4, len(frames), 4):
        frames[index] = frames[index - 1].copy()
    repeating = stream.model_copy(update={"payload": SyntheticFrames(frames=frames)})

    result = frozen_frame_pct(
        stream_context(repeating, episode_streams=[_moving_actions(100)])
    )

    assert result.value == 0


def test_a_dim_camera_is_not_read_as_frozen():
    """A camera at a fifth of full contrast still changes between frames."""

    assert _full_scan_freeze(_flickering(0.2)).value == 0


def test_a_freeze_in_dim_footage_is_still_found():
    """Scaling a dim camera's differences leaves its real repeats frozen."""

    result = _full_scan_freeze(_flickering(0.2, freeze=(40, 70)))

    assert result.value is not None
    assert result.value > 0


def test_a_blank_pair_is_never_frozen():
    """A run of black frames breaks the camera's picture, not its frame rate."""

    stream = _flickering(0.2)
    black = [numpy.zeros((64, 64, 3), numpy.uint8)] * 100

    result = _full_scan_freeze(_with_frames(stream, black, 40, 70))

    assert result.value == 0
    assert result.evidence["n_blank_pairs"] >= 29


def test_blurring_lowers_sharpness_score():
    """A blurred copy of sharp frames loses less of its edge energy to a re-blur."""

    stream = clean_frames(frames=30, height=64, width=64)

    sharp = sharpness_score(stream_context(stream)).value
    blurred = sharpness_score(stream_context(blur_frames(stream))).value

    assert sharp is not None and blurred is not None
    band = load_default_policy().metrics["vision.sharpness_score"].thresholds["default"]
    good = band.good
    assert good is not None and sharp >= good
    assert blurred < sharp


def test_a_flat_frame_reads_as_fully_blurred():
    """A frame with no edges has nothing a re-blur could remove."""

    flat = [numpy.full((16, 16, 3), 128, numpy.uint8)] * 30
    stream = clean_frames(frames=30).model_copy(
        update={"payload": SyntheticFrames(frames=flat)}
    )

    assert sharpness_score(stream_context(stream)).value == 0.0


def test_a_graded_video_stream_serializes_to_json(_fixture_stream):
    """Every vision value and evidence entry is a plain Python type."""

    graded, _findings = grade_stream(
        _fixture_stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
    )

    for name in ("sharpness_score", "exposure_shift_pct", "frame_count_vs_timebase"):
        assert graded.metrics[name].value is not None
    json.loads(graded.model_dump_json())


def _count_decodes(monkeypatch) -> list[int]:
    """Count the frames each `gray_windows` call yields, one entry per call."""

    calls: list[int] = []
    gray_windows = SyntheticFrames.gray_windows

    def counting(self, windows, size, native=frozenset(), **caps):
        calls.append(0)
        for item in gray_windows(self, windows, size, native, **caps):
            calls[-1] += 1
            yield item

    monkeypatch.setattr(SyntheticFrames, "gray_windows", counting)
    return calls


def test_blur_exposure_and_freezes_decode_the_stream_once(monkeypatch):
    """The three frame metrics read one shared pass rather than decoding three times."""

    calls = _count_decodes(monkeypatch)

    ctx = stream_context(
        clean_frames(frames=1000), episode_streams=[_moving_actions(1000)]
    )

    results = [
        func(ctx) for func in (sharpness_score, exposure_shift_pct, frozen_frame_pct)
    ]

    assert [result.status for result in results] == [MetricStatus.REPORT_ONLY] * 3
    assert calls == [300]
    assert results[0].evidence["n_sampled"] == 10


def test_without_an_action_stream_only_the_sampled_frames_are_decoded(monkeypatch):
    """With no freeze to look for, blur and exposure decode their own frames alone."""

    calls = _count_decodes(monkeypatch)
    ctx = stream_context(clean_frames(frames=1000))

    sharpness_score(ctx)
    exposure_shift_pct(ctx)

    assert calls == [10]


def test_a_depth_stream_sharing_a_cached_payload_is_still_skipped():
    """A camera's cached sample is never handed to a depth stream built on it."""

    camera = clean_frames(frames=30)
    depth = camera.model_copy(
        update={
            "taxonomy_type": "extero.depth",
            "mapping_source": MappingSource.DICTIONARY,
        }
    )

    sharpness_score(stream_context(camera, is_regular=True))

    _assert_not_applicable(sharpness_score(stream_context(depth, is_regular=True)))


def test_a_different_read_decodes_its_own_frames():
    """Two sample counts, or a full scan, on one payload each measure their own."""

    stream = clean_frames(frames=1000)

    ten = sharpness_score(stream_context(stream))
    three = sharpness_score(stream_context(stream, vision_samples=3))
    full = sharpness_score(stream_context(stream, full_frame_scan=True))

    assert ten.evidence["n_sampled"] == 10
    assert three.evidence["n_sampled"] == 3
    assert full.evidence["n_sampled"] == 1000


def test_a_stream_without_a_frame_rate_samples_single_frames():
    """With no timestamp gap to size a window by, each window is one frame."""

    stream = clean_frames(frames=100)
    still = stream.model_copy(update={"timestamps": pl.Series("time_s", [0.0] * 100)})

    result = sharpness_score(stream_context(still))

    assert result.evidence["n_sampled"] == 10


def test_a_full_scan_finds_a_blown_out_moment_the_windows_miss():
    """Exposure over every frame sees a flash that falls between the sampled windows."""

    stream = clean_frames(frames=1000)
    assert isinstance(stream.payload, SyntheticFrames)
    frames = list(stream.payload.frames)
    for index in range(40, 70):
        frames[index] = frames[index] * 0 + 255
    flashed = stream.model_copy(update={"payload": SyntheticFrames(frames=frames)})

    sampled = exposure_shift_pct(stream_context(flashed))
    full = exposure_shift_pct(stream_context(flashed, full_frame_scan=True))

    assert sampled.value == 0
    assert full.value == pytest.approx(3.0)


def test_exposure_catches_part_of_a_stream_blown_out():
    """Frames overexposed against the rest of their camera read as bad."""

    clean = clean_frames(frames=100)
    blown = clip_frames(clean)
    assert isinstance(blown.payload, SyntheticFrames)

    check_metric(
        exposure_shift_pct,
        clean=stream_context(clean),
        defective=stream_context(_with_frames(clean, blown.payload.frames, 20, 60)),
        policy=load_default_policy(),
    )


def test_exposure_catches_frames_darker_than_their_camera():
    """A light going out part-way through reads as darker frames."""

    clean = clean_frames(frames=100)
    assert isinstance(clean.payload, SyntheticFrames)
    dim = [(frame * 0.3).astype(frame.dtype) for frame in clean.payload.frames]

    result = exposure_shift_pct(stream_context(_with_frames(clean, dim, 20, 60)))

    assert result.evidence["darker"] == 4
    assert result.value == pytest.approx(40.0)


def test_a_camera_exposed_one_way_throughout_reads_clean():
    """A camera overexposed in every frame, like a white scene, is its own norm."""

    result = exposure_shift_pct(stream_context(clip_frames(clean_frames(frames=100))))

    assert result.value == 0


def test_a_blank_frame_is_bad_on_any_camera():
    """A uniform frame fails even where most of the camera's frames are blank too."""

    clean = clean_frames(frames=100)
    assert isinstance(clean.payload, SyntheticFrames)
    grey = [frame * 0 + 128 for frame in clean.payload.frames]

    partly = exposure_shift_pct(stream_context(_with_frames(clean, grey, 20, 60)))
    wholly = exposure_shift_pct(stream_context(_with_frames(clean, grey, 0, 100)))

    assert partly.evidence["blank"] == 4
    assert wholly.value == 100


def test_exposure_level_is_the_median_frame_mean():
    """A camera filming one gray level reads it, with nothing crushed or clipped."""

    frames = [numpy.full((16, 16, 3), 100, numpy.uint8) for _ in range(30)]
    stream = clean_frames(frames=30).model_copy(
        update={"payload": SyntheticFrames(frames=frames)}
    )

    result = exposure_level(stream_context(stream))

    assert result.value == pytest.approx(100.0)
    assert result.evidence["dark_share"] == 0
    assert result.evidence["bright_share"] == 0


@pytest.mark.parametrize(("level", "share"), [(0, "dark_share"), (255, "bright_share")])
def test_exposure_level_reports_crushed_and_clipped_pixels(level, share):
    """An all-black camera is wholly crushed, an all-white one wholly clipped."""

    frames = [numpy.full((16, 16, 3), level, numpy.uint8) for _ in range(30)]
    stream = clean_frames(frames=30).model_copy(
        update={"payload": SyntheticFrames(frames=frames)}
    )

    result = exposure_level(stream_context(stream))

    assert result.evidence[share] == 1


def test_the_metadata_tier_reads_no_frames(monkeypatch, _fixture_stream):
    """Under the metadata tier every vision metric is skipped before any decode."""

    monkeypatch.setattr(
        VideoPayload,
        "gray_windows",
        lambda self, windows, size, native=frozenset(): pytest.fail(
            "metadata tier decoded frames"
        ),
    )
    monkeypatch.setattr(
        VideoPayload,
        "packet_times",
        lambda self: pytest.fail("metadata tier counted frames"),
    )

    graded, _findings = grade_stream(
        _fixture_stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
        tier=ExecutionTier.METADATA,
    )

    for name in (
        "sharpness_score",
        "exposure_shift_pct",
        "frame_count_vs_timebase",
        "pts_defect_pct",
        "max_gap_s",
    ):
        assert "metadata tier" in _assert_not_applicable(graded.metrics[name])
    with use_tier(ExecutionTier.METADATA):
        skew = _skew(_fixture_stream, _fixture_stream)
    assert "metadata tier" in _assert_not_applicable(skew)


def test_vision_metric_coverage_leaves_depth_out():
    """A depth stream is no camera: it neither counts as one nor holds a camera back."""

    camera, depth = (
        grade_stream(
            stream,
            policy=load_default_policy(),
            is_regular=True,
            episode_id="episode_0",
            category=None,
        )[0]
        for stream in (
            clean_frames(frames=30),
            clean_frames(frames=30, taxonomy_type="extero.depth"),
        )
    )

    assert vision_metric_coverage([]) == (False, False)
    assert vision_metric_coverage([camera, depth]) == (True, False)


def test_a_read_that_runs_out_early_leaves_the_camera_unsampled(monkeypatch):
    """A truncated file the metrics still compute on has not finished its sample."""

    gray_windows = SyntheticFrames.gray_windows

    def truncated(self, windows, size, native=frozenset(), **caps):
        for index, item in enumerate(gray_windows(self, windows, size, native, **caps)):
            if index == 3:
                return
            yield item

    monkeypatch.setattr(SyntheticFrames, "gray_windows", truncated)
    graded, _findings = grade_stream(
        clean_frames(frames=30),
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
    )

    assert graded.frames is not None and graded.frames.missing_rows
    assert graded.metrics["sharpness_score"].value is not None
    assert vision_metric_coverage([graded]) == (False, False)


def test_coverage_counts_the_frames_the_vision_metrics_examined():
    """Without the diagnostic, coverage reports the frames the metrics measured."""

    report = kalanos.api.grade(str(LEROBOT_FIXTURE), vision_samples=2)

    assert report.coverage is not None
    assert report.coverage.decoded_frames_examined == 2 * len(report.episodes)
    assert report.coverage.eligible_visual_frames == 8 + 6
    assert not any(
        line.startswith("Visual quality not evaluated")
        for line in coverage_lines(report.coverage)
    )


def test_a_capped_read_is_unavailable_and_keeps_its_frames():
    """A read stopped by max_decode_frames grades nothing and shows what it read."""

    ctx = StreamContext(
        stream=clean_frames(frames=100),
        is_regular=True,
        vision=VisionSpec(sample_frames=2, max_decode_frames=4),
    )

    for func in (sharpness_score, exposure_shift_pct, exposure_level):
        assert func(ctx).availability == Availability.UNAVAILABLE
    frames = camera_frames(ctx)
    assert frames is not None
    assert frames.availability == Availability.UNAVAILABLE
    assert len(frames.frames) == 4
    assert frames.missing_rows == frames.requested_rows[4:]


def test_a_full_scan_reads_past_the_decode_budget():
    """A full scan is asked for every frame, so max_decode_frames does not stop it."""

    ctx = StreamContext(
        stream=clean_frames(frames=100),
        is_regular=True,
        full_frame_scan=True,
        vision=VisionSpec(sample_frames=2, max_decode_frames=4),
    )

    frames = camera_frames(ctx)

    assert frames is not None
    assert frames.availability == Availability.COMPUTED
    assert len(frames.frames) == 100


def test_a_decode_failure_is_an_error_and_keeps_its_frames(monkeypatch):
    """A decode that fails part-way is an error, and the frame before it is kept."""

    gray_windows = SyntheticFrames.gray_windows

    def failing(self, windows, size, native=frozenset(), **caps):
        yield next(gray_windows(self, windows, size, native, **caps))
        raise DecodeFailed("corrupt packet")

    monkeypatch.setattr(SyntheticFrames, "gray_windows", failing)
    ctx = stream_context(clean_frames(frames=100))

    assert sharpness_score(ctx).availability == Availability.ERROR
    frames = camera_frames(ctx)
    assert frames is not None
    assert (frames.availability, frames.reason) == (
        Availability.ERROR,
        "corrupt packet",
    )
    assert len(frames.frames) == 1


def test_a_reordered_stream_is_not_decoded(monkeypatch):
    """Frame positions only name source rows while the adapter kept the frame order."""

    monkeypatch.setattr(
        SyntheticFrames,
        "gray_windows",
        lambda *args, **kwargs: pytest.fail("a reordered stream was decoded"),
    )
    stream = clean_frames(frames=30)
    reordered = stream.model_copy(
        update={
            "source_order": SourceOrder(
                preserved=False, original_index=list(range(30))[::-1]
            )
        }
    )

    reason = _assert_not_applicable(sharpness_score(stream_context(reordered)))
    assert "frame order" in reason


def test_exposure_shift_locates_its_bad_frames():
    """Each sampled frame the light went out on is one support interval."""

    clean = clean_frames(frames=100)
    assert isinstance(clean.payload, SyntheticFrames)
    dim = [(frame * 0.3).astype(frame.dtype) for frame in clean.payload.frames]

    result = exposure_shift_pct(stream_context(_with_frames(clean, dim, 20, 60)))

    intervals = [(i.start, i.end_exclusive) for i in result.support.intervals]
    assert len(intervals) == 4
    assert all(20 <= start and end <= 60 for start, end in intervals)
    assert intervals == [(i, i + 1) for i in result.evidence["bad_indices"]]


def test_frozen_frames_locate_their_runs():
    """A counted freeze is supported by exactly the frames that repeat."""

    frames = freeze_frames(clean_frames(frames=1000), start=220, length=20)

    result = frozen_frame_pct(
        stream_context(frames, episode_streams=[_moving_actions(1000)])
    )

    assert [(i.start, i.end_exclusive) for i in result.support.intervals] == [
        (220, 240)
    ]


@pytest.mark.parametrize("full_frame_scan", [False, True])
def test_frame_count_finds_frames_past_the_timestamps_on_every_read(full_frame_scan):
    """Six frames under four timestamps read as surplus under a full scan too."""

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=4, start_s=8 / 30, end_s=14 / 30
    )

    result = frame_count_vs_timebase(
        stream_context(_video_stream(payload, 4), full_frame_scan=full_frame_scan)
    )

    assert (result.evidence["present"], result.evidence["implied"]) == (6, 4)
    assert result.value == pytest.approx(0.5)


def test_evidence_frames_alone_carry_hashes_on_a_full_scan():
    """A full scan lists every frame, hashing only the sample's frames and bad ones."""

    clean = clean_frames(frames=100)
    assert isinstance(clean.payload, SyntheticFrames)
    dim = [(frame * 0.3).astype(frame.dtype) for frame in clean.payload.frames]
    stream = _with_frames(clean, dim, 20, 60)
    sampled = camera_frames(stream_context(stream))
    full_ctx = stream_context(stream, full_frame_scan=True)

    full = camera_frames(full_ctx)
    bad = exposure_shift_pct(full_ctx).evidence["bad_indices"]

    assert sampled is not None and full is not None
    assert len(full.frames) == 100
    assert bad == list(range(20, 60))
    assert {f.source_row for f in full.frames if f.luma_sha256} == {
        *sampled.requested_rows,
        *bad,
    }


def test_a_critical_vision_result_carries_its_worst_frame():
    """A soft camera's sharpness shows its worst frame; a sharp one's does not."""

    y, x = numpy.mgrid[0:64, 0:64]
    soft = [
        numpy.repeat(
            (128 + 100 * numpy.sin(2 * numpy.pi * (x + k) / 32) * numpy.sin(y / 5))[
                ..., None
            ],
            3,
            axis=2,
        )
        .round()
        .astype(numpy.uint8)
        for k in range(30)
    ]
    sharp = clean_frames(frames=30, height=64, width=64)

    results = [
        grade_stream(
            stream,
            policy=load_default_policy(),
            is_regular=True,
            episode_id="episode_0",
            category=None,
        )[0].metrics["sharpness_score"]
        for stream in (
            sharp.model_copy(update={"payload": SyntheticFrames(frames=soft)}),
            sharp,
        )
    ]

    assert results[0].status == MetricStatus.CRITICAL
    assert results[0].evidence["thumbnail_png_base64"]
    assert results[1].status == MetricStatus.GOOD
    assert "thumbnail_png_base64" not in results[1].evidence


def test_a_payload_that_raises_still_grades_its_stream(monkeypatch):
    """A read failing outside the decode errors marks the metrics, not the whole run."""

    def broken(self, *args, **kwargs):
        raise ValueError("plugin bug")
        yield

    monkeypatch.setattr(SyntheticFrames, "gray_windows", broken)

    graded, _findings = grade_stream(
        clean_frames(frames=30),
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
    )

    assert graded.metrics["sharpness_score"].availability == Availability.ERROR
    assert graded.frames is None
