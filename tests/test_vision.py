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
from fsspec.implementations.memory import MemoryFile, MemoryFileSystem
from upath import UPath

# Internal
import kalanos.api
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.video import (
    ContainerHeader,
    DeclaredFormat,
    DecodeFailed,
    SegmentPackets,
    VideoPayload,
)
from kalanos.analysis.coverage import coverage_lines, vision_metric_coverage
from kalanos.analysis.execution import use_tier
from kalanos.analysis.metrics import vision
from kalanos.analysis.metrics.continuity import (
    _plan_cuts,
    _window_thumbnails,
    cut_count,
)
from kalanos.analysis.metrics.registry import run_episode_metrics, run_stream_metrics
from kalanos.analysis.metrics.vision import (
    blank_frame_pct,
    bright_frame_pct,
    camera_frames,
    camera_stream_present,
    contrast_score,
    dark_frame_pct,
    exposure_level,
    exposure_shift_pct,
    footage_vector,
    frame_count_vs_timebase,
    frozen_frame_pct,
    max_gap_s,
    pts_defect_pct,
    sharpness_score,
    video_format_consistency,
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
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import CameraSummary, GradedEpisode
from kalanos.analysis.models.scoring import Finding
from kalanos.analysis.reporting.assemble import (
    grade_episode,
    grade_stream,
    scope_policy,
)
from kalanos.analysis.scoring.cameras import (
    _TOO_FEW_FOOTAGE,
    FootageVariation,
    camera_key,
    camera_samples,
    compare_cameras,
)
from kalanos.analysis.scoring.score import score_episode
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


def _write_paced_video(path: UPath, pts: list[int], vary: bool = False) -> None:
    """Encode one 64x64 gray frame at each of `pts`, in ticks of 1/30 s, at 30 fps.

    With `vary`, each frame is a ramp whose level follows its tick,
    so no two frames look alike and none reads as blank.
    """

    av = pytest.importorskip("av")

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=30, options={"bf": "0"})
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    stream.codec_context.time_base = Fraction(1, 30)
    for tick in pts:
        pixels = numpy.full((64, 64, 3), 128, dtype=numpy.uint8)
        if vary:
            ramp = 20 + (tick * 7) % 150 + numpy.arange(64)
            pixels[:] = ramp[None, :, None].astype(numpy.uint8)
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


CAMERA_FIELD = "observation.images.up"


def _camera(
    path: UPath, declared: DeclaredFormat | None = None, start_s: float = 8 / 30
) -> Stream:
    """A six-frame camera stream over `path` from `start_s`, as a LeRobot video key."""

    payload = VideoPayload(
        path=path,
        frame_count=6,
        start_s=start_s,
        end_s=start_s + 6 / 30,
        declared=declared,
    )
    return _video_stream(payload, 6).model_copy(update={"source_field": CAMERA_FIELD})


def _frameless_mp4(path: UPath) -> UPath:
    """Write an mp4 whose one h264 stream holds no frame."""

    av = pytest.importorskip("av")
    # Without empty_moov the muxer drops a track that never received a packet.
    with av.open(str(path), "w", options={"movflags": "empty_moov"}) as out:
        stream = out.add_stream("h264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        out.start_encoding()
    return path


def _unreadable(tmp_path) -> UPath:
    """Random bytes no demuxer recognises."""

    corrupt = UPath(tmp_path / "corrupt.mp4")
    corrupt.write_bytes(random.Random(42).randbytes(4096))
    return corrupt


@pytest.mark.parametrize(
    "build",
    [
        lambda tmp_path: _camera(UPath(tmp_path / "absent.mp4")),
        lambda tmp_path: _camera(_unreadable(tmp_path)),
        lambda tmp_path: _camera(_frameless_mp4(UPath(tmp_path / "frameless.mp4"))),
        # A shared file that ends before this episode's segment begins.
        lambda tmp_path: _camera(LEROBOT_VIDEO, start_s=500.0),
        lambda tmp_path: _camera(UPath("memory://videos/absent.mp4")),
    ],
    ids=["missing", "unreadable", "frameless", "segment_past_the_end", "remote"],
)
def test_an_absent_camera_blocks_only_under_a_video_scope(tmp_path, build):
    """A camera with no first frame to read blocks through its contract, naming it."""

    stream = build(tmp_path)
    video_scope = scope_policy(
        load_default_policy(),
        RequirementsSection(required_capabilities=["sampled_video_quality"]),
    )
    default_scope = scope_policy(load_default_policy(), RequirementsSection())
    metric_id = "vision.camera_stream_present"

    finding = _grade(stream, video_scope)[metric_id]
    assert (finding.consequence, finding.route) == (
        Consequence.BLOCK,
        BlockingRoute.CONTRACT,
    )
    assert finding.source_field == CAMERA_FIELD
    assert all(
        finding.consequence != Consequence.BLOCK
        for finding in _grade(stream, default_scope).values()
    )


@pytest.mark.parametrize("failing", ["open", "read"])
def test_a_network_error_leaves_presence_unknown_and_uncached(monkeypatch, failing):
    """A remote read that fails outside the file neither blocks nor sticks."""

    remote = UPath("memory://videos/flaky.mp4")
    remote.write_bytes(LEROBOT_VIDEO.read_bytes())
    stream = _camera(remote)
    video_scope = scope_policy(
        load_default_policy(),
        RequirementsSection(required_capabilities=["sampled_video_quality"]),
    )

    def unreachable(*_args, **_kwargs):
        """Fail as a hub answering 503 does."""

        raise ConnectionError("503 Service Unavailable")

    target = MemoryFileSystem if failing == "open" else MemoryFile
    method = "_open" if failing == "open" else "read"
    try:
        with monkeypatch.context() as patched:
            patched.setattr(target, method, unreachable)
            result = camera_stream_present(stream_context(stream))
            findings = _grade(stream, video_scope)

        assert result.value is None
        assert result.availability == Availability.UNAVAILABLE
        assert "vision.camera_stream_present" not in findings
        assert camera_stream_present(stream_context(stream)).value == 1.0
    finally:
        video.close_video_files()
        remote.unlink()


def test_a_dropped_first_frame_is_still_present(tmp_path):
    """A segment whose first frame the encoder dropped reads its next one."""

    path = UPath(tmp_path / "dropped.mp4")
    _write_paced_video(path, [tick for tick in range(20) if tick != 8])

    result = camera_stream_present(stream_context(_camera(path)))

    assert result.value == 1.0
    assert result.evidence["reason"] is None


def test_presence_reads_the_shared_decode_and_format_decodes_none(monkeypatch):
    """Presence takes its first frame from the vision read; format reads no frame."""

    calls: list[int] = []
    gray_windows = VideoPayload.gray_windows

    def counting(self, windows, size, native=frozenset(), **caps):
        calls.append(0)
        for item in gray_windows(self, windows, size, native, **caps):
            calls[-1] += 1
            yield item

    monkeypatch.setattr(VideoPayload, "gray_windows", counting)
    ctx = stream_context(_camera(LEROBOT_VIDEO, DeclaredFormat(fps=30.0)))

    sharpness_score(ctx)
    assert len(calls) == 1
    assert camera_stream_present(ctx).value == 1.0
    assert video_format_consistency(ctx).value == 0.0
    assert len(calls) == 1


def _indexed_decodes(monkeypatch) -> list[list[int]]:
    """Record the frame indices each `VideoPayload.gray_windows` call yields."""

    calls: list[list[int]] = []
    gray_windows = VideoPayload.gray_windows

    def recording(self, windows, size, native=frozenset(), **caps):
        calls.append([])
        counts = [0] * len(windows)
        for item in gray_windows(self, windows, size, native, **caps):
            position = item[0]
            calls[-1].append(windows[position][0] + counts[position])
            counts[position] += 1
            yield item

    monkeypatch.setattr(VideoPayload, "gray_windows", recording)
    return calls


def test_every_camera_metric_shares_one_decode(monkeypatch, tmp_path):
    """The vision metrics, cut_count and presence decode each frame once in all."""

    calls = _indexed_decodes(monkeypatch)
    camera = _paced_camera(UPath(tmp_path / "paced.mp4"), list(range(60)), "up")
    ctx = stream_context(camera, episode_streams=[camera, _moving_actions(60)])

    results = run_stream_metrics(ctx)

    assert results["cut_count"].status == MetricStatus.REPORT_ONLY
    assert results["camera_stream_present"].value == 1.0
    # Sixty frames make one vision window, which holds every cut window too.
    assert calls == [list(range(60))]


def test_a_dropped_first_frame_is_present_with_one_sample(tmp_path):
    """With no sampled range at a dropped first frame, presence reads one alone."""

    path = UPath(tmp_path / "dropped.mp4")
    pts = [tick for tick in range(80) if tick != 8]
    camera = _paced_camera(path, pts, CAMERA_FIELD, frames=60, start=8)

    result = camera_stream_present(stream_context(camera, vision_samples=1))

    assert result.value == 1.0
    assert result.evidence["reason"] is None


def test_a_failure_between_ranges_leaves_presence_to_read_alone(monkeypatch, tmp_path):
    """A stop the shared read cannot place leaves presence to read its first frame."""

    path = UPath(tmp_path / "dropped.mp4")
    pts = [tick for tick in range(80) if tick != 8]
    camera = _paced_camera(path, pts, CAMERA_FIELD, frames=60, start=8)
    gray_windows = VideoPayload.gray_windows
    calls: list[int] = []

    def failing(self, windows, size, native=frozenset(), **caps):
        calls.append(1)
        for item in gray_windows(self, windows, size, native, **caps):
            if len(calls) == 1 and item[0] > 0:
                raise DecodeFailed("corrupt packet")
            yield item

    monkeypatch.setattr(VideoPayload, "gray_windows", failing)
    result = camera_stream_present(stream_context(camera, vision_samples=1))

    assert result.value == 1.0


def test_a_decode_cap_before_the_first_frame_leaves_presence_unknown(tmp_path):
    """A cap the seek preroll alone exhausts says nothing about the file."""

    camera = _paced_camera(
        UPath(tmp_path / "late.mp4"),
        list(range(100)),
        CAMERA_FIELD,
        frames=60,
        start=40,
    )
    ctx = stream_context(camera).model_copy(
        update={"vision": VisionSpec(sample_frames=2, max_decode_frames=2)}
    )

    result = camera_stream_present(ctx)

    assert result.value is None
    assert result.availability == Availability.UNAVAILABLE


@pytest.mark.parametrize(
    ("windows", "cut_windows"),
    [([(10, 30)], [(5, 20)]), ([(5, 30)], [(10, 20)])],
    ids=["cut_first", "cut_inside"],
)
def test_the_shared_read_numbers_frames_as_each_read_alone(
    tmp_path, windows, cut_windows
):
    """On video missing a frame, each window takes the frames its own read would."""

    path = UPath(tmp_path / "gap.mp4")
    _write_paced_video(path, [tick for tick in range(60) if tick != 7], vary=True)
    payload = VideoPayload(path=path, frame_count=60, start_s=0.0, end_s=2.0)
    sampled = [15, 25]
    spec = VisionSpec()
    limits = {
        "dark_level": spec.dark_level,
        "bright_level": spec.bright_level,
        "max_decode_frames": spec.max_decode_frames,
        "max_pixels": spec.max_pixels,
    }

    ranges = vision._split(windows + cut_windows + [(0, 1)], 60)
    shared = vision._pass(
        payload, ranges, windows, sampled, sampled, cut_windows, **limits
    )
    alone = vision._read(payload, windows, sampled, sampled, **limits)
    cut, failure = _window_thumbnails(payload, cut_windows)

    assert isinstance(shared.frames, vision._Frames)
    assert isinstance(alone, vision._Frames)
    assert (shared.frames.pts, shared.frames.luma_sha256) == (
        alone.pts,
        alone.luma_sha256,
    )
    [(start, diffs)] = shared.frames.diffs
    assert start == alone.diffs[0][0]
    assert numpy.array_equal(diffs, alone.diffs[0][1])
    assert (shared.cut_failure, failure) == (None, None)
    assert shared.cut.keys() == cut.keys()
    assert all(numpy.array_equal(shared.cut[index], cut[index]) for index in cut)


def test_a_failing_control_stream_fails_cut_count_alone(monkeypatch):
    """A control payload that cannot be fetched leaves the vision read whole."""

    action = _moving_actions(1000)
    payload_type = type(action.payload)
    fetch = payload_type.fetch
    calls: list[int] = []

    def failing(self, *args, **kwargs):
        if self is action.payload:
            calls.append(1)
            raise OSError("503 Service Unavailable")
        return fetch(self, *args, **kwargs)

    monkeypatch.setattr(payload_type, "fetch", failing)
    ctx = stream_context(clean_frames(frames=1000), episode_streams=[action])

    assert sharpness_score(ctx).status == MetricStatus.REPORT_ONLY
    with pytest.raises(OSError):
        cut_count(ctx)
    assert len(calls) == 1


def test_a_pixel_limit_leaves_later_cut_windows_as_before():
    """A frame over max_pixels stops the vision read, and the cut windows still read."""

    ctx = stream_context(
        clean_frames(frames=1000, height=16, width=16),
        episode_streams=[_moving_actions(1000)],
    )
    windows, _sampled = vision._layout(ctx, False)
    plan = _plan_cuts(ctx)
    assert not isinstance(plan, MetricResult)
    assert any(start >= windows[0][1] for start, _end in plan.windows)
    default = cut_count(ctx)
    vision.clear_cache()
    capped = ctx.model_copy(update={"vision": VisionSpec(max_pixels=64)})

    assert sharpness_score(capped).availability == Availability.UNAVAILABLE
    assert cut_count(capped).evidence == default.evidence


def test_a_pixel_limit_leaves_presence_and_cuts_as_before(tmp_path):
    """A frame over max_pixels stops the vision metrics alone."""

    camera = _paced_camera(UPath(tmp_path / "paced.mp4"), list(range(60)), "up")
    episode = [camera, _moving_actions(60)]
    default = cut_count(stream_context(camera, episode_streams=episode))
    vision.clear_cache()
    ctx = stream_context(camera, episode_streams=episode).model_copy(
        update={"vision": VisionSpec(max_pixels=64)}
    )

    assert sharpness_score(ctx).availability == Availability.UNAVAILABLE
    assert camera_stream_present(ctx).value == 1.0
    assert cut_count(ctx).evidence == default.evidence


@pytest.mark.parametrize(
    "doctored",
    [
        {"width": 64},
        {"height": 64},
        {"fps": 25.0},
        {"pix_fmt": "yuv444p"},
    ],
    ids=lambda doctored: next(iter(doctored)),
)
def test_video_format_consistency_names_each_differing_field(doctored):
    """One declared field that disagrees with the header is the one reported."""

    declared = {"width": 32, "height": 32, "fps": 30.0, "pix_fmt": "yuv420p"}
    declared.update(doctored)
    result = video_format_consistency(
        stream_context(_camera(LEROBOT_VIDEO, DeclaredFormat(**declared)))
    )

    field = next(iter(doctored))
    assert result.value == 1.0
    assert [entry["field"] for entry in result.evidence["differs"]] == [field]
    assert result.evidence["differs"][0]["declared"] == doctored[field]
    assert result.evidence["undeclared"] == []


def test_video_format_consistency_lists_undeclared_fields():
    """Undeclared fields are listed, not compared; with none declared, nothing is."""

    result = video_format_consistency(
        stream_context(_camera(LEROBOT_VIDEO, DeclaredFormat(fps=30.0)))
    )
    assert result.value == 0.0
    assert result.evidence["undeclared"] == ["width", "height", "pix_fmt"]

    _assert_not_applicable(
        video_format_consistency(stream_context(_camera(LEROBOT_VIDEO)))
    )


def test_format_reads_the_header_without_numpy(monkeypatch):
    """Comparing the header needs only the decoder; presence still names numpy."""

    monkeypatch.setattr(vision, "load_numpy", lambda: None)
    stream = _camera(LEROBOT_VIDEO, DeclaredFormat(fps=30.0))

    assert video_format_consistency(stream_context(stream)).value == 0.0
    assert "numpy" in _assert_not_applicable(
        camera_stream_present(stream_context(stream))
    )


def test_a_field_the_container_does_not_state_is_not_a_difference(monkeypatch):
    """A header with no average rate or pixel format contradicts no declaration."""

    monkeypatch.setattr(
        VideoPayload,
        "header",
        lambda self: ContainerHeader(width=32, height=32, fps=None, pix_fmt=None),
    )
    declared = DeclaredFormat(width=32, height=32, fps=30.0, pix_fmt="yuv420p")

    result = video_format_consistency(stream_context(_camera(LEROBOT_VIDEO, declared)))

    assert result.value == 0.0
    assert result.evidence["differs"] == []
    assert result.evidence["unstated"] == ["fps", "pix_fmt"]


@pytest.mark.parametrize(
    ("found", "declared", "differs"),
    [
        (30000 / 1001, 29.97, False),
        (30.0, 30.0, False),
        (29.0, 30.0, True),
        (25.0, 30.0, True),
    ],
)
def test_fps_tolerance_absorbs_ntsc_rounding_only(found, declared, differs):
    """A rounded NTSC rate matches its declaration; a real rate change does not."""

    assert vision._fps_differs(found, declared) is differs


def test_presence_and_format_skip_in_memory_frames():
    """Frames held in memory have no file to be missing or to declare a format."""

    stream = clean_frames()
    for check in (camera_stream_present, video_format_consistency):
        _assert_not_applicable(check(stream_context(stream)))


def test_presence_leaves_the_vision_score_unchanged_on_clean_video():
    """A readable camera grades presence under a video scope, vision score unmoved."""

    requirements = RequirementsSection(required_capabilities=["sampled_video_quality"])
    with_presence = scope_policy(load_default_policy(), requirements)
    without = load_default_policy()
    del without.metrics["vision.camera_stream_present"]
    without = scope_policy(without, requirements)
    stream = _camera(LEROBOT_VIDEO)

    def graded(policy):
        """Grade the clean camera under `policy`."""

        stream_grade, _findings = grade_stream(
            stream, policy=policy, is_regular=True, episode_id="e0", category=None
        )
        return stream_grade

    present, absent = graded(with_presence), graded(without)
    assert present.metrics["camera_stream_present"].status == MetricStatus.GOOD
    assert present.score.families["vision"] == pytest.approx(
        absent.score.families["vision"]
    )
    assert score_episode([present], {}, policy=with_presence).families[
        "vision"
    ] == pytest.approx(score_episode([absent], {}, policy=without).families["vision"])


def test_an_absent_camera_alone_gives_vision_no_score(tmp_path):
    """Presence failing with nothing else graded leaves vision unscored, not zero."""

    video_scope = scope_policy(
        load_default_policy(),
        RequirementsSection(required_capabilities=["sampled_video_quality"]),
    )
    stream_grade, _findings = grade_stream(
        _camera(UPath(tmp_path / "absent.mp4")),
        policy=video_scope,
        is_regular=True,
        episode_id="e0",
        category=None,
    )

    assert stream_grade.metrics["camera_stream_present"].points == 0.0
    assert "vision" not in stream_grade.score.families
    episode = score_episode([stream_grade], {}, policy=video_scope)
    assert "vision" not in episode.families
    assert episode.metrics["vision.camera_stream_present"].points == 0.0


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


def _gray_frames(kind: str) -> list:
    """Build square gray frames of side `_SAMPLE_SIZE`, as `gray_windows` yields them.

    Parameters
    ----------
    kind : str
        `random`, `black`, `white`, `dim` (a narrow spread around level 30),
        `two_level` (half black, half white)
        or `between_levels` (P5 and P95 both fall between two gray levels).

    Returns
    -------
    list of numpy.ndarray
        50 uint8 frames, or one for a constant or `between_levels` kind.
    """

    rng = numpy.random.default_rng(42)
    side = vision._SAMPLE_SIZE
    if kind == "black":
        return [numpy.zeros((side, side), numpy.uint8)]
    if kind == "white":
        return [numpy.full((side, side), 255, numpy.uint8)]
    if kind == "between_levels":
        # P5 sits between ranks 819 and 820, P95 between 15563 and 15564,
        # so 820 pixels at 10 and 820 at 30 put each pair on two levels.
        frame = numpy.full(side * side, 20, numpy.uint8)
        frame[:820] = 10
        frame[-820:] = 30
        return [frame.reshape(side, side)]
    frames = []
    for _ in range(50):
        if kind == "random":
            frame = rng.integers(0, 256, (side, side), dtype=numpy.uint8)
        elif kind == "dim":
            frame = numpy.clip(rng.normal(30, 3, (side, side)), 0, 255).astype(
                numpy.uint8
            )
        else:
            frame = numpy.where(rng.random((side, side)) < 0.5, 0, 255).astype(
                numpy.uint8
            )
        frames.append(frame)
    return frames


FRAME_KINDS = ["random", "black", "white", "dim", "two_level", "between_levels"]


@pytest.mark.parametrize("kind", FRAME_KINDS)
def test_frame_percentiles_match_numpy_to_the_bit(kind):
    """P5 and P95 from a frame's histogram equal `numpy.percentile` exactly."""

    for frame in _gray_frames(kind):
        cumulative = numpy.cumsum(numpy.bincount(frame.ravel(), minlength=256))
        for q in (5, 95):
            assert vision._percentile(cumulative, q) == float(
                numpy.percentile(frame, q)
            )


@pytest.mark.parametrize("kind", FRAME_KINDS)
def test_thumbnails_match_the_block_mean_to_the_bit(kind):
    """Integer block sums give the same float32 thumbnail as a float32 block mean."""

    for frame in _gray_frames(kind):
        for size in (vision._FREEZE_SIZE, vision._VARIATION_SIZE):
            k = vision._SAMPLE_SIZE // size
            thumbnail = vision._thumbnail(frame, size)
            assert thumbnail.dtype == numpy.float32
            assert numpy.array_equal(
                thumbnail,
                frame.reshape(size, k, size, k).mean(axis=(1, 3), dtype=numpy.float32),
            )


def test_a_graded_video_stream_serializes_to_json(_fixture_stream):
    """Every vision value and evidence entry is a plain Python type."""

    graded, _findings = grade_stream(
        _fixture_stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
    )

    for name in (
        "sharpness_score",
        "exposure_shift_pct",
        "frame_count_vs_timebase",
        "camera_stream_present",
    ):
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
    """The three frame metrics read one shared pass rather than decoding three times.

    The pass also reads cut_count's windows, around the actions' largest jumps.
    """

    calls = _count_decodes(monkeypatch)

    ctx = stream_context(
        clean_frames(frames=1000), episode_streams=[_moving_actions(1000)]
    )
    windows, _sampled = vision._layout(ctx, False)
    plan = _plan_cuts(ctx)
    assert not isinstance(plan, MetricResult)
    ranges = vision._merge(windows + plan.windows, 1000)

    results = [
        func(ctx) for func in (sharpness_score, exposure_shift_pct, frozen_frame_pct)
    ]

    assert [result.status for result in results] == [MetricStatus.REPORT_ONLY] * 3
    assert calls == [sum(end - start for start, end in ranges)]
    assert results[0].evidence["n_sampled"] == 10


def test_a_failure_in_a_cut_range_leaves_vision_as_before(monkeypatch):
    """A decode failure in frames only cut_count reads leaves the vision read whole."""

    ctx = stream_context(
        clean_frames(frames=1000), episode_streams=[_moving_actions(1000)]
    )
    windows, _sampled = vision._layout(ctx, False)
    plan = _plan_cuts(ctx)
    assert not isinstance(plan, MetricResult)
    cut_only = [
        (start, end)
        for start, end in vision._merge(windows + plan.windows, 1000)
        if not any(s < end and start < e for s, e in windows)
    ]
    # A vision window after the failure, which the pass never reaches.
    failing_at = next(
        start for start, _end in cut_only if any(s > start for s, _e in windows)
    )
    clean = sharpness_score(ctx)
    vision.clear_cache()

    gray_windows = SyntheticFrames.gray_windows
    calls: list[int] = []

    def failing(self, windows, size, native=frozenset(), **caps):
        calls.append(1)
        counts = [0] * len(windows)
        for item in gray_windows(self, windows, size, native, **caps):
            index = windows[item[0]][0] + counts[item[0]]
            counts[item[0]] += 1
            if len(calls) == 1 and index == failing_at:
                raise DecodeFailed("corrupt packet")
            yield item

    monkeypatch.setattr(SyntheticFrames, "gray_windows", failing)
    result = sharpness_score(ctx)

    assert len(calls) > 1
    assert (result.value, result.evidence) == (clean.value, clean.evidence)


def test_without_an_action_stream_only_the_sampled_frames_are_decoded(monkeypatch):
    """With no freeze to look for, blur and exposure decode their own frames alone."""

    calls = _count_decodes(monkeypatch)
    ctx = stream_context(clean_frames(frames=1000))

    sharpness_score(ctx)
    exposure_shift_pct(ctx)

    assert calls == [10]


def test_new_frame_metrics_decode_nothing_extra(monkeypatch):
    """The four frame-statistic metrics decode no frame beyond the shared sample."""

    calls = _count_decodes(monkeypatch)
    ctx = stream_context(clean_frames(frames=1000))

    for func in (
        sharpness_score,
        exposure_shift_pct,
        blank_frame_pct,
        contrast_score,
        dark_frame_pct,
        bright_frame_pct,
    ):
        func(ctx)

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


def _uniform_stream(level: int) -> Stream:
    """30 frames of 16x16x3 at one gray level."""

    frames = [numpy.full((16, 16, 3), level, numpy.uint8) for _ in range(30)]
    return clean_frames(frames=30).model_copy(
        update={"payload": SyntheticFrames(frames=frames)}
    )


def test_blank_frame_pct_reads_100_on_blank_frames():
    """A stream blank throughout reads 100, and exposure_shift_pct still counts it."""

    ctx = stream_context(_uniform_stream(128))

    assert blank_frame_pct(ctx).value == 100
    assert exposure_shift_pct(ctx).value == 100


def test_blank_frame_pct_counts_only_blank_frames():
    """Blank frames spliced into a clean stream are counted and located."""

    clean = clean_frames(frames=100)
    assert isinstance(clean.payload, SyntheticFrames)
    grey = [frame * 0 + 128 for frame in clean.payload.frames]

    result = blank_frame_pct(stream_context(_with_frames(clean, grey, 20, 60)))

    assert result.value == pytest.approx(40.0)
    assert all(20 <= index < 60 for index in result.evidence["bad_indices"])
    intervals = [(i.start, i.end_exclusive) for i in result.support.intervals]
    assert intervals == [(i, i + 1) for i in result.evidence["bad_indices"]]


def test_contrast_score_is_low_on_a_flat_stream():
    """Compressing a stream's gray levels to a narrow band lowers its contrast score."""

    clean = clean_frames(frames=30)
    assert isinstance(clean.payload, SyntheticFrames)
    flat = clean.model_copy(
        update={
            "payload": SyntheticFrames(
                frames=[
                    (frame * (10 / 255) + 120).astype(numpy.uint8)
                    for frame in clean.payload.frames
                ]
            )
        }
    )

    low = contrast_score(stream_context(flat)).value
    high = contrast_score(stream_context(clean)).value
    assert low is not None and high is not None
    assert low < 15
    assert high > 50


def test_a_blank_dark_stream_is_flagged_by_both():
    """A uniformly dark camera is dark and blank, so exposure_shift_pct fails it too."""

    ctx = stream_context(_uniform_stream(2))

    assert dark_frame_pct(ctx).value == 100
    assert bright_frame_pct(ctx).value == 0
    assert exposure_shift_pct(ctx).value == 100


def test_a_stream_dim_throughout_is_flagged_but_reads_clean_on_shift():
    """A camera dimmed to a tenth reads 100 dark, though few of its pixels are crushed.

    Its P95 sits near 20 and under a sixth of its pixels at or below the dark level,
    both far from the rules' thresholds;
    exposure_shift_pct reads it clean, since every frame is dimmed alike.
    """

    clean = clean_frames(frames=30)
    assert isinstance(clean.payload, SyntheticFrames)
    dim = clean.model_copy(
        update={
            "payload": SyntheticFrames(
                frames=[
                    (frame * 0.1).astype(numpy.uint8) for frame in clean.payload.frames
                ]
            )
        }
    )
    ctx = stream_context(dim)

    assert dark_frame_pct(stream_context(clean)).value == 0
    assert dark_frame_pct(ctx).value == 100
    assert blank_frame_pct(ctx).value == 0
    assert exposure_shift_pct(ctx).value == 0


def test_a_stream_mostly_crushed_but_not_dim_is_flagged():
    """A frame three quarters black is dark, though its P95 is bright."""

    frame = numpy.zeros((16, 16, 3), numpy.uint8)
    frame[12:] = 200
    crushed = clean_frames(frames=30).model_copy(
        update={"payload": SyntheticFrames(frames=[frame.copy() for _ in range(30)])}
    )

    assert dark_frame_pct(stream_context(crushed)).value == 100


def test_bright_frame_pct_flags_a_white_stream():
    """A stream white in every frame is wholly bright and not dark."""

    ctx = stream_context(_uniform_stream(255))

    assert bright_frame_pct(ctx).value == 100
    assert dark_frame_pct(ctx).value == 0


@pytest.mark.parametrize(
    "func", [blank_frame_pct, contrast_score, dark_frame_pct, bright_frame_pct]
)
def test_new_frame_metrics_are_report_only(func):
    """Each new frame metric is reported, and the default policy grades none of them."""

    stream = clean_frames(frames=30)

    assert func(stream_context(stream)).status == MetricStatus.REPORT_ONLY
    graded, _findings = grade_stream(
        stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
    )
    result = graded.metrics[func.__name__]
    assert result.status == MetricStatus.REPORT_ONLY
    assert result.evidence["ungraded_reason"] == "report_only by policy design"


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
    monkeypatch.setattr(
        VideoPayload,
        "header",
        lambda self: pytest.fail("metadata tier read the header"),
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
        "camera_stream_present",
        "video_format_consistency",
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

    for func in (
        sharpness_score,
        exposure_shift_pct,
        exposure_level,
        blank_frame_pct,
        contrast_score,
        dark_frame_pct,
        bright_frame_pct,
    ):
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
    assert "worst_index" in results[0].evidence
    assert "thumbnail_png_base64" not in results[0].evidence
    assert results[1].status == MetricStatus.GOOD


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


def _compare(
    episodes: list[GradedEpisode], **options
) -> tuple[list[CameraSummary], list[Finding]]:
    """Compare the cameras of graded episodes, through their camera samples.

    Parameters
    ----------
    episodes : list[GradedEpisode]
        The graded episodes.
    **options
        Passed on to `compare_cameras`.

    Returns
    -------
    tuple of (list of CameraSummary, list of Finding)
        What `compare_cameras` returns.
    """

    return compare_cameras(
        [sample for e in episodes for sample in camera_samples(e)], **options
    )


def _visual_variation(streams: list[Stream]):
    """Grade each of `streams` as one episode; return the camera's visual variation."""

    footage = FootageVariation()
    episodes = [
        grade_episode(
            Episode(id=f"episode_{index}", streams=[stream]),
            adapter="test",
            adapter_confidence=1.0,
            policy=load_default_policy(),
            dictionary=load_default_dictionary(),
            footage=footage,
        )[0]
        for index, stream in enumerate(streams)
    ]
    (summary,), _findings = _compare(episodes, grades_vision=True, footage=footage)
    return summary.visual_variation


def test_noisy_copies_vary_less_than_distinct_episodes():
    """50 copies of one episode with 1% noise vary far less than 50 distinct ones."""

    base = clean_frames(frames=30, height=128, width=128, seed=0)
    assert isinstance(base.payload, SyntheticFrames)
    frames = base.payload.frames
    rng = numpy.random.default_rng(42)
    noisy = [
        base.model_copy(
            update={
                "payload": SyntheticFrames(
                    frames=[
                        numpy.clip(frame + rng.normal(0, 2.55, frame.shape), 0, 255)
                        .round()
                        .astype(numpy.uint8)
                        for frame in frames
                    ]
                )
            }
        )
        for _ in range(50)
    ]
    distinct = [
        clean_frames(frames=30, height=128, width=128, seed=1000 * index)
        for index in range(50)
    ]

    copies, varied = _visual_variation(noisy), _visual_variation(distinct)
    assert copies is not None and varied is not None
    assert copies.value is not None and varied.value is not None
    assert copies.episodes == varied.episodes == 50
    assert copies.value * 10 < varied.value


def test_footage_reuses_the_vision_read(monkeypatch):
    """The footage vector comes from the read the vision metrics already made."""

    calls = []
    read = vision._pass

    def counting(*args, **kwargs):
        calls.append(1)
        return read(*args, **kwargs)

    monkeypatch.setattr(vision, "_pass", counting)
    stream = clean_frames(frames=30, height=128, width=128)

    footage = FootageVariation()

    graded, _findings = grade_stream(
        stream,
        policy=load_default_policy(),
        is_regular=True,
        episode_id="episode_0",
        category=None,
        footage=footage,
    )
    vector = footage_vector(stream_context(stream))

    assert len(calls) == 1
    assert footage.result(camera_key(graded)).episodes == 1
    assert vector is not None and len(vector) == 10 * 16 * 16


def test_full_scan_pools_the_sampled_frames():
    """A full scan pools the frames a sample takes, so its vector matches a sample's."""

    stream = clean_frames(frames=30, height=128, width=128)

    sampled = footage_vector(stream_context(stream))
    scanned = footage_vector(stream_context(stream, full_frame_scan=True))

    assert sampled is not None and scanned is not None
    assert numpy.array_equal(sampled, scanned)


def test_an_incomplete_read_is_not_counted():
    """A read the decode cap cuts short misses sampled frames, so it counts for none."""

    stream = clean_frames(frames=30, height=128, width=128)
    counted = {}
    for name, spec in (
        ("capped", VisionSpec(sample_frames=2, max_decode_frames=3)),
        ("whole", None),
    ):
        vision.clear_cache()
        footage = FootageVariation()
        graded, _findings = grade_stream(
            stream,
            policy=load_default_policy(),
            is_regular=True,
            episode_id="episode_0",
            category=None,
            vision=spec,
            footage=footage,
        )
        counted[name] = footage.result(camera_key(graded))

    assert counted["capped"].episodes == 0
    assert counted["capped"].reason == _TOO_FEW_FOOTAGE
    assert counted["whole"].episodes == 1
