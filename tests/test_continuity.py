"""Verifies the metrics that check camera footage against the robot's own signals."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import polars as pl
import pytest

# Internal
from kalanos.analysis.metrics import continuity, vision
from kalanos.analysis.metrics.continuity import (
    _EDGE,
    _plan_cuts,
    cut_count,
    dropped_chunk_count,
    video_control_lag_ms,
)
from kalanos.analysis.metrics.vision import sharpness_score
from kalanos.analysis.models.diagnostics import VisionSpec
from kalanos.analysis.models.domain import FramePayload, Stream
from kalanos.analysis.models.metrics import MetricResult, MetricStatus
from kalanos.assets.policy import load_default_policy
from kalanos.testing import (
    check_metric,
    clean_frames,
    clean_recording,
    cut_frames,
    drop_chunk,
    shift_video,
    stream_context,
)
from kalanos.testing.injectors import SyntheticFrames


numpy = pytest.importorskip("numpy")


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# An unsmoothed walk jumps from a still step to a long one from frame to frame,
# which reads as a cut; a robot's speed changes over several frames.
_SMOOTH_FRAMES = 9

# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _tracking(
    samples: int = 300,
    seed: int = 42,
    control_hz: float = 30.0,
    cut: tuple[float, int] | None = None,
) -> tuple[Stream, Stream, Stream]:
    """A 30 Hz camera watching a square the state moves, with its state and action.

    State channel 0 is a random walk averaged over `_SMOOTH_FRAMES`,
    and action equals it.
    Each 64x64 frame is mid-gray, with a 12x12 white square
    whose x position follows the walk, scaled to [0, 52],
    drawn to a fraction of a pixel,
    so the footage moves exactly when, and as much as, the robot does.

    Parameters
    ----------
    samples : int
        How many frames the camera carries.
    seed : int
        The seed of the walk.
    control_hz : float
        The rate of state and action, in whole Hz.
        The walk is built at the least common multiple of it and 30,
        and each stream samples it at its own rate.
    cut : tuple of (float, int) or None
        `(start, length)` frames to remove from the walk before anything is built,
        so every stream is cut by time at its own rate.
        `start` may fall between two frames, as a real cut does.

    Returns
    -------
    tuple of (Stream, Stream, Stream)
        `(video, state, action)`.
    """

    base_hz = math.lcm(30, round(control_hz))
    per_frame = base_hz // 30
    per_row = base_hz // round(control_hz)
    removed = cut[1] if cut else 0
    rows = (samples + removed) * per_frame
    width = _SMOOTH_FRAMES * per_frame
    walk = numpy.random.default_rng(seed).normal(size=rows + width).cumsum()
    walk = numpy.convolve(walk, numpy.ones(width) / width, mode="valid")[:rows]
    if cut:
        start, length = cut
        first = round(start * per_frame)
        walk = numpy.delete(walk, numpy.arange(first, first + length * per_frame))
    x = (walk - walk.min()) / (walk.max() - walk.min()) * 52
    readings = walk[::per_row]

    columns = numpy.arange(64)
    frames = []
    for index in range(samples):
        left = x[index * per_frame]
        # The share of each pixel column the square covers.
        cover = numpy.clip(
            numpy.minimum(columns + 1, left + 12) - numpy.maximum(columns, left), 0, 1
        )
        frame = numpy.full((64, 64, 3), 128.0)
        frame[26:38] += 127 * cover[None, :, None]
        frames.append(numpy.round(frame).astype(numpy.uint8))
    video = clean_frames(frames=samples, height=64, width=64).model_copy(
        update={"payload": SyntheticFrames(frames=frames)}
    )

    def control(taxonomy_type: str) -> Stream:
        """A control stream at `control_hz` whose one channel is the walk."""

        stream = clean_recording(
            hz=control_hz,
            samples=len(readings),
            channels=("x",),
            taxonomy_type=taxonomy_type,
        )
        return stream.model_copy(
            update={"payload": FramePayload(frame=pl.DataFrame({"x": readings}))}
        )

    return (
        video,
        control("unmapped.observation.state"),
        control("action.joint_position_command"),
    )


def _largest_move(state: Stream, length: int) -> int:
    """The first frame of the `length` frames over which the state moves furthest.

    A drop is visible only when the robot ends it somewhere else,
    so a drop test places it where the walk travels.
    The search keeps clear of `_EDGE` frames at either end, where nothing is searched.
    """

    assert isinstance(state.payload, FramePayload)
    walk = state.payload.frame["x"].to_numpy()
    starts = numpy.arange(_EDGE + 1, len(walk) - length - _EDGE - 1)
    return int(starts[numpy.argmax(numpy.abs(walk[starts + length] - walk[starts]))])


def _context(video: Stream, *controls: Stream, full_frame_scan: bool = False):
    """The camera's context, with `controls` and the camera as its episode."""

    return stream_context(
        video, episode_streams=[video, *controls], full_frame_scan=full_frame_scan
    )


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(autouse=True)
def _empty_vision_cache():
    """Start every test with no frames sampled, so no test reads another's."""

    vision.clear_cache()


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_cut_count_locates_a_cut_during_motion():
    """Forty frames cut from footage and controls alike read as one cut at frame 150."""

    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)

    result = cut_count(_context(cut_video, *cut_controls))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 1
    [index] = result.evidence["cut_indices"]
    assert abs(index - 150) <= 2
    assert [(i.start, i.end_exclusive) for i in result.support.intervals] == [
        (index, index + 1)
    ]


def test_cut_count_is_quiet_on_clean_footage():
    """Uncut footage reads no cut and supports no frame."""

    result = cut_count(_context(*_tracking()))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 0
    assert result.support.intervals == []


def test_cut_count_without_controls_is_not_applicable():
    """A camera alone has no control jump to look for a cut at."""

    video, _state, _action = _tracking()

    result = cut_count(_context(video))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "state or action" in result.evidence["reason"]


def test_a_cut_is_found_when_control_runs_at_twice_the_frame_rate():
    """60 Hz controls are placed on the 30 Hz camera by their timestamps."""

    result = cut_count(_context(*_tracking(control_hz=60.0, cut=(150, 40))))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 1
    [index] = result.evidence["cut_indices"]
    assert abs(index - 150) <= 2


def test_a_cut_between_frames_is_found_when_control_runs_at_50_hz():
    """A 50 Hz control step that ends nearer the frame before the jump still finds it.

    The cut starts a fifth of a frame after frame 151,
    so the first frame after it is 152,
    while the control row after it lies nearer frame 151.
    """

    result = cut_count(_context(*_tracking(control_hz=50.0, cut=(151.2, 40))))

    assert result.value == 1
    assert result.evidence["cut_indices"] == [152]


def test_a_mostly_still_stream_does_not_hide_a_cut():
    """A gripper that moves at three steps does not crowd the cut out."""

    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)
    rows = len(cut_video.timestamps)
    opening = numpy.zeros(rows)
    opening[50:], opening[100:], opening[200:] = 1.0, 0.5, 0.8
    gripper = clean_recording(
        hz=30.0, samples=rows, channels=("g",), taxonomy_type="unmapped.gripper"
    ).model_copy(update={"payload": FramePayload(frame=pl.DataFrame({"g": opening}))})

    result = cut_count(_context(cut_video, *cut_controls, gripper))

    assert result.value == 1
    [index] = result.evidence["cut_indices"]
    assert abs(index - 150) <= 2


def _reward(rows: int) -> Stream:
    """A reward that holds still, drifts slightly at 20 steps and jumps at 3."""

    reward = numpy.zeros(rows)
    for step in range(20, 280, 13):
        reward[step:] += 0.001
    reward[60:], reward[110:], reward[220:] = 1.0, 0.2, 0.9
    return clean_recording(
        hz=30.0, samples=rows, channels=("r",), taxonomy_type="reward.frame_reward"
    ).model_copy(update={"payload": FramePayload(frame=pl.DataFrame({"r": reward}))})


def test_a_reward_does_not_hide_a_cut():
    """Reward jumps follow the task, not the robot, so none is a cut candidate."""

    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)
    reward = _reward(len(cut_video.timestamps))

    result = cut_count(_context(cut_video, *cut_controls, reward))

    assert result.value == 1
    [index] = result.evidence["cut_indices"]
    assert abs(index - 150) <= 2


def test_cut_count_passes_the_contract():
    """cut_count holds every property `check_metric` demands, and tells a cut apart."""

    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)

    check_metric(
        cut_count,
        clean=_context(video, state, action),
        defective=_context(cut_video, *cut_controls),
        policy=load_default_policy(),
    )


def test_cut_count_decodes_nothing_after_the_shared_read(monkeypatch):
    """cut_count takes its windows from the read the vision metrics made."""

    calls: list[int] = []
    gray_windows = SyntheticFrames.gray_windows

    def counting(self, windows, size, native=frozenset(), **caps):
        calls.append(0)
        for item in gray_windows(self, windows, size, native, **caps):
            calls[-1] += 1
            yield item

    monkeypatch.setattr(SyntheticFrames, "gray_windows", counting)
    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)
    ctx = _context(cut_video, *cut_controls)

    sharpness_score(ctx)
    result = cut_count(ctx)

    assert result.status == MetricStatus.REPORT_ONLY
    assert len(calls) == 1


def test_cut_count_matches_its_own_read():
    """The shared read holds the thumbnails cut_count's own read of its windows does."""

    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)
    ctx = _context(cut_video, *cut_controls)
    plan = _plan_cuts(ctx)
    assert not isinstance(plan, MetricResult)
    assert isinstance(cut_video.payload, SyntheticFrames)

    read = vision._shared_read(ctx)
    alone, failure = continuity._window_thumbnails(cut_video.payload, plan.windows)

    assert not isinstance(read, MetricResult)
    assert (read.cut_failure, failure) == (None, None)
    assert read.cut.keys() == alone.keys()
    assert all(numpy.array_equal(read.cut[index], alone[index]) for index in alone)
    assert cut_count(ctx).value == 1


def test_the_decode_cap_counts_cut_frames():
    """A decode cap that stops the shared read inside a cut window stops cut_count."""

    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)
    ctx = _context(cut_video, *cut_controls)
    plan = _plan_cuts(ctx)
    assert not isinstance(plan, MetricResult)
    # The whole stream is one vision window, so the cap is the index it stops at.
    capped = ctx.model_copy(
        update={"vision": VisionSpec(max_decode_frames=plan.windows[0][0] + 1)}
    )

    result = cut_count(capped)

    assert result.evidence["decode_error"] == "max_decode_frames reached"


def test_a_padded_drop_is_located():
    """Twenty frames lost from the footage alone read as one drop where they were."""

    video, state, action = _tracking()
    start = _largest_move(state, 20)
    dropped = drop_chunk(video, start=start, length=20, pad=True)

    result = dropped_chunk_count(_context(dropped, state, action, full_frame_scan=True))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 1
    assert abs(result.evidence["drop_index"] - start) <= 2
    [interval] = result.support.intervals
    assert interval.start == result.evidence["drop_index"]


def test_a_cut_is_not_read_as_a_drop():
    """A cut jumps the state with the footage, so it is no drop."""

    video, state, action = _tracking()
    cut_video, cut_controls = cut_frames(video, [state, action], start=150, length=40)

    result = dropped_chunk_count(
        _context(cut_video, *cut_controls, full_frame_scan=True)
    )

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 0
    assert result.evidence["drop_index"] is None


def test_dropped_chunk_count_needs_a_full_scan():
    """Sampled windows cannot place a drop, and the reason names the flag to set."""

    result = dropped_chunk_count(_context(*_tracking()))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "--full-frame-scan" in result.evidence["reason"]


def test_dropped_chunk_count_without_state_is_not_applicable():
    """Action alone cannot tell a drop from a cut."""

    video, _state, action = _tracking()

    result = dropped_chunk_count(_context(video, action, full_frame_scan=True))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "state" in result.evidence["reason"]


def test_a_reward_is_not_state_to_a_drop():
    """A reward cannot tell a drop from a cut, so with action it leaves no state."""

    video, _state, action = _tracking()
    reward = _reward(len(video.timestamps))

    result = dropped_chunk_count(_context(video, action, reward, full_frame_scan=True))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "state" in result.evidence["reason"]


def test_dropped_chunk_count_with_no_step_clear_of_the_ends_is_not_applicable():
    """Twenty frames leave no step outside the ends, where nothing is searched."""

    video, state, action = _tracking(samples=2 * _EDGE)

    result = dropped_chunk_count(_context(video, state, action, full_frame_scan=True))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "ends" in result.evidence["reason"]


def test_dropped_chunk_count_passes_the_contract():
    """dropped_chunk_count holds every property `check_metric` demands."""

    video, state, action = _tracking()
    dropped = drop_chunk(video, start=_largest_move(state, 20), length=20, pad=True)

    check_metric(
        dropped_chunk_count,
        clean=_context(video, state, action, full_frame_scan=True),
        defective=_context(dropped, state, action, full_frame_scan=True),
        policy=load_default_policy(),
    )


def test_a_known_video_offset_is_reported():
    """Footage shown 3 frames early reads 3 frames further ahead than when clean."""

    video, state, action = _tracking()
    clean = video_control_lag_ms(_context(video, state, action, full_frame_scan=True))
    vision.clear_cache()
    shifted = video_control_lag_ms(
        _context(shift_video(video, frames=3), state, action, full_frame_scan=True)
    )

    assert clean.status == shifted.status == MetricStatus.REPORT_ONLY
    lead = shifted.evidence["lag_frames"] - clean.evidence["lag_frames"]
    assert lead == pytest.approx(3, abs=1)


def test_the_value_is_against_state_when_state_exists():
    """With state and action both present, the value is the lag against state."""

    video, state, action = _tracking()

    result = video_control_lag_ms(_context(video, state, action, full_frame_scan=True))

    assert result.evidence["reference"] == "state"
    assert result.value == result.evidence["lag_state_ms"]
    assert result.evidence["lag_action_ms"] is not None


def test_the_value_falls_back_to_action_without_state():
    """Without state, the value is the lag against action."""

    video, _state, action = _tracking()

    result = video_control_lag_ms(_context(video, action, full_frame_scan=True))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.evidence["reference"] == "action"
    assert result.value == result.evidence["lag_action_ms"]
    assert result.evidence["lag_state_ms"] is None
    assert result.value == pytest.approx(result.evidence["lag_frames"] * 1000 / 30)


def test_the_value_falls_back_to_action_when_state_holds_still():
    """A state that never moves times nothing, so the value is against action."""

    video, state, action = _tracking()
    rows = len(state.timestamps)
    still = state.model_copy(
        update={"payload": FramePayload(frame=pl.DataFrame({"x": numpy.zeros(rows)}))}
    )

    result = video_control_lag_ms(_context(video, still, action, full_frame_scan=True))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.evidence["reference"] == "action"
    assert result.evidence["lag_state_ms"] is None
    assert result.value == result.evidence["lag_action_ms"]


def test_video_control_lag_needs_a_full_scan():
    """Sampled windows cannot time the footage, and the reason names the flag to set."""

    result = video_control_lag_ms(_context(*_tracking()))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "--full-frame-scan" in result.evidence["reason"]


def test_video_control_lag_on_a_short_episode_is_not_applicable():
    """Eight frames are too few to correlate at any lag, and the metric says so."""

    video, state, action = _tracking(samples=8)

    result = video_control_lag_ms(_context(video, state, action, full_frame_scan=True))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "correlates" in result.evidence["reason"]


def test_video_control_lag_passes_the_contract():
    """video_control_lag_ms holds every property `check_metric` demands."""

    video, state, action = _tracking()

    check_metric(
        video_control_lag_ms,
        clean=_context(video, state, action, full_frame_scan=True),
        defective=_context(
            shift_video(video, frames=3), state, action, full_frame_scan=True
        ),
        policy=load_default_policy(),
    )
