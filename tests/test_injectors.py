"""Verifies the synthetic recording builders and the twelve defect injectors."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import pytest

# Internal
from kalanos.analysis.models.domain import FramePayload
from kalanos.testing import (
    Defect,
    add_noise,
    apply_defect,
    clean_frames,
    clean_recording,
    clean_taxels,
    drift_channel,
    drop_samples,
    freeze_frames,
    jitter_clock,
    kill_taxels,
    null_run,
    saturate_channel,
    skew_unloading,
    spike_channel,
    stick_channel,
    stream_context,
    stretch_clock,
)
from kalanos.testing.injectors import SyntheticFrames


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_clean_recording_is_regular_and_full():
    """The built stream has uniform gaps, no nulls, and one column per channel."""

    stream = clean_recording(hz=100.0, samples=100)

    assert isinstance(stream.payload, FramePayload)
    frame = stream.payload.frame
    assert frame.width == len(stream.channels)
    assert frame.height == len(stream.timestamps) == 100
    assert frame.null_count().sum_horizontal().item() == 0

    gaps = stream.timestamps.diff().drop_nulls().to_list()
    assert gaps == pytest.approx([0.01] * (len(gaps)), abs=1e-9)


def test_a_stuck_channel_holds_one_value_and_leaves_its_siblings_alone():
    """The stuck window is constant; the rest of the channel and its siblings aren't."""

    clean = clean_recording()
    stuck = stick_channel(clean, "tcp_pose_x_mm", start=20, length=40)

    assert isinstance(stuck.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    stuck_window = stuck.payload.frame["tcp_pose_x_mm"][20:60].to_list()
    assert len(set(stuck_window)) == 1

    outside_window = stuck.payload.frame["tcp_pose_x_mm"][60:].to_list()
    clean_outside = clean.payload.frame["tcp_pose_x_mm"][60:].to_list()
    assert outside_window == clean_outside

    for other in ("tcp_pose_y_mm", "tcp_pose_z_mm"):
        assert (
            stuck.payload.frame[other].to_list() == clean.payload.frame[other].to_list()
        )

    assert stuck.timestamps.to_list() == clean.timestamps.to_list()


def test_dropping_samples_removes_timestamps_and_rows_together():
    """Frame and timestamps shrink together by `count`, staying Stream-valid."""

    clean = clean_recording()
    dropped = drop_samples(clean, start=40, count=10)

    assert isinstance(dropped.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    assert dropped.payload.frame.height == clean.payload.frame.height - 10
    assert len(dropped.timestamps) == len(clean.timestamps) - 10

    surviving = dropped.timestamps.to_list()
    expected = clean.timestamps.to_list()[:40] + clean.timestamps.to_list()[50:]
    assert surviving == expected


def test_jittering_the_clock_keeps_the_timestamps_strictly_increasing():
    """Jitter leaves the clock monotonic but breaks uniform gaps."""

    clean = clean_recording()
    jittered = jitter_clock(clean, milliseconds=2.0)

    times = jittered.timestamps.to_list()
    assert all(b > a for a, b in zip(times, times[1:], strict=False))

    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert len(set(round(gap, 9) for gap in gaps)) > 1

    with pytest.raises(ValueError, match="monotonicity"):
        jitter_clock(clean, milliseconds=1000.0)


def test_a_null_run_nulls_only_its_own_window():
    """The null count equals `length`, and no other channel gains a null."""

    clean = clean_recording()
    nulled = null_run(clean, "tcp_pose_y_mm", start=30, length=10)

    assert isinstance(nulled.payload, FramePayload)
    assert nulled.payload.frame["tcp_pose_y_mm"].null_count() == 10
    assert nulled.payload.frame["tcp_pose_x_mm"].null_count() == 0
    assert nulled.payload.frame["tcp_pose_z_mm"].null_count() == 0
    assert len(nulled.timestamps) == len(clean.timestamps)


def test_stretching_the_clock_ramps_the_gaps_while_the_payload_holds():
    """The clock stretches monotonically from its first gap to its last,
    payload alone."""

    clean = clean_recording()
    stretched = stretch_clock(clean, factor=1.4)

    times = stretched.timestamps.to_list()
    assert all(b > a for a, b in zip(times, times[1:], strict=False))

    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert gaps[-1] > gaps[0]

    assert stretched.payload is clean.payload

    with pytest.raises(ValueError, match="positive"):
        stretch_clock(clean, factor=0)


def test_saturating_a_channel_clips_its_peaks_and_leaves_its_siblings_alone():
    """The clipped channel's range shrinks to the limit; its siblings don't move."""

    clean = clean_recording()
    saturated = saturate_channel(clean, "tcp_pose_x_mm", limit=0.5)

    assert isinstance(saturated.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    saturated_values = saturated.payload.frame["tcp_pose_x_mm"]
    assert saturated_values.max() == pytest.approx(0.5)
    assert saturated_values.min() == pytest.approx(-0.5)
    assert saturated_values.n_unique() < clean.payload.frame["tcp_pose_x_mm"].n_unique()

    for other in ("tcp_pose_y_mm", "tcp_pose_z_mm"):
        assert (
            saturated.payload.frame[other].to_list()
            == clean.payload.frame[other].to_list()
        )
    assert saturated.timestamps.to_list() == clean.timestamps.to_list()


def test_drifting_a_channel_adds_a_monotonic_ramp_and_leaves_its_siblings_alone():
    """The drifted channel's difference from clean is a ramp ending at `total`."""

    clean = clean_recording()
    drifted = drift_channel(clean, "tcp_pose_x_mm", total=1.0)

    assert isinstance(drifted.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    clean_values = clean.payload.frame["tcp_pose_x_mm"].to_list()
    drifted_values = drifted.payload.frame["tcp_pose_x_mm"].to_list()
    difference = [d - c for d, c in zip(drifted_values, clean_values, strict=False)]

    assert difference[0] == pytest.approx(0.0)
    assert difference[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(difference, difference[1:], strict=False))

    for other in ("tcp_pose_y_mm", "tcp_pose_z_mm"):
        assert (
            drifted.payload.frame[other].to_list()
            == clean.payload.frame[other].to_list()
        )


def test_freezing_frames_repeats_the_predecessor_across_its_own_window():
    """The frozen window's frames all equal their predecessor; the rest don't."""

    numpy = pytest.importorskip("numpy")

    clean = clean_frames(frames=30)
    frozen = freeze_frames(clean, start=10, length=10)

    assert isinstance(frozen.payload, SyntheticFrames) and isinstance(
        clean.payload, SyntheticFrames
    )
    predecessor = frozen.payload.frames[9]
    for frame in frozen.payload.frames[10:20]:
        assert numpy.array_equal(frame, predecessor)

    assert not numpy.array_equal(frozen.payload.frames[20], predecessor)
    assert not numpy.array_equal(frozen.payload.frames[9], frozen.payload.frames[8])

    assert len(frozen.payload.frames) == len(clean.payload.frames)
    assert frozen.timestamps.to_list() == clean.timestamps.to_list()


def test_apply_defect_builds_frozen_frames_on_a_channelless_image_stream():
    """A `Kind.IMAGE` stream has no channels, and `FROZEN_FRAMES` needs none."""

    pytest.importorskip("numpy")

    ctx = stream_context(clean_frames(frames=30))
    injected = apply_defect(ctx, Defect.FROZEN_FRAMES)

    assert isinstance(injected, type(ctx))
    assert isinstance(injected.stream.payload, SyntheticFrames) and isinstance(
        ctx.stream.payload, SyntheticFrames
    )
    assert len(injected.stream.payload.frames) == len(ctx.stream.payload.frames)


def test_a_settled_clean_recording_ends_static_and_is_otherwise_unchanged():
    """`settle=20` holds the tail constant; everything before it matches `settle=0`."""

    unsettled = clean_recording(settle=0)
    settled = clean_recording(settle=20)

    assert isinstance(settled.payload, FramePayload) and isinstance(
        unsettled.payload, FramePayload
    )
    for name in settled.channels:
        tail = settled.payload.frame[name.name][-20:].to_list()
        assert len(set(tail)) == 1

        head = settled.payload.frame[name.name][:-20].to_list()
        unsettled_head = unsettled.payload.frame[name.name][:-20].to_list()
        assert head == unsettled_head

    assert settled.timestamps.to_list() == unsettled.timestamps.to_list()


def test_clean_taxels_builds_one_rise_then_fall_aggregate_per_cell():
    """Every cell traces a half-sine that peaks at the recording's midpoint."""

    stream = clean_taxels(samples=101, cells=4)

    assert isinstance(stream.payload, FramePayload)
    frame = stream.payload.frame
    assert frame.width == 4
    assert [channel.name for channel in stream.channels] == [
        "taxel_00",
        "taxel_01",
        "taxel_02",
        "taxel_03",
    ]

    for name in frame.columns:
        values = frame[name].to_list()
        peak_index = max(range(len(values)), key=lambda index: values[index])
        assert peak_index == len(values) // 2
        assert values[0] == pytest.approx(0.0, abs=1e-9)
        assert values[-1] == pytest.approx(0.0, abs=1e-9)

    assert frame["taxel_00"].to_list() != frame["taxel_01"].to_list()


def test_spike_channel_moves_exactly_one_sample_and_leaves_its_siblings_alone():
    """The spiked sample jumps by `magnitude` standard deviations; nothing else
    moves."""

    clean = clean_recording()
    spiked = spike_channel(clean, "tcp_pose_x_mm", magnitude=20.0)

    assert isinstance(spiked.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    clean_values = clean.payload.frame["tcp_pose_x_mm"].to_list()
    spiked_values = spiked.payload.frame["tcp_pose_x_mm"].to_list()
    midpoint = len(clean_values) // 2

    for index, (before, after) in enumerate(
        zip(clean_values, spiked_values, strict=True)
    ):
        if index == midpoint:
            assert after != before
        else:
            assert after == pytest.approx(before)

    for other in ("tcp_pose_y_mm", "tcp_pose_z_mm"):
        assert (
            spiked.payload.frame[other].to_list()
            == clean.payload.frame[other].to_list()
        )
    assert spiked.timestamps.to_list() == clean.timestamps.to_list()


def test_add_noise_raises_step_to_step_variation_without_moving_the_mean():
    """Alternating +/-amplitude raises the step size but averages out to zero."""

    clean = clean_recording()
    noisy = add_noise(clean, "tcp_pose_x_mm", amplitude=0.2)

    assert isinstance(noisy.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    clean_values = clean.payload.frame["tcp_pose_x_mm"].to_list()
    noisy_values = noisy.payload.frame["tcp_pose_x_mm"].to_list()
    difference = [n - c for n, c in zip(noisy_values, clean_values, strict=True)]

    assert sum(difference) / len(difference) == pytest.approx(0.0, abs=1e-9)
    assert set(round(d, 9) for d in difference) == {0.2, -0.2}

    for other in ("tcp_pose_y_mm", "tcp_pose_z_mm"):
        assert (
            noisy.payload.frame[other].to_list() == clean.payload.frame[other].to_list()
        )


def test_kill_taxels_leaves_exactly_count_channels_constant():
    """The first `count` cells hold their row-0 value; the rest keep tracing
    the cycle."""

    clean = clean_taxels(cells=8)
    killed = kill_taxels(clean, count=3)

    assert isinstance(killed.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    dead_names = [channel.name for channel in clean.channels[:3]]
    alive_names = [channel.name for channel in clean.channels[3:]]

    for name in dead_names:
        assert killed.payload.frame[name].n_unique() == 1

    for name in alive_names:
        assert (
            killed.payload.frame[name].to_list() == clean.payload.frame[name].to_list()
        )


def test_skew_unloading_changes_the_second_half_and_not_the_first():
    """Scaling only the second half breaks the loading/unloading retrace."""

    clean = clean_taxels(samples=100, cells=2)
    skewed = skew_unloading(clean, factor=1.5)

    assert isinstance(skewed.payload, FramePayload) and isinstance(
        clean.payload, FramePayload
    )
    midpoint = 100 // 2

    for name in ("taxel_00", "taxel_01"):
        clean_values = clean.payload.frame[name].to_list()
        skewed_values = skewed.payload.frame[name].to_list()
        assert skewed_values[:midpoint] == pytest.approx(clean_values[:midpoint])
        assert skewed_values[midpoint:] == pytest.approx(
            [value * 1.5 for value in clean_values[midpoint:]]
        )
