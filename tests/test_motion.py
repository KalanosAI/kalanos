"""Verifies the motion family's own logic, beyond what the contract sweep checks."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import polars as pl
import pytest

# Internal
from kalanos.analysis.metrics.motion import (
    action_chatter,
    energy_proxy,
    hf_vibration_ratio,
    log_dimensionless_jerk,
    mean_jerk_norm,
    still_drift,
    velocity_spike_pct,
)
from kalanos.analysis.metrics.registry import run_channel_metrics
from kalanos.analysis.models.domain import Channel, Episode, FramePayload, Stream
from kalanos.analysis.models.metrics import ChannelContext, EpisodeContext, MetricStatus
from kalanos.assets.policy import load_default_policy
from kalanos.testing import (
    check_metric,
    clean_recording,
    null_run,
    saturate_channel,
    step_channel,
    stick_channel,
    stream_context,
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _scaled(stream: Stream, factor: float) -> Stream:
    """Multiply every channel's values by `factor`, leaving the timestamps alone."""

    assert isinstance(stream.payload, FramePayload)
    frame = stream.payload.frame
    scaled_frame = frame.select(
        [(pl.col(name) * factor).alias(name) for name in frame.columns]
    )
    return stream.model_copy(update={"payload": FramePayload(frame=scaled_frame)})


def _channel_ctx(stream: Stream, name: str = "tcp_pose_x_mm") -> ChannelContext:
    """Build a ChannelContext over one channel of a regularly sampled stream."""

    assert isinstance(stream.payload, FramePayload)
    channel = next(channel for channel in stream.channels if channel.name == name)
    return ChannelContext(
        channel=channel,
        values=stream.payload.frame[name],
        stream=stream_context(stream, is_regular=True),
    )


def _energy_episode(
    *, torque: Stream | None = None, velocity: Stream | None = None
) -> EpisodeContext:
    """Build an EpisodeContext pairing a torque and a velocity stream on one
    instance."""

    torque = torque or clean_recording(
        taxonomy_type="proprio.joint_torque", instance="arm0"
    )
    velocity = velocity or clean_recording(
        taxonomy_type="proprio.joint_velocity", instance="arm0"
    )
    return EpisodeContext(episode=Episode(id="episode", streams=[torque, velocity]))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_mean_jerk_norm_is_scale_free():
    """The same signal scaled ten times reports the same normalised jerk."""

    stream = clean_recording(taxonomy_type="proprio.joint_position")
    scaled = _scaled(stream, 10.0)

    result = mean_jerk_norm(stream_context(stream, is_regular=True))
    scaled_result = mean_jerk_norm(stream_context(scaled, is_regular=True))

    assert result.value == pytest.approx(scaled_result.value, rel=1e-9)


def test_still_drift_needs_a_settled_tail():
    """An unsettled recording declines; a settled one reports a value."""

    unsettled = clean_recording(taxonomy_type="proprio.joint_position")
    settled = clean_recording(taxonomy_type="proprio.joint_position", settle=20)

    unsettled_result = still_drift(stream_context(unsettled, is_regular=True))
    settled_result = still_drift(stream_context(settled, is_regular=True))

    assert unsettled_result.status == MetricStatus.NOT_APPLICABLE
    assert settled_result.status == MetricStatus.REPORT_ONLY
    assert settled_result.value == pytest.approx(0.0, abs=1e-9)


def test_mean_jerk_norm_survives_a_null_run():
    """A run of nulls in one channel does not crash the whole-stream computation."""

    stream = null_run(
        clean_recording(taxonomy_type="proprio.joint_position"), "tcp_pose_x_mm"
    )

    result = mean_jerk_norm(stream_context(stream, is_regular=True))

    assert result.status in (MetricStatus.REPORT_ONLY, MetricStatus.NOT_APPLICABLE)


def test_action_chatter_survives_a_null_run():
    """A run of nulls in one channel does not crash the whole-stream computation."""

    stream = null_run(
        clean_recording(taxonomy_type="proprio.joint_velocity"), "tcp_pose_x_mm"
    )

    result = action_chatter(stream_context(stream, is_regular=True))

    assert result.status in (MetricStatus.REPORT_ONLY, MetricStatus.NOT_APPLICABLE)


def test_still_drift_survives_a_null_run_in_the_tail():
    """A run of nulls reaching into the settled tail does not crash the computation."""

    stream = null_run(
        clean_recording(taxonomy_type="proprio.joint_position", settle=20),
        "tcp_pose_x_mm",
        start=85,
        length=10,
    )

    result = still_drift(stream_context(stream, is_regular=True))

    assert result.status in (MetricStatus.REPORT_ONLY, MetricStatus.NOT_APPLICABLE)


def test_hf_vibration_ratio_is_not_applicable_below_twice_the_cutoff():
    """A sampling rate under twice the cutoff cannot represent anything above it."""

    pytest.importorskip("numpy")

    stream = clean_recording(hz=30.0, samples=64, taxonomy_type="proprio.joint_torque")
    assert isinstance(stream.payload, FramePayload)
    channel = stream.channels[0]
    ctx = ChannelContext(
        channel=channel,
        values=stream.payload.frame[channel.name],
        stream=stream_context(stream, is_regular=True),
    )

    result = hf_vibration_ratio(ctx)

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_energy_proxy_tells_clean_from_a_saturated_torque_stream():
    """energy_proxy's own contract check, since no injector builds an episode defect."""

    saturated_torque = saturate_channel(
        clean_recording(taxonomy_type="proprio.joint_torque", instance="arm0"),
        "tcp_pose_x_mm",
        limit=0.3,
    )

    check_metric(
        energy_proxy,
        clean=_energy_episode(),
        defective=_energy_episode(torque=saturated_torque),
        policy=load_default_policy(),
    )


def test_energy_proxy_is_not_applicable_when_timestamps_disagree():
    """Torque and velocity sampled on different clocks share no timebase to pair
    against."""

    torque = clean_recording(
        taxonomy_type="proprio.joint_torque", instance="arm0", hz=100.0
    )
    velocity = clean_recording(
        taxonomy_type="proprio.joint_velocity", instance="arm0", hz=90.0
    )

    result = energy_proxy(_energy_episode(torque=torque, velocity=velocity))

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_energy_proxy_is_not_applicable_when_declared_axes_share_none():
    """Streams that declare disjoint axes must not fall back to pairing positionally."""

    torque = clean_recording(
        taxonomy_type="proprio.joint_torque",
        instance="arm0",
        channels=["j1", "j2"],
    ).model_copy(
        update={
            "channels": [Channel(name="j1", axis="1"), Channel(name="j2", axis="2")]
        }
    )
    velocity = clean_recording(
        taxonomy_type="proprio.joint_velocity",
        instance="arm0",
        channels=["j7", "j8"],
    ).model_copy(
        update={
            "channels": [Channel(name="j7", axis="7"), Channel(name="j8", axis="8")]
        }
    )

    result = energy_proxy(_energy_episode(torque=torque, velocity=velocity))

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_velocity_spike_pct_fires_on_a_step_and_is_zero_on_a_clean_sine():
    """A sine changes smoothly; a step makes one change no smooth motion makes."""

    clean = clean_recording(taxonomy_type="proprio.joint_position")
    stepped = step_channel(clean, "tcp_pose_x_mm")

    clean_result = velocity_spike_pct(_channel_ctx(clean))
    stepped_result = velocity_spike_pct(_channel_ctx(stepped))

    assert clean_result.value == 0.0
    assert stepped_result.value is not None and stepped_result.value > 0
    assert stepped_result.evidence["sample_indices"] == [50]


def test_velocity_spike_pct_catches_a_step_while_the_joint_holds_still():
    """A step in a joint otherwise held still fires."""

    stream = clean_recording(taxonomy_type="proprio.joint_position")
    assert isinstance(stream.payload, FramePayload)
    held = [0.0] * 50 + [1.0] * 50
    frame = stream.payload.frame.with_columns(pl.Series("tcp_pose_x_mm", held))
    stream = stream.model_copy(update={"payload": FramePayload(frame=frame)})

    result = velocity_spike_pct(_channel_ctx(stream))

    assert result.value is not None and result.value > 0


def test_log_dimensionless_jerk_is_the_same_at_two_playback_speeds():
    """One trajectory at 100 Hz and stretched three times longer agrees to rel=1e-9.

    The agreement is exact up to floating point: dt cancels out, leaving (N - 1)^5.
    """

    normal = clean_recording(taxonomy_type="proprio.joint_position")
    slow = clean_recording(hz=100.0 / 3, taxonomy_type="proprio.joint_position")

    normal_result = log_dimensionless_jerk(_channel_ctx(normal))
    slow_result = log_dimensionless_jerk(_channel_ctx(slow))

    assert normal_result.value == pytest.approx(slow_result.value, rel=1e-9)


def test_log_dimensionless_jerk_is_the_same_at_two_amplitudes():
    """The same trajectory scaled ten times agrees to rel=1e-9."""

    stream = clean_recording(taxonomy_type="proprio.joint_position")

    result = log_dimensionless_jerk(_channel_ctx(stream))
    scaled_result = log_dimensionless_jerk(_channel_ctx(_scaled(stream, 10.0)))

    assert result.value == pytest.approx(scaled_result.value, rel=1e-9)


def test_log_dimensionless_jerk_is_not_applicable_on_a_still_channel_or_a_null_run():
    """A channel that never moves, or one with a gap, has no smoothness to report."""

    stream = clean_recording(taxonomy_type="proprio.joint_position")
    still = stick_channel(stream, "tcp_pose_x_mm", start=0, length=100)
    gapped = null_run(stream, "tcp_pose_x_mm")

    for defective, reason in ((still, "never moves"), (gapped, "null")):
        result = log_dimensionless_jerk(_channel_ctx(defective))

        assert result.status is MetricStatus.NOT_APPLICABLE
        assert reason in result.evidence["reason"]


def test_velocity_spike_and_dimensionless_jerk_need_no_binding_and_skip_other_types():
    """Both gate on joint position only, with no binding needed to run."""

    names = ("velocity_spike_pct", "log_dimensionless_jerk")
    unmapped = clean_recording(taxonomy_type="unmapped.observation.state")
    unbound = clean_recording(taxonomy_type="proprio.joint_position")

    unmapped_results = run_channel_metrics(_channel_ctx(unmapped))
    unbound_results = run_channel_metrics(_channel_ctx(unbound))

    for name in names:
        assert unmapped_results[name].status is MetricStatus.NOT_APPLICABLE
        assert "proprio.joint_position" in unmapped_results[name].evidence["reason"]
        assert unbound_results[name].value is not None, name
