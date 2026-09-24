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
    mean_jerk_norm,
    still_drift,
)
from kalanos.analysis.models.domain import Channel, Episode, FramePayload, Stream
from kalanos.analysis.models.metrics import ChannelContext, EpisodeContext, MetricStatus
from kalanos.assets.policy import load_default_policy
from kalanos.testing import (
    check_metric,
    clean_recording,
    null_run,
    saturate_channel,
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
