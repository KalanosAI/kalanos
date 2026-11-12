"""Verifies that the metrics reading an angle channel undo its wraps first."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import random

# External
import numpy as np
import polars as pl
import pytest

# Internal
from kalanos.analysis.angles import unwrap
from kalanos.analysis.metrics.integrity import drift, flatline_pct, spike_pct
from kalanos.analysis.metrics.motion import (
    log_dimensionless_jerk,
    max_abs_jerk,
    mean_jerk_norm,
    still_drift,
    velocity_spike_pct,
)
from kalanos.analysis.models.binding import ChannelBinding, Representation
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.metrics import MetricStatus
from kalanos.testing import clean_recording, stream_context

# Local
from helpers import angle_ctx, wrap_angle, yaw_swing


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The channel metrics that unwrap an angle channel, beside snr_db.
_CHANNEL_METRICS = [
    spike_pct,
    drift,
    flatline_pct,
    velocity_spike_pct,
    log_dimensionless_jerk,
]

# The stream metrics that unwrap each angle channel they read.
_STREAM_METRICS = [mean_jerk_norm, max_abs_jerk, still_drift]

# The channel metrics a wrap visibly distorts on the yaw swing.
_DISTORTED = [drift, velocity_spike_pct, log_dimensionless_jerk]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _turn_then_hold() -> list[float]:
    """Turn the yaw up to π, then hold it there, creeping across π as it settles."""

    rng = random.Random(5)
    turn = [(math.pi - 1e-4) * i / 600 for i in range(600)]
    hold = [math.pi - 1e-4 + 1e-6 * i + rng.gauss(0, 1e-7) for i in range(200)]
    return turn + hold


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_unwrap_matches_numpy():
    wrapped = wrap_angle(yaw_swing(1.0), math.pi)
    unwrapped, steps = unwrap(pl.Series(wrapped, dtype=pl.Float64), 2 * math.pi)
    expected = np.unwrap(np.array(wrapped), period=2 * math.pi)
    corrections = np.diff(expected - np.array(wrapped))

    np.testing.assert_allclose(unwrapped.to_numpy(), expected, rtol=0, atol=1e-12)
    assert steps == int(np.count_nonzero(np.round(corrections, 6)))


def test_unwrap_keeps_nan_in_place_and_compares_across_it():
    values = [3.1, float("nan"), -3.1, -3.0]
    unwrapped, steps = unwrap(pl.Series(values), 2 * math.pi)

    assert math.isnan(unwrapped[1])
    assert unwrapped[2] == pytest.approx(-3.1 + 2 * math.pi)
    assert steps == 1


@pytest.mark.parametrize("metric", _CHANNEL_METRICS, ids=lambda m: m.__name__)
def test_wrapped_angle_reads_as_its_unwrapped_motion(metric):
    yaw = yaw_swing(1.0)
    wrapped = metric(angle_ctx(wrap_angle(yaw, math.pi), unit="rad"))
    unwrapped = metric(
        angle_ctx(yaw, unit="rad", representation=Representation.CONTINUOUS)
    )

    assert wrapped.value == pytest.approx(unwrapped.value, rel=1e-9, abs=1e-9)
    assert wrapped.evidence["angle_period"] == 2 * math.pi
    assert wrapped.evidence["n_unwrapped_steps"] > 0


@pytest.mark.parametrize("metric", _DISTORTED, ids=lambda m: m.__name__)
def test_wrap_distorts_the_metric_without_the_angle_binding(metric):
    """Guards the fixture: without unwrapping, these metrics read the wrap."""

    wrapped = wrap_angle(yaw_swing(1.0), math.pi)
    as_angle = metric(angle_ctx(wrapped, unit="rad"))
    as_continuous = metric(
        angle_ctx(wrapped, unit="rad", representation=Representation.CONTINUOUS)
    )

    assert as_continuous.value != pytest.approx(as_angle.value, rel=0.01)


@pytest.mark.parametrize("metric", _CHANNEL_METRICS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("unit", [None, "degrees"])
def test_angle_without_a_known_unit_is_unavailable(metric, unit):
    result = metric(angle_ctx(wrap_angle(yaw_swing(1.0), math.pi), unit=unit))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert result.availability == Availability.UNAVAILABLE
    assert "unit" in result.evidence["reason"]


@pytest.mark.parametrize(
    "metric", _CHANNEL_METRICS + _STREAM_METRICS, ids=lambda m: m.__name__
)
def test_other_channels_gain_no_angle_evidence(metric):
    ctx = angle_ctx(
        _turn_then_hold(), unit="rad", representation=Representation.CONTINUOUS
    )
    result = metric(ctx.stream if metric in _STREAM_METRICS else ctx)

    assert result.value is not None
    assert (
        not {
            "angle_period",
            "n_unwrapped_steps",
            "n_unwrapped_steps_by_channel",
            "skipped_angle_channels",
        }
        & result.evidence.keys()
    )


def test_flatline_reads_both_signs_of_the_bound_as_the_same_angle():
    values = [3.0, math.pi, -math.pi, math.pi, 3.0]
    as_angle = flatline_pct(angle_ctx(values, unit="rad"))
    as_continuous = flatline_pct(
        angle_ctx(values, unit="rad", representation=Representation.CONTINUOUS)
    )

    assert as_angle.evidence["n_unchanged"] == 2
    assert as_continuous.evidence["n_unchanged"] == 0


def test_drift_unwraps_in_time_order_when_rows_are_stored_out_of_order():
    ctx = angle_ctx(wrap_angle(yaw_swing(1.0), math.pi), unit="rad")
    expected = drift(ctx).value
    # drift reads the first and last stored rows as the time span, so they stay put.
    middle = list(range(1, len(ctx.values) - 1))
    random.Random(3).shuffle(middle)
    order = [0, *middle, len(ctx.values) - 1]
    ctx.values = pl.Series([ctx.values[i] for i in order])
    ctx.stream.stream.timestamps = pl.Series(
        [ctx.stream.stream.timestamps[i] for i in order]
    )

    assert drift(ctx).value == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("metric", [mean_jerk_norm, max_abs_jerk])
def test_stream_jerk_reads_a_wrapped_angle_as_its_unwrapped_motion(metric):
    yaw = yaw_swing(1.0)
    wrapped = metric(angle_ctx(wrap_angle(yaw, math.pi), unit="rad").stream)
    unwrapped = metric(
        angle_ctx(yaw, unit="rad", representation=Representation.CONTINUOUS).stream
    )

    assert wrapped.value == pytest.approx(unwrapped.value, rel=1e-6)
    assert wrapped.evidence["n_unwrapped_steps_by_channel"]["value"] > 0


def test_still_drift_reads_a_hold_on_the_bound_as_still():
    wrapped = wrap_angle(_turn_then_hold(), math.pi)
    as_angle = still_drift(angle_ctx(wrapped, unit="rad").stream)
    as_continuous = still_drift(
        angle_ctx(wrapped, unit="rad", representation=Representation.CONTINUOUS).stream
    )

    assert as_angle.value is not None and as_angle.value < 0.01
    assert as_angle.evidence["n_unwrapped_steps_by_channel"]["value"] > 0
    assert as_continuous.value is None or as_continuous.value > 1.0


@pytest.mark.parametrize("metric", _STREAM_METRICS, ids=lambda m: m.__name__)
def test_stream_metrics_leave_out_an_angle_without_a_known_unit(metric):
    result = metric(angle_ctx(_turn_then_hold(), unit=None).stream)

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert result.evidence["reason"] == (
        "every channel is an angle without a known wrap period: value"
    )


@pytest.mark.parametrize("metric", _STREAM_METRICS, ids=lambda m: m.__name__)
def test_stream_metrics_name_the_angle_channel_they_leave_out(metric):
    stream = clean_recording(samples=200, channels=["yaw", "x"], settle=60)
    stream.channels[0].binding = ChannelBinding(
        feature="observation.yaw",
        index=0,
        taxonomy_type="unmapped.observation.yaw",
        representation=Representation.ANGLE,
    )
    result = metric(stream_context(stream))

    assert result.value is not None
    assert result.evidence["skipped_angle_channels"] == {
        "yaw": "angle channel declares no unit, so its wrap period is unknown"
    }
