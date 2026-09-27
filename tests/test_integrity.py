"""Verifies the integrity family's own logic, beyond what the contract sweep checks."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections.abc import Sequence

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.metrics.integrity import (
    drift,
    flatline_pct,
    hysteresis,
    missing_pct,
    snr_db,
    spike_pct,
)
from kalanos.analysis.models.domain import Channel, Clock, FramePayload, Kind, Stream
from kalanos.analysis.models.metrics import ChannelContext, MetricStatus, StreamContext
from kalanos.testing import clean_taxels, skew_unloading, stream_context


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_SOURCE_PATH = UPath("test_integrity.csv")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _channel_ctx(
    values: Sequence[float | None], *, rate_hz: float = 100.0
) -> ChannelContext:
    """Wrap a plain list of values in a ChannelContext, on a regular clock."""

    timestamps = pl.Series("time_s", [index / rate_hz for index in range(len(values))])
    payload = FramePayload(frame=pl.DataFrame({"value": values}))
    stream = Stream(
        taxonomy_type="unmapped.test",
        kind=Kind.SERIES,
        timestamps=timestamps,
        payload=payload,
        source_path=_SOURCE_PATH,
        clock=Clock.CAPTURE,
        channels=[Channel(name="value")],
    )
    stream_ctx = StreamContext(stream=stream, is_regular=True)
    return ChannelContext(
        channel=stream.channels[0], values=payload.frame["value"], stream=stream_ctx
    )


def _taxel_stream_ctx(columns: dict[str, list[float]]) -> StreamContext:
    """Wrap a dict of cell columns in a StreamContext, on a regular 50 Hz clock."""

    n_samples = len(next(iter(columns.values())))
    timestamps = pl.Series("time_s", [index / 50.0 for index in range(n_samples)])
    stream = Stream(
        taxonomy_type="extero.taxel_pressure",
        kind=Kind.SERIES,
        timestamps=timestamps,
        payload=FramePayload(frame=pl.DataFrame(columns)),
        source_path=_SOURCE_PATH,
        clock=Clock.CAPTURE,
        channels=[Channel(name=name) for name in columns],
    )
    return StreamContext(stream=stream, is_regular=True)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_missing_pct_counts_nan_alongside_null():
    """A NaN and a null both count as missing, not just the null."""

    ctx = _channel_ctx([1.0, float("nan"), None, 2.0, 3.0])

    result = missing_pct(ctx)

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.evidence["n_missing"] == 2
    assert result.value == pytest.approx(100 * 2 / 5)


def test_flatline_pct_reports_the_longest_run_not_the_total():
    """Two separate stuck runs: evidence names the longer one, not their sum."""

    values = [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0]
    ctx = _channel_ctx(values)

    result = flatline_pct(ctx)

    assert result.evidence["longest_run"] == 6
    assert result.value == pytest.approx(100 * 7 / 11)


def test_spike_pct_is_not_applicable_on_a_constant_channel():
    """A channel with no local spread has nothing to measure a spike against."""

    ctx = _channel_ctx([1.0] * 60)

    result = spike_pct(ctx)

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert result.value is None


def test_drift_r_squared_is_near_one_on_a_ramp_and_near_zero_on_a_sine():
    """A pure ramp fits a line almost perfectly; whole cycles of a sine don't."""

    ramp_ctx = _channel_ctx([float(index) for index in range(60)])
    sine_ctx = _channel_ctx([math.sin(2 * math.pi * index / 10) for index in range(60)])

    ramp_result = drift(ramp_ctx)
    sine_result = drift(sine_ctx)

    assert ramp_result.evidence["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert sine_result.evidence["r_squared"] == pytest.approx(0.0, abs=0.1)


def test_snr_db_is_not_applicable_on_a_noiseless_signal():
    """A pure ramp's centred rolling mean reproduces it exactly: no residual to
    call noise."""

    ctx = _channel_ctx([float(index) for index in range(60)])

    result = snr_db(ctx)

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_hysteresis_grows_with_skew_severity():
    """A harder unloading skew must read as more hysteresis,
    not a value capped by bin width."""

    clean = clean_taxels(samples=101, cells=4)
    mild = hysteresis(stream_context(skew_unloading(clean, factor=1.1)))
    severe = hysteresis(stream_context(skew_unloading(clean, factor=5.0)))

    assert mild.value is not None
    assert severe.value is not None
    assert severe.value > mild.value


def test_hysteresis_is_not_applicable_on_a_monotonic_rise():
    """A recording that only ever loads has no unloading branch to compare against."""

    rising = {
        "taxel_00": [float(index) for index in range(20)],
        "taxel_01": [float(index) * 2 for index in range(20)],
    }
    ctx = _taxel_stream_ctx(rising)

    result = hysteresis(ctx)

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_a_two_valued_channel_is_a_switch_so_flatline_spike_and_snr_stand_down():
    """A gripper open/close command or a done flag takes two values by design.

    Staying on one value for most of the episode is that switch working, not a
    stuck sensor, so flatline, spike and signal-to-noise don't apply to it.
    """

    gripper = [-1.0] * 30 + [1.0] * 30
    ctx = _channel_ctx(gripper)

    for check in (flatline_pct, spike_pct, snr_db):
        result = check(ctx)
        assert result.status == MetricStatus.NOT_APPLICABLE, check.__name__
        assert "two values" in result.evidence["reason"], check.__name__


def test_a_done_flag_with_a_null_is_still_a_switch():
    """Nulls don't count as a third value."""

    ctx = _channel_ctx([0.0] * 58 + [None, 1.0])

    assert flatline_pct(ctx).status == MetricStatus.NOT_APPLICABLE


def test_a_constant_channel_is_still_checked_for_flatline():
    """One value is not a switch: a channel that never moves may be a dead sensor."""

    result = flatline_pct(_channel_ctx([0.5] * 20))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == pytest.approx(100.0)


def test_a_switch_with_glitches_on_it_is_measured_again():
    """A third value (a glitch on the command) makes it a signal to check once more."""

    values = [-1.0] * 30 + [1.0] * 30
    values[10] = 2.7
    ctx = _channel_ctx(values)

    assert flatline_pct(ctx).status == MetricStatus.REPORT_ONLY
    assert spike_pct(ctx).status == MetricStatus.REPORT_ONLY


def _noisy_sine(n: int, rate_hz: float) -> list[float]:
    """A 0.5 Hz motion with a small, fixed-seed noise floor on top."""

    import random

    rng = random.Random(7)
    return [
        math.sin(2 * math.pi * 0.5 * i / rate_hz) + rng.gauss(0.0, 0.01)
        for i in range(n)
    ]


@pytest.mark.parametrize("rate_hz", [50.0, 100.0, 1000.0])
def test_snr_keeps_its_five_sample_window_at_50_hz_and_faster(rate_hz):
    """At 50 Hz and above the window is unchanged: 5 samples, 0.1 s or less."""

    result = snr_db(_channel_ctx(_noisy_sine(400, rate_hz), rate_hz=rate_hz))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.evidence["smoothing_window"] == 5
    assert result.evidence["smoothing_span_s"] <= 0.1 + 1e-9
    assert result.evidence["rate_hz"] == pytest.approx(rate_hz)


@pytest.mark.parametrize("rate_hz", [5.0, 10.0, 15.0, 30.0])
def test_snr_is_not_applicable_when_sampled_too_slowly_to_tell_noise_from_motion(
    rate_hz,
):
    """Below ~45 Hz five samples span more than 0.1 s and smooth away real motion.

    Found grading real 5-15 Hz LeRobot datasets: the ratio tracked the sampling
    rate rather than the robot, so it is not reported rather than misreported.
    """

    result = snr_db(_channel_ctx(_noisy_sine(200, rate_hz), rate_hz=rate_hz))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert (
        "too slowly to separate sensor noise from motion" in (result.evidence["reason"])
    )
