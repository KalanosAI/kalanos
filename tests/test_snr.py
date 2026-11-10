"""Verifies contextual SNR and the validation of its noise-floor reference."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import random

# External
import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from upath import UPath

# Internal
from kalanos.analysis.coverage import state_of
from kalanos.analysis.metrics.integrity import rotation_snr_db, snr_db
from kalanos.analysis.models.binding import (
    ChannelBinding,
    NoiseFloor,
    Quantity,
    Representation,
    Validation,
)
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.domain import (
    Channel,
    Clock,
    ClockInfo,
    FramePayload,
    Kind,
    SourceOrder,
    Stream,
    TimestampDtype,
)
from kalanos.analysis.models.eligibility import Consequence
from kalanos.analysis.models.metrics import Level, MetricStatus
from kalanos.analysis.models.scoring import FindingLocation
from kalanos.analysis.scoring.score import score_metrics
from kalanos.assets.policy import load_default_policy
from kalanos.testing import stream_context

# Local
from helpers import channel_ctx as _channel_ctx


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_QUATERNION = ("qx", "qy", "qz", "qw")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def context(scale=1.0, floor=None):
    """Use a seeded quiet hold, with independently specified reference metadata."""
    rng = random.Random(73)
    ctx = _channel_ctx([scale * rng.gauss(0, 0.001) for _ in range(400)])
    binding = ChannelBinding(
        feature="position",
        index=0,
        taxonomy_type="proprio.joint_position",
        quantity=Quantity.POSITION,
        representation=Representation.CONTINUOUS,
        unit="rad",
        source_identity="test-session",
        noise_floor=floor,
    )
    current = binding.model_dump(mode="json")
    current["identity"] = "position[0]"
    binding.validations = [
        Validation(
            property=k,
            value=current[k],
            scope="test-session",
            validator="test reference",
            evidence="fixture-only",
            capability="noise",
        )
        for k in ("identity", "quantity", "unit", "noise_floor")
        if current[k] is not None
    ]
    ctx.channel.binding = binding
    return ctx


def reference(sd=0.01, **changes):
    """A test-only residual standard-deviation reference in native radians."""
    return NoiseFloor(
        **dict(  # pyright: ignore[reportArgumentType]
            dict(
                standard_deviation=sd,
                unit="rad",
                source="reference_capture",
                reference="fixture-only",
                sensor_configuration="test-sensor",
                sample_rate_hz=100,
                bandwidth_hz=50,
                estimator="centered_mean_5_residual_std_v1",
            ),
            **changes,
        )
    )


def scored(ctx):
    """Run the real scorer, retaining findings and policy-independent evidence."""
    return score_metrics(
        {"snr_db": snr_db(ctx)},
        level=Level.CHANNEL,
        taxonomy_type=ctx.taxonomy_type,
        policy=load_default_policy(),
        location=FindingLocation(
            episode_id="e", stream=ctx.taxonomy_type, channel="position"
        ),
    )


def _quaternion_rows(
    angle_deg,
    *,
    n=200,
    rate_hz=50.0,
    noise_deg=0.09,
    seed=8,
    canonical=False,
):
    """Rows `(qx, qy, qz, qw)` of a rotation about one fixed axis, with small noise.

    The defaults are a 10° rms, 0.75 Hz swing near the identity rotation,
    with 0.09° rms of noise: what episode 8 of the synthetic action-spike set reads.

    Parameters
    ----------
    angle_deg : callable
        The rotation angle in degrees at time `t` in seconds.
    n : int
        How many rows.
    rate_hz : float
        The sample rate the angle is evaluated at.
    noise_deg : float
        The rms angle of the random rotation applied to each row.
    seed : int
        The noise generator's seed.
    canonical : bool
        Whether to flip every row to `qw >= 0`, as many sensors store a quaternion.

    Returns
    -------
    list[tuple[float, float, float, float]]
    """

    rng = random.Random(seed)
    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14.0)
    sigma = math.radians(noise_deg) / math.sqrt(3.0)
    rows = []
    for i in range(n):
        half = math.radians(angle_deg(i / rate_hz)) / 2
        motion = np.array([*(math.sin(half) * axis), math.cos(half)])
        rv = np.array([rng.gauss(0, sigma) for _ in range(3)])
        theta = float(np.linalg.norm(rv))
        noise = np.array([*(math.sin(theta / 2) * rv / theta), math.cos(theta / 2)])
        # Hamilton product motion * noise, both stored as (x, y, z, w).
        (x1, y1, z1, w1), (x2, y2, z2, w2) = motion, noise
        q = (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
        if canonical and q[3] < 0:
            q = tuple(-c for c in q)
        rows.append(q)
    return rows


def _swing(t):
    """A 0.75 Hz swing of 10° rms."""

    return 10 * math.sqrt(2) * math.sin(2 * math.pi * 0.75 * t)


def _quaternion_ctx(rows, *, rate_hz=50.0, names=_QUATERNION):
    """Wrap quaternion rows in a stream whose four channels are bound as quaternion.

    Parameters
    ----------
    rows : list of tuple
        Rows `(qx, qy, qz, qw)`.
    rate_hz : float
        The rate of the regular clock the rows sit on.
    names : tuple[str, ...]
        The channel order; each name picks its component from the row.

    Returns
    -------
    StreamContext
    """

    columns = {name: [row[_QUATERNION.index(name)] for row in rows] for name in names}
    stream = Stream(
        taxonomy_type="unmapped.observation.eef_quat",
        kind=Kind.SERIES,
        timestamps=pl.Series("time_s", [i / rate_hz for i in range(len(rows))]),
        payload=FramePayload(frame=pl.DataFrame(columns)),
        source_path=UPath("quaternion.parquet"),
        timestamp_dtype=TimestampDtype.FLOAT64,
        clock=Clock.CAPTURE,
        channels=[
            Channel(
                name=name,
                binding=ChannelBinding(
                    feature="observation.eef_quat",
                    index=_QUATERNION.index(name),
                    taxonomy_type="unmapped.observation.eef_quat",
                    representation=Representation.QUATERNION,
                ),
            )
            for name in names
        ],
    )
    return stream_context(stream)


def _probe_rotation_snr(rows):
    """The investigation probe's rotation SNR in numpy, as an independent reference."""

    q = np.array(rows, dtype=np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    k = np.ones(5) / 5
    sm = np.stack([np.convolve(q[:, j], k, mode="valid") for j in range(4)], 1)
    sm /= np.linalg.norm(sm, axis=1, keepdims=True)
    mean = sm.mean(0)
    mean /= np.linalg.norm(mean)

    def angle(a, b):
        return 2 * np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1))

    signal = (angle(sm, np.broadcast_to(mean, sm.shape)) ** 2).mean()
    noise = (angle(q[2:-2], sm) ** 2).mean()
    return 10 * math.log10(signal / noise)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


@pytest.mark.parametrize(
    "taxonomy", ["reward.reward", "annotation.label", "metadata.episode_index"]
)
def test_semantic_scalar_exclusions(taxonomy):
    ctx = context()
    assert ctx.channel.binding is not None
    ctx.channel.binding.taxonomy_type = taxonomy
    r = snr_db(ctx)
    assert r.status == MetricStatus.NOT_APPLICABLE
    assert r.availability is not None and r.availability.value == "not_applicable"


@pytest.mark.parametrize(
    "change", [{"representation": "discrete"}, {"quantity": "binary"}]
)
def test_declared_discrete_excluded_even_with_more_than_two_values(change):
    ctx = context()
    assert ctx.channel.binding is not None
    ctx.channel.binding = ctx.channel.binding.model_copy(update=change)
    result = snr_db(ctx)
    assert result.availability is not None
    assert result.availability.value == "not_applicable"


def test_quiet_hold_with_validated_floor_is_measured_without_noise_penalty():
    results, score, findings = scored(context(floor=reference()))
    assert results["snr_db"].value is not None and results["snr_db"].value < 15
    assert results["snr_db"].status == MetricStatus.REPORT_ONLY
    assert state_of(results["snr_db"]).value == "computed"
    assert score.score is None and not findings
    assert (
        results["snr_db"].evidence["noise_assessment"]["status"] == "within_reference"
    )


def test_quiet_hold_without_floor_stays_unassessed_and_review():
    results, _, findings = scored(context())
    assert results["snr_db"].evidence["noise_assessment"]["status"] == "unassessed"
    assert findings and findings[0].consequence.value == "review"
    assert findings[0].route is None


def test_corruption_above_reference_remains_visible():
    results, _, findings = scored(context(scale=1000, floor=reference()))
    assert results["snr_db"].evidence["noise_assessment"]["status"] == "above_reference"
    assert findings and findings[0].consequence.value == "review"


@pytest.mark.parametrize(
    "change",
    [
        {"unit": "degrees"},
        {"sample_rate_hz": 200},
        {"scale_transform": "normalized"},
    ],
)
def test_reference_mismatch_preserves_measurement_but_abstains(change):
    ctx = context(floor=reference(**change))
    result = snr_db(ctx)
    assert result.value is not None and math.isfinite(result.value)
    assert result.evidence["noise_assessment"]["status"] == "unassessed"


@pytest.mark.parametrize("property", ["unit", "quantity", "identity", "noise_floor"])
def test_each_reference_validation_required(property):
    ctx = context(floor=reference())
    assert ctx.channel.binding is not None
    ctx.channel.binding.validations = [
        v for v in ctx.channel.binding.validations if v.property != property
    ]
    assert snr_db(ctx).evidence["noise_assessment"]["status"] == "unassessed"


def test_changed_reference_or_source_invalidates_validation():
    ctx = context(floor=reference())
    assert ctx.channel.binding is not None
    ctx.channel.binding.source_identity = "another-session"
    assert snr_db(ctx).evidence["noise_assessment"]["status"] == "unassessed"
    ctx.channel.binding.source_identity = "test-session"
    ctx.channel.binding.noise_floor = reference(sd=1)
    assert snr_db(ctx).evidence["noise_assessment"]["status"] == "unassessed"


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_nonfinite_values_break_windows_without_contaminating_finite_results(invalid):
    ctx = context()
    ctx.values = ctx.values.scatter([200], [invalid])
    r = snr_db(ctx)
    assert r.value is not None and math.isfinite(r.value)
    assert r.evidence["n_invalid_samples"] == 1
    assert r.evidence["n_samples"] == 391


def test_unknown_native_clock_units_abstain():
    ctx = context()
    ctx.stream.stream.native_timestamps = pl.Series(range(400))
    ctx.stream.stream.clock_info = ClockInfo(native_unit="unknown")
    assert snr_db(ctx).value is None


def test_transformed_signal_restores_source_order():
    ctx = context()
    expected = snr_db(ctx)
    ctx.values = ctx.values.reverse()
    ctx.stream.stream.timestamps = ctx.stream.stream.timestamps.reverse()
    ctx.stream.stream.source_order = SourceOrder(
        preserved=False, original_index=list(reversed(range(400)))
    )
    assert snr_db(ctx).value == pytest.approx(expected.value)
    ctx.stream.stream.source_order = SourceOrder(preserved=False)
    assert snr_db(ctx).value is None


@pytest.mark.parametrize(
    "changes",
    [
        {"standard_deviation": 0},
        {"standard_deviation": float("nan")},
        {"bandwidth_hz": 51},
        {"estimator": "unspecified"},
        {"reference": " "},
    ],
)
def test_invalid_noise_reference_rejected(changes):
    with pytest.raises(ValidationError):
        reference(**changes)


def test_matching_manifest_cannot_promote_unassessed_snr(monkeypatch):
    from kalanos.analysis.calibration import apply_calibration

    _, _, findings = scored(context())
    monkeypatch.setattr("kalanos.analysis.calibration.context_for", lambda *a: {})
    monkeypatch.setattr(
        "kalanos.analysis.calibration.evaluate", lambda *a: {"accepted": True}
    )
    result = apply_calibration(findings, {"e": load_default_policy()}, None, None, None)
    assert result[0].consequence.value == "review"
    assert result[0].calibration["accepted"] is False
    _, _, above = scored(context(scale=1000, floor=reference()))
    promoted = apply_calibration(above, {"e": load_default_policy()}, None, None, None)
    assert promoted[0].consequence.value == "block"


@pytest.mark.parametrize(
    "values, expected", [([True, False, True], 0), ([True, None, False], 100 / 3)]
)
def test_boolean_missing_values_are_measured_without_snr(values, expected):
    from kalanos.analysis.metrics.integrity import missing_pct

    ctx = _channel_ctx(values)
    ctx.values = pl.Series(values, dtype=pl.Boolean)
    assert missing_pct(ctx).value == pytest.approx(expected)
    assert snr_db(ctx).value is None


def test_reference_without_source_scope_is_unassessed():
    ctx = context(floor=reference())
    assert ctx.channel.binding is not None
    ctx.channel.binding.source_identity = None
    for validation in ctx.channel.binding.validations:
        validation.scope = None
    assert snr_db(ctx).evidence["noise_assessment"]["status"] == "unassessed"


@pytest.mark.parametrize("floor", [None, reference()])
def test_snr_cannot_bypass_calibration_with_contract_label(floor):
    from kalanos.analysis.calibration import apply_calibration
    from kalanos.analysis.models.eligibility import BlockingRoute, Consequence

    policy = load_default_policy()
    policy.metrics["integrity.snr_db"].route = BlockingRoute.CONTRACT
    _, _, findings = score_metrics(
        {"snr_db": snr_db(context(scale=1000, floor=floor))},
        level=Level.CHANNEL,
        taxonomy_type="proprio.joint_position",
        policy=policy,
        location=FindingLocation(episode_id="e", channel="position"),
    )
    assert findings[0].consequence == Consequence.REVIEW
    result = apply_calibration(findings, {"e": policy}, None, None, None)
    assert result[0].consequence == Consequence.REVIEW
    assert not result[0].calibration["accepted"]


def test_explicit_report_only_noise_policy_stays_report_only():
    from kalanos.analysis.models.eligibility import Consequence

    policy = load_default_policy()
    policy.metrics["integrity.snr_db"].consequence = Consequence.REPORT_ONLY
    _, _, findings = score_metrics(
        {"snr_db": snr_db(context())},
        level=Level.CHANNEL,
        taxonomy_type="proprio.joint_position",
        policy=policy,
        location=FindingLocation(episode_id="e", channel="position"),
    )
    assert findings[0].consequence == Consequence.REPORT_ONLY


def test_qw_alone_reads_as_noise_near_the_identity_rotation():
    """qw = cos(θ/2) barely moves near the identity, so per-component SNR flags it."""

    qw = [row[3] for row in _quaternion_rows(_swing)]

    value = snr_db(_channel_ctx(qw, rate_hz=50.0)).value

    assert value is not None and value < 30.0


def test_rotation_snr_reads_the_whole_rotation_above_the_warning_band():
    result = rotation_snr_db(_quaternion_ctx(_quaternion_rows(_swing)))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value is not None and result.value >= 30.0
    assert result.evidence["noise_assessment"]["status"] == "unassessed"


def test_rotation_snr_ignores_quaternion_sign_and_component_order():
    rows = _quaternion_rows(_swing)
    flipped = [tuple(-c for c in row) if i % 2 else row for i, row in enumerate(rows)]

    original = rotation_snr_db(_quaternion_ctx(rows)).value
    reordered = rotation_snr_db(
        _quaternion_ctx(flipped, names=("qw", "qx", "qy", "qz"))
    ).value

    assert reordered == pytest.approx(original, abs=1e-9)


def test_rotation_snr_is_not_applicable_without_unit_length():
    rows = [tuple(2 * c for c in row) for row in _quaternion_rows(_swing)]
    result = rotation_snr_db(_quaternion_ctx(rows))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "unit length" in result.evidence["reason"]
    assert result.evidence["max_norm_error"] == pytest.approx(1.0)


def test_rotation_snr_is_not_applicable_below_45_hz():
    rows = _quaternion_rows(_swing, rate_hz=30.0)
    result = rotation_snr_db(_quaternion_ctx(rows, rate_hz=30.0))

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "too slowly" in result.evidence["reason"]


def test_snr_db_stands_down_on_a_quaternion_component():
    ctx = _channel_ctx([row[3] for row in _quaternion_rows(_swing)], rate_hz=50.0)
    ctx.channel.binding = ChannelBinding(
        feature="observation.eef_quat",
        index=3,
        taxonomy_type="unmapped.observation.eef_quat",
        representation=Representation.QUATERNION,
    )
    result = snr_db(ctx)

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert result.availability == Availability.NOT_APPLICABLE
    assert "rotation_snr_db" in result.evidence["reason"]


def test_rotation_snr_survives_a_half_turn():
    """A 200° sweep stored with qw >= 0 flips sign mid-episode; alignment undoes it."""

    rows = _quaternion_rows(lambda t: 50 * t, canonical=True)
    assert any(
        sum(x * y for x, y in zip(a, b, strict=True)) < 0
        for a, b in zip(rows, rows[1:], strict=False)
    )

    result = rotation_snr_db(_quaternion_ctx(rows))

    assert result.value == pytest.approx(_probe_rotation_snr(rows), abs=0.1)


def test_a_critical_rotation_snr_finding_is_reviewed_never_blocked():
    ctx = _quaternion_ctx(_quaternion_rows(_swing, noise_deg=10.0))
    _, _, findings = score_metrics(
        {"rotation_snr_db": rotation_snr_db(ctx)},
        level=Level.STREAM,
        taxonomy_type=ctx.taxonomy_type,
        policy=load_default_policy(),
        location=FindingLocation(episode_id="e", stream=ctx.taxonomy_type),
    )

    assert findings[0].severity.value == "critical"
    assert findings[0].consequence == Consequence.REVIEW


def test_rotation_snr_skips_invalid_rows_and_counts_them():
    rows = _quaternion_rows(_swing)
    rows[60] = (None, None, None, None)
    rows[120] = (rows[120][0], math.nan, rows[120][2], rows[120][3])
    result = rotation_snr_db(_quaternion_ctx(rows))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value is not None and result.value >= 30.0
    assert result.evidence["n_invalid_samples"] == 2


@pytest.mark.parametrize("is_regular", [True, False])
def test_rotation_snr_is_inapplicable_on_a_short_stream_without_a_quaternion(
    is_regular,
):
    """A stream with no quaternion is inapplicable whatever its length or sampling."""

    from kalanos.analysis.metrics.registry import run_stream_metrics

    ctx = _channel_ctx([0.0, 1.0, 0.5], rate_hz=50.0).stream
    ctx.is_regular = is_regular

    result = run_stream_metrics(ctx)["rotation_snr_db"]

    assert result.availability == Availability.NOT_APPLICABLE
