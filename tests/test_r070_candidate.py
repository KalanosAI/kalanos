"""0.7.0 controls for contextual SNR and reference validation."""

import math
import random

import polars as pl
import pytest
from pydantic import ValidationError
from test_integrity import _channel_ctx

from kalanos.analysis.coverage import state_of
from kalanos.analysis.metrics.integrity import snr_db
from kalanos.analysis.models.binding import ChannelBinding, NoiseFloor, Validation
from kalanos.analysis.models.domain import ClockInfo, SourceOrder
from kalanos.analysis.models.metrics import Level, MetricStatus
from kalanos.analysis.models.scoring import FindingLocation
from kalanos.analysis.scoring.score import score_metrics
from kalanos.assets.policy import load_default_policy


def context(scale=1.0, floor=None):
    """Use a seeded quiet hold, with independently specified reference metadata."""
    rng = random.Random(73)
    ctx = _channel_ctx([scale * rng.gauss(0, 0.001) for _ in range(400)])
    binding = ChannelBinding(
        feature="position",
        index=0,
        taxonomy_type="proprio.joint_position",
        quantity="position",
        representation="continuous",
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
        **dict(
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


@pytest.mark.parametrize(
    "taxonomy", ["reward.reward", "annotation.label", "metadata.episode_index"]
)
def test_semantic_scalar_exclusions(taxonomy):
    ctx = context()
    ctx.channel.binding.taxonomy_type = taxonomy
    r = snr_db(ctx)
    assert r.status == MetricStatus.NOT_APPLICABLE
    assert r.availability.value == "not_applicable"


@pytest.mark.parametrize(
    "change", [{"representation": "discrete"}, {"quantity": "binary"}]
)
def test_declared_discrete_excluded_even_with_more_than_two_values(change):
    ctx = context()
    ctx.channel.binding = ctx.channel.binding.model_copy(update=change)
    assert snr_db(ctx).availability.value == "not_applicable"


def test_quiet_hold_with_validated_floor_is_measured_without_noise_penalty():
    results, score, findings = scored(context(floor=reference()))
    assert results["snr_db"].value < 15
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
    assert math.isfinite(result.value)
    assert result.evidence["noise_assessment"]["status"] == "unassessed"


@pytest.mark.parametrize("property", ["unit", "quantity", "identity", "noise_floor"])
def test_each_reference_validation_required(property):
    ctx = context(floor=reference())
    ctx.channel.binding.validations = [
        v for v in ctx.channel.binding.validations if v.property != property
    ]
    assert snr_db(ctx).evidence["noise_assessment"]["status"] == "unassessed"


def test_changed_reference_or_source_invalidates_validation():
    ctx = context(floor=reference())
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
    assert math.isfinite(r.value)
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
