"""Verifies only an accepted, matching calibration manifest promotes a finding."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from datetime import datetime, timedelta, timezone

# External
import pytest
from calibration_helpers import synthetic_manifest
from pydantic import ValidationError

# Internal
from kalanos.analysis.calibration import binomial_upper, evaluate
from kalanos.assets.policy import _parse_policy, load_default_policy


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _context() -> dict[str, str | None]:
    return dict(
        metric_id="integrity.spike_pct",
        taxonomy_type="proprio.joint_position",
        detector_digest="1" * 64,
        thresholds_digest="2" * 64,
        binding_digest="3" * 64,
        scope_digest="4" * 64,
    )


def _policy(manifest):
    return load_default_policy().model_copy(
        update={"calibration_manifests": [manifest]}
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


@pytest.mark.parametrize(
    "key",
    [
        "metric_id",
        "taxonomy_type",
        "detector_digest",
        "thresholds_digest",
        "binding_digest",
        "scope_digest",
    ],
)
def test_every_manifest_identity_must_match(key):
    context = _context()
    manifest = synthetic_manifest(context)
    changed = dict(context)
    changed[key] = (
        "motion.p99_torque"
        if key == "metric_id"
        else "proprio.joint_torque"
        if key == "taxonomy_type"
        else "b" * 64
    )
    assert not evaluate(_policy(manifest), changed)["accepted"]
    assert evaluate(_policy(manifest), context)["accepted"]


@pytest.mark.parametrize(
    "change",
    [
        {"status": "draft"},
        {"status": "revoked"},
        {"accepted_by": None},
        {"accepted_at": None},
        {"accepted_at": datetime.now(timezone.utc) + timedelta(days=1)},
        {"valid_until": datetime(2021, 1, 1, tzinfo=timezone.utc)},
        {"independent_episodes": False},
        {"real_fault_validation": False},
        {"combined_policy_validated": False},
        {"validation_sessions": []},
        {"valid_episodes": 599},
        {"false_blocks": 1},
        {"fault_episodes": 99},
        {"detected_faults": 95},
    ],
)
def test_insufficient_or_unaccepted_validation_never_promotes(change):
    context = _context()
    assert not evaluate(_policy(synthetic_manifest(context, **change)), context)[
        "accepted"
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"false_blocks": 601},
        {"detected_faults": 101},
        {"accepted_at": datetime(2020, 1, 1)},
        {"validation_sessions": ["same"], "tuning_sessions": ["same"]},
        {"validation_sessions": ["same", "same"]},
        {"detector_digest": "not-a-digest"},
        {"valid_until": datetime(2019, 1, 1, tzinfo=timezone.utc)},
    ],
)
def test_malformed_manifests_are_rejected(change):
    with pytest.raises(ValidationError):
        synthetic_manifest(_context(), **change)


def test_confidence_bounds_match_the_declared_acceptance_gate():
    assert binomial_upper(0, 600) == pytest.approx(1 - 0.05 ** (1 / 600))
    assert binomial_upper(0, 500) > 0.005
    assert 1 - binomial_upper(4, 100) > 0.90
    assert 1 - binomial_upper(5, 100) < 0.90


def test_legacy_name_list_is_not_calibration_and_false_is_rejected():
    policy = load_default_policy().model_copy(
        update={"calibrated_metrics": {"integrity.spike_pct": "approved"}}
    )
    assert not evaluate(policy, _context())["accepted"]
    with pytest.raises(ValidationError):
        type(policy).model_validate(
            {**policy.model_dump(), "enforce_calibration": False}
        )


def test_missing_runtime_identity_never_authorizes_blocking():
    context = _context()
    manifest = synthetic_manifest(context)
    context["binding_digest"] = None
    assert not evaluate(_policy(manifest), context)["accepted"]


def test_duplicate_manifest_ids_and_yaml_keys_are_configuration_errors():
    manifest = synthetic_manifest(_context())
    data = load_default_policy().model_dump()
    with pytest.raises(ValidationError, match="duplicate"):
        type(load_default_policy()).model_validate(
            {**data, "calibration_manifests": [manifest, manifest]}
        )
    with pytest.raises(ValueError, match="duplicate"):
        _parse_policy(
            "extends: default\nenforce_calibration: true\nenforce_calibration: true",
            source="test",
        )
