"""Synthetic acceptance fixtures ONLY; these are not validation results or releases."""

from datetime import datetime, timezone

from kalanos.analysis.models.calibration import CalibrationManifest
from kalanos.analysis.models.provenance import content_digest
from kalanos.api import grade
from kalanos.assets.policy import load_default_policy


def synthetic_manifest(context, **changes):
    data = dict(
        id="test-only-" + content_digest(context)[:16],
        **context,
        status="accepted",
        accepted_by="synthetic unit-test fixture",
        accepted_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        validation_report="fixture://synthetic-not-real-validation",
        validation_report_digest="a" * 64,
        operating_scope="unit tests only",
        independent_episodes=True,
        real_fault_validation=True,
        combined_policy_validated=True,
        tuning_sessions=["fixture-tuning"],
        validation_sessions=["fixture-validation"],
        valid_episodes=600,
        false_blocks=0,
        fault_episodes=100,
        detected_faults=96,
    )
    data.update(changes)
    return CalibrationManifest(**data)


def grade_with_test_calibration(path, *, policy=None, **kwargs):
    """Exercise blocking/gate invariants with explicit matching test manifests."""
    policy = policy or load_default_policy()
    initial = grade(path, policy=policy, **kwargs)
    contexts = {
        content_digest(f.calibration["context"]): f.calibration["context"]
        for f in initial.findings
        if f.calibration
    }
    approved = policy.model_copy(
        update={
            "calibration_manifests": [synthetic_manifest(c) for c in contexts.values()]
        }
    )
    return grade(path, policy=approved, **kwargs)
