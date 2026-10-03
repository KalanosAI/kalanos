"""Verifies bundles: what a profile sets, how its policy resolves, and its limits."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from pathlib import Path

# External
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis.models.binding import Bundle
from kalanos.analysis.models.diagnostics import VisionSpec
from kalanos.analysis.models.eligibility import SufficiencyStatus
from kalanos.analysis.models.errors import ConfigurationError, MappingOverrideError
from kalanos.analysis.models.provenance import ExecutionTier, content_digest
from kalanos.api import grade
from kalanos.assets.bundle import prepare_configuration, resolve_vision
from kalanos.assets.policy import load_default_policy
from kalanos.benchmark import benchmark_dataset
from kalanos.cli import app
from kalanos.core.settings import Settings

# Local
from helpers import write_spiked_arm as _write_arm


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"
FIXTURE = Path(__file__).parent / "fixtures/lerobot_v3_tiny"


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_bundle_sets_scope_and_requirements_and_a_sidecar_cannot(tmp_path):
    """`--profile` decides the scope; a sidecar only asserts mappings."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 5, glitched=set())
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "requirements": {"id": "strict-v1", "min_pass_episodes": 10},
                "policy": {"id": "default-decisions-v1"},
                "execution": {"tier": "metadata"},
            }
        )
    )
    report = grade(path, bundle=bundle)
    assert report.scope is not None
    assert report.scope.requirements_id == "strict-v1"
    assert report.scope.tier.value == "metadata"
    assert report.sufficiency is not None
    assert report.sufficiency.status == SufficiencyStatus.INSUFFICIENT
    assert report.run is not None and report.run.requirements is not None
    assert report.run.requirements.id == "strict-v1"
    assert report.run.requirements.digest


# A bundle's policy path resolves beside the bundle, and an explicit policy wins


def test_a_relative_policy_path_resolves_beside_the_bundle_from_any_cwd(
    tmp_path, monkeypatch
):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "review-policy.yaml").write_text("extends: default\n")
    bundle = profile_dir / "bundle.yaml"
    bundle.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "policy": {"id": "review-v1", "path": "review-policy.yaml"},
            }
        )
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    report = grade(TINY_V3, bundle=bundle)
    assert report.scope is not None and report.scope.policy_id == "review-v1"
    assert report.run is not None and report.run.policy is not None
    assert report.run.policy.origin is not None
    assert report.run.policy.origin.endswith("review-policy.yaml")


def test_relative_bundle_policy_is_shared_by_grade_and_benchmark(tmp_path):
    policy = load_default_policy()
    (tmp_path / "policy.yaml").write_text(
        yaml.safe_dump(policy.model_dump(mode="json"))
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text("schema_version: 1\npolicy:\n  id: local\n  path: policy.yaml\n")
    report = grade(FIXTURE, bundle=profile)
    bench = benchmark_dataset(str(FIXTURE), bundle=profile, sample=0)
    assert report.run is not None
    assert bench.configuration["policy"] == report.run.policy


def test_missing_or_malformed_policy_is_a_configuration_error_exit_two(tmp_path):
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(
        yaml.safe_dump({"schema_version": 1, "policy": {"path": "nope.yaml"}})
    )
    with pytest.raises(ConfigurationError, match="could not be loaded"):
        grade(TINY_V3, bundle=bundle)
    assert (
        CliRunner()
        .invoke(app, ["grade", str(TINY_V3), "--profile", str(bundle)])
        .exit_code
        == 2
    )
    (tmp_path / "bad.yaml").write_text("schema_version: [\n")
    bundle.write_text(
        yaml.safe_dump({"schema_version": 1, "policy": {"path": "bad.yaml"}})
    )
    assert (
        CliRunner()
        .invoke(app, ["grade", str(TINY_V3), "--profile", str(bundle)])
        .exit_code
        == 2
    )


def test_an_in_memory_bundle_cannot_use_a_relative_policy_path():
    bundle = Bundle.model_validate({"schema_version": 1, "policy": {"path": "x.yaml"}})
    with pytest.raises(ConfigurationError, match="no file location"):
        grade(TINY_V3, bundle=bundle)


def test_explicit_policy_wins_consistently_over_bundle_policy(tmp_path):
    profile = Bundle.model_validate({"policy": {"path": "missing.yaml"}})
    explicit = load_default_policy()
    loaded, _, _ = prepare_configuration(
        UPath(tmp_path), policy=explicit, bundle=profile
    )
    assert loaded is explicit


# Vision settings, the execution identity and profile validation


def test_vision_settings_follow_cli_then_environment_then_bundle():
    """The command line beats KALANOS_*, which beats the bundle; the full tier scans."""

    bundle = VisionSpec(sample_frames=7, max_pixels=4096)
    unset = Settings(vision_samples=None, full_frame_scan=None)
    environment = Settings(vision_samples=5, full_frame_scan=True)

    def resolve(
        settings, samples=None, full_frame_scan=None, tier=ExecutionTier.STANDARD
    ):
        return resolve_vision(
            bundle,
            settings,
            samples=samples,
            full_frame_scan=full_frame_scan,
            tier=tier,
        )

    assert (resolve(unset).sample_frames, resolve(unset).full_frame_scan) == (7, False)
    assert resolve(unset).max_pixels == 4096
    assert (
        resolve(environment).sample_frames,
        resolve(environment).full_frame_scan,
    ) == (
        5,
        True,
    )
    cli = resolve(environment, samples=3, full_frame_scan=False)
    assert (cli.sample_frames, cli.full_frame_scan) == (3, False)
    assert resolve(unset, tier=ExecutionTier.FULL).full_frame_scan


def test_a_default_vision_section_keeps_the_execution_identity(tmp_path):
    """Only a changed vision section enters the execution digest."""

    def configuration(bundle):
        return prepare_configuration(UPath(tmp_path), bundle=bundle)[2]

    default = configuration(Bundle(vision=VisionSpec()))
    changed = configuration(Bundle(vision=VisionSpec(max_pixels=4096)))
    assert default.execution_id is not None and changed.execution_id is not None

    assert default.execution_id.digest == content_digest(
        {
            "tier": default.scope.tier.value,
            "limits": {
                "max_bytes": default.limits.max_bytes,
                "max_files": default.limits.max_files,
            },
        }
    )
    assert changed.execution_id.digest != default.execution_id.digest


@pytest.mark.parametrize("override", [{"full_frame_scan": True}, {"vision_samples": 3}])
def test_a_command_line_vision_override_changes_the_execution_identity(
    tmp_path, override
):
    """A full scan or another sample count is a different run to compare."""

    default = prepare_configuration(UPath(tmp_path), bundle=Bundle())[2]
    overridden = prepare_configuration(UPath(tmp_path), bundle=Bundle(), **override)[2]
    assert default.execution_id is not None and overridden.execution_id is not None

    assert overridden.execution_id.digest != default.execution_id.digest


def test_benchmark_cli_accepts_profile_and_rejects_conflicting_maps(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text("schema_version: 1\nexecution:\n  tier: metadata\n")
    output = tmp_path / "bench.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(FIXTURE),
            "--profile",
            str(profile),
            "--sample",
            "0",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["datasets"][0]["scope"]["tier"] == "metadata"
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(FIXTURE),
            "--map",
            "observation.state=proprio.joint_position",
            "--map",
            "observation.state=proprio.joint_torque",
        ],
    )
    assert result.exit_code == 2


def test_in_memory_bundle_cannot_bypass_schema_version_validation():
    with pytest.raises(MappingOverrideError, match="schema_version"):
        grade(FIXTURE, bundle=Bundle(schema_version=999))


# Resource limits


def test_bundle_resource_limits_are_enforced_before_adapter_reads(monkeypatch):
    import kalanos.api as api
    import kalanos.benchmark as benchmark
    from kalanos.analysis.models.discovery import SourceInfo
    from kalanos.analysis.models.errors import SourceTooLarge

    fake = SourceInfo(
        uri="hf://datasets/test/source@revision",
        protocol="hf",
        size_bytes=100,
        file_count=10,
    )
    monkeypatch.setattr(api, "resolve_source", lambda path: (UPath(FIXTURE), fake))
    monkeypatch.setattr(
        benchmark, "resolve_source", lambda path: (UPath(FIXTURE), fake)
    )
    bundle = Bundle.model_validate({"execution": {"limits": {"max_bytes": 99}}})
    with pytest.raises(SourceTooLarge):
        grade(FIXTURE, bundle=bundle)
    with pytest.raises(SourceTooLarge):
        benchmark_dataset(str(FIXTURE), bundle=bundle, sample=0)


def test_bundle_budget_can_tighten_but_cannot_lift_an_explicit_limit(tmp_path):
    from kalanos.analysis.models.discovery import SourceLimits

    bundle = Bundle.model_validate(
        {"execution": {"limits": {"max_bytes": 500, "max_files": 3}}}
    )
    _, _, config = prepare_configuration(
        UPath(tmp_path), bundle=bundle, limits=SourceLimits(max_bytes=100, max_files=10)
    )
    assert config.limits == SourceLimits(max_bytes=100, max_files=3)


@pytest.mark.parametrize(
    "limits", [{"unknown_budget": 1}, {"max_files": -1}, {"max_bytes": True}]
)
def test_unknown_or_invalid_resource_limits_are_rejected(limits):
    with pytest.raises(ValidationError):
        Bundle.model_validate({"execution": {"limits": limits}})
