"""Controls for the 0.7.0 branch review findings F1–F7.

Each test reproduces the case the review reported against 32e36fb and asserts
the corrected behaviour: an incomplete inventory cannot read as 100%
readiness, an unexamined stream cannot pass, the metadata tier reads no
payloads, dataset compatibility follows the counts, configuration digests
follow the effective configuration, same-priority conflicts are refused, and
bundle policy paths resolve against the bundle.
"""

# Built-in
import shutil
from pathlib import Path
from unittest.mock import patch

# External
import h5py
import numpy as np
import polars as pl
import pytest
import yaml
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis.models.binding import (
    BindingOrigin,
    Bundle,
    FeatureAssertion,
    SamePriorityConflict,
    resolve_feature_types,
)
from kalanos.analysis.models.domain import FramePayload
from kalanos.analysis.models.eligibility import EligibilityStatus
from kalanos.analysis.models.errors import ConfigurationError, MappingOverrideError
from kalanos.analysis.models.provenance import Inventory
from kalanos.analysis.models.report import PayloadStatus, Report
from kalanos.analysis.reporting.assemble import grade_episode
from kalanos.api import grade
from kalanos.assets.bundle import load_bundle, resolve_run_configuration
from kalanos.assets.dictionary import load_dictionary
from kalanos.assets.policy import load_policy
from kalanos.assets.yaml_strict import DuplicateKeyError, safe_load_strict
from kalanos.cli import app


TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"


def _write_arm(path: Path, n_episodes: int, glitched: set[int]) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(str(path), "w") as store:
        for index in range(n_episodes):
            group = store.create_group(f"data/demo_{index}")
            t = np.arange(200) / 50.0
            signal = np.sin(t) + rng.normal(0, 0.01, 200)
            if index in glitched:
                signal[50:60] += 40.0
            group.create_dataset("joint_pos", data=signal)
            group.create_dataset("timestamp", data=t)


# ── F1: a declared episode that never loads is an inventory gap ──


def _truncated_tiny(tmp_path: Path) -> Path:
    """The tiny LeRobot fixture with one episode-index row removed, count kept."""

    copy = tmp_path / "lerobot_v3_tiny"
    shutil.copytree(TINY_V3, copy)
    index = copy / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    frame = pl.read_parquet(index)
    assert frame.height == 2
    frame.head(1).write_parquet(index)
    return copy


def test_f1_a_missing_index_row_is_an_unresolved_episode_not_a_smaller_denominator(
    tmp_path,
):
    report = grade(_truncated_tiny(tmp_path))
    inv = report.inventory
    assert inv is not None
    assert (inv.expected, inv.loaded, inv.unresolved, inv.complete) == (2, 1, 1, False)
    assert inv.notes and "declared 2" in inv.notes[0]
    counts = report.eligibility_counts
    assert counts is not None
    assert (counts.total, counts.pass_count, counts.unknown) == (2, 1, 1)
    assert counts.inventory_complete is False
    assert counts.confirmed_eligible_share is None
    assert report.readiness is not None and report.readiness.score is None
    assert any("incomplete" in r for r in report.readiness.reasons)
    assert report.score.train_ready is None
    assert report.run is not None and report.run.completion.value == "partial"


def test_f1_the_default_gate_refuses_an_incomplete_audit(tmp_path):
    result = CliRunner().invoke(app, ["grade", str(_truncated_tiny(tmp_path))])
    assert result.exit_code == 1


def test_f1_an_inventory_cannot_claim_a_gap_and_completeness():
    with pytest.raises(ValueError, match="cannot be complete"):
        Inventory(loaded=1, unresolved=1, complete=True)


# ── F2: a stream with no payload cannot pass on another stream's evidence ──


def _tiny_episode():
    from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter

    return next(iter(LeRobotV3Adapter().episodes(UPath(TINY_V3))))


def test_f2_a_required_stream_without_a_payload_makes_the_episode_unknown():
    episode = _tiny_episode()
    with_channels = [s for s in episode.streams if s.channels]
    assert len(with_channels) >= 2, "fixture needs two numeric streams"
    starved = with_channels[0].model_copy(update={"payload": None})
    episode = episode.model_copy(
        update={
            "streams": [
                starved if s is with_channels[0] else s for s in episode.streams
            ]
        }
    )
    graded, _ = grade_episode(
        episode, adapter="lerobot_v3", adapter_confidence=1.0, policy=load_policy(None)
    )
    statuses = {s.taxonomy_type: s.evaluation.payload for s in graded.streams}
    assert statuses[starved.taxonomy_type] == PayloadStatus.MISSING_INPUT
    assert statuses[with_channels[1].taxonomy_type] == PayloadStatus.COMPUTED

    from kalanos.analysis.models.binding import RequirementsSection
    from kalanos.analysis.scoring.eligibility import eligibility_of

    decision = eligibility_of(
        graded, [], requirements=RequirementsSection(), policy_id="p"
    )
    assert decision.status == EligibilityStatus.UNKNOWN
    assert any(r.id == f"payload:{starved.taxonomy_type}" for r in decision.reasons)


def test_f2_a_stream_without_channels_is_not_required():
    graded, _ = grade_episode(
        _tiny_episode(),
        adapter="lerobot_v3",
        adapter_confidence=1.0,
        policy=load_policy(None),
    )
    for stream in graded.streams:
        if not stream.channels:
            assert stream.evaluation.payload == PayloadStatus.NOT_REQUIRED
        else:
            assert stream.evaluation.payload == PayloadStatus.COMPUTED
            assert stream.evaluation.n_channels_graded == len(stream.channels)


# ── F3: the metadata tier is an execution boundary ──


def test_f3_metadata_tier_fetches_no_numeric_payload_and_leaves_required_checks_unknown(
    tmp_path,
):
    bundle = tmp_path / "meta.yaml"
    bundle.write_text(
        yaml.safe_dump({"schema_version": 1, "execution": {"tier": "metadata"}})
    )
    with patch.object(FramePayload, "fetch", autospec=True) as fetch:
        report = grade(TINY_V3, bundle=bundle)
    assert fetch.call_count == 0
    assert all(
        s.evaluation.payload == PayloadStatus.SKIPPED
        for e in report.episodes
        for s in e.streams
        if s.channels
    )
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.unknown == len(report.episodes)
    assert report.readiness is not None and report.readiness.score is None
    assert (
        CliRunner().invoke(app, ["grade", str(TINY_V3), "--tier", "metadata"]).exit_code
        == 1
    )


def test_f3_standard_tier_still_reads_payloads():
    with patch.object(
        FramePayload, "fetch", autospec=True, side_effect=FramePayload.fetch
    ) as fetch:
        report = grade(TINY_V3)
    assert fetch.call_count > 0
    assert (
        report.eligibility_counts is not None
        and report.eligibility_counts.pass_count == 2
    )


# ── F4: dataset train_ready follows the counts, gate or no gate ──


def test_f4_a_failed_input_beside_passing_episodes_leaves_dataset_train_ready_null(
    tmp_path,
):
    folder = tmp_path / "mixed"
    shutil.copytree(TINY_V3, folder / "lerobot_v3_tiny")
    (folder / "no_time.csv").write_text("a,b\n1,2\n3,4\n")
    report = grade(folder)
    c = report.eligibility_counts
    assert c is not None and c.pass_count == 2 and c.unknown >= 1
    assert report.score.train_ready is None


def test_f4_all_pass_without_a_letter_gate_is_train_ready(tmp_path):
    path = tmp_path / "clean.hdf5"
    _write_arm(path, 3, glitched=set())
    report = grade(path, policy=load_policy(Path("legacy_0_5")))
    assert report.gate is None
    assert report.score.train_ready is True


def test_f4_a_forged_dataset_train_ready_is_refused(tmp_path):
    path = tmp_path / "dirty.hdf5"
    _write_arm(path, 3, glitched={1})
    report = grade(path)
    assert report.score.train_ready is False
    payload = report.model_dump(mode="json")
    payload["score"]["train_ready"] = True
    with pytest.raises(ValueError, match="dataset train_ready"):
        Report.model_validate(payload)


# ── F5: digests identify the effective configuration ──


def _config(tmp_path: Path, **kw):
    return resolve_run_configuration(
        UPath(tmp_path),
        dictionary=load_dictionary(None),
        policy_digest="p",
        sidecar=False,
        **kw,
    )


def test_f5_an_argument_override_changes_the_effective_binding_digest(tmp_path):
    bundle = Bundle.model_validate(
        {
            "schema_version": 1,
            "binding": {
                "id": "b1",
                "features": {"observation.state": "proprio.joint_position"},
            },
        }
    )
    plain = _config(tmp_path, bundle=bundle)
    overridden = _config(
        tmp_path, bundle=bundle, mapping={"observation.state": "proprio.joint_velocity"}
    )
    assert plain.binding_id is not None and overridden.binding_id is not None
    assert plain.binding_id.digest != overridden.binding_id.digest
    # The declared bundle is the same file either way.
    assert plain.bundle_id is not None and overridden.bundle_id is not None
    assert plain.bundle_id.digest == overridden.bundle_id.digest
    # Equivalent resolution is stable.
    assert _config(tmp_path, bundle=bundle).binding_id.digest == plain.binding_id.digest


def test_f5_a_mapping_without_a_bundle_still_has_a_binding_identity(tmp_path):
    cfg = _config(tmp_path, mapping={"observation.state": "proprio.joint_position"})
    assert cfg.binding_id is not None and cfg.binding_id.id == "effective-mappings"
    assert _config(tmp_path).binding_id is None


def test_f5_the_dictionary_digest_covers_its_content_not_just_its_keys(tmp_path):
    dictionary = load_dictionary(None)
    key = next(iter(dictionary.entries))
    changed = dictionary.model_copy(
        update={
            "entries": {
                **dictionary.entries,
                key: dictionary.entries[key].model_copy(update={"unit": "furlong"}),
            }
        }
    )
    a = resolve_run_configuration(
        UPath(tmp_path), dictionary=dictionary, policy_digest="p", sidecar=False
    )
    b = resolve_run_configuration(
        UPath(tmp_path), dictionary=changed, policy_digest="p", sidecar=False
    )
    assert a.dictionary_id.digest != b.dictionary_id.digest


def test_f5_the_execution_identity_follows_the_tier(tmp_path):
    from kalanos.analysis.models.provenance import ExecutionTier

    std = _config(tmp_path)
    meta = _config(tmp_path, tier=ExecutionTier.METADATA)
    assert std.execution_id is not None and meta.execution_id is not None
    assert std.execution_id.digest != meta.execution_id.digest


# ── F6: same-priority conflicts are refused, wherever they hide ──


def test_f6_two_map_flags_for_one_feature_exit_two_before_grading():
    result = CliRunner().invoke(
        app,
        [
            "grade",
            str(TINY_V3),
            "--map",
            "observation.state=proprio.joint_position",
            "--map",
            "observation.state=proprio.joint_velocity",
        ],
    )
    assert result.exit_code == 2
    assert "conflicting argument assertions" in result.output + (result.stderr or "")


def test_f6_a_conflict_below_the_winner_is_still_refused():
    with pytest.raises(SamePriorityConflict, match="sidecar"):
        resolve_feature_types(
            [
                FeatureAssertion(
                    feature="f", taxonomy_type="a", origin=BindingOrigin.ARGUMENT
                ),
                FeatureAssertion(
                    feature="f", taxonomy_type="b", origin=BindingOrigin.SIDECAR
                ),
                FeatureAssertion(
                    feature="f", taxonomy_type="c", origin=BindingOrigin.SIDECAR
                ),
            ]
        )
    # Repeating the same value is not a conflict.
    resolve_feature_types(
        [
            FeatureAssertion(feature="f", taxonomy_type="a", origin=BindingOrigin.FILE),
            FeatureAssertion(feature="f", taxonomy_type="a", origin=BindingOrigin.FILE),
        ]
    )


def test_f6_duplicate_yaml_keys_are_refused_in_bundles_and_mapping_files(tmp_path):
    with pytest.raises(DuplicateKeyError, match="observation.state"):
        safe_load_strict("features:\n  observation.state: a\n  observation.state: b\n")
    bundle = tmp_path / "dup.yaml"
    bundle.write_text(
        "schema_version: 1\nbinding:\n  id: x\n  features:\n"
        "    observation.state: proprio.joint_position\n"
        "    observation.state: proprio.joint_velocity\n"
    )
    with pytest.raises(MappingOverrideError, match="duplicate key"):
        load_bundle(UPath(bundle))
    result = CliRunner().invoke(app, ["grade", str(TINY_V3), "--profile", str(bundle)])
    assert result.exit_code == 2


# ── F7: bundle policy paths resolve against the bundle; failures exit 2 ──


def test_f7_a_relative_policy_path_resolves_beside_the_bundle_from_any_cwd(
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
    assert report.run.policy.origin.endswith("review-policy.yaml")


def test_f7_missing_or_malformed_policy_is_a_configuration_error_exit_two(tmp_path):
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


def test_f7_an_in_memory_bundle_cannot_use_a_relative_policy_path():
    bundle = Bundle.model_validate({"schema_version": 1, "policy": {"path": "x.yaml"}})
    with pytest.raises(ConfigurationError, match="no file location"):
        grade(TINY_V3, bundle=bundle)
