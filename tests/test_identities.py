"""Verifies that run configuration digests identify the effective configuration."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path
from typing import cast

# External
from upath import UPath

# Internal
from kalanos.analysis.models.binding import Bundle
from kalanos.api import grade
from kalanos.assets.bundle import resolve_run_configuration
from kalanos.assets.dictionary import load_dictionary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TINY_V3 = Path(__file__).parent / "fixtures" / "lerobot_v3_tiny"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _config(tmp_path: Path, **kw):
    return resolve_run_configuration(
        UPath(tmp_path),
        dictionary=load_dictionary(None),
        policy_digest="p",
        sidecar=False,
        **kw,
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_an_argument_override_changes_the_effective_binding_digest(tmp_path):
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
    resolved = _config(tmp_path, bundle=bundle)
    assert resolved.binding_id is not None
    assert resolved.binding_id.digest == plain.binding_id.digest


def test_a_mapping_without_a_bundle_still_has_a_binding_identity(tmp_path):
    cfg = _config(tmp_path, mapping={"observation.state": "proprio.joint_position"})
    assert cfg.binding_id is not None and cfg.binding_id.id == "effective-mappings"
    assert _config(tmp_path).binding_id is None


def test_the_dictionary_digest_covers_its_content_not_just_its_keys(tmp_path):
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


def test_the_execution_identity_follows_the_tier(tmp_path):
    from kalanos.analysis.models.provenance import ExecutionTier

    std = _config(tmp_path)
    meta = _config(tmp_path, tier=ExecutionTier.METADATA)
    assert std.execution_id is not None and meta.execution_id is not None
    assert std.execution_id.digest != meta.execution_id.digest


def test_producer_and_run_are_recorded_and_config_digests_are_stable(tmp_path):
    a = grade(TINY_V3)
    b = grade(TINY_V3)
    assert a.producer is not None and a.producer.version
    assert a.run is not None and b.run is not None
    assert a.run.id != b.run.id
    assert a.run.policy == b.run.policy
    assert a.run.requirements == b.run.requirements
    assert (
        a.run.source.complete is False and a.run.source.digest is None
    )  # not hashed yet: honest
    assert cast(str, a.run.tier.value) == "standard"
