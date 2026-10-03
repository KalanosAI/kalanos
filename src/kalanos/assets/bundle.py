"""Load a configuration bundle and resolve every mapping input through one path.

`resolve_run_configuration` is the single resolver: `grade` and `benchmark`
both call it, so a calibration result describes the same semantic
interpretation a user's grade run used. Legacy mapping inputs are normalised
into the binding model with their origin recorded; nothing on disk is
rewritten.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

# External
import yaml
from pydantic import ValidationError
from upath import UPath

# Internal
from kalanos.analysis.models.binding import (
    BUILT_IN_PROFILES,
    BUNDLE_SCHEMA_VERSION,
    BindingConflict,
    BindingOrigin,
    Bundle,
    EvaluationScope,
    FeatureAssertion,
    RequirementsSection,
    SamePriorityConflict,
    assertions_from_overrides,
    resolve_feature_types,
)
from kalanos.analysis.models.diagnostics import VisionSpec
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.discovery import SourceLimits
from kalanos.analysis.models.errors import ConfigurationError, MappingOverrideError
from kalanos.analysis.models.mapping import MappingOverride, OverrideOrigin
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.provenance import (
    ConfigIdentity,
    ExecutionTier,
    content_digest,
)
from kalanos.assets.dictionary import load_dictionary, use_dictionary
from kalanos.assets.mapping import (
    check_taxonomy_types,
    load_mapping_file,
    sidecar_path,
)
from kalanos.assets.policy import load_policy
from kalanos.assets.yaml_strict import safe_load_strict
from kalanos.core.settings import Settings, get_settings


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_ORIGIN_OF = {
    BindingOrigin.ARGUMENT: OverrideOrigin.ARGUMENT,
    BindingOrigin.FILE: OverrideOrigin.FILE,
    BindingOrigin.BUNDLE: OverrideOrigin.BUNDLE,
    BindingOrigin.SIDECAR: OverrideOrigin.SIDECAR,
}


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class RunConfiguration:
    """Everything a run resolved about its configuration, in one place.

    Attributes
    ----------
    overrides : list[MappingOverride]
        The winning feature assertions, in the shape the pipeline applies.
    conflicts : list[BindingConflict]
        Every feature where a lower-precedence source disagreed.
    scope : EvaluationScope
        What decisions are relative to.
    requirements : RequirementsSection
    requirements_id, policy_id, binding_id, dictionary_id : ConfigIdentity
        Identities with content digests, for the report's `run`.
    vision : VisionSpec
        How camera footage is read, after the command line and environment.
    """

    overrides: list[MappingOverride]
    conflicts: list[BindingConflict]
    scope: EvaluationScope
    requirements: RequirementsSection
    requirements_id: ConfigIdentity
    policy_id: ConfigIdentity
    binding_id: ConfigIdentity | None
    dictionary_id: ConfigIdentity
    execution_id: ConfigIdentity | None = None
    bundle_id: ConfigIdentity | None = None
    bundle: Bundle = field(default_factory=Bundle)
    limits: SourceLimits = field(default_factory=SourceLimits)
    vision: VisionSpec = field(default_factory=VisionSpec)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def load_bundle(path: UPath) -> Bundle:
    """Read one bundle YAML.

    Raises
    ------
    MappingOverrideError
        If the file is missing, is not YAML, does not match `Bundle`, or
        declares a `schema_version` other than `BUNDLE_SCHEMA_VERSION`.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MappingOverrideError(
            f"no bundle at {path}; built-in profiles: {', '.join(BUILT_IN_PROFILES)}"
        ) from exc
    try:
        payload = safe_load_strict(raw)
    except yaml.YAMLError as exc:
        raise MappingOverrideError(f"{path} is not valid YAML: {exc}") from exc
    try:
        bundle = Bundle.model_validate(payload)
    except ValidationError as exc:
        raise MappingOverrideError(
            f"{path} does not match the bundle schema: {exc}"
        ) from exc
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        raise MappingOverrideError(
            f"{path} declares schema_version {bundle.schema_version}; "
            f"this version of kalanos reads {BUNDLE_SCHEMA_VERSION}"
        )
    return bundle


def resolve_policy_path(bundle: Bundle, bundle_path: UPath | None) -> UPath | None:
    """Where the bundle's policy file is, relative to the bundle file itself.

    A relative `policy.path` is taken against the bundle's own directory, so a
    profile behaves the same from any working directory. An absolute path or a
    URL is kept as given. An in-memory bundle has no directory, so a relative
    path there is a configuration error.

    Raises
    ------
    ConfigurationError
    """

    if bundle.policy.path is None:
        return None
    declared = UPath(bundle.policy.path)
    if declared.is_absolute() or "://" in str(bundle.policy.path):
        return declared
    if bundle_path is None:
        raise ConfigurationError(
            f"policy.path {bundle.policy.path!r} is relative but the bundle has no "
            "file location; use an absolute path"
        )
    return UPath(bundle_path).parent / declared


def load_bundle_policy(bundle: Bundle, bundle_path: UPath | None) -> Policy | None:
    """The bundle's policy from where the bundle says, or `None` for the default.

    Every way this can fail — a missing file, unreadable YAML, a policy that
    does not validate — is a configuration error, not a crash.

    Raises
    ------
    ConfigurationError
    """

    path = resolve_policy_path(bundle, bundle_path)
    if path is None:
        return None
    try:
        return load_policy(path)  # type: ignore[arg-type]
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"bundle policy {path} could not be loaded: {exc}"
        ) from exc


def resolve_run_configuration(
    root: UPath,
    *,
    dictionary: Dictionary,
    policy_digest: str,
    bundle: Bundle | None = None,
    bundle_path: UPath | None = None,
    mapping: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    mapping_file: UPath | None = None,
    sidecar: bool = True,
    tier: ExecutionTier | None = None,
) -> RunConfiguration:
    """Resolve every configuration input by precedence, recording conflicts.

    Precedence for a feature's taxonomy type is
    `--map` > `--map-file` > bundle `binding.features` > discovered sidecar.
    A sidecar or mapping file asserts mappings only; requirements, policy and
    tier come from the bundle or the defaults. Two assertions at the same
    precedence that disagree are a configuration error.

    Parameters
    ----------
    root : UPath
        The path being graded, for sidecar discovery.
    dictionary : Dictionary
        The dictionary every asserted type must exist in.
    policy_digest : str
        The content digest of the loaded policy, for its identity.
    bundle : Bundle or None
        An explicit bundle, when the caller loaded one.
    bundle_path : UPath or None
        Where it came from, for provenance.
    mapping, mapping_file, sidecar
        The legacy inputs, exactly as `api.grade` accepts them.
    tier : ExecutionTier or None
        An explicit tier, overriding the bundle's. Never changes requirements.

    Returns
    -------
    RunConfiguration

    Raises
    ------
    MappingOverrideError
        On a malformed input, an unknown taxonomy type, or a same-priority conflict.
    """

    bundle = bundle or Bundle()
    if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
        raise MappingOverrideError(
            f"unsupported bundle schema_version {bundle.schema_version}"
        )
    assertions: list[FeatureAssertion] = []

    if sidecar:
        candidate = sidecar_path(root)
        if candidate.exists():
            assertions += assertions_from_overrides(
                load_mapping_file(candidate, origin=OverrideOrigin.SIDECAR)
            )
    assertions += bundle.feature_assertions(bundle_path)
    if mapping_file is not None:
        assertions += assertions_from_overrides(
            load_mapping_file(mapping_file, origin=OverrideOrigin.FILE)
        )
    # Arguments arrive as pairs, not a dict, so two `--map` flags for one
    # feature reach the resolver and can be refused as a conflict.
    pairs = mapping.items() if isinstance(mapping, Mapping) else (mapping or [])
    assertions += [
        FeatureAssertion(feature=k, taxonomy_type=v, origin=BindingOrigin.ARGUMENT)
        for k, v in pairs
    ]

    try:
        resolved = resolve_feature_types(assertions)
    except SamePriorityConflict as exc:
        raise MappingOverrideError(str(exc)) from exc

    overrides = [
        MappingOverride(
            feature=feature,
            taxonomy_type=winner.taxonomy_type,
            origin=_ORIGIN_OF[winner.origin],
            path=winner.path,
        )
        for feature, winner in resolved.types.items()
    ]
    check_taxonomy_types(overrides, dictionary)
    if bundle.binding is not None:
        check_taxonomy_types(
            [
                MappingOverride(
                    feature=c.feature,
                    taxonomy_type=c.taxonomy_type,
                    origin=OverrideOrigin.BUNDLE,
                )
                for c in bundle.binding.channels
            ],
            dictionary,
        )

    effective_tier = tier or bundle.execution.tier
    requirements = bundle.requirements
    # The binding identity is the *effective* mapping after precedence: the
    # winning type and origin per feature, plus any per-channel semantics.
    # Two runs that resolve the same interpretation share it; an argument
    # that overrides a bundle changes it. Paths are not part of it.
    effective = {
        feature: {"taxonomy_type": w.taxonomy_type, "origin": w.origin.value}
        for feature, w in sorted(resolved.types.items())
    }
    channels = (
        [c.model_dump(mode="json") for c in bundle.binding.channels]
        if bundle.binding is not None
        else []
    )
    binding_id = (
        ConfigIdentity(
            id=bundle.binding.id
            if bundle.binding is not None
            else "effective-mappings",
            digest=content_digest({"features": effective, "channels": channels}),
            origin="resolved",
        )
        if effective or channels
        else None
    )
    bundle_id = (
        ConfigIdentity(
            id=bundle.binding.id if bundle.binding is not None else "bundle",
            digest=content_digest(
                bundle.model_dump(mode="json", exclude={"policy": {"path"}})
            ),
            origin=str(bundle_path) if bundle_path else "in-memory",
        )
        if bundle_path is not None or bundle != Bundle()
        else None
    )
    execution_id = ConfigIdentity(
        id=f"tier-{effective_tier.value}",
        digest=content_digest(
            {
                "tier": effective_tier.value,
                "limits": bundle.execution.limits,
                **(
                    {"diagnostics": bundle.diagnostics.model_dump(mode="json")}
                    if bundle.diagnostics
                    else {}
                ),
                **_vision_identity(bundle.vision),
            }
        ),
        origin="resolved",
    )
    return RunConfiguration(
        overrides=overrides,
        conflicts=resolved.conflicts,
        scope=EvaluationScope(
            requirements_id=requirements.id,
            policy_id=bundle.policy.id,
            binding_id=binding_id.id if binding_id else None,
            tier=effective_tier,
        ),
        requirements=requirements,
        requirements_id=ConfigIdentity(
            id=requirements.id,
            digest=content_digest(requirements.model_dump(mode="json")),
            origin=str(bundle_path) if bundle_path else "builtin",
        ),
        policy_id=ConfigIdentity(
            id=bundle.policy.id,
            digest=policy_digest,
            origin=str(bundle.policy.path) if bundle.policy.path else "builtin",
        ),
        binding_id=binding_id,
        # The whole dictionary, not its key list: a changed unit or alias is
        # a changed interpretation.
        dictionary_id=ConfigIdentity(
            id=f"dictionary-v{dictionary.schema_version}",
            digest=content_digest(dictionary.model_dump(mode="json")),
            origin="loaded",
        ),
        execution_id=execution_id,
        bundle_id=bundle_id,
        bundle=bundle,
    )


def _vision_identity(vision: VisionSpec) -> dict[str, object]:
    """How camera footage is read, for the execution identity, empty when default.

    Left out at its default so a run without vision settings keeps its digest.
    """

    if vision == VisionSpec():
        return {}
    return {"vision": vision.model_dump(mode="json")}


def resolve_vision(
    bundle_vision: VisionSpec,
    settings: Settings,
    *,
    samples: int | None,
    full_frame_scan: bool | None,
    tier: ExecutionTier,
) -> VisionSpec:
    """Resolve how camera footage is read: CLI, then environment, then bundle.

    Parameters
    ----------
    bundle_vision : VisionSpec
        The bundle's `vision` section.
    settings : Settings
        The process settings, carrying any `KALANOS_*` override.
    samples, full_frame_scan : int, bool or None
        The command line's values, `None` where it gave none.
    tier : ExecutionTier
        The run's tier; the full tier always scans every frame.

    Returns
    -------
    VisionSpec
        The bundle's section with the sample count and the full scan resolved.
    """

    sample_frames = next(
        value
        for value in (samples, settings.vision_samples, bundle_vision.sample_frames)
        if value is not None
    )
    scan = next(
        value
        for value in (
            full_frame_scan,
            settings.full_frame_scan,
            bundle_vision.full_frame_scan,
        )
        if value is not None
    )
    return VisionSpec.model_validate(
        {
            **bundle_vision.model_dump(),
            "sample_frames": sample_frames,
            "full_frame_scan": scan or tier == ExecutionTier.FULL,
        }
    )


def overrides_for(config: RunConfiguration) -> Sequence[MappingOverride]:
    """The pipeline-shaped overrides a resolved configuration carries."""

    return config.overrides


def prepare_configuration(
    root: UPath,
    *,
    policy: Policy | None = None,
    dictionary: Dictionary | None = None,
    bundle: Bundle | UPath | str | os.PathLike[str] | None = None,
    mapping: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    mapping_file: UPath | None = None,
    sidecar: bool = True,
    tier: ExecutionTier | None = None,
    limits: SourceLimits | None = None,
    vision_samples: int | None = None,
    full_frame_scan: bool | None = None,
) -> tuple[Policy, Dictionary, RunConfiguration]:
    """Load grade/benchmark inputs identically; explicit policy wins over bundle.

    `vision_samples` and `full_frame_scan` are the command line's values,
    resolved with the environment and the bundle by `resolve_vision`.
    """

    settings = get_settings()
    explicit_policy = policy is not None
    dictionary = dictionary or load_dictionary(settings.dictionary_path)
    use_dictionary(dictionary)
    bundle_path = (
        None if bundle is None or isinstance(bundle, Bundle) else UPath(bundle)
    )
    # A built-in profile name stands in for a bundle unless such a file exists.
    if (
        bundle_path is not None
        and str(bundle) in BUILT_IN_PROFILES
        and not bundle_path.exists()
    ):
        bundle = Bundle(requirements=RequirementsSection(id=str(bundle)))
        bundle_path = None
    loaded = (
        bundle
        if isinstance(bundle, Bundle)
        else load_bundle(bundle_path)
        if bundle_path is not None
        else None
    )
    if policy is None and loaded is not None:
        policy = load_bundle_policy(loaded, bundle_path)
    policy = policy or load_policy(settings.policy_path)
    from kalanos.analysis.diagnostics.runner import validate_review_plan

    validate_review_plan(loaded.diagnostics if loaded else None, policy)
    config = resolve_run_configuration(
        root,
        dictionary=dictionary,
        policy_digest=content_digest(policy.model_dump(mode="json")),
        bundle=loaded,
        bundle_path=bundle_path,
        mapping=mapping,
        mapping_file=mapping_file,
        sidecar=sidecar,
        tier=tier,
    )
    if explicit_policy:
        config = replace(
            config,
            policy_id=config.policy_id.model_copy(
                update={"id": "explicit-policy", "origin": "argument"}
            ),
            scope=config.scope.model_copy(update={"policy_id": "explicit-policy"}),
        )
    base_limits = limits or SourceLimits(
        max_bytes=settings.remote_max_bytes, max_files=settings.remote_max_files
    )
    effective = {}
    for name in ("max_bytes", "max_files"):
        candidates = [
            getattr(base_limits, name),
            config.bundle.execution.limits.get(name),
        ]
        caps = [v for v in candidates if v is not None]
        effective[name] = min(caps) if caps else None
    vision = resolve_vision(
        config.bundle.vision,
        settings,
        samples=vision_samples,
        full_frame_scan=full_frame_scan,
        tier=config.scope.tier,
    )
    config = replace(
        config,
        limits=SourceLimits(**effective),
        vision=vision,
        execution_id=ConfigIdentity(
            id=f"tier-{config.scope.tier.value}",
            origin="resolved",
            digest=content_digest(
                {
                    "tier": config.scope.tier.value,
                    "limits": effective,
                    **(
                        {
                            "diagnostics": config.bundle.diagnostics.model_dump(
                                mode="json"
                            )
                        }
                        if config.bundle.diagnostics
                        else {}
                    ),
                    **_vision_identity(vision),
                }
            ),
        ),
    )
    return policy, dictionary, config


__all__ = [
    "RunConfiguration",
    "load_bundle",
    "load_bundle_policy",
    "overrides_for",
    "prepare_configuration",
    "resolve_policy_path",
    "resolve_run_configuration",
    "resolve_vision",
]
