"""Channel bindings and the configuration bundle.

A binding says what a channel *is*: which source field and index, what kind of
actuator, which physical quantity, in what unit, commanded how. Older mapping
inputs (`--map`, `--map-file`, `kalanos-map.yaml`) type a whole feature at
once; they remain supported and are normalised into this representation with
their origin recorded. Nothing here validates a physical claim: a declaration
is a declaration, an override is an assertion, and validation is a separate
record with its own evidence.

The bundle carries four logical sections with separate identities. Reducing
the execution tier never reduces the requirements; a sidecar can assert
mappings but never change requirements, policy or tier.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

# External
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal
from kalanos.analysis.models.mapping import MappingOverride, OverrideOrigin
from kalanos.analysis.models.paths import AnyPath
from kalanos.analysis.models.provenance import ExecutionTier


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

BUNDLE_SCHEMA_VERSION = 1

# The built-in evaluation scopes. `numeric-core` is the default: it asks for
# readable numeric input under an explicit missing-value contract and nothing
# more. It does not silently demand physical units, capture timing or video.
DEFAULT_REQUIREMENTS_ID = "numeric-core-v1"
DEFAULT_POLICY_ID = "default-decisions-v1"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class ActuatorKind(str, Enum):
    """What sort of thing a channel belongs to. Orthogonal to `Quantity`."""

    # fmt: off
    JOINT   = "joint"
    GRIPPER = "gripper"
    BASE    = "base"
    SENSOR  = "sensor"
    UNKNOWN = "unknown"
    # fmt: on


class Quantity(str, Enum):
    """The physical quantity a channel measures or commands."""

    # fmt: off
    POSITION     = "position"
    VELOCITY     = "velocity"
    ACCELERATION = "acceleration"
    EFFORT       = "effort"
    CURRENT      = "current"
    BINARY       = "binary"
    UNKNOWN      = "unknown"
    # fmt: on


class Representation(str, Enum):
    """How the quantity is encoded."""

    # fmt: off
    CONTINUOUS = "continuous"
    DISCRETE   = "discrete"
    ANGLE      = "angle"       # Continuous, and wraps
    UNKNOWN    = "unknown"
    # fmt: on


class CommandSemantics(str, Enum):
    """For an action channel, what the number means relative to the state."""

    # fmt: off
    ABSOLUTE = "absolute"
    DELTA    = "delta"
    RATE     = "rate"
    NONE     = "none"     # An observation, not a command
    UNKNOWN  = "unknown"
    # fmt: on


class BindingOrigin(str, Enum):
    """Where a binding assertion came from, in precedence order.

    `resolve_feature_types` applies exactly this order:
    argument > file > bundle > sidecar > declared > inferred.
    """

    # fmt: off
    ARGUMENT = "argument"  # `--map`
    FILE     = "file"      # `--map-file`
    BUNDLE   = "bundle"    # An explicit bundle's `binding` section
    SIDECAR  = "sidecar"   # An automatically discovered `kalanos-map.yaml`
    DECLARED = "declared"  # The source's own metadata
    INFERRED = "inferred"  # The dictionary or a heuristic
    # fmt: on


_PRECEDENCE = list(BindingOrigin)


def _rank(origin: BindingOrigin) -> int:
    return _PRECEDENCE.index(origin)


class ValidationStatus(str, Enum):
    """Whether a binding property has been checked against evidence.

    Precedence does not validate. An argument override outranks a sidecar and is
    still an assertion; only a validation record with evidence makes a property
    validated, and replacing a validated value invalidates it.
    """

    # fmt: off
    VALIDATED  = "validated"
    DECLARED   = "declared"
    ASSERTED   = "asserted"
    INFERRED   = "inferred"
    UNRESOLVED = "unresolved"
    # fmt: on


class Validation(BaseModel):
    """Evidence that one binding property is what it claims to be.

    Attributes
    ----------
    property : str
        The property validated: `quantity`, `unit`, `limits`, `identity`.
    validator : str
        Who or what validated it.
    evidence : str
        A reference to the evidence.
    scope : str or None
        The dataset/session/revision the validation applies to.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    property: str = Field(min_length=1)
    validator: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    scope: str | None = None
    value: Any = None
    capability: str | None = None


class PropertyConflict(BaseModel):
    """A displaced channel assertion; conflicts never disappear into a hash."""

    property: str
    retained: Any
    displaced: Any
    origin: BindingOrigin
    reason: str


class CapabilityCheck(BaseModel):
    """Prerequisite readiness, not a claim that a detector ran or passed."""

    ready: bool
    reasons: list[str] = Field(default_factory=list)


class ChannelBinding(BaseModel):
    """What one channel is, and how confidently.

    Attributes
    ----------
    feature : str
        The source field the channel lives in.
    index : int or None
        Its position within that field, or `None` for a scalar field.
    name : str or None
        The channel name, as declared or assigned.
    taxonomy_type : str
        The dictionary type it grades under.
    actuator : ActuatorKind
    quantity : Quantity
    representation : Representation
    unit : str or None
        The declared unit, or `None` when none was declared.
    command : CommandSemantics
    device : str or None
        The arm or device, e.g. `left`.
    origin : BindingOrigin
        Where the winning assertion came from.
    status : ValidationStatus
        Whether any of it has been validated.
    validations : list[Validation]
        The validation records, one per validated property.
    """

    model_config = ConfigDict(extra="forbid")

    feature: str
    index: int | None = Field(default=None, ge=0)
    name: str | None = None
    taxonomy_type: str
    actuator: ActuatorKind = ActuatorKind.UNKNOWN
    quantity: Quantity = Quantity.UNKNOWN
    representation: Representation = Representation.UNKNOWN
    unit: str | None = None
    command: CommandSemantics = CommandSemantics.UNKNOWN
    device: str | None = None
    origin: BindingOrigin = BindingOrigin.INFERRED
    status: ValidationStatus = ValidationStatus.UNRESOLVED
    validations: list[Validation] = Field(default_factory=list)
    source_identity: str | None = None
    frame: str | None = None
    sign: int | None = None
    alignment: str | None = None
    calibration_transform: str | None = None
    limits: tuple[float, float] | None = None
    noise_reference: str | None = None
    property_origins: dict[str, BindingOrigin] = Field(default_factory=dict)
    property_status: dict[str, ValidationStatus] = Field(default_factory=dict)
    conflicts: list[PropertyConflict] = Field(default_factory=list)
    invalidated_validations: list[Validation] = Field(default_factory=list)
    capabilities: dict[str, CapabilityCheck] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent_declarations(self) -> "ChannelBinding":
        """Reject malformed limits/signs, without treating assertions as evidence."""

        expected = {
            "proprio.joint_position": {Quantity.POSITION},
            "proprio.gripper_width": {Quantity.POSITION},
            "action.joint_position_command": {Quantity.POSITION},
            "proprio.joint_velocity": {Quantity.VELOCITY},
            "action.joint_velocity_command": {Quantity.VELOCITY},
            "proprio.joint_acceleration": {Quantity.ACCELERATION},
            "proprio.joint_torque": {Quantity.EFFORT, Quantity.CURRENT},
            "action.joint_torque_command": {Quantity.EFFORT, Quantity.CURRENT},
        }.get(self.taxonomy_type)
        if (
            expected
            and self.quantity != Quantity.UNKNOWN
            and self.quantity not in expected
        ):
            raise ValueError("quantity contradicts the selected taxonomy type")
        if self.sign not in (None, -1, 1):
            raise ValueError("sign must be -1 or 1")
        if self.limits is not None and (
            not all(math.isfinite(x) for x in self.limits)
            or not self.limits[0] < self.limits[1]
        ):
            raise ValueError("limits must be an increasing lower/upper pair")
        return self


class FeatureAssertion(BaseModel):
    """One assertion that a whole feature has a taxonomy type.

    This is the shape every legacy mapping input reduces to.

    Attributes
    ----------
    feature : str
    taxonomy_type : str
    origin : BindingOrigin
    path : UPath or None
        The file the assertion came from, when it came from one.
    """

    model_config = ConfigDict(frozen=True)

    feature: str
    taxonomy_type: str
    origin: BindingOrigin
    path: AnyPath | None = None


class BindingConflict(BaseModel):
    """Two assertions about one feature that disagreed, and how it was settled.

    Attributes
    ----------
    feature : str
    winner : FeatureAssertion
    displaced : list[FeatureAssertion]
        Every lower-precedence assertion that said something else.
    """

    feature: str
    winner: FeatureAssertion
    displaced: list[FeatureAssertion] = Field(default_factory=list)


class ResolvedFeatureTypes(BaseModel):
    """The outcome of merging every feature-level assertion by precedence.

    Attributes
    ----------
    types : dict[str, FeatureAssertion]
        Each feature's winning assertion.
    conflicts : list[BindingConflict]
        Every feature where a lower-precedence source disagreed.
    """

    types: dict[str, FeatureAssertion] = Field(default_factory=dict)
    conflicts: list[BindingConflict] = Field(default_factory=list)


class SamePriorityConflict(ValueError):
    """Two assertions at the same precedence disagreed: a configuration error."""


def resolve_feature_types(
    assertions: Iterable[FeatureAssertion],
) -> ResolvedFeatureTypes:
    """Merge feature-level assertions by origin precedence, recording every conflict.

    Parameters
    ----------
    assertions : Iterable[FeatureAssertion]
        From any source, in any order; the origin decides precedence.

    Returns
    -------
    ResolvedFeatureTypes
        The winner per feature and the displaced disagreements.

    Raises
    ------
    SamePriorityConflict
        If two assertions at the same origin give one feature different types.
    """

    by_feature: dict[str, list[FeatureAssertion]] = {}
    for assertion in assertions:
        by_feature.setdefault(assertion.feature, []).append(assertion)

    types: dict[str, FeatureAssertion] = {}
    conflicts: list[BindingConflict] = []
    for feature, group in by_feature.items():
        ordered = sorted(group, key=lambda a: _rank(a.origin))
        winner = ordered[0]
        # Disagreement within *any* origin group is a configuration error,
        # even one a higher-priority assertion would have outranked: a
        # contradiction is not resolved by being hidden.
        by_origin: dict[BindingOrigin, set[str]] = {}
        for a in ordered:
            by_origin.setdefault(a.origin, set()).add(a.taxonomy_type)
        for origin, asserted in by_origin.items():
            if len(asserted) > 1:
                raise SamePriorityConflict(
                    f"{feature}: conflicting {origin.value} assertions "
                    f"({', '.join(sorted(asserted))})"
                )
        displaced = [
            a
            for a in ordered
            if a.origin != winner.origin and a.taxonomy_type != winner.taxonomy_type
        ]
        types[feature] = winner
        if displaced:
            conflicts.append(
                BindingConflict(feature=feature, winner=winner, displaced=displaced)
            )
    return ResolvedFeatureTypes(types=types, conflicts=conflicts)


def assertions_from_overrides(
    overrides: Sequence[MappingOverride],
) -> list[FeatureAssertion]:
    """Normalise legacy mapping overrides into feature assertions, origin preserved."""

    origin_of = {
        OverrideOrigin.ARGUMENT: BindingOrigin.ARGUMENT,
        OverrideOrigin.FILE: BindingOrigin.FILE,
        OverrideOrigin.BUNDLE: BindingOrigin.BUNDLE,
        OverrideOrigin.SIDECAR: BindingOrigin.SIDECAR,
    }
    return [
        FeatureAssertion(
            feature=o.feature,
            taxonomy_type=o.taxonomy_type,
            origin=origin_of[o.origin],
            path=o.path,
        )
        for o in overrides
    ]


class BindingSection(BaseModel):
    """The bundle's `binding` section: what the channels are.

    Attributes
    ----------
    id : str
    features : dict[str, str]
        Whole-feature taxonomy assertions, the legacy shape.
    channels : list[ChannelBinding]
        Per-channel semantics. Explicit per-channel semantics take precedence
        within a resolved binding over a whole-feature assertion.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    features: dict[str, str] = Field(default_factory=dict)
    channels: list[ChannelBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_channels(self) -> "BindingSection":
        """Never allow duplicate selectors to silently replace one another."""

        keys = [(c.source_identity, c.feature, c.index) for c in self.channels]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "duplicate binding channel selector (source, feature, index)"
            )
        return self


class RequirementsSection(BaseModel):
    """The bundle's `requirements` section: what a pass needs.

    Attributes
    ----------
    id : str
    required_families : list[str]
        Metric families that must have graded at least one result in an
        episode for it to be decidable; otherwise the episode is `unknown`.
    require_resolved_bindings : bool
        Whether an episode with any unmapped stream is `unknown`.
    require_numeric_payloads : bool
        Whether every stream that declares channels must have had its
        payload read and graded. `True` under numeric-core: a stream with a
        missing, skipped or errored payload makes the episode `unknown`,
        however well another stream scored. Streams without channels
        (video, text) are not required by this.
    min_pass_episodes : int or None
        A dataset-level sufficiency constraint, evaluated in 0.7.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = DEFAULT_REQUIREMENTS_ID
    required_families: list[str] = Field(default_factory=lambda: ["integrity"])
    require_resolved_bindings: bool = False
    require_numeric_payloads: bool = True
    min_pass_episodes: int | None = Field(default=None, ge=0)
    required_capabilities: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def preset(cls, data: Any) -> Any:
        """A named built-in preset has actual requirements, even with only an id."""

        if isinstance(data, dict) and data.get("id") == "vision-imitation-v1":
            data = dict(data)
            data["required_capabilities"] = sorted(
                set(data.get("required_capabilities", [])) | {"video_quality"}
            )
        return data


class PolicySection(BaseModel):
    """The bundle's `policy` section: which decision policy applies.

    Attributes
    ----------
    id : str
    path : UPath or None
        A policy file to load instead of the built-in one.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = DEFAULT_POLICY_ID
    path: AnyPath | None = None


class ExecutionSection(BaseModel):
    """The bundle's `execution` section: what this run attempts."""

    model_config = ConfigDict(extra="forbid")

    tier: ExecutionTier = ExecutionTier.STANDARD
    limits: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def supported_limits(self) -> "ExecutionSection":
        """Reject budgets the runtime cannot enforce."""

        if self.limits.keys() - {"max_bytes", "max_files"}:
            raise ValueError("execution.limits supports only max_bytes and max_files")
        if any(
            v is not None and (type(v) is not int or v < 0)
            for v in self.limits.values()
        ):
            raise ValueError("execution limits must be nonnegative integers or null")
        return self


class Bundle(BaseModel):
    """One configuration file, four identities.

    Attributes
    ----------
    schema_version : int
    binding : BindingSection or None
    requirements : RequirementsSection
    policy : PolicySection
    execution : ExecutionSection
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = BUNDLE_SCHEMA_VERSION
    binding: BindingSection | None = None
    requirements: RequirementsSection = Field(default_factory=RequirementsSection)
    policy: PolicySection = Field(default_factory=PolicySection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)

    def feature_assertions(self, path: AnyPath | None = None) -> list[FeatureAssertion]:
        """The binding section's whole-feature assertions, at bundle precedence."""

        if self.binding is None:
            return []
        return [
            FeatureAssertion(
                feature=feature,
                taxonomy_type=taxonomy_type,
                origin=BindingOrigin.BUNDLE,
                path=path,
            )
            for feature, taxonomy_type in self.binding.features.items()
        ]


class EvaluationScope(BaseModel):
    """What a report's decisions are relative to, named on every surface.

    Attributes
    ----------
    requirements_id, policy_id : str
    binding_id : str or None
    tier : ExecutionTier
    """

    requirements_id: str
    policy_id: str
    binding_id: str | None = None
    tier: ExecutionTier = ExecutionTier.STANDARD


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_POLICY_ID",
    "DEFAULT_REQUIREMENTS_ID",
    "ActuatorKind",
    "BindingConflict",
    "BindingOrigin",
    "BindingSection",
    "Bundle",
    "ChannelBinding",
    "CommandSemantics",
    "EvaluationScope",
    "ExecutionSection",
    "FeatureAssertion",
    "PolicySection",
    "Quantity",
    "Representation",
    "RequirementsSection",
    "ResolvedFeatureTypes",
    "SamePriorityConflict",
    "Validation",
    "ValidationStatus",
    "assertions_from_overrides",
    "resolve_feature_types",
]
