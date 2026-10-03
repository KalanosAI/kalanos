"""Resolve channel semantics once, before either grading or benchmarking.

Source arrays and indices are never rewritten.
Whole-feature mappings remain compatibility defaults.
A specific channel interpretation survives a conflicting whole-feature assertion,
with both values recorded on the channel.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Iterable, Sequence
from typing import Any

# External
from pydantic import ValidationError

# Internal
from kalanos.analysis.inference.roles import roles
from kalanos.analysis.models.binding import (
    ActuatorKind,
    BindingOrigin,
    CapabilityCheck,
    ChannelBinding,
    CommandSemantics,
    PropertyConflict,
    Quantity,
    Representation,
    ValidationStatus,
)
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.domain import (
    Attribution,
    Channel,
    Episode,
    FramePayload,
    MappingSource,
    Stream,
)
from kalanos.analysis.models.errors import MappingOverrideError
from kalanos.analysis.models.mapping import MappingOverride
from kalanos.analysis.models.provenance import ConfigIdentity, content_digest


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_PROPERTIES = (
    "taxonomy_type",
    "actuator",
    "quantity",
    "representation",
    "unit",
    "command",
    "device",
    "frame",
    "sign",
    "alignment",
    "calibration_transform",
    "limits",
    "noise_reference",
    "noise_floor",
)
_TYPES = {
    "proprio.joint_position": (ActuatorKind.JOINT, Quantity.POSITION),
    "proprio.joint_velocity": (ActuatorKind.JOINT, Quantity.VELOCITY),
    "proprio.joint_acceleration": (ActuatorKind.JOINT, Quantity.ACCELERATION),
    "proprio.joint_torque": (ActuatorKind.JOINT, Quantity.EFFORT),
    "proprio.gripper_width": (ActuatorKind.GRIPPER, Quantity.POSITION),
    "action.joint_position_command": (ActuatorKind.JOINT, Quantity.POSITION),
    "action.joint_velocity_command": (ActuatorKind.JOINT, Quantity.VELOCITY),
    "action.joint_torque_command": (ActuatorKind.JOINT, Quantity.EFFORT),
    "action.gripper_command": (ActuatorKind.GRIPPER, Quantity.UNKNOWN),
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _known(value: object) -> bool:
    return value is not None and value != "unknown"


def _inferred(
    stream: Stream, channel: Channel, dictionary: Dictionary
) -> ChannelBinding:
    """Interpret names in one place and explicitly label the result inference."""

    taxonomy = stream.taxonomy_type
    [role] = roles([channel.declared_name or channel.name], dictionary)
    # A partly recognized declared vector must not type its unresolved members.
    if taxonomy.startswith("unmapped."):
        taxonomy = role.taxonomy_type or taxonomy
    actuator, quantity = _TYPES.get(taxonomy, (ActuatorKind.UNKNOWN, Quantity.UNKNOWN))
    if "gripper" in (channel.declared_name or channel.name).lower():
        actuator = ActuatorKind.GRIPPER
        if taxonomy in ("unmapped.observation.state", "unmapped.state"):
            taxonomy, quantity = "proprio.gripper_width", Quantity.POSITION
        # Device kind never turns effort/current into position or a command.
        if quantity == Quantity.POSITION and taxonomy.startswith("proprio."):
            taxonomy = "proprio.gripper_width"
        elif taxonomy == "action.action_vector":
            taxonomy = "action.gripper_command"
    command = (
        CommandSemantics.UNKNOWN
        if taxonomy.startswith("action.")
        else CommandSemantics.NONE
    )
    binding = ChannelBinding(
        feature=stream.source_field or channel.name,
        index=channel.source_index,
        name=channel.declared_name or channel.name,
        taxonomy_type=taxonomy,
        actuator=actuator,
        quantity=quantity,
        command=command,
        representation=(
            Representation.CONTINUOUS
            if quantity != Quantity.UNKNOWN
            else Representation.UNKNOWN
        ),
        origin=BindingOrigin.INFERRED,
        status=(
            ValidationStatus.UNRESOLVED
            if taxonomy.startswith("unmapped.")
            else ValidationStatus.INFERRED
        ),
    )
    binding.property_origins = {p: BindingOrigin.INFERRED for p in _PROPERTIES}
    binding.property_origins["name"] = (
        BindingOrigin.DECLARED
        if channel.declared_name is not None
        else BindingOrigin.INFERRED
    )
    binding.property_origins["identity"] = BindingOrigin.DECLARED
    return binding


def capability_checks(
    binding: ChannelBinding, *, numeric: bool
) -> dict[str, CapabilityCheck]:
    """Evaluate explicit prerequisites; readiness does not mean a metric ran.

    Generic inspection needs a numeric payload, not physical calibration.
    Physical claims require scoped validation of identity, quantity and units.
    Unit strings and metadata declarations alone do not validate dimensions.
    """

    validated = {
        p
        for p, status in binding.property_status.items()
        if status == ValidationStatus.VALIDATED
    }

    def validated_for(capability: str) -> set[str]:
        return validated | {
            v.property for v in binding.validations if v.capability == capability
        }

    def physical(capability: str) -> list[str]:
        missing = [
            f"{p} lacks scoped validation"
            for p in ("identity", "quantity", "unit")
            if p not in validated_for(capability)
        ]
        if not binding.unit:
            missing.append("unit is undeclared")
        if binding.quantity == Quantity.UNKNOWN:
            missing.append("quantity is unresolved")
        return missing

    derivatives = physical("derivatives")
    if binding.quantity not in (
        Quantity.POSITION,
        Quantity.VELOCITY,
        Quantity.ACCELERATION,
    ):
        derivatives.append("quantity is not kinematic")
    if binding.representation not in (Representation.CONTINUOUS, Representation.ANGLE):
        derivatives.append("representation is not continuous")
    if binding.taxonomy_type.startswith("action.") and binding.command not in (
        CommandSemantics.ABSOLUTE,
        CommandSemantics.RATE,
    ):
        derivatives.append("action is not an absolute/rate command")
    limits = physical("limits")
    if binding.limits is None or "limits" not in validated_for("limits"):
        limits.append("declared limits lack scoped validation")
    noise = physical("noise")
    if (
        binding.taxonomy_type.startswith("reward.")
        or binding.representation == Representation.DISCRETE
    ):
        noise.append("reward/discrete signals do not support this noise claim")
    if not (
        binding.noise_floor is not None
        and "noise_floor" in validated_for("noise")
        or binding.noise_reference
        and "noise_reference" in validated_for("noise")
    ):
        noise.append("matching noise reference lacks scoped validation")
    return {
        "numeric": CapabilityCheck(
            ready=numeric, reasons=[] if numeric else ["numeric payload unavailable"]
        ),
        "derivatives": CapabilityCheck(
            ready=numeric and not derivatives,
            reasons=derivatives + ([] if numeric else ["numeric payload unavailable"]),
        ),
        "limits": CapabilityCheck(
            ready=numeric and not limits,
            reasons=limits + ([] if numeric else ["numeric payload unavailable"]),
        ),
        "noise": CapabilityCheck(
            ready=numeric and not noise,
            reasons=noise + ([] if numeric else ["numeric payload unavailable"]),
        ),
    }


def _validate(binding: ChannelBinding, source: str) -> None:
    """Retain only evidence matching this property's value and source scope."""

    binding.property_status = {
        p: (
            ValidationStatus.UNRESOLVED
            if not _known(getattr(binding, p))
            else ValidationStatus.INFERRED
            if binding.property_origins.get(p) == BindingOrigin.INFERRED
            else ValidationStatus.ASSERTED
        )
        for p in _PROPERTIES
    }
    if binding.taxonomy_type.startswith("unmapped."):
        binding.property_status["taxonomy_type"] = ValidationStatus.UNRESOLVED
    binding.property_status["identity"] = ValidationStatus.DECLARED
    current = binding.model_dump(mode="json")
    current["identity"] = f"{binding.feature}[{binding.index}]"
    active = []
    for validation in binding.validations:
        matches = (
            validation.property in current
            and validation.value is not None
            and validation.value == current[validation.property]
            and validation.scope == source
        )
        if matches:
            active.append(validation)
            # Capability-specific evidence never validates unrelated uses.
            if validation.capability is None:
                binding.property_status[validation.property] = (
                    ValidationStatus.VALIDATED
                )
        else:
            binding.invalidated_validations.append(validation)
    binding.validations = active
    # A record validating units must not upgrade the entire channel to validated.
    binding.status = binding.property_status["taxonomy_type"]


def resolve_stream(
    stream: Stream,
    *,
    dictionary: Dictionary,
    channels: Sequence[ChannelBinding] = (),
    override: MappingOverride | None = None,
    source_identity: str | None = None,
    matched: set[tuple[str | None, str, int | None]] | None = None,
) -> Stream:
    """Return a stream with resolved bindings, preserving its payload and order."""

    source = source_identity or str(stream.source_path)
    resolved = []
    named_roles = roles(
        [c.declared_name or c.name for c in stream.channels], dictionary
    )
    partly_named = (
        stream.taxonomy_type.startswith("unmapped.")
        and any(r.taxonomy_type for r in named_roles)
        and any(r.taxonomy_type is None for r in named_roles)
    )
    for position, channel in enumerate(stream.channels):
        if channel.source_index is None and len(stream.channels) > 1:
            channel = channel.model_copy(update={"source_index": position})
        base = channel.binding or _inferred(stream, channel, dictionary)
        binding = base.model_copy(deep=True)
        binding.source_identity = source
        specific = (
            partly_named
            or channel.binding is not None
            or (
                channel.declared_name is not None
                and binding.taxonomy_type != stream.taxonomy_type
            )
        )
        assertions = [
            c
            for c in channels
            if c.feature == stream.source_field
            and c.index == channel.source_index
            and c.source_identity in (None, source)
        ]
        if len(assertions) > 1:
            raise MappingOverrideError(
                f"overlapping channel bindings for {source}: "
                f"{binding.feature}[{binding.index}]"
            )
        if override is not None:
            origin = BindingOrigin(override.origin.value)
            if specific and override.taxonomy_type != binding.taxonomy_type:
                binding.conflicts.append(
                    PropertyConflict(
                        property="taxonomy_type",
                        retained=binding.taxonomy_type,
                        displaced=override.taxonomy_type,
                        origin=origin,
                        reason="channel semantics retained over feature default",
                    )
                )
            elif not specific:
                if override.taxonomy_type != binding.taxonomy_type:
                    binding.invalidated_validations += binding.validations
                    binding.validations = []
                    binding.conflicts.append(
                        PropertyConflict(
                            property="taxonomy_type",
                            retained=override.taxonomy_type,
                            displaced=binding.taxonomy_type,
                            origin=binding.origin,
                            reason="feature mapping displaced inferred type",
                        )
                    )
                binding.taxonomy_type = override.taxonomy_type
                binding.origin = origin
                binding.property_origins["taxonomy_type"] = origin
                # A legacy taxonomy assertion does not assert units/physical facts.
                inferred_actuator, binding.quantity = _TYPES.get(
                    override.taxonomy_type, (ActuatorKind.UNKNOWN, Quantity.UNKNOWN)
                )
                if binding.actuator == ActuatorKind.UNKNOWN:
                    binding.actuator = inferred_actuator
                binding.property_origins["quantity"] = BindingOrigin.INFERRED
                binding.property_origins["actuator"] = BindingOrigin.INFERRED
        if assertions:
            assertion = assertions[0]
            if assertion.name is not None and assertion.name != (
                channel.declared_name or channel.name
            ):
                raise MappingOverrideError(
                    f"binding name mismatch at {binding.feature}[{binding.index}]: "
                    f"expected {assertion.name!r}, found {channel.name!r}"
                )
            if matched is not None:
                matched.add(
                    (assertion.source_identity, assertion.feature, assertion.index)
                )
            if any(
                p in assertion.model_fields_set
                and getattr(binding, p) != getattr(assertion, p)
                for p in _PROPERTIES
            ):
                binding.invalidated_validations += binding.validations
                binding.validations = []
            for prop in _PROPERTIES:
                if prop not in assertion.model_fields_set:
                    continue
                old, new = getattr(binding, prop), getattr(assertion, prop)
                if old != new and _known(old):
                    binding.conflicts.append(
                        PropertyConflict(
                            property=prop,
                            retained=new,
                            displaced=old,
                            origin=binding.property_origins.get(prop, binding.origin),
                            reason="channel assertion displaced prior interpretation",
                        )
                    )
                setattr(binding, prop, new)
                binding.property_origins[prop] = BindingOrigin.BUNDLE
            binding.validations += assertion.validations
            binding.origin = BindingOrigin.BUNDLE
            if override is not None and override.taxonomy_type != binding.taxonomy_type:
                binding.conflicts.append(
                    PropertyConflict(
                        property="taxonomy_type",
                        retained=binding.taxonomy_type,
                        displaced=override.taxonomy_type,
                        origin=BindingOrigin(override.origin.value),
                        reason="channel semantics retained over feature default",
                    )
                )
        try:
            binding = ChannelBinding.model_validate(binding.model_dump())
        except ValidationError as exc:
            raise MappingOverrideError(
                f"inconsistent binding at {binding.feature}[{binding.index}]: {exc}"
            ) from exc
        _validate(binding, source)
        numeric = (
            isinstance(stream.payload, FramePayload)
            and channel.name in stream.payload.frame.columns
            and stream.payload.frame[channel.name].dtype.is_numeric()
        )
        binding.capabilities = capability_checks(binding, numeric=numeric)
        resolved.append(channel.model_copy(update={"binding": binding}))
    updates = {"channels": resolved, "source_identity": source}
    if override is not None:
        updates.update(
            taxonomy_type=override.taxonomy_type, mapping_source=MappingSource.OVERRIDE
        )
    return stream.model_copy(update=updates)


def typed_views(stream: Stream) -> list[Stream]:
    """Expose homogeneous channel groups without reindexing source columns.

    Each view keeps the original field/path and each channel's source index.
    The source stream and its arrays remain untouched.
    Views are signals, not extra physical sensors;
    consumers must count devices from binding identity.
    """

    groups: dict[tuple, list[Channel]] = {}
    for channel in stream.channels:
        b = channel.binding
        if b is None:
            return [stream]
        key = (
            b.taxonomy_type,
            b.device,
            b.unit,
            b.quantity,
            b.command,
            b.representation,
            b.frame,
        )
        groups.setdefault(key, []).append(channel)
    if not groups:
        return [stream]
    views = []
    for key, channels in groups.items():
        payload = stream.payload
        if len(channels) != len(stream.channels) and isinstance(payload, FramePayload):
            payload = FramePayload(
                frame=payload.frame.select([c.name for c in channels])
            )
        taxonomy = key[0]
        mapping_source = stream.mapping_source
        if taxonomy.startswith("unmapped."):
            mapping_source = None
        elif taxonomy != stream.taxonomy_type:
            mapping_source = (
                MappingSource.OVERRIDE
                # every channel here passed the binding-is-None check above
                if any(
                    c.binding.origin == BindingOrigin.BUNDLE
                    for c in channels
                    if c.binding is not None
                )
                else MappingSource.DECLARED_NAMES
            )
        updates: dict[str, Any] = dict(
            taxonomy_type=taxonomy,
            channels=channels,
            payload=payload,
            mapping_source=mapping_source,
        )
        if key[1] is not None:
            updates.update(instance=key[1], attribution=Attribution.KEYED)
        views.append(stream.model_copy(update=updates))
    return views


def resolve_episodes(
    episodes: Sequence[Episode],
    *,
    dictionary: Dictionary,
    channels: Sequence[ChannelBinding] = (),
    overrides: Sequence[MappingOverride] = (),
    source_identity: str | None = None,
    matched: set[tuple[str | None, str, int | None]] | None = None,
) -> list[Episode]:
    """Shared interpretation for raw episodes in grade and benchmark."""

    by_feature: dict[str | None, MappingOverride] = {o.feature: o for o in overrides}
    return [
        episode.model_copy(
            update={
                "streams": [
                    view
                    for s in episode.streams
                    for view in typed_views(
                        resolve_stream(
                            s,
                            dictionary=dictionary,
                            channels=channels,
                            override=by_feature.get(s.source_field),
                            source_identity=source_identity,
                            matched=matched,
                        )
                    )
                ]
            }
        )
        for episode in episodes
    ]


def binding_records(episodes: Iterable[Episode]) -> dict[str, dict]:
    """Unique semantic records, with no payload arrays or display labels."""

    records = {}
    for episode in episodes:
        for stream in episode.streams:
            for channel in stream.channels:
                if channel.binding is not None:
                    data = channel.binding.model_dump(
                        mode="json", exclude={"name", "capabilities"}
                    )
                    records[content_digest(data)] = data
    return records


def binding_identity(
    episodes: Iterable[Episode], configured: ConfigIdentity | None
) -> ConfigIdentity:
    """Hash actual resolved semantics, independent of display names/episode counts."""

    return identity_from_records(binding_records(episodes), configured)


def identity_from_records(
    records: dict[str, dict], configured: ConfigIdentity | None
) -> ConfigIdentity:
    """The streaming benchmark and the pipeline use identical canonicalization."""

    return ConfigIdentity(
        id=configured.id if configured else "resolved-bindings",
        digest=content_digest(
            {
                "configured": configured.digest if configured else None,
                "channels": [records[k] for k in sorted(records)],
            }
        ),
        origin="resolved",
    )


def check_matched(
    channels: Sequence[ChannelBinding], matched: set[tuple[str | None, str, int | None]]
) -> None:
    """A typo/out-of-range selector is a configuration error, not silent success."""

    missing = [
        (c.source_identity, c.feature, c.index)
        for c in channels
        if (c.source_identity, c.feature, c.index) not in matched
    ]
    if missing:
        raise MappingOverrideError(f"channel bindings matched no input: {missing}")
