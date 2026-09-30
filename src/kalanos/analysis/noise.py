"""Applicability and reference evidence for smooth/residual diagnostics."""

import math

from kalanos.analysis.models.binding import Quantity, Representation


def snr_exclusion(ctx):
    """Return known semantic exclusions; unknown semantics remain diagnostic only."""
    binding = ctx.channel.binding
    if ctx.taxonomy_type.startswith(("reward.", "annotation.", "metadata.")):
        return (
            "reward, annotation and metadata signals do not support an SNR assessment"
        )
    if binding and (
        binding.quantity == Quantity.BINARY
        or binding.representation == Representation.DISCRETE
    ):
        return "declared binary/discrete signal does not support an SNR assessment"
    return None


def noise_assessment(ctx, rate_hz, residual_std, signal_std):
    """Compare with a scoped reference only after all required evidence matches.

    Parameters
    ----------
    ctx : ChannelContext
        Resolved channel and source identity.
    rate_hz, residual_std, signal_std : float
        Measured rate and component standard deviations in the channel scale.

    Returns
    -------
    dict
        An unassessed reason or reference-relative level; never a health verdict.
    """
    binding = ctx.channel.binding
    if binding is None or binding.noise_floor is None:
        return {"status": "unassessed", "reason": "no scoped native-scale noise floor"}
    floor = binding.noise_floor
    current = binding.model_dump(mode="json")
    current["identity"] = f"{binding.feature}[{binding.index}]"
    checked = {
        v.property
        for v in binding.validations
        if v.capability in (None, "noise")
        and bool(binding.source_identity)
        and v.scope == binding.source_identity
        and v.value is not None
        and v.value == current.get(v.property)
    }
    # Recheck records: a capability flag alone is not evidence.
    required = {"identity", "quantity", "unit", "noise_floor"}
    reasons = []
    if not required <= checked:
        reasons.append(
            "identity, quantity, unit and noise floor require "
            "matching scoped validation"
        )
    if binding.quantity == Quantity.UNKNOWN:
        reasons.append("quantity is unresolved")
    if binding.representation not in (Representation.CONTINUOUS, Representation.ANGLE):
        reasons.append("continuous representation is not declared")
    if floor.unit != binding.unit:
        reasons.append("reference unit differs from native channel unit")
    if floor.scale_transform != binding.calibration_transform:
        reasons.append("reference scale transform does not match")
    # Tolerance is numeric clock roundoff only, not a new supported rate regime.
    if not math.isclose(floor.sample_rate_hz, rate_hz, rel_tol=1e-6):
        reasons.append("reference sample rate does not match")
    if reasons:
        return {"status": "unassessed", "reason": "; ".join(reasons)}
    return {
        "status": "within_reference"
        if residual_std <= floor.standard_deviation
        else "above_reference",
        "reference": floor.model_dump(mode="json"),
        "residual_to_reference": residual_std / floor.standard_deviation,
        "signal_to_reference": signal_std / floor.standard_deviation,
        "near_reference_signal": signal_std <= floor.standard_deviation,
        "reason": (
            "comparison with a declared validated reference, not proof of sensor health"
        ),
    }
