"""Shared prerequisite checks and native source-coordinate access."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import bisect
import math
from collections.abc import Sequence
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Literal, cast, overload

# Internal
from kalanos.analysis.models.binding import ChannelBinding
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.diagnostics import (
    ClockRelation,
    CohortSpec,
    DiagnosticResult,
    MotionSpec,
    Selector,
    TimingSpec,
    TrackingSpec,
    WindowSpec,
)
from kalanos.analysis.models.domain import Channel, Episode, FramePayload, Stream


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# Source rows, payload positions and relative seconds, float-valued or exact.
_Axis = tuple[list[int], list[int], list[float]]
_ExactAxis = tuple[list[int], list[int], list[Fraction]]

# One diagnostic's own spec; vision carries only the identifier result() uses.
DiagnosticSpec = (
    TimingSpec | TrackingSpec | MotionSpec | WindowSpec | CohortSpec | SimpleNamespace
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Unavailable(ValueError):
    """Required evidence is absent; this is not a successful measurement."""


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def select(episode: Episode, selector: Selector) -> tuple[Stream, Channel | None]:
    """Resolve exactly one stream/channel; never choose the first ambiguous match."""
    streams = [
        s
        for s in episode.streams
        if s.source_field == selector.feature
        and s.instance == selector.instance
        and (selector.source_path is None or str(s.source_path) == selector.source_path)
        and (
            selector.index is None
            or any(c.source_index == selector.index for c in s.channels)
        )
    ]
    if len(streams) != 1:
        raise Unavailable(
            f"selector matched {len(streams)} streams: {selector.model_dump()}"
        )
    stream = streams[0]
    channels = [
        c
        for c in stream.channels
        if selector.index is None or c.source_index == selector.index
    ]
    return stream, channels[0] if len(channels) == 1 else None


def ordered(stream: Stream) -> tuple[list[int], list[int], list[float], float]:
    """Return source rows, payload positions and relative native seconds.

    Integer ticks are subtracted before conversion to retain epoch precision.
    The separate origin allows two streams to share one subtraction origin.
    """
    order = stream.source_order
    if not order.preserved and order.original_index is None:
        raise Unavailable("source row order is unavailable")
    rows = (
        order.original_index
        if order.original_index is not None
        else list(range(len(stream.timestamps)))
    )
    positions = sorted(range(len(rows)), key=lambda i: rows[i])
    rows = [rows[i] for i in positions]
    info = stream.clock_info
    values = stream.native_timestamps
    factor = 1.0
    if values is not None:
        factor = (
            info.tick_period_s
            # A None or unknown unit matches no key and leaves the factor None.
            or {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}.get(
                cast(str, info.native_unit)
            )
            if info
            else None
        )
        if factor is None:
            raise Unavailable("native clock units do not establish seconds")
    else:
        values = stream.timestamps
    ticks = [values[i] for i in positions]
    if len(ticks) < 2 or any(v is None or not math.isfinite(v) for v in ticks):
        raise Unavailable("at least two finite timestamps are required")
    if any(b <= a for a, b in zip(ticks, ticks[1:], strict=False)):
        raise Unavailable("source time axis is not strictly increasing")
    return rows, positions, ticks, factor


def _rational(value: float) -> Fraction:
    """Preserve integer ticks and the recorded decimal value of float inputs.

    No epsilon or rounding is applied:
    a distinct recorded future timestamp remains distinct.
    Decimal scale/rate declarations do not introduce another binary float multiplication
    or origin-subtraction error.
    """
    return Fraction(str(value))


@overload
def axis(stream: Stream, *, exact: Literal[False] = False) -> _Axis: ...
@overload
def axis(stream: Stream, *, exact: Literal[True]) -> _ExactAxis: ...
def axis(stream: Stream, *, exact: bool = False) -> _Axis | _ExactAxis:
    """Return relative seconds, optionally rational for causal grid comparisons."""
    rows, positions, ticks, factor = ordered(stream)
    if exact:
        origin, scale = _rational(ticks[0]), _rational(factor)
        return rows, positions, [(_rational(t) - origin) * scale for t in ticks]
    return rows, positions, [(t - ticks[0]) * factor for t in ticks]


def scope_of(stream: Stream) -> str:
    """Use resolved source identity, falling back to the adapter's source path."""
    if stream.source_identity:
        return stream.source_identity
    scopes = {
        c.binding.source_identity
        for c in stream.channels
        if c.binding and c.binding.source_identity
    }
    return next(iter(scopes)) if len(scopes) == 1 else str(stream.source_path)


@overload
def paired_axes(
    left: Stream,
    right: Stream,
    relation: ClockRelation,
    *,
    exact: Literal[False] = False,
) -> tuple[_Axis, _Axis]: ...
@overload
def paired_axes(
    left: Stream, right: Stream, relation: ClockRelation, *, exact: Literal[True]
) -> tuple[_ExactAxis, _ExactAxis]: ...
def paired_axes(
    left: Stream, right: Stream, relation: ClockRelation, *, exact: bool = False
) -> tuple[_Axis, _Axis] | tuple[_ExactAxis, _ExactAxis]:
    """Apply only the declared right-to-left transform, recording no inferred sync."""
    if scope_of(left) != relation.left_scope or scope_of(right) != relation.right_scope:
        raise Unavailable("clock relation source scope does not match")
    if relation.claim == "capture_alignment" and not all(
        s.clock_info and s.clock_info.certifies_acquisition for s in (left, right)
    ):
        raise Unavailable(
            "capture alignment requires producer capture evidence on both streams"
        )
    lr, lp, lt, lf = ordered(left)
    rr, rp, rt, rf = ordered(right)
    if exact:
        left_origin, right_origin = _rational(lt[0]), _rational(rt[0])
        left_scale, right_scale = _rational(lf), _rational(rf)
        drift = _rational(relation.drift_ppm) / 1_000_000
        shift = (
            right_origin * right_scale
            - left_origin * left_scale
            + (right_origin * right_scale - _rational(relation.anchor_s)) * drift
            + _rational(relation.offset_s)
        )
        return (
            (lr, lp, [(_rational(t) - left_origin) * left_scale for t in lt]),
            (
                rr,
                rp,
                [
                    (_rational(t) - right_origin) * right_scale * (1 + drift) + shift
                    for t in rt
                ],
            ),
        )
    # Same-scale integer epochs are differenced exactly before conversion.
    delta = (
        (rt[0] - lt[0]) * lf
        if lf == rf
        else float(
            Fraction(rt[0]) * Fraction(str(rf)) - Fraction(lt[0]) * Fraction(str(lf))
        )
    )
    left_t = [(v - lt[0]) * lf for v in lt]
    scale = 1 + relation.drift_ppm * 1e-6
    drift_origin = (
        float(Fraction(rt[0]) * Fraction(str(rf)) - Fraction(str(relation.anchor_s)))
        * relation.drift_ppm
        * 1e-6
    )
    right_t = [
        (v - rt[0]) * rf * scale + delta + drift_origin + relation.offset_s for v in rt
    ]
    if not all(math.isfinite(v) for v in right_t):
        raise Unavailable("clock transform is nonfinite")
    return (lr, lp, left_t), (rr, rp, right_t)


def nearest(
    times: Sequence[float] | Sequence[Fraction],
    target: float | Fraction,
    mode: Literal["nearest", "previous"] = "nearest",
) -> int | None:
    """Find a nearest or causal index, with deterministic earlier-sample ties."""
    i = bisect.bisect_right(times, target)
    if mode == "previous":
        return i - 1 if i else None
    candidates = [j for j in (i - 1, i) if 0 <= j < len(times)]
    return (
        min(candidates, key=lambda j: (abs(times[j] - target), j))
        if candidates
        else None
    )


def numeric(
    stream: Stream, channel: Channel | None, positions: Sequence[int]
) -> list[float | int]:
    """Read one scalar payload without invoking an unbounded lazy fetch."""
    if channel is None or not isinstance(stream.payload, FramePayload):
        raise Unavailable("one numeric channel with an eager payload is required")
    values = stream.payload.frame[channel.name]
    if not values.dtype.is_numeric():
        raise Unavailable("channel is not numeric")
    return [values[i] for i in positions]


def validated(
    channel: Channel | None, properties: Sequence[str], capability: str
) -> ChannelBinding:
    """Require current, scoped per-property evidence for a physical comparison."""
    b = channel.binding if channel else None
    if b is None or not b.source_identity:
        raise Unavailable("scoped channel binding is missing")
    current = b.model_dump(mode="json")
    current["identity"] = f"{b.feature}[{b.index}]"
    checked = {
        v.property
        for v in b.validations
        if v.scope == b.source_identity
        and v.capability in (None, capability)
        and v.value is not None
        and v.value == current.get(v.property)
    }
    if not set(properties) <= checked:
        raise Unavailable(
            "missing scoped validation: " + ", ".join(sorted(set(properties) - checked))
        )
    return b


def finite(v: float | None) -> bool:
    """Whether a scalar contributes observed finite evidence."""
    return v is not None and math.isfinite(v)


def result(
    kind: Literal["timing", "tracking", "motion", "vision", "windows", "cohort"],
    spec: DiagnosticSpec,
    episode: Episode | None = None,
    **kwargs: Any,
) -> DiagnosticResult:
    """Construct one consistently addressed result."""
    return DiagnosticResult(
        id=spec.id,
        kind=kind,
        episode_id=episode.id if episode else None,
        availability=Availability.COMPUTED,
        **kwargs,
    )
