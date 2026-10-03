"""Shared diagnostics builders: validated synthetic channels, episodes and assembly."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

# External
import numpy as np
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.models.binding import (
    ChannelBinding,
    CommandSemantics,
    EvaluationScope,
    Quantity,
    Representation,
    RequirementsSection,
    Validation,
)
from kalanos.analysis.models.diagnostics import (
    ClockRelation,
    DiagnosticPlan,
    DiagnosticReviewPolicy,
    ModalitySpec,
    Selector,
    VisionSpec,
    WindowSpec,
)
from kalanos.analysis.models.domain import (
    Channel,
    ClockInfo,
    ClockOrigin,
    Episode,
    FramePayload,
    Kind,
    MappingSource,
    OriginEvidence,
    Stream,
    TimestampDtype,
)
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import AnalysedEpisode, Report
from kalanos.analysis.reporting.assemble import assemble_report
from kalanos.assets.policy import load_default_policy


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def signal(
    feature: str = "state",
    values: Iterable[float] | None = None,
    times: Iterable[float] | None = None,
    command: str = "none",
) -> Stream:
    """Create an explicitly validated synthetic channel, never production evidence.

    Parameters
    ----------
    feature : str, default "state"
        Feature name bound to the single channel and used as the source field.
    values : Iterable[float] or None, optional
        Channel values. Defaults to 101 points linearly spaced over `[0, 1]`.
    times : Iterable[float] or None, optional
        Timestamps, one per value. Defaults to a 100 Hz series starting at 0.
    command : str, default "none"
        Command recorded on the channel binding.

    Returns
    -------
    Stream
        A single-channel stream with a fully validated binding.
    """
    values = list(values if values is not None else np.linspace(0, 1, 101))
    times = list(times if times is not None else np.arange(len(values)) / 100)
    b = ChannelBinding(
        feature=feature,
        index=0,
        taxonomy_type="proprio.joint_position",
        quantity=Quantity.POSITION,
        representation=Representation.CONTINUOUS,
        unit="rad",
        frame="joint",
        command=CommandSemantics(command),
        source_identity="fixture-session",
    )
    current = b.model_dump(mode="json")
    current["identity"] = f"{feature}[0]"
    b.validations = [
        Validation(
            property=k,
            value=current[k],
            scope="fixture-session",
            validator="test",
            evidence="synthetic only",
        )
        for k in ("identity", "quantity", "unit", "frame", "representation", "command")
    ]
    return Stream(
        taxonomy_type="proprio.joint_position",
        mapping_source=MappingSource.DECLARED_NAMES,
        kind=Kind.SERIES,
        source_path=UPath("fixture.csv"),
        source_field=feature,
        timestamps=pl.Series(times),
        timestamp_dtype=TimestampDtype.FLOAT64,
        clock_info=ClockInfo(
            origin=ClockOrigin.CAPTURE,
            origin_evidence=OriginEvidence.PRODUCER,
            native_unit="s",
            domain="fixture",
        ),
        is_regular=True,
        payload=FramePayload(frame=pl.DataFrame({"value": values})),
        channels=[Channel(name="value", source_index=0, binding=b)],
    )


def relation(**updates: Any) -> ClockRelation:
    """Return a fixture-only reviewed relation between two source scopes.

    Parameters
    ----------
    **updates : Any
        Field overrides forwarded to ClockRelation, such as
        `offset_s`, `drift_ppm`, `anchor_s` or `claim`.

    Returns
    -------
    ClockRelation
        The relation, with both scopes fixed to `"fixture-session"`.
    """
    return ClockRelation(
        left_scope="fixture-session",
        right_scope="fixture-session",
        evidence="fixture-only clock wiring",
        domain="fixture",
        **updates,
    )


def episode(*streams: Stream, identifier: str = "e") -> Episode:
    """Build one canonical episode without passing through an adapter.

    Parameters
    ----------
    *streams : Stream
        Streams to include in the episode.
    identifier : str, default "e"
        Episode identifier.

    Returns
    -------
    Episode
        An episode containing the given streams and a fixture task.
    """
    return Episode(id=identifier, streams=list(streams), tasks=["fixture task"])


def audit(
    episodes: Sequence[Episode],
    plan: DiagnosticPlan | None = None,
    requirements: RequirementsSection | None = None,
    tier: str = "standard",
    review: DiagnosticReviewPolicy | None = None,
    vision: VisionSpec | None = None,
) -> Report:
    """Exercise the real assembly, coverage and single eligibility calculation.

    Parameters
    ----------
    episodes : Sequence[Episode]
        Episodes to analyse, each wrapped in an AnalysedEpisode with
        the default policy and a CSV adapter of full confidence.
    plan : DiagnosticPlan or None, optional
        Diagnostics plan forwarded to assemble_report.
    requirements : RequirementsSection or None, optional
        Pass/fail requirements forwarded to assemble_report.
    tier : str, default "standard"
        Execution tier recorded on the evaluation scope.
    review : DiagnosticReviewPolicy or None, optional
        Overrides the default policy's `diagnostic_reviews` when given.
    vision : VisionSpec or None, optional
        Vision sampling settings. Defaults to VisionSpec defaults.

    Returns
    -------
    Report
        The assembled report.
    """
    policy = load_default_policy()
    if review:
        policy.diagnostic_reviews = review
    vision = vision or VisionSpec()
    return assemble_report(
        root=UPath("fixture"),
        analysed=[
            AnalysedEpisode(
                episode=e, adapter="csv", policy=policy, adapter_confidence=1
            )
            for e in episodes
        ],
        policy=policy,
        diagnostics_plan=plan,
        vision_samples=vision.sample_frames,
        full_frame_scan=vision.full_frame_scan,
        vision=vision,
        requirements=requirements,
        scope=EvaluationScope(
            requirements_id="test", policy_id="test", tier=ExecutionTier(tier)
        ),
    )


def window_spec(**updates: Any) -> WindowSpec:
    """Declare a ten-hertz, two-history plus two-future window contract.

    Parameters
    ----------
    **updates : Any
        Field overrides forwarded to WindowSpec, such as `stride`,
        `max_windows`, `max_probes`, `padding` or `interpolation`.

    Returns
    -------
    WindowSpec
        The window specification, anchored on `state[0]`.
    """
    return WindowSpec(
        id="train",
        anchor=Selector(feature="state", index=0),
        modalities=[
            ModalitySpec(
                selector=Selector(feature="state", index=0),
                max_age_s=0.00001,
                matching="nearest",
            )
        ],
        sample_rate_hz=10,
        history_steps=2,
        prediction_steps=2,
        max_gap_s=0.11,
        **updates,
    )


def two_channels(*, missing: bool = False) -> Episode:
    """Create two original indices sharing one feature and source clock.

    Parameters
    ----------
    missing : bool, default False
        When set, blanks out the fifth sample of the second channel.

    Returns
    -------
    Episode
        A single-stream episode with two channels, `value` and `other`.
    """
    s = signal(values=list(np.arange(10, dtype=float)), times=np.arange(10) / 10)
    other = s.channels[0].model_copy(deep=True)
    other.name = "other"
    other.source_index = 1
    assert other.binding is not None
    other.binding.index = 1
    s.channels.append(other)
    values: list[float | None] = list(np.arange(10, dtype=float))
    if missing:
        values[4] = None
    s.payload = FramePayload(
        frame=pl.DataFrame({"value": list(np.arange(10, dtype=float)), "other": values})
    )
    return episode(s)
