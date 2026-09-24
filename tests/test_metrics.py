"""Verifies metrics: the registry, Requires filtering, and the timing family."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Sequence

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.metrics.registry import (
    metric,
    run_channel_metrics,
    run_episode_metrics,
    run_stream_metrics,
)
from kalanos.analysis.models.domain import Channel, Episode, FramePayload, Kind, Stream
from kalanos.analysis.models.metrics import (
    ChannelContext,
    EpisodeContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
    Requires,
    StreamContext,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# A source path for the stand-in streams below — its content is never read.
_SOURCE_PATH = UPath("test_metrics.csv")

# A 100 Hz clock wandering ±2 ms around its own nominal gap — regular enough
# to pass inference's tolerance band, irregular enough to give dt_jitter_ms
# something to measure.
_JITTERY_CLOCK_GAPS = [0.010, 0.012, 0.008, 0.010]

# A 100 Hz clock that otherwise held, save for three samples missing in a
# row midway through — the "burst" a single dropped-sample counter would
# average away.
_BURST_DROP_TIMESTAMPS = [0.0, 0.01, 0.02, 0.03, 0.07, 0.08, 0.09, 0.10]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _stream_context(
    timestamps: Sequence[float | None],
    *,
    is_regular: bool = True,
    taxonomy_type: str = "unmapped.tcp_pose_x_mm",
) -> StreamContext:
    """Build a StreamContext from a plain list of timestamps.

    The timing family reads only `timestamps` and `is_regular`, so the
    underlying stream carries no payload.

    Parameters
    ----------
    timestamps : Sequence[float or None]
        The time column, in seconds, row order preserved. A `None` stands
        in for a row whose timestamp did not resolve.
    is_regular : bool
        The regularity verdict to hand the context, bypassing inference entirely.
    taxonomy_type : str
        The stream's taxonomy type, for tests that check evidence or naming.

    Returns
    -------
    StreamContext
        A context ready to hand to a metric function or to `run_stream_metrics`.
    """

    stream = Stream(
        taxonomy_type=taxonomy_type,
        kind=Kind.SERIES,
        timestamps=pl.Series("time_s", timestamps),
        source_path=_SOURCE_PATH,
    )
    return StreamContext(stream=stream, is_regular=is_regular)


def _timestamps_from_gaps(gaps: list[float]) -> list[float]:
    """Turn a list of gaps into a cumulative timestamp series starting at zero.

    Parameters
    ----------
    gaps : list[float]
        Consecutive gaps to accumulate.

    Returns
    -------
    list[float]
        `len(gaps) + 1` timestamps, `timestamps[0] == 0.0`.
    """

    timestamps = [0.0]
    for gap in gaps:
        timestamps.append(timestamps[-1] + gap)
    return timestamps


def _episode_stream(taxonomy_type: str) -> Stream:
    """Build a bare Stream of one taxonomy type, for an EpisodeContext's own streams.

    Parameters
    ----------
    taxonomy_type : str
        The stream's taxonomy type.

    Returns
    -------
    Stream
        A minimal stream; its timestamps and payload are never read by the
        taxonomy gate itself.
    """

    return Stream(
        taxonomy_type=taxonomy_type,
        kind=Kind.SERIES,
        timestamps=pl.Series("time_s", [0.0, 1.0]),
        source_path=_SOURCE_PATH,
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_effective_hz_matches_the_hand_computed_rate_on_a_jittery_clock():
    """Verify the median-gap rate, not the mean, drives effective_hz.

    Gaps sorted are [0.008, 0.010, 0.010, 0.012]; the median of the two
    middle values is exactly 0.010 s, so the rate is exactly 100 Hz.
    """

    ctx = _stream_context(_timestamps_from_gaps(_JITTERY_CLOCK_GAPS))

    results = run_stream_metrics(ctx)

    assert results["effective_hz"].status == MetricStatus.REPORT_ONLY
    assert results["effective_hz"].value == pytest.approx(100.0)
    assert results["effective_hz"].unit == "Hz"


def test_dt_jitter_ms_matches_the_hand_computed_spread_on_a_jittery_clock():
    """Verify the reported spread is the sample standard deviation, in ms.

    Hand-computed from the same four gaps: mean 0.010 s, sample variance
    8e-6 / 3 s², spread ≈ 1.633 ms.
    """

    ctx = _stream_context(_timestamps_from_gaps(_JITTERY_CLOCK_GAPS))

    results = run_stream_metrics(ctx)

    assert results["dt_jitter_ms"].status == MetricStatus.REPORT_ONLY
    assert results["dt_jitter_ms"].value == pytest.approx(1.633, abs=1e-3)
    assert results["dt_jitter_ms"].unit == "ms"


def test_drop_rate_matches_the_hand_computed_fraction_on_a_burst_drop():
    """Verify a three-sample gap in the middle of an otherwise steady clock.

    Median gap stays 0.01 s despite the burst. Duration is 0.10 s, so the
    expected count is 0.10 / 0.01 + 1 = 11 against 8 observed —
    a drop_rate of 3/11.
    """

    ctx = _stream_context(_BURST_DROP_TIMESTAMPS)

    results = run_stream_metrics(ctx)

    assert results["drop_rate"].status == MetricStatus.REPORT_ONLY
    assert results["drop_rate"].value == pytest.approx(3 / 11)
    assert results["drop_rate"].unit == "fraction"


def test_all_three_timing_metrics_are_not_applicable_on_an_irregular_series():
    """Verify the irregular flag alone gates the whole family, without recomputing it.

    The timestamps here are the jitter fixture again — plausible, evenly
    spaced data — but `is_regular=False` must still turn every one of the
    three away, proving the registry trusts the flag rather than judging
    the gaps for itself.
    """

    ctx = _stream_context(_timestamps_from_gaps(_JITTERY_CLOCK_GAPS), is_regular=False)

    results = run_stream_metrics(ctx)

    for name in ("effective_hz", "dt_jitter_ms", "drop_rate"):
        assert results[name].status == MetricStatus.NOT_APPLICABLE
        assert results[name].value is None
        assert "reason" in results[name].evidence


def test_a_collapsed_clock_returns_not_applicable_rather_than_raising_or_zero():
    """Verify duplicate timestamps fail closed instead of dividing by zero.

    Three identical timestamps carry two zero-length gaps: a genuine clock
    would never do this, so the honest answer is "could not evaluate",
    not a jitter of 0 ms or an unbounded rate.
    """

    ctx = _stream_context([0.0, 0.0, 0.0])

    results = run_stream_metrics(ctx)

    for name in ("effective_hz", "dt_jitter_ms", "drop_rate"):
        assert results[name].status == MetricStatus.NOT_APPLICABLE
        assert results[name].value is None


def test_a_null_timestamp_is_dropped_rather_than_crashing_the_metric():
    """Verify a hole in the time column is skipped, not subtracted against None.

    `loading` sorts unresolved timestamps to the tail rather than dropping
    them, so a well-formed stream can still hand this stage a null partway
    through. Four valid timestamps survive the drop here —
    [0.0, 0.01, 0.03, 0.04] — giving gaps of [0.01, 0.02, 0.01].
    """

    ctx = _stream_context([0.0, 0.01, None, 0.03, 0.04])

    results = run_stream_metrics(ctx)

    assert results["effective_hz"].value == pytest.approx(100.0)
    assert results["dt_jitter_ms"].value == pytest.approx(5.7735, abs=1e-3)
    assert results["drop_rate"].value == pytest.approx(0.2)


def test_a_stream_rejects_a_payload_and_timestamps_length_mismatch():
    """Verify the Stream itself refuses a payload that cannot pair up index-for-index.

    `StreamContext` reads `timestamps` and `payload` straight through to the
    stream it wraps rather than holding its own copies, so this invariant is
    checked once, here, rather than a second time on the context.
    """

    payload = FramePayload(frame=pl.DataFrame({"tcp_pose_x_mm": [0.0, 1.0]}))

    with pytest.raises(ValueError, match="Stream must carry one timestamp"):
        Stream(
            taxonomy_type="unmapped.tcp_pose_x_mm",
            kind=Kind.SERIES,
            timestamps=pl.Series("time_s", [0.0, 1.0, 2.0]),
            payload=payload,
            source_path=_SOURCE_PATH,
        )


def test_channel_context_rejects_a_values_and_timestamps_length_mismatch():
    """Verify the model itself refuses two series that cannot pair up index-for-index.

    Catching this here means no metric function ever has to guard against
    it — a mismatch would otherwise surface many calls later as a
    `StatisticsError` inside whichever function first zipped the two together.
    """

    stream_ctx = _stream_context([0.0, 1.0])

    with pytest.raises(ValueError, match="ChannelContext must carry one timestamp"):
        ChannelContext(
            channel=Channel(name="tcp_pose_x_mm"),
            values=pl.Series("tcp_pose_x_mm", [0.0, 1.0, 2.0]),
            stream=stream_ctx,
        )


def test_the_registry_declines_a_metric_below_its_min_samples_requirement(monkeypatch):
    """Verify min_samples is enforced by the registry, not by a metric's own code.

    A stub metric with an unreasonably high `min_samples` is registered
    against a temporary copy of the registry, so this test cannot leak a
    fake metric into any other test's `run_stream_metrics` call.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(
        level=Level.STREAM,
        family=Family.TIMING,
        requires=Requires(min_samples=1000),
    )
    def _needs_a_thousand_samples(ctx: StreamContext) -> MetricResult:
        raise AssertionError("the registry should never have called this")

    ctx = _stream_context(_timestamps_from_gaps(_JITTERY_CLOCK_GAPS))

    results = run_stream_metrics(ctx)

    result = results["_needs_a_thousand_samples"]
    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "1000" in result.evidence["reason"]


def test_run_channel_metrics_routes_a_channel_level_metric_to_its_own_context(
    monkeypatch,
):
    """Verify the channel runner calls a channel-level metric with a ChannelContext.

    Registers a stub against a temporary copy of the registry, alongside
    whatever channel-level metrics are already built in, so the assertion
    checks that the stub ran rather than pinning the full registered set.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.CHANNEL, family=Family.INTEGRITY)
    def peak_value(ctx: ChannelContext) -> MetricResult:
        return MetricResult(
            value=max(ctx.values.to_list()), unit=None, status=MetricStatus.REPORT_ONLY
        )

    stream_ctx = _stream_context(_timestamps_from_gaps(_JITTERY_CLOCK_GAPS))
    ctx = ChannelContext(
        channel=Channel(name="tcp_pose_x_mm"),
        values=pl.Series("tcp_pose_x_mm", [0.0, 3.0, 1.0, 2.0, 0.5]),
        stream=stream_ctx,
    )

    results = run_channel_metrics(ctx)

    assert "peak_value" in results
    assert results["peak_value"].value == pytest.approx(3.0)


def test_an_unknown_family_is_refused_at_decoration_time():
    """Verify @metric coerces family, so a typo fails at import rather than at grading.

    A family outside the nine has no weight key in the policy, so a metric
    carrying one would compute a number nothing could ever grade.
    """

    with pytest.raises(ValueError, match="not_a_family"):

        @metric(level=Level.STREAM, family="not_a_family")
        def _wrong_family(ctx: StreamContext) -> MetricResult:
            raise AssertionError("the decorator should never have registered this")


def test_stream_taxonomy_gating_is_any_of(monkeypatch):
    """Verify Requires.taxonomy on a stream metric runs on either declared type.

    A metric wanting one of two proprioceptive types runs on a stream of
    the first; on an unrelated type it comes back not_applicable with a
    reason naming both declared types and the type it actually got.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(
        level=Level.STREAM,
        family=Family.MOTION,
        requires=Requires(taxonomy=["proprio.joint_position", "proprio.ee_position"]),
    )
    def _wants_proprio(ctx: StreamContext) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    matching = run_stream_metrics(
        _stream_context([0.0, 1.0], taxonomy_type="proprio.joint_position")
    )["_wants_proprio"]
    other = run_stream_metrics(_stream_context([0.0, 1.0], taxonomy_type="extero.rgb"))[
        "_wants_proprio"
    ]

    assert matching.status == MetricStatus.REPORT_ONLY
    assert other.status == MetricStatus.NOT_APPLICABLE
    reason = other.evidence["reason"]
    assert "proprio.joint_position" in reason
    assert "proprio.ee_position" in reason
    assert "extero.rgb" in reason


def test_channel_taxonomy_gating_reads_through_to_its_stream(monkeypatch):
    """Verify a channel-level taxonomy gate reads the channel's stream, not its name.

    `ChannelContext.taxonomy_type` reads through to `self.stream`, so the
    same channel name must gate differently depending only on which
    stream it was built against.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(
        level=Level.CHANNEL,
        family=Family.MOTION,
        requires=Requires(taxonomy=["proprio.joint_position"]),
    )
    def _wants_joint_position(ctx: ChannelContext) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    def _channel_ctx(taxonomy_type: str) -> ChannelContext:
        return ChannelContext(
            channel=Channel(name="tcp_pose_x_mm"),
            values=pl.Series("tcp_pose_x_mm", [0.0, 1.0]),
            stream=_stream_context([0.0, 1.0], taxonomy_type=taxonomy_type),
        )

    matching = run_channel_metrics(_channel_ctx("proprio.joint_position"))[
        "_wants_joint_position"
    ]
    other = run_channel_metrics(_channel_ctx("extero.rgb"))["_wants_joint_position"]

    assert matching.status == MetricStatus.REPORT_ONLY
    assert other.status == MetricStatus.NOT_APPLICABLE


def test_episode_taxonomy_gating_is_all_of(monkeypatch):
    """Verify Requires.taxonomy on an episode metric needs every declared type present.

    An episode carrying only one of two declared types comes back
    not_applicable naming the missing one, in a shape distinct from the
    stream-level any-of reason; carrying both, the metric runs.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(
        level=Level.EPISODE,
        family=Family.CONSISTENCY,
        requires=Requires(taxonomy=["proprio.joint_position", "extero.rgb"]),
    )
    def _compares_two_streams(ctx: EpisodeContext) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    one_type = Episode(id="e1", streams=[_episode_stream("proprio.joint_position")])
    both_types = Episode(
        id="e2",
        streams=[
            _episode_stream("proprio.joint_position"),
            _episode_stream("extero.rgb"),
        ],
    )

    missing = run_episode_metrics(EpisodeContext(episode=one_type))[
        "_compares_two_streams"
    ]
    complete = run_episode_metrics(EpisodeContext(episode=both_types))[
        "_compares_two_streams"
    ]

    assert missing.status == MetricStatus.NOT_APPLICABLE
    reason = missing.evidence["reason"]
    assert "extero.rgb" in reason
    assert "carries no" in reason
    assert "needs one of" not in reason
    assert complete.status == MetricStatus.REPORT_ONLY


def test_an_episode_metric_with_min_samples_is_refused_at_decoration_time():
    """Verify min_samples or regular_sampling on an EPISODE metric raises at import.

    An episode has no single sample count or regularity verdict, so an
    episode metric must check anything structural itself, inside the
    function, where the reason can name the stream it applies to.
    """

    with pytest.raises(ValueError, match="_needs_min_samples"):

        @metric(
            level=Level.EPISODE,
            family=Family.CONSISTENCY,
            requires=Requires(min_samples=64),
        )
        def _needs_min_samples(ctx: EpisodeContext) -> MetricResult:
            raise AssertionError("the decorator should never have registered this")


def test_an_episode_metric_receives_the_episode_and_can_reach_its_streams(monkeypatch):
    """Verify an episode metric is handed an EpisodeContext it can query by type.

    Returns a value derived from `streams_of` on both declared types,
    proving the function can reach the actual streams, not just that the
    registry called it.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(
        level=Level.EPISODE,
        family=Family.CONSISTENCY,
        requires=Requires(taxonomy=["proprio.joint_position", "extero.rgb"]),
    )
    def _stream_count_gap(ctx: EpisodeContext) -> MetricResult:
        assert isinstance(ctx, EpisodeContext)
        joint = ctx.streams_of("proprio.joint_position")
        rgb = ctx.streams_of("extero.rgb")
        return MetricResult(
            value=float(len(joint) - len(rgb)),
            unit=None,
            status=MetricStatus.REPORT_ONLY,
        )

    episode = Episode(
        id="e1",
        streams=[
            _episode_stream("proprio.joint_position"),
            _episode_stream("proprio.joint_position"),
            _episode_stream("extero.rgb"),
        ],
    )

    result = run_episode_metrics(EpisodeContext(episode=episode))["_stream_count_gap"]

    assert result.value == pytest.approx(1.0)
