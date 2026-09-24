"""Verifies reporting: assembling the graded Report, and its four renderers."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging

# External
import polars as pl
import pytest
import yaml
from upath import UPath

# Internal
from kalanos.analysis.adapters.csv import CsvAdapter
from kalanos.analysis.models.discovery import SkippedSource, SkipReason
from kalanos.analysis.models.domain import (
    Attribution,
    Channel,
    Episode,
    FramePayload,
    Kind,
    Stream,
)
from kalanos.analysis.models.metrics import (
    Family,
    Level,
    MetricResult,
    MetricStatus,
)
from kalanos.analysis.models.policy import Band, MetricPolicy, Policy
from kalanos.analysis.models.report import (
    CURRENT_SCHEMA_VERSION,
    AnalysedEpisode,
    GradedChannel,
    GradedEpisode,
    GradedStream,
    Report,
)
from kalanos.analysis.models.schema import SourceSchema, UnresolvedSource
from kalanos.analysis.models.scoring import (
    Finding,
    FindingLocation,
    ScoreResult,
    Severity,
)
from kalanos.analysis.reporting.assemble import (
    assemble_report,
    grade_episode,
    grade_stream,
)
from kalanos.analysis.reporting.card import (
    _BUCKET_LABELS,
    _MAX_EPISODE_ROWS,
    _METRIC_BUCKETS,
    _RAIL_CELLS,
    _RAIL_MIN_WIDTH,
    _metric_counts,
    render_terminal,
)
from kalanos.analysis.reporting.registry import registered_reporters
from kalanos.analysis.reporting.render import render_html, render_json, render_yaml
from kalanos.analysis.reporting.write import write_report
from kalanos.analysis.scoring.score import grade_for, rollup, score_metrics
from kalanos.assets.policy import load_default_policy

# Local
from helpers import DisclosureStateCollector, ScoreAttributeCollector, score_attr


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FIXTURE = UPath(__file__).parent / "fixtures" / "arm_multi_device.csv"

# The letters table `_score`'s stand-in ScoreResults grade against —
# loaded once, since every test in this module shares the same default policy.
_POLICY = load_default_policy()


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _score(level: Level, score: float | None) -> ScoreResult:
    """Build a ScoreResult with just enough fields set for a Graded* node.

    Mirrors `test_scoring.py`'s own helper of the same name.

    Parameters
    ----------
    level : Level
        The level this stand-in result attaches to.
    score : float or None
        The score to carry; `grade` and `train_ready` follow from it.

    Returns
    -------
    ScoreResult
        A result usable anywhere a Graded* model wants one.
    """

    return ScoreResult(
        level=level,
        score=score,
        grade=grade_for(score, _POLICY) if score is not None else None,
        train_ready=score >= 70.0 if score is not None else None,
        n_contributing=1 if score is not None else 0,
    )


def _metric(
    value: float | None, status: MetricStatus, *, unit: str | None = "fraction"
) -> MetricResult:
    """Build a MetricResult for a rendering test, evidence included.

    Parameters
    ----------
    value : float or None
        The metric's value.
    status : MetricStatus
        The status to render against.
    unit : str or None
        The value's unit.

    Returns
    -------
    MetricResult
        A result carrying a small, fixed evidence dict.
    """

    return MetricResult(
        value=value, unit=unit, status=status, evidence={"reason": "test evidence"}
    )


def _graded_channel(
    name: str, score: float | None, metrics: dict[str, MetricResult] | None = None
) -> GradedChannel:
    """Build a GradedChannel, one level of a hand-built Report tree.

    Parameters
    ----------
    name : str
        The channel's name.
    score : float or None
        The channel's rolled-up score.
    metrics : dict[str, MetricResult] or None
        The channel's metric results, keyed by name.

    Returns
    -------
    GradedChannel
        A channel ready to nest in a GradedStream.
    """

    return GradedChannel(
        channel=Channel(name=name),
        score=_score(Level.CHANNEL, score),
        metrics=metrics or {},
    )


def _sample_report() -> Report:
    """Build a small, fully nested Report for rendering tests.

    One episode, one stream carrying a warning metric of its own, one
    channel nested in it carrying a critical and a not_applicable metric
    side by side, and one skipped source. `findings` mirrors the tree's
    own two defects, worst first.

    Returns
    -------
    Report
        A report exercising every level of the graded tree at once.
    """

    channel = _graded_channel(
        "tcp_pose_z_mm",
        50.0,
        metrics={
            "drop_rate": _metric(0.081, MetricStatus.CRITICAL),
            "effective_hz": _metric(None, MetricStatus.NOT_APPLICABLE, unit=None),
        },
    )
    stream = GradedStream(
        taxonomy_type="unmapped.tcp_pose",
        instance="armC",
        score=_score(Level.STREAM, 50.0),
        metrics={"dt_jitter_ms": _metric(3.5, MetricStatus.WARNING, unit="ms")},
        channels=[channel],
    )
    episode = GradedEpisode(
        id="arm_multi_device",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[UPath("arm_multi_device.csv")],
        score=_score(Level.EPISODE, 50.0),
        streams=[stream],
    )
    return Report(
        root=UPath("fixtures"),
        score=_score(Level.DATASET, 50.0),
        episodes=[episode],
        findings=[
            Finding(
                metric_id="timing.drop_rate",
                family="timing",
                severity=Severity.CRITICAL,
                value=0.081,
                unit="fraction",
                points=0.0,
                episode_id="arm_multi_device",
                stream="unmapped.tcp_pose",
                instance="armC",
                channel="tcp_pose_z_mm",
                evidence={"reason": "test evidence"},
            ),
            Finding(
                metric_id="timing.dt_jitter_ms",
                family="timing",
                severity=Severity.WARNING,
                value=3.5,
                unit="ms",
                points=55.0,
                episode_id="arm_multi_device",
                stream="unmapped.tcp_pose",
                instance="armC",
                channel=None,
                evidence={"reason": "test evidence"},
            ),
        ],
        skipped=[
            SkippedSource(path=UPath("video_meta.json"), reason=SkipReason.NO_ADAPTER)
        ],
        policy_version=1,
        duration_s=0.31,
    )


def _analysed_fixture() -> tuple[Episode, str]:
    """Run the real fixture through CsvAdapter.

    Returns
    -------
    tuple[Episode, str]
        The pair `assemble_report` expects for one analysed recording:
        the fixture's episode, and the name of the adapter that read it.
    """

    return next(CsvAdapter().episodes(FIXTURE)), "csv"


def _series(timestamps: list[float]) -> Stream:
    """Build a one-channel series stream sampled at the given times.

    Parameters
    ----------
    timestamps : list[float]
        The stream's sample times, in seconds.

    Returns
    -------
    Stream
        A series stream with one all-zero channel, one row per timestamp.
    """

    return Stream(
        taxonomy_type="robot.joint.position",
        kind=Kind.SERIES,
        timestamps=pl.Series(timestamps, dtype=pl.Float64),
        payload=FramePayload(frame=pl.DataFrame({"x": [0.0] * len(timestamps)})),
        source_path=UPath("recording.csv"),
        channels=[Channel(name="x")],
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_report_carries_the_current_schema_version():
    """Verify a freshly assembled Report defaults to the version it was built under."""

    report = assemble_report(root=UPath("."), analysed=[], policy=load_default_policy())

    assert report.schema_version == CURRENT_SCHEMA_VERSION


def test_skipped_and_unresolved_sources_land_on_the_report_verbatim():
    """Verify a run with nothing analysed still reports what it declined to."""

    skipped = [
        SkippedSource(path=UPath("video_meta.json"), reason=SkipReason.NO_ADAPTER)
    ]
    unresolved = [
        UnresolvedSource(
            path=UPath("mystery.csv"),
            schema_so_far=SourceSchema(),
            reason="no time column found",
        )
    ]

    report = assemble_report(
        root=UPath("."),
        analysed=[],
        policy=load_default_policy(),
        skipped=skipped,
        unresolved=unresolved,
    )

    assert report.skipped == skipped
    assert report.unresolved == unresolved
    assert report.score.score is None


def test_a_stream_with_no_channels_is_never_fetched():
    """Verify a channel-less stream, such as a video, is graded without fetching.

    A `Payload` whose `fetch` fails the test if called stands in for a
    camera stream: decoding every frame of every episode to grade a
    dataset is not a plausible thing to do.
    """

    mapped, _ = _analysed_fixture()
    original = mapped.streams[0]

    class _ExplodingPayload:
        def __len__(self) -> int:
            return len(original.timestamps)

        def fetch(self):
            pytest.fail("payload.fetch() was called on a channel-less stream")

    video_like = original.model_copy(
        update={
            "payload": _ExplodingPayload(),
            "channels": [],
            "kind": Kind.VIDEO,
        }
    )

    graded, _findings = grade_stream(
        video_like, policy=load_default_policy(), is_regular=True, episode_id=mapped.id
    )

    assert graded.channels == []


def test_a_stream_with_no_payload_grades_its_own_metrics():
    """Verify payload=None no longer raises: the stream's own metrics still run."""

    mapped, _ = _analysed_fixture()
    empty = mapped.streams[0].model_copy(update={"payload": None, "channels": []})

    graded, _findings = grade_stream(
        empty, policy=load_default_policy(), is_regular=True, episode_id=mapped.id
    )

    assert graded.channels == []
    assert graded.metrics


def test_a_stream_with_channels_but_no_payload_warns_and_grades_no_channels(caplog):
    """Verify a stream that declares channels but carries no payload warns,
    rather than fetching a payload that isn't there or raising.
    """

    mapped, _ = _analysed_fixture()
    original = mapped.streams[0]
    assert original.channels, (
        "fixture stream must declare channels for this to mean anything"
    )
    orphaned = original.model_copy(update={"payload": None})

    with caplog.at_level(logging.WARNING):
        graded, _findings = grade_stream(
            orphaned,
            policy=load_default_policy(),
            is_regular=True,
            episode_id=mapped.id,
        )

    assert graded.channels == []
    assert any(
        "no payload" in record.getMessage()
        and original.taxonomy_type in record.getMessage()
        for record in caplog.records
    )


def test_grade_episode_rolls_up_an_episode_with_no_streams_to_score_none():
    """Verify an episode with no streams at all doesn't crash.

    A stream-less Episode is a real shape an adapter can produce — a source
    with nothing left after excluding its time and key columns — so
    `grade_episode`'s rollups need to fold an empty child list into
    `score=None` rather than raising on it.
    """

    episode = Episode(id="time_only", streams=[])

    graded, findings = grade_episode(
        episode, adapter="csv", adapter_confidence=0.9, policy=load_default_policy()
    )

    assert graded.streams == []
    assert graded.score.score is None
    assert graded.duration_s is None
    assert graded.sample_count == 0
    assert findings == []


def test_an_episode_spans_from_its_earliest_to_its_latest_timestamp():
    """Verify an episode's duration covers every stream, not just the first one.

    Streams in one recording start and stop at different times;
    the recording spans all of them.
    """

    episode = Episode(
        id="rec", streams=[_series([0.0, 0.5, 1.0]), _series([0.25, 1.5])]
    )

    graded, _ = grade_episode(
        episode, adapter="csv", adapter_confidence=1.0, policy=load_default_policy()
    )

    assert graded.duration_s == pytest.approx(1.5)
    assert graded.sample_count == 3


def test_grade_episode_score_is_unchanged_with_no_episode_metrics_registered(
    monkeypatch,
):
    """Verify an episode's rollup still equals a plain rollup of its streams.

    Filters EPISODE-level entries out of a temporary registry copy, rather
    than relying on none being registered in the shipped catalogue — that
    would break, for no real reason, the moment a first episode metric
    lands. The invariant under test is that an empty `results` dict
    contributes nothing to the rollup, never a shifting zero.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(
        registry,
        "_REGISTRY",
        [entry for entry in registry._REGISTRY if entry.level is not Level.EPISODE],
    )

    raw_episode, adapter = _analysed_fixture()
    policy = load_default_policy()

    graded, _findings = grade_episode(
        raw_episode, adapter=adapter, adapter_confidence=0.9, policy=policy
    )

    assert graded.metrics == {}
    assert graded.score == rollup(
        Level.EPISODE, [gs.score for gs in graded.streams], policy=policy
    )


def test_grade_episode_folds_its_own_score_in_as_one_more_equal_weight_child(
    monkeypatch,
):
    """Verify an episode's score is the plain mean of its own metrics and its streams'.

    Mirrors `test_grade_stream_folds_its_own_score_in_as_one_more_equal_weight_child`:
    a stub episode metric graded to one end of the scale, contrasted
    against a stub stream metric graded to the other, makes the two
    contributions distinguishable.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.STREAM, family=Family.INTEGRITY)
    def stub_stream_metric(ctx) -> MetricResult:
        return MetricResult(value=0.0, unit=None, status=MetricStatus.REPORT_ONLY)

    @registry.metric(level=Level.EPISODE, family=Family.CONSISTENCY)
    def stub_episode_metric(ctx) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    policy = Policy(
        schema_version=1,
        metrics={
            "stub.stub_stream_metric": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=1.0)}
            ),
            "stub.stub_episode_metric": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=1.0)}
            ),
        },
        letters={"A": 90.0, "B": 80.0, "C": 70.0, "D": 60.0},
    )
    stream = Stream(
        taxonomy_type="unmapped.tcp_pose",
        instance="armA",
        kind=Kind.SERIES,
        timestamps=pl.Series("time_s", [0.0, 1.0]),
        source_path=UPath("test_reporting.csv"),
    )
    episode = Episode(id="episode_0", streams=[stream])

    graded, _findings = grade_episode(
        episode, adapter="csv", adapter_confidence=0.9, policy=policy
    )

    assert graded.metrics["stub_episode_metric"].status == MetricStatus.CRITICAL
    assert graded.score.score == pytest.approx(50.0)


def test_an_episode_level_finding_names_no_stream_or_channel(monkeypatch):
    """Verify a finding an episode-level metric raises is addressed by episode alone.

    A stream-level metric on the same episode still names its stream, so
    the two findings' addresses stay distinguishable in the same list.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.STREAM, family=Family.INTEGRITY)
    def stub_stream_metric(ctx) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    @registry.metric(level=Level.EPISODE, family=Family.CONSISTENCY)
    def stub_episode_metric(ctx) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    policy = Policy(
        schema_version=1,
        metrics={
            "stub.stub_stream_metric": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=1.0)}
            ),
            "stub.stub_episode_metric": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=1.0)}
            ),
        },
        letters={"A": 90.0, "B": 80.0, "C": 70.0, "D": 60.0},
    )
    stream = Stream(
        taxonomy_type="unmapped.tcp_pose",
        instance="armA",
        kind=Kind.SERIES,
        timestamps=pl.Series("time_s", [0.0, 1.0]),
        source_path=UPath("test_reporting.csv"),
    )
    episode = Episode(id="episode_0", streams=[stream])

    _graded, findings = grade_episode(
        episode, adapter="csv", adapter_confidence=0.9, policy=policy
    )

    by_metric = {finding.metric_id: finding for finding in findings}
    episode_finding = by_metric["stub.stub_episode_metric"]
    stream_finding = by_metric["stub.stub_stream_metric"]
    assert episode_finding.episode_id == "episode_0"
    assert episode_finding.stream is None
    assert episode_finding.channel is None
    assert stream_finding.stream == "unmapped.tcp_pose"


def test_assemble_report_grades_the_fixtures_four_instances_to_known_scores():
    """Verify assemble_report's actual numbers against the fixture's own pathologies.

    `tests/fixtures/README.md` names what each instance is: `armA` is the
    negative control, `armB` carries a flatlined `tcp_pose_z_mm` channel,
    `armC` carries a burst of dropped samples, `armD` has a jittery clock
    that stays ungraded. With `integrity` bands grading alongside `timing`,
    every instance's own noise floor and flatline runs move its score off
    100. `armC`'s score and the rollups pin an exact number, since
    `drop_rate` is a settled band; the other three assert only that they
    stay high and that armB — the one with an outright flatlined channel —
    scores below its clean siblings, since their own bands are still
    candidates and pinning their sixteen-digit values would break on the
    next confirmation run. `drop_rate` runs at stream level, so it is
    armC's own metrics that carry its critical status, not any of its
    channels'; `flatline_pct` runs at channel level, so armB's critical
    finding lands on `tcp_pose_z_mm` specifically.
    """

    raw_episode, adapter = _analysed_fixture()
    policy = load_default_policy()

    report = assemble_report(
        root=FIXTURE.parent,
        analysed=[
            AnalysedEpisode(
                episode=raw_episode,
                adapter=adapter,
                adapter_confidence=0.9,
                policy=policy,
            )
        ],
        policy=policy,
    )

    [episode] = report.episodes
    assert len(episode.streams) == 4
    scores_by_instance = {
        stream.instance: stream.score.score for stream in episode.streams
    }
    arm_a, arm_b, arm_c, arm_d = (
        scores_by_instance["armA"],
        scores_by_instance["armB"],
        scores_by_instance["armC"],
        scores_by_instance["armD"],
    )
    assert arm_a is not None
    assert arm_b is not None
    assert arm_c is not None
    assert arm_d is not None

    assert arm_c == pytest.approx(59.826, abs=1e-3)
    assert arm_a > 90.0
    assert arm_d > 90.0
    assert arm_b > 80.0
    assert arm_b < arm_a
    assert arm_b < arm_d

    [armc] = [stream for stream in episode.streams if stream.instance == "armC"]
    assert armc.channels, "armC's arm stream lost its channels somewhere in the walk"
    assert armc.metrics["drop_rate"].status == MetricStatus.CRITICAL

    assert episode.score.score == pytest.approx(84.037, abs=1e-3)
    assert report.score.score == pytest.approx(84.037, abs=1e-3)

    # Two nodes carry the dataset's only critical findings: the
    # stream-level drop_rate on armC, and the channel-level flatline_pct
    # on armB's flatlined channel — proves both levels of grade_stream
    # thread episode_id/instance/channel correctly, not just that a
    # finding of the right severity exists somewhere in the list.
    critical = {
        (finding.metric_id, finding.instance, finding.channel)
        for finding in report.findings
        if finding.severity == Severity.CRITICAL
    }
    assert critical == {
        ("timing.drop_rate", "armC", None),
        ("integrity.flatline_pct", "armB", "tcp_pose_z_mm"),
    }
    assert {finding.episode_id for finding in report.findings} == {episode.id}


def test_assemble_report_rolls_up_exactly_like_an_independent_rollup_of_its_children():
    """Verify every level's score equals `rollup` applied to its own children.

    A stream's score folds in its own metrics alongside its channels', so
    its independent reconstruction re-derives that contribution by calling
    `score_metrics` again over `stream.metrics` — deterministic, since
    interpolation depends only on the value each metric already carries.
    Pins the wiring itself — that `assemble_report` calls `rollup` with the
    right children at the right level — as a separate concern from the
    previous test's real numbers, which pin that the walk reaches the
    right channels in the first place.
    """

    raw_episode, adapter = _analysed_fixture()
    policy = load_default_policy()

    report = assemble_report(
        root=FIXTURE.parent,
        analysed=[
            AnalysedEpisode(
                episode=raw_episode,
                adapter=adapter,
                adapter_confidence=0.9,
                policy=policy,
            )
        ],
        policy=policy,
    )

    [episode] = report.episodes
    assert len(episode.streams) == 4
    assert (
        rollup(Level.DATASET, [ge.score for ge in report.episodes], policy=policy)
        == report.score
    )
    assert (
        rollup(Level.EPISODE, [gs.score for gs in episode.streams], policy=policy)
        == episode.score
    )
    for stream in episode.streams:
        assert stream.channels, f"{stream.instance} lost its channels"
        _, own_score, _ = score_metrics(
            stream.metrics,
            level=Level.STREAM,
            taxonomy_type=stream.taxonomy_type,
            policy=policy,
            location=FindingLocation(episode_id=episode.id, instance=stream.instance),
        )
        assert (
            rollup(
                Level.STREAM,
                [*(gc.score for gc in stream.channels), own_score],
                policy=policy,
            )
            == stream.score
        )


def test_grade_stream_folds_its_own_score_in_as_one_more_equal_weight_child(
    monkeypatch,
):
    """Verify a stream's score is the plain mean of its own metrics and its channels'.

    The fixture corpus alone cannot pin this: every channel currently scores
    `None` (no channel-level metric is registered yet), so its own rollup
    collapses to the stream's own score regardless of how the two would
    combine. Registering one stub metric at each level, graded to opposite
    ends of the scale, makes the two contributions distinguishable.
    """

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.STREAM, family=Family.INTEGRITY)
    def stub_stream_metric(ctx) -> MetricResult:
        return MetricResult(value=0.0, unit=None, status=MetricStatus.REPORT_ONLY)

    @registry.metric(level=Level.CHANNEL, family=Family.INTEGRITY)
    def stub_channel_metric(ctx) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    policy = Policy(
        schema_version=1,
        metrics={
            "stub.stub_stream_metric": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=1.0)}
            ),
            "stub.stub_channel_metric": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=1.0)}
            ),
        },
        letters={"A": 90.0, "B": 80.0, "C": 70.0, "D": 60.0},
    )
    stream = Stream(
        taxonomy_type="unmapped.tcp_pose",
        instance="armA",
        kind=Kind.SERIES,
        timestamps=pl.Series("time_s", [0.0, 1.0]),
        payload=FramePayload(frame=pl.DataFrame({"tcp_pose_x_mm": [0.0, 1.0]})),
        source_path=UPath("test_reporting.csv"),
        channels=[Channel(name="tcp_pose_x_mm")],
    )

    graded, _findings = grade_stream(
        stream, policy=policy, is_regular=True, episode_id="episode_0"
    )

    assert graded.metrics["stub_stream_metric"].status == MetricStatus.GOOD
    [channel] = graded.channels
    assert channel.metrics["stub_channel_metric"].status == MetricStatus.CRITICAL
    assert graded.score.score == pytest.approx(50.0)


def test_assemble_report_regrades_metrics_rather_than_keeping_report_only():
    """Verify scoring's resolve_status actually ran, rather than metrics' own output.

    Every metric in `metrics.timing` returns `report_only` by design, since it
    has no policy to grade against — a Report showing only `report_only`
    everywhere would mean `score_metrics` never ran. `drop_rate` now runs at
    stream level, so the statuses to check live on the stream, not a channel.
    """

    raw_episode, adapter = _analysed_fixture()
    policy = load_default_policy()
    report = assemble_report(
        root=FIXTURE.parent,
        analysed=[
            AnalysedEpisode(
                episode=raw_episode,
                adapter=adapter,
                adapter_confidence=0.9,
                policy=policy,
            )
        ],
        policy=policy,
    )

    statuses = {
        metric.status
        for episode in report.episodes
        for stream in episode.streams
        for metric in stream.metrics.values()
    }
    assert statuses & {MetricStatus.GOOD, MetricStatus.WARNING, MetricStatus.CRITICAL}


def test_report_round_trips_through_json():
    """Verify a Report survives a JSON render and re-parse unchanged."""

    report = _sample_report()

    assert Report.model_validate_json(render_json(report)) == report


def test_write_report_writes_a_json_file_that_parses_back_into_the_model(tmp_path):
    """Verify the file write_report produces is real, parseable JSON."""

    report = _sample_report()
    path = write_report(report, tmp_path / "r.json")

    assert path == tmp_path / "r.json"
    assert Report.model_validate_json(path.read_text()) == report


def test_write_report_writes_an_html_file_containing_the_reports_own_data(tmp_path):
    """Verify write_report's .html path is exercised, not just render_html directly."""

    report = _sample_report()
    path = write_report(report, tmp_path / "r.html")

    html = path.read_text()
    assert "video_meta.json" in html
    assert 'data-status="critical"' in html


def test_write_report_raises_on_an_unknown_suffix(tmp_path):
    """Verify an unsupported extension fails loudly, naming what is supported."""

    report = _sample_report()

    with pytest.raises(ValueError, match=r"unsupported report suffix.*\.html.*\.json"):
        write_report(report, tmp_path / "r.txt")


def test_the_registry_claims_exactly_the_suffixes_write_report_accepts():
    """Verify the registry claims the same four suffixes the suffix map used to.

    `write_report` reads its renderer out of the registry now, so a decoration
    that claimed the wrong extension would change which files a `--report`
    path accepts without any round-trip test noticing.
    """

    claimed = {
        extension
        for reporter in registered_reporters()
        for extension in reporter.extensions
    }

    assert claimed == {".json", ".yaml", ".yml", ".html"}


def test_writing_a_second_report_to_the_same_path_replaces_the_first(tmp_path):
    """Verify a re-run replaces a previous report rather than appending to it.

    Also pins that writing never mutates the Report object handed to it —
    `first` must still describe what it described before the second write.
    """

    first = _sample_report()
    second = first.model_copy(
        update={"episodes": [], "score": _score(Level.DATASET, None)}
    )
    path = tmp_path / "r.json"

    write_report(first, path)
    write_report(second, path)

    reloaded = Report.model_validate_json(path.read_text())
    assert reloaded == second
    assert first.episodes[0].source_paths == [UPath("arm_multi_device.csv")]


def test_every_data_score_in_the_html_matches_the_models_score_at_that_level():
    """Verify the rendered HTML's score attributes match the model, node for node."""

    report = _sample_report()

    collector = ScoreAttributeCollector()
    collector.feed(render_html(report))

    expected = [("dataset", str(report.root), score_attr(report.score))]
    for episode in report.episodes:
        expected.append(("episode", episode.id, score_attr(episode.score)))
        for stream in episode.streams:
            expected.append(("stream", stream.taxonomy_type, score_attr(stream.score)))
            for graded in stream.channels:
                expected.append(
                    ("channel", graded.channel.name, score_attr(graded.score))
                )

    assert collector.rows == expected


def test_a_not_applicable_metric_renders_an_em_dash_and_critical_shows_its_value():
    """Verify the two statuses are visually distinct, and neither is a bare zero."""

    html = render_html(_sample_report())

    assert 'data-status="critical"' in html
    assert 'data-status="not_applicable"' in html
    assert "drop_rate: 0.081 fraction" in html
    assert "effective_hz: —" in html
    assert "effective_hz: 0" not in html


def test_a_streams_own_metric_renders_alongside_its_channels():
    """Verify a metric the stream carries directly, not one of its channels, renders."""

    html = render_html(_sample_report())

    assert 'data-status="warning"' in html
    assert "dt_jitter_ms: 3.5 ms" in html


def test_every_skipped_source_appears_in_the_html_with_its_reason():
    """Verify a file Kalanos declined to analyse is shown, not dropped."""

    html = render_html(_sample_report())

    assert "video_meta.json" in html
    assert "no_adapter" in html


def test_the_cards_overall_line_carries_the_dataset_score_and_grade():
    """Verify the card's first score row is the dataset score, not an episode's.

    Gives the dataset a score no episode shares, so the assertion can only
    pass if `render_terminal` actually reads `report.score` for the OVERALL
    row rather than, say, the first or worst episode's.
    """

    base = _sample_report()
    report = base.model_copy(update={"score": _score(Level.DATASET, 92.0)})

    overall_line = next(
        line for line in render_terminal(report).splitlines() if "OVERALL" in line
    )
    assert "A" in overall_line  # grade_for(92.0) is A
    assert "92.0" in overall_line
    assert "50.0" not in overall_line


def test_an_episode_with_no_score_renders_n_a_on_the_card_not_a_zero_or_an_f():
    """Verify score=None sinks to n/a on the card, the same rule the HTML follows."""

    ungraded = GradedEpisode(
        id="capture_index",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[UPath("capture_index.json")],
        score=_score(Level.EPISODE, None),
        streams=[],
    )
    base = _sample_report()
    report = base.model_copy(update={"episodes": [*base.episodes, ungraded]})

    text = render_terminal(report)

    row = next(line for line in text.splitlines() if "capture_index" in line)
    assert "n/a" in row
    assert "0.0" not in row
    assert " F " not in row


def test_a_finding_on_the_card_is_addressed_down_to_its_instance_and_channel():
    """Verify the finding line names which subject and which channel raised it.

    Four arms in one recording produce four streams of the same type, so a
    finding that omitted the instance would not say which arm to go and look at.
    """

    finding = next(
        line
        for line in render_terminal(_sample_report()).splitlines()
        if line.startswith("FAIL")
    )

    assert "armC:unmapped.tcp_pose" in finding
    assert "tcp_pose_z_mm.drop_rate" in finding


def test_a_streams_own_finding_is_addressed_without_a_channel_segment():
    """Verify a finding raised on the stream itself omits the channel from its address.

    `armC`'s stream carries a warning `dt_jitter_ms` result of its own,
    distinct from its channel's critical `drop_rate` — the two addresses
    must stay visibly different so a reader can tell which node raised which.
    """

    finding = next(
        line
        for line in render_terminal(_sample_report()).splitlines()
        if line.startswith("WARN")
    )

    assert "armC:unmapped.tcp_pose.dt_jitter_ms" in finding
    assert "unmapped.tcp_pose/." not in finding


def test_a_stream_with_no_instance_is_addressed_by_its_type_alone():
    """Verify a single-subject recording gets no empty instance in its address."""

    base = _sample_report()
    [episode] = base.episodes
    [stream] = episode.streams
    report = base.model_copy(
        update={
            "episodes": [
                episode.model_copy(
                    update={"streams": [stream.model_copy(update={"instance": None})]}
                )
            ],
            "findings": [
                finding.model_copy(update={"instance": None})
                for finding in base.findings
            ],
        }
    )

    finding = next(
        line for line in render_terminal(report).splitlines() if line.startswith("FAIL")
    )

    assert "/unmapped.tcp_pose/" in finding
    assert ":unmapped.tcp_pose" not in finding


def test_an_episode_level_finding_is_addressed_by_episode_id_alone():
    """Verify a finding above stream level names no stream, instance or channel.

    No literal stands in for the missing stream — a reader must not mistake
    a placeholder like `episode` for a real stream name.
    """

    base = _sample_report()
    report = base.model_copy(
        update={
            "findings": [
                base.findings[0].model_copy(
                    update={"stream": None, "instance": None, "channel": None}
                )
            ]
        }
    )

    finding = next(
        line for line in render_terminal(report).splitlines() if line.startswith("FAIL")
    )

    assert "arm_multi_device.drop_rate" in finding


def test_the_card_names_the_file_an_episode_was_read_from():
    """Verify the summary row stays actionable when two recordings share a stem.

    An episode id is a recording name, so `runs/a/joints.csv` and `runs/b/joints.csv`
    both land on `joints` and the row alone would not say which one scored badly.
    """

    text = render_terminal(_sample_report())

    row = next(line for line in text.splitlines() if "arm_multi_device.csv" in line)
    assert "50.0" in row


def test_the_card_strips_the_dataset_root_from_every_path_cell():
    """Verify a path under report.root loses that prefix everywhere it renders.

    The plate already prints the root in full one line above, so the RECORDING
    row, the NOT ANALYSED row and the NO SCHEMA row should each carry only the
    part that tells one row apart from another.
    """

    root = UPath("/srv/robot-logs/2026/09/run-14")
    episode = GradedEpisode(
        id="arm",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[root / "arm.csv"],
        score=_score(Level.EPISODE, 50.0),
    )
    report = Report(
        root=root,
        score=_score(Level.DATASET, 50.0),
        episodes=[episode],
        skipped=[
            SkippedSource(path=root / "video_meta.json", reason=SkipReason.NO_ADAPTER)
        ],
        unresolved=[
            UnresolvedSource(
                path=root / "mystery.csv",
                schema_so_far=SourceSchema(),
                reason="no time column found",
            )
        ],
    )

    text = render_terminal(report)
    lines = text.splitlines()

    assert text.count(str(root)) == 1
    recording_row = next(line for line in lines if "arm.csv" in line)
    skipped_row = next(line for line in lines if "video_meta.json" in line)
    unresolved_row = next(line for line in lines if "mystery.csv" in line)
    assert str(root) not in recording_row
    assert str(root) not in skipped_row
    assert str(root) not in unresolved_row


def test_a_path_outside_the_root_still_renders_in_full():
    """Verify a source on another branch, or another protocol, keeps its full path.

    `..` segments back out of the root would read worse than the absolute
    location, so `_under_root` falls back to the full path instead.
    """

    root = UPath("/srv/robot-logs/run-14")
    report = Report(
        root=root,
        score=_score(Level.DATASET, 50.0),
        skipped=[
            SkippedSource(
                path=UPath("/var/tmp/stray.csv"), reason=SkipReason.NO_ADAPTER
            )
        ],
        unresolved=[
            UnresolvedSource(
                path=UPath("s3://bucket/mystery.csv"),
                schema_so_far=SourceSchema(),
                reason="no time column found",
            )
        ],
    )

    text = render_terminal(report)

    assert "/var/tmp/stray.csv" in text
    assert "s3://bucket/mystery.csv" in text
    assert ".." not in text


def test_a_single_file_run_names_the_file_by_its_bare_name():
    """Verify a run pointed at one file still names it, rather than an empty cell.

    `report.root` is then the file itself, so relativising it against itself
    collapses to `.` — the bare file name is what a reader can act on instead.
    """

    path = UPath("/srv/robot-logs/run-14/arm.csv")
    episode = GradedEpisode(
        id="arm",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[path],
        score=_score(Level.EPISODE, 50.0),
    )
    report = Report(root=path, score=_score(Level.DATASET, 50.0), episodes=[episode])

    text = render_terminal(report)
    recordings = text.split("RECORDINGS", 1)[1]

    row = next(line for line in recordings.splitlines() if "arm.csv" in line)
    assert str(path.parent) not in row


def test_a_skipped_source_appears_on_the_card_with_its_reason():
    """Verify the card reports a declined file rather than omitting it."""

    text = render_terminal(_sample_report())

    assert "video_meta.json" in text
    assert "no_adapter" in text


def test_render_terminal_emits_no_ansi_escape_by_default_and_color_true_does():
    """Verify plain output stays plain, and color is strictly opt-in."""

    report = _sample_report()

    assert "\x1b[" not in render_terminal(report)
    assert "\x1b[" in render_terminal(report, color=True)


def test_render_yaml_round_trips_back_into_the_model():
    """Verify a Report survives a YAML render and re-parse unchanged."""

    report = _sample_report()

    assert Report.model_validate(yaml.safe_load(render_yaml(report))) == report


@pytest.mark.parametrize("filename", ["r.yaml", "r.yml"])
def test_write_report_writes_yaml_for_both_suffixes(tmp_path, filename):
    """Verify both YAML suffixes reach render_yaml rather than one raising."""

    report = _sample_report()
    path = write_report(report, tmp_path / filename)

    assert path == tmp_path / filename
    assert Report.model_validate(yaml.safe_load(path.read_text())) == report


def test_every_skipped_source_appears_in_the_yaml_with_its_reason():
    """Verify a file Kalanos declined to analyse is shown, not dropped."""

    text = render_yaml(_sample_report())

    assert "video_meta.json" in text
    assert "no_adapter" in text


def test_a_not_applicable_metric_is_distinct_from_critical_in_the_yaml():
    """Verify the two statuses are visually distinct, and neither is a bare zero."""

    loaded = yaml.safe_load(render_yaml(_sample_report()))
    [channel] = loaded["episodes"][0]["streams"][0]["channels"]

    assert channel["metrics"]["effective_hz"]["status"] == "not_applicable"
    assert channel["metrics"]["effective_hz"]["value"] is None
    assert channel["metrics"]["drop_rate"]["status"] == "critical"
    assert channel["metrics"]["drop_rate"]["value"] == pytest.approx(0.081)


def test_all_four_renderers_agree_on_every_score():
    """Verify YAML, JSON, HTML and the terminal card all report the same scores."""

    report = _sample_report()

    assert Report.model_validate(yaml.safe_load(render_yaml(report))) == report
    assert Report.model_validate_json(render_json(report)) == report

    collector = ScoreAttributeCollector()
    collector.feed(render_html(report))
    expected = [("dataset", str(report.root), score_attr(report.score))]
    for episode in report.episodes:
        expected.append(("episode", episode.id, score_attr(episode.score)))
        for stream in episode.streams:
            expected.append(("stream", stream.taxonomy_type, score_attr(stream.score)))
            for graded in stream.channels:
                expected.append(
                    ("channel", graded.channel.name, score_attr(graded.score))
                )
    assert collector.rows == expected

    card = render_terminal(report)
    overall_line = next(line for line in card.splitlines() if "OVERALL" in line)
    assert f"{report.score.score:.1f}" in overall_line
    for episode in report.episodes:
        label = ", ".join(str(path) for path in episode.source_paths)
        row = next(line for line in card.splitlines() if label in line)
        assert f"{episode.score.score:.1f}" in row


def test_the_card_prints_its_score_table_before_its_findings():
    """Verify the card leads with the score, then names what went wrong."""

    lines = render_terminal(_sample_report()).splitlines()

    overall_index = next(i for i, line in enumerate(lines) if "OVERALL" in line)
    fail_index = next(i for i, line in enumerate(lines) if line.startswith("FAIL"))
    assert overall_index < fail_index


def test_the_card_reads_report_findings_rather_than_rewalking_the_tree():
    """Verify an empty findings list renders no findings, even with a critical metric.

    The sample tree still carries a critical `drop_rate` result, so this can
    only pass if the card stopped deriving findings by walking it.
    """

    report = _sample_report().model_copy(update={"findings": []})

    text = render_terminal(report)
    findings_section = text.split("FINDINGS", 1)[1]
    lines = findings_section.splitlines()

    assert not any(line.startswith("FAIL") for line in lines)
    assert not any(line.startswith("WARN") for line in lines)
    assert "no findings" in findings_section


def test_the_cards_metric_count_includes_the_episodes_own_metrics():
    """Verify _metric_counts folds in an episode's own metrics, not just its streams'.

    `_sample_report`'s tree already reads `"   0    1    1    1"` on its own
    (one WARN, one FAIL, one SKIP); adding one critical episode-level metric
    on top of it should move the FAIL count to 2 and leave the rest alone.
    """

    report = _sample_report()
    [episode] = report.episodes
    episode = episode.model_copy(
        update={"metrics": {"stream_consistency": _metric(0.5, MetricStatus.CRITICAL)}}
    )
    report = report.model_copy(update={"episodes": [episode]})

    text = render_terminal(report)

    row = next(line for line in text.splitlines() if "arm_multi_device.csv" in line)
    assert "   0    1    2    1" in row


def test_every_metric_status_lands_in_exactly_one_bucket():
    """Verify every MetricStatus member is mapped, so a new one fails here first."""

    assert set(_METRIC_BUCKETS) == set(MetricStatus)
    assert set(_METRIC_BUCKETS.values()) == set(_BUCKET_LABELS)


def test_a_report_only_metric_counts_as_a_skip():
    """Verify a report_only metric lands in the SKIP bucket, next to not_applicable."""

    episode = GradedEpisode(
        id="episode",
        adapter="csv",
        adapter_confidence=0.9,
        score=_score(Level.EPISODE, None),
        metrics={"drop_rate": _metric(0.1, MetricStatus.REPORT_ONLY)},
    )

    counts = _metric_counts(episode)

    assert counts == (0, 0, 0, 1)


def test_the_bucket_labels_appear_once_in_the_header_and_never_in_a_row():
    """Verify PASS/SKIP name a bucket once, in the header, not repeated per row.

    FAIL and WARN are excluded from this check because `_severity_cell` also
    prints them in the FINDINGS section, for the same meaning.
    """

    text = render_terminal(_sample_report())

    assert text.count("PASS") == 1
    assert text.count("SKIP") == 1

    row = next(line for line in text.splitlines() if "arm_multi_device.csv" in line)
    metrics_tokens = row.split()[-len(_BUCKET_LABELS) :]
    assert all(token.isdigit() for token in metrics_tokens)


def test_a_run_with_warnings_reads_differently_from_one_without():
    """Verify a WARNING metric moves the WARN count and changes the rendered row."""

    warning_report = _sample_report()
    [episode] = warning_report.episodes
    [stream] = episode.streams
    good_stream = stream.model_copy(
        update={"metrics": {"dt_jitter_ms": _metric(3.5, MetricStatus.GOOD, unit="ms")}}
    )
    good_episode = episode.model_copy(update={"streams": [good_stream]})
    good_report = warning_report.model_copy(update={"episodes": [good_episode]})

    warning_text = render_terminal(warning_report)
    good_text = render_terminal(good_report)

    warning_row = next(
        line for line in warning_text.splitlines() if "arm_multi_device.csv" in line
    )
    good_row = next(
        line for line in good_text.splitlines() if "arm_multi_device.csv" in line
    )

    assert warning_row != good_row
    assert "   0    1    1    1" in warning_row
    assert "   1    0    1    1" in good_row


def test_an_episode_where_nothing_ran_does_not_read_as_four_zeroes():
    """Verify an episode with no metrics at all reads as prose, not zero counts."""

    episode = GradedEpisode(
        id="untouched",
        adapter="csv",
        adapter_confidence=0.9,
        score=_score(Level.EPISODE, None),
    )
    report = Report(
        root=UPath("fixtures"), score=_score(Level.DATASET, None), episodes=[episode]
    )

    text = render_terminal(report)

    row = next(line for line in text.splitlines() if "untouched" in line)
    assert "no metrics ran" in row
    assert not any(char.isdigit() for char in row)


def test_the_overall_row_sums_the_bucket_counts_of_every_episode():
    """Verify OVERALL carries the element-wise sum of every episode's bucket counts."""

    base = _sample_report()
    [episode] = base.episodes
    second = episode.model_copy(
        update={
            "id": "second_episode",
            "source_paths": [UPath("second_episode.csv")],
            "metrics": {"stream_consistency": _metric(0.5, MetricStatus.WARNING)},
        }
    )
    report = base.model_copy(update={"episodes": [episode, second]})

    text = render_terminal(report)

    overall_row = next(line for line in text.splitlines() if "OVERALL" in line)
    assert "   0    3    2    2" in overall_row


def test_a_five_digit_bucket_count_keeps_every_column_aligned():
    """Verify a bucket count past 9999 widens every column's field, header included."""

    metrics = {f"metric_{i}": _metric(1.0, MetricStatus.GOOD) for i in range(10_000)}
    episode = GradedEpisode(
        id="huge",
        adapter="csv",
        adapter_confidence=0.9,
        score=_score(Level.EPISODE, 100.0),
        metrics=metrics,
    )
    report = Report(
        root=UPath("fixtures"), score=_score(Level.DATASET, 100.0), episodes=[episode]
    )

    text = render_terminal(report, width=200)
    lines = text.splitlines()

    for line in lines:
        assert len(line) <= 200

    header_line = next(line for line in lines if "PASS" in line)
    row_line = next(line for line in lines if "huge" in line)

    header_ends = [header_line.rindex(label) + len(label) for label in _BUCKET_LABELS]
    numbers = row_line.split()[-len(_BUCKET_LABELS) :]
    row_ends = []
    cursor = 0
    for number in numbers:
        idx = row_line.index(number, cursor)
        row_ends.append(idx + len(number))
        cursor = idx + len(number)

    assert header_ends == row_ends


def test_the_html_report_carries_none_of_the_cards_bucket_words():
    """Verify the HTML report's metric vocabulary never collides with the card's.

    The HTML exposes metric status only as a raw enum value in a class and a
    `data-status` attribute, so there is nothing here for the two surfaces to
    disagree about. This test is what a future visible pill label would have
    to touch and reconcile.
    """

    html = render_html(_sample_report())

    for label in _BUCKET_LABELS:
        assert label not in html


@pytest.mark.parametrize("width", [80, 90, 120, 200])
def test_the_card_fits_the_width_it_is_given(width):
    """Verify no rendered line exceeds the width the card was laid out against."""

    text = render_terminal(_sample_report(), width=width)

    for line in text.splitlines():
        assert len(line) <= width


def _container_report(
    *members: tuple[str, float | None], root: UPath | None = None
) -> Report:
    """Build a Report whose episodes carry the ids and scores given.

    Parameters
    ----------
    *members : tuple[str, float or None]
        One `(id, score)` pair per episode.
    root : UPath or None
        The report's root, `fixtures` unless a test needs another.

    Returns
    -------
    Report
        A dataset-level report, each episode carrying one source path
        under `root` and no streams.
    """

    root = root if root is not None else UPath("fixtures")
    episodes = [
        GradedEpisode(
            id=episode_id,
            adapter="csv",
            adapter_confidence=0.9,
            source_paths=[root / "data" / "file-000.parquet"],
            score=_score(Level.EPISODE, score),
        )
        for episode_id, score in members
    ]
    return Report(root=root, score=_score(Level.DATASET, 100.0), episodes=episodes)


def test_episodes_from_one_container_render_under_one_header_row():
    """Verify two episodes of one container render under one header naming it."""

    report = _container_report(
        ("lerobot_v3_tiny::episode_000000", 40.0),
        ("lerobot_v3_tiny::episode_000001", 95.0),
    )

    text = render_terminal(report)
    recordings = text.split("RECORDINGS", 1)[1].split("FINDINGS", 1)[0]
    lines = [line for line in recordings.splitlines() if line.strip()]

    # An exact match proves the header's GRADE, SCORE and METRICS cells are
    # blank: any of them carrying real content would leave stray text beside
    # the container name.
    next(line for line in lines if line.strip() == "lerobot_v3_tiny")

    for episode_number in ("episode_000000", "episode_000001"):
        row = next(line for line in lines if episode_number in line)
        assert "lerobot_v3_tiny" not in row


def test_two_episodes_read_from_one_file_are_distinguishable_at_the_default_width():
    """Verify two episodes sharing a container and a source file render distinct rows.

    The two episodes name the same container and the same one file, so their
    episode id is the only thing that can tell their rows apart.
    """

    report = _container_report(
        ("hdf5_tiny.hdf5::runs_session_a", 40.0),
        ("hdf5_tiny.hdf5::runs_session_b", 95.0),
    )

    text = render_terminal(report)
    recordings = text.split("RECORDINGS", 1)[1].split("FINDINGS", 1)[0]
    lines = [line for line in recordings.splitlines() if line.strip()]

    row_a = next(line for line in lines if "runs_session_a" in line)
    row_b = next(line for line in lines if "runs_session_b" in line)
    assert row_a != row_b
    assert sum("runs_session_a" in line for line in lines) == 1
    assert sum("runs_session_b" in line for line in lines) == 1


def test_a_grouped_episode_renders_without_an_ellipsis_at_120_columns():
    """Verify a grouped episode's own row carries no ellipsis at 120 columns.

    A child's label is its bare episode id, so its row never carries the
    parquet-plus-mp4 source paths its header already covers.
    """

    root = UPath("fixtures")
    long_paths = [
        root / "lerobot_v3_tiny" / "data" / "chunk-000" / "file-000.parquet",
        root
        / "lerobot_v3_tiny"
        / "videos"
        / "chunk-000"
        / "observation.images.up"
        / "file-000.mp4",
    ]
    episodes = [
        GradedEpisode(
            id=f"lerobot_v3_tiny::episode_00000{i}",
            adapter="lerobot_v3",
            adapter_confidence=0.9,
            source_paths=long_paths,
            score=_score(Level.EPISODE, score),
        )
        for i, score in enumerate((40.0, 95.0))
    ]
    report = Report(root=root, score=_score(Level.DATASET, 100.0), episodes=episodes)

    text = render_terminal(report, width=120)
    recordings = text.split("RECORDINGS", 1)[1].split("FINDINGS", 1)[0]

    assert "…" not in recordings


def test_an_episode_that_is_its_own_container_renders_as_a_single_row():
    """Verify an unqualified episode id renders as one row, not a header plus one."""

    report = _container_report(("arm_multi_device", 82.0))
    [episode] = report.episodes

    text = render_terminal(report)
    rows = [line for line in text.splitlines() if "file-000.parquet" in line]

    assert len(rows) == 1
    assert episode.score.grade is not None
    assert episode.score.grade.value in rows[0]


def test_a_container_sorts_at_its_worst_episode_not_after_the_ungrouped_rows():
    """Verify a container's header takes the position of its worst episode.

    A container holding a 40.0 and a 95.0 episode is worse than a lone 90.0
    episode, so its header must sort above the ungrouped row, not below it.
    """

    report = _container_report(
        ("solo_run", 90.0),
        ("multi_run::a", 40.0),
        ("multi_run::b", 95.0),
    )

    text = render_terminal(report)
    recordings = text.split("RECORDINGS", 1)[1].split("FINDINGS", 1)[0]
    lines = [line for line in recordings.splitlines() if line.strip()]

    overall_index = next(i for i, line in enumerate(lines) if "OVERALL" in line)
    header_index = next(
        i for i, line in enumerate(lines) if line.strip() == "multi_run"
    )
    solo_index = next(i for i, line in enumerate(lines) if "90.0" in line)
    a_index = next(i for i, line in enumerate(lines) if "40.0" in line)
    b_index = next(i for i, line in enumerate(lines) if "95.0" in line)

    assert overall_index < header_index < solo_index
    assert a_index < b_index


def test_the_row_cap_counts_episodes_and_headers_cost_no_budget():
    """Verify the cap counts episodes, not printed lines, and a header costs nothing.

    A container whose second episode is cut by the cap still prints its header,
    because grouping follows the container's real size rather than how many of
    its children survived.
    """

    report = _container_report(
        ("cA::e1", 1.0),
        ("cA::e2", 2.0),
        ("cA::e3", 3.0),
        ("cC::e1", 4.0),
        ("cC::e2", 200.0),
        ("solo1", 5.0),
        ("solo2", 6.0),
        ("solo3", 7.0),
        ("solo4", 8.0),
        ("solo5", 9.0),
    )

    text = render_terminal(report)
    recordings = text.split("RECORDINGS", 1)[1].split("FINDINGS", 1)[0]
    lines = [line for line in recordings.splitlines() if line.strip()]

    score_rows = [
        line
        for line in lines
        if any(
            f"{score:.1f}" in line for score in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        )
    ]
    assert len(score_rows) == _MAX_EPISODE_ROWS

    remaining_row = next(line for line in lines if "more" in line)
    assert "+2 more" in remaining_row

    assert any(line.strip() == "cC" for line in lines)
    assert not any("200.0" in line for line in lines)


def test_a_grouped_episode_with_more_than_one_file_names_its_count():
    """Verify a child row states its file count, once the header absorbs its paths.

    A grouped episode's row shows only its bare id, so an episode drawing on
    several files would otherwise lose that fact entirely.
    """

    root = UPath("fixtures")
    episodes = [
        GradedEpisode(
            id="lerobot_v3_tiny::episode_000000",
            adapter="lerobot_v3",
            adapter_confidence=0.9,
            source_paths=[
                root / "data" / "file-000.parquet",
                root / "videos" / "up.mp4",
            ],
            score=_score(Level.EPISODE, 40.0),
        ),
        GradedEpisode(
            id="lerobot_v3_tiny::episode_000001",
            adapter="lerobot_v3",
            adapter_confidence=0.9,
            source_paths=[root / "data" / "file-000.parquet"],
            score=_score(Level.EPISODE, 95.0),
        ),
    ]
    report = Report(root=root, score=_score(Level.DATASET, 100.0), episodes=episodes)

    text = render_terminal(report)

    two_file_row = next(line for line in text.splitlines() if "episode_000000" in line)
    one_file_row = next(line for line in text.splitlines() if "episode_000001" in line)
    assert "(2 files)" in two_file_row
    assert "files)" not in one_file_row


def test_the_grade_column_survives_long_content_at_a_narrow_width():
    """Verify GRADE never disappears when RECORDING's content overflows a narrow width.

    Sizing RECORDING to its own measured content, rather than as a ratio
    column, risks Rich's last-resort column collapse once total content
    exceeds the console width — a step that ignores every column's fixed
    width and `no_wrap` and can crush GRADE to nothing.
    """

    root = UPath("fixtures")
    long_paths = [
        root / "lerobot_v2_0_tiny" / "data" / "chunk-000" / "episode_000000.parquet",
        root
        / "lerobot_v2_0_tiny"
        / "videos"
        / "chunk-000"
        / "observation.images.up"
        / "episode_000000.mp4",
    ]
    episode = GradedEpisode(
        id="solo",
        adapter="lerobot_v2",
        adapter_confidence=0.9,
        source_paths=long_paths,
        score=_score(Level.EPISODE, 82.0),
    )
    report = Report(root=root, score=_score(Level.DATASET, 82.0), episodes=[episode])
    assert report.score.grade is not None

    for width in (80, 90, 120):
        text = render_terminal(report, width=width)
        for line in text.splitlines():
            assert len(line) <= width
        overall_row = next(line for line in text.splitlines() if "OVERALL" in line)
        assert overall_row.split()[0] == report.score.grade.value


def test_the_rail_reads_without_colour_and_tracks_the_score():
    """Verify the score rail fills by score, not by row order or colour alone."""

    base = _sample_report()
    [episode] = base.episodes
    episodes = [
        episode.model_copy(
            update={
                "id": f"clip_{score}",
                "source_paths": [UPath(f"clip_{score}.mp4")],
                "score": _score(Level.EPISODE, score),
            }
        )
        for score in (0.0, 50.0, 100.0)
    ]
    report = base.model_copy(update={"episodes": episodes})

    text = render_terminal(report, color=False)

    counts = [
        next(line for line in text.splitlines() if f"clip_{score}.mp4" in line).count(
            "█"
        )
        for score in (0.0, 50.0, 100.0)
    ]

    assert counts[0] == 0
    assert counts[2] == _RAIL_CELLS
    assert counts[0] < counts[1] < counts[2]


def test_the_bar_sits_beside_its_score_under_one_header():
    """Verify the bar and its figure share the SCORE column, not a separate RAIL one."""

    base = _sample_report()
    [episode] = base.episodes
    episode = episode.model_copy(update={"score": _score(Level.EPISODE, 100.0)})
    report = base.model_copy(
        update={"episodes": [episode], "score": _score(Level.DATASET, 100.0)}
    )

    text = render_terminal(report, color=False)

    assert "RAIL" not in text
    assert "SCORE" in text
    row = next(line for line in text.splitlines() if "arm_multi_device.csv" in line)
    assert "█" * _RAIL_CELLS + " 100.0" in row


@pytest.mark.parametrize(
    ("width", "expect_bar"), [(_RAIL_MIN_WIDTH, True), (_RAIL_MIN_WIDTH - 1, False)]
)
def test_a_narrow_card_drops_the_bar_and_keeps_the_score(width, expect_bar):
    """Verify a card too narrow for the bar still keeps the score figure."""

    text = render_terminal(_sample_report(), width=width)

    row = next(line for line in text.splitlines() if "arm_multi_device.csv" in line)
    assert "50.0" in row
    assert ("█" in row) is expect_bar


def test_a_long_not_analysed_list_truncates_with_a_more_tail():
    """Verify a source list past `_MAX_SOURCE_ROWS` renders a "+N more" tail."""

    skipped = [
        SkippedSource(path=UPath(f"clip_{n}.mp4"), reason=SkipReason.NO_ADAPTER)
        for n in range(9)
    ]
    report = _sample_report().model_copy(update={"skipped": skipped})

    text = render_terminal(report)

    assert "clip_0.mp4" in text
    assert "clip_8.mp4" not in text
    assert "+4 more" in text


def test_the_html_page_fetches_nothing_from_the_network():
    """Verify the page carries no external reference, so it opens from disk alone."""

    html = render_html(_sample_report())

    for marker in ("http://", "https://", "<link", "@import", " src="):
        assert marker not in html


def test_an_unattributed_stream_renders_its_attribution_everywhere():
    """Verify Attribution.UNATTRIBUTED reaches the HTML attribute, label, and JSON."""

    channel = _graded_channel("tcp_pose_x_mm", 50.0)
    stream = GradedStream(
        taxonomy_type="unmapped.tcp_pose",
        instance=None,
        attribution=Attribution.UNATTRIBUTED,
        score=_score(Level.STREAM, 50.0),
        channels=[channel],
    )
    episode = GradedEpisode(
        id="arm_multi_device",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[UPath("arm_multi_device.csv")],
        score=_score(Level.EPISODE, 50.0),
        streams=[stream],
    )
    report = Report(
        root=UPath("fixtures"), score=_score(Level.DATASET, 50.0), episodes=[episode]
    )

    html = render_html(report)
    assert 'data-attribution="unattributed"' in html
    assert "&middot; unattributed" in html

    assert '"unattributed"' in render_json(report)


def test_a_stream_carrying_a_finding_starts_expanded_and_a_clean_one_stays_shut():
    """Verify a stream a finding names opens by default, and a clean one stays shut."""

    flagged = _graded_channel("tcp_pose_z_mm", 50.0)
    clean = _graded_channel("tcp_pose_y_mm", 100.0)
    flagged_stream = GradedStream(
        taxonomy_type="unmapped.tcp_pose",
        instance="armC",
        score=_score(Level.STREAM, 50.0),
        channels=[flagged],
    )
    clean_stream = GradedStream(
        taxonomy_type="unmapped.tcp_pose",
        instance="armD",
        score=_score(Level.STREAM, 100.0),
        channels=[clean],
    )
    episode = GradedEpisode(
        id="arm_multi_device",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[UPath("arm_multi_device.csv")],
        score=_score(Level.EPISODE, 75.0),
        streams=[flagged_stream, clean_stream],
    )
    report = Report(
        root=UPath("fixtures"),
        score=_score(Level.DATASET, 75.0),
        episodes=[episode],
        findings=[
            Finding(
                metric_id="timing.drop_rate",
                family="timing",
                severity=Severity.CRITICAL,
                value=0.081,
                unit="fraction",
                points=0.0,
                episode_id="arm_multi_device",
                stream="unmapped.tcp_pose",
                instance="armC",
                channel="tcp_pose_z_mm",
            )
        ],
    )

    collector = DisclosureStateCollector()
    collector.feed(render_html(report))

    states = {(name, instance): open_ for _, name, instance, open_ in collector.rows}
    assert states[("unmapped.tcp_pose", "armC")] is True
    assert states[("unmapped.tcp_pose", "armD")] is False
    [episode_row] = [row for row in collector.rows if row[0] == "episode"]
    assert episode_row[3] is True


def test_an_episode_with_no_findings_starts_shut():
    """Verify a report with no findings opens with the whole tree collapsed."""

    report = _sample_report().model_copy(update={"findings": []})

    collector = DisclosureStateCollector()
    collector.feed(render_html(report))

    assert collector.rows
    assert all(open_ is False for _, _, _, open_ in collector.rows)


def test_every_finding_renders_with_its_severity_metric_and_location():
    """Verify a finding's severity, metric id and location all reach the findings table.

    Scoped to the findings table itself, not the whole page — the tree below
    it renders some of the same substrings (an instance tag, a channel name)
    for reasons that have nothing to do with findings.

    An empty findings list falls back to the page's own empty-state copy,
    never a bare table with nothing in it.
    """

    html = render_html(_sample_report())
    start = html.index('<table class="findings">')
    table = html[start : html.index("</table>", start)]

    for marker in (
        "timing.drop_rate",
        "timing.dt_jitter_ms",
        "critical",
        "warning",
        "tcp_pose_z_mm",
        "armC",
    ):
        assert marker in table

    empty = render_html(_sample_report().model_copy(update={"findings": []}))
    assert 'class="empty"' in empty
    assert 'class="findings"' not in empty


def test_a_stream_less_finding_omits_the_stream_separator_in_the_html_table():
    """Verify the findings table drops the stream segment, not just its label.

    A finding above stream level names no stream, so its location cell must
    carry no `&middot;` at all — not the episode id followed by an empty separator.
    """

    base = _sample_report()
    episode_level = base.findings[0].model_copy(
        update={"stream": None, "instance": None, "channel": None}
    )
    report = base.model_copy(update={"findings": [episode_level]})
    html = render_html(report)
    start = html.index('<table class="findings">')
    table = html[start : html.index("</table>", start)]

    assert "arm_multi_device" in table
    assert "&middot;" not in table


def test_findings_beyond_the_preview_go_behind_a_disclosure():
    """Verify only the worst findings show up front, the rest behind a disclosure."""

    base = _sample_report().findings[0]
    findings = [
        base.model_copy(update={"metric_id": f"timing.metric_{i}"}) for i in range(10)
    ]
    report = _sample_report().model_copy(update={"findings": findings})

    html = render_html(report)
    preview = html[: html.index("2 more findings")]

    assert preview.count('<tr class="sev-') == 8
    assert "timing.metric_9" not in preview
    assert "timing.metric_9" in html


def test_a_level_with_no_score_renders_no_rail():
    """Verify a level with no score renders 'not graded' rather than an empty rail."""

    episode = GradedEpisode(
        id="capture_index",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[UPath("capture_index.json")],
        score=_score(Level.EPISODE, None),
        streams=[],
    )
    report = Report(
        root=UPath("fixtures"),
        score=_score(Level.DATASET, None),
        episodes=[episode],
    )

    html = render_html(report)

    assert "not graded" in html
    assert 'class="rail"' not in html
