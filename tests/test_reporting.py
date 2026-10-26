"""Verifies reporting: assembling the graded Report, and its four renderers."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import dataclasses
import json
import logging
import shutil

# External
import polars as pl
import pytest
import yaml
from calibration_helpers import grade_with_test_calibration
from upath import UPath

# Internal
from kalanos.analysis.adapters.csv import CsvAdapter
from kalanos.analysis.adapters.video import (
    ContainerHeader,
    DeclaredFormat,
    VideoPayload,
)
from kalanos.analysis.models.discovery import SkippedSource, SkipReason
from kalanos.analysis.models.domain import (
    Attribution,
    Channel,
    Episode,
    FramePayload,
    Kind,
    Stream,
    TimestampDtype,
)
from kalanos.analysis.models.metrics import (
    Family,
    Level,
    MetricResult,
    MetricStatus,
)
from kalanos.analysis.models.policy import Band, MetricPolicy, Policy
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import (
    CURRENT_SCHEMA_VERSION,
    AnalysedEpisode,
    DatasetWarning,
    GradedChannel,
    GradedEpisode,
    GradedStream,
    Report,
    VideoFormat,
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
from kalanos.analysis.reporting.families import FAMILY_LABELS, family_scores
from kalanos.analysis.reporting.omit import omit_empty_defaults
from kalanos.analysis.reporting.registry import registered_reporters
from kalanos.analysis.reporting.render import (
    render_compact_json,
    render_html,
    render_json,
    render_yaml,
)
from kalanos.analysis.reporting.write import write_report
from kalanos.analysis.scoring.gate import worse
from kalanos.analysis.scoring.score import (
    grade_for,
    rollup,
    score_episode,
    score_metrics,
)
from kalanos.api import grade
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.assets.policy import load_default_policy

# Local
from helpers import (
    FIXTURES_DIR,
    DisclosureStateCollector,
    ScoreAttributeCollector,
    decided,
    score_attr,
    write_arm,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FIXTURE = UPath(__file__).parent / "fixtures" / "arm_multi_device.csv"

# The letters table `_score`'s stand-in ScoreResults grade against —
# loaded once, since every test in this module shares the same default policy.
_POLICY = load_default_policy()

# What `assemble_report` looks each stream's category up in, for calls that bypass it.
_DICTIONARY = load_default_dictionary()


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
    episode = decided(
        GradedEpisode(
            id="arm_multi_device",
            adapter="csv",
            adapter_confidence=0.9,
            source_paths=[UPath("arm_multi_device.csv")],
            score=_score(Level.EPISODE, 50.0),
            streams=[stream],
        )
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
        timestamp_dtype=TimestampDtype.FLOAT64,
        channels=[Channel(name="x")],
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_report_carries_the_current_schema_version():
    """Verify a freshly assembled Report defaults to the version it was built under."""

    report = assemble_report(root=UPath("."), analysed=[], policy=load_default_policy())

    assert report.schema_version == CURRENT_SCHEMA_VERSION


def test_a_mapped_stream_carries_its_category_and_an_unmapped_one_carries_none():
    """Verify each graded stream is labelled with its dictionary category."""

    raw_episode, adapter = _analysed_fixture()
    unmapped = raw_episode.streams[1].model_copy(
        update={"taxonomy_type": "unmapped.mystery", "mapping_source": None}
    )
    episode = raw_episode.model_copy(
        update={"streams": [raw_episode.streams[0], unmapped]}
    )

    report = assemble_report(
        root=FIXTURE.parent,
        analysed=[
            AnalysedEpisode(
                episode=episode, adapter=adapter, adapter_confidence=0.9, policy=_POLICY
            )
        ],
        policy=_POLICY,
    )

    [graded] = report.episodes
    assert {s.taxonomy_type: s.category for s in graded.streams} == {
        "proprio.ee_pose": "proprioceptive_state",
        "unmapped.mystery": None,
    }


def test_a_report_maps_every_category_to_its_group():
    """Verify a reader can group stream categories from the report alone."""

    report = assemble_report(root=UPath("."), analysed=[], policy=_POLICY)

    assert report.categories == _DICTIONARY.category_groups


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
        video_like,
        policy=load_default_policy(),
        is_regular=True,
        episode_id=mapped.id,
        category=None,
    )

    assert graded.channels == []


def test_a_stream_with_no_payload_grades_its_own_metrics():
    """Verify payload=None no longer raises: the stream's own metrics still run."""

    mapped, _ = _analysed_fixture()
    empty = mapped.streams[0].model_copy(update={"payload": None, "channels": []})

    graded, _findings = grade_stream(
        empty,
        policy=load_default_policy(),
        is_regular=True,
        episode_id=mapped.id,
        category=None,
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
            category=None,
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
        episode,
        adapter="csv",
        adapter_confidence=0.9,
        policy=load_default_policy(),
        dictionary=_DICTIONARY,
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
        episode,
        adapter="csv",
        adapter_confidence=1.0,
        policy=load_default_policy(),
        dictionary=_DICTIONARY,
    )

    assert graded.duration_s == pytest.approx(1.5)
    assert graded.sample_count == 3


def test_grade_episode_score_is_unchanged_with_no_episode_metrics_registered(
    monkeypatch,
):
    """Verify an episode's score still equals the fold of its streams alone.

    Filters EPISODE-level entries out of a temporary registry copy, rather
    than relying on none being registered in the shipped catalogue — that
    would break, for no real reason, the moment a first episode metric
    lands. The invariant under test is that an empty `results` dict
    contributes nothing to the score, never a shifting zero.
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
        raw_episode,
        adapter=adapter,
        adapter_confidence=0.9,
        policy=policy,
        dictionary=_DICTIONARY,
    )

    assert graded.metrics == {}
    assert graded.score == score_episode(graded.streams, {}, policy=policy)


def test_grade_episode_folds_its_own_metrics_in_beside_its_stream_metrics(
    monkeypatch,
):
    """Verify an episode metric enters its family's fold beside the stream metrics.

    A stub episode metric graded to one end of the scale, contrasted
    against a stub stream metric graded to the other, both in one family
    with undecided weights, makes the two contributions distinguishable.
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
        timestamp_dtype=TimestampDtype.FLOAT64,
    )
    episode = Episode(id="episode_0", streams=[stream])

    graded, _findings = grade_episode(
        episode,
        adapter="csv",
        adapter_confidence=0.9,
        policy=policy,
        dictionary=_DICTIONARY,
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
        timestamp_dtype=TimestampDtype.FLOAT64,
    )
    episode = Episode(id="episode_0", streams=[stream])

    _graded, findings = grade_episode(
        episode,
        adapter="csv",
        adapter_confidence=0.9,
        policy=policy,
        dictionary=_DICTIONARY,
    )

    by_metric = {finding.metric_id: finding for finding in findings}
    episode_finding = by_metric["stub.stub_episode_metric"]
    stream_finding = by_metric["stub.stub_stream_metric"]
    assert episode_finding.episode_id == "episode_0"
    assert episode_finding.stream is None
    assert episode_finding.channel is None
    assert stream_finding.stream == "unmapped.tcp_pose"


def test_report_grades_integrity_without_unverified_acquisition_timing():
    """CSV timestamps have unknown origin. Recorded gaps stay descriptive;
    integrity findings still grade and retain their instance/channel addresses.
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

    assert arm_c > 80.0
    assert arm_a > 90.0
    assert arm_d > 90.0
    assert arm_b > 80.0
    assert arm_b < arm_a
    assert arm_b < arm_d

    [armc] = [stream for stream in episode.streams if stream.instance == "armC"]
    assert armc.channels, "armC's arm stream lost its channels somewhere in the walk"
    assert armc.metrics["drop_rate"].status == MetricStatus.NOT_APPLICABLE
    assert armc.metrics["recorded_drop_estimate"].status == MetricStatus.REPORT_ONLY
    recorded_drop_estimate = armc.metrics["recorded_drop_estimate"].value
    assert recorded_drop_estimate is not None
    assert recorded_drop_estimate > 0.05

    assert report.score.score == episode.score.score
    report_score = report.score.score
    assert report_score is not None
    # Integrity alone scores, each metric at its worst channel: flatline_pct
    # 50 and missing_pct 100 at weight 4, snr_db 7.24 at weight 2.
    assert report_score == pytest.approx(61.448, abs=1e-3)

    # Missing provenance must not turn a recorded-axis estimate into a
    # critical acquisition finding. The real channel finding keeps its address.
    by_severity = {
        (finding.metric_id, finding.instance, finding.channel): finding.severity
        for finding in report.findings
    }
    assert not any(key[0] == "timing.drop_rate" for key in by_severity)
    assert (
        by_severity[("integrity.flatline_pct", "armB", "tcp_pose_z_mm")]
        == Severity.WARNING
    )
    assert {finding.episode_id for finding in report.findings} == {episode.id}


def test_assemble_report_rolls_up_exactly_like_an_independent_rollup_of_its_children():
    """Verify every level's score equals its fold applied to its own children.

    The episode reconstructs through `score_episode` over its streams and own
    metrics; the stream and the dataset through `rollup`.

    A stream's score folds in its own metrics alongside its channels', so
    its independent reconstruction re-derives that contribution by calling
    `score_metrics` again over `stream.metrics` — deterministic, since
    interpolation depends only on the value each metric already carries.
    Pins the wiring itself — that `assemble_report` calls each fold with the
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
    # The number, families and contributors are the rollup's exactly; the gate
    # only caps the letter (and train-readiness) by the share of failing
    # episodes — here the fixture's one episode, whose critical findings fail it.
    independent = rollup(Level.DATASET, [episode.score], policy=policy)
    assert report.score.score == independent.score
    assert report.score.families == independent.families
    assert report.score.n_contributing == independent.n_contributing
    assert report.gate is not None
    assert report.gate.uncapped_grade == independent.grade
    assert report.score.grade == worse(independent.grade, report.gate.cap)
    # The episode's `train_ready` is the compatibility mirror of its
    # eligibility, filled in by assembly; an independent rollup leaves it
    # `None`, so compare everything else.
    independent_episode = score_episode(episode.streams, episode.metrics, policy=policy)
    assert (
        independent_episode.model_copy(
            update={"train_ready": episode.score.train_ready}
        )
        == episode.score
    )
    assert episode.eligibility is not None
    assert episode.score.train_ready == episode.eligibility.compatibility_train_ready
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


def test_a_graded_report_records_family_tails_on_the_dataset_score_only():
    """Verify the tails count every graded episode and leave every score unchanged.

    The fixture folder has one episode scoring integrity below 70 among those
    that scored it, so the count is above zero on real grading.
    """

    report = grade(FIXTURES_DIR)
    integrity = [
        e.score.families["integrity"]
        for e in report.episodes
        if "integrity" in e.score.families
    ]

    tail = report.score.family_tails["integrity"]

    assert tail.n_scored == len(integrity) == len(report.episodes)
    assert tail.n_below >= 1
    assert tail.lowest == min(integrity)
    assert all(e.score.family_tails == {} for e in report.episodes)
    independent = rollup(
        Level.DATASET,
        [e.score for e in report.episodes],
        policy=load_default_policy(),
    )
    assert report.score.families == independent.families


def test_blocked_episodes_count_toward_the_family_tails(tmp_path):
    """Verify a blocked episode counts toward its family's tail, as it does the mean.

    The four glitched episodes are blocked and are the only ones scoring
    integrity below 70, so leaving blocked episodes out would read 0 of 16.
    """

    path = tmp_path / "arm.hdf5"
    write_arm(path, 20, glitched={2, 7, 11, 16})

    report = grade_with_test_calibration(path)

    assert report.eligibility_counts is not None
    assert report.eligibility_counts.blocked == 4
    tail = report.score.family_tails["integrity"]
    assert tail.n_scored == 20
    assert tail.n_below == 4


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
        timestamp_dtype=TimestampDtype.FLOAT64,
        channels=[Channel(name="tcp_pose_x_mm")],
    )

    graded, _findings = grade_stream(
        stream, policy=policy, is_regular=True, episode_id="episode_0", category=None
    )

    assert graded.metrics["stub_stream_metric"].status == MetricStatus.GOOD
    [channel] = graded.channels
    assert channel.metrics["stub_channel_metric"].status == MetricStatus.CRITICAL
    assert graded.score.score == pytest.approx(50.0)


def test_assemble_report_regrades_metrics_rather_than_keeping_report_only():
    """Integrity metrics still grade when unknown-clock acquisition checks abstain."""

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
        for channel in stream.channels
        for metric in channel.metrics.values()
    }
    assert statuses & {MetricStatus.GOOD, MetricStatus.WARNING, MetricStatus.CRITICAL}


def test_report_round_trips_through_json():
    """Verify a Report survives a JSON render and re-parse unchanged."""

    report = _sample_report()

    assert Report.model_validate_json(render_json(report)) == report


def test_compact_json_keeps_evidence_and_omits_only_empty_defaults():
    """Compact JSON keeps evidence nulls, numeric precision and string content."""

    report = _sample_report()
    evidence = {
        "instruction": 'Grasp  the towel — 左手\nHold\t"still"',
        "samples": [None, 0, False, 0.12345678901234567, 1e-20],
        "empty": {},
    }
    report.findings[0].evidence = evidence
    before = report.model_dump(mode="json")
    pretty = report.model_dump_json(indent=2)

    compact = render_json(report)
    parsed = json.loads(compact)

    assert "\n" not in compact
    assert len(compact.encode("utf-8")) < len(pretty.encode("utf-8"))
    assert parsed == omit_empty_defaults(report, before)
    # Compared as text so False and 0, or a float's last digit, can't pass as equal.
    assert json.dumps(parsed["findings"][0]["evidence"]) == json.dumps(evidence)
    assert report.model_dump(mode="json") == before


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


def test_write_report_compact_writes_the_compact_render(tmp_path):
    """Verify compact=True writes `render_compact_json`'s text to a .json path."""

    report = _sample_report()
    path = write_report(report, tmp_path / "r.json", compact=True)

    assert path.read_text() == render_compact_json(report)


def test_write_report_compact_refuses_a_path_other_than_json(tmp_path):
    """Verify compact=True refuses a suffix that does not hold JSON, writing nothing."""

    with pytest.raises(ValueError, match="use a .json path"):
        write_report(_sample_report(), tmp_path / "r.html", compact=True)
    assert not (tmp_path / "r.html").exists()


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
    for family, _, score in family_scores(report.score.families):
        expected.append(("family", family, repr(score)))
    for episode in report.episodes:
        expected.append(("episode", episode.id, score_attr(episode.score)))
        for stream in episode.streams:
            expected.append(("stream", stream.taxonomy_type, score_attr(stream.score)))
            for graded in stream.channels:
                expected.append(
                    ("channel", graded.channel.name, score_attr(graded.score))
                )

    assert collector.rows == expected


def test_family_scores_order_named_families_first_and_fall_back_to_the_id():
    """Verify the named families lead in their own order, and any other shows its id."""

    scores = family_scores({"consistency": 80.0, "vision": 50.0, "timing": 90.0})

    assert scores == [
        ("timing", "Temporal", 90.0),
        ("vision", "Vision", 50.0),
        ("consistency", "consistency", 80.0),
    ]


def test_every_family_label_names_a_known_family():
    """Verify a display name never hangs off a family id the engine cannot score."""

    assert set(FAMILY_LABELS) <= {family.value for family in Family}


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
    """Verify YAML, JSON and HTML all report the same scores."""

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
    episode = decided(
        GradedEpisode(
            id="arm_multi_device",
            adapter="csv",
            adapter_confidence=0.9,
            source_paths=[UPath("arm_multi_device.csv")],
            score=_score(Level.EPISODE, 50.0),
            streams=[stream],
        )
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
    episode = decided(
        GradedEpisode(
            id="arm_multi_device",
            adapter="csv",
            adapter_confidence=0.9,
            source_paths=[UPath("arm_multi_device.csv")],
            score=_score(Level.EPISODE, 75.0),
            streams=[flagged_stream, clean_stream],
        )
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


def test_html_shows_dataset_warnings():
    """A dataset warning renders once, in its own section,
    and a report with none renders no such section."""

    warning = DatasetWarning(
        code="near_duplicates_saturated", message="5 of 5 episodes are alike."
    )
    report = _sample_report().model_copy(update={"dataset_warnings": [warning]})

    html = render_html(report)

    column = html[html.index('<div class="column">') :]
    assert column.count('data-warning="near_duplicates_saturated"') == 1
    assert (
        '<p data-warning="near_duplicates_saturated">5 of 5 episodes are alike.</p>'
        in column
    )
    assert "Dataset warnings" not in render_html(_sample_report())


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
    assert (
        "timing.metric_9"
        not in preview.split('<section class="findings-section">', 1)[1]
    )
    assert "timing.metric_9" in html


def test_a_level_with_no_score_renders_no_rail():
    """Verify a level with no score renders 'not graded' rather than an empty rail."""

    episode = decided(
        GradedEpisode(
            id="capture_index",
            adapter="csv",
            adapter_confidence=0.9,
            source_paths=[UPath("capture_index.json")],
            score=_score(Level.EPISODE, None),
            streams=[],
        )
    )
    report = Report(
        root=UPath("fixtures"),
        score=_score(Level.DATASET, None),
        episodes=[episode],
    )

    html = render_html(report)

    assert "not graded" in html
    assert 'class="rail"' not in html


def _source_stream(report: Report, source_field: str) -> GradedStream:
    """Return the first episode's stream read from `source_field`."""

    return next(s for s in report.episodes[0].streams if s.source_field == source_field)


def test_a_camera_stream_reports_its_declared_and_container_format():
    """Verify a camera carries both formats even where they agree, and they reload."""

    pytest.importorskip("av")
    report = grade(FIXTURES_DIR / "lerobot_v2_0_tiny")

    reloaded = Report.model_validate_json(report.model_dump_json())
    for graded in (report, reloaded):
        camera = _source_stream(graded, "observation.images.up")
        assert camera.video_format is not None
        declared, found = camera.video_format.declared, camera.video_format.found
        assert declared.fps == 30.0
        assert declared.width is None and declared.height is None
        assert (found.width, found.height, found.fps) == (32, 32, 30.0)
        assert (found.codec, found.profile) == ("h264", "High")
        assert (found.bit_rate, found.frame_count) == (26910, 8)
        assert found.duration_s == pytest.approx(4096 / 15360)
        # The fixture's colour codes are all unspecified.
        assert found.color_range is None and found.color_primaries is None
        assert found.color_transfer is None and found.color_space is None
        assert found.encoder == "Lavf62.12.102"
        assert found.camera_make is None and found.camera_model is None
        assert declared.codec is None and declared.channels is None
        assert declared.is_depth_map is None and declared.has_audio is None
    assert _source_stream(report, "observation.state").video_format is None


def test_the_metadata_tier_reports_no_container_format(monkeypatch):
    """Verify the metadata tier states the declared format and opens no container."""

    monkeypatch.setattr(
        VideoPayload,
        "header",
        lambda self: pytest.fail("metadata tier read the header"),
    )
    report = grade(FIXTURES_DIR / "lerobot_v2_0_tiny", tier=ExecutionTier.METADATA)

    camera = _source_stream(report, "observation.images.up")
    assert camera.video_format is not None
    assert camera.video_format.declared.fps == 30.0
    found = camera.video_format.found.model_dump()
    assert found and all(value is None for value in found.values())


def test_every_adapter_format_field_reaches_the_report():
    """Verify `VideoFormat` has a field for each the dataclasses carry, by name.

    `camera_format` builds it from their fields by keyword,
    and Pydantic ignores a keyword it has no field for.
    """

    fields = set(VideoFormat.model_fields)
    assert {f.name for f in dataclasses.fields(DeclaredFormat)} <= fields
    assert {f.name for f in dataclasses.fields(ContainerHeader)} <= fields


def _copy_v2_fixture(tmp_path) -> UPath:
    """Copy `lerobot_v2_0_tiny` to `tmp_path`, so a test may change it."""

    root = tmp_path / "dataset"
    shutil.copytree(str(FIXTURES_DIR / "lerobot_v2_0_tiny"), root)
    return UPath(root)


def test_a_camera_stream_reports_what_its_feature_declares(tmp_path):
    """Verify the codec, channels and depth-map and audio flags `info` declares."""

    root = _copy_v2_fixture(tmp_path)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.up"]["info"].update(
        {
            "video.codec": "av1",
            "video.channels": 3,
            "video.is_depth_map": False,
            "has_audio": False,
        }
    )
    info_path.write_text(json.dumps(info))

    report = grade(root, tier=ExecutionTier.METADATA)

    camera = _source_stream(report, "observation.images.up")
    assert camera.video_format is not None
    declared = camera.video_format.declared
    assert (declared.codec, declared.channels) == ("av1", 3)
    assert declared.is_depth_map is False and declared.has_audio is False


def test_a_camera_stream_reports_its_tags_and_colour_but_no_location(tmp_path):
    """Verify make, model and colour reach the report, and a location tag does not."""

    av = pytest.importorskip("av")
    numpy = pytest.importorskip("numpy")
    root = _copy_v2_fixture(tmp_path)
    location = "+48.8577+002.2950+035.000/"
    path = root / "videos/chunk-000/observation.images.up/episode_000000.mp4"
    container = av.open(str(path), "w", options={"movflags": "use_metadata_tags"})
    container.metadata.update(
        {
            "com.apple.quicktime.make": "Acme",
            "com.apple.quicktime.model": "Cam One",
            "com.apple.quicktime.location.ISO6709": location,
            "location": location,
        }
    )
    stream = container.add_stream("libx264", rate=30)
    stream.width = stream.height = 32
    stream.pix_fmt = "yuv420p"
    codec = stream.codec_context
    codec.color_range = codec.color_primaries = codec.color_trc = 1
    codec.colorspace = 1
    for _ in range(8):
        pixels = numpy.full((32, 32, 3), 128, dtype=numpy.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    with av.open(str(path)) as written:
        assert location in written.metadata.values()

    report = grade(root)

    camera = _source_stream(report, "observation.images.up")
    assert camera.video_format is not None
    found = camera.video_format.found
    assert (found.camera_make, found.camera_model) == ("Acme", "Cam One")
    assert (
        found.color_range,
        found.color_primaries,
        found.color_transfer,
        found.color_space,
    ) == ("tv", "bt709", "bt709", "bt709")
    assert location not in report.model_dump_json()


def test_every_episode_in_a_shared_file_reports_the_whole_file():
    """Verify both episodes of one v3 mp4 report the file's frame count."""

    pytest.importorskip("av")
    report = grade(FIXTURES_DIR / "lerobot_v3_tiny")

    counts = [
        stream.video_format.found.frame_count
        for episode in report.episodes
        for stream in episode.streams
        if stream.video_format is not None
    ]
    assert counts == [14, 14]


def test_a_graded_episode_carries_its_split():
    """Verify the graded report keeps each episode's split, and reloads it."""

    report = grade(FIXTURES_DIR / "lerobot_v2_0_tiny")

    reloaded = Report.model_validate_json(report.model_dump_json())
    assert [e.split for e in reloaded.episodes] == ["train", "train"]
