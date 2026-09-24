"""Round-trip and composition tests for the domain models."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path
from typing import Any

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.domain import (
    Attribution,
    Channel,
    Clock,
    Episode,
    FramePayload,
    Kind,
    Stream,
)
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.report import (
    CURRENT_SCHEMA_VERSION,
    GradedChannel,
    GradedEpisode,
    GradedStream,
    Report,
)
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult, Severity


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


class RefusingPayload:
    """A payload that fails the moment anything asks it for data.

    Stands in for the expensive case, a video nobody has decoded: every fetch
    raises, so a test can show that nothing reached for the data.
    """

    def __len__(self) -> int:
        """Report a sample count, which costs nothing to know.

        Returns
        -------
        int
            A fixed count; no data is read to produce it.
        """

        return 2

    def fetch(self) -> Any:
        """Refuse to hand over any data.

        Raises
        ------
        AssertionError
            Always.
        """

        raise AssertionError("the payload was fetched")


def _sample_report() -> Report:
    """Build a Report with a graded channel for the round-trip test.

    Returns
    -------
    Report
        A populated report, no field left at a placeholder value.
    """

    metric = MetricResult(
        value=0.02,
        unit="fraction",
        status=MetricStatus.GOOD,
        evidence={"dropped": 2, "expected": 100},
    )
    channel_score = ScoreResult(
        level=Level.CHANNEL,
        score=100.0,
        grade=Grade.A,
        train_ready=True,
        n_contributing=1,
    )
    channel = GradedChannel(
        channel=Channel(name="tcp_pose_x_mm"),
        score=channel_score,
        metrics={"drop_rate": metric},
    )
    episode = GradedEpisode(
        id="arm_multi_device",
        adapter="csv",
        adapter_confidence=0.9,
        source_paths=[UPath("/data/example/arm_multi_device.csv")],
        score=channel_score,
        streams=[
            GradedStream(
                taxonomy_type="unmapped.tcp_pose",
                instance="armA",
                score=channel_score,
                channels=[channel],
            )
        ],
    )
    return Report(root=UPath("/data/example"), score=channel_score, episodes=[episode])


def _stream(source_path: Path, **overrides: Any) -> Stream:
    """Build a minimal series Stream, overriding whatever a test cares about.

    Parameters
    ----------
    source_path : Path
        The file the stream claims to come from.
    **overrides : Any
        Field values replacing the defaults.

    Returns
    -------
    Stream
        A one-channel stream over two samples.
    """

    fields = {
        "taxonomy_type": "unmapped.tcp_pose",
        "kind": Kind.SERIES,
        "timestamps": pl.Series("time_s", [0.0, 0.1]),
        "payload": FramePayload(frame=pl.DataFrame({"tcp_pose_x_mm": [1.0, 2.0]})),
        "source_path": source_path,
        "source_field": "tcp_pose",
        "channels": [Channel(name="tcp_pose_x_mm")],
    }
    return Stream(**{**fields, **overrides})


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_report_round_trips_through_json():
    """Verify a Report serialises to JSON and parses back to an equal model."""

    report = _sample_report()
    restored = Report.model_validate_json(report.model_dump_json())
    assert restored == report


def test_report_schema_version_is_populated_not_a_placeholder():
    """Verify schema_version defaults to a real version, not an empty placeholder."""

    report = _sample_report()
    assert report.schema_version == CURRENT_SCHEMA_VERSION
    assert report.schema_version not in ("", "unset", "TODO", "TBD")


def test_metric_result_status_is_an_enum_over_the_five_statuses():
    """Verify MetricStatus is a closed enum over the five documented statuses."""

    assert {status.value for status in MetricStatus} == {
        "good",
        "warning",
        "critical",
        "report_only",
        "not_applicable",
    }


def test_metric_result_rejects_an_unknown_status():
    """Verify MetricResult refuses a status string outside the closed enum."""

    with pytest.raises(ValueError):
        MetricResult.model_validate(
            {"value": 1.0, "unit": "hz", "status": "excellent", "evidence": {}}
        )


def test_level_is_the_four_levels_the_rollup_composes_over():
    """Verify Level carries no file level and nothing below Channel."""

    values = {level.value for level in Level}
    assert values == {"channel", "stream", "episode", "dataset"}
    assert not values & {"source", "entity", "group"}


def test_severity_is_a_closed_enum_over_only_the_two_defect_verdicts():
    """Verify Severity has no member for good, report_only or not_applicable.

    A Finding exists to report a defect; a third member here would give a
    non-defect status somewhere to go.
    """

    assert {severity.value for severity in Severity} == {"critical", "warning"}


def test_finding_rejects_a_severity_outside_the_closed_enum():
    """Verify Finding refuses a severity string the enum does not carry."""

    with pytest.raises(ValueError):
        Finding.model_validate(
            {
                "metric_id": "timing.drop_rate",
                "family": "timing",
                "severity": "good",
                "value": 0.2,
                "unit": "fraction",
                "points": 0.0,
                "episode_id": "episode_0",
                "stream": "unmapped.tcp_pose",
            }
        )


def test_stream_with_no_instance_is_the_single_subject_case():
    """Verify a stream from a recording with one subject uses the same model."""

    stream = _stream(Path("/data/example/single_device.csv"))
    assert stream.instance is None
    assert stream.clock is Clock.UNKNOWN


def test_reading_timestamps_never_fetches_the_payload():
    """Verify timestamps are eager and independent of the payload behind them.

    A stream backed by an undecoded video has to be able to answer every timing
    question without paying for the decode.
    """

    stream = _stream(Path("/data/example/wrist_camera.mp4"), payload=RefusingPayload())
    payload = stream.payload
    assert payload is not None

    assert stream.timestamps.to_list() == [0.0, 0.1]
    assert len(payload) == 2


def test_stream_defaults_attribution_from_whether_instance_is_set():
    """Verify a Stream with an instance defaults to KEYED, and without to SINGLE."""

    keyed = _stream(Path("/data/example/keyed.csv"), instance="armA")
    single = _stream(Path("/data/example/single.csv"))
    assert keyed.attribution is Attribution.KEYED
    assert single.attribution is Attribution.SINGLE


def test_stream_rejects_an_attribution_that_contradicts_instance():
    """Verify KEYED with no instance, and SINGLE or UNATTRIBUTED with one, raise."""

    with pytest.raises(ValueError):
        _stream(
            Path("/data/example/bad.csv"),
            instance=None,
            attribution=Attribution.KEYED,
        )
    with pytest.raises(ValueError):
        _stream(
            Path("/data/example/bad.csv"),
            instance="armA",
            attribution=Attribution.SINGLE,
        )
    with pytest.raises(ValueError):
        _stream(
            Path("/data/example/bad.csv"),
            instance="armA",
            attribution=Attribution.UNATTRIBUTED,
        )


def test_channel_defaults_axis_to_none():
    """Verify a Channel with no axis given defaults to None."""

    assert Channel(name="tcp_pose_x_mm").axis is None


def test_episode_source_paths_derive_from_its_streams():
    """Verify an episode's files come from its streams, each listed once."""

    joints = Path("/data/example/joints.csv")
    camera = Path("/data/example/wrist.mp4")
    episode = Episode(
        id="episode_000",
        streams=[_stream(joints), _stream(camera), _stream(joints)],
    )
    assert episode.source_paths == [joints, camera]


def test_dataset_info_preserves_a_upath_that_is_not_a_pathlib_path():
    """Verify a memory:// path survives DatasetInfo without being coerced."""

    path = UPath("memory://data/arm.csv")
    info = DatasetInfo(adapter="x", path=path)

    assert info.path == path
    assert info.path.protocol == "memory"
    assert not isinstance(info.path, Path)


def test_dataset_info_round_trips_a_remote_path_through_json():
    """Verify a UPath's protocol survives model_dump_json -> model_validate_json."""

    path = UPath("memory://data/arm.csv")
    info = DatasetInfo(adapter="x", path=path)
    restored = DatasetInfo.model_validate_json(info.model_dump_json())

    assert restored.path == path
    assert restored.path.protocol == "memory"


def test_dataset_info_round_trips_a_plain_path_handed_in_as_a_upath():
    """Verify a bare pathlib.Path handed to `path` comes back a UPath."""

    # DatasetInfo(path=Path(...)) is what a caller would write, but its
    # declared type is UPath: model_validate takes the same value through
    # AnyPath's coercion without a static type mismatch at the call site.
    local = DatasetInfo.model_validate(
        {"adapter": "x", "path": Path("/data/example/arm.csv")}
    )
    restored_local = DatasetInfo.model_validate_json(local.model_dump_json())
    assert restored_local.path == UPath("/data/example/arm.csv")
