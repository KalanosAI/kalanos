"""Verifies per-run mapping overrides from an argument, a file or the sidecar."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import shutil
from pathlib import Path

# External
import pytest
import yaml

# Internal
import kalanos
from kalanos import MappingOverrideError, MappingSource, OverrideOrigin
from kalanos.analysis.models.metrics import MetricStatus
from kalanos.analysis.models.report import GradedStream, Report
from kalanos.assets.mapping import SIDECAR_NAME, parse_map_argument

# Local
from helpers import CSV_FIXTURE, LEROBOT_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

STATE = "observation.state"
UNMAPPED_STATE = f"unmapped.{STATE}"
JOINT_POSITION = "proprio.joint_position"

# The prefix of the reason a metric gives when a stream's type does not qualify.
_TAXONOMY_GATE = "needs one of"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _streams_typed(report: Report, taxonomy_type: str) -> list[GradedStream]:
    """Return every stream in the report typed as `taxonomy_type`."""

    return [
        stream
        for episode in report.episodes
        for stream in episode.streams
        if stream.taxonomy_type == taxonomy_type
    ]


def _write_mapping(path: Path, features: dict[str, str]) -> Path:
    """Write a mapping file in the on-disk shape and return its path."""

    path.write_text(yaml.safe_dump({"schema_version": 1, "features": features}))
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A private copy of the LeRobot fixture, so a sidecar can be written into it."""

    return Path(shutil.copytree(str(LEROBOT_FIXTURE), tmp_path / "dataset"))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_an_argument_override_types_the_state_stream_as_joint_position():
    """Verify `mapping=` retypes observation.state in every episode, and it grades."""

    report = kalanos.grade(LEROBOT_FIXTURE, mapping={STATE: JOINT_POSITION})

    states = _streams_typed(report, JOINT_POSITION)
    assert len(states) == len(report.episodes)
    assert _streams_typed(report, UNMAPPED_STATE) == []
    for stream in states:
        assert stream.mapping_source is MappingSource.OVERRIDE

    [override] = report.mapping_overrides
    assert (override.feature, override.taxonomy_type) == (STATE, JOINT_POSITION)
    assert override.origin is OverrideOrigin.ARGUMENT
    assert override.path is None

    # The four joint-position metrics now run on the stream's own data.
    # The tiny fixture is too short and too busy for still_drift to find a settled
    # tail, so that one is checked only for having got past the taxonomy gate.
    [first, *_] = states
    for name in ("max_abs_jerk", "mean_jerk_norm"):
        assert first.metrics[name].status is not MetricStatus.NOT_APPLICABLE
    for channel in first.channels:
        status = channel.metrics["limit_proximity_pct"].status
        assert status is not MetricStatus.NOT_APPLICABLE
    for stream in states:
        reason = stream.metrics["still_drift"].evidence.get("reason", "")
        assert not reason.startswith(_TAXONOMY_GATE)


def test_an_override_naming_a_missing_feature_fails_and_lists_what_was_seen():
    with pytest.raises(MappingOverrideError) as excinfo:
        kalanos.grade(LEROBOT_FIXTURE, mapping={"observation.stat": JOINT_POSITION})

    message = str(excinfo.value)
    assert "'observation.stat' (from argument)" in message
    assert STATE in message


def test_an_override_naming_an_unknown_taxonomy_type_fails():
    with pytest.raises(MappingOverrideError, match="proprio.no_such_type"):
        kalanos.grade(LEROBOT_FIXTURE, mapping={STATE: "proprio.no_such_type"})


def test_a_mapping_file_beats_the_sidecar_and_an_argument_beats_both(dataset, tmp_path):
    """Verify precedence argument > file > sidecar, and the winner's origin is kept."""

    _write_mapping(dataset / SIDECAR_NAME, {STATE: "proprio.joint_velocity"})
    mapping_file = _write_mapping(tmp_path / "map.yaml", {STATE: JOINT_POSITION})

    from_file = kalanos.grade(dataset, mapping_file=mapping_file)

    [override] = from_file.mapping_overrides
    assert override.taxonomy_type == JOINT_POSITION
    assert override.origin is OverrideOrigin.FILE
    assert override.path is not None
    assert Path(str(override.path)) == mapping_file

    from_argument = kalanos.grade(
        dataset,
        mapping_file=mapping_file,
        mapping={STATE: "proprio.joint_torque"},
    )

    [override] = from_argument.mapping_overrides
    assert override.taxonomy_type == "proprio.joint_torque"
    assert override.origin is OverrideOrigin.ARGUMENT
    torque = _streams_typed(from_argument, "proprio.joint_torque")
    assert len(torque) == len(from_argument.episodes)


def test_a_sidecar_applies_on_its_own_and_sidecar_false_ignores_it(dataset):
    _write_mapping(dataset / SIDECAR_NAME, {STATE: JOINT_POSITION})

    applied = kalanos.grade(dataset)
    ignored = kalanos.grade(dataset, sidecar=False)

    [override] = applied.mapping_overrides
    assert override.origin is OverrideOrigin.SIDECAR
    assert override.path is not None
    assert Path(str(override.path)) == dataset / SIDECAR_NAME
    assert ignored.mapping_overrides == []
    assert len(_streams_typed(ignored, UNMAPPED_STATE)) == len(ignored.episodes)


@pytest.mark.parametrize(
    "text",
    [
        "features: [unclosed\n",
        "schema_version: 1\nfeatures: {}\nextra: true\n",
        f"schema_version: 2\nfeatures:\n  {STATE}: {JOINT_POSITION}\n",
    ],
    ids=["not-yaml", "extra-key", "wrong-schema-version"],
)
def test_a_malformed_mapping_file_fails_naming_its_path(tmp_path, text):
    mapping_file = tmp_path / "map.yaml"
    mapping_file.write_text(text)

    with pytest.raises(MappingOverrideError, match="map.yaml"):
        kalanos.grade(LEROBOT_FIXTURE, mapping_file=mapping_file)


def test_a_missing_mapping_file_fails_naming_its_path(tmp_path):
    with pytest.raises(MappingOverrideError, match="absent.yaml"):
        kalanos.grade(LEROBOT_FIXTURE, mapping_file=tmp_path / "absent.yaml")


def test_an_override_retypes_a_csv_stream_by_its_grouped_stem():
    """Verify the override works for any adapter: `tcp_pose` groups `tcp_pose_*_mm`."""

    report = kalanos.grade(CSV_FIXTURE, mapping={"tcp_pose": JOINT_POSITION})

    streams = [s for e in report.episodes for s in e.streams]
    assert streams
    for stream in streams:
        assert stream.taxonomy_type == JOINT_POSITION
        assert stream.mapping_source is MappingSource.OVERRIDE


def test_a_map_argument_splits_on_its_last_equals_sign():
    assert parse_map_argument("a=b=proprio.joint_position") == (
        "a=b",
        "proprio.joint_position",
    )
    assert parse_map_argument(" observation.state = proprio.joint_position ") == (
        STATE,
        JOINT_POSITION,
    )


@pytest.mark.parametrize("text", ["nonsense", "=proprio.joint_position", "state="])
def test_a_map_argument_with_an_empty_side_is_refused(text):
    with pytest.raises(MappingOverrideError):
        parse_map_argument(text)
