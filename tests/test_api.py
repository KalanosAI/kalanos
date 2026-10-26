"""Pins the public library surface and `kalanos.grade`'s behaviour on its edges.

An embedder imports from `kalanos` and nowhere else,
so the names exported there are the contract.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from importlib import resources
from pathlib import Path

# External
import pytest
import yaml

# Internal
import kalanos
from kalanos.analysis.models.provenance import ExecutionTier, content_digest
from kalanos.analysis.models.report import CURRENT_SCHEMA_VERSION
from kalanos.assets.policy import load_default_policy, load_policy

# Local
from helpers import FIXTURES_DIR


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


PUBLIC_SURFACE = {
    "grade",
    "compare",
    "load_report",
    "load_policy",
    "load_dictionary",
    "Policy",
    "Dictionary",
    "Category",
    "CategoryGroup",
    "Report",
    "GradedEpisode",
    "GradedStream",
    "GradedChannel",
    "ScoreResult",
    "Grade",
    "Level",
    "MetricResult",
    "Finding",
    "Severity",
    "SkippedSource",
    "SkipReason",
    "UnresolvedSource",
    "KalanosError",
    "NothingToGrade",
    "SourceUnavailable",
    "AdapterTie",
    "SourceInfo",
    "SourceLimits",
    "SourceTooLarge",
    "DatasetInfo",
    "MappingOverride",
    "MappingOverrideError",
    "MappingSource",
    "OverrideOrigin",
    "Bundle",
    "ExecutionTier",
}


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture
def heavier_integrity_policy(tmp_path) -> Path:
    """A full policy file, the default with integrity's family weight raised to 7."""

    payload = _default_policy_payload()
    payload["family_weights"]["integrity"] = 7
    return _write_policy(tmp_path / "policy.yaml", payload)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _default_policy_payload() -> dict:
    """The packaged default policy as a dict, to edit into a full policy."""

    text = (
        resources.files("kalanos.assets")
        .joinpath("policies", "default.yaml")
        .read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def _write_policy(path: Path, payload: dict) -> Path:
    """Write `payload` as a policy file at `path` and return the path."""

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_public_surface_is_exactly_what_is_pinned():
    """Removing or renaming a public name must fail here, not in an embedder's code."""

    assert set(kalanos.__all__) == PUBLIC_SURFACE
    for name in kalanos.__all__:
        assert getattr(kalanos, name) is not None


def test_grade_returns_a_report_for_a_fixture_dataset():
    """Verify grading a fixture dataset through the library returns a Report."""

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert isinstance(report, kalanos.Report)
    assert report.episodes


def test_grade_accepts_a_plain_string():
    """Verify a string path is parsed as a UPath rather than rejected."""

    report = kalanos.grade(str(FIXTURES_DIR / "lerobot_v3_tiny"))

    assert report.episodes


def test_grade_refuses_a_missing_path(tmp_path):
    """Verify a missing path raises SourceUnavailable before any configuration loads."""

    with pytest.raises(kalanos.SourceUnavailable, match="does not exist"):
        kalanos.grade(tmp_path / "missing")


def test_grade_refuses_a_sample_count_below_one(tmp_path):
    """Verify vision_samples=0 is refused before the path is even looked at."""

    with pytest.raises(ValueError, match="vision_samples"):
        kalanos.grade(tmp_path / "missing", vision_samples=0)


def test_grade_refuses_an_empty_folder(tmp_path):
    """Verify a folder holding nothing at all raises NothingToGrade."""

    with pytest.raises(kalanos.NothingToGrade, match="nothing to grade"):
        kalanos.grade(tmp_path)


def test_every_public_error_is_a_kalanos_error():
    """Verify one `except KalanosError` catches every error raised on purpose."""

    for error in (
        kalanos.NothingToGrade,
        kalanos.SourceUnavailable,
        kalanos.AdapterTie,
    ):
        assert issubclass(error, kalanos.KalanosError)


def test_a_graded_report_records_its_source():
    """Verify the Report says what it graded: a local root and its file count."""

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert report.source is not None
    assert report.source.protocol == "file"
    assert report.source.file_count > 0
    assert report.schema_version == CURRENT_SCHEMA_VERSION


def test_the_full_tier_reads_every_camera_frame():
    """Verify `--tier full` implies a full frame scan whatever the sample count."""

    report = kalanos.grade(
        str(FIXTURES_DIR / "lerobot_v3_tiny"),
        tier=ExecutionTier.FULL,
        vision_samples=2,
    )

    for episode in report.episodes:
        camera = next(s for s in episode.streams if s.kind == "video")
        evidence = camera.metrics["sharpness_score"].evidence
        assert evidence["n_sampled"] == evidence["n_frames"]


def test_a_default_grade_records_the_default_family_weights():
    """Verify the report carries the family weights the run graded with."""

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert report.family_weights == {
        "integrity": 3,
        "timing": 2,
        "motion": 2,
        "vision": 2,
        "annotation": 1,
    }


def test_a_7_0_0_report_without_family_weights_still_loads():
    """Verify the field is additive: an older report validates with no weights."""

    payload = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny").model_dump(mode="json")
    del payload["family_weights"]
    payload["schema_version"] = "7.0.0"

    assert kalanos.Report.model_validate(payload).family_weights == {}


def test_a_duplicate_index_outside_the_episodes_is_refused():
    """Verify `duplicates` can only name episodes the report holds."""

    payload = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny").model_dump(mode="json")
    payload["duplicates"] = {"exact": [[0, len(payload["episodes"])]]}

    with pytest.raises(ValueError, match="outside episodes"):
        kalanos.Report.model_validate(payload)


def test_a_graded_report_rolls_up_each_metric_and_round_trips_them():
    """Verify the dataset and episode scores carry per-metric rollups that reload."""

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert report.score.metrics
    assert all(e.score.metrics for e in report.episodes)
    reloaded = kalanos.Report.model_validate(report.model_dump(mode="json"))
    assert reloaded.score.metrics == report.score.metrics
    assert [e.score.metrics for e in reloaded.episodes] == [
        e.score.metrics for e in report.episodes
    ]


def test_a_7_0_0_report_without_family_tails_still_loads():
    """Verify `family_tails` is additive: a 7.0.0 report validates without it."""

    payload = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny").model_dump(mode="json")
    del payload["score"]["family_tails"]
    for episode in payload["episodes"]:
        del episode["score"]["family_tails"]
    payload["schema_version"] = "7.0.0"

    assert kalanos.Report.model_validate(payload).score.family_tails == {}


def test_a_7_0_0_report_without_metric_rollups_still_loads():
    """Verify the per-metric fields are additive: a 7.0.0 report loads without them."""

    payload = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny").model_dump(mode="json")
    del payload["score"]["metrics"]
    for episode in payload["episodes"]:
        del episode["score"]["metrics"]
        for result in episode["metrics"].values():
            del result["points"]
        for stream in episode["streams"]:
            for result in stream["metrics"].values():
                del result["points"]
            for channel in stream["channels"]:
                for result in channel["metrics"].values():
                    del result["points"]
    payload["schema_version"] = "7.0.0"

    report = kalanos.Report.model_validate(payload)

    assert report.score.metrics == {}
    assert all(e.score.metrics == {} for e in report.episodes)
    assert all(
        result.points is None
        for e in report.episodes
        for s in e.streams
        for result in s.metrics.values()
    )


def test_a_policy_passed_to_grade_overrides_the_default_weights(
    heavier_integrity_policy,
):
    """Verify `policy=` grades with, and records, the policy's own family weights."""

    policy = load_policy(heavier_integrity_policy)
    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny", policy=policy)

    assert policy.family_weights["integrity"] == 7
    assert report.family_weights["integrity"] == 7


def test_a_policy_from_the_environment_overrides_the_default_weights(
    heavier_integrity_policy, monkeypatch
):
    """Verify `KALANOS_POLICY_PATH` grades with, and records, its own family weights."""

    monkeypatch.setenv("KALANOS_POLICY_PATH", str(heavier_integrity_policy))

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert report.family_weights["integrity"] == 7


def test_a_packaged_policy_from_the_environment_is_named_in_the_report(monkeypatch):
    """Verify the report names a packaged policy from `KALANOS_POLICY_PATH`."""

    monkeypatch.setenv("KALANOS_POLICY_PATH", "language_conditioned")

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert report.run is not None and report.run.policy is not None
    assert report.run.policy.id == "language_conditioned"
    assert report.run.policy.origin == "builtin"
    assert report.scope is not None
    assert report.scope.policy_id == "language_conditioned"
    for episode in report.episodes:
        assert episode.eligibility is not None
        assert episode.eligibility.policy_id == "language_conditioned"


def test_a_policy_file_from_the_environment_is_named_by_its_path(tmp_path, monkeypatch):
    """Verify a policy file from the environment is named by its stem and path."""

    path = tmp_path / "strict.yaml"
    path.write_text("extends: default\n")
    monkeypatch.setenv("KALANOS_POLICY_PATH", str(path))

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert report.run is not None and report.run.policy is not None
    assert report.run.policy.origin == str(path)
    assert report.run.policy.id == "strict"
    assert report.scope is not None
    assert report.scope.policy_id == "strict"


def test_without_a_policy_in_the_environment_the_report_names_the_default():
    """Verify a run with no policy override names the built-in default."""

    report = kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny")

    assert report.run is not None and report.run.policy is not None
    assert report.run.policy.id == "default-decisions-v1"
    assert report.run.policy.origin == "builtin"
    assert report.run.policy.digest == content_digest(
        load_default_policy().model_dump(mode="json")
    )
    assert report.scope is not None
    assert report.scope.policy_id == "default-decisions-v1"


def test_an_explicit_policy_wins_over_the_environment(monkeypatch):
    """Verify `policy=` keeps its own identity when `KALANOS_POLICY_PATH` is set."""

    monkeypatch.setenv("KALANOS_POLICY_PATH", "language_conditioned")

    report = kalanos.grade(
        FIXTURES_DIR / "lerobot_v3_tiny", policy=load_default_policy()
    )

    assert report.run is not None and report.run.policy is not None
    assert report.run.policy.id == "explicit-policy"
    assert report.run.policy.origin == "argument"


def test_grading_fails_naming_a_metric_left_without_a_weight(tmp_path):
    """Verify a graded metric without a weight beside weighted ones fails by name."""

    flatline = _default_policy_payload()["metrics"]["integrity.flatline_pct"]
    policy = _write_policy(
        tmp_path / "policy.yaml",
        {
            "extends": "default",
            "metrics": {"integrity.flatline_pct": {**flatline, "weight": None}},
        },
    )

    with pytest.raises(ValueError, match="flatline_pct"):
        kalanos.grade(FIXTURES_DIR / "lerobot_v3_tiny", policy=load_policy(policy))
