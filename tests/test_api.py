"""Pins the public library surface and `kalanos.grade`'s behaviour on its edges.

An embedder imports from `kalanos` and nowhere else,
so the names exported there are the contract.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import pytest

# Internal
import kalanos

# Local
from helpers import FIXTURES_DIR


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


PUBLIC_SURFACE = {
    "grade",
    "load_policy",
    "load_dictionary",
    "Policy",
    "Dictionary",
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
}


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
    assert report.schema_version == "6.0.0"
