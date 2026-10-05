"""Verifies the terminal report card: its sections, caps, wording and two looks."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# External
import pytest
from rich.cells import cell_len
from upath import UPath

# Internal
from kalanos.analysis.models.coverage import Coverage, CoverageRow
from kalanos.analysis.models.eligibility import (
    Consequence,
    EligibilityCounts,
    EligibilityReason,
    EligibilityStatus,
    EpisodeEligibility,
    ReasonKind,
    Sufficiency,
    SufficiencyCheck,
    SufficiencyStatus,
)
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.card import render_terminal
from kalanos.analysis.reporting.card.sections import _under_root
from kalanos.api import grade

# Local
from helpers import CSV_FIXTURE, FIXTURES_DIR, LEROBOT_FIXTURE


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(scope="module")
def csv_report() -> Report:
    """The CSV fixture graded under the default scope: one passing episode."""

    return grade(CSV_FIXTURE)


@pytest.fixture(scope="module")
def vision_report() -> Report:
    """The LeRobot fixture graded under the vision scope: two episodes in review."""

    return grade(LEROBOT_FIXTURE, bundle=Path("vision-imitation-v1"))


@pytest.fixture(scope="module")
def corpus_report() -> Report:
    """The whole fixture folder, with a skipped and an unresolved file in it."""

    return grade(FIXTURES_DIR)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _section(text: str, heading: str) -> list[str]:
    """The lines under one section heading, up to the blank line that ends it."""

    lines = text.splitlines()
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if not lines[i]), len(lines))
    return lines[start:end]


def _row(text: str, label: str) -> str:
    """The value beside `label` in a two-column row, whatever the column padding."""

    for line in text.splitlines():
        parts = line.split(maxsplit=1)
        if parts and parts[0] == label:
            return parts[1] if len(parts) > 1 else ""
    raise AssertionError(f"no {label!r} row")


def _with_findings(report: Report, *where: tuple[str, str, Consequence]) -> Report:
    """A copy of `report` with one finding per `(episode, stream, consequence)`."""

    base = report.findings[0]
    findings = [
        base.model_copy(
            update={
                "id": f"f{i}",
                "episode_id": episode,
                "stream": stream,
                "instance": None,
                "channel": None,
                "consequence": consequence,
            }
        )
        for i, (episode, stream, consequence) in enumerate(where)
    ]
    return report.model_copy(update={"findings": findings})


def _flagged(
    report: Report, *episodes: tuple[str, EligibilityStatus, float | None]
) -> Report:
    """A copy of `report` with one low-sharpness episode per `(id, status, score)`."""

    base = report.episodes[0]
    flagged = [
        base.model_copy(
            update={
                "id": episode_id,
                "source_paths": [],
                "score": base.score.model_copy(update={"score": score}),
                "eligibility": EpisodeEligibility.model_construct(
                    status=status,
                    scope_id="s",
                    policy_id="p",
                    reasons=[
                        EligibilityReason(
                            id="cam.sharpness_score",
                            kind=ReasonKind.FINDING,
                            status=status,
                        )
                    ],
                ),
            }
        )
        for episode_id, status, score in episodes
    ]
    return report.model_copy(update={"episodes": flagged})


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_findings_name_the_source_and_condition_without_debug_tails(csv_report):
    text = render_terminal(csv_report)
    rows = _section(text, "Findings")

    armb = next(row for row in rows if "armB:proprio.ee_pose/tcp_pose_z_mm" in row)
    assert "repeated values" in armb
    assert "1 episode " in armb
    assert any("low signal-to-noise" in row for row in rows)
    for tail in ("adapter=", "consequence=", "support="):
        assert tail not in text


def test_findings_count_each_episode_once_and_split_by_consequence(csv_report):
    report = _with_findings(
        csv_report,
        ("ep1", "cam", Consequence.REVIEW),
        ("ep1", "cam", Consequence.REVIEW),
        ("ep2", "cam", Consequence.REVIEW),
        ("ep1", "cam", Consequence.BLOCK),
    )

    rows = _section(render_terminal(report), "Findings")

    assert len(rows) == 2
    assert "1 episode " in rows[0] and rows[0].endswith("block")
    assert "2 episodes" in rows[1] and rows[1].endswith("review")


def test_findings_cap_at_three_sources_and_never_total_them(csv_report):
    report = _with_findings(
        csv_report,
        *((f"ep{i}", f"stream{i}", Consequence.REPORT_ONLY) for i in range(5)),
    )

    rows = _section(render_terminal(report), "Findings")

    assert len(rows) == 4
    assert rows[-1].strip() == "+ 2 more sources in report"
    assert not any("5 episodes" in row for row in rows)


def test_coverage_shows_only_the_capabilities_that_apply(csv_report, vision_report):
    numeric = _section(render_terminal(csv_report), "Coverage")
    vision = _section(render_terminal(vision_report), "Coverage")

    assert any(row.split()[0] == "numeric" for row in numeric)
    assert not any("video" in row for row in numeric)
    video = next(row for row in vision if "video quality" in row)
    assert "frames examined" in video


def test_a_coverage_gap_reads_as_a_gap_and_never_as_zero(csv_report):
    gap = Coverage(
        capabilities=[
            CoverageRow(
                key="video_quality",
                unit="episode",
                computed=1,
                unavailable=1,
                eligible=2,
                reasons={"no decodable frames": 1},
            )
        ]
    )

    rows = _section(
        render_terminal(csv_report.model_copy(update={"coverage": gap})), "Coverage"
    )

    video = next(row for row in rows if "video quality" in row)
    assert "1/2" in video and "unavailable" in video
    assert "no decodable frames" in video
    assert " 0 " not in video


def test_needs_attention_is_absent_when_every_episode_passes(csv_report):
    assert "Needs attention" not in render_terminal(csv_report)


def test_needs_attention_lists_each_episode_in_review(vision_report):
    rows = _section(render_terminal(vision_report), "Needs attention")

    assert len(rows) == 2
    assert all("review" in row and "low sharpness" in row for row in rows)


def test_needs_attention_caps_at_three_rows(csv_report):
    review = EligibilityStatus.REVIEW
    report = _flagged(csv_report, *((f"ep{i}", review, 80.0) for i in range(5)))

    rows = _section(render_terminal(report), "Needs attention")

    assert len(rows) == 4
    assert rows[-1].strip() == "+ 2 more in report"


def test_needs_attention_orders_by_status_then_score_with_none_last(csv_report):
    report = _flagged(
        csv_report,
        ("review_low", EligibilityStatus.REVIEW, 40.0),
        ("unknown_none", EligibilityStatus.UNKNOWN, None),
        ("blocked", EligibilityStatus.BLOCKED, 90.0),
        ("unknown_low", EligibilityStatus.UNKNOWN, 10.0),
    )

    rows = _section(render_terminal(report), "Needs attention")

    assert [row.split()[0] for row in rows] == [
        "blocked",
        "unknown_low",
        "unknown_none",
        "+",
    ]


def test_an_episode_inside_a_container_is_named_by_its_id(csv_report):
    episode = csv_report.episodes[0]
    report = _flagged(csv_report, ("export::episode_7", EligibilityStatus.REVIEW, 1.0))
    report.episodes[0].source_paths = episode.source_paths

    rows = _section(render_terminal(report), "Needs attention")

    assert rows[0].split()[0] == "export/episode_7"


@pytest.mark.parametrize(
    ("stream", "instance", "channel", "source"),
    [
        ("proprio.joint_position", "armA", "j1", "armA:proprio.joint_position/j1"),
        ("proprio.joint_position", "armA", None, "armA:proprio.joint_position"),
        ("proprio.joint_position", None, "j1", "proprio.joint_position/j1"),
        (None, None, None, "episode"),
    ],
)
def test_a_finding_is_addressed_by_instance_stream_and_channel(
    csv_report, stream, instance, channel, source
):
    finding = csv_report.findings[0].model_copy(
        update={"stream": stream, "instance": instance, "channel": channel}
    )
    report = csv_report.model_copy(update={"findings": [finding]})

    rows = _section(render_terminal(report), "Findings")

    assert rows[0].split()[0] == source


@pytest.mark.parametrize(
    ("path", "root", "shown"),
    [
        ("data/sub/a.csv", "data", "sub/a.csv"),
        ("elsewhere/a.csv", "data", "elsewhere/a.csv"),
        ("data/a.csv", "data/a.csv", "a.csv"),
    ],
)
def test_a_path_is_shown_under_the_root(path, root, shown):
    assert _under_root(UPath(path), UPath(root)) == shown


@pytest.mark.parametrize(
    ("counts", "errors", "verdict"),
    [
        ((1, 1, 0, 0, 0, True), [{"error": "x"}], "Incomplete - operational errors"),
        (None, [], "Not graded"),
        ((1, 1, 0, 0, 0, False), [], "Incomplete - inventory not fully read"),
        ((3, 0, 1, 1, 1, True), [], "Blocked episodes found"),
        ((2, 0, 0, 1, 1, True), [], "Review required"),
        ((1, 0, 0, 0, 1, True), [], "Evidence missing"),
        ((1, 1, 0, 0, 0, True), [], "Ready"),
    ],
)
def test_the_assessment_is_the_first_verdict_that_holds(
    csv_report, counts, errors, verdict
):
    eligibility_counts = (
        None
        if counts is None
        else EligibilityCounts(
            total=counts[0],
            pass_count=counts[1],
            blocked=counts[2],
            review=counts[3],
            unknown=counts[4],
            inventory_complete=counts[5],
        )
    )
    report = csv_report.model_copy(
        update={"eligibility_counts": eligibility_counts, "operational_errors": errors}
    )

    assert _row(render_terminal(report), "Assessment") == verdict


def test_the_card_never_lists_what_could_not_be_observed(vision_report):
    text = render_terminal(vision_report)

    assert "Not observable" not in text
    assert "needs one of" not in text


def test_sufficiency_shows_only_when_it_is_known(csv_report):
    unknown = Sufficiency(status=SufficiencyStatus.UNKNOWN)
    insufficient = Sufficiency(
        status=SufficiencyStatus.INSUFFICIENT,
        checks=[
            SufficiencyCheck(
                requirement="min_pass_episodes",
                status=SufficiencyStatus.INSUFFICIENT,
                detail="1 pass episode, 10 required",
            )
        ],
    )

    hidden = render_terminal(csv_report.model_copy(update={"sufficiency": unknown}))
    shown = render_terminal(csv_report.model_copy(update={"sufficiency": insufficient}))

    assert "Sufficiency" not in hidden
    assert _row(shown, "Sufficiency") == "insufficient - 1 pass episode, 10 required"


def test_plain_output_is_ascii_and_color_is_styled(corpus_report):
    for text in (
        render_terminal(corpus_report),
        render_terminal(corpus_report, color=False),
    ):
        assert all(ord(c) < 128 for c in text)
        assert "\x1b[" not in text
    assert "\x1b[" in render_terminal(corpus_report, color=True)


@pytest.mark.parametrize("width", [40, 60, 80])
def test_no_line_exceeds_the_width(corpus_report, width):
    text = render_terminal(corpus_report, width=width)

    assert max(cell_len(line) for line in text.splitlines()) <= width


def test_not_analysed_names_skipped_and_unresolved_files(corpus_report):
    rows = _section(render_terminal(corpus_report), "Not analysed")

    assert any("README.md" in row and "no adapter" in row for row in rows)
    assert any("video_meta.json" in row and "no schema: " in row for row in rows)


def test_the_footer_points_at_the_report_only_when_one_was_written(csv_report):
    written = render_terminal(csv_report, report_path="x/report.json").splitlines()
    unwritten = render_terminal(csv_report).splitlines()

    assert any(
        line.startswith("  Report") and "x/report.json" in line for line in written
    )
    assert not any(line.startswith("  Report") for line in unwritten)


def test_unmapped_stream_types_get_a_mapping_row(csv_report, vision_report):
    assert not any(
        row.split()[0] == "mapping"
        for row in _section(render_terminal(csv_report), "Coverage")
    )
    assert _row(render_terminal(vision_report), "mapping").startswith(
        "2 stream types unmapped - type them with --map"
    )
