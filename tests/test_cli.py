"""Verifies the CLI's shape and `grade`'s wiring through the whole pipeline.

`grade` is the first subcommand and not the only one. Typer collapses a
one-command app into a bare `kalanos <args>`, which would silently rename
every documented invocation the moment a second command lands, so the group
form is pinned here rather than left to a default.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
from pathlib import Path

# External
import yaml
from typer.testing import CliRunner

# Internal
from kalanos.analysis.models.report import Report
from kalanos.cli import app

# Local
from helpers import CSV_FIXTURE, FIXTURES_DIR, ScoreAttributeCollector, score_attr


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


runner = CliRunner()


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _write_collision_csv(path: Path) -> None:
    """Write a CSV that resolves a time axis but that `CsvAdapter` must refuse.

    `t_ms` is a clean, regularly sampled time column, so it wins the time-axis
    score outright. The file also carries an unrelated column already named
    `time_s` — the canonical name the adapter renames the time axis to.
    `time_s` is deliberately non-monotonic here, so it cannot tie or beat
    `t_ms` on the same score and flip which column gets picked. Normalising
    `t_ms` into `time_s` would silently overwrite that column's real data,
    so `episodes` raises `AdapterRefusal` instead of reading the file; see
    `kalanos.analysis.adapters.tabular._normalise_time`.

    Parameters
    ----------
    path : Path
        Where to write the file.
    """

    rows = [["t_ms", "time_s", "val"]]
    rows += [[str(i * 10), str(100 - (i % 7)), str(i * 1.5)] for i in range(30)]
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_help_lists_grade_as_a_subcommand():
    """Verify `kalanos --help` presents a command group containing grade."""

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "COMMAND [ARGS]" in result.output
    assert "grade" in result.output


def test_grade_is_invoked_as_a_subcommand_not_as_the_bare_command():
    """Verify `kalanos <path>` is not accepted: the path belongs after `grade`."""

    result = runner.invoke(app, [str(CSV_FIXTURE)])

    assert result.exit_code != 0


def test_grade_refuses_a_path_that_does_not_exist():
    """Verify a nonexistent path is rejected before any analysis is attempted."""

    result = runner.invoke(app, ["grade", "/nonexistent/recording.csv"])

    assert result.exit_code != 0
    assert result.stdout == ""
    assert result.stderr != ""


def test_grade_a_file_prints_a_card_and_exits_zero():
    """Verify grading a single file runs the whole pipeline and reports it."""

    result = runner.invoke(app, ["grade", str(CSV_FIXTURE)])

    assert result.exit_code == 0
    assert "OVERALL" in result.stdout


def test_grade_a_folder_prints_a_card_and_exits_zero():
    """Verify grading a folder goes through the same code path as a single file."""

    result = runner.invoke(app, ["grade", str(FIXTURES_DIR)])

    assert result.exit_code == 0
    assert "OVERALL" in result.stdout
    # README.md and the fixtures with no CSV pathology are real, explained
    # refusals — the card names each one rather than dropping it silently.
    assert "NOT ANALYSED" in result.stdout


def test_grade_refuses_a_folder_with_nothing_to_grade(tmp_path):
    """Verify an empty folder is refused rather than reported as a clean run.

    A folder where every file was skipped with a reason is still a real
    report; an empty folder has no reason to give, so it is unusable input
    instead.
    """

    result = runner.invoke(app, ["grade", str(tmp_path)])

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "nothing to grade" in result.stderr


def test_json_flag_prints_the_model_with_no_card():
    """Verify --json opts out of the card and onto the parseable Report."""

    result = runner.invoke(app, ["grade", str(CSV_FIXTURE), "--json"])

    assert result.exit_code == 0
    assert "OVERALL" not in result.stdout
    report = Report.model_validate_json(result.stdout)
    assert report.episodes


def test_report_json_writes_a_file_that_parses_back_into_the_model(tmp_path):
    """Verify --report x.json writes the same model render_json produces."""

    destination = tmp_path / "report.json"

    result = runner.invoke(
        app, ["grade", str(CSV_FIXTURE), "--report", str(destination)]
    )

    assert result.exit_code == 0
    report = Report.model_validate_json(destination.read_text())
    assert report.episodes


def test_report_yaml_writes_a_file_that_parses_back_into_the_model(tmp_path):
    """Verify --report x.yaml writes a file that parses back into the model."""

    destination = tmp_path / "report.yaml"

    result = runner.invoke(
        app, ["grade", str(CSV_FIXTURE), "--report", str(destination)]
    )

    assert result.exit_code == 0
    report = Report.model_validate(yaml.safe_load(destination.read_text()))
    assert report.episodes


def test_report_html_writes_a_rendered_page(tmp_path):
    """Verify --report x.html writes a page carrying the report's own path."""

    destination = tmp_path / "report.html"

    result = runner.invoke(
        app, ["grade", str(CSV_FIXTURE), "--report", str(destination)]
    )

    assert result.exit_code == 0
    html = destination.read_text()
    assert str(CSV_FIXTURE) in html


def test_html_and_json_reports_carry_the_same_scores(tmp_path):
    """Verify the HTML and JSON renders of one run agree, node for node.

    'HTML is a read-only view of the model and can never disagree with the
    score' is the acceptance criterion. `--json` and `--report` are
    independent flags, so one invocation produces both outputs from the
    same `Report` object — this walks both and checks every level rather
    than inspecting one file by eye.
    """

    html_path = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["grade", str(FIXTURES_DIR), "--json", "--report", str(html_path)],
    )

    assert result.exit_code == 0
    report = Report.model_validate_json(result.stdout)
    collector = ScoreAttributeCollector()
    collector.feed(html_path.read_text())

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


def test_report_with_an_unsupported_suffix_exits_nonzero_and_writes_nothing(
    tmp_path,
):
    """Verify a bad --report suffix fails before stdout carries anything.

    The reason belongs on stderr and stdout must stay empty — a partial
    card ahead of a write failure would be worse than no output at all.
    """

    destination = tmp_path / "report.txt"

    result = runner.invoke(
        app, ["grade", str(CSV_FIXTURE), "--report", str(destination)]
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "report.txt" in result.stderr
    assert not destination.exists()


def test_report_to_an_unwritable_path_exits_nonzero_with_the_os_error_on_stderr(
    tmp_path,
):
    """Verify a filesystem failure on --report fails as cleanly as a bad suffix.

    `write_report` ends in `Path.write_text`, so a missing parent directory
    raises `OSError` rather than `write_report`'s own `ValueError` — that
    has to fail the same way, not surface as an unhandled traceback.
    """

    destination = tmp_path / "does-not-exist" / "report.json"

    result = runner.invoke(
        app, ["grade", str(CSV_FIXTURE), "--report", str(destination)]
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert result.stderr != ""


def test_a_source_loading_refuses_is_unresolved_rather_than_aborting_the_run(
    tmp_path,
):
    """Verify one pathological file does not stop the rest of a folder from grading.

    The adapter raises `AdapterRefusal` on a schema it resolved but still
    can't carry forward — see `_write_collision_csv`. The pipeline catches
    that per source, so a folder with one such file still grades everything
    else in it.
    """

    good = tmp_path / "arm_multi_device.csv"
    good.write_bytes(CSV_FIXTURE.read_bytes())
    _write_collision_csv(tmp_path / "collides.csv")

    result = runner.invoke(app, ["grade", str(tmp_path), "--json"])

    assert result.exit_code == 0
    report = Report.model_validate_json(result.stdout)
    analysed = [
        path.name for episode in report.episodes for path in episode.source_paths
    ]
    assert analysed == ["arm_multi_device.csv"]
    assert len(report.unresolved) == 1
    assert report.unresolved[0].path.name == "collides.csv"
    assert "time_s" in report.unresolved[0].reason


def test_a_source_that_fails_to_parse_is_unresolved_rather_than_aborting_the_run(
    tmp_path,
):
    """Verify a file polars itself refuses to read does not stop the run either.

    `CsvAdapter.parse` reads the file with polars before there is a schema
    to raise anything else against, so a malformed CSV — a ragged row
    count, here — surfaces as `polars.exceptions.ComputeError`, which
    `TabularAdapter.episodes` folds into `AdapterRefusal` the same way it
    does `_write_collision_csv`'s case. That must be caught, or the whole
    folder aborts on the one file polars itself cannot parse.
    """

    good = tmp_path / "arm_multi_device.csv"
    good.write_bytes(CSV_FIXTURE.read_bytes())
    (tmp_path / "ragged.csv").write_text("a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n")

    result = runner.invoke(app, ["grade", str(tmp_path), "--json"])

    assert result.exit_code == 0
    report = Report.model_validate_json(result.stdout)
    analysed = [
        path.name for episode in report.episodes for path in episode.source_paths
    ]
    assert analysed == ["arm_multi_device.csv"]
    assert len(report.unresolved) == 1
    assert report.unresolved[0].path.name == "ragged.csv"


def test_kalanos_reports_dir_redirects_a_relative_report_path(monkeypatch, tmp_path):
    """Verify KALANOS_REPORTS_DIR resolves a relative --report against it."""

    monkeypatch.setenv("KALANOS_REPORTS_DIR", str(tmp_path))

    result = runner.invoke(app, ["grade", str(CSV_FIXTURE), "--report", "report.json"])

    assert result.exit_code == 0
    assert (tmp_path / "report.json").exists()


def test_a_second_subcommand_can_be_added_without_touching_grade():
    """Verify registering another command leaves `grade`'s own behaviour unchanged.

    Demonstrates the acceptance criterion directly: a trivial command is
    registered on `app` for the duration of this test alone, so nothing
    about the group's shape or `grade`'s wiring had to change to add it.
    """

    @app.command()
    def ping() -> None:
        """Answer, to prove a second command needs no changes to grade."""

        print("pong")

    try:
        ping_result = runner.invoke(app, ["ping"])
        grade_result = runner.invoke(app, ["grade", str(CSV_FIXTURE)])

        assert ping_result.exit_code == 0
        assert ping_result.stdout.strip() == "pong"
        assert grade_result.exit_code == 0
        assert "OVERALL" in grade_result.stdout
    finally:
        app.registered_commands = [
            command
            for command in app.registered_commands
            if command.callback is not ping
        ]


def test_adapters_lists_the_built_in_csv_adapter():
    """Verify `kalanos adapters` names a built-in and where it came from."""

    result = runner.invoke(app, ["adapters"])

    assert result.exit_code == 0
    assert "csv" in result.stdout
    assert "kalanos" in result.stdout


def test_metrics_lists_a_built_in_metric():
    """Verify `kalanos metrics` names a registered metric."""

    result = runner.invoke(app, ["metrics"])

    assert result.exit_code == 0
    assert "drop_rate" in result.stdout


def test_reporters_lists_a_built_in_renderer():
    """Verify `kalanos reporters` names a registered renderer."""

    result = runner.invoke(app, ["reporters"])

    assert result.exit_code == 0
    assert "html" in result.stdout


def test_plugins_summarises_all_three_groups():
    """Verify `kalanos plugins` covers adapters, metrics and reporters at once."""

    result = runner.invoke(app, ["plugins"])

    assert result.exit_code == 0
    for group in ("adapters", "metrics", "reporters"):
        assert group in result.stdout


def test_metrics_family_filters_to_that_family_alone():
    """Verify --family keeps its own family's metrics and drops every other one."""

    timing = runner.invoke(app, ["metrics", "--family", "timing"])
    vision = runner.invoke(app, ["metrics", "--family", "vision"])

    assert timing.exit_code == 0
    assert "drop_rate" in timing.stdout
    assert vision.exit_code == 0
    assert "drop_rate" not in vision.stdout


def test_new_adapter_writes_a_package(tmp_path):
    """Verify `kalanos new adapter` writes a package rooted at the name it was given."""

    result = runner.invoke(app, ["new", "adapter", "myformat", "--into", str(tmp_path)])

    assert result.exit_code == 0
    assert (tmp_path / "kalanos-myformat" / "pyproject.toml").is_file()


def test_new_adapter_twice_exits_two_without_overwriting(tmp_path):
    """Verify a second run refuses rather than replacing what is already there."""

    runner.invoke(app, ["new", "adapter", "myformat", "--into", str(tmp_path)])
    written = tmp_path / "kalanos-myformat" / "src" / "kalanos_myformat" / "__init__.py"
    written.write_text("# mine\n")

    result = runner.invoke(app, ["new", "adapter", "myformat", "--into", str(tmp_path)])

    assert result.exit_code == 2
    assert written.read_text() == "# mine\n"
