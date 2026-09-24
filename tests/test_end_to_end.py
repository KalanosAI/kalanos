"""Runs `kalanos grade` against the tracked fixture corpus and checks the report."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import os
from collections.abc import Iterator
from pathlib import Path

# External
import pytest
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.common import CONFIDENCE
from kalanos.analysis.models.discovery import SkipReason
from kalanos.analysis.models.metrics import MetricResult, MetricStatus
from kalanos.analysis.models.report import Report
from kalanos.analysis.models.schema import RefusalCode
from kalanos.cli import app
from kalanos.core.settings import get_settings

# Local
from helpers import (
    CAPTURE_INDEX_FIXTURE,
    CSV_FIXTURE,
    FIXTURES_DIR,
    LEROBOT_FIXTURE,
    LEROBOT_V2_0_FIXTURE,
    LEROBOT_V2_1_FIXTURE,
    NO_TIMESERIES_FIXTURE,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


runner = CliRunner()


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _grade(path: Path | UPath) -> Report:
    """Invoke `kalanos grade <path> --json` and parse the resulting report.

    Parameters
    ----------
    path : Path or UPath
        A recording or a folder of them, passed straight to `grade`.

    Returns
    -------
    Report
        The parsed model behind the run's stdout.
    """

    result = runner.invoke(app, ["grade", str(path), "--json"])
    assert result.exit_code == 0, result.output or repr(result.exception)
    return Report.model_validate_json(result.stdout)


def _iter_metric_results(report: Report) -> Iterator[tuple[str, MetricResult]]:
    """Walk every stream and channel in a graded report and yield their metrics.

    Parameters
    ----------
    report : Report
        A parsed grading result to walk.

    Yields
    ------
    tuple of (str, MetricResult)
        Every `(metric name, result)` pair found on any stream or channel,
        across every episode in the report.
    """

    for episode in report.episodes:
        for stream in episode.streams:
            yield from stream.metrics.items()
            for channel in stream.channels:
                yield from channel.metrics.items()


def _csv_ids() -> set[str]:
    """Read the distinct `id` column values straight out of the CSV fixture.

    Returns
    -------
    set of str
        Every distinct instance the fixture carries, independent of
        whatever the adapter did with them.
    """

    with CSV_FIXTURE.open() as handle:
        return {row["id"] for row in csv.DictReader(handle)}


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(scope="module", autouse=True)
def _clean_kalanos_env_for_module() -> Iterator[None]:
    """Strip KALANOS_ env vars around the module-scoped report fixtures below.

    `conftest.py`'s own env-clearing fixture is function-scoped, since it
    needs `monkeypatch`, so it runs after `corpus_report` and
    `single_file_report` are already built. Without this, an ambient
    `KALANOS_POLICY_PATH` from outside the test run would grade the whole
    corpus under the wrong policy.
    """

    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith("KALANOS_"):
                mp.delenv(key, raising=False)
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def corpus_report() -> Report:
    """Grade the whole fixture corpus once, shared by every test that reads it.

    Returns
    -------
    Report
        The report produced from `tests/fixtures/` as a folder.
    """

    return _grade(FIXTURES_DIR)


@pytest.fixture(scope="module")
def single_file_report() -> Report:
    """Grade the CSV fixture alone once, shared by every test that reads it.

    Returns
    -------
    Report
        The report produced from `arm_multi_device.csv` as a single file.
    """

    return _grade(CSV_FIXTURE)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_every_top_level_input_is_accounted_for(corpus_report):
    """Verify every top-level entry under the corpus is accounted for.

    A plain file must show up in the report at least once. A directory a
    whole-subtree adapter claims (`lerobot_v3_tiny/`) counts as accounted
    for once any file under it becomes a stream's source. `meta/info.json`
    and the episode index never become one, and are not reported as
    skipped either: that is the accepted cost of claiming a subtree
    wholesale instead of one file at a time. A path can legitimately show
    up more than once, since several episodes can share a parquet chunk,
    so this checks coverage rather than counting.
    """

    analysed = [
        path for episode in corpus_report.episodes for path in episode.source_paths
    ]
    skipped = [item.path for item in corpus_report.skipped]
    unresolved = [item.path for item in corpus_report.unresolved]
    all_reported = analysed + skipped + unresolved

    for entry in FIXTURES_DIR.iterdir():
        if entry.is_dir():
            assert any(entry in path.parents for path in all_reported), (
                f"{entry} was never claimed by anything the report accounts for"
            )
        else:
            assert entry in all_reported, f"{entry} is missing from the report"

    assert all(path.exists() for path in all_reported), "a reported path is not on disk"


def test_the_no_timeseries_fixture_is_refused_for_having_no_time_index(corpus_report):
    """Verify the flat metadata blob is read and refused, not merely unclaimed.

    `JsonAdapter` bids on this file and reads it, so it lands in `unresolved`
    with its own reason rather than in `skipped` with the generic
    `no_adapter`. Asserting on the code, rather than merely on the file's
    absence from the analysed episodes, catches a refusal landing on the
    wrong reason.
    """

    unresolved = next(
        item for item in corpus_report.unresolved if item.path == NO_TIMESERIES_FIXTURE
    )
    assert unresolved.code is RefusalCode.NO_TIME_INDEX


def test_rate_metrics_are_not_applicable_on_the_irregular_json_fixture(corpus_report):
    """Verify the irregular-sampling gate reaches end to end, not just at the adapter.

    `capture_index.json`'s streams are irregular, so `effective_hz` must come
    back `not_applicable` there while the CSV fixture's regular streams keep
    reporting a value.
    """

    capture_episode = next(
        episode
        for episode in corpus_report.episodes
        if CAPTURE_INDEX_FIXTURE in episode.source_paths
    )
    csv_episode = next(
        episode
        for episode in corpus_report.episodes
        if CSV_FIXTURE in episode.source_paths
    )

    capture_statuses = [
        result.status
        for stream in capture_episode.streams
        for name, result in stream.metrics.items()
        if name == "effective_hz"
    ]
    csv_statuses = [
        result.status
        for stream in csv_episode.streams
        for name, result in stream.metrics.items()
        if name == "effective_hz"
    ]

    assert capture_statuses
    assert all(status == MetricStatus.NOT_APPLICABLE for status in capture_statuses)
    assert csv_statuses
    assert MetricStatus.NOT_APPLICABLE not in csv_statuses


def test_a_folder_with_one_gradeable_file_and_one_unclaimed_file(tmp_path):
    """Verify an unclaimed file next to a gradeable one doesn't stop the run.

    One `.csv` an adapter reads, and one `.bin` nothing claims: the run
    must still produce a graded episode for the first and a `no_adapter`
    skip for the second, rather than aborting or losing either.
    """

    (tmp_path / "arm_multi_device.csv").write_bytes(CSV_FIXTURE.read_bytes())
    (tmp_path / "unclaimed.bin").write_bytes(b"\x00\x01\x02")

    report = _grade(tmp_path)

    assert len(report.episodes) == 1
    assert len(report.skipped) == 1
    assert report.skipped[0].reason == SkipReason.NO_ADAPTER
    assert report.skipped[0].path.name == "unclaimed.bin"


def test_the_graded_episode_names_the_adapter_that_read_it(single_file_report):
    """Verify GradedEpisode.adapter names `csv`, end to end through the CLI."""

    (episode,) = single_file_report.episodes
    assert episode.adapter == "csv"


def test_the_csv_fixture_yields_one_instance_per_id(single_file_report):
    """Verify the instance split produced exactly the ids the fixture carries."""

    (episode,) = single_file_report.episodes
    reported = [stream.instance for stream in episode.streams]
    assert sorted(reported) == sorted(_csv_ids())


def test_drop_rate_is_graded_for_the_csv_fixture(single_file_report):
    """Verify drop_rate discriminates rather than merely being marked graded.

    `armC`'s burst of dropped samples must actually surface as `critical`;
    a metric stuck reporting `good` everywhere would pass a weaker check
    that only looked for the absence of `report_only`.
    """

    statuses = [
        result.status
        for name, result in _iter_metric_results(single_file_report)
        if name == "drop_rate"
    ]

    assert statuses
    assert MetricStatus.REPORT_ONLY not in statuses
    assert MetricStatus.CRITICAL in statuses


def test_dt_jitter_is_report_only_for_the_csv_fixture(single_file_report):
    """Verify dt_jitter_ms stays report-only on every channel, per the policy."""

    statuses = [
        result.status
        for name, result in _iter_metric_results(single_file_report)
        if name == "dt_jitter_ms"
    ]

    assert statuses
    assert all(status == MetricStatus.REPORT_ONLY for status in statuses)


def test_grading_a_single_file_reports_that_file_alone(single_file_report):
    """Verify a single-file run reports only that file, with nothing skipped."""

    assert single_file_report.root == CSV_FIXTURE
    assert [
        path for episode in single_file_report.episodes for path in episode.source_paths
    ] == [CSV_FIXTURE]
    assert single_file_report.skipped == []
    assert single_file_report.unresolved == []


def test_a_file_and_its_folder_grade_it_identically(corpus_report, single_file_report):
    """Verify the CSV fixture grades the same whether passed alone or inside its folder.

    The headline command passes a single file; the corpus case
    passes a folder. Both go through `discovery.walk_folder` the same way,
    so the two `GradedEpisode` trees are compared whole, node for node,
    rather than by their top-level scores alone.
    """

    (single,) = single_file_report.episodes
    (from_corpus,) = [
        episode
        for episode in corpus_report.episodes
        if CSV_FIXTURE in episode.source_paths
    ]

    assert from_corpus == single


def test_the_lerobot_fixture_grades_end_to_end():
    """Verify the LeRobot fixture grades end to end through the CLI.

    Two episodes, each naming several source paths, `effective_hz` graded
    rather than refused for a missing limit, and nothing skipped.
    """

    report = _grade(LEROBOT_FIXTURE)

    assert len(report.episodes) == 2
    assert report.skipped == []
    for episode in report.episodes:
        assert len(episode.source_paths) >= 2
        for stream in episode.streams:
            reason = stream.metrics["effective_hz"].evidence.get("ungraded_reason", "")
            assert "declared under limits" not in reason


def test_the_lerobot_fixtures_episodes_carry_the_winning_adapters_bid():
    """Verify the adapter's confidence on its winning bid reaches each episode.

    `LeRobotV3Adapter` bids `CONFIDENCE` on this fixture; that same number
    should land on `GradedEpisode.adapter_confidence` for both episodes,
    rather than being discarded once selection has already decided the winner.
    """

    report = _grade(LEROBOT_FIXTURE)

    assert len(report.episodes) == 2
    for episode in report.episodes:
        assert episode.adapter_confidence == pytest.approx(CONFIDENCE)


def test_the_lerobot_v2_0_fixture_grades_end_to_end():
    """Verify the LeRobot v2.0 fixture grades end to end through the CLI.

    Two episodes, each naming both its own parquet and its own mp4,
    `effective_hz` graded rather than refused for a missing limit, and
    nothing skipped.
    """

    report = _grade(LEROBOT_V2_0_FIXTURE)

    assert len(report.episodes) == 2
    assert report.skipped == []
    for episode in report.episodes:
        assert len(episode.source_paths) >= 2
        for stream in episode.streams:
            reason = stream.metrics["effective_hz"].evidence.get("ungraded_reason", "")
            assert "declared under limits" not in reason


def test_the_lerobot_v2_1_fixture_grades_end_to_end():
    """Verify the LeRobot v2.1 fixture grades end to end through the CLI.

    Two episodes, each naming exactly one source path since this fixture
    declares no camera, `effective_hz` graded rather than refused for a
    missing limit, and nothing skipped.
    """

    report = _grade(LEROBOT_V2_1_FIXTURE)

    assert len(report.episodes) == 2
    assert report.skipped == []
    for episode in report.episodes:
        assert len(set(episode.source_paths)) == 1
        for stream in episode.streams:
            reason = stream.metrics["effective_hz"].evidence.get("ungraded_reason", "")
            assert "declared under limits" not in reason
