"""Verifies the detector benchmark on synthetic episodes and a LeRobot fixture."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib.metadata
from pathlib import Path

# External
from typer.testing import CliRunner

# Internal
from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.models.domain import Episode
from kalanos.assets.policy import load_policy
from kalanos.benchmark import (
    Benchmark,
    BenignRate,
    DatasetBenchmark,
    DetectionRate,
    _Rates,
    benchmark_episodes,
    render_markdown,
)
from kalanos.cli import app
from kalanos.testing import Defect, clean_recording

# Local
from helpers import FIXTURES_DIR


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _episodes(count: int = 3) -> list[Episode]:
    stream = clean_recording(samples=200).model_copy(update={"is_regular": True})
    return [Episode(id=f"episode_{index}", streams=[stream]) for index in range(count)]


def _benchmark(sample: int) -> _Rates:
    episodes = _episodes()
    return benchmark_episodes(
        episodes, policy=load_policy(None), n_episodes=len(episodes), sample=sample
    )


def _tables(markdown: str) -> list[list[str]]:
    """Group the document's consecutive `|` lines into tables."""

    tables: list[list[str]] = []
    previous_was_row = False
    for line in markdown.splitlines():
        is_row = line.startswith("|")
        if is_row and not previous_was_row:
            tables.append([])
        if is_row:
            tables[-1].append(line)
        previous_was_row = is_row
    return tables


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_markdown_tables_stay_well_formed():
    """Verify each rendered table keeps one column count across all its rows."""

    dataset = DatasetBenchmark(
        uri="/data/local",
        repo_id=None,
        revision=None,
        adapter="lerobot_v3",
        n_episodes=2,
        n_sampled=1,
        benign=[
            BenignRate(
                metric="spike_pct",
                family="integrity",
                observed_episodes=2,
                firing_episodes=1,
                rate=0.5,
            ),
            BenignRate(
                metric="effective_hz",
                family="timing",
                observed_episodes=0,
                firing_episodes=0,
                rate=None,
                not_observable="timestamps are reconstructed",
            ),
        ],
        detection=[
            DetectionRate(
                defect=Defect.SPIKE,
                metric="spike_pct",
                eligible=3,
                detected=3,
                rate=1.0,
            ),
            DetectionRate(
                defect=Defect.NULLS,
                metric="missing_pct",
                eligible=3,
                detected=2,
                rate=2 / 3,
            ),
        ],
        not_injected={Defect.FROZEN_FRAMES: "no stream it applies to"},
    )
    benchmark = Benchmark(
        kalanos_version="0.0.0", policy_version=1, injection={}, datasets=[dataset]
    )

    markdown = render_markdown(benchmark)

    [benign, detection] = _tables(markdown)
    for table in (benign, detection):
        assert len({line.count("|") for line in table}) == 1, table
    assert "| spike_pct | integrity | 1 / 2 | 50.0% |" in benign
    assert "not observable: timestamps are reconstructed" in benign[-1]
    assert "| spike |  | 3/3 |" in detection
    assert not any(line.startswith("| frozen_frames") for line in detection)
    assert "Revision" not in markdown


def test_every_registered_metric_gets_a_row_and_an_unobservable_one_never_reads_zero():
    """Verify an unobserved metric is listed with its reason instead of as 0% firing."""

    rates = _benchmark(sample=0)

    assert sorted(row.metric for row in rates.benign) == sorted(
        entry.name for entry in registered_metrics()
    )
    unobserved = [row for row in rates.benign if row.observed_episodes == 0]
    assert all(row.rate is None and row.not_observable for row in unobserved)
    assert "dead_taxel_pct" in {row.metric for row in unobserved}


def test_a_spike_is_detected_wherever_spike_pct_graded_clean():
    """Verify a 20-std spike fires `spike_pct` on every channel it graded good."""

    rates = _benchmark(sample=3)

    [cell] = [
        row
        for row in rates.detection
        if row.defect == Defect.SPIKE and row.metric == "spike_pct"
    ]
    assert cell.eligible > 0
    assert cell.detected == cell.eligible


def test_a_metric_not_graded_on_the_clean_stream_has_no_detection_cell():
    """Verify timing metrics a reconstructed clock leaves ungraded get no cell."""

    rates = _benchmark(sample=3)

    timing = {"effective_hz", "dt_jitter_ms", "drop_rate"}
    assert not [row for row in rates.detection if row.metric in timing]


def test_a_defect_with_nothing_to_inject_into_is_listed_with_a_reason():
    """Verify defects needing taxels or image frames are reported as not injected."""

    rates = _benchmark(sample=3)

    for defect in (Defect.DEAD_TAXEL, Defect.HYSTERESIS, Defect.FROZEN_FRAMES):
        assert rates.not_injected[defect]


def test_the_cli_benchmarks_a_lerobot_dataset_into_json(tmp_path: Path):
    """Verify `kalanos benchmark --out` writes a Benchmark for a local LeRobot set."""

    out = tmp_path / "b.json"
    result = CliRunner().invoke(
        app, ["benchmark", str(FIXTURES_DIR / "lerobot_v3_tiny"), "--out", str(out)]
    )

    assert result.exit_code == 0, result.output
    benchmark = Benchmark.model_validate_json(out.read_text())
    assert benchmark.kalanos_version == importlib.metadata.version("kalanos")
    [dataset] = benchmark.datasets
    assert dataset.n_episodes == 2
    # The fixture's camera stream is a decoded video, which has no frames to freeze.
    assert Defect.FROZEN_FRAMES in dataset.not_injected
