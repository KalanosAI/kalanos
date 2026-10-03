"""Verifies the detector benchmark on synthetic episodes and a LeRobot fixture."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib.metadata
from pathlib import Path
from unittest.mock import patch

# External
from typer.testing import CliRunner

# Internal
from kalanos import benchmark
from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.models.binding import Bundle, RequirementsSection
from kalanos.analysis.models.domain import Clock, ClockInfo, Episode
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
from kalanos.testing import Defect, clean_frames, clean_recording

# Local
from helpers import FIXTURES_DIR


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# A scope under which vision grades, so cameras are decoded for injection.
VIDEO_SCOPE = Bundle(
    requirements=RequirementsSection(required_capabilities=["sampled_video_quality"])
)


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


def _camera_episodes(count: int = 3) -> list[Episode]:
    camera = clean_frames(frames=100, hz=30.0)
    actions = clean_recording(
        hz=30.0, samples=100, taxonomy_type="action.joint_position_command"
    ).model_copy(update={"is_regular": True})
    return [
        Episode(id=f"episode_{index}", streams=[camera, actions])
        for index in range(count)
    ]


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
    """Generated clocks remain ineligible for acquisition checks after injection."""
    episodes = _episodes()
    for episode in episodes:
        episode.streams = [
            stream.model_copy(
                update={
                    "clock": Clock.RECONSTRUCTED,
                    "clock_info": ClockInfo.from_legacy(Clock.RECONSTRUCTED),
                }
            )
            for stream in episode.streams
        ]
    rates = benchmark_episodes(
        episodes, policy=load_policy(None), n_episodes=len(episodes), sample=3
    )

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
    # The default scope does not grade vision, so no camera is decoded for injection.
    for defect in (Defect.BLUR, Defect.CLIPPED, Defect.FROZEN_FRAMES):
        assert dataset.not_injected[defect] == "the scope does not grade vision"


def test_blur_and_clipping_are_injected_into_in_memory_frames():
    """Verify the camera defects reach a stream holding its frames in memory."""

    episodes = _camera_episodes()
    rates = benchmark_episodes(
        episodes, policy=load_policy(None), n_episodes=len(episodes), sample=3
    )

    for defect in (Defect.BLUR, Defect.CLIPPED, Defect.FROZEN_FRAMES):
        assert defect not in rates.not_injected, rates.not_injected.get(defect)


def test_the_benchmark_keeps_vision_report_only_outside_a_video_scope():
    """Verify the benchmark scopes vision as grade does, and closes remote video."""

    closed: list[None] = []
    with patch.object(benchmark, "close_remote_handles", lambda: closed.append(None)):
        dataset = benchmark.benchmark_dataset(
            str(FIXTURES_DIR / "lerobot_v3_tiny"), sample=0
        )

    vision_rows = [row for row in dataset.benign if row.family == "vision"]
    assert vision_rows
    for row in vision_rows:
        assert row.observed_episodes == 0, row
    assert closed


def test_a_video_that_cannot_be_decoded_leaves_the_camera_defects_with_a_reason(
    monkeypatch,
):
    """Verify a decode failure is reported per camera defect, not raised."""

    def fail(self, size):
        raise benchmark.DecodeFailed("broken file")

    monkeypatch.setattr(benchmark.VideoPayload, "rgb_frames", fail)
    dataset = benchmark.benchmark_dataset(
        str(FIXTURES_DIR / "lerobot_v3_tiny"), sample=2, bundle=VIDEO_SCOPE
    )

    for defect in (Defect.BLUR, Defect.CLIPPED, Defect.FROZEN_FRAMES):
        assert "broken file" in dataset.not_injected[defect]


def test_a_video_over_the_decode_cap_is_reported_not_decoded(monkeypatch):
    """Verify a camera over the cap is refused before its frames are decoded."""

    def fail(self, size):
        raise AssertionError("decoded a video over the cap")

    monkeypatch.setattr(benchmark.VideoPayload, "rgb_frames", fail)
    monkeypatch.setattr(benchmark, "_MAX_DECODE_BYTES", 1)
    dataset = benchmark.benchmark_dataset(
        str(FIXTURES_DIR / "lerobot_v3_tiny"), sample=2, bundle=VIDEO_SCOPE
    )

    for defect in (Defect.BLUR, Defect.CLIPPED, Defect.FROZEN_FRAMES):
        assert "decode cap" in dataset.not_injected[defect]


def test_camera_defects_are_injected_at_native_resolution(monkeypatch):
    """Verify a camera defect goes into frames at the video's own size."""

    shapes: set[tuple[int, ...]] = set()
    inject = benchmark._inject

    def record(stream, channel, defect):
        if isinstance(stream.payload, benchmark.SyntheticFrames):
            shapes.update(frame.shape for frame in stream.payload.frames)
        return inject(stream, channel, defect)

    monkeypatch.setattr(benchmark, "_inject", record)
    dataset = benchmark.benchmark_dataset(
        str(FIXTURES_DIR / "lerobot_v3_tiny"), sample=2, bundle=VIDEO_SCOPE
    )

    for defect in (Defect.BLUR, Defect.CLIPPED, Defect.FROZEN_FRAMES):
        assert defect not in dataset.not_injected, dataset.not_injected.get(defect)

    assert shapes == {(32, 32, 3)}
