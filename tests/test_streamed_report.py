"""Verifies a streamed report writes what a whole `Report` would, on every output.

`kalanos grade` holds graded episodes as compressed JSON when it writes JSON,
and as models otherwise.
Both paths grade every fixture here, and each output must come out the same.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import gc
import io
import json
import os
import tracemalloc
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

# External
import numpy as np
import pytest

# Local
from calibration_helpers import synthetic_manifest
from diagnostics_helpers import episode, signal, window_spec
from pydantic_core import to_json
from test_diagnostics import camera, camera_read, cohort_spec, pair
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos import api, publish
from kalanos.analysis import calibration, pipeline
from kalanos.analysis.adapters.discover import (
    AdapterDiscovery,
    LoadedAdapter,
    PluginSource,
)
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.compact_keys import decode_keys, encode_keys
from kalanos.analysis.models.diagnostics import (
    DiagnosticPlan,
    DiagnosticReviewPolicy,
    MotionSpec,
    Selector,
)
from kalanos.analysis.models.domain import Episode, Stream
from kalanos.analysis.models.errors import MappingOverrideError
from kalanos.analysis.models.legacy import load_any
from kalanos.analysis.models.mapping import MappingOverride, OverrideOrigin
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.report import AnalysedEpisode, Report
from kalanos.analysis.models.scoring import Finding
from kalanos.analysis.reporting import assemble
from kalanos.analysis.reporting.assemble import assemble_report
from kalanos.analysis.reporting.card import render_terminal
from kalanos.analysis.reporting.omit import omit_empty_defaults
from kalanos.analysis.reporting.render import render_compact_json, render_json
from kalanos.analysis.reporting.stream import StreamedReport
from kalanos.analysis.reporting.write import write_report
from kalanos.assets.policy import load_default_policy
from kalanos.cli import app
from kalanos.core.settings import get_settings
from kalanos.testing.injectors import SyntheticFrames

from helpers import (
    CSV_FIXTURE,
    FIXTURES_DIR,
    HDF5_FIXTURE,
    LEROBOT_FIXTURE,
    LEROBOT_V2_0_FIXTURE,
    LEROBOT_V2_1_FIXTURE,
    MCAP_FIXTURE,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

SOURCES = [
    FIXTURES_DIR,
    CSV_FIXTURE,
    HDF5_FIXTURE,
    MCAP_FIXTURE,
    LEROBOT_FIXTURE,
    LEROBOT_V2_0_FIXTURE,
    LEROBOT_V2_1_FIXTURE,
]

KEY = "klns_live_" + "ab" * 16

runner = CliRunner()


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(scope="module", autouse=True)
def _clean_kalanos_env_for_module() -> Iterator[None]:
    """Strip KALANOS_ env vars around the module-scoped reports below."""

    with pytest.MonkeyPatch.context() as mp:
        for key in list(os.environ):
            if key.startswith("KALANOS_"):
                mp.delenv(key, raising=False)
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


@pytest.fixture(scope="module", params=SOURCES, ids=lambda path: path.name)
def graded(request: pytest.FixtureRequest) -> tuple[Report, StreamedReport]:
    """Grade one fixture of the corpus, or the corpus as a folder, on both paths.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Carries the source to grade.

    Returns
    -------
    tuple of (Report, StreamedReport)
        The run that kept its episodes, given the other run's identity and duration,
        and the run that streamed them.
    """

    source: UPath = request.param
    kept = api._grade(source, keep_episodes=True)
    streamed = api._grade(source, keep_episodes=False)
    assert isinstance(kept, Report) and isinstance(streamed, StreamedReport)
    return _aligned(kept, streamed.header), streamed


@pytest.fixture(scope="module")
def report(graded: tuple[Report, StreamedReport]) -> Report:
    """The run of `graded` that kept its episodes.

    Parameters
    ----------
    graded : tuple of (Report, StreamedReport)
        Both runs.

    Returns
    -------
    Report
        The kept run.
    """

    return graded[0]


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Capture every request body `publish_report` sends, answering each as accepted.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the HTTP opener and points `KALANOS_HUB_URL` at a local address.

    Returns
    -------
    list[bytes]
        The bodies sent, in order.
    """

    bodies: list[bytes] = []
    answer = {"report_id": "rep_1", "page": None, "dashboard": "http://127.0.0.1/d"}

    class Opener:
        def open(self, request, timeout):
            bodies.append(request.data)
            return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr(publish.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setenv("KALANOS_HUB_URL", "http://127.0.0.1:9")
    return bodies


# ░█░█░█▀▀░█░░░█▀█░█▀▀░█▀▄░█▀▀
# ░█▀█░█▀▀░█░░░█▀▀░█▀▀░█▀▄░▀▀█
# ░▀░▀░▀▀▀░▀▀▀░▀░░░▀▀▀░▀░▀░▀▀▀


def _whole_dump(report: Report, *, compact: bool) -> bytes:
    """Dump a report as one whole-report dict.

    Parameters
    ----------
    report : Report
        The report to dump.
    compact : bool
        Replace every key with its alias.

    Returns
    -------
    bytes
        The report's JSON.
    """

    data = omit_empty_defaults(report, report.model_dump(mode="json"))
    return to_json(encode_keys(data) if compact else data, inf_nan_mode="null")


def _aligned(report: Report, other: Report) -> Report:
    """Give `report` the run identity and duration of `other`, which differ per run.

    Parameters
    ----------
    report : Report
        A graded report.
    other : Report
        Another run of the same grade.

    Returns
    -------
    Report
        A copy of `report` with `other`'s `run` and `duration_s`.
    """

    return report.model_copy(update={"run": other.run, "duration_s": other.duration_s})


def _grade_cli(*args: str) -> bytes:
    """Run `kalanos grade` and return its stdout.

    Parameters
    ----------
    *args : str
        The arguments after `grade`.

    Returns
    -------
    bytes
        What the command wrote to stdout.
    """

    result = runner.invoke(app, ["grade", *args])
    assert result.exit_code in (0, 1), result.output or repr(result.exception)
    return result.stdout_bytes


def _as_report(data: dict) -> Report:
    """Load a parsed report file, compact or not.

    Parameters
    ----------
    data : dict
        A parsed report.

    Returns
    -------
    Report
        The report it holds.
    """

    return Report.model_validate(decode_keys(data) if "_compact" in data else data)


class _OneSource:
    """An adapter reading `n` single-stream episodes from any path, logging each read.

    Parameters
    ----------
    n : int
        How many episodes the source holds.
    events : list or None
        When given, receives `("read", id)` as each episode is yielded.
    alive : list or None
        When given, receives, as each episode is read,
        whether each earlier episode's payload frame is still in memory.
    """

    name = "one_source"

    def __init__(
        self, n: int, *, events: list | None = None, alive: list | None = None
    ) -> None:
        """Hold the source's shape and what to record.

        Parameters
        ----------
        n : int
            How many episodes the source holds.
        events : list or None
            When given, receives `("read", id)` as each episode is yielded.
        alive : list or None
            When given, receives, as each episode is read,
            whether each earlier episode's payload frame is still in memory.
        """

        self._n = n
        self._events = events
        self._alive = alive

    def detect(self, path: UPath) -> float:
        """Claim any path."""

        return 1.0

    def describe(self, path: UPath) -> DatasetInfo:
        """Declare how many episodes the source holds.

        Parameters
        ----------
        path : UPath
            The candidate.

        Returns
        -------
        DatasetInfo
            The declared episode count.
        """

        return DatasetInfo(adapter=self.name, path=path, episode_count=self._n)

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield each episode, built only when asked for.

        Parameters
        ----------
        path : UPath
            The candidate.
        sample : int or None
            Unused.

        Yields
        ------
        Episode
            One single-stream recording.
        """

        frames: list[weakref.ref] = []
        for k in range(self._n):
            if self._alive is not None:
                gc.collect()
                self._alive.append([ref() is not None for ref in frames])
            if self._events is not None:
                self._events.append(("read", f"e{k}"))
            stream = signal()
            assert stream.payload is not None
            frames.append(weakref.ref(stream.payload.fetch()))
            yield episode(stream, identifier=f"e{k}")
            del stream


def _one_source_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter: _OneSource,
    **options,
):
    """Grade a one-file folder that only `adapter` reads.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces adapter discovery with `adapter` alone.
    tmp_path : Path
        Where the file to grade is written.
    adapter : _OneSource
        The adapter that reads it.
    **options
        Passed to `pipeline.run`.

    Returns
    -------
    Report or StreamedReport
        What `pipeline.run` returned.
    """

    loaded = LoadedAdapter(
        name=adapter.name,
        adapter=adapter,
        source=PluginSource.ENTRY_POINT,
        origin="test",
    )
    monkeypatch.setattr(
        pipeline,
        "discover_adapters",
        lambda: AdapterDiscovery(adapters=[loaded], failures=[]),
    )
    (tmp_path / "recording.bin").write_bytes(b"recording")
    return pipeline.run(UPath(tmp_path), policy=load_default_policy(), **options)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_render_json_writes_the_whole_dump(report):
    assert render_json(report).encode() == _whole_dump(report, compact=False)


def test_render_compact_json_writes_the_whole_compact_dump(report):
    assert render_compact_json(report).encode() == _whole_dump(report, compact=True)


@pytest.mark.parametrize("compact", [False, True])
def test_write_report_writes_the_whole_dump(report, tmp_path, compact):
    path = write_report(report, tmp_path / "report.json", compact=compact)

    assert path.read_bytes() == _whole_dump(report, compact=compact)


def test_a_report_without_episodes_leaves_them_out(report):
    empty = report.model_copy(update={"episodes": []})

    assert render_json(empty).encode() == _whole_dump(empty, compact=False)
    assert render_compact_json(empty).encode() == _whole_dump(empty, compact=True)


@pytest.mark.parametrize("compact", [False, True])
def test_both_paths_write_the_same_report_file(graded, tmp_path, compact):
    kept, streamed = graded
    kept_path = write_report(kept, tmp_path / "kept.json", compact=compact)
    streamed_path = tmp_path / "streamed.json"
    with streamed_path.open("wb") as out:
        streamed.write(out, compact=compact)

    assert streamed_path.read_bytes() == kept_path.read_bytes()


def test_a_streamed_episode_reads_back_as_the_kept_one(graded):
    kept, streamed = graded

    assert list(streamed.graded_episodes()) == kept.episodes


def test_both_paths_print_the_same_card(graded):
    kept, streamed = graded
    card = render_terminal(
        streamed.header,
        width=100,
        episodes=streamed.card_rows,
        unmapped=streamed.unmapped_types,
    )

    assert card == render_terminal(kept, width=100)


@pytest.mark.parametrize("compact", [[], ["--compact"]], ids=["full", "compact"])
def test_the_cli_writes_a_report_file_as_the_kept_report_renders(
    report, tmp_path, compact
):
    source = report.root
    path = tmp_path / "report.json"
    _grade_cli(str(source), "--report", str(path), *compact)

    written = path.read_bytes()
    loaded = load_any(UPath(path))
    assert isinstance(loaded, Report)
    expected = _aligned(report, loaded)
    render = render_compact_json if compact else render_json
    assert written == render(expected).encode()


@pytest.mark.parametrize("compact", [[], ["--compact"]], ids=["full", "compact"])
def test_the_cli_prints_json_as_the_kept_report_renders(report, compact):
    stdout = _grade_cli(str(report.root), "--json", *compact)

    expected = _aligned(report, _as_report(json.loads(stdout)))
    render = render_compact_json if compact else render_json
    assert stdout == render(expected).encode() + b"\n"


@pytest.mark.parametrize("compact", [[], ["--compact"]], ids=["full", "compact"])
def test_the_cli_publishes_the_kept_report(report, sent, compact):
    _grade_cli(
        str(report.root), "--publish", "--api-key", KEY, "--name", "acme/set", *compact
    )

    [body] = [json.loads(b) for b in sent]
    expected = _aligned(report, _as_report(body["report"]))
    render = render_compact_json if compact else render_json
    assert body == {"report": json.loads(render(expected)), "name": "acme/set"}


@pytest.mark.parametrize(
    ("suffix", "keeps"),
    [(".json", False), (".yaml", True), (".yml", True), (".html", True)],
)
def test_only_a_json_report_streams_its_episodes(monkeypatch, tmp_path, suffix, keeps):
    calls: list[bool] = []
    grade = api._grade

    def spy(*args, **kwargs):
        calls.append(kwargs["keep_episodes"])
        result = grade(*args, **kwargs)
        assert isinstance(result, Report) is keeps
        return result

    monkeypatch.setattr(api, "_grade", spy)
    _grade_cli(str(CSV_FIXTURE), "--report", str(tmp_path / f"report{suffix}"))

    assert calls == [keeps]


def _graded_both_ways(
    episodes: Callable[[], list[Episode]],
    plan: DiagnosticPlan,
    tmp_path: Path,
    *,
    policy: Policy | None = None,
) -> tuple[Report, StreamedReport, bytes, bytes]:
    """Grade the same episodes under a diagnostics plan, kept and streamed.

    Parameters
    ----------
    episodes : Callable[[], list[Episode]]
        Builds the episodes, afresh for each run.
    plan : DiagnosticPlan
        The diagnostics to run.
    tmp_path : Path
        Where both reports are written.
    policy : Policy or None
        The policy every episode grades against; the default when `None`.

    Returns
    -------
    tuple of (Report, StreamedReport, bytes, bytes)
        Both runs, then the kept run's report file and the streamed run's.
    """

    policy = policy or load_default_policy()

    def run(keep: bool) -> Report | StreamedReport:
        """Grade the episodes once."""

        return assemble_report(
            root=UPath("fixture"),
            analysed=deque(
                AnalysedEpisode(
                    episode=e, adapter="csv", adapter_confidence=1, policy=policy
                )
                for e in episodes()
            ),
            policy=policy,
            diagnostics_plan=plan,
            vision_samples=10,
            keep_episodes=keep,
        )

    kept, streamed = run(True), run(False)
    assert isinstance(kept, Report) and isinstance(streamed, StreamedReport)
    kept_path = write_report(kept, tmp_path / "kept.json")
    streamed_path = tmp_path / "streamed.json"
    with streamed_path.open("wb") as out:
        streamed.write(out)
    return kept, streamed, kept_path.read_bytes(), streamed_path.read_bytes()


def _dark_camera() -> Stream:
    """A camera whose fourth frame is black, which the vision review flags."""

    cam = camera()
    assert isinstance(cam.payload, SyntheticFrames)
    cam.payload.frames[3] = np.zeros((16, 16, 3), dtype=np.uint8)
    return cam


def _gappy_signal() -> Stream:
    """A state signal missing two fifths of its rows, which is a critical finding."""

    values = [
        float("nan") if 20 <= i < 60 else v
        for i, v in enumerate(np.linspace(0, 1, 101))
    ]
    return signal(values=values)


def test_a_diagnostics_plan_streams_its_episodes():
    policy = load_default_policy()
    analysed = [
        AnalysedEpisode(
            episode=episode(signal("command"), signal()),
            adapter="csv",
            adapter_confidence=1,
            policy=policy,
        )
    ]

    result = assemble_report(
        root=UPath("fixture"),
        analysed=analysed,
        policy=policy,
        diagnostics_plan=DiagnosticPlan(windows=[window_spec()]),
        keep_episodes=False,
    )

    assert isinstance(result, StreamedReport)
    assert result.header.diagnostics is not None
    assert [r.kind for r in result.header.diagnostics.results] == ["windows"]


@pytest.mark.parametrize(
    "plan",
    [
        DiagnosticPlan(timing=[pair()]),
        DiagnosticPlan(
            motion=[
                MotionSpec(
                    id="m", channel=Selector(feature="state", index=0), max_gap_s=0.1
                )
            ]
        ),
        DiagnosticPlan(vision=True),
        DiagnosticPlan(windows=[window_spec()]),
        DiagnosticPlan(cohorts=[cohort_spec()]),
    ],
    ids=["timing", "motion", "vision", "windows", "cohorts"],
)
def test_a_diagnostics_plan_streams_the_same_report(plan, tmp_path):
    policy = load_default_policy()
    if plan.vision:
        policy.diagnostic_reviews = DiagnosticReviewPolicy(
            vision_max_clipped_fraction=0.9
        )

    def episodes() -> list[Episode]:
        """Two episodes the cohort lists, each with a camera that trips the review."""

        return [
            episode(signal("command"), _gappy_signal(), _dark_camera(), identifier=x)
            for x in ("a", "b")
        ]

    kept, _, kept_file, streamed_file = _graded_both_ways(
        episodes, plan, tmp_path, policy=policy
    )

    assert kept.diagnostics is not None
    assert all(r.availability.value != "error" for r in kept.diagnostics.results)
    if plan.vision:
        assert [camera_read(kept).consequence] == ["review"]
    assert streamed_file == kept_file


def test_a_calibrated_window_resolves_after_grading(monkeypatch, tmp_path):
    """A manifest accepts the missing-data finding only once the producer is known."""

    evaluate = calibration.evaluate

    def accept_once_identified(policy, context, now=None):
        """Accept every manifest match whose detector the run has identified."""

        if context["detector_digest"] is None:
            return evaluate(policy, context, now)
        return {"accepted": True, "manifest_id": "fixture", "context": context}

    monkeypatch.setattr(calibration, "evaluate", accept_once_identified)
    policy = load_default_policy()
    policy = policy.model_copy(
        update={
            "calibration_manifests": [
                synthetic_manifest(
                    {
                        "metric_id": "integrity.missing_pct",
                        "taxonomy_type": "proprio.joint_position",
                        "detector_digest": "1" * 64,
                        "thresholds_digest": "2" * 64,
                        "binding_digest": "3" * 64,
                        "scope_digest": "4" * 64,
                    }
                )
            ]
        }
    )

    kept, streamed, kept_file, streamed_file = _graded_both_ways(
        lambda: [episode(_gappy_signal())],
        DiagnosticPlan(windows=[window_spec()]),
        tmp_path,
        policy=policy,
    )

    [missing] = [f for f in kept.findings if f.metric_id == "integrity.missing_pct"]
    assert isinstance(missing, Finding) and missing.consequence.value == "block"
    assert kept.diagnostics is not None
    [result] = kept.diagnostics.results
    records = result.evidence["windows"]
    assert result.measurements["counts"] == {
        "pass": 0,
        "blocked": 8,
        "review": 0,
        "unknown": 0,
    }
    assert all(r["status"] == "blocked" for r in records)
    assert all(f"finding:{missing.id}" in r["reasons"] for r in records)
    assert not any("pending" in r for r in records)
    assert streamed_file == kept_file


def test_a_cohort_holds_no_episode(monkeypatch, tmp_path):
    graded: list[weakref.ref] = []
    alive: list[list[bool]] = []
    grade_episode = assemble.grade_episode

    def spy(episode, **options):
        gc.collect()
        alive.append([ref() is not None for ref in graded])
        graded.append(weakref.ref(episode))
        return grade_episode(episode, **options)

    monkeypatch.setattr(assemble, "grade_episode", spy)
    plan = DiagnosticPlan(cohorts=[cohort_spec()])
    kept, streamed, _, _ = _graded_both_ways(
        lambda: [episode(signal(), identifier=x) for x in ("a", "b")],
        plan,
        tmp_path,
    )

    # The kept run grades first, and lets go of its episodes once it returns.
    assert alive == [[], [True], [False, False], [False, False, False]]
    assert kept.diagnostics is not None and streamed.header.diagnostics is not None
    assert streamed.header.diagnostics.results == kept.diagnostics.results
    assert kept.diagnostics.results[0].availability.value == "computed"


def _write_peak(
    tmp_path: Path,
    n: int,
    *,
    keep: bool,
    plan: DiagnosticPlan | None = None,
    cameras: bool = False,
) -> int:
    """Grade `n` synthetic episodes and write their report, measuring the peak memory.

    Parameters
    ----------
    tmp_path : Path
        Where the report is written, outside the traced memory.
    n : int
        How many episodes, each a copy of one recording under its own id.
    keep : bool
        Grade into a `Report`, written by `write_report`,
        rather than a `StreamedReport`.
    plan : DiagnosticPlan or None
        The diagnostics to run.
    cameras : bool
        Give each episode a camera too.

    Returns
    -------
    int
        The traced peak, in bytes, from grading through writing.
        The recordings themselves were read before tracing started.
    """

    policy = load_default_policy()
    analysed = deque(
        AnalysedEpisode(
            episode=episode(
                signal("command"),
                signal(),
                *([camera()] if cameras else []),
                identifier=f"e{k}",
            ),
            adapter="csv",
            adapter_confidence=1,
            policy=policy,
        )
        for k in range(n)
    )
    path = tmp_path / f"{n}-{keep}-{plan is not None}-{cameras}.json"
    tracemalloc.start()
    try:
        result = assemble_report(
            root=UPath("fixture"),
            analysed=analysed,
            policy=policy,
            diagnostics_plan=plan,
            keep_episodes=keep,
        )
        if isinstance(result, Report):
            write_report(result, path)
        else:
            with path.open("wb") as out:
                result.write(out)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.slow
def test_a_streamed_report_stays_flat_as_episodes_grow(tmp_path):
    kept = _write_peak(tmp_path, 400, keep=True)
    small = _write_peak(tmp_path, 50, keep=False)
    large = _write_peak(tmp_path, 400, keep=False)

    assert large < 0.4 * kept
    assert large < 3 * small


@pytest.mark.slow
def test_a_diagnostics_plan_streams_within_twice_a_run_without_one(tmp_path):
    """The vision results stay in the report header, so they still grow with it."""

    plan = DiagnosticPlan(vision=True)
    kept = _write_peak(tmp_path, 400, keep=True, plan=plan, cameras=True)
    plain = _write_peak(tmp_path, 400, keep=False, cameras=True)
    planned = _write_peak(tmp_path, 400, keep=False, plan=plan, cameras=True)

    assert planned < 0.4 * kept
    assert planned < 2 * plain


def test_a_streamed_run_lets_go_of_each_recording_once_graded(monkeypatch):
    """Two episodes read from one source: the first is freed while the second grades."""

    graded: list[weakref.ref] = []
    alive: list[list[bool]] = []
    grade_episode = assemble.grade_episode

    def spy(episode, **options):
        gc.collect()
        alive.append([ref() is not None for ref in graded])
        graded.append(weakref.ref(episode))
        return grade_episode(episode, **options)

    monkeypatch.setattr(assemble, "grade_episode", spy)
    result = pipeline.run(
        UPath(LEROBOT_FIXTURE), policy=load_default_policy(), keep_episodes=False
    )

    assert isinstance(result, StreamedReport)
    assert alive == [[], [False]]


def test_a_run_grades_each_episode_before_reading_the_next(monkeypatch, tmp_path):
    events: list[tuple[str, str]] = []
    graded: list[weakref.ref] = []
    alive: list[list[bool]] = []
    grade_episode = assemble.grade_episode

    def spy(episode, **options):
        gc.collect()
        alive.append([ref() is not None for ref in graded])
        graded.append(weakref.ref(episode))
        events.append(("grade", episode.id.rsplit("::", 1)[-1]))
        return grade_episode(episode, **options)

    monkeypatch.setattr(assemble, "grade_episode", spy)
    result = _one_source_run(
        monkeypatch, tmp_path, _OneSource(4, events=events), keep_episodes=False
    )

    assert isinstance(result, StreamedReport)
    assert events == [(step, f"e{k}") for k in range(4) for step in ("read", "grade")]
    assert alive == [[False] * k for k in range(4)]


def test_a_run_lets_go_of_each_recording_before_reading_the_next(monkeypatch, tmp_path):
    alive: list[list[bool]] = []
    result = _one_source_run(
        monkeypatch, tmp_path, _OneSource(4, alive=alive), keep_episodes=False
    )

    assert isinstance(result, StreamedReport)
    assert alive == [[False] * k for k in range(4)]


def test_an_unmatched_override_fails_once_every_episode_is_read(monkeypatch, tmp_path):
    graded: list[str] = []
    grade_episode = assemble.grade_episode

    def spy(episode, **options):
        graded.append(episode.id)
        return grade_episode(episode, **options)

    monkeypatch.setattr(assemble, "grade_episode", spy)
    override = MappingOverride(
        feature="missing",
        taxonomy_type="proprio.joint_position",
        origin=OverrideOrigin.ARGUMENT,
    )
    with pytest.raises(MappingOverrideError) as excinfo:
        _one_source_run(monkeypatch, tmp_path, _OneSource(3), overrides=[override])

    message = str(excinfo.value)
    assert "'missing' (from argument)" in message
    assert "state" in message
    assert len(graded) == 3


def test_a_refusal_raised_while_grading_propagates(monkeypatch, tmp_path):
    def refuse(episode, **options):
        raise AdapterRefusal(UPath(tmp_path), "payload could not be read")

    monkeypatch.setattr(assemble, "grade_episode", refuse)
    with pytest.raises(AdapterRefusal, match="payload could not be read"):
        _one_source_run(monkeypatch, tmp_path, _OneSource(2))
