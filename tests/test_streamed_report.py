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
from collections.abc import Iterator
from pathlib import Path

# External
import pytest

# Local
from diagnostics_helpers import episode, signal, window_spec
from pydantic_core import to_json
from typer.testing import CliRunner
from upath import UPath

# Internal
from kalanos import api, publish
from kalanos.analysis import pipeline
from kalanos.analysis.models.compact_keys import decode_keys, encode_keys
from kalanos.analysis.models.diagnostics import DiagnosticPlan
from kalanos.analysis.models.legacy import load_any
from kalanos.analysis.models.report import AnalysedEpisode, Report
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


def test_a_diagnostics_plan_keeps_its_episodes():
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

    assert isinstance(result, Report)
    assert result.diagnostics is not None
    assert [e.id for e in result.episodes] == ["e"]


def _write_peak(tmp_path: Path, n: int, *, keep: bool) -> int:
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

    Returns
    -------
    int
        The traced peak, in bytes, from grading through writing.
        The recordings themselves were read before tracing started.
    """

    policy = load_default_policy()
    analysed = deque(
        AnalysedEpisode(
            episode=episode(signal("command"), signal(), identifier=f"e{k}"),
            adapter="csv",
            adapter_confidence=1,
            policy=policy,
        )
        for k in range(n)
    )
    path = tmp_path / f"{n}-{keep}.json"
    tracemalloc.start()
    try:
        result = assemble_report(
            root=UPath("fixture"), analysed=analysed, policy=policy, keep_episodes=keep
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
