"""Checks the report-size benchmark script without grading or touching the network."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# External
import pytest
from rich.console import Console
from typer.testing import CliRunner


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "scripts/report_size"
spec = importlib.util.spec_from_file_location(
    "report_size", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
)
assert spec is not None and spec.loader is not None
sys.modules["report_size"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules["report_size"])
cli = importlib.import_module("report_size.__main__")
config = importlib.import_module("report_size.config")
grading = importlib.import_module("report_size.grading")
hub = importlib.import_module("report_size.hub")
table = importlib.import_module("report_size.table")
versions = importlib.import_module("report_size.versions")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _partial_path(command: list[str]) -> Path:
    """Return the path a grade command was told to write its report to.

    Parameters
    ----------
    command : list[str]
        The command passed to `subprocess.run`.

    Returns
    -------
    Path
        The argument after `--report`.
    """

    return Path(command[command.index("--report") + 1])


def _repo(path: Path) -> list[str]:
    """Create a git repository with one commit and return the git command prefix.

    Parameters
    ----------
    path : Path
        Where the repository goes.

    Returns
    -------
    list[str]
        `git -C <path>` with an identity, ready for more arguments.
    """

    path.mkdir()
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    return git


def _sha(git: list[str], ref: str) -> str:
    """Return the commit a ref points at.

    Parameters
    ----------
    git : list[str]
        The git command prefix from `_repo`.
    ref : str
        The ref.

    Returns
    -------
    str
        The full SHA.
    """

    return subprocess.run(
        [*git, "rev-parse", ref], check=True, capture_output=True, text=True
    ).stdout.strip()


def _comparison():
    """Return a comparison with a smaller, an equal and a mismatched dataset.

    Returns
    -------
    Comparison
        pusht halves its bytes per episode, utokyo is unchanged, and usc_cloth_sim
        loads 8 episodes instead of 10 at the same bytes per episode.
    """

    before = versions.Version(label="53087a3", key="53087a3d" * 5, source="")
    after = versions.Version(label="HEAD", key="abcdef01" * 5, source="")
    pusht, utokyo, cloth = config.DATASETS
    rows = [
        (pusht, grading.Measurement(1000, 10), grading.Measurement(500, 10)),
        (utokyo, grading.Measurement(1000, 10), grading.Measurement(1000, 10)),
        (cloth, grading.Measurement(1000, 10), grading.Measurement(800, 8)),
    ]
    return table.compare(before, after, rows)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_resolve_version_commit_and_wheel(tmp_path):
    """A ref resolves to its commit SHA, a wheel to its content hash."""

    repo = tmp_path / "repo"
    sha = _sha(_repo(repo), "HEAD")

    commit = versions.resolve_version("HEAD", repo)
    assert (commit.label, commit.key, commit.source) == ("HEAD", sha, sha)

    wheel = tmp_path / "x-0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    built = versions.resolve_version(str(wheel), repo)
    assert built.is_wheel and len(built.key) == len("wheel-") + 12
    assert built.source == str(wheel.resolve())

    with pytest.raises(ValueError, match="no-such-ref"):
        versions.resolve_version("no-such-ref", repo)
    with pytest.raises(ValueError, match="missing.whl"):
        versions.resolve_version(str(tmp_path / "missing.whl"), repo)


def test_merge_base_is_where_the_branch_started(tmp_path):
    """The default before side is the commit the branch left its base at."""

    repo = tmp_path / "repo"
    git = _repo(repo)
    start = _sha(git, "HEAD")
    subprocess.run([*git, "branch", "-q", "base"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "branch"], check=True)
    subprocess.run([*git, "checkout", "-q", "base"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True)
    subprocess.run([*git, "checkout", "-q", "-"], check=True)

    after = versions.resolve_version("HEAD", repo)
    found = versions.merge_base(after, "base", repo)
    assert found.key == start
    assert found.label == "merge-base of HEAD and base"

    with pytest.raises(ValueError, match="no-such-base"):
        versions.merge_base(after, "no-such-base", repo)


def test_pin_dataset_keeps_a_commit_and_resolves_the_rest(monkeypatch):
    """A 40-hex revision is kept; a branch or no revision becomes the Hub's commit."""

    asked = []

    class FakeApi:
        def dataset_info(self, repo_id, revision=None):
            asked.append((repo_id, revision))
            return SimpleNamespace(sha="f" * 40)

    monkeypatch.setattr(hub, "HfApi", FakeApi)

    pinned = f"lerobot/pusht@{'a' * 40}"
    assert hub.pin_dataset(pinned) == pinned
    assert hub.pin_dataset("lerobot/pusht") == f"lerobot/pusht@{'f' * 40}"
    assert hub.pin_dataset("lerobot/pusht@main") == f"lerobot/pusht@{'f' * 40}"
    assert asked == [("lerobot/pusht", None), ("lerobot/pusht", "main")]


def test_grade_accepts_gate_exit_and_rejects_failure(tmp_path, monkeypatch):
    """Exit 1 keeps the report; exit 2 raises and leaves no file behind."""

    report = tmp_path / "pusht.json"

    def fake_run(code):
        def run(command, **kwargs):
            _partial_path(command).write_text("{}")
            return subprocess.CompletedProcess(command, code, stderr="boom\n")

        return run

    monkeypatch.setattr(grading.subprocess, "run", fake_run(1))
    grading.grade("kalanos", "lerobot/pusht@abc", report)
    assert report.exists()

    report.unlink()
    monkeypatch.setattr(grading.subprocess, "run", fake_run(2))
    with pytest.raises(grading.GradeFailed, match="boom"):
        grading.grade("kalanos", "lerobot/pusht@abc", report)
    assert list(tmp_path.iterdir()) == []


def test_rerun_grades_only_the_new_side(tmp_path, monkeypatch):
    """A version and dataset already measured is reused."""

    calls = []

    def fake_grade(install, dataset, report):
        calls.append(install)
        body = json.dumps({"inventory": {"loaded": 10}})
        report.write_text(body + " " * (1000 - len(body)))

    monkeypatch.setattr(grading, "grade", fake_grade)
    monkeypatch.setattr(
        grading, "requirement", lambda version, repo, cache: version.key
    )

    def version(key):
        return versions.Version(label=key, key=key, source=key)

    def measure_all(*sides):
        for side in sides:
            for dataset in config.DATASETS:
                measured = grading.ensure_measurement(side, dataset, ROOT, tmp_path)
                assert measured == grading.Measurement(bytes=1000, episodes=10)

    measure_all(version("a"), version("b"))
    assert len(calls) == 6
    measure_all(version("a"), version("c"))
    assert calls[6:] == ["c", "c", "c"]


def test_options_key_follows_bundle(tmp_path, monkeypatch):
    """Editing the bundle gives a new cache key."""

    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("vision:\n  preview_size: 192\n")
    second.write_text("vision:\n  preview_size: 96\n")
    monkeypatch.setattr(config, "BUNDLE", first)
    key = grading.options_key()
    monkeypatch.setattr(config, "BUNDLE", second)
    assert grading.options_key() != key


def test_markdown_names_versions_options_and_change():
    """The Markdown table names both versions and the options, and shows each change."""

    lines = table.to_markdown(_comparison()).splitlines()

    assert lines[:3] == [
        "Before: 53087a3 (53087a3d5308)",
        "After: HEAD (abcdef01abcd)",
        "Grade options: --tier full --profile scripts/report_size/profile.yaml",
    ]
    assert lines[6:] == [
        "| lerobot/pusht@7628202 | 10 | 1,000 | 500 | 100 | 50 | -50.0% |",
        "| lerobot/utokyo_xarm_bimanual@13533b9 | 10 | 1,000 | 1,000 | 100 | 100 "
        "| +0.0% |",
        "| lerobot/usc_cloth_sim@6c4fb6a | 10 → 8 | 1,000 | 800 | 100 | 100 | +0.0% |",
        "| Mean |  | 1,000 | 767 | 100 | 83 | -16.7% |",
        "",
        "The two versions loaded a different number of episodes for "
        "lerobot/usc_cloth_sim.",
    ]


def test_terminal_render_keeps_every_number_whole_on_a_narrow_terminal():
    """At 80 columns every cell and the note show, without pipes or cut numbers."""

    before = versions.Version(label="main", key="d" * 40, source="")
    after = versions.Version(label="HEAD", key="e" * 40, source="")
    pusht, utokyo, cloth = config.DATASETS
    rows = [
        (
            pusht,
            grading.Measurement(61_225_507, 206),
            grading.Measurement(48_489_170, 206),
        ),
        (
            utokyo,
            grading.Measurement(63_945_758, 70),
            grading.Measurement(16_663_922, 70),
        ),
        (
            cloth,
            grading.Measurement(493_045_422, 1000),
            grading.Measurement(315_241_906, 990),
        ),
    ]
    comparison = table.compare(before, after, rows)
    console = Console(width=80, record=True)
    console.print(table.to_rich(comparison, 80))
    text = console.export_text()

    assert "Before: main (dddddddddddd)" in text
    for cells in [*comparison.rows, comparison.mean]:
        for cell in cells[1:]:
            assert cell in text
    assert "different number of episodes" in text
    assert "|" not in text and "…" not in text


def test_terminal_render_colours_the_change_by_sign():
    """A smaller report per episode is green, a larger one red, no change plain."""

    before = versions.Version(label="a", key="a" * 40, source="")
    after = versions.Version(label="b", key="b" * 40, source="")
    pusht, utokyo, cloth = config.DATASETS
    rows = [
        (pusht, grading.Measurement(1000, 10), grading.Measurement(500, 10)),
        (utokyo, grading.Measurement(1000, 10), grading.Measurement(1000, 10)),
        (cloth, grading.Measurement(1000, 10), grading.Measurement(2000, 10)),
    ]
    console = Console(width=120, force_terminal=True, color_system="standard")
    with console.capture() as capture:
        console.print(table.to_rich(table.compare(before, after, rows), 120))
    out = capture.get()

    # Right-justified padding sits inside the colour code.
    assert re.search(r"\x1b\[32m *-50\.0%", out)
    assert re.search(r"\x1b\[31m *\+100\.0%", out)
    assert re.search(r" \+0\.0% ", out)


def test_measure_rejects_a_report_with_no_episodes(tmp_path):
    """A grade that loaded nothing is an error, so it is never cached."""

    report = tmp_path / "empty.json"
    report.write_text(json.dumps({"inventory": {"loaded": 0}}))
    with pytest.raises(ValueError, match="loaded no episodes"):
        grading.measure(report)


def test_cli_reports_an_unknown_version():
    """A ref that doesn't resolve exits 1 with a message and grades nothing."""

    result = CliRunner().invoke(
        cli.app, ["--after", "no-such-ref", "--dataset", config.DATASETS[0]]
    )
    assert result.exit_code == 1
    assert "report_size: no-such-ref" in result.output


@pytest.mark.parametrize("terminal", [False, True])
def test_cli_writes_markdown_to_output_whatever_stdout_is(
    tmp_path, monkeypatch, terminal
):
    """--output gets Markdown; stdout gets the same, or a table on a terminal."""

    monkeypatch.setattr(cli, "pin_dataset", lambda spec: spec)
    monkeypatch.setattr(
        cli,
        "ensure_measurement",
        lambda version, dataset, repo, cache: grading.Measurement(1000, 10),
    )
    stdout = SimpleNamespace(isatty=lambda: terminal)
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdout=stdout, stderr=sys.stderr))
    output = tmp_path / "sizes.md"

    result = CliRunner().invoke(
        cli.app,
        ["--before", "HEAD", "--dataset", config.DATASETS[0], "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert "| lerobot/pusht@7628202 | 10 |" in output.read_text()
    if terminal:
        assert "|" not in result.stdout and "lerobot/pusht@7628202" in result.stdout
    else:
        assert output.read_text() == result.stdout
