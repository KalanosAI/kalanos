"""Checks the report-size benchmark script without grading or touching the network."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# External
import pytest
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


def test_table_names_versions_options_and_change():
    """The table names both versions and the options, and shows each change."""

    before = versions.Version(label="53087a3", key="53087a3d" * 5, source="")
    after = versions.Version(label="HEAD", key="abcdef01" * 5, source="")
    pusht, utokyo, cloth = config.DATASETS
    rows = [
        (pusht, grading.Measurement(1000, 10), grading.Measurement(500, 10)),
        (utokyo, grading.Measurement(1000, 10), grading.Measurement(1000, 10)),
        (cloth, grading.Measurement(1000, 10), grading.Measurement(800, 8)),
    ]
    lines = table.render_table(before, after, rows).splitlines()

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
