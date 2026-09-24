"""Verifies the wheel actually installs and runs: the console script resolves,
and both files shipped outside `src/kalanos/*.py` survive packaging.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import shutil
import subprocess
import venv
from dataclasses import dataclass
from pathlib import Path

# External
import pytest

# Local
from helpers import CSV_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


REPO_ROOT = Path(__file__).parent.parent

# Building the wheel and resolving its dependencies from PyPI is what makes
# `integration` slow, so every test below shares one install rather than
# each paying that cost — see `installed_kalanos`.
pytestmark = pytest.mark.integration


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass
class ScaffoldedPlugin:
    """A package `kalanos new` wrote, installed beside the wheel that wrote it.

    Attributes
    ----------
    plugin_name : str
        The name the plugin registers under, as a listing prints it.
    dist_name : str
        The distribution and version a listing names as its source.
    directory : Path
        The generated package, for running its own test suite from.
    listing : str
        The subcommand that lists this plugin's group.
    """

    plugin_name: str
    dist_name: str
    directory: Path
    listing: str


@dataclass
class InstalledWheel:
    """A `kalanos` wheel installed into a throwaway venv, isolated from the checkout.

    Attributes
    ----------
    python : Path
        The venv's interpreter.
    console_script : Path
        The `kalanos` entry point `[project.scripts]` installed.
    working_directory : Path
        A directory outside the repository to run subprocesses from, so
        nothing under test can fall back to a path the checkout provides.
    """

    python: Path
    console_script: Path
    working_directory: Path


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(scope="session")
def installed_kalanos(tmp_path_factory) -> InstalledWheel:
    """Build the `kalanos` wheel once and install it into a clean venv.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Pytest's session-scoped temp directory factory.

    Returns
    -------
    InstalledWheel
        The venv's interpreter and console-script paths, plus a working
        directory outside the repository for tests to run from.
    """

    # `uv run pytest` cannot reach this fixture without uv on PATH already,
    # so a missing uv here would mean something broke the runner itself —
    # that must fail the job, not skip it quietly to a false green.
    uv = shutil.which("uv")
    assert uv is not None, "uv is not on PATH; cannot build the wheel for this test"

    # Step 1: build the wheel exactly as it would ship.
    dist_dir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(dist_dir), str(REPO_ROOT)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"

    # Step 2: install it into a venv that has never seen the source tree.
    # `with_pip=False` skips ensurepip entirely — `uv pip install` targets
    # the venv's interpreter directly and needs no pip inside it.
    venv_dir = tmp_path_factory.mktemp("venv")
    venv.create(venv_dir, with_pip=False)
    venv_python = venv_dir / "bin" / "python"

    subprocess.run(
        [uv, "pip", "install", "--quiet", "--python", str(venv_python), str(wheels[0])],
        check=True,
        capture_output=True,
        text=True,
    )

    return InstalledWheel(
        python=venv_python,
        console_script=venv_dir / "bin" / "kalanos",
        working_directory=tmp_path_factory.mktemp("cwd"),
    )


@pytest.fixture(scope="session")
def scaffolded_plugins(
    installed_kalanos, tmp_path_factory
) -> dict[str, ScaffoldedPlugin]:
    """Scaffold one plugin of each kind with the installed CLI, and install both.

    Parameters
    ----------
    installed_kalanos : InstalledWheel
        The wheel whose console script writes the packages and then lists them.
    tmp_path_factory : pytest.TempPathFactory
        Pytest's session-scoped temp directory factory.

    Returns
    -------
    dict[str, ScaffoldedPlugin]
        One entry per scaffold kind, keyed `"adapter"` and `"metric"`.
    """

    uv = shutil.which("uv")
    assert uv is not None, "uv is not on PATH; cannot install the scaffolded packages"

    into = tmp_path_factory.mktemp("scaffold")
    plugins = {
        "adapter": ScaffoldedPlugin(
            plugin_name="myformat",
            dist_name="kalanos-myformat 0.1.0",
            directory=into / "kalanos-myformat",
            listing="adapters",
        ),
        "metric": ScaffoldedPlugin(
            plugin_name="myjitter",
            dist_name="kalanos-myjitter 0.1.0",
            directory=into / "kalanos-myjitter",
            listing="metrics",
        ),
    }

    for kind, plugin in plugins.items():
        written = subprocess.run(
            [
                str(installed_kalanos.console_script),
                "new",
                kind,
                plugin.plugin_name,
                "--into",
                str(into),
            ],
            cwd=installed_kalanos.working_directory,
            capture_output=True,
            text=True,
        )
        assert written.returncode == 0, written.stderr
        assert plugin.directory.is_dir(), written.stdout

    # Two installs, because --no-deps is right for the generated packages and
    # wrong for pytest: kalanos and polars are already here, pluggy is not.
    for arguments in (
        ["pytest"],
        ["--no-deps", *(str(plugin.directory) for plugin in plugins.values())],
    ):
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--quiet",
                "--python",
                str(installed_kalanos.python),
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    return plugins


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_installed_console_script_runs(installed_kalanos):
    """Verify the `[project.scripts]` entry point resolves in a clean install.

    `uv run kalanos` and `python -m kalanos.cli` both work from a checkout
    regardless of what the entry point declares, so only running the
    installed script itself catches a wrong target.
    """

    result = subprocess.run(
        [str(installed_kalanos.console_script), "--help"],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "grade" in result.stdout


def test_the_default_policy_loads_from_the_wheel(installed_kalanos):
    """Verify the packaged policy loads with no source tree on the path at all.

    Catches a missing package-data include, or a loader regressed back to
    a `Path(__file__)` walk that only works from a checkout.
    """

    result = subprocess.run(
        [
            str(installed_kalanos.python),
            "-c",
            "from kalanos.assets.policy import load_default_policy\n"
            "policy = load_default_policy()\n"
            "assert policy.entry_for('drop_rate') is not None\n"
            "print('ok')",
        ],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_default_dictionary_loads_from_the_wheel(installed_kalanos):
    """Verify the packaged dictionary loads with no source tree on the path at all.

    Same failure mode as the policy check above: a missing package-data
    include, or a loader that only works from a checkout.
    """

    result = subprocess.run(
        [
            str(installed_kalanos.python),
            "-c",
            "from kalanos.assets.dictionary import load_default_dictionary\n"
            "dictionary = load_default_dictionary()\n"
            "assert len(dictionary.entries) == 202\n"
            "print('ok')",
        ],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_contract_suite_imports_from_the_wheel(installed_kalanos):
    """Verify `kalanos.testing` resolves with no source tree on the path at all.

    An out-of-tree plugin package imports the contract suite and injectors
    through this same path, since it has no checkout to import from.
    """

    result = subprocess.run(
        [
            str(installed_kalanos.python),
            "-c",
            "from kalanos.testing import check_metric, clean_recording\nprint('ok')",
        ],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_installed_cli_lists_the_lerobot_v3_adapter(installed_kalanos):
    """Verify the lerobot_v3 entry point is declared and loads from the wheel."""

    result = subprocess.run(
        [str(installed_kalanos.console_script), "adapters"],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "lerobot_v3" in result.stdout


def test_the_installed_cli_lists_hdf5_as_unavailable(installed_kalanos):
    """Verify a slim install without h5py reports the hdf5 entry point as FAILED.

    `installed_kalanos` installs the wheel with no extras, so the module's
    guarded `import h5py` raises on entry-point load. That failure has to
    reach this listing rather than crash the whole `adapters` command, and
    has to name the extra: this is the only test that sees the message a
    reader of a base install actually gets.
    """

    result = subprocess.run(
        [str(installed_kalanos.console_script), "adapters"],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    listed = [line for line in result.stdout.splitlines() if "hdf5" in line]
    assert len(listed) == 1, result.stdout
    assert "FAILED" in listed[0], result.stdout
    assert "kalanos[hdf5]" in listed[0], result.stdout
    assert "ImportError" not in listed[0], result.stdout


def test_the_installed_cli_lists_mcap_as_unavailable(installed_kalanos):
    """Verify a slim install without mcap reports the mcap entry point as FAILED.

    `installed_kalanos` installs the wheel with no extras, so the module's
    top-level `from mcap...` imports raise on entry-point load. That failure
    has to reach this listing rather than crash the whole `adapters` command.
    """

    result = subprocess.run(
        [str(installed_kalanos.console_script), "adapters"],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    listed = [line for line in result.stdout.splitlines() if "mcap" in line]
    assert len(listed) == 1, result.stdout
    assert "FAILED" in listed[0], result.stdout


def test_the_installed_cli_grades_a_recording_to_html(installed_kalanos):
    """Verify a full grade run renders HTML from the wheel's own template.

    `PackageLoader` only checks at import time that `templates/` exists —
    a missing directory already fails `test_the_installed_console_script_runs`
    above. What a real render catches, and nothing else does, is
    `report.html.j2` itself missing from an otherwise-present `templates/`.
    """

    report_path = installed_kalanos.working_directory / "report.html"

    result = subprocess.run(
        [
            str(installed_kalanos.console_script),
            "grade",
            str(CSV_FIXTURE),
            "--report",
            str(report_path),
        ],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert report_path.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("kind", ["adapter", "metric"])
def test_a_scaffolded_plugin_installs_and_is_listed(
    installed_kalanos, scaffolded_plugins, kind
):
    """Verify what `kalanos new` writes is found by the next listing, with no edits.

    This is the whole promise of the scaffold: an author installs what it wrote
    and Kalanos already knows about it. A wrong entry point in the generated
    `pyproject.toml` shows up here and nowhere else.
    """

    plugin = scaffolded_plugins[kind]

    result = subprocess.run(
        [str(installed_kalanos.console_script), plugin.listing],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    listed = [line for line in result.stdout.splitlines() if plugin.plugin_name in line]
    assert len(listed) == 1, result.stdout
    assert plugin.dist_name in listed[0]
    assert "FAILED" not in listed[0]


@pytest.mark.parametrize("kind", ["adapter", "metric"])
def test_a_scaffolded_plugins_own_test_passes_unedited(
    installed_kalanos, scaffolded_plugins, kind
):
    """Verify the generated contract test is green before its author changes anything.

    A scaffold whose test starts red teaches an author to ignore it, which
    defeats the reason it ships a contract test at all.
    """

    plugin = scaffolded_plugins[kind]

    result = subprocess.run(
        [
            str(installed_kalanos.python),
            "-m",
            "pytest",
            str(plugin.directory / "tests"),
        ],
        cwd=installed_kalanos.working_directory,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
