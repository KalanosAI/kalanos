"""Verifies the CLI's verbosity option and the logging module it configures.

`kalanos grade` must print exactly what it prints today on stdout regardless
of verbosity — a caller piping `--json` onward cannot have log records
land in the middle of the model.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import ast
import logging
from pathlib import Path

# External
from typer.testing import CliRunner

# Internal
from kalanos.analysis import pipeline
from kalanos.analysis.models.report import Report
from kalanos.assets.policy import load_policy
from kalanos.cli import app

# Local
from helpers import CSV_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


runner = CliRunner()

PACKAGE_ROOT = Path(__file__).parent.parent / "src" / "kalanos"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _getlogger_calls(source: Path) -> list[ast.Call]:
    """Collect every `logging.getLogger(...)` call site in a module.

    Parameters
    ----------
    source : Path
        Python file to parse.

    Returns
    -------
    list[ast.Call]
        Every call node whose callee is `getLogger`, dotted or bare.
    """

    tree = ast.parse(source.read_text(), filename=str(source))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "getLogger":
            calls.append(node)
        elif isinstance(func, ast.Name) and func.id == "getLogger":
            calls.append(node)
    return calls


def _calls_any(source: Path, names: set[str]) -> bool:
    """Detect whether a module calls any of the given attribute names.

    Parameters
    ----------
    source : Path
        Python file to parse.
    names : set[str]
        Attribute names to look for at a call site, e.g. `{"addHandler"}`.

    Returns
    -------
    bool
        `True` if any call in the module targets one of `names`.
    """

    tree = ast.parse(source.read_text(), filename=str(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in names:
            return True
        if isinstance(func, ast.Name) and func.id in names:
            return True
    return False


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_default_verbosity_leaves_stdout_to_the_report_card():
    """Verify the default verbosity leaves stdout byte-identical to the card."""

    result = runner.invoke(app, ["grade", str(CSV_FIXTURE)])

    assert result.exit_code == 0
    assert "OVERALL" in result.stdout
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        assert level not in result.stdout


def test_the_default_verbosity_emits_no_debug_records():
    """Verify WARNING (the default) does not let a DEBUG record through."""

    result = runner.invoke(app, ["grade", str(CSV_FIXTURE)])

    assert result.exit_code == 0
    assert "DEBUG" not in result.stderr


def test_debug_verbosity_writes_the_inference_trace_to_stderr():
    """Verify --verbosity debug surfaces inference's DEBUG records on stderr."""

    result = runner.invoke(app, ["--verbosity", "debug", "grade", str(CSV_FIXTURE)])

    assert result.exit_code == 0
    assert "OVERALL" in result.stdout
    assert "kalanos.analysis.inference" in result.stderr
    assert "DEBUG" in result.stderr


def test_json_stdout_stays_parseable_at_debug_verbosity():
    """Verify --json stays parseable even with every log record enabled.

    A caller piping `--json` onward would find a stray record on stdout
    the same way this test does: the model fails to parse.
    """

    result = runner.invoke(
        app, ["--verbosity", "debug", "grade", str(CSV_FIXTURE), "--json"]
    )

    assert result.exit_code == 0
    report = Report.model_validate_json(result.stdout)
    assert report.episodes


def test_an_unresolved_source_is_logged_as_a_warning(tmp_path, caplog):
    """Verify pipeline.run logs an unresolved file as a WARNING, with no handler.

    Calling the pipeline directly, rather than through the CLI, is
    deliberate: it proves the library logs without any handler being
    configured, and that `propagate` was left at its default.
    """

    good = tmp_path / "arm_multi_device.csv"
    good.write_bytes(CSV_FIXTURE.read_bytes())
    (tmp_path / "ragged.csv").write_text("a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n")

    with caplog.at_level(logging.DEBUG, logger="kalanos"):
        pipeline.run(tmp_path, policy=load_policy(None))

    # Other files in the run may log warnings of their own,
    # so filter to the one naming ragged.csv rather than counting all of them.
    ragged_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "ragged.csv" in record.getMessage()
    ]
    assert len(ragged_warnings) == 1


def test_every_module_logger_is_named_for_its_module():
    """Verify every `logging.getLogger(...)` call under src/kalanos passes `__name__`.

    A bare `getLogger()` — the root logger — fails this too: no module may
    configure the root, only its own named logger. `core/log.py` is
    excluded: it deliberately names the `"kalanos"` hierarchy root that every
    other module's `__name__` logger nests under, rather than a module logger
    of its own.
    """

    offenders: dict[Path, list[str]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path == PACKAGE_ROOT / "core" / "log.py":
            continue
        bad = []
        for call in _getlogger_calls(path):
            if len(call.args) != 1 or not (
                isinstance(call.args[0], ast.Name) and call.args[0].id == "__name__"
            ):
                bad.append(ast.dump(call))
        if bad:
            offenders[path.relative_to(PACKAGE_ROOT)] = bad

    assert not offenders, f"getLogger call(s) not passed __name__: {offenders}"


def test_no_library_module_configures_logging():
    """Verify only `kalanos.core.log` attaches a handler or sets a level.

    `cli.py` passes because it delegates to `configure_logging`
    rather than touching `logging` itself.
    """

    configuring_calls = {"basicConfig", "addHandler", "setLevel"}
    offenders = [
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        if path != PACKAGE_ROOT / "core" / "log.py"
        and _calls_any(path, configuring_calls)
    ]

    assert not offenders, (
        f"module(s) configuring logging outside core/log.py: {offenders}"
    )
