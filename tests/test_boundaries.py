"""Enforces the load-bearing boundaries from docs/ARCHITECTURE.md.

Retrofitting these later is expensive, so they are pinned as tests
before any pipeline code exists.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import ast
from pathlib import Path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


PACKAGE_ROOT = Path(__file__).parent.parent / "src" / "kalanos"
ANALYSIS_ROOT = PACKAGE_ROOT / "analysis"

# A stage may import models/ and stages before it, never after — and models/
# itself is the contract every stage agrees on, so it may not import any of them.
# Order matters here: it is also the ordering the boundary test below checks.
STAGE_ORDER = [
    "discovery",
    "metrics",
    "scoring",
    "reporting",
]
STAGE_PACKAGES = set(STAGE_ORDER)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


# Method names that write regardless of any mode argument —
# a Path write method, or a polars DataFrame writer, since a future metric # or fixture
# helper reaching for df.write_csv would otherwise write files invisibly to this check.
_ALWAYS_WRITE_METHODS = {
    "write_text",
    "write_bytes",
    "mkdir",
    "touch",
    "write_csv",
    "write_parquet",
    "write_ndjson",
    "write_json",
}


def _open_mode(node: ast.Call, *, mode_index: int) -> str | None:
    """Extract an `open()`-shaped call's mode argument, if it's a literal string.

    Parameters
    ----------
    node : ast.Call
        A call already known to be `open(...)` or `path.open(...)`.
    mode_index : int
        Where the mode falls in `node.args` — `1` for the builtin
        `open(file, mode, ...)`, `0` for `Path.open(mode='r', ...)`,
        since only the builtin also takes the file itself as an argument.

    Returns
    -------
    str or None
        The mode string, or `None` when no mode was given or it isn't a
        literal this check can read — a non-literal mode (built from a
        variable) is not seen in this codebase today.
    """

    mode_node = node.args[mode_index] if len(node.args) > mode_index else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode_node = keyword.value
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return mode_node.value
    return None


def _writes_files(source: Path) -> bool:
    """Detect whether a module performs write-mode file access.

    Parameters
    ----------
    source : Path
        Path to the Python file to parse.

    Returns
    -------
    bool
        `True` if the module calls a write method, or opens a file with a
        mode string containing `w`, `a`, `x` or `+` — through either the
        builtin `open(...)` or `Path.open(...)`, since both reach the
        filesystem the same way.
    """

    tree = ast.parse(source.read_text(), filename=str(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _ALWAYS_WRITE_METHODS:
            return True

        if isinstance(func, ast.Name) and func.id == "open":
            mode = _open_mode(node, mode_index=1)
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            mode = _open_mode(node, mode_index=0)
        else:
            continue

        if isinstance(mode, str) and any(letter in mode for letter in "wax+"):
            return True

    return False


def _constructs(source: Path, name: str) -> bool:
    """Detect whether a module calls a constructor of the given bare name.

    Catches both `Name(...)` and a dotted `module.Name(...)`, since a metric
    module could import the class either way.

    Parameters
    ----------
    source : Path
        Path to the Python file to parse.
    name : str
        The bare class name to look for at a call site.

    Returns
    -------
    bool
        `True` if the module contains a call whose callee is `name`.
    """

    tree = ast.parse(source.read_text(), filename=str(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False


def _imported_stage_packages(source: Path) -> set[str]:
    """Collect which ``analysis.<stage>`` packages a source file imports from.

    Looks one level below the top-level import, since a root of ``kalanos``
    does not by itself say whether a stage package was reached.

    Parameters
    ----------
    source : Path
        Path to the Python file to parse.

    Returns
    -------
    set[str]
        Stage package names (e.g. ``"discovery"``) imported by this file.
    """

    tree = ast.parse(source.read_text(), filename=str(source))
    hits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
            if len(parts) >= 3 and parts[0] == "kalanos" and parts[1] == "analysis":
                if parts[2] in STAGE_PACKAGES:
                    hits.add(parts[2])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 3 and parts[0] == "kalanos" and parts[1] == "analysis":
                    if parts[2] in STAGE_PACKAGES:
                        hits.add(parts[2])
    return hits


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_models_import_no_pipeline_stage():
    """Verify no module under ``kalanos/analysis/models`` imports a stage package."""

    models_files = list((ANALYSIS_ROOT / "models").rglob("*.py"))
    assert models_files, "expected at least one module under kalanos/analysis/models/"

    offenders = {}
    for path in models_files:
        hit = _imported_stage_packages(path)
        if hit:
            offenders[path.relative_to(ANALYSIS_ROOT)] = hit

    assert not offenders, f"pipeline-stage imports found in models/: {offenders}"


def test_the_inference_library_imports_no_pipeline_stage():
    """Verify no module under ``kalanos/analysis/inference`` imports a stage package.

    `inference` is a library any stage or adapter may call, not a step in
    STAGE_ORDER. A module here reaching into a stage package is the library
    turning back into a stage — the same drift `test_models_import_no_pipeline_stage`
    catches for `models/`.
    """

    inference_files = list((ANALYSIS_ROOT / "inference").rglob("*.py"))
    assert inference_files, "expected at least one module under analysis/inference/"

    offenders = {}
    for path in inference_files:
        hit = _imported_stage_packages(path)
        if hit:
            offenders[path.relative_to(ANALYSIS_ROOT)] = hit

    assert not offenders, f"pipeline-stage imports found in inference/: {offenders}"


def test_the_adapters_library_imports_no_pipeline_stage():
    """Verify no module under ``kalanos/analysis/adapters`` imports a stage package.

    `adapters` is a library any stage may call, not a step in STAGE_ORDER. A
    module here reaching into a stage package is the same drift
    `test_the_inference_library_imports_no_pipeline_stage` catches for `inference/`.
    """

    adapters_files = list((ANALYSIS_ROOT / "adapters").rglob("*.py"))
    assert adapters_files, "expected at least one module under analysis/adapters/"

    offenders = {}
    for path in adapters_files:
        hit = _imported_stage_packages(path)
        if hit:
            offenders[path.relative_to(ANALYSIS_ROOT)] = hit

    assert not offenders, f"pipeline-stage imports found in adapters/: {offenders}"


def test_a_stage_never_imports_a_later_stage():
    """Verify every stage package only imports stages that precede it in STAGE_ORDER."""

    offenders = {}
    for index, stage in enumerate(STAGE_ORDER):
        later_stages = set(STAGE_ORDER[index + 1 :])
        stage_files = list((ANALYSIS_ROOT / stage).rglob("*.py"))
        # A missing or renamed stage directory would otherwise pass this
        # check vacuously — nothing to import, nothing to catch.
        assert stage_files, f"expected at least one module under analysis/{stage}/"
        for path in stage_files:
            hit = _imported_stage_packages(path) & later_stages
            if hit:
                offenders[path.relative_to(ANALYSIS_ROOT)] = hit

    assert not offenders, f"forward stage imports found: {offenders}"


def test_reporting_is_the_only_module_that_writes_files():
    """Verify only `analysis/reporting/` opens a file for writing.

    `docs/ARCHITECTURE.md`: "reporting is the only stage that opens a file for writing"
    — the reason a future SQLite backing store stays a local change is that nothing
    else in the pipeline touches storage at all.
    """

    offenders = [
        path.relative_to(ANALYSIS_ROOT)
        for path in ANALYSIS_ROOT.rglob("*.py")
        if path.relative_to(ANALYSIS_ROOT).parts[0] != "reporting"
        and _writes_files(path)
    ]

    assert not offenders, f"modules writing files outside reporting/: {offenders}"


def test_pipeline_is_the_only_module_that_reaches_across_stages():
    """Verify `analysis/pipeline.py` is the sole module spanning more than one stage.

    `pipeline.py` sits beside the stage packages rather than inside one so it
    can import several of them, which is otherwise exactly what STAGE_ORDER
    forbids. A second orchestrator drifting in elsewhere would defeat the
    ordering rule this file exists to enforce, so this pins the exception to
    stay singular.
    """

    pipeline_path = ANALYSIS_ROOT / "pipeline.py"
    assert pipeline_path.is_file(), "expected analysis/pipeline.py to exist"
    assert len(_imported_stage_packages(pipeline_path)) > 1

    offenders = {}
    for path in ANALYSIS_ROOT.rglob("*.py"):
        if path == pipeline_path or path.relative_to(ANALYSIS_ROOT).parts[0] in (
            "models",
            *STAGE_PACKAGES,
        ):
            continue
        hit = _imported_stage_packages(path)
        if len(hit) > 1:
            offenders[path.relative_to(ANALYSIS_ROOT)] = hit

    assert not offenders, f"non-stage modules reaching across stages: {offenders}"


def test_no_metric_module_constructs_a_finding():
    """Verify severity stays a scoring-only decision.

    `docs/ARCHITECTURE.md`: severity is assigned by the policy, in scoring.
    A metric module constructing a `Finding` directly would mean it had
    already decided how bad its own result was, before the policy ran.
    """

    offenders = [
        path.relative_to(ANALYSIS_ROOT)
        for path in (ANALYSIS_ROOT / "metrics").rglob("*.py")
        if _constructs(path, "Finding")
    ]

    assert not offenders, f"modules under metrics/ constructing a Finding: {offenders}"
