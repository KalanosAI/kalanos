"""Guards the packaging invariants in pyproject.toml that no runtime test reaches.

Extras, base dependencies and entry points are declarative data:
nothing that runs a pipeline reads them, so a drift here is invisible
until an install fails or a plugin goes missing.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import ast
import importlib
import re
from pathlib import Path

# External
import pytest


try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # pyright: ignore[reportMissingImports]


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
ANALYSIS_ROOT = SRC_ROOT / "kalanos" / "analysis"

# Each entry-point group, and the decorator whose presence in a module means
# that module has something to register through it.
REGISTERING_DECORATORS = {
    "kalanos.adapters": "adapter",
    "kalanos.metrics": "metric",
    "kalanos.reporters": "reporter",
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _pyproject() -> dict:
    """Parse pyproject.toml at the repo root."""

    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _distribution_name(requirement: str) -> str:
    """Normalise a PEP 508 requirement string to its bare distribution name.

    Parameters
    ----------
    requirement : str
        A requirement string, e.g. ``"av>=13"`` or
        ``"tomli>=2 ; python_full_version < '3.11'"``.

    Returns
    -------
    str
        The distribution name, lowercased with ``_``/``.`` normalised to
        ``-`` per PEP 503.
    """

    name = re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].strip()
    return re.sub(r"[-_.]+", "-", name).lower()


def _entry_points(group: str) -> dict[str, str]:
    """Read one entry-point group from pyproject.toml.

    Parameters
    ----------
    group : str
        The entry-point group name, e.g. ``"kalanos.adapters"``.

    Returns
    -------
    dict[str, str]
        Mapping of entry-point name to its target string.
    """

    return _pyproject()["project"].get("entry-points", {}).get(group, {})


def _declared_modules(group: str) -> set[str]:
    """Collect the dotted module names declared for an entry-point group.

    Parameters
    ----------
    group : str
        The entry-point group name.

    Returns
    -------
    set[str]
        Each target's dotted module, with any ``:attr`` suffix dropped.
    """

    return {target.split(":")[0] for target in _entry_points(group).values()}


def _module_name(path: Path) -> str:
    """Convert a source file path to its dotted ``kalanos.…`` module name.

    Parameters
    ----------
    path : Path
        Path to a Python file under ``src/kalanos``.

    Returns
    -------
    str
        The dotted module name.
    """

    relative = path.relative_to(REPO_ROOT / "src").with_suffix("")
    return ".".join(relative.parts)


def _named_targets(source: Path) -> list[tuple[str, str | None]]:
    """Collect what every `MissingDependency` in a module tells the reader to install.

    Parameters
    ----------
    source : Path
        Path to the Python file to parse.

    Returns
    -------
    list of tuple of (str, str or None)
        One `(distribution, extra)` pair per call whose arguments are string
        literals, which is the only form the guards use. The extra is `None`
        where the call named a distribution alone.
    """

    tree = ast.parse(source.read_text(), filename=str(source))
    named: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "MissingDependency" or not node.args:
            continue
        literals = [
            argument.value if isinstance(argument, ast.Constant) else None
            for argument in node.args
        ]
        if not isinstance(literals[0], str):
            continue
        extra = literals[1] if len(literals) > 1 else None
        named.append((literals[0], extra if isinstance(extra, str) else None))

    return named


def _decorator_names(source: Path) -> set[str]:
    """Collect the name of every decorator applied to a def or class in a module.

    Parameters
    ----------
    source : Path
        Path to the Python file to parse.

    Returns
    -------
    set[str]
        Bare decorator names, e.g. ``{"adapter"}`` for ``@adapter`` or
        ``@adapter(...)``.
    """

    tree = ast.parse(source.read_text(), filename=str(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                names.add(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Name
            ):
                names.add(decorator.func.id)
    return names


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_no_optional_dependency_appears_in_base_dependencies():
    """Verify no extra's library is also required by the base install.

    Python packaging gives no way to opt out of a base dependency, so a
    decoder placed there is one a CI job grading CSVs has to take.
    """

    project = _pyproject()["project"]
    base = {_distribution_name(req) for req in project["dependencies"]}

    optional = set()
    for extra, requirements in project["optional-dependencies"].items():
        if extra == "all":
            continue
        optional.update(_distribution_name(req) for req in requirements)

    overlap = base & optional
    assert not overlap, (
        f"base dependencies also declared in an extra: {overlap} — a slim "
        "install cannot decline a library that lives in base"
    )


def test_all_composes_every_other_extra():
    """Verify the `all` extra self-references exactly the other declared extras."""

    optional = _pyproject()["project"]["optional-dependencies"]
    extras = {_distribution_name(name) for name in optional if name != "all"}

    composed = set()
    for entry in optional["all"]:
        match = re.fullmatch(r"kalanos\[([\w.-]+)\]", entry)
        assert match, f"`all` entry {entry!r} is not a kalanos[<extra>] self-reference"
        composed.add(_distribution_name(match.group(1)))

    missing = extras - composed
    extra_names = composed - extras
    assert not missing and not extra_names, (
        f"`all` is out of step with the declared extras: missing={missing}, "
        f"no-longer-exists={extra_names}"
    )


@pytest.mark.parametrize("group", ["kalanos.metrics", "kalanos.reporters"])
def test_every_declared_entry_point_module_imports(group):
    """Verify every declared target in a metrics/reporters entry-point group imports.

    Adapters are covered by
    `tests/test_adapters.py::test_declared_entry_points_resolve`, which
    additionally checks the loaded target satisfies `Adapter`. Metrics and
    reporters name a module rather than a callable, so importing it — which
    runs the decorations inside — is the whole check.
    """

    for target in _entry_points(group).values():
        importlib.import_module(target.split(":")[0])


@pytest.mark.parametrize("group", REGISTERING_DECORATORS.keys())
def test_every_registering_module_is_declared(group):
    """Verify every built-in module using a group's decorator is declared for it.

    A built-in that registers nowhere ships invisible to `kalanos adapters`,
    `kalanos metrics` or `kalanos reporters`.
    """

    decorator = REGISTERING_DECORATORS[group]
    declared = _declared_modules(group)

    undeclared = []
    for path in ANALYSIS_ROOT.rglob("*.py"):
        if decorator in _decorator_names(path):
            module = _module_name(path)
            if module not in declared:
                undeclared.append(module)

    assert not undeclared, (
        f"modules decorated with @{decorator} but not declared in "
        f'[project.entry-points."{group}"]: {undeclared}'
    )


def test_every_extra_a_plugin_names_is_declared():
    """Verify every `MissingDependency` in the source names an extra that exists.

    The extra is a literal at the import it guards, so nothing but this
    comparison keeps it honest: an extra that was renamed here sends the
    reader to an install that fails, which is worse than the raw import error.

    A third-party plugin may name a distribution with no extra at all, but
    every optional dependency this repo ships is behind one of its own extras,
    so a bare distribution here is a mistake rather than that case.
    """

    optional = _pyproject()["project"]["optional-dependencies"]

    named = [
        (path, distribution, extra)
        for path in SRC_ROOT.rglob("*.py")
        for distribution, extra in _named_targets(path)
    ]
    assert named, "no plugin names an extra — the guards have gone missing"

    for path, distribution, extra in named:
        assert distribution == "kalanos", (
            f"{path.name} names {distribution!r}, which this repo does not publish"
        )
        assert extra is not None, (
            f"{path.name} names kalanos with no extra, but every optional "
            "dependency here is behind one"
        )
        assert extra in optional, (
            f"{path.name} names kalanos[{extra}], which pyproject.toml does not declare"
        )
