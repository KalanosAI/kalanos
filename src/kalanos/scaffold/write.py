"""Write a publishable adapter or metric package from the bundled templates.

Both scaffolds produce something that installs, is discovered, and passes its
own contract test before a single line is edited, so an author's first run is green
and their first edit is the part that is actually theirs.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import re
from pathlib import Path

# External
from jinja2 import Environment, PackageLoader, StrictUndefined

# Internal
from kalanos.analysis.models.metrics import Family


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The generated package name has to be both a distribution name and a Python
# module name, which leaves the intersection of the two: lowercase, no dashes.
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# PackageLoader resolves through the package's own loader,
# so this works whether `kalanos` is an editable checkout or unzipped from a wheel.
_TEMPLATE_ENVIRONMENT = Environment(
    loader=PackageLoader("kalanos.scaffold", "templates"),
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)

# The path the generated adapter must decline. Prose rather than a table,
# so nothing about it could be mistaken for a recording.
_DECOY = """Not a recording. The adapter must bid zero on this file and either
yield nothing or raise AdapterRefusal when asked to read it anyway.
"""


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _prepare(name: str, into: Path) -> tuple[dict[str, str], Path]:
    """Validate a plugin name and claim the directory its package will fill.

    Parameters
    ----------
    name : str
        The plugin's name, lowercased before it is checked.
    into : Path
        The parent directory to write the package into.

    Returns
    -------
    tuple[dict[str, str], Path]
        The template context, and the directory to write into.

    Raises
    ------
    ValueError
        If `name` is not a lowercase identifier.
    FileExistsError
        If the target directory already exists.
    """

    name = name.lower()
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            f"{name!r} is not a usable plugin name; expected a lowercase name "
            "starting with a letter, made of letters, digits and underscores"
        )

    context = {
        "name": name,
        "dist_name": f"kalanos-{name.replace('_', '-')}",
        "module_name": f"kalanos_{name}",
        "suffix": f".{name}",
    }

    destination = into / context["dist_name"]
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; nothing was written")

    return context, destination


def _render(kind: str, template: str, destination: Path, context: dict) -> None:
    """Render one template into a file, creating the directories above it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = _TEMPLATE_ENVIRONMENT.get_template(f"{kind}/{template}").render(context)
    destination.write_text(rendered, encoding="utf-8")


def _write_common(kind: str, context: dict, destination: Path) -> None:
    """Render the four files both scaffolds share."""

    module = destination / "src" / context["module_name"]
    _render(kind, "pyproject.toml.j2", destination / "pyproject.toml", context)
    _render(kind, "README.md.j2", destination / "README.md", context)
    _render(kind, "package.py.j2", module / "__init__.py", context)
    _render(
        kind, "test_contract.py.j2", destination / "tests" / "test_contract.py", context
    )


def scaffold_adapter(name: str, into: Path) -> Path:
    """Write a publishable adapter package, fixtures and contract test included.

    Parameters
    ----------
    name : str
        The adapter's name. Also its file suffix, its module name and half
        its distribution name.
    into : Path
        The parent directory to write the package into.

    Returns
    -------
    Path
        The directory that was written.

    Raises
    ------
    ValueError
        If `name` is not a lowercase identifier.
    FileExistsError
        If the target directory already exists.
    """

    context, destination = _prepare(name, into)
    context["class_name"] = f"{name.replace('_', ' ').title().replace(' ', '')}Adapter"

    _write_common("adapter", context, destination)

    fixtures = destination / "tests" / "fixtures"
    _render(
        "adapter", "sample.csv.j2", fixtures / f"sample{context['suffix']}", context
    )
    (fixtures / "decoy.txt").write_text(_DECOY, encoding="utf-8")

    return destination


def scaffold_metric(name: str, family: Family, into: Path) -> Path:
    """Write a publishable metric package, contract test included.

    Parameters
    ----------
    name : str
        The metric's name. Also its module name and half its distribution name.
    family : Family
        The question the metric asks, written into its `@metric` decoration.
    into : Path
        The parent directory to write the package into.

    Returns
    -------
    Path
        The directory that was written.

    Raises
    ------
    ValueError
        If `name` is not a lowercase identifier.
    FileExistsError
        If the target directory already exists.
    """

    context, destination = _prepare(name, into)
    context["family"] = family.value
    context["family_constant"] = family.name

    _write_common("metric", context, destination)

    return destination
