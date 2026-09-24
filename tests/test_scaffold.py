"""Verifies what `kalanos new` writes: valid Python, and a correct entry point.

A bad entry point only shows up once someone installs the package and finds
nothing was discovered, so it's checked directly here instead of waiting on
the integration test that installs the result.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import ast

# External
import pytest


try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

# Internal
from kalanos.analysis.models.metrics import Family
from kalanos.scaffold import scaffold_adapter, scaffold_metric


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _entry_points(package, group: str) -> dict[str, str]:
    """Read one entry-point group out of a generated package's pyproject.toml."""

    data = tomllib.loads((package / "pyproject.toml").read_text())
    return data["project"]["entry-points"][group]


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_generated_adapter_declares_its_entry_point_as_a_class(tmp_path):
    """Verify the adapter scaffold points `kalanos.adapters` at the adapter class."""

    package = scaffold_adapter("myformat", tmp_path)

    assert _entry_points(package, "kalanos.adapters") == {
        "myformat": "kalanos_myformat:MyformatAdapter"
    }


def test_the_generated_metric_declares_its_entry_point_as_a_bare_module(tmp_path):
    """Verify the metric scaffold points `kalanos.metrics` at a bare module.

    Discovery imports the target for its registration side effect, so a
    function-valued entry point would load something that registers nothing.
    """

    package = scaffold_metric("myjitter", Family.TIMING, tmp_path)

    assert _entry_points(package, "kalanos.metrics") == {"myjitter": "kalanos_myjitter"}


def test_the_chosen_family_reaches_the_generated_decoration(tmp_path):
    """Verify --family is what the generated @metric declares, not a fixed default."""

    package = scaffold_metric("myjitter", Family.MOTION, tmp_path)

    source = (package / "src" / "kalanos_myjitter" / "__init__.py").read_text()
    assert "family=Family.MOTION," in source


@pytest.mark.parametrize("name", ["My-Format", "9lives", "my format", ""])
def test_a_name_that_is_not_a_module_name_is_refused(tmp_path, name):
    """Verify a name that could not be a module is refused before any write."""

    with pytest.raises(ValueError):
        scaffold_adapter(name, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_writing_over_an_existing_package_is_refused(tmp_path):
    """Verify a second run leaves the first package alone rather than overwriting it."""

    package = scaffold_adapter("myformat", tmp_path)
    (package / "src" / "kalanos_myformat" / "__init__.py").write_text("# mine\n")

    with pytest.raises(FileExistsError):
        scaffold_adapter("myformat", tmp_path)

    assert (package / "src" / "kalanos_myformat" / "__init__.py").read_text() == (
        "# mine\n"
    )


def test_every_generated_python_file_parses(tmp_path):
    """Verify neither scaffold can ship Python that fails on a syntax error."""

    packages = [
        scaffold_adapter("myformat", tmp_path),
        scaffold_metric("myjitter", Family.TIMING, tmp_path),
    ]

    written = [path for package in packages for path in package.rglob("*.py")]
    assert written, "expected the scaffolds to write Python files"
    for path in written:
        ast.parse(path.read_text(), filename=str(path))
