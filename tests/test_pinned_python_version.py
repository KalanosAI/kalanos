"""The published Python floor must stay in step with what pyproject declares.

`requires-python` is the promise to users on the floor; this guards it against
silent drift.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

from pathlib import Path


try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # pyright: ignore[reportMissingImports]


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


REPO_ROOT = Path(__file__).parent.parent

# The floor kalanos publishes as its minimum supported Python
# (pyproject.toml requires-python).
MINIMUM_SUPPORTED_VERSION = (3, 10)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _requires_python(pyproject: Path) -> str:
    """Read the ``requires-python`` constraint from a pyproject.toml file.

    Parameters
    ----------
    pyproject : Path
        Path to the pyproject.toml file to read.

    Returns
    -------
    str
        The value of ``project.requires-python``.
    """

    data = tomllib.loads(pyproject.read_text())
    return data["project"]["requires-python"]


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_pyproject_declares_the_expected_floor():
    """Verify that pyproject.toml requires at least the minimum supported version."""

    major, minor = MINIMUM_SUPPORTED_VERSION
    expected = f">={major}.{minor}"

    assert _requires_python(REPO_ROOT / "pyproject.toml") == expected
