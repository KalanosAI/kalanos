"""Verifies release metadata agreement and prerelease policy, without publishing."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib.util
from pathlib import Path

# External
import pytest


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_release", ROOT / "scripts/check_release.py"
)
assert spec is not None and spec.loader is not None
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


@pytest.mark.parametrize("version, prerelease", [("0.7.0rc1", True), ("0.7.0", False)])
def test_metadata_agreement(tmp_path, version, prerelease):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CITATION.cff").write_text(f'version: "{version}"\n')
    assert release.check(tmp_path, "v" + version) == (version, prerelease)
    with pytest.raises(ValueError, match="tag"):
        release.check(tmp_path, "v0.6.0")
    (tmp_path / "CITATION.cff").write_text('version: "0.6.0"\n')
    with pytest.raises(ValueError, match="CITATION"):
        release.check(tmp_path)


def test_checkout_metadata_agrees():
    """The checked-out pyproject.toml and CITATION.cff name the same version."""

    release.check(ROOT)
