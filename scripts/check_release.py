"""Check release metadata without publishing or contacting external services."""

import argparse
import re
import sys
from pathlib import Path


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def check(root, tag=None):
    """Return the version and prerelease flag, rejecting inconsistent metadata."""
    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:rc[1-9]\d*)?", version):
        raise ValueError("release version must be X.Y.Z or X.Y.ZrcN")
    citation = re.search(
        r'^version:\s*"?([^"\n]+)"?$', (root / "CITATION.cff").read_text(), re.M
    )
    if citation is None or citation[1].strip() != version:
        raise ValueError("CITATION.cff and package versions differ")
    if tag is not None and tag != "v" + version:
        raise ValueError("release tag and package versions differ")
    return version, "rc" in version


def main():
    """Print metadata for humans and fail before any upload if inconsistent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    args = parser.parse_args()
    version, prerelease = check(Path(__file__).resolve().parents[1], args.tag)
    print(f"version={version} prerelease={str(prerelease).lower()}")


if __name__ == "__main__":
    main()
