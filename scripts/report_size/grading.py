"""Grade a dataset with one version, measure the report, and cache the result."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Internal
from . import config
from .versions import Version, requirement


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Measurement:
    """The size of one report.

    Attributes
    ----------
    bytes : int
        The report file's size.
    episodes : int
        The episodes the grade loaded, from `inventory.loaded`.
    """

    bytes: int
    episodes: int

    @property
    def per_episode(self) -> float:
        """Return the report bytes per loaded episode.

        Returns
        -------
        float
            `bytes / episodes`.
        """

        return self.bytes / self.episodes


class GradeFailed(Exception):
    """A grade exited with an operational error or wrote no report."""


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def options_key() -> str:
    """Return a key that changes whenever the grade options or the bundle change.

    The bundle's path is left out, so the key doesn't depend on where the repository
    is checked out.

    Returns
    -------
    str
        The first 12 hex characters of a sha256 over the flags and the bundle's bytes.
    """

    digest = hashlib.sha256()
    for flag in config.GRADE_FLAGS:
        digest.update(flag.encode() + b"\0")
    digest.update(config.BUNDLE.read_bytes())
    return digest.hexdigest()[:12]


def grade(install: str, dataset: str, report: Path) -> None:
    """Grade one dataset and write its report.

    The report is written to a `.partial.json` sibling first and renamed only on
    success, so a crash never leaves a file that looks complete.

    Parameters
    ----------
    install : str
        The kalanos requirement to grade with.
    dataset : str
        A Hugging Face dataset as `<owner>/<name>@<revision>`.
    report : Path
        Where the report goes.

    Raises
    ------
    GradeFailed
        If the grade exits with anything but 0 or 1, or writes no report.
    """

    partial = report.with_name(report.stem + ".partial.json")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            install,
            "kalanos",
            "grade",
            f"hf://datasets/{dataset}",
            *config.GRADE_FLAGS,
            "--profile",
            str(config.BUNDLE),
            "--no-color",
            "--report",
            str(partial),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Exit 1 means the eligibility gate tripped, and the report is still complete.
    if result.returncode in (0, 1) and partial.is_file():
        partial.rename(report)
        return

    partial.unlink(missing_ok=True)
    tail = "\n".join(result.stderr.splitlines()[-20:])
    raise GradeFailed(f"grading {dataset} exited {result.returncode}:\n{tail}")


def measure(report: Path) -> Measurement:
    """Read a report's size and loaded-episode count.

    Parameters
    ----------
    report : Path
        A JSON report.

    Returns
    -------
    Measurement
        The file's size and `inventory.loaded`.

    Raises
    ------
    ValueError
        If the report has no `inventory.loaded`, or it is 0.
    """

    with report.open("rb") as stream:
        parsed = json.load(stream)
    try:
        episodes = parsed["inventory"]["loaded"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"no inventory.loaded in {report}") from exc
    # Exit 1 also covers a grade that loaded nothing, which has no size per episode.
    if not episodes:
        raise ValueError(f"{report} loaded no episodes")
    return Measurement(bytes=report.stat().st_size, episodes=episodes)


def ensure_measurement(
    version: Version, dataset: str, repo: Path, cache: Path
) -> Measurement:
    """Return a dataset's measurement at a version, grading only when none is cached.

    The sidecar `<dataset>.size.json` decides reuse and is written last. The report
    stays beside it for inspection.

    Parameters
    ----------
    version : Version
        The kalanos version.
    dataset : str
        A Hugging Face dataset as `<owner>/<name>@<revision>`.
    repo : Path
        The git repository holding commit versions.
    cache : Path
        The cache directory.

    Returns
    -------
    Measurement
        The report's size and loaded-episode count.
    """

    folder = cache / version.key / options_key()
    stem = dataset.replace("/", "__")
    report = folder / f"{stem}.json"
    sidecar = folder / f"{stem}.size.json"

    if sidecar.is_file():
        print(f"reusing {dataset} for {version.label}", file=sys.stderr)
        return Measurement(**json.loads(sidecar.read_text()))

    print(f"grading {dataset} with {version.label}", file=sys.stderr)
    folder.mkdir(parents=True, exist_ok=True)
    grade(requirement(version, repo, cache), dataset, report)
    measured = measure(report)
    staging = sidecar.with_name(sidecar.name + ".tmp")
    staging.write_text(
        json.dumps({"bytes": measured.bytes, "episodes": measured.episodes})
    )
    staging.replace(sidecar)
    return measured
