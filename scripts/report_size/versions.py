"""Resolve the kalanos versions to compare, and the requirement that installs each."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Version:
    """A kalanos version to grade with.

    Attributes
    ----------
    label : str
        How the table names it: the git ref or wheel path the user passed, or how the
        merge-base was found.
    key : str
        The cache key: the full commit SHA, or `wheel-` and the first 12 hex
        characters of the wheel's sha256.
    source : str
        The commit SHA, or the wheel's absolute path.
    """

    label: str
    key: str
    source: str

    @property
    def is_wheel(self) -> bool:
        """Whether this version installs from a built wheel.

        Returns
        -------
        bool
            True for a wheel, False for a commit.
        """

        return self.key.startswith("wheel-")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _git(repo: Path, *args: str) -> str | None:
    """Run a git command and return its trimmed stdout.

    Parameters
    ----------
    repo : Path
        The git repository to run in.
    *args : str
        The git arguments.

    Returns
    -------
    str | None
        The output, or None when git exits with an error.
    """

    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_version(spec: str, repo: Path) -> Version:
    """Turn a git ref or a wheel path into a version.

    Parameters
    ----------
    spec : str
        A git ref `repo` resolves, or a path ending in `.whl`.
    repo : Path
        The git repository that resolves refs.

    Returns
    -------
    Version
        The version, keyed by commit SHA or wheel content.

    Raises
    ------
    ValueError
        If the wheel doesn't exist or the ref doesn't name a commit.
    """

    if spec.endswith(".whl"):
        wheel = Path(spec).resolve()
        if not wheel.is_file():
            raise ValueError(f"no wheel at {spec}")
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        return Version(label=spec, key=f"wheel-{digest[:12]}", source=str(wheel))

    sha = _git(repo, "rev-parse", "--verify", "--quiet", f"{spec}^{{commit}}")
    if sha is None:
        raise ValueError(f"{spec} doesn't name a commit in {repo}")
    return Version(label=spec, key=sha, source=sha)


def merge_base(after: Version, base: str, repo: Path) -> Version:
    """Return the commit where the after side branched off `base`.

    A wheel is taken to be built from `HEAD`, so its merge-base is `HEAD`'s.

    Parameters
    ----------
    after : Version
        The version being measured.
    base : str
        The git ref the after side branched from.
    repo : Path
        The git repository that resolves refs.

    Returns
    -------
    Version
        The merge-base commit, labelled with the two refs that found it.

    Raises
    ------
    ValueError
        If the two refs share no history or `base` doesn't resolve.
    """

    ref = "HEAD" if after.is_wheel else after.source
    sha = _git(repo, "merge-base", ref, base)
    if sha is None:
        raise ValueError(f"no merge-base of {ref} and {base} in {repo}")
    label = f"merge-base of {'HEAD' if after.is_wheel else after.label} and {base}"
    return Version(label=label, key=sha, source=sha)


def requirement(version: Version, repo: Path, cache: Path) -> str:
    """Return the requirement that installs kalanos at a version.

    A commit is extracted with `git archive` once, under `cache / "src"`, and
    installed from that directory.

    Parameters
    ----------
    version : Version
        The version to install.
    repo : Path
        The git repository holding the commit.
    cache : Path
        The cache directory.

    Returns
    -------
    str
        A `kalanos[all] @ file://…` requirement for `uv run --with`.

    Raises
    ------
    subprocess.CalledProcessError
        If `git archive` or `tar` fails.
    """

    if version.is_wheel:
        return f"kalanos[all] @ {Path(version.source).as_uri()}"

    target = cache / "src" / version.source
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        # Extract beside the target and rename at the end, so an interrupted extract
        # is never reused.
        staging = Path(
            tempfile.mkdtemp(prefix=f".{version.source}-", dir=target.parent)
        )
        archive = None
        try:
            archive = subprocess.Popen(
                ["git", "-C", str(repo), "archive", "--format=tar", version.source],
                stdout=subprocess.PIPE,
            )
            assert archive.stdout is not None
            subprocess.run(
                ["tar", "-x", "-C", str(staging)], stdin=archive.stdout, check=True
            )
            archive.stdout.close()
            if archive.wait() != 0:
                raise subprocess.CalledProcessError(archive.returncode, archive.args)
            staging.rename(target)
        finally:
            # A failed tar leaves git archive blocked on a full pipe.
            if archive is not None and archive.poll() is None:
                archive.kill()
                archive.wait()
            shutil.rmtree(staging, ignore_errors=True)
    return f"kalanos[all] @ {target.as_uri()}"
