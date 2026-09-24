"""Turn what the user typed into a root to walk, and refuse one too large to stream."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import re

# External
from upath import UPath

# Internal
from kalanos.analysis.models.discovery import SourceInfo, SourceLimits
from kalanos.analysis.models.errors import SourceTooLarge, SourceUnavailable
from kalanos.analysis.optional import load_huggingface_hub


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_HF_PAGE = re.compile(
    r"^https?://huggingface\.co/datasets/(?P<repo>[^/@]+/[^/@]+)"
    r"(?:/tree/(?P<rev>[^/]+))?(?P<rest>/.*)?$"
)
_HF_URI = re.compile(
    r"^hf://datasets/(?P<repo>[^/@]+/[^/@]+)(?:@(?P<rev>[^/]+))?(?P<rest>/.*)?$"
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def resolve_source(path: str | os.PathLike[str] | UPath) -> tuple[UPath, SourceInfo]:
    """Turn what the user typed into the root to walk, and describe that root.

    Parameters
    ----------
    path : str, PathLike or UPath
        A local path, any UPath, or a reference to a hosted dataset.

    Returns
    -------
    tuple[UPath, SourceInfo]
        The root to grade, pinned to one commit when its host versions it,
        and its description.

    Raises
    ------
    SourceUnavailable
        If the path does not exist, or a hosted dataset cannot be read:
        the repository or revision is missing, gated or private,
        or the extra its host needs is not installed.
    """

    for resolve_hosted in (_resolve_hf,):
        resolved = resolve_hosted(str(path))
        if resolved is not None:
            return resolved
    return _resolve_path(path)


def _resolve_path(path: str | os.PathLike[str] | UPath) -> tuple[UPath, SourceInfo]:
    """Describe a root fsspec reads as-is, and total up what it holds.

    Parameters
    ----------
    path : str, PathLike or UPath
        A local path or any UPath.

    Returns
    -------
    tuple[UPath, SourceInfo]
        The root, unchanged, and its description.

    Raises
    ------
    SourceUnavailable
        If the path does not exist.
    """

    # A UPath is kept as given, so its storage options survive.
    root = path if isinstance(path, UPath) else UPath(path)
    if not root.exists():
        raise SourceUnavailable(f"{root} does not exist")
    files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else [root]
    source = SourceInfo(
        uri=str(root),
        # A local UPath may report "", "file" or "local";
        # one spelling keeps is_remote honest.
        protocol="file" if root.protocol in ("", "file", "local") else root.protocol,
        size_bytes=sum(p.stat().st_size for p in files),
        file_count=len(files),
    )
    return root, source


def _resolve_hf(text: str) -> tuple[UPath, SourceInfo] | None:
    """Pin a Hugging Face dataset reference to one commit and total up what it holds.

    Parameters
    ----------
    text : str
        What the user typed. Claimed when it is one of
        `hf://datasets/<org>/<name>[@<rev>][/<path>]` or
        `https://huggingface.co/datasets/<org>/<name>[/tree/<rev>][/<path>]`,
        where `<rev>` is a branch, tag or commit, and the default branch without one.

    Returns
    -------
    tuple[UPath, SourceInfo] or None
        `hf://datasets/<org>/<name>@<sha>[/<path>]` and its description,
        or `None` when `text` is not a Hugging Face dataset reference.

    Raises
    ------
    SourceUnavailable
        If the `hf` extra is missing, or the repository or revision cannot be read.
    """

    # Step 1: claim the reference
    match = _HF_PAGE.match(text) or _HF_URI.match(text)
    if match is None:
        return None
    repo_id, revision = match["repo"], match["rev"]
    path_in_repo = (match["rest"] or "").strip("/")

    # Step 2: pin it to one commit
    hub = load_huggingface_hub()
    if hub is None:
        raise SourceUnavailable(
            "grading a Hugging Face dataset needs the hf extra: "
            "pip install 'kalanos[hf]'"
        )
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    # Gated before not-found, because GatedRepoError subclasses RepositoryNotFoundError.
    try:
        info = hub.HfApi().dataset_info(repo_id, revision=revision, files_metadata=True)
    except GatedRepoError as exc:
        raise SourceUnavailable(
            f"{repo_id} is gated: accept its terms on huggingface.co, then set HF_TOKEN"
        ) from exc
    except RevisionNotFoundError as exc:
        raise SourceUnavailable(f"{repo_id} has no revision {revision!r}") from exc
    except RepositoryNotFoundError as exc:
        raise SourceUnavailable(
            f"{repo_id} was not found, or is private and HF_TOKEN is not set"
        ) from exc
    except HfHubHTTPError as exc:
        raise SourceUnavailable(
            f"could not reach {repo_id} on Hugging Face: {exc}"
        ) from exc

    # Step 3: total up the files under it
    prefix = f"{path_in_repo}/" if path_in_repo else ""
    files = [
        s
        for s in info.siblings or []
        if s.rfilename == path_in_repo or s.rfilename.startswith(prefix)
    ]
    uri = f"hf://datasets/{repo_id}@{info.sha}" + (
        f"/{path_in_repo}" if path_in_repo else ""
    )
    source = SourceInfo(
        uri=uri,
        protocol="hf",
        repo_id=repo_id,
        revision=info.sha,
        size_bytes=sum(s.size or 0 for s in files),
        file_count=len(files),
    )
    return UPath(uri), source


def enforce_limits(source: SourceInfo, limits: SourceLimits) -> None:
    """Refuse a remote root that is over either limit, before any of its files is read.

    Parameters
    ----------
    source : SourceInfo
        The resolved root.
    limits : SourceLimits
        The limits to hold it to. A local root is never limited.

    Raises
    ------
    SourceTooLarge
        If the root is remote and over `max_bytes` or `max_files`.
    """

    if not source.is_remote:
        return
    if limits.max_bytes is not None and source.size_bytes > limits.max_bytes:
        raise SourceTooLarge(
            f"{source.uri} holds {source.size_bytes / 1e9:.3g} GB, "
            f"over the {limits.max_bytes / 1e9:g} GB limit; "
            "raise it with --max-remote-gb or KALANOS_REMOTE_MAX_BYTES"
        )
    if limits.max_files is not None and source.file_count > limits.max_files:
        raise SourceTooLarge(
            f"{source.uri} holds {source.file_count} files, "
            f"over the {limits.max_files} file limit; "
            "raise it with --max-remote-files or KALANOS_REMOTE_MAX_FILES"
        )
