"""Turn what the user typed into a root to walk, and refuse one too large to stream."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os

# External
from upath import UPath

from kalanos.analysis.hosts import LOCAL_PROTOCOLS, hosts

# Internal
from kalanos.analysis.models.discovery import SourceInfo, SourceLimits
from kalanos.analysis.models.errors import SourceTooLarge, SourceUnavailable


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

    for host in hosts():
        resolved = host.resolve(str(path))
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
        protocol="file" if root.protocol in LOCAL_PROTOCOLS else root.protocol,
        size_bytes=sum(p.stat().st_size for p in files),
        file_count=len(files),
    )
    return root, source


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
