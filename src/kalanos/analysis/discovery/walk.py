"""Walk a dataset folder and classify each file."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging

# External
from upath import UPath

# Internal
from kalanos.analysis.models.discovery import SkippedSource, SkipReason, SourceCandidate


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


def _classify(path: UPath) -> SourceCandidate | SkippedSource:
    """Classify one file as a candidate or a skip.

    Checks:
    1. Readability
    2. Emptiness (a zero-byte file, from the same `stat()` call)

    Parameters
    ----------
    path : UPath
        File to classify.

    Returns
    -------
    SourceCandidate or SkippedSource
        The classification result for this file.
    """

    try:
        size = path.stat().st_size
    except OSError:
        return SkippedSource(path=path, reason=SkipReason.UNREADABLE)

    if size == 0:
        return SkippedSource(path=path, reason=SkipReason.EMPTY)

    return SourceCandidate(path=path)


def _log_classification(item: SourceCandidate | SkippedSource) -> None:
    """Log one file's classification at the level its outcome deserves.

    Parameters
    ----------
    item : SourceCandidate or SkippedSource
        The result `_classify` returned for one file.
    """

    if isinstance(item, SourceCandidate):
        logger.debug("%s: candidate", item.path)
        return

    logger.warning("%s: skipped (%s)", item.path, item.reason.value)


def _hidden_depth(path: UPath, root: UPath) -> int | None:
    """Find the first dot-named part of `path` below `root`.

    Parameters
    ----------
    path : UPath
        An entry found under `root`.
    root : UPath
        The walked root, never itself treated as hidden.

    Returns
    -------
    int or None
        The index of that part within `path.relative_to(root).parts`,
        or `None` when no part is hidden.
    """

    parts = path.relative_to(root).parts
    return next((i for i, part in enumerate(parts) if part.startswith(".")), None)


def walk_folder(root: UPath) -> list[SourceCandidate | SkippedSource]:
    """Walk `root` and classify every file and directory found.

    Parameters
    ----------
    root : UPath
        Recording to classify, or a dataset folder to walk recursively.

    Returns
    -------
    list of SourceCandidate or SkippedSource
        One entry per file or directory found, in sorted path order.
        A hidden entry below the root is one skip, and nothing inside it is listed.
    """

    if root.is_file():
        item = _classify(root)
        _log_classification(item)
        return [item]

    # Excludes nothing but files and directories: a dangling symlink
    # or a FIFO must still reach `_classify` to be reported as a skip,
    # and `is_file()` alone would silently drop both.
    found = [root]
    found.extend(root.rglob("*"))

    results: list[SourceCandidate | SkippedSource] = []
    for path in sorted(found):
        if path != root and (depth := _hidden_depth(path, root)) is not None:
            # Only the outermost hidden entry is reported;
            # whatever lies inside it never reaches an adapter.
            if depth == len(path.relative_to(root).parts) - 1:
                results.append(SkippedSource(path=path, reason=SkipReason.HIDDEN))
            continue
        try:
            is_dir = path.is_dir()
        except OSError:
            # `_classify` re-stats and reports the same failure as UNREADABLE;
            # treating it as a file here would silently swallow it instead.
            is_dir = False
        # A directory has no size and no readability question of its own;
        # one that cannot be listed surfaces through its missing contents.
        results.append(SourceCandidate(path=path) if is_dir else _classify(path))

    for item in results:
        _log_classification(item)
    return results
