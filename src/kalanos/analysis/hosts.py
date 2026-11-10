"""Where a dataset can be hosted: how a reference resolves, and how files are read.

Each host declares both in one `Host`, so neither can be added without the other.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import re
from collections.abc import Callable
from typing import NamedTuple

# External
from upath import UPath

# Internal
from kalanos.analysis.models.discovery import SourceInfo
from kalanos.analysis.models.errors import SourceUnavailable
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

# A local UPath may report any of these.
LOCAL_PROTOCOLS = frozenset({"", "file", "local"})


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Host(NamedTuple):
    """A hosted dataset source: how a reference resolves, and how its files are read.

    Attributes
    ----------
    resolve : Callable[[str], tuple[UPath, SourceInfo] or None]
        Turns what the user typed into a pinned root and its description,
        or returns `None` when the reference is not for this host.
    direct_location : Callable[[UPath], tuple[str, dict[str, str] or None] or None]
        Gives the URI and storage options a reader with its own range requests
        reads a file of this host from, or `None` when the path is not this host's.
    """

    resolve: Callable[[str], tuple[UPath, SourceInfo] | None]
    direct_location: Callable[[UPath], tuple[str, dict[str, str] | None] | None]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def hosts() -> tuple[Host, ...]:
    """List the hosted sources, in the order a reference is offered to them.

    Returns
    -------
    tuple[Host, ...]
        Every host known to resolve a reference or read a file directly.
    """

    return (Host(resolve=_resolve_hf, direct_location=_hf_direct_location),)


def direct_location(path: UPath) -> tuple[str, dict[str, str] | None] | None:
    """Give the location a reader with its own range requests reads `path` from.

    polars reads only the byte ranges of the columns it is asked for when it opens
    a location itself, while a read from an fsspec file handle fetches the whole file.

    Parameters
    ----------
    path : UPath
        A file under a resolved root.

    Returns
    -------
    tuple[str, dict[str, str] or None] or None
        The plain path of a local file, or the URI and storage options its host
        gives. `None` when no host claims the path; read it through its handle.
    """

    if path.protocol in LOCAL_PROTOCOLS:
        return path.path, None
    for host in hosts():
        location = host.direct_location(path)
        if location is not None:
            return location
    return None


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


def _hf_direct_location(path: UPath) -> tuple[str, dict[str, str] | None] | None:
    """Give the URI and token a direct reader opens a Hugging Face file with.

    Parameters
    ----------
    path : UPath
        A file under a resolved root.

    Returns
    -------
    tuple[str, dict[str, str] or None] or None
        The full `hf://` URI, with the token `huggingface_hub` finds as
        storage options, or `None` when `path` is not on Hugging Face.
    """

    if path.protocol != "hf":
        return None
    # polars reads only `HF_TOKEN` from the environment,
    # so a token saved by `hf auth login` has to be passed explicitly.
    hub = load_huggingface_hub()
    token = hub.get_token() if hub is not None else None
    return str(path), {"token": token} if token else None
