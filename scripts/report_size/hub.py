"""Pin Hugging Face datasets to a commit, so cached reports never go stale."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import re

# External
import httpx
from huggingface_hub import HfApi


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

COMMIT = re.compile(r"[0-9a-f]{40}")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def pin_dataset(spec: str) -> str:
    """Return a dataset pinned to the commit its revision points at.

    A dataset already pinned to a 40-hex commit is returned as is. Otherwise the Hub
    resolves the revision, or the latest commit when there is none.

    Parameters
    ----------
    spec : str
        A Hugging Face dataset as `<owner>/<name>` or `<owner>/<name>@<revision>`.

    Returns
    -------
    str
        The dataset as `<owner>/<name>@<40-hex commit>`.

    Raises
    ------
    ValueError
        If the Hub can't resolve the dataset or its revision.
    """

    repo_id, _, revision = spec.partition("@")
    if COMMIT.fullmatch(revision):
        return spec
    try:
        info = HfApi().dataset_info(repo_id, revision=revision or None)
    except (httpx.HTTPError, OSError) as exc:
        raise ValueError(
            f"can't resolve {spec} on the Hugging Face Hub: {exc}"
        ) from exc
    if info.sha is None:
        raise ValueError(f"the Hugging Face Hub returned no commit for {spec}")
    return f"{repo_id}@{info.sha}"
