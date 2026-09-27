"""Normalise the task instructions an adapter found, and apply the dataset rule.

One rule, shared by every adapter so they can't drift: instructions apply to a
dataset only when at least one of its episodes has a non-blank one. A dataset
never annotated with language gets `None` for every episode, so the annotation
metric reads it as not applicable rather than as every episode missing one.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar


Key = TypeVar("Key")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def as_task_list(raw: Any) -> list[str]:
    """Turn whatever a source stored as an episode's instructions into strings.

    Parameters
    ----------
    raw : Any
        A string, bytes, a list or array of either, or `None`.

    Returns
    -------
    list[str]
        One string per instruction, blank ones kept so the metric can count them.
        Empty when `raw` holds nothing.
    """

    if raw is None:
        return []
    if isinstance(raw, bytes):
        return [raw.decode("utf-8", errors="replace")]
    if isinstance(raw, str):
        return [raw]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, Iterable):
        return [item for value in raw for item in as_task_list(value)]
    return [str(raw)]


def dataset_tasks(
    per_episode: Mapping[Key, list[str]],
) -> dict[Key, list[str]] | None:
    """Apply the dataset rule: keep the instructions only if any are non-blank.

    Parameters
    ----------
    per_episode : Mapping
        Each episode's instructions, keyed however the adapter identifies it.

    Returns
    -------
    dict or None
        `per_episode` as a dict when at least one episode has a non-blank
        instruction; `None` when none does, so every episode's `tasks` stays `None`.
    """

    if any(task.strip() for tasks in per_episode.values() for task in tasks):
        return dict(per_episode)
    return None
