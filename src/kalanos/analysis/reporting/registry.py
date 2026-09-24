"""The @reporter decorator and the registry `write_report` picks a renderer from.

A renderer declares the file extensions it handles. `write_report` looks up a format
by checking the registry. Only reporters this process has already imported show up here
— loading one from a `--report` path isn't wired up yet.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

# Internal
from kalanos.analysis.models.report import Report


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


RenderFunction = Callable[[Report], str]

F = TypeVar("F", bound=RenderFunction)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class RegisteredReporter:
    """One renderer as the registry knows it: name, extensions and body.

    Attributes
    ----------
    name : str
        The reporter's name, as `kalanos reporters` prints it.
    extensions : tuple[str, ...]
        The suffixes this reporter claims, lowercased and dot-prefixed.
    render : RenderFunction
        The decorated function itself.
    module : str
        The module that registered it, so discovery can attribute a reporter to
        the entry point whose import put it here.
    """

    name: str
    extensions: tuple[str, ...]
    render: RenderFunction
    module: str


# Populated by every @reporter-decorated function as its module is imported.
_REGISTRY: list[RegisteredReporter] = []


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def reporter(*, name: str, extensions: Sequence[str]) -> Callable[[F], F]:
    """Register a function as the renderer for one or more file suffixes.

    Parameters
    ----------
    name : str
        The reporter's name.
    extensions : Sequence[str]
        The suffixes it claims, each with its leading dot.

    Returns
    -------
    Callable
        A decorator that registers `func` unchanged and returns it.

    Raises
    ------
    ValueError
        If `name` is empty, `extensions` is empty, or an extension is empty
        or lacks its leading dot.
    """

    if not name:
        raise ValueError("a reporter needs a non-empty name")
    if not extensions:
        raise ValueError(f"reporter {name!r} claims no extensions")

    claimed: list[str] = []
    for extension in extensions:
        if not extension.startswith(".") or len(extension) < 2:
            raise ValueError(
                f"reporter {name!r} declared extension {extension!r}; "
                "an extension must start with a dot and name a suffix"
            )
        claimed.append(extension.lower())

    def decorator(func: F) -> F:
        _REGISTRY.append(
            RegisteredReporter(
                name=name,
                extensions=tuple(claimed),
                render=func,
                module=func.__module__,
            )
        )
        return func

    return decorator


def registered_reporters() -> list[RegisteredReporter]:
    """List every registered reporter, in registration order.

    Returns
    -------
    list[RegisteredReporter]
        Every reporter registered so far.
    """

    return list(_REGISTRY)
