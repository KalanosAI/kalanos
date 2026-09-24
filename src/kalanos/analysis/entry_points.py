"""Load one entry-point group, reporting what failed instead of raising.

Adapters, metrics and reporters each declare their own group.
A load failure is recorded as data, so one broken package still shows up in
the listing alongside the ones that loaded.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib.metadata
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from types import ModuleType


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


logger = logging.getLogger(__name__)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class MissingDependency(ImportError):
    """An optional dependency is absent, and the plugin knows what provides it.

    Raised by a plugin's own module at its optional import.
    If this information is provided, we can provide a better error handling message
    back to the user, explaining what needs to be installed to include that plugin.

    Attributes
    ----------
    distribution : str
        The distribution to install, as an installer names it.
    extra : str or None
        Its extra that provides the missing dependency, or `None` when the
        distribution provides it on its own.
    """

    def __init__(self, distribution: str, extra: str | None = None) -> None:
        """Name what makes this plugin importable.

        Parameters
        ----------
        distribution : str
            The distribution to install.
        extra : str or None
            The extra to install it with, when one is needed.
        """

        self.distribution = distribution
        self.extra = extra
        super().__init__(f"{self.target} is not installed")

    @property
    def target(self) -> str:
        """What an installer would be asked for, as a reader would type it.

        Returns
        -------
        str
            `distribution[extra]`, or `distribution` when there is no extra.
        """

        return f"{self.distribution}[{self.extra}]" if self.extra else self.distribution


class PluginSource(str, Enum):
    """Where a loaded plugin, or a load failure, came from."""

    # fmt: off
    ENTRY_POINT = "entry_point"
    PLUGIN_FILE = "plugin_file"
    PLUGIN_DIR  = "plugin_dir"
    # fmt: on


@dataclass(frozen=True)
class PluginLoadFailure:
    """One plugin discovery tried to load and could not.

    Attributes
    ----------
    name : str
        The entry point's or plugin file's name.
    source : PluginSource
        Which discovery path produced the failure.
    origin : str
        The distribution or file path this came from.
    reason : str
        What went wrong, in the form a `kalanos plugins` listing prints after FAILED.
    """

    name: str
    source: PluginSource
    origin: str
    reason: str


@dataclass(frozen=True)
class LoadedEntryPoint:
    """One entry point that resolved, and where it came from.

    Attributes
    ----------
    name : str
        The entry point's declared name.
    value : object
        Whatever `entry.load()` returned.
    origin : str
        The distribution or the entry point's target when the distribution is unknown.
    """

    name: str
    value: object
    origin: str


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def describe_load_failure(exc: Exception) -> str:
    """Render one load failure as the STATUS and ERROR columns print it.

    Parameters
    ----------
    exc : Exception
        Whatever the load raised.

    Returns
    -------
    str
        For a plugin that named what provides its optional dependency,
        a line naming that; otherwise the exception's class and message.
    """

    if isinstance(exc, MissingDependency):
        return f"not installed — add {exc.target}"

    return f"{type(exc).__name__}: {exc}"


def load_group(group: str) -> tuple[list[LoadedEntryPoint], list[PluginLoadFailure]]:
    """Load every entry point declared under one group.

    Parameters
    ----------
    group : str
        The entry-point group to walk, such as `kalanos.adapters`.

    Returns
    -------
    tuple[list[LoadedEntryPoint], list[PluginLoadFailure]]
        What loaded and what raised, each in the order it was tried.
    """

    loaded: list[LoadedEntryPoint] = []
    failures: list[PluginLoadFailure] = []

    # Called with `group=` as a keyword: the tests' fakes stand in for `entry_points`
    # with that signature, and a switch to the newer `entry_points().select(group=...)`
    # has to update them in step.
    for entry in importlib.metadata.entry_points(group=group):
        origin = (
            f"{entry.dist.name} {entry.dist.version}" if entry.dist else entry.value
        )
        try:
            value = entry.load()
        except Exception as exc:
            # A third-party module body can raise anything on import;
            # none of it may abort the run, so this is the one place a
            # bare `except Exception` is right.
            logger.warning("entry point %s failed to load: %s", entry.name, exc)
            failures.append(
                PluginLoadFailure(
                    name=entry.name,
                    source=PluginSource.ENTRY_POINT,
                    origin=origin,
                    reason=describe_load_failure(exc),
                )
            )
            continue
        loaded.append(LoadedEntryPoint(name=entry.name, value=value, origin=origin))

    return loaded, failures


def load_registering_modules(
    group: str,
) -> tuple[list[tuple[LoadedEntryPoint, ModuleType]], list[PluginLoadFailure]]:
    """Load a group whose entry points name modules imported for their side effect.

    Metrics and reporters register themselves by decoration, so importing the
    module is the whole of loading it; anything that is not a module cannot
    have registered anything and is reported as a failure.

    Parameters
    ----------
    group : str
        The entry-point group to walk, such as `kalanos.metrics`.

    Returns
    -------
    tuple[list[tuple[LoadedEntryPoint, ModuleType]], list[PluginLoadFailure]]
        Each entry paired with the module it imported, and every failure.
    """

    loaded, failures = load_group(group)

    modules: list[tuple[LoadedEntryPoint, ModuleType]] = []
    for item in loaded:
        if not inspect.ismodule(item.value):
            failures.append(
                PluginLoadFailure(
                    name=item.name,
                    source=PluginSource.ENTRY_POINT,
                    origin=item.origin,
                    reason=(
                        f"TypeError: {group} entry points must name a module, "
                        f"got {type(item.value).__name__}"
                    ),
                )
            )
            continue
        modules.append((item, item.value))

    return modules, failures
