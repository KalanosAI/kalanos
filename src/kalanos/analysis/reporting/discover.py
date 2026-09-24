"""Find every reporter available to this process, and say which package registered it.

A reporter entry point points to a module.
Loading the entry point imports that module, which runs its `@reporter` decorations.
Attribution works backward from there, matching each registry entry to the module,
or a submodule, that imported it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass

# Internal
from kalanos.analysis.entry_points import (
    PluginLoadFailure,
    PluginSource,
    load_registering_modules,
)
from kalanos.analysis.reporting.registry import (
    RegisteredReporter,
    registered_reporters,
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class LoadedReporter:
    """One registered reporter discovery found, and where it came from.

    Attributes
    ----------
    name : str
        The reporter's own name.
    reporter : RegisteredReporter
        The registry entry itself.
    source : PluginSource
        Which discovery path found it.
    origin : str
        The distribution ("kalanos 0.1.0") that declared the entry point.
    """

    name: str
    reporter: RegisteredReporter
    source: PluginSource
    origin: str


@dataclass(frozen=True)
class ReporterDiscovery:
    """Everything one discovery run found.

    Attributes
    ----------
    reporters : list[LoadedReporter]
        Every reporter that loaded, in the order it was found.
    failures : list[PluginLoadFailure]
        Every entry point that raised or named something other than a module.
    """

    reporters: list[LoadedReporter]
    failures: list[PluginLoadFailure]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def discover_reporters(*, entry_points: bool = True) -> ReporterDiscovery:
    """Find every reporter registered under the `kalanos.reporters` entry-point group.

    Does not deduplicate by name. Two modules registering the same reporter name
    are both reported, so the collision is visible.

    Parameters
    ----------
    entry_points : bool
        Whether to walk the entry-point group at all. `False` returns an empty
        discovery, for a caller that wants to isolate itself from what is installed.

    Returns
    -------
    ReporterDiscovery
        Every reporter found and every load failure.
    """

    reporters: list[LoadedReporter] = []
    failures: list[PluginLoadFailure] = []

    if entry_points:
        modules, failures = load_registering_modules("kalanos.reporters")
        for item, module in modules:
            reporters.extend(
                LoadedReporter(
                    name=entry.name,
                    reporter=entry,
                    source=PluginSource.ENTRY_POINT,
                    origin=item.origin,
                )
                for entry in registered_reporters()
                if entry.module == module.__name__
                or entry.module.startswith(f"{module.__name__}.")
            )

    return ReporterDiscovery(reporters=reporters, failures=failures)
