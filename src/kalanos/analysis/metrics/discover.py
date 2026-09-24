"""Find every metric available to this process, and say which package registered it.

A metric entry point names a module. Importing it runs the `@metric`
decorations inside. Attribution works backward from there, matching each
registry entry to the module, or a submodule, that imported it.
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
from kalanos.analysis.metrics.registry import RegisteredMetric, registered_metrics


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class LoadedMetric:
    """One registered metric discovery found, and where it came from.

    Attributes
    ----------
    name : str
        The metric's own name.
    metric : RegisteredMetric
        The registry entry itself.
    source : PluginSource
        Which discovery path found it.
    origin : str
        The distribution ("kalanos 0.1.0") that declared the entry point.
    """

    name: str
    metric: RegisteredMetric
    source: PluginSource
    origin: str


@dataclass(frozen=True)
class MetricDiscovery:
    """Everything one discovery run found.

    Attributes
    ----------
    metrics : list[LoadedMetric]
        Every metric that loaded, in the order it was found.
    failures : list[PluginLoadFailure]
        Every entry point that raised or named something other than a module.
    """

    metrics: list[LoadedMetric]
    failures: list[PluginLoadFailure]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def discover_metrics(*, entry_points: bool = True) -> MetricDiscovery:
    """Find every metric registered under the `kalanos.metrics` entry-point group.

    Does not deduplicate by name. Two modules registering the same metric name
    are both reported, so the collision is visible.

    Parameters
    ----------
    entry_points : bool
        Whether to walk the entry-point group at all. `False` returns an empty
        discovery, for a caller that wants to isolate itself from what is installed.

    Returns
    -------
    MetricDiscovery
        Every metric found and every load failure.
    """

    metrics: list[LoadedMetric] = []
    failures: list[PluginLoadFailure] = []

    if entry_points:
        modules, failures = load_registering_modules("kalanos.metrics")
        for item, module in modules:
            metrics.extend(
                LoadedMetric(
                    name=entry.name,
                    metric=entry,
                    source=PluginSource.ENTRY_POINT,
                    origin=item.origin,
                )
                for entry in registered_metrics()
                if entry.module == module.__name__
                or entry.module.startswith(f"{module.__name__}.")
            )

    return MetricDiscovery(metrics=metrics, failures=failures)
