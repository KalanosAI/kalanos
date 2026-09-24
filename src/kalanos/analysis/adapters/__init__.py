"""Find and register the readers that teach Kalanos a format.

A library beside the stages, not one of them: an adapter reads a family of
formats and produces episodes, and never applies a threshold. Discovery
merges two sources — Python entry points under `kalanos.adapters`, the
distribution mechanism a third-party package uses, and local registration
through the `@adapter` decorator, for a single file passed with `--plugin`
or dropped in the user plugin directory.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.adapters.csv import CsvAdapter
from kalanos.analysis.adapters.delimited import DelimitedTextAdapter
from kalanos.analysis.adapters.discover import (
    AdapterDiscovery,
    LoadedAdapter,
    discover_adapters,
)
from kalanos.analysis.adapters.json import JsonAdapter
from kalanos.analysis.adapters.jsonl import JsonlAdapter
from kalanos.analysis.adapters.registry import (
    adapter,
    clear_local_registry,
    locally_registered,
)
from kalanos.analysis.adapters.select import Selection, select_adapter
from kalanos.analysis.adapters.tabular import ParsedTable, TabularAdapter
from kalanos.analysis.entry_points import PluginLoadFailure, PluginSource
from kalanos.analysis.models.adapters import AdapterTie


__all__ = [
    "AdapterDiscovery",
    "AdapterTie",
    "CsvAdapter",
    "DelimitedTextAdapter",
    "JsonAdapter",
    "JsonlAdapter",
    "LoadedAdapter",
    "ParsedTable",
    "PluginLoadFailure",
    "PluginSource",
    "Selection",
    "TabularAdapter",
    "adapter",
    "clear_local_registry",
    "discover_adapters",
    "locally_registered",
    "select_adapter",
]
