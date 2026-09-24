"""Find every adapter available to this process: entry points and local files.

Picking a winner for a path is selection's job.
Discovery deliberately returns duplicate names and leaves resolving them to the caller.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import importlib.util
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Internal
from kalanos.analysis.adapters.registry import locally_registered
from kalanos.analysis.entry_points import (
    PluginLoadFailure,
    PluginSource,
    describe_load_failure,
    load_group,
)
from kalanos.analysis.models.adapters import Adapter


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


logger = logging.getLogger(__name__)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class LoadedAdapter:
    """One adapter instance discovery found, and where it came from.

    Attributes
    ----------
    name : str
        The adapter's own name.
    adapter : Adapter
        The instance itself.
    source : PluginSource
        Which discovery path found it.
    origin : str
        The distribution ("kalanos 0.1.0") for an entry point,
        or the file path otherwise.
    """

    name: str
    adapter: Adapter
    source: PluginSource
    origin: str


@dataclass(frozen=True)
class AdapterDiscovery:
    """Everything one discovery run found.

    Attributes
    ----------
    adapters : list[LoadedAdapter]
        Every adapter that loaded, in the order it was found.
    failures : list[PluginLoadFailure]
        Every entry point or plugin file that raised, in the order it was tried.
    """

    adapters: list[LoadedAdapter]
    failures: list[PluginLoadFailure]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _instantiate(loaded: object) -> Adapter:
    """Turn what an entry point resolved to into an adapter instance.

    Parameters
    ----------
    loaded : object
        Whatever `entry.load()` returned: an adapter class or a ready instance.

    Returns
    -------
    Adapter
        An instance constructed with no arguments when `loaded` is a class,
        or `loaded` itself otherwise.

    Raises
    ------
    TypeError
        If the result does not satisfy `Adapter`.
    """

    instance = loaded() if isinstance(loaded, type) else loaded
    if not isinstance(instance, Adapter):
        raise TypeError(f"{loaded!r} does not satisfy the Adapter protocol")
    return instance


def _load_plugin_file(
    path: Path, source: PluginSource
) -> tuple[list[LoadedAdapter], list[PluginLoadFailure]]:
    """Execute one plugin file and collect the adapters it registered.

    Parameters
    ----------
    path : Path
        The `.py` file to execute.
    source : PluginSource
        `PLUGIN_FILE` or `PLUGIN_DIR`, recorded against whatever this file registers.

    Returns
    -------
    tuple[list[LoadedAdapter], list[PluginLoadFailure]]
        The adapters this file registered,
        and a single-element failure list if executing it raised.
    """

    before = len(locally_registered())
    # Keyed by the resolved path, not the bare stem, so two plugin files
    # sharing a filename in different directories don't collide in sys.modules.
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
    module_name = f"kalanos_plugin_{path.stem}_{digest}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registered before exec so a plugin that pickles or dataclasses its
        # own classes can still resolve them by module name.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        logger.warning("plugin file %s failed to load: %s", path, exc)
        return [], [
            PluginLoadFailure(
                name=path.stem,
                source=source,
                origin=str(path),
                reason=describe_load_failure(exc),
            )
        ]

    registered = locally_registered()[before:]
    return [
        LoadedAdapter(
            name=instance.name, adapter=instance, source=source, origin=str(path)
        )
        for instance in registered
    ], []


def discover_adapters(
    *,
    plugin_files: Sequence[Path] = (),
    plugin_dir: Path | None = None,
    entry_points: bool = True,
) -> AdapterDiscovery:
    """Find every adapter available from entry points and local plugin files.

    Does not deduplicate by name. Two adapters answering to the same name
    are left for selection to sort out.

    Parameters
    ----------
    plugin_files : Sequence[Path]
        Individual plugin files to load, such as `--plugin FILE`.
    plugin_dir : Path or None
        A directory to scan for `*.py` plugin files. A missing directory is
        not a failure.
    entry_points : bool
        Whether to discover adapters registered under the `kalanos.adapters`
        entry-point group. Default `True`.

    Returns
    -------
    AdapterDiscovery
        Every adapter found and every load failure, from all three sources.
    """

    adapters: list[LoadedAdapter] = []
    failures: list[PluginLoadFailure] = []

    if entry_points:
        loaded_entries, entry_failures = load_group("kalanos.adapters")
        failures.extend(entry_failures)
        for item in loaded_entries:
            try:
                instance = _instantiate(item.value)
            except Exception as exc:
                logger.warning("entry point %s is not an adapter: %s", item.name, exc)
                failures.append(
                    PluginLoadFailure(
                        name=item.name,
                        source=PluginSource.ENTRY_POINT,
                        origin=item.origin,
                        reason=describe_load_failure(exc),
                    )
                )
                continue
            adapters.append(
                LoadedAdapter(
                    name=instance.name,
                    adapter=instance,
                    source=PluginSource.ENTRY_POINT,
                    origin=item.origin,
                )
            )

    for path in plugin_files:
        loaded, failed = _load_plugin_file(path, PluginSource.PLUGIN_FILE)
        adapters.extend(loaded)
        failures.extend(failed)

    if plugin_dir is not None and plugin_dir.is_dir():
        for path in sorted(plugin_dir.glob("*.py")):
            loaded, failed = _load_plugin_file(path, PluginSource.PLUGIN_DIR)
            adapters.extend(loaded)
            failures.extend(failed)

    return AdapterDiscovery(adapters=adapters, failures=failures)
