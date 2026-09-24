"""Verifies the Adapter protocol, local registration, and discovery.

Retrofitting these later is expensive, so they are pinned as tests
before any adapter exists.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib.metadata
import textwrap
from pathlib import Path

# External
import pytest
from upath import UPath


try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

# Internal
from kalanos.analysis.adapters import (
    adapter,
    clear_local_registry,
    discover_adapters,
    locally_registered,
)
from kalanos.analysis.adapters.discover import PluginSource, _instantiate
from kalanos.analysis.models.adapters import Adapter, DatasetInfo


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


REPO_ROOT = Path(__file__).parent.parent
_PLUGIN_BODY = textwrap.dedent(
    """\
    from kalanos.analysis.adapters import adapter


    @adapter
    class PluginAdapter:
        name = {name!r}

        def detect(self, path):
            return 1.0

        def describe(self, path):
            from kalanos.analysis.models.adapters import DatasetInfo

            return DatasetInfo(adapter=self.name, path=path)

        def episodes(self, path, sample=None):
            return iter(())
    """
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _ConformingAdapter:
    """A minimal adapter used only to exercise the protocol and registry."""

    name = "conforming"

    def detect(self, path: UPath) -> float:
        return 1.0

    def describe(self, path: UPath) -> DatasetInfo:
        return DatasetInfo(adapter=self.name, path=path)

    def episodes(self, path: UPath, sample: int | None = None):
        return iter(())


class _MissingEpisodes:
    """Has `detect` and `describe` but not `episodes` — not an Adapter."""

    name = "incomplete"

    def detect(self, path: UPath) -> float:
        return 1.0

    def describe(self, path: UPath) -> DatasetInfo:
        return DatasetInfo(adapter=self.name, path=path)


class _FakeDist:
    """Stands in for `importlib.metadata`'s distribution object on an entry point."""

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version


class _FakeEntryPoint:
    """Stands in for `importlib.metadata.EntryPoint` without touching real packages.

    `load_fn` is called, not returned, so a fake that must fail on import
    can raise from inside it exactly as a real broken module would.
    """

    def __init__(self, name: str, load_fn, dist: _FakeDist | None = None) -> None:
        self.name = name
        self.dist = dist
        self.value = f"fake:{name}"
        self._load_fn = load_fn

    def load(self):
        return self._load_fn()


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(autouse=True)
def _clear_registry():
    """Reset the local adapter registry before and after every test in this module."""

    clear_local_registry()
    yield
    clear_local_registry()


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_conforming_class_satisfies_the_protocol():
    """Verify a class with all three methods is Adapter-shaped, one missing isn't."""

    assert isinstance(_ConformingAdapter(), Adapter)
    assert not isinstance(_MissingEpisodes(), Adapter)


def test_the_decorator_registers_an_instance():
    """Verify @adapter puts a ready instance in locally_registered()."""

    adapter(_ConformingAdapter)
    registered = locally_registered()
    assert len(registered) == 1
    assert isinstance(registered[0], _ConformingAdapter)

    clear_local_registry()
    assert locally_registered() == []


def test_the_decorator_rejects_a_non_adapter():
    """Verify @adapter raises TypeError naming a class that doesn't satisfy Adapter."""

    with pytest.raises(TypeError, match="_MissingEpisodes"):
        adapter(_MissingEpisodes)


def test_entry_points_are_discovered_with_no_built_in_special_case(monkeypatch):
    """Verify a synthetic entry point is discovered the same way a built-in would be."""

    fake = _FakeEntryPoint(
        "conforming",
        lambda: _ConformingAdapter,
        dist=_FakeDist("kalanos-conforming", "1.2.3"),
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [fake])

    result = discover_adapters(entry_points=True)

    assert len(result.adapters) == 1
    loaded = result.adapters[0]
    assert loaded.name == "conforming"
    assert loaded.source == PluginSource.ENTRY_POINT
    assert loaded.origin == "kalanos-conforming 1.2.3"
    assert result.failures == []


def test_a_broken_entry_point_is_reported_and_the_others_still_load(monkeypatch):
    """Verify one raising entry point becomes a failure without stopping the rest."""

    def _raise():
        raise ImportError("No module named 'pyzed'")

    broken = _FakeEntryPoint("zed", _raise)
    good = _FakeEntryPoint("conforming", lambda: _ConformingAdapter)
    monkeypatch.setattr(
        importlib.metadata, "entry_points", lambda group: [broken, good]
    )

    result = discover_adapters()

    assert [loaded.name for loaded in result.adapters] == ["conforming"]
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.name == "zed"
    assert "No module named 'pyzed'" in failure.reason


def test_an_entry_point_resolving_to_a_non_adapter_is_a_failure(monkeypatch):
    """Verify an entry point loading to a plain int lands in failures, not adapters."""

    fake = _FakeEntryPoint("not_an_adapter", lambda: 42)
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [fake])

    result = discover_adapters()

    assert result.adapters == []
    assert len(result.failures) == 1
    assert result.failures[0].name == "not_an_adapter"


def test_a_plugin_file_is_loaded_from_a_path(tmp_path):
    """Verify a single plugin file registers its adapter with source=PLUGIN_FILE."""

    plugin = tmp_path / "my_adapter.py"
    plugin.write_text(_PLUGIN_BODY.format(name="fromfile"))

    result = discover_adapters(plugin_files=[plugin], entry_points=False)

    assert len(result.adapters) == 1
    loaded = result.adapters[0]
    assert loaded.name == "fromfile"
    assert loaded.source == PluginSource.PLUGIN_FILE
    assert loaded.origin == str(plugin)
    assert result.failures == []


def test_a_plugin_file_that_raises_on_import_is_reported(tmp_path):
    """Verify a plugin file whose body raises lands in failures, without aborting."""

    plugin = tmp_path / "broken_adapter.py"
    plugin.write_text("raise RuntimeError('bad plugin')\n")

    result = discover_adapters(plugin_files=[plugin], entry_points=False)

    assert result.adapters == []
    assert len(result.failures) == 1
    assert "bad plugin" in result.failures[0].reason


def test_the_plugin_directory_is_scanned(tmp_path):
    """Verify every .py file in a plugin directory is loaded with source=PLUGIN_DIR."""

    (tmp_path / "one.py").write_text(_PLUGIN_BODY.format(name="one"))
    (tmp_path / "two.py").write_text(_PLUGIN_BODY.format(name="two"))

    result = discover_adapters(plugin_dir=tmp_path, entry_points=False)

    assert {loaded.name for loaded in result.adapters} == {"one", "two"}
    assert all(loaded.source == PluginSource.PLUGIN_DIR for loaded in result.adapters)


def test_a_missing_plugin_directory_is_not_an_error(tmp_path):
    """Verify a nonexistent plugin directory yields no adapters and no failures."""

    result = discover_adapters(
        plugin_dir=tmp_path / "does-not-exist", entry_points=False
    )

    assert result.adapters == []
    assert result.failures == []


def test_declared_entry_points_resolve():
    """Verify every kalanos.adapters entry point in pyproject.toml is loadable."""

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = data["project"].get("entry-points", {}).get("kalanos.adapters", {})

    for name, target in declared.items():
        entry = importlib.metadata.EntryPoint(name, target, "kalanos.adapters")
        instance = _instantiate(entry.load())
        assert isinstance(instance, Adapter), f"{name} does not satisfy Adapter"
