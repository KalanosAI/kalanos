"""Verifies the listing surface: what is installed, where from, and what failed.

Every test here drives `importlib.metadata.entry_points` through a fake, so the
listing is exercised against entry points this checkout does not have to install.
These rows exist to catch load failures, so they're pinned down here just as
carefully as the successes.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib
import importlib.metadata
import sys
from dataclasses import replace
from types import ModuleType

# External
import pytest
from upath import UPath

# Internal
import kalanos.analysis.metrics.registry as registry
from kalanos.analysis.entry_points import MissingDependency
from kalanos.analysis.metrics import timing
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.metrics import (
    Family,
    Level,
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.reporting import render
from kalanos.plugins import list_adapters, list_metrics, list_reporters


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _ListableAdapter:
    """A minimal adapter, enough for discovery to instantiate and list it."""

    name = "listable"

    def detect(self, path: UPath) -> float:
        return 1.0

    def describe(self, path: UPath) -> DatasetInfo:
        return DatasetInfo(adapter=self.name, path=path)

    def episodes(self, path: UPath, sample: int | None = None):
        return iter(())


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


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _serve(monkeypatch, **by_group) -> None:
    """Answer `entry_points(group=...)` from a per-group mapping, empty elsewhere."""

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: by_group.get(group, []),
    )


def _package_re_exporting(module_name: str, submodule_name: str) -> ModuleType:
    """Register a metric as a submodule of `module_name` would, and return the package.

    A plugin package commonly registers in `pkg.core` and re-exports from
    `pkg/__init__.py`, which leaves `RegisteredMetric.module` naming the
    submodule while the entry point names the package.
    """

    @metric(level=Level.STREAM, family=Family.MOTION)
    def _re_exported(ctx: StreamContext) -> MetricResult:
        return MetricResult(value=0.0, unit=None, status=MetricStatus.REPORT_ONLY)

    # The decoration recorded this test module; rewrite it to the submodule
    # the registration would really have run in.
    registry._REGISTRY[-1] = replace(
        registry._REGISTRY[-1], module=f"{module_name}.{submodule_name}"
    )

    return ModuleType(module_name)


def _broken(message: str):
    """Build a loader that raises, standing in for a package that fails on import."""

    def _raise():
        raise ImportError(message)

    return _raise


def _missing_module(name: str):
    """Build a loader that raises the way a missing import does, with `name` set."""

    def _raise():
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return _raise


def _needs(distribution: str, extra: str | None = None):
    """Build a loader that raises the way a plugin with an optional dependency does."""

    def _raise():
        raise MissingDependency(distribution, extra)

    return _raise


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_an_installed_adapter_is_listed_with_its_distribution(monkeypatch):
    """Verify a loaded adapter carries the distribution that declared it."""

    _serve(
        monkeypatch,
        **{
            "kalanos.adapters": [
                _FakeEntryPoint(
                    "listable",
                    lambda: _ListableAdapter,
                    dist=_FakeDist("kalanos-listable", "1.2.3"),
                )
            ]
        },
    )

    rows = list_adapters()

    assert [(row.name, row.origin, row.error) for row in rows] == [
        ("listable", "kalanos-listable 1.2.3", None)
    ]


def test_an_adapter_that_fails_to_import_is_listed_with_its_error(monkeypatch):
    """Verify a broken entry point appears with its error and does not hide the rest.

    Omitting it would make an installed-but-broken package indistinguishable
    from one that was never installed, which is the failure this listing exists
    to make visible.
    """

    _serve(
        monkeypatch,
        **{
            "kalanos.adapters": [
                _FakeEntryPoint("zed", _broken("No module named 'pyzed'")),
                _FakeEntryPoint("listable", lambda: _ListableAdapter),
            ]
        },
    )

    rows = {row.name: row for row in list_adapters()}

    assert rows["listable"].error is None
    broken = rows["zed"].error
    assert broken is not None and "No module named 'pyzed'" in broken


def test_a_plugin_behind_an_extra_names_the_extra_that_provides_it(monkeypatch):
    """Verify a plugin that named its extra is listed as absent, not as a crash."""

    _serve(
        monkeypatch,
        **{"kalanos.adapters": [_FakeEntryPoint("hdf5", _needs("kalanos", "hdf5"))]},
    )

    rows = {row.name: row for row in list_adapters()}

    error = rows["hdf5"].error
    assert error == "not installed — add kalanos[hdf5]"
    assert "Error" not in error
    assert "pip" not in error
    assert "uv" not in error


def test_a_third_party_plugin_names_its_own_distribution(monkeypatch):
    """Verify the rendered extra follows the plugin, not this distribution.

    A plugin published elsewhere gates on its own extra, and the listing has
    to send the reader to that install rather than to a kalanos one.
    """

    _serve(
        monkeypatch,
        **{"kalanos.adapters": [_FakeEntryPoint("bag", _needs("kalanos-ros", "bag"))]},
    )

    rows = {row.name: row for row in list_adapters()}

    assert rows["bag"].error == "not installed — add kalanos-ros[bag]"


def test_a_plugin_gated_on_a_whole_distribution_names_it_alone(monkeypatch):
    """Verify a plugin with no extra to name is rendered without empty brackets.

    An optional dependency is not always an extra of the plugin's own
    distribution: a plugin may gate on a separate package it does not require,
    and the install it points at is that package's name on its own.
    """

    _serve(
        monkeypatch,
        **{"kalanos.adapters": [_FakeEntryPoint("bag", _needs("rosbags"))]},
    )

    rows = {row.name: row for row in list_adapters()}

    assert rows["bag"].error == "not installed — add rosbags"


def test_a_missing_module_no_plugin_claimed_keeps_the_underlying_error(monkeypatch):
    """Verify an unclaimed missing module still prints the raw error.

    A plugin that named no extra is not optional: its dependencies should have
    arrived with its install, so this is a broken install and the import error
    is the message that leads somewhere.
    """

    _serve(
        monkeypatch,
        **{"kalanos.adapters": [_FakeEntryPoint("zed", _missing_module("pyzed"))]},
    )

    rows = {row.name: row for row in list_adapters()}

    error = rows["zed"].error
    assert error is not None
    assert "No module named 'pyzed'" in error
    assert "kalanos[" not in error


def test_a_load_failure_that_is_not_a_missing_module_is_reported_in_full(monkeypatch):
    """Verify a genuine crash still names its exception class."""

    def _malformed_policy():
        raise RuntimeError("policy file is malformed")

    _serve(
        monkeypatch,
        **{"kalanos.metrics": [_FakeEntryPoint("policy", _malformed_policy)]},
    )

    rows = {row.name: row for row in list_metrics()}

    assert rows["policy"].error == "RuntimeError: policy file is malformed"


@pytest.mark.parametrize(
    ("module", "dependencies", "extra"),
    [
        ("kalanos.analysis.adapters.hdf5", ("h5py",), "hdf5"),
        ("kalanos.analysis.adapters.mcap", ("mcap", "mcap_ros2"), "mcap"),
    ],
)
def test_an_optional_built_in_names_its_extra_when_its_dependency_is_absent(
    monkeypatch, module, dependencies, extra
):
    """Verify each optional built-in guards its own import.

    Every test above drives the rendering through a fake, so this is the only
    unit-level check that the adapters themselves raise what the rendering
    expects; without it a new optional built-in ships reporting a raw
    `ModuleNotFoundError` and nothing fails until a base install is run.

    Parameters
    ----------
    module : str
        The adapter module to import.
    dependencies : tuple of str
        The top-level modules its extra provides, made unimportable for the test.
    extra : str
        The extra the adapter is expected to name.
    """

    # `None` in `sys.modules` is what makes an installed module unimportable.
    # Every already-imported submodule needs the same treatment: a cached
    # `mcap.reader` is returned without the import system ever consulting
    # `mcap`, so blocking the top-level name alone leaves the import working.
    for dependency in dependencies:
        blocked = {dependency} | {
            name for name in sys.modules if name.startswith(f"{dependency}.")
        }
        for name in blocked:
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, module, raising=False)

    with pytest.raises(MissingDependency) as raised:
        importlib.import_module(module)

    assert raised.value.distribution == "kalanos"
    assert raised.value.extra == extra


def test_two_packages_claiming_one_adapter_name_are_both_listed(monkeypatch):
    """Verify a name collision shows as two rows rather than one silently winning."""

    _serve(
        monkeypatch,
        **{
            "kalanos.adapters": [
                _FakeEntryPoint(
                    "listable",
                    lambda: _ListableAdapter,
                    dist=_FakeDist("kalanos-listable", "1.0.0"),
                ),
                _FakeEntryPoint(
                    "listable",
                    lambda: _ListableAdapter,
                    dist=_FakeDist("kalanos-other", "2.0.0"),
                ),
            ]
        },
    )

    rows = list_adapters()

    assert [row.name for row in rows] == ["listable", "listable"]
    assert [row.origin for row in rows] == [
        "kalanos-listable 1.0.0",
        "kalanos-other 2.0.0",
    ]


def test_the_timing_metrics_are_listed_with_their_family_and_distribution(monkeypatch):
    """Verify a metric entry point lists every metric the module it named registered."""

    _serve(
        monkeypatch,
        **{
            "kalanos.metrics": [
                _FakeEntryPoint(
                    "timing",
                    lambda: timing,
                    dist=_FakeDist("kalanos", "0.1.0"),
                )
            ]
        },
    )

    rows = {row.name: row for row in list_metrics()}

    assert {"effective_hz", "dt_jitter_ms", "drop_rate"} <= rows.keys()
    assert all(row.family == Family.TIMING for row in rows.values())
    assert rows["drop_rate"].origin == "kalanos 0.1.0"
    assert rows["drop_rate"].error is None


def test_filtering_by_family_drops_other_families_but_keeps_failures(monkeypatch):
    """Verify --family filters loaded metrics without hiding what never imported.

    A failed entry point has no family to filter on, so dropping it would make
    `--family` a way to accidentally hide a broken install.
    """

    _serve(
        monkeypatch,
        **{
            "kalanos.metrics": [
                _FakeEntryPoint("timing", lambda: timing),
                _FakeEntryPoint("broken", _broken("No module named 'nowhere'")),
            ]
        },
    )

    rows = list_metrics(Family.MOTION)

    assert [row.name for row in rows] == ["broken"]
    broken = rows[0].error
    assert broken is not None and "No module named 'nowhere'" in broken


def test_a_metric_entry_point_that_is_not_a_module_is_a_failure(monkeypatch):
    """Verify a metric entry point resolving to a function is refused, naming the group.

    The group names a module imported for its registration side effect, so
    pointing it at the metric function itself registers nothing at all.
    """

    _serve(
        monkeypatch,
        **{"kalanos.metrics": [_FakeEntryPoint("wrong", lambda: len)]},
    )

    rows = list_metrics()

    assert len(rows) == 1
    assert rows[0].name == "wrong"
    refused = rows[0].error
    assert refused is not None
    assert "kalanos.metrics entry points must name a module" in refused


def test_the_built_in_reporters_are_listed_with_the_suffixes_they_claim(monkeypatch):
    """Verify each renderer lists under its own name, with the extensions it claims."""

    _serve(
        monkeypatch,
        **{
            "kalanos.reporters": [
                _FakeEntryPoint(
                    "render", lambda: render, dist=_FakeDist("kalanos", "0.1.0")
                )
            ]
        },
    )

    rows = {row.name: row for row in list_reporters()}

    assert rows["json"].extensions == (".json",)
    assert rows["yaml"].extensions == (".yaml", ".yml")
    assert rows["html"].extensions == (".html",)
    assert all(row.error is None for row in rows.values())


def test_a_plugin_registering_from_a_submodule_is_still_attributed(monkeypatch):
    """Verify a package re-exporting from a submodule is listed, not dropped.

    Attribution matches the registry's `module` against the imported one. An
    entry point naming the package while the decoration ran in `pkg.core` is
    the ordinary layout, and it must not fall through into neither the rows
    nor the failures.
    """

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))
    package = _package_re_exporting("kalanos_split", "core")
    _serve(
        monkeypatch,
        **{
            "kalanos.metrics": [
                _FakeEntryPoint(
                    "split", lambda: package, dist=_FakeDist("kalanos-split", "1.0.0")
                )
            ]
        },
    )

    rows = list_metrics()

    assert [(row.name, row.origin) for row in rows] == [
        ("_re_exported", "kalanos-split 1.0.0")
    ]


def test_a_broken_reporter_entry_point_is_listed_with_its_error(monkeypatch):
    """Verify the reporter group reports a failed import the way the others do."""

    _serve(
        monkeypatch,
        **{
            "kalanos.reporters": [
                _FakeEntryPoint("pdf", _broken("No module named 'weasyprint'"))
            ]
        },
    )

    rows = list_reporters()

    assert [(row.name, row.extensions) for row in rows] == [("pdf", ())]
    broken = rows[0].error
    assert broken is not None and "No module named 'weasyprint'" in broken


def test_a_reporter_entry_point_that_is_not_a_module_is_a_failure(monkeypatch):
    """Verify a reporter entry point pointed at a render function is refused."""

    _serve(
        monkeypatch,
        **{"kalanos.reporters": [_FakeEntryPoint("wrong", lambda: render.render_json)]},
    )

    rows = list_reporters()

    refused = rows[0].error
    assert refused is not None
    assert "kalanos.reporters entry points must name a module" in refused
