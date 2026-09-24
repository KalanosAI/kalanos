"""What is installed, where it came from, and whether it loaded.

One row per entry, across all three plugin groups. Entries are not deduplicated by name,
so a built-in and an installed package claiming the same name both show up.
Hiding that here would just bury the collision instead of surfacing it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass

# Internal
from kalanos.analysis.adapters.discover import discover_adapters
from kalanos.analysis.metrics.discover import discover_metrics
from kalanos.analysis.models.metrics import Family, Level, Requires
from kalanos.analysis.reporting.discover import discover_reporters


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_NO_REQUIREMENTS = "—"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class AdapterRow:
    """One adapter as a listing prints it.

    Attributes
    ----------
    name : str
        The adapter's name, or the entry point's name when it failed to load.
    origin : str
        The distribution or file path it came from.
    error : str or None
        Why it failed to load, or `None` when it loaded.
    """

    name: str
    origin: str
    error: str | None


@dataclass(frozen=True)
class MetricRow:
    """One metric as a listing prints it.

    A failed entry point has no registered metric behind it, so `level`,
    `family` and `requires` carry only what discovery could know.

    Attributes
    ----------
    name : str
        The metric's name, or the entry point's name when it failed to load.
    level : Level or None
        Where in the rollup it attaches.
    family : Family or None
        The question it asks.
    requires : str
        Its requirements as a short phrase.
    origin : str
        The distribution it came from.
    error : str or None
        Why it failed to load, or `None` when it loaded.
    """

    name: str
    level: Level | None
    family: Family | None
    requires: str
    origin: str
    error: str | None


@dataclass(frozen=True)
class ReporterRow:
    """One reporter as a listing prints it.

    Attributes
    ----------
    name : str
        The reporter's name, or the entry point's name when it failed to load.
    extensions : tuple[str, ...]
        The suffixes it claims.
    origin : str
        The distribution it came from.
    error : str or None
        Why it failed to load, or `None` when it loaded.
    """

    name: str
    extensions: tuple[str, ...]
    origin: str
    error: str | None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _describe_requires(requires: Requires) -> str:
    """Render a metric's requirements as a phrase short enough for a table cell."""

    parts = []
    if requires.regular_sampling:
        parts.append("regular sampling")
    if requires.min_samples:
        parts.append(f"≥{requires.min_samples} samples")
    return ", ".join(parts) if parts else _NO_REQUIREMENTS


def list_adapters() -> list[AdapterRow]:
    """List every adapter installed in this process, loaded or not.

    Uses the same discovery a `grade` run would, so the listing matches
    what the pipeline can actually reach for.

    Returns
    -------
    list[AdapterRow]
        One row per adapter and per load failure, sorted by name and origin.
    """

    discovery = discover_adapters()
    rows = [
        AdapterRow(name=loaded.name, origin=loaded.origin, error=None)
        for loaded in discovery.adapters
    ]
    rows.extend(
        AdapterRow(name=failure.name, origin=failure.origin, error=failure.reason)
        for failure in discovery.failures
    )
    return sorted(rows, key=lambda row: (row.name, row.origin))


def list_metrics(family: Family | None = None) -> list[MetricRow]:
    """List every metric installed in this process, loaded or not.

    Parameters
    ----------
    family : Family or None
        Keep only metrics of this family. Load failures are kept either way:
        an entry point that never imported has no family to filter on.

    Returns
    -------
    list[MetricRow]
        One row per metric and per load failure, sorted by name and origin.
    """

    discovery = discover_metrics()
    rows = [
        MetricRow(
            name=loaded.name,
            level=loaded.metric.level,
            family=loaded.metric.family,
            requires=_describe_requires(loaded.metric.requires),
            origin=loaded.origin,
            error=None,
        )
        for loaded in discovery.metrics
        if family is None or loaded.metric.family == family
    ]
    rows.extend(
        MetricRow(
            name=failure.name,
            level=None,
            family=None,
            requires=_NO_REQUIREMENTS,
            origin=failure.origin,
            error=failure.reason,
        )
        for failure in discovery.failures
    )
    return sorted(rows, key=lambda row: (row.name, row.origin))


def list_reporters() -> list[ReporterRow]:
    """List every reporter installed in this process, loaded or not.

    Returns
    -------
    list[ReporterRow]
        One row per reporter and per load failure, sorted by name and origin.
    """

    discovery = discover_reporters()
    rows = [
        ReporterRow(
            name=loaded.name,
            extensions=loaded.reporter.extensions,
            origin=loaded.origin,
            error=None,
        )
        for loaded in discovery.reporters
    ]
    rows.extend(
        ReporterRow(
            name=failure.name,
            extensions=(),
            origin=failure.origin,
            error=failure.reason,
        )
        for failure in discovery.failures
    )
    return sorted(rows, key=lambda row: (row.name, row.origin))
