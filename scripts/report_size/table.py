"""Render the before and after measurements as a Markdown table."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# Internal
from . import config
from .grading import Measurement
from .versions import Version


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _describe(version: Version) -> str:
    """Name a version for the table header.

    Parameters
    ----------
    version : Version
        The version.

    Returns
    -------
    str
        A commit's label and short SHA, or a wheel's file name and key.
    """

    if version.is_wheel:
        return f"{Path(version.source).name} ({version.key})"
    return f"{version.label} ({version.key[:12]})"


def _change(before: float, after: float) -> str:
    """Format the change from `before` to `after` as a signed percentage.

    Parameters
    ----------
    before, after : float
        The two values.

    Returns
    -------
    str
        The change with one decimal place, `+0.0%` when it rounds to zero.
    """

    percent = round((after - before) / before * 100, 1)
    return f"{percent if percent != 0 else 0.0:+.1f}%"


def render_table(
    before: Version, after: Version, rows: list[tuple[str, Measurement, Measurement]]
) -> str:
    """Render the comparison as Markdown.

    Parameters
    ----------
    before, after : Version
        The two versions.
    rows : list[tuple[str, Measurement, Measurement]]
        Each dataset with its before and after measurements.

    Returns
    -------
    str
        A header naming the versions and grade options, then the table and a mean row.
    """

    profile = config.BUNDLE.relative_to(config.ROOT)
    options = " ".join([*config.GRADE_FLAGS, "--profile", str(profile)])
    lines = [
        f"Before: {_describe(before)}",
        f"After: {_describe(after)}",
        f"Grade options: {options}",
        "",
        "| Dataset | Episodes | Before bytes | After bytes"
        " | Before bytes/episode | After bytes/episode | Change |",
        "|---|---|---|---|---|---|---|",
    ]

    changes = []
    differing = []
    for dataset, old, new in rows:
        name, revision = dataset.split("@")
        episodes = f"{old.episodes:,}"
        if old.episodes != new.episodes:
            episodes = f"{old.episodes:,} → {new.episodes:,}"
            differing.append(name)
        changes.append((new.per_episode - old.per_episode) / old.per_episode * 100)
        cells = [
            f"{name}@{revision[:7]}",
            episodes,
            f"{old.bytes:,}",
            f"{new.bytes:,}",
            f"{old.per_episode:,.0f}",
            f"{new.per_episode:,.0f}",
            _change(old.per_episode, new.per_episode),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    def mean(values: list[float]) -> float:
        """Return the arithmetic mean of `values`."""

        return sum(values) / len(values)

    mean_change = round(mean(changes), 1)
    cells = [
        "Mean",
        "",
        f"{mean([old.bytes for _, old, _ in rows]):,.0f}",
        f"{mean([new.bytes for _, _, new in rows]):,.0f}",
        f"{mean([old.per_episode for _, old, _ in rows]):,.0f}",
        f"{mean([new.per_episode for _, _, new in rows]):,.0f}",
        f"{mean_change if mean_change != 0 else 0.0:+.1f}%",
    ]
    lines.append("| " + " | ".join(cells) + " |")

    if differing:
        lines += [
            "",
            "The two versions loaded a different number of episodes for "
            f"{', '.join(differing)}.",
        ]
    return "\n".join(lines)
