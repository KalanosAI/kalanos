"""Lay out the before and after measurements, for Markdown or for the terminal."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass
from pathlib import Path

# External
from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text

# Internal
from . import config
from .grading import Measurement
from .versions import Version


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

COLUMNS = (
    "Dataset",
    "Episodes",
    "Before bytes",
    "After bytes",
    "Before bytes/episode",
    "After bytes/episode",
    "Change",
)

# The same columns with headers that fold to short words, so the table fits 80 columns.
TERMINAL_COLUMNS = (
    "Dataset",
    "Episodes",
    "Before bytes",
    "After bytes",
    "Before per episode",
    "After per episode",
    "Change",
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Comparison:
    """The formatted cells of a comparison, shared by both renderings.

    Attributes
    ----------
    header : list[str]
        The lines naming the two versions and the grade options.
    rows : list[list[str]]
        One row of cells per dataset, in `COLUMNS` order.
    mean : list[str]
        The mean row's cells.
    note : str | None
        A line saying which datasets loaded a different number of episodes, if any.
    """

    header: list[str]
    rows: list[list[str]]
    mean: list[str]
    note: str | None


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


def _percent(value: float) -> str:
    """Format a percentage with a sign and one decimal place.

    Parameters
    ----------
    value : float
        The percentage.

    Returns
    -------
    str
        For example `-12.3%`, and `+0.0%` when it rounds to zero.
    """

    rounded = round(value, 1)
    return f"{rounded if rounded != 0 else 0.0:+.1f}%"


def compare(
    before: Version, after: Version, rows: list[tuple[str, Measurement, Measurement]]
) -> Comparison:
    """Format every cell of the comparison.

    Parameters
    ----------
    before, after : Version
        The two versions.
    rows : list[tuple[str, Measurement, Measurement]]
        Each dataset with its before and after measurements.

    Returns
    -------
    Comparison
        The header lines, one row per dataset, the mean row and the episode note.
    """

    profile = config.BUNDLE.relative_to(config.ROOT)
    options = " ".join([*config.GRADE_FLAGS, "--profile", str(profile)])
    header = [
        f"Before: {_describe(before)}",
        f"After: {_describe(after)}",
        f"Grade options: {options}",
    ]

    cells = []
    changes = []
    differing = []
    for dataset, old, new in rows:
        name, revision = dataset.split("@")
        episodes = f"{old.episodes:,}"
        if old.episodes != new.episodes:
            episodes = f"{old.episodes:,} → {new.episodes:,}"
            differing.append(name)
        change = (new.per_episode - old.per_episode) / old.per_episode * 100
        changes.append(change)
        cells.append(
            [
                f"{name}@{revision[:7]}",
                episodes,
                f"{old.bytes:,}",
                f"{new.bytes:,}",
                f"{old.per_episode:,.0f}",
                f"{new.per_episode:,.0f}",
                _percent(change),
            ]
        )

    def mean(values: list[float]) -> float:
        """Return the arithmetic mean of `values`."""

        return sum(values) / len(values)

    mean_row = [
        "Mean",
        "",
        f"{mean([old.bytes for _, old, _ in rows]):,.0f}",
        f"{mean([new.bytes for _, _, new in rows]):,.0f}",
        f"{mean([old.per_episode for _, old, _ in rows]):,.0f}",
        f"{mean([new.per_episode for _, _, new in rows]):,.0f}",
        _percent(mean(changes)),
    ]

    note = None
    if differing:
        note = (
            "The two versions loaded a different number of episodes for "
            f"{', '.join(differing)}."
        )
    return Comparison(header=header, rows=cells, mean=mean_row, note=note)


def to_markdown(comparison: Comparison) -> str:
    """Render a comparison as Markdown, for pasting into a story or saving.

    Parameters
    ----------
    comparison : Comparison
        The formatted cells.

    Returns
    -------
    str
        The header lines, the table with its mean row, and the note if any.
    """

    def line(cells: list[str] | tuple[str, ...]) -> str:
        """Join cells into one Markdown table row."""

        return "| " + " | ".join(cells) + " |"

    separator = "|" + "|".join(["---"] * len(COLUMNS)) + "|"
    lines = [*comparison.header, "", line(COLUMNS), separator]
    lines += [line(cells) for cells in [*comparison.rows, comparison.mean]]
    if comparison.note:
        lines += ["", comparison.note]
    return "\n".join(lines)


def to_rich(comparison: Comparison, width: int) -> Group:
    """Render a comparison for the terminal.

    Each number column is as wide as its widest value or header word, and the dataset
    column takes the rest, folding names that don't fit. A smaller report per episode
    shows its change in green, a larger one in red.

    Parameters
    ----------
    comparison : Comparison
        The formatted cells.
    width : int
        The terminal width, in columns.

    Returns
    -------
    Group
        The header lines, the table with its mean row, and the note if any.
    """

    body = [*comparison.rows, comparison.mean]
    numbers = [
        max(*(len(cells[index]) for cells in body), *map(len, header.split()))
        for index, header in enumerate(TERMINAL_COLUMNS)
        if index > 0
    ]
    # Rich's own shrinking crops the last columns once the dataset names don't fit,
    # so the widths are set here. Each column is padded by one space on either side,
    # and a one-character divider sits between neighbouring columns.
    names = max(width - sum(numbers) - 3 * len(TERMINAL_COLUMNS) + 1, 1)

    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column(TERMINAL_COLUMNS[0], overflow="fold", max_width=names)
    for header, size in zip(TERMINAL_COLUMNS[1:], numbers, strict=True):
        table.add_column(header, justify="right", width=size)

    def styled(cells: list[str]) -> list[str | Text]:
        """Colour the change cell by its sign."""

        change = cells[-1]
        style = "green" if change.startswith("-") else "red"
        return [*cells[:-1], Text(change, style="" if change == "+0.0%" else style)]

    for cells in comparison.rows:
        table.add_row(*styled(cells))
    table.add_section()
    table.add_row(*styled(comparison.mean), style="bold")

    parts: list[Text | Table] = [Text(line) for line in comparison.header]
    parts += [Text(), table]
    if comparison.note:
        parts.append(Text(comparison.note, style="yellow"))
    return Group(*parts)
