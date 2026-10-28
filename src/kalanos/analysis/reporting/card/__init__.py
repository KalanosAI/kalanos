"""Render a Report as a short, verdict-first summary for the terminal.

Every adapter's report renders through the same sections in the same order;
only the rows inside them follow the dataset.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import io
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet

# External
from rich.console import Console
from rich.text import Text

# Internal
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.card.sections import (
    CardContext,
    CardRow,
    card_rows,
    coverage,
    families,
    findings,
    flagged_channels,
    footer,
    headline,
    needs_attention,
    not_analysed,
    title,
    unmapped_types,
)
from kalanos.analysis.reporting.card.theme import ASCII, UNICODE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# fmt: off
_DEFAULT_WIDTH = 120
_MIN_WIDTH     = 40
# fmt: on

Section = Callable[[Report, CardContext], list[Text]]

SECTIONS: tuple[Section, ...] = (
    title,
    headline,
    families,
    flagged_channels,
    needs_attention,
    findings,
    coverage,
    not_analysed,
    footer,
)

_HEADINGS: dict[Section, str] = {
    families: "Family scores",
    flagged_channels: "Flagged channels",
    needs_attention: "Needs attention",
    findings: "Findings",
    coverage: "Coverage",
    not_analysed: "Not analysed",
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def render_terminal(
    report: Report,
    *,
    color: bool = False,
    width: int | None = None,
    report_path: str | None = None,
    episodes: Sequence[CardRow] | None = None,
    unmapped: AbstractSet[str] | None = None,
) -> str:
    """Render a Report as a report card for the terminal.

    Parameters
    ----------
    report : Report
        The report to render.
    color : bool
        Styled Unicode when set, plain ASCII otherwise.
        Off by default, so a captured string stays plain.
    width : int or None
        The column count to lay the card out against, floored at `_MIN_WIDTH`.
        `None` falls back to a fixed default.
    report_path : str or None
        Where the full report was written, for the footer to point at.
    episodes : Sequence[CardRow] or None
        The episodes that did not pass, when `report` holds none of its episodes.
        `None` reads them from `report.episodes`.
    unmapped : AbstractSet[str] or None
        The unmapped stream types, when `report` holds none of its episodes.
        `None` reads them from `report.episodes`.

    Returns
    -------
    str
        The rendered card, without a trailing newline.
    """

    ctx = CardContext(
        theme=UNICODE if color else ASCII,
        width=max(width or _DEFAULT_WIDTH, _MIN_WIDTH),
        report_path=report_path,
        episodes=tuple(card_rows(report.episodes) if episodes is None else episodes),
        unmapped_types=frozenset(
            unmapped_types(report.episodes) if unmapped is None else unmapped
        ),
    )
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=color,
        no_color=not color,
        width=ctx.width,
        # Explicit, so a TERM=dumb environment cannot veto a forced color.
        color_system="truecolor" if color else None,
        highlight=False,
        markup=False,
        emoji=False,
    )

    for section in SECTIONS:
        lines = section(report, ctx)
        if not lines:
            continue
        if section is not title:
            console.print()
        if section in _HEADINGS:
            console.print(Text(_HEADINGS[section], style=ctx.theme.style("heading")))
        for line in lines:
            # Lines arrive fitted to the width; the one that is not means to overflow.
            console.print(line, no_wrap=True, overflow="ignore", crop=False)

    return buffer.getvalue().rstrip("\n")
