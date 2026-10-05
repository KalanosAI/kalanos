"""The two looks the report card comes in: styled Unicode, or plain ASCII."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass

# External
from rich.cells import cell_len
from rich.text import Text


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Similar colors from the HTML report
# fmt: off
_GOOD    = "#1c8c91"
_WARNING = "#b07107"
_ACCENT  = "#e63f32"
_RULE    = "#807c74"
# fmt: on

_STYLES = {
    "heading": "bold",
    "muted": _RULE,
    "good": _GOOD,
    "warn": _WARNING,
    "bad": _ACCENT,
}


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Theme:
    """The glyphs and styles one rendering of the card uses.

    Attributes
    ----------
    sep : str
        Separates the parts of one cell.
    ellipsis : str
        Ends a cell cut short to fit the width.
    color : bool
        Whether `style` returns anything at all.
    """

    sep: str
    ellipsis: str
    color: bool

    def style(self, role: str) -> str:
        """Resolve a role to a rich style.

        Parameters
        ----------
        role : str
            One of `heading`, `muted`, `good`, `warn` or `bad`.

        Returns
        -------
        str
            The style for `role`, or `""` when the theme has no color.
        """

        return _STYLES[role] if self.color else ""

    def fit(self, text: Text | str, width: int) -> Text:
        """Cut `text` down to `width` cells, marking the cut with `ellipsis`.

        Parameters
        ----------
        text : Text or str
            The text to fit. Its styles stay on the part that survives.
        width : int
            The most cells the result may span.

        Returns
        -------
        Text
            `text` unchanged when it fits, else its head and `ellipsis`.
            A width too small for the mark gets the bare head.
        """

        fitted = Text(text) if isinstance(text, str) else text.copy()
        width = max(width, 0)
        if fitted.cell_len <= width:
            return fitted
        mark = self.ellipsis if cell_len(self.ellipsis) < width else ""
        fitted.truncate(width - cell_len(mark), overflow="crop")
        fitted.rstrip()
        fitted.append(mark)
        return fitted


UNICODE = Theme(sep="·", ellipsis="…", color=True)
ASCII = Theme(sep="-", ellipsis="...", color=False)
