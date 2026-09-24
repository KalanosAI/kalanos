"""Write a rendered Report to disk.

Everything above `reporting` computes;
this is where a result finally leaves the process.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# Internal
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.registry import (
    RegisteredReporter,
    registered_reporters,
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _reporter_for(suffix: str) -> RegisteredReporter | None:
    """Find the registered reporter claiming one file suffix."""

    for entry in registered_reporters():
        if suffix in entry.extensions:
            return entry
    return None


def write_report(report: Report, path: Path) -> Path:
    """Render a Report and write it to `path`, picking the format from the suffix.

    Parameters
    ----------
    report : Report
        The report to render and write.
    path : Path
        Where to write it. `.json` writes the raw model; `.yaml`/`.yml` write
        YAML; `.html` renders a page.

    Returns
    -------
    Path
        `path`, unchanged, for a caller that wants to chain the call.

    Raises
    ------
    ValueError
        If no registered reporter claims `path`'s suffix.
    """

    entry = _reporter_for(path.suffix.lower())
    if entry is None:
        claimed = sorted(
            {
                extension
                for known in registered_reporters()
                for extension in known.extensions
            }
        )
        raise ValueError(
            f"{path}: unsupported report suffix {path.suffix!r}; "
            f"expected one of {claimed}"
        )

    # write_text truncates rather than appends, so a re-run replaces a
    # previous report at the same path instead of corrupting it.
    path.write_text(entry.render(report), encoding="utf-8")
    return path
