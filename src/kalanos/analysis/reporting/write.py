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
from kalanos.analysis.reporting.render import render_json
from kalanos.analysis.reporting.stream import episode_data, write_report_stream


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _reporter_for(suffix: str) -> RegisteredReporter | None:
    """Find the registered reporter claiming one file suffix."""

    for entry in registered_reporters():
        if suffix in entry.extensions:
            return entry
    return None


def _write_json(report: Report, path: Path, *, compact: bool) -> Path:
    """Write a Report's JSON to `path` an episode at a time.

    Parameters
    ----------
    report : Report
        The report to write.
    path : Path
        Where to write it, replacing any file there.
    compact : bool
        Replace every key with a short alias.

    Returns
    -------
    Path
        `path`, unchanged.
    """

    with path.open("wb") as out:
        write_report_stream(
            report.model_copy(update={"episodes": []}),
            lambda: (episode_data(e) for e in report.episodes),
            out,
            compact=compact,
        )
    return path


def write_report(report: Report, path: Path, *, compact: bool = False) -> Path:
    """Render a Report and write it to `path`, picking the format from the suffix.

    Parameters
    ----------
    report : Report
        The report to render and write.
    path : Path
        Where to write it. `.json` writes the raw model; `.yaml`/`.yml` write
        YAML; `.html` renders a page.
    compact : bool
        Write the JSON with every key replaced by a short alias.
        `path` must then end in `.json`.

    Returns
    -------
    Path
        `path`, unchanged, for a caller that wants to chain the call.

    Raises
    ------
    ValueError
        If no registered reporter claims `path`'s suffix,
        or `compact` is set and `path` does not end in `.json`.
    """

    if compact:
        if path.suffix.lower() != ".json":
            raise ValueError(f"{path}: a compact report is JSON; use a .json path")
        return _write_json(report, path, compact=True)

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

    if entry.render is render_json:
        return _write_json(report, path, compact=False)
    # write_text truncates rather than appends, so a re-run replaces a
    # previous report at the same path instead of corrupting it.
    path.write_text(entry.render(report), encoding="utf-8")
    return path
