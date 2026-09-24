"""The `kalanos` command group.

`grade` is the first subcommand, not the only one. Everything here parses
arguments and prints; the work belongs to `kalanos.analysis`, so that a
second command reuses the pipeline rather than reimplementing part of it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import shutil
import sys
from pathlib import Path
from typing import Annotated

# External
import typer
from rich import box
from rich.console import Console
from rich.table import Table

# Internal
from kalanos import api
from kalanos.analysis.models.discovery import SourceLimits
from kalanos.analysis.models.errors import KalanosError
from kalanos.analysis.models.metrics import Family
from kalanos.analysis.reporting.card import render_terminal
from kalanos.analysis.reporting.render import render_json
from kalanos.analysis.reporting.write import write_report
from kalanos.core.log import Verbosity, configure_logging
from kalanos.core.settings import get_settings
from kalanos.plugins import list_adapters, list_metrics, list_reporters
from kalanos.scaffold import scaffold_adapter, scaffold_metric


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


app = typer.Typer(
    name="kalanos",
    help="Score sensor and telemetry recordings for training-data quality.",
    no_args_is_help=True,
    add_completion=False,
)

# What a listing is laid out against when stdout is not a terminal, so a
# captured row does not depend on the width of whoever ran it.
_LISTING_WIDTH = 120


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _console() -> Console:
    """Open a console for a listing, sized to the terminal when there is one."""

    is_terminal = sys.stdout.isatty()
    width = shutil.get_terminal_size().columns if is_terminal else _LISTING_WIDTH
    # Plugin names and load errors are arbitrary third-party text, not markup.
    return Console(width=width, highlight=False, markup=False)


def _table(*headers: str) -> Table:
    """Build an unboxed listing table with one left-aligned column per header."""

    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    for header in headers:
        table.add_column(header, overflow="fold")
    return table


def _status(error: str | None) -> str:
    """Render a row's load outcome as the STATUS column shows it."""

    return "ok" if error is None else f"FAILED: {error}"


@app.command(help="Grade a recording, or every recording in a folder.")
def grade(
    path: Annotated[
        str,
        typer.Argument(help="A recording to grade, or a folder of them."),
    ],
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help=(
                "Write the report here. `.html` renders it, "
                "`.json`/`.yaml`/`.yml` dump the model."
            ),
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the Report model to stdout instead of the report card.",
        ),
    ] = False,
    max_remote_gb: Annotated[
        float | None,
        typer.Option(
            "--max-remote-gb",
            help=(
                "Refuse a remote dataset larger than this. "
                "Defaults to KALANOS_REMOTE_MAX_BYTES (20 GB)."
            ),
        ),
    ] = None,
    max_remote_files: Annotated[
        int | None,
        typer.Option(
            "--max-remote-files",
            help=(
                "Refuse a remote dataset with more files than this. "
                "Defaults to KALANOS_REMOTE_MAX_FILES (10,000)."
            ),
        ),
    ] = None,
) -> None:
    """Grade a recording, or every recording in a folder.

    Parameters
    ----------
    path : str
        File or folder to analyse, parsed as a `UPath`.
    report : Path or None
        Where to write the report. `.html` renders it, `.json`/`.yaml`/`.yml`
        dump the model. Omitting it only affects the file write — stdout is
        decided by `as_json` either way.
    as_json : bool
        Print the Report model to stdout instead of the report card.
    max_remote_gb : float or None
        Refuse a remote dataset whose listed size is over this many gigabytes.
        `None` falls back to `Settings.remote_max_bytes`.
    max_remote_files : int or None
        Refuse a remote dataset listing more files than this.
        `None` falls back to `Settings.remote_max_files`.

    Raises
    ------
    typer.Exit
        Code 2 when grading raised a `KalanosError` — `path` does not exist,
        is a remote dataset over a limit, held nothing to report on at all,
        or two adapters tied on the same file —
        or when writing `report` failed: an unsupported suffix, a missing directory,
        an unwritable path.
        The reason goes to stderr and nothing is written to stdout.
    """

    settings = get_settings()
    limits = SourceLimits(
        max_bytes=int(max_remote_gb * 1e9)
        if max_remote_gb is not None
        else settings.remote_max_bytes,
        max_files=max_remote_files
        if max_remote_files is not None
        else settings.remote_max_files,
    )

    # Step 1: grade. The reason is printed, not logged: it explains a non-zero exit,
    # and must reach the user even at a verbosity that silences ERROR records.
    try:
        result = api.grade(path, limits=limits)
    except KalanosError as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    # Step 2: write the report file, if asked, before anything reaches stdout,
    # so a failed write leaves stdout empty rather than a half card.
    if report is not None:
        destination = (
            report if report.is_absolute() else get_settings().reports_dir / report
        )
        try:
            write_report(result, destination)
        except (ValueError, OSError) as exc:
            print(f"kalanos: {exc}", file=sys.stderr)
            raise typer.Exit(code=2) from exc

    # Step 3: stdout is the model for a caller that wants to pipe it onward,
    # or the report card for a person reading the terminal.
    if as_json:
        print(render_json(result))
    else:
        is_terminal = sys.stdout.isatty()
        width = shutil.get_terminal_size().columns if is_terminal else None
        print(render_terminal(result, color=is_terminal, width=width))


@app.command(help="Summarise every installed plugin, and report what failed to load.")
def plugins() -> None:
    """Summarise the three plugin groups, then name whatever failed to load."""

    groups = {
        "adapters": list_adapters(),
        "metrics": list_metrics(),
        "reporters": list_reporters(),
    }

    counts = _table("GROUP", "LOADED", "FAILED")
    for group, rows in groups.items():
        failed = [row for row in rows if row.error is not None]
        counts.add_row(group, str(len(rows) - len(failed)), str(len(failed)))
    _console().print(counts)

    failures = _table("GROUP", "NAME", "FROM", "ERROR")
    for group, rows in groups.items():
        for row in rows:
            if row.error is not None:
                failures.add_row(group, row.name, row.origin, row.error)
    if failures.row_count:
        console = _console()
        console.print("FAILED")
        console.print(failures)


@app.command(help="List every installed adapter and where it came from.")
def adapters() -> None:
    """List every installed adapter, loaded or not."""

    table = _table("NAME", "FROM", "STATUS")
    for row in list_adapters():
        table.add_row(row.name, row.origin, _status(row.error))
    _console().print(table)


@app.command(help="List every installed metric and where it came from.")
def metrics(
    family: Annotated[
        Family | None,
        typer.Option(
            "--family",
            case_sensitive=False,
            help="Show only metrics asking this question.",
        ),
    ] = None,
) -> None:
    """List every installed metric, loaded or not.

    Parameters
    ----------
    family : Family or None
        Keep only metrics of this family. Load failures are listed either way,
        since an entry point that never imported has no family to filter on.
    """

    table = _table("NAME", "LEVEL", "FAMILY", "REQUIRES", "FROM", "STATUS")
    for row in list_metrics(family):
        table.add_row(
            row.name,
            row.level.value if row.level else "",
            row.family.value if row.family else "",
            row.requires,
            row.origin,
            _status(row.error),
        )
    _console().print(table)


@app.command(help="List every installed reporter and the suffixes it claims.")
def reporters() -> None:
    """List every installed reporter, loaded or not."""

    table = _table("NAME", "EXTENSIONS", "FROM", "STATUS")
    for row in list_reporters():
        table.add_row(
            row.name, " ".join(row.extensions), row.origin, _status(row.error)
        )
    _console().print(table)


new_app = typer.Typer(
    name="new",
    help="Write a publishable plugin package to start from.",
    no_args_is_help=True,
)
app.add_typer(new_app)


@new_app.command("adapter", help="Write an adapter package for a new format.")
def new_adapter(
    name: Annotated[
        str,
        typer.Argument(help="The adapter's name, which is also its file suffix."),
    ],
    into: Annotated[
        Path,
        typer.Option("--into", help="The directory to write the package into."),
    ] = Path(),
) -> None:
    """Write a publishable adapter package.

    Parameters
    ----------
    name : str
        The adapter's name.
    into : Path
        The parent directory to write the package into.

    Raises
    ------
    typer.Exit
        Code 2 when `name` is unusable or the target directory already exists.
    """

    try:
        written = scaffold_adapter(name, into)
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    print(f"Wrote {written}")
    print(f"Next: pip install -e {written} && pytest {written}")


@new_app.command("metric", help="Write a metric package for a new measurement.")
def new_metric(
    name: Annotated[
        str,
        typer.Argument(help="The metric's name, as a listing and a policy print it."),
    ],
    family: Annotated[
        Family,
        typer.Option(
            "--family",
            case_sensitive=False,
            help="The question the metric asks.",
        ),
    ] = Family.INTEGRITY,
    into: Annotated[
        Path,
        typer.Option("--into", help="The directory to write the package into."),
    ] = Path(),
) -> None:
    """Write a publishable metric package.

    Parameters
    ----------
    name : str
        The metric's name.
    family : Family
        The question the metric asks.
    into : Path
        The parent directory to write the package into.

    Raises
    ------
    typer.Exit
        Code 2 when `name` is unusable or the target directory already exists.
    """

    try:
        written = scaffold_metric(name, family, into)
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    print(f"Wrote {written}")
    print(f"Next: pip install -e {written} && pytest {written}")


@app.callback()
def main(
    verbosity: Annotated[
        Verbosity | None,
        typer.Option(
            "--verbosity",
            "-v",
            case_sensitive=False,
            help="How much of the run to report on stderr.",
        ),
    ] = None,
) -> None:
    """Anchor the command group, and configure logging before any subcommand runs.

    Typer collapses a single-command app into a bare `kalanos <args>`.
    An explicit callback keeps the group form from the start,
    so adding a command changes nothing a user types.

    Parameters
    ----------
    verbosity : Verbosity or None
        Overrides `Settings.verbosity` for this invocation.
        `None` falls back to the settings default.
    """

    configure_logging(verbosity if verbosity is not None else get_settings().verbosity)


if __name__ == "__main__":
    app()
