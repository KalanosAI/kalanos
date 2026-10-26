"""Command line for the report-size benchmark."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import subprocess
import sys
from pathlib import Path
from typing import Annotated

# External
import typer
from rich.console import Console

# Internal
from . import __doc__ as usage
from . import compact as compact_table
from . import config
from .grading import GradeFailed, ensure_measurement, report_path
from .hub import pin_dataset
from .table import compare, to_markdown, to_rich
from .versions import merge_base, resolve_version


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

# Markdown mode reflows the package docstring, which is wrapped at 88 columns.
app = typer.Typer(add_completion=False, rich_markup_mode="markdown")


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


@app.command(help=usage)
def main(
    after: Annotated[
        str, typer.Option(help="The version to measure: a git ref or a .whl path.")
    ] = "HEAD",
    before: Annotated[
        str | None,
        typer.Option(
            help="The version to compare against. Defaults to the merge-base of "
            "--after and --base."
        ),
    ] = None,
    base: Annotated[
        str,
        typer.Option(
            help="The branch --after started from, to find --before: "
            "origin/release/X.Y.Z for work on a release."
        ),
    ] = config.DEFAULT_BASE,
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            help="A Hugging Face dataset as owner/name or owner/name@revision, "
            "repeatable. Without a revision, the latest commit is used. Replaces "
            "the default set."
        ),
    ] = None,
    cache_dir: Annotated[
        Path,
        typer.Option(
            help="Where reports and their sizes are cached.",
            show_default="reports/report-size",
        ),
    ] = config.CACHE_DIR,
    output: Annotated[
        Path | None,
        typer.Option(help="Also write the Markdown table to this file."),
    ] = None,
    compact: Annotated[
        bool,
        typer.Option(
            help="Also measure the compact encoding of each --after report: size, "
            "encode time, load time and peak memory."
        ),
    ] = False,
) -> None:
    """Measure every dataset at both versions and print the table.

    Parameters
    ----------
    after : str
        The version to measure.
    before : str | None
        The version to compare against, or None for the merge-base of `after` and
        `base`.
    base : str
        The branch `after` started from.
    dataset : list[str] | None
        The datasets to grade, or None for the defaults.
    cache_dir : Path
        Where reports and their sizes are cached.
    output : Path | None
        A file to write the Markdown table to, or None.
    compact : bool
        Also measure each `after` report's compact encoding, in a second table.

    Raises
    ------
    typer.Exit
        With code 1 when a version or dataset doesn't resolve, a grade fails, a
        compact measurement fails or the table can't be written.
    """

    # kalanos grade puts a relative --report under KALANOS_REPORTS_DIR.
    cache = cache_dir.resolve()
    try:
        datasets = [pin_dataset(spec) for spec in dataset or config.DATASETS]
        new_side = resolve_version(after, config.ROOT)
        old_side = (
            resolve_version(before, config.ROOT)
            if before is not None
            else merge_base(new_side, base, config.ROOT)
        )
        old = [ensure_measurement(old_side, d, config.ROOT, cache) for d in datasets]
        new = [ensure_measurement(new_side, d, config.ROOT, cache) for d in datasets]
    except (ValueError, GradeFailed, subprocess.CalledProcessError) as exc:
        print(f"report_size: {exc}", file=sys.stderr)
        raise typer.Exit(code=1) from exc

    comparison = compare(old_side, new_side, list(zip(datasets, old, new, strict=True)))
    if sys.stdout.isatty():
        console = Console(highlight=False)
        console.print(to_rich(comparison, console.width))
    else:
        print(to_markdown(comparison))

    markdown = to_markdown(comparison)
    if compact:
        try:
            measured = [
                (d, compact_table.measure_compact(report_path(new_side, d, cache)))
                for d in datasets
            ]
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"report_size: {exc}", file=sys.stderr)
            raise typer.Exit(code=1) from exc
        compact_markdown = compact_table.to_markdown(measured)
        print(f"\n{compact_markdown}")
        markdown += f"\n\n{compact_markdown}"

    # Written after printing, so a bad path still leaves the table on screen.
    if output is not None:
        try:
            output.write_text(markdown + "\n")
        except OSError as exc:
            print(f"report_size: {exc}", file=sys.stderr)
            raise typer.Exit(code=1) from exc
        print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    app()
