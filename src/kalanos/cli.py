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
from upath import UPath

# Internal
from kalanos import api
from kalanos.analysis.models.discovery import SourceLimits
from kalanos.analysis.models.eligibility import EligibilityStatus
from kalanos.analysis.models.errors import KalanosError
from kalanos.analysis.models.legacy import LegacyReport, load_any
from kalanos.analysis.models.metrics import Family
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import Report
from kalanos.analysis.reporting.card import render_terminal
from kalanos.analysis.reporting.render import render_json
from kalanos.analysis.reporting.write import write_report
from kalanos.assets.mapping import parse_map_argument
from kalanos.benchmark import (
    DEFAULT_SAMPLE,
    REFERENCE_DATASETS,
    render_markdown,
    run_benchmark,
)
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
    map_: Annotated[
        list[str] | None,
        typer.Option(
            "--map",
            help=(
                "Type one source field for this run, as FEATURE=TYPE, "
                "e.g. observation.state=proprio.joint_position. "
                "Repeatable; beats --map-file and the sidecar."
            ),
        ),
    ] = None,
    map_file: Annotated[
        Path | None,
        typer.Option(
            "--map-file",
            help=(
                "A YAML mapping file: `schema_version: 1` and a `features:` table "
                "of FEATURE: TYPE. Beats the sidecar."
            ),
        ),
    ] = None,
    no_sidecar: Annotated[
        bool,
        typer.Option(
            "--no-sidecar",
            help=(
                "Ignore a kalanos-map.yaml in or beside the graded path. "
                "--map, --map-file and --profile still apply."
            ),
        ),
    ] = False,
    profile: Annotated[
        Path | None,
        typer.Option(
            "--profile",
            help=(
                "A configuration bundle: binding, requirements, policy and "
                "execution sections. Defaults to the numeric-core scope."
            ),
        ),
    ] = None,
    tier: Annotated[
        ExecutionTier | None,
        typer.Option(
            "--tier",
            case_sensitive=False,
            help=(
                "Which capabilities to attempt. Never changes the requirements: "
                "a skipped required check makes episodes unknown."
            ),
        ),
    ] = None,
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on",
            help=(
                "Comma-separated eligibility statuses that make the audit fail "
                "(exit 1): any of blocked, review, unknown. "
                "Default blocked,unknown; a training gate adds review; "
                "blocked alone is exploratory."
            ),
        ),
    ] = "blocked,unknown",
) -> None:
    """Grade a recording, or every recording in a folder.

    Exit codes
    ----------
    0
        The audit completed and no episode carries a status in `fail_on`.
    1
        The audit completed and at least one episode carries such a status,
        or the inventory is incomplete (a refused source, or declared
        episodes that never loaded) and `unknown` is in `fail_on`.
        CI gate failure is not source corruption: `review` and `unknown` say
        the data needs a decision or more evidence.
    2
        Invalid configuration or an operational failure: a missing path, a
        malformed bundle, an unknown taxonomy type, an unwritable report.
        Takes precedence over 1 when both apply.

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
    map_ : list[str] or None
        `FEATURE=TYPE` overrides, each typing one source field for this run.
    map_file : Path or None
        A YAML mapping file of overrides, beaten by `map_`.
    no_sidecar : bool
        Ignore a `kalanos-map.yaml` in the graded root.

    Raises
    ------
    typer.Exit
        Code 2 when grading raised a `KalanosError` — `path` does not exist,
        is a remote dataset over a limit, held nothing to report on at all,
        two adapters tied on the same file,
        or a mapping override was malformed or matched nothing —
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

    # Step 0: validate the gate before any expensive work.
    try:
        gate_on = parse_fail_on(fail_on)
    except ValueError as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    # Step 1: grade. The reason is printed, not logged: it explains a non-zero exit,
    # and must reach the user even at a verbosity that silences ERROR records.
    try:
        # Pairs, not a dict: two `--map` flags for one feature must reach the
        # resolver so a contradiction is refused instead of last-one-wins.
        mapping = [parse_map_argument(text) for text in map_ or []]
        result = api.grade(
            path,
            limits=limits,
            mapping=mapping,
            mapping_file=map_file,
            sidecar=not no_sidecar,
            bundle=profile,
            tier=tier,
        )
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

    # Step 4: the decision gate, from the one place decisions live. An
    # incomplete audit fails the default gate; operational errors already
    # left with exit 2 above, so this never masks one.
    for warning in gate_warnings(result, gate_on):
        print(f"kalanos: {warning}", file=sys.stderr)
    if failing_statuses(result, gate_on):
        raise typer.Exit(code=1)


def parse_fail_on(text: str) -> set[EligibilityStatus]:
    """Parse `--fail-on`: a comma-separated set of non-pass statuses.

    Raises
    ------
    ValueError
        On an unknown status, or on `pass`, which cannot fail an audit.
    """

    allowed = {s.value: s for s in EligibilityStatus if s != EligibilityStatus.PASS}
    chosen = set()
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(
                f"--fail-on {item!r} is not one of {', '.join(sorted(allowed))}"
            )
        chosen.add(allowed[item])
    if not chosen:
        raise ValueError("--fail-on needs at least one status")
    return chosen


def failing_statuses(report: Report, gate_on: set[EligibilityStatus]) -> int:
    """How many gate conditions the report trips; 0 means the gate passes.

    Episode statuses count one each. An incomplete inventory — a refused
    source, or a declared count the adapter did not deliver — is unresolved
    evidence about episodes that have no identities, so it counts once
    under `unknown` without inventing an episode count. Under the default
    `blocked,unknown` an incomplete audit therefore fails, even when every
    loaded episode passed or none loaded at all. `--fail-on blocked`
    (exploratory) does not fail on it; `gate_warnings` names it instead.
    """

    counts = report.eligibility_counts
    if counts is None:
        return 0
    tripped = sum(
        {
            EligibilityStatus.BLOCKED: counts.blocked,
            EligibilityStatus.REVIEW: counts.review,
            EligibilityStatus.UNKNOWN: counts.unknown,
        }[status]
        for status in gate_on
    )
    if EligibilityStatus.UNKNOWN in gate_on and not counts.inventory_complete:
        tripped += 1
    return tripped


def gate_warnings(report: Report, gate_on: set[EligibilityStatus]) -> list[str]:
    """What the gate let through that a reader should still hear about."""

    counts, inventory = report.eligibility_counts, report.inventory
    warnings = []
    if (
        counts is not None
        and not counts.inventory_complete
        and EligibilityStatus.UNKNOWN not in gate_on
    ):
        refused = len(inventory.refused_sources) if inventory else 0
        gap = inventory.unresolved if inventory else 0
        warnings.append(
            "audit incomplete and not gated: "
            + ", ".join(
                part
                for part in (
                    f"{refused} source(s) refused" if refused else "",
                    f"{gap} declared episode(s) not loaded" if gap else "",
                )
                if part
            )
            + " (add 'unknown' to --fail-on to fail on this)"
        )
    return warnings


@app.command(help="Summarise a saved report: scope, decisions, readiness, reasons.")
def inspect(
    report: Annotated[Path, typer.Argument(help="A report JSON of any schema.")],
    episode: Annotated[
        str | None,
        typer.Option("--episode", help="Show one episode's reasons in full."),
    ] = None,
) -> None:
    """Read-only inspection of a saved report, current or legacy.

    Raises
    ------
    typer.Exit
        Code 2 when the file is missing or not a report.
    """

    try:
        loaded = load_any(UPath(report))
    except (OSError, ValueError) as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    console = _console()
    if isinstance(loaded, LegacyReport):
        console.print(
            f"schema {loaded.schema_version} (legacy) · sha256 {loaded.sha256}"
        )
        summary = loaded.summary
        console.print(
            f"{summary.n_episodes} episodes; gate lists {summary.n_gate_failing} "
            f"failing; {summary.n_score_train_ready} carry score.train_ready=true; "
            f"legacy readiness {summary.legacy_readiness}"
        )
        if loaded.contradictions:
            console.print(
                f"{len(loaded.contradictions)} contradiction(s): episodes both "
                "failing and train_ready=true"
            )
            for item in loaded.contradictions:
                console.print(f"  {item.episode_id}")
        console.print("not recorded in this schema: " + "; ".join(loaded.unknown))
        return

    scope = loaded.scope
    counts = loaded.eligibility_counts
    console.print(f"schema {loaded.schema_version}")
    if scope is not None:
        console.print(
            f"scope {scope.requirements_id} · policy {scope.policy_id} · "
            f"tier {scope.tier.value}"
        )
    if counts is not None:
        console.print(
            f"{counts.pass_count}/{counts.total} pass, {counts.blocked} blocked, "
            f"{counts.review} review, {counts.unknown} unknown"
        )
    if loaded.readiness is not None:
        r = loaded.readiness
        console.print(
            f"readiness {r.score:.2f}"
            if r.score is not None
            else "readiness undefined: " + "; ".join(r.reasons)
        )
    if loaded.sufficiency is not None:
        console.print(f"sufficiency {loaded.sufficiency.status.value}")
    table = _table("EPISODE", "STATUS", "REASONS")
    for item in loaded.episodes:
        if episode is not None and item.id != episode:
            continue
        e = item.eligibility
        if e is None:
            continue
        reasons = (
            "; ".join(r.detail or r.id for r in e.reasons)
            if episode
            else str(len(e.reasons))
        )
        table.add_row(item.id, e.status.value, reasons)
    console.print(table)


@app.command(
    help=(
        "Measure how often each metric fires on reference datasets, "
        "and how often it catches an injected defect."
    )
)
def benchmark(
    paths: Annotated[
        list[str] | None,
        typer.Argument(
            help="Datasets to benchmark. Defaults to the pinned reference datasets."
        ),
    ] = None,
    sample: Annotated[
        int,
        typer.Option(
            "--sample",
            min=0,
            help="How many episodes per dataset to inject defects into.",
        ),
    ] = DEFAULT_SAMPLE,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Write the result here: `.md` renders the table, "
                "`.json` dumps the model."
            ),
        ),
    ] = None,
) -> None:
    """Benchmark each dataset's benign firing and detection rates.

    Parameters
    ----------
    paths : list[str] or None
        Datasets to benchmark, each graded whole by one adapter.
        `None` benchmarks `REFERENCE_DATASETS`.
    sample : int
        How many episodes per dataset to inject defects into.
    out : Path or None
        Where to write the result, relative to the working directory.
        `None` prints the Markdown to stdout instead.

    Raises
    ------
    typer.Exit
        Code 2 when benchmarking raised a `KalanosError`,
        or when writing `out` failed or its suffix is neither `.md` nor `.json`.
    """

    # Step 1: refuse an unwritable suffix before the slow run, not after it.
    if out is not None and out.suffix not in {".md", ".json"}:
        print(f"kalanos: cannot write a benchmark as {out.suffix!r}", file=sys.stderr)
        raise typer.Exit(code=2)

    # Step 2: benchmark every dataset before writing anything.
    try:
        result = run_benchmark(paths or REFERENCE_DATASETS, sample=sample)
    except KalanosError as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    # Step 3: `out` is taken as given, outside `reports_dir`.
    if out is None:
        print(render_markdown(result), end="")
        return
    if out.suffix == ".md":
        text = render_markdown(result)
    else:
        text = result.model_dump_json(indent=2)
    try:
        out.write_text(text)
    except OSError as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc


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
