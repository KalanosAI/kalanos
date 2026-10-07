"""The `kalanos` command group.

`grade` is the first subcommand, not the only one.
Everything here parses arguments and prints; the work belongs to `kalanos.analysis`,
so that a second command reuses the pipeline rather than reimplementing part of it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import logging
import os
import shutil
import sys
from contextlib import nullcontext
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
from kalanos.analysis.coverage import coverage_lines
from kalanos.analysis.models.binding import (
    BUILT_IN_PROFILES,
    Bundle,
    RequirementsSection,
)
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
from kalanos.assets.bundle import load_bundle, load_bundle_policy
from kalanos.assets.mapping import parse_map_argument
from kalanos.benchmark import (
    DEFAULT_SAMPLE,
    REFERENCE_DATASETS,
    render_markdown,
    run_benchmark,
)
from kalanos.core.log import Verbosity, configure_logging, logging_to
from kalanos.core.settings import get_settings
from kalanos.plugins import list_adapters, list_metrics, list_reporters
from kalanos.publish import (
    PublishError,
    check_hub_url,
    check_key,
    check_name,
    load_report,
    publish_report,
)
from kalanos.scaffold import scaffold_adapter, scaffold_metric


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Listing width when stdout is not a terminal, so rows do not depend on who ran it.
_LISTING_WIDTH = 120


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


app = typer.Typer(
    name="kalanos",
    help="Score sensor and telemetry recordings for training-data quality.",
    no_args_is_help=True,
    add_completion=False,
)


new_app = typer.Typer(
    name="new",
    help="Write a publishable plugin package to start from.",
    no_args_is_help=True,
)
app.add_typer(new_app)


profiles_app = typer.Typer(help="Inspect and validate requirements bundles.")
app.add_typer(profiles_app, name="profiles")


diagnostics_app = typer.Typer(
    help="Inspect explicit diagnostic plans and validation evidence."
)
app.add_typer(diagnostics_app, name="diagnostics")


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
    color: Annotated[
        bool | None,
        typer.Option(
            "--color/--no-color",
            help=(
                "Style the report card for a terminal, or print plain ASCII. "
                "Defaults to styled when stdout is a terminal; "
                "NO_COLOR also turns it off."
            ),
        ),
    ] = None,
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
    vision_samples: Annotated[
        int | None,
        typer.Option(
            "--vision-samples",
            min=1,
            help=(
                "Frames sharpness and exposure sample per camera, "
                "and 1 s windows the frozen-frame metric reads. "
                "Defaults to KALANOS_VISION_SAMPLES, "
                "then the bundle's vision section (10)."
            ),
        ),
    ] = None,
    full_frame_scan: Annotated[
        bool,
        typer.Option(
            "--full-frame-scan",
            help=(
                "Read every video frame for frame metrics instead of a sample. "
                "Slower; on a remote dataset it downloads every video segment. "
                "Implied by --tier full."
            ),
        ),
    ] = False,
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
                "A configuration bundle file, or a built-in profile name from "
                "`kalanos profiles list`. Defaults to the numeric-core scope."
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
    hash_source: Annotated[
        bool,
        typer.Option(
            "--hash-source",
            help="Hash local source bytes before and after analysis for comparison.",
        ),
    ] = False,
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
        or the inventory is incomplete
        (a refused source, or declared episodes that never loaded)
        and `unknown` is in `fail_on`.
        CI gate failure is not source corruption:
        `review` and `unknown` say the data needs a decision or more evidence.
    2
        Invalid configuration or an operational failure: a missing path,
        a malformed bundle, an unknown taxonomy type, an unwritable report.
        Takes precedence over 1 when both apply.

    Parameters
    ----------
    path : str
        File or folder to analyse, parsed as a `UPath`.
    report : Path or None
        Where to write the report.
        `.html` renders it, `.json`/`.yaml`/`.yml` dump the model.
        Omitting it only affects the file write —
        stdout is decided by `as_json` either way.
    as_json : bool
        Print the Report model to stdout instead of the report card.
    color : bool or None
        Style the card with color and Unicode, or print it as plain ASCII.
        `None` styles it only when stdout is a terminal and `NO_COLOR` is unset.
    max_remote_gb : float or None
        Refuse a remote dataset whose listed size is over this many gigabytes.
        `None` falls back to `Settings.remote_max_bytes`.
    max_remote_files : int or None
        Refuse a remote dataset listing more files than this.
        `None` falls back to `Settings.remote_max_files`.
    vision_samples : int or None
        Frames sharpness and exposure sample per camera,
        and 1 s windows the frozen-frame metric reads.
        `None` falls back to `Settings.vision_samples`,
        then the bundle's `vision` section.
    full_frame_scan : bool
        Read every video frame for frame metrics instead of a sample.
        When absent, `Settings.full_frame_scan` decides,
        then the bundle's `vision` section, or `--tier full` implies it.
    map_ : list[str] or None
        `FEATURE=TYPE` overrides, each typing one source field for this run.
    map_file : Path or None
        A YAML mapping file of overrides, beaten by `map_`.
    no_sidecar : bool
        Ignore a `kalanos-map.yaml` in the graded root.
    profile : Path or None
        A configuration bundle file, or a built-in profile name.
        `None` grades under the built-in `numeric-core` scope.
    tier : ExecutionTier or None
        Overrides the bundle's execution tier without changing its requirements.
    hash_source : bool
        Hash local source bytes before and after analysis,
        and withhold the report if they changed.
    fail_on : str
        Comma-separated eligibility statuses that fail the audit.

    Raises
    ------
    typer.Exit
        Code 2 when `fail_on` names an unknown status or `pass`, or no status at all,
        when the report records operational errors,
        or when grading raised a `KalanosError` — `path` does not exist,
        is a remote dataset over a limit, held nothing to report on at all,
        two adapters tied on the same file,
        or a mapping override was malformed or matched nothing —
        or when writing `report` failed: an unsupported suffix, a missing directory,
        an unwritable path.
        The reason goes to stderr and nothing is written to stdout.
        Code 1 when the gate in `fail_on` trips.
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
    use_color = (
        color
        if color is not None
        else not os.environ.get("NO_COLOR") and sys.stdout.isatty()
    )

    # Step 1: grade. The reason is printed, not logged: it explains a non-zero exit,
    # and must reach the user even at a verbosity that silences ERROR records.
    try:
        # Pairs, not a dict: two `--map` flags for one feature must reach the
        # resolver so a contradiction is refused instead of last-one-wins.
        mapping = [parse_map_argument(text) for text in map_ or []]
        show_progress = (
            sys.stderr.isatty()
            and not as_json
            and logging.getLogger(__name__).getEffectiveLevel() > logging.INFO
        )
        progress = (
            Console(stderr=True).status(f"Grading {path}", spinner="line")
            if show_progress
            else nullcontext()
        )
        # Entering the status replaces sys.stderr with a proxy that prints log
        # records above the spinner, so the handler is pointed at it afterwards.
        with progress, logging_to(sys.stderr):
            result = api.grade(
                path,
                limits=limits,
                mapping=mapping,
                mapping_file=map_file,
                sidecar=not no_sidecar,
                bundle=profile,
                tier=tier,
                vision_samples=vision_samples,
                full_frame_scan=True if full_frame_scan else None,
                hash_source=hash_source,
            )
    except KalanosError as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    # Step 2: write the report file, if asked, before anything reaches stdout,
    # so a failed write leaves stdout empty rather than a half card.
    destination: Path | None = None
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
        width = shutil.get_terminal_size().columns if sys.stdout.isatty() else None
        print(
            render_terminal(
                result,
                color=use_color,
                width=width,
                report_path=str(destination) if destination is not None else None,
            )
        )

    # Step 4: the decision gate, from the one place decisions live. An
    # incomplete audit fails the default gate; operational errors already
    # left with exit 2 above, so this never masks one.
    if result.operational_errors:
        print(
            "kalanos: analysis encountered operational errors; "
            "inspect the saved report",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)
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
    """Summarise a saved report, current or legacy, without changing it.

    Parameters
    ----------
    report : Path
        The report JSON, of the current schema or a legacy one.
    episode : str or None
        Show only this episode's findings, diagnostics and reasons,
        with each reason in full.
        `None` shows every episode with a count of its reasons.

    Raises
    ------
    typer.Exit
        Code 2 when the file is missing or not a report,
        or `episode` names no episode in it.
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
        if episode is not None:
            item = next(
                (e for e in loaded.data.get("episodes", []) if e.get("id") == episode),
                None,
            )
            if item is None:
                print(f"kalanos: no episode {episode!r}", file=sys.stderr)
                raise typer.Exit(code=2)
            console.print(json.dumps(item), markup=False)
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
    selected = (
        next((e for e in loaded.episodes if e.id == episode), None) if episode else None
    )
    if episode and selected is None:
        print(f"kalanos: no episode {episode!r}", file=sys.stderr)
        raise typer.Exit(code=2)
    for line in coverage_lines(selected.coverage if selected else loaded.coverage):
        console.print(line, markup=False)
    if loaded.diagnostics:
        from kalanos.analysis.diagnostics.runner import diagnostic_lines

        for line in diagnostic_lines(loaded.diagnostics):
            console.print(line, markup=False)
        for diagnostic in loaded.diagnostics.results:
            if episode is None or diagnostic.episode_id == episode:
                console.print(diagnostic.model_dump_json(indent=2), markup=False)
    for finding in loaded.findings:
        if episode is None or finding.episode_id == episode:
            console.print(
                json.dumps(
                    {
                        "id": finding.id,
                        "episode": finding.episode_id,
                        "metric": finding.metric_id,
                        "source_path": finding.source_path,
                        "source_field": finding.source_field,
                        "channel": finding.channel,
                        "consequence": finding.consequence.value,
                        "support": finding.support.model_dump(mode="json"),
                        "calibration": finding.calibration,
                    }
                ),
                markup=False,
            )
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


@app.command(help="Publish a saved report to your private space on hub.kalanos.ai.")
def publish(
    report: Annotated[
        Path,
        typer.Argument(help="A report JSON written by `kalanos grade --report`."),
    ],
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="KALANOS_API_KEY",
            show_envvar=True,
            help="Your hub API key. Prefer the KALANOS_API_KEY environment variable.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            help=(
                "ORG/NAME to publish a local dataset under. "
                "Hosted datasets use their own name."
            ),
        ),
    ] = None,
) -> None:
    """Send a saved report, unchanged, to the hub account `api_key` belongs to.

    Any report `grade` wrote can be sent, whatever its exit code was.

    Exit codes
    ----------
    0
        The hub accepted the report.
    2
        Anything was refused or failed: a missing or malformed key,
        an unreadable report, a local dataset without `name`,
        or a hub that refused the report or could not be reached.
        Nothing is published, and the reason goes to stderr.

    Parameters
    ----------
    report : Path
        The report JSON, read relative to the working directory.
    api_key : str or None
        A `klns_` key from the hub's settings page.
        Only its first 16 characters ever appear in a message.
    name : str or None
        `ORG/NAME` for a report of a local dataset, which has no name of its own.

    Raises
    ------
    typer.Exit
        Code 2 on any `PublishError`.
    """

    try:
        hub = check_hub_url(get_settings().hub_url)
        key = check_key(api_key, hub)
        payload = load_report(report)
        name = check_name(payload, name)
        published = publish_report(report=payload, api_key=key, name=name, hub_url=hub)
    except PublishError as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc

    print(f"Submitted {published.report_id}. It is private to your account.")
    print(f"Check its status at {published.dashboard}")
    if published.page:
        print(f"Report: {published.page}")


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
    profile: Annotated[
        Path | None,
        typer.Option("--profile", help="Configuration bundle, as for grade."),
    ] = None,
    map_: Annotated[
        list[str] | None,
        typer.Option("--map", help="FEATURE=TYPE; repeatable, as for grade."),
    ] = None,
    map_file: Annotated[
        Path | None, typer.Option("--map-file", help="Explicit mapping file.")
    ] = None,
    no_sidecar: Annotated[
        bool, typer.Option("--no-sidecar", help="Ignore automatic mapping sidecars.")
    ] = False,
    tier: Annotated[
        ExecutionTier | None,
        typer.Option("--tier", help="Execution tier; requirements remain unchanged."),
    ] = None,
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
    profile : Path or None
        A configuration bundle file, or a built-in profile name, as for `grade`.
    map_ : list[str] or None
        `FEATURE=TYPE` overrides, as for `grade`.
    map_file : Path or None
        A YAML mapping file of overrides, as for `grade`.
    no_sidecar : bool
        Ignore a `kalanos-map.yaml` in or beside each dataset.
    tier : ExecutionTier or None
        Overrides the bundle's execution tier without changing its requirements.
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
        result = run_benchmark(
            paths or REFERENCE_DATASETS,
            sample=sample,
            bundle=profile,
            mapping=[parse_map_argument(text) for text in map_ or []],
            mapping_file=UPath(map_file) if map_file is not None else None,
            sidecar=not no_sidecar,
            tier=tier,
        )
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


@app.command(
    "compare", help="Compare saved reports without inventing missing identities."
)
def compare_command(
    old: Annotated[
        Path, typer.Argument(help="The earlier report JSON, of any schema.")
    ],
    new: Annotated[Path, typer.Argument(help="The later report JSON, of any schema.")],
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Also write the comparison JSON here. It must not be either input.",
        ),
    ] = None,
) -> None:
    """Compare two saved reports and print the comparison as JSON.

    Parameters
    ----------
    old : Path
        The earlier report.
    new : Path
        The later report.
    report : Path or None
        Where to also write the comparison, relative to the working directory.

    Raises
    ------
    typer.Exit
        Code 2 when a report cannot be read or loaded,
        when `report` is one of the inputs, or when writing it fails.
        Code 1 when the reports are not comparable.
    """

    try:
        result = api.compare(old, new)
        text = result.model_dump_json(indent=2)
        if report:
            if report.resolve() in (old.resolve(), new.resolve()):
                raise ValueError("comparison output must not overwrite an input report")
            report.write_text(text, encoding="utf-8")
        print(text)
    except (OSError, ValueError) as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc
    if not result.comparable:
        raise typer.Exit(code=1)


@profiles_app.command("list", help="List the built-in profile names.")
def profiles_list() -> None:
    """Print each built-in profile name on its own line."""

    print("\n".join(BUILT_IN_PROFILES))


@profiles_app.command("show", help="Print the bundle a built-in profile stands for.")
def profiles_show(
    name: Annotated[
        str,
        typer.Argument(help="A built-in profile name, from `kalanos profiles list`."),
    ],
) -> None:
    """Print the bundle a built-in profile stands for, as JSON.

    Parameters
    ----------
    name : str
        A built-in profile name.

    Raises
    ------
    typer.Exit
        Code 2 when `name` is not a built-in profile.
    """

    if name not in BUILT_IN_PROFILES:
        print(f"kalanos: unknown profile {name!r}", file=sys.stderr)
        raise typer.Exit(code=2)
    print(Bundle(requirements=RequirementsSection(id=name)).model_dump_json(indent=2))


@profiles_app.command(
    "validate", help="Check a configuration bundle file and the policy it names."
)
def profiles_validate(
    path: Annotated[Path, typer.Argument(help="A configuration bundle YAML file.")],
) -> None:
    """Load a configuration bundle and its policy, and report whether both are valid.

    Dataset bindings and coverage are left to a grade run,
    which has a dataset to check them against.

    Parameters
    ----------
    path : Path
        The bundle file.

    Raises
    ------
    typer.Exit
        Code 2 when the bundle or its policy cannot be read or does not validate.
    """

    try:
        bundle = load_bundle(UPath(path))
        load_bundle_policy(bundle, UPath(path))
        print(
            "Configuration structure is valid; "
            "dataset bindings and coverage require a grade run."
        )
    except (OSError, ValueError, KalanosError) as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc


@diagnostics_app.command(
    "summarize-study",
    help="Summarize labelled JSON observations as an unaccepted validation draft.",
)
def summarize_study_command(
    path: Annotated[
        Path,
        typer.Argument(help="A JSON list of labelled episode observations."),
    ],
) -> None:
    """Summarize labelled JSON observations as an unaccepted validation draft.

    Parameters
    ----------
    path : Path
        A JSON list of observations, one per episode,
        each in the tuning or the validation split.

    Raises
    ------
    typer.Exit
        Code 2 when the file cannot be read or parsed,
        an observation does not validate,
        episode ids repeat,
        or a session appears in both splits.
    """

    from kalanos.analysis.diagnostics.validation import summarize_study

    try:
        print(json.dumps(summarize_study(json.loads(path.read_text())), indent=2))
    except (OSError, ValueError, TypeError) as exc:
        print(f"kalanos: {exc}", file=sys.stderr)
        raise typer.Exit(code=2) from exc


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀

if __name__ == "__main__":
    app()
