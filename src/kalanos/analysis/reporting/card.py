"""Render a Report as a short, glanceable terminal summary.

The card only ever reads fields the model already carries —
one row per episode, using the score `scoring` already rolled up for it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import io
from collections import Counter

# External
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from upath import UPath

# Internal
from kalanos.analysis.models.metrics import MetricStatus
from kalanos.analysis.models.report import GradedEpisode, Report
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult, Severity


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# fmt: off
_MAX_EPISODE_ROWS = 8
_MAX_FINDINGS     = 3
_MAX_SOURCE_ROWS  = 5
_DEFAULT_WIDTH    = 120
_RAIL_CELLS       = 12
_RAIL_MIN_WIDTH   = 90
_SCORE_CELLS      = 5
# fmt: on

_CHILD_INDENT = "  "

_BUCKET_LABELS = ("PASS", "WARN", "FAIL", "SKIP")

# The minimum width of one bucket's number column: every label is four characters,
# so a wider count is what widens the field, never the label.
_BUCKET_CELLS = 4

# Multi-column can break if we accept a lower width.
_MIN_WIDTH = 40

# Similar colors from the HTML report
# fmt: off
_PLATE   = "#0a2540"
_PAPER   = "#f4f1ea"
_GOOD    = "#1c8c91"
_WARNING = "#b07107"
_ACCENT  = "#e63f32"
_RULE    = "#807c74"
# fmt: on

# Only consulted when color is on — a plain card carries no style at all.
_GRADE_STYLE = {
    Grade.A: _GOOD,
    Grade.B: _GOOD,
    Grade.C: _WARNING,
    Grade.D: _WARNING,
    Grade.F: _ACCENT,
}

# Every MetricStatus lands in exactly one bucket.
# SKIP holds the two that never reached the score:
# - not_applicable never ran
# - report_only ran but nothing graded it
_METRIC_BUCKETS: dict[MetricStatus, str] = {
    MetricStatus.GOOD: "PASS",
    MetricStatus.WARNING: "WARN",
    MetricStatus.CRITICAL: "FAIL",
    MetricStatus.REPORT_ONLY: "SKIP",
    MetricStatus.NOT_APPLICABLE: "SKIP",
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _grade_cell(score: ScoreResult) -> Text:
    """Render one ScoreResult's grade letter, styled by how good it is.

    Parameters
    ----------
    score : ScoreResult
        The score to render.

    Returns
    -------
    Text
        The grade letter, styled by severity when the `Console` has color enabled.
        `"–"` when `score.score` is `None`, so an ungraded row never reads as an "F".
    """

    if score.score is None:
        return Text(text="–")
    if score.grade is None:
        return Text(text="?")
    return Text(text=score.grade.value, style=_GRADE_STYLE.get(score.grade, ""))


def _score_cell(score: ScoreResult) -> str:
    """Render one ScoreResult's number, one decimal place.

    Parameters
    ----------
    score : ScoreResult
        The score to render.

    Returns
    -------
    str
        `"n/a"` when nothing rolled up to this level — never `"0.0"`,
        which would misread as a real, low score.
    """

    return "n/a" if score.score is None else f"{score.score:.1f}"


def _rail(score: ScoreResult, cells: int = _RAIL_CELLS) -> Text:
    """Render one ScoreResult as a filled/empty block rail.

    Parameters
    ----------
    score : ScoreResult
        The score to render.
    cells : int
        How many character cells the rail spans.

    Returns
    -------
    Text
        A bar of `"█"` (styled by grade) then `"░"` for the remainder.
        All `"░"` when `score.score is None`.
    """

    if score.score is None:
        return Text("░" * cells, style=_RULE)

    pct = max(0.0, min(100.0, score.score))
    filled = round(pct / 100 * cells)
    grade = score.grade
    fill_style = _GRADE_STYLE.get(grade, _RULE) if grade is not None else _RULE
    rail = Text("█" * filled, style=fill_style)
    rail.append("░" * (cells - filled), style=_RULE)
    return rail


def _score_rail_cell(score: ScoreResult, *, rail: bool) -> Text:
    """Render one ScoreResult for the recordings table: its bar, then its figure.

    Parameters
    ----------
    score : ScoreResult
        The score to render.
    rail : bool
        Whether the bar is drawn. False renders the figure alone,
        for a terminal too narrow to carry the bar as well.

    Returns
    -------
    Text
        The figure padded to `_SCORE_CELLS`,
        preceded by a space and the bar when `rail` is set.
        The bar is what gives the figure its 0-100 range, so the two never separate.
    """

    figure = Text(text=_score_cell(score).rjust(_SCORE_CELLS))
    if not rail:
        return figure
    return Text.assemble(_rail(score), " ", figure)


def _verdict_cell(score: ScoreResult) -> Text:
    """Render one ScoreResult's train-readiness verdict.

    Parameters
    ----------
    score : ScoreResult
        The score to read `train_ready` from.

    Returns
    -------
    Text
        A reverse-video pill reading `TRAIN READY` or `NOT TRAIN READY` for
        `True`/`False`, or plain italic `train-readiness unknown` for `None`.
    """

    if score.train_ready is None:
        return Text(text="train-readiness unknown", style=f"italic {_RULE}")
    if score.train_ready:
        return Text(text="▌ TRAIN READY ▐", style=f"bold {_PLATE} on {_GOOD}")
    return Text(text="▌ NOT TRAIN READY ▐", style=f"bold {_PAPER} on {_ACCENT}")


def _counts_line(report: Report) -> str:
    """Describe a Report's dataset-wide counts for the plate's third line.

    Parameters
    ----------
    report : Report
        The report to count over.

    Returns
    -------
    str
        `"{n} recordings · {n} findings · {n} not analysed · {n} no schema"`,
        singularising `recording`/`finding` at 1. All four segments always
        appear, including zeros.
    """

    n_episodes = len(report.episodes)
    n_findings = len(report.findings)
    return (
        f"{n_episodes} recording{'' if n_episodes == 1 else 's'} · "
        f"{n_findings} finding{'' if n_findings == 1 else 's'} · "
        f"{len(report.skipped)} not analysed · "
        f"{len(report.unresolved)} no schema"
    )


def _under_root(path: UPath, root: UPath) -> str:
    """Render one path as the reader sees it: relative to the root on the plate.

    Parameters
    ----------
    path : UPath
        The path to render.
    root : UPath
        The dataset root the plate already prints in full.

    Returns
    -------
    str
        `path` relative to `root`. Its bare name when the two are the same path,
        so a run pointed at one file still names it. `path` in full when it sits
        outside `root` or on another protocol — `..` segments back out of a root
        would read worse than the absolute location.
    """

    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return path.name if str(relative) == "." else str(relative)


def _episode_label(episode: GradedEpisode, root: UPath) -> str:
    """Name an episode that is its own container, by the files it was read from.

    An episode's id is a recording name rather than a path, and two recordings
    under different folders can share one, so the files are what a reader can act on.
    A grouped episode is named by `_child_label` instead, under a header that
    already carries the container's path.

    Parameters
    ----------
    episode : GradedEpisode
        The episode to name.
    root : UPath
        The dataset root to render each source file relative to.

    Returns
    -------
    str
        The episode's source files, comma-separated and relative to `root`,
        or its id when it has none.
    """

    if not episode.source_paths:
        return episode.id
    return ", ".join(_under_root(path, root) for path in episode.source_paths)


def _container_of(episode: GradedEpisode) -> str | None:
    """Name the container an episode was read from, or None when it is its own.

    `pipeline` qualifies an id with its container only when one candidate returned
    more than one episode, so the separator is what distinguishes the two cases.
    """

    container, separator, _ = episode.id.partition("::")
    return container if separator else None


def _child_label(episode: GradedEpisode) -> str:
    """Name one grouped episode by the part of its id its container does not carry.

    Returns
    -------
    str
        The bare episode id, with its file count in parentheses when it drew on
        more than one file — the header absorbs the episode's source paths,
        so that count would otherwise vanish from the row entirely.
    """

    label = episode.id.split("::", 1)[1]
    n_files = len(episode.source_paths)
    return f"{label} ({n_files} files)" if n_files > 1 else label


def _group_episodes(
    episodes: list[GradedEpisode],
) -> list[tuple[str | None, list[GradedEpisode]]]:
    """Gather episodes into container blocks, keeping the order they arrive in.

    Parameters
    ----------
    episodes : list[GradedEpisode]
        The episodes to render, already ordered worst-first.

    Returns
    -------
    list[tuple[str or None, list[GradedEpisode]]]
        One entry per block: the container to head it with,
        or `None` for an episode that is its own container and renders as a bare row.
        A container takes the position of the first episode that named it,
        which in a worst-first list is its worst one.
    """

    blocks: list[tuple[str | None, list[GradedEpisode]]] = []
    # seen holds the same list objects the blocks carry, so appending a member
    # fills the block already in position rather than a copy of it.
    seen: dict[str, list[GradedEpisode]] = {}
    for episode in episodes:
        container = _container_of(episode)
        if container is None:
            blocks.append((None, [episode]))
            continue
        members = seen.get(container)
        if members is None:
            members = []
            seen[container] = members
            blocks.append((container, members))
        members.append(episode)
    return blocks


def _metric_counts(episode: GradedEpisode) -> tuple[int, ...]:
    """Count an episode's metric results into one number per bucket.

    Parameters
    ----------
    episode : GradedEpisode
        The graded episode to count over.

    Returns
    -------
    tuple[int, ...]
        One count per `_BUCKET_LABELS` entry, in that order, across the
        episode's own metrics and every stream and channel within it.
    """

    results = list(episode.metrics.values())
    for stream in episode.streams:
        results.extend(stream.metrics.values())
        for channel in stream.channels:
            results.extend(channel.metrics.values())

    counted = Counter(_METRIC_BUCKETS[metric.status] for metric in results)
    return tuple(counted[label] for label in _BUCKET_LABELS)


def _bucket_cell(counts: tuple[int, ...], field: int) -> str:
    """Render one row's bucket counts as bare numbers aligned under their labels."""

    if not sum(counts):
        return "no metrics ran"
    return " ".join(f"{count:>{field}}" for count in counts)


def _bucket_header(field: int) -> Group:
    """Build the two-row METRICS header: the group name over its four bucket labels."""

    labels = " ".join(f"{label:>{field}}" for label in _BUCKET_LABELS)
    return Group(Text(text="METRICS", justify="center"), Text(text=labels))


def _severity_cell(finding: Finding) -> Text:
    """Render one Finding's severity label, styled to match the HTML's verdict colours.

    Parameters
    ----------
    finding : Finding
        The finding to label.

    Returns
    -------
    Text
        `"FAIL"` in accent for a critical finding, `"WARN"` in warning otherwise.
    """

    if finding.severity == Severity.CRITICAL:
        return Text(text="FAIL", style=f"bold {_ACCENT}")
    return Text(text="WARN", style=f"bold {_WARNING}")


def _finding_location(finding: Finding) -> str:
    """Address one finding down to its instance and channel, for its own line.

    Parameters
    ----------
    finding : Finding
        The finding to address.

    Returns
    -------
    str
        `"{episode}/{instance}:{stream}/{channel}.{metric}"`, with the `instance:`
        prefix omitted when `finding.instance is None`, and the `/{stream}` and
        `/{channel}` segments each omitted when `finding.stream`/`finding.channel`
        is `None`.
    """

    where = finding.episode_id
    if finding.stream is not None:
        stream_label = (
            finding.stream
            if finding.instance is None
            else f"{finding.instance}:{finding.stream}"
        )
        where = f"{where}/{stream_label}"
    if finding.channel is not None:
        where = f"{where}/{finding.channel}"
    metric_name = finding.metric_id.removeprefix(f"{finding.family}.")
    return f"{where}.{metric_name}"


def _finding_detail(finding: Finding) -> str:
    """Describe one finding's value and evidence, for the line under its address.

    Parameters
    ----------
    finding : Finding
        The finding to describe.

    Returns
    -------
    str
        `"{value}{unit}"`, followed by `"; {evidence}"` when `finding.evidence`
        is non-empty.
    """

    unit = f" {finding.unit}" if finding.unit else ""
    value = f"{finding.value:.4g}{unit}"
    evidence = ", ".join(f"{key}={val}" for key, val in finding.evidence.items())
    return f"{value}; {evidence}" if evidence else value


def _eyebrow(title: str) -> Rule:
    """Build a section rule labelled with an uppercase title.

    The card's only heading style.

    Parameters
    ----------
    title : str
        The section's name, already uppercase.

    Returns
    -------
    Rule
        A hairline rule with `title` set into its left edge, styled `_RULE`
        throughout so the heading recedes rather than competing with the data.
    """

    return Rule(Text(title, style=_RULE), align="left", characters="─", style=_RULE)


def _source_table(rows: list[tuple[str, str]]) -> Table:
    """Build a two-column table of source files and why each didn't reach grading.

    Parameters
    ----------
    rows : list[tuple[str, str]]
        `(path, reason)` pairs, in the order they should render.

    Returns
    -------
    Table
        A borderless `FILE`/`REASON` table shaped like the recordings table,
        holding at most `_MAX_SOURCE_ROWS` rows plus a trailing
        `"+N more"` row when `rows` held more than that.
    """

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, expand=True, header_style=_RULE)
    table.add_column(header="FILE", no_wrap=True, overflow="ellipsis", ratio=3)
    table.add_column(header="REASON", overflow="fold", ratio=2)

    for path, reason in rows[:_MAX_SOURCE_ROWS]:
        table.add_row(path, reason)

    remaining = len(rows) - _MAX_SOURCE_ROWS
    if remaining > 0:
        table.add_row(Text(text=f"+{remaining} more", style=_RULE), "")

    return table


def render_terminal(
    report: Report, *, color: bool = False, width: int | None = None
) -> str:
    """Render a Report as a report card for the terminal.

    Parameters
    ----------
    report : Report
        The report to render.
    color : bool
        Whether to style grades, rails and finding labels. Off by default,
        so the string stays plain wherever it is captured — a caller talking
        to a real terminal opts in explicitly.
    width : int or None
        The column count to lay the card out against, floored at `_MIN_WIDTH`.
        `None` falls back to a fixed default.

    Returns
    -------
    str
        The rendered card, without a trailing newline.
    """

    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=color,
        no_color=not color,
        width=max(width or _DEFAULT_WIDTH, _MIN_WIDTH),
        highlight=False,
        # Channel/instance names are arbitrary file data, not markup — a name like
        # `acc[x,y,z]` (docs/ARCHITECTURE.md) would otherwise trip rich's parser.
        markup=False,
    )

    # Step 1: the plate, the one painted element on the card — a full-bleed
    # navy panel for the one thing a person runs `kalanos grade` to find out.
    plate = Table.grid(padding=(0, 1), expand=True)
    plate.add_column()
    plate.add_column(justify="right")
    plate.add_row(Text("KALANOS · DATASET REPORT", style=f"{_PAPER} dim"), "")
    plate.add_row(
        Text(str(report.root), style="bold"),
        Text.assemble(
            _grade_cell(report.score),
            "  ",
            _score_cell(report.score),
            "  ",
            _verdict_cell(report.score),
        ),
    )
    plate.add_row(Text(_counts_line(report), style=f"{_PAPER} dim"), "")
    console.print(
        Panel(
            renderable=plate,
            box=box.ROUNDED,
            padding=(0, 1),
            style=f"{_PAPER} on {_PLATE}",
        )
    )

    # Step 2: the dataset-level row and every episode row share one table,
    # so rich measures every cell's visible width together and pads them
    # to the same columns, whether or not a given cell carries a style.
    console.print(_eyebrow("RECORDINGS"))
    show_rail = console.width >= _RAIL_MIN_WIDTH

    # worst score first,
    # score=None sinks to the bottom rather than sorting as if it were zero.
    ordered = sorted(
        report.episodes,
        key=lambda episode: (episode.score.score is None, episode.score.score or 0.0),
    )
    shown = ordered[:_MAX_EPISODE_ROWS]
    overall_counts = tuple(
        sum(counts)
        for counts in zip(
            *(_metric_counts(episode) for episode in report.episodes), strict=True
        )
    ) or (0,) * len(_BUCKET_LABELS)
    # One field width shared by every row. overall_counts sums every episode
    # in the report, so it is always >= any single shown episode's counts —
    # sizing against it alone still covers a dataset-wide total in five digits.
    field = max(_BUCKET_CELLS, *(len(str(count)) for count in overall_counts))

    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        expand=True,
        padding=(0, 1),
        header_style=_RULE,
    )
    # rich puts a one-line header on the lower row of a multi-line header block;
    # the trailing newline lifts these three to the upper row, beside METRICS.
    table.add_column(header="GRADE\n", justify="center", no_wrap=True, width=5)
    # The one column that grows: RECORDING is the only cell whose content is
    # dataset-dependent, so it is the only place surplus width can usefully go.
    # A ratio column is also the only way expand=True stays safe — Rich sizes
    # it to exactly fill what GRADE/SCORE/METRICS leave over, so the table
    # width never overflows and never forces Rich's last-resort column
    # collapse, which does not respect any column's fixed width or no_wrap.
    table.add_column(header="RECORDING\n", no_wrap=True, overflow="ellipsis", ratio=3)
    table.add_column(
        header="SCORE\n",
        justify="right",
        no_wrap=True,
        width=_RAIL_CELLS + 1 + _SCORE_CELLS if show_rail else _SCORE_CELLS,
    )
    table.add_column(
        header=_bucket_header(field),
        justify="right",
        no_wrap=True,
        overflow="ellipsis",
        width=len(_BUCKET_LABELS) * field + len(_BUCKET_LABELS) - 1,
    )

    def _score_row(
        label: RenderableType, score: ScoreResult, detail: str
    ) -> list[RenderableType]:
        return [
            _grade_cell(score),
            label,
            _score_rail_cell(score, rail=show_rail),
            detail,
        ]

    table.add_row(
        *_score_row("OVERALL", report.score, _bucket_cell(overall_counts, field)),
        style="bold",
    )
    table.add_section()

    for container, members in _group_episodes(shown):
        if container is not None:
            header_row: list[RenderableType] = [""] * len(table.columns)
            header_row[1] = Text(text=container)
            table.add_row(*header_row)
        for episode in members:
            label = (
                f"{_CHILD_INDENT}{_child_label(episode)}"
                if container is not None
                else _episode_label(episode, report.root)
            )
            table.add_row(
                *_score_row(
                    label, episode.score, _bucket_cell(_metric_counts(episode), field)
                )
            )

    remaining = len(ordered) - _MAX_EPISODE_ROWS
    if remaining > 0:
        more_row: list[RenderableType] = [""] * len(table.columns)
        more_row[1] = Text(text=f"+{remaining} more", style=_RULE)
        table.add_row(*more_row)
    console.print(table)

    # Step 3: report.findings arrives already sorted worst-first via sort_findings,
    # so the card only slices it and never re-sorts.
    console.print(_eyebrow("FINDINGS"))
    findings = report.findings[:_MAX_FINDINGS]
    if findings:
        # Severity and address are one assembled Text, not separate columns,
        # so a narrow ellipsis only ever eats the tail — never the "FAIL"/
        # "WARN" label at the start.
        for finding in findings:
            address = Text.assemble(
                _severity_cell(finding), " ", _finding_location(finding)
            )
            console.print(address, no_wrap=True, overflow="ellipsis")
            detail = Text(text=f"    {_finding_detail(finding)}", style=_RULE)
            console.print(detail, no_wrap=True, overflow="ellipsis")
    else:
        console.print(Text(text="  no findings", style=_GOOD))

    # Step 4: a file Kalanos declined to analyse is reported here too,
    # never dropped just because the card is a summary.
    if report.skipped:
        console.print(_eyebrow("NOT ANALYSED"))
        console.print(
            _source_table(
                [
                    (_under_root(item.path, report.root), item.reason.value)
                    for item in report.skipped
                ]
            )
        )
    if report.unresolved:
        console.print(_eyebrow("NO SCHEMA"))
        console.print(
            _source_table(
                [
                    (_under_root(item.path, report.root), item.reason)
                    for item in report.unresolved
                ]
            )
        )

    # Step 5: the footer, matching the plate meta line in the HTML report.
    footer_parts = [f"schema {report.schema_version}"]
    if report.policy_version is not None:
        footer_parts.append(f"policy v{report.policy_version}")
    if report.duration_s is not None:
        footer_parts.append(f"{report.duration_s:.2f}s")
    console.print(Text(text=" · ".join(footer_parts), style=_RULE))

    return buffer.getvalue().rstrip("\n")
