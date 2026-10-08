"""The report card's sections, each a function of the report and the layout.

A section returns its lines already fitted to the width,
or no lines at all when it has nothing to say about this report.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections import Counter
from dataclasses import dataclass

# External
from rich.cells import cell_len
from rich.text import Text
from upath import UPath

# Internal
from kalanos.analysis.metrics.registry import metric_label
from kalanos.analysis.models.domain import UNMAPPED_TAXONOMY_PREFIX
from kalanos.analysis.models.eligibility import (
    _PRECEDENCE,
    Consequence,
    EligibilityReason,
    EligibilityStatus,
    ReasonKind,
    SufficiencyStatus,
)
from kalanos.analysis.models.report import GradedEpisode, Report
from kalanos.analysis.models.scoring import Finding
from kalanos.analysis.reporting.card.theme import Theme


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# fmt: off
_MAX_ROWS   = 3
_INDENT     = 2
_GUTTER     = 2
_LEFT_LIMIT = 40
# fmt: on

_CONSEQUENCE_ORDER = (Consequence.BLOCK, Consequence.REVIEW, Consequence.REPORT_ONLY)

_CONSEQUENCE_ROLE = {
    Consequence.BLOCK: "bad",
    Consequence.REVIEW: "warn",
    Consequence.REPORT_ONLY: "muted",
}

Cell = Text | str


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class CardContext:
    """What every section needs besides the report.

    Attributes
    ----------
    theme : Theme
        The glyphs and styles to render with.
    width : int
        The column count no line may exceed.
    report_path : str or None
        Where `--report` wrote the full report, when it did.
    """

    theme: Theme
    width: int
    report_path: str | None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


# LAYOUT


def _rows(pairs: list[tuple[Cell, Cell]], ctx: CardContext) -> list[Text]:
    """Lay out two-column rows, the left column as wide as its widest cell.

    Parameters
    ----------
    pairs : list[tuple[Text or str, Text or str]]
        One `(left, right)` per row.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        One indented line per pair, never wider than `ctx.width`.
    """

    if not pairs:
        return []
    # Half the line at most, so a narrow terminal still shows the right column.
    left_width = min(
        max(cell_len(str(left)) for left, _ in pairs),
        _LEFT_LIMIT,
        max((ctx.width - _INDENT - _GUTTER) // 2, 0),
    )
    right_width = ctx.width - _INDENT - left_width - _GUTTER
    lines = []
    for left, right in pairs:
        line = Text(" " * _INDENT)
        line.append_text(ctx.theme.fit(left, left_width))
        line.pad_right(_INDENT + left_width + _GUTTER - line.cell_len)
        line.append_text(ctx.theme.fit(right, right_width))
        line.rstrip()
        lines.append(line)
    return lines


def _note(text: str, role: str, ctx: CardContext) -> Text:
    """Lay out one indented line that belongs to no column, such as a `+ N more`.

    Parameters
    ----------
    text : str
        The line's content.
    role : str
        The theme role to style the whole line with.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    Text
        `text` indented like a row and fitted to `ctx.width`.
    """

    styled = Text(text, style=ctx.theme.style(role))
    return Text(" " * _INDENT) + ctx.theme.fit(styled, ctx.width - _INDENT)


def _parts(*parts: Cell, sep: str) -> Text:
    """Join cell parts with the theme's separator, each keeping its own style.

    Parameters
    ----------
    *parts : Text or str
        The parts to join, in order.
    sep : str
        The separator glyph, padded with a space on each side.

    Returns
    -------
    Text
        One cell, unfitted, for `_rows` to fit to its column.
    """

    return Text(f" {sep} ").join(Text(p) if isinstance(p, str) else p for p in parts)


# NAMES


def _under_root(path: UPath, root: UPath) -> str:
    """Render one path as the reader sees it: relative to the root in the title.

    Parameters
    ----------
    path : UPath
        The path to render.
    root : UPath
        The dataset root the title already prints in full.

    Returns
    -------
    str
        `path` relative to `root`.
        Its bare name when the two are the same path, so a run pointed at one file still
        names it. `path` in full when it sits outside `root` or on another protocol —
        `..` segments back out of a root would read worse than the absolute location.
    """

    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return path.name if str(relative) == "." else str(relative)


def _episode_label(episode: GradedEpisode, root: UPath) -> str:
    """Name an episode by the files it was read from, or by its id.

    An episode read out of a container shares its files with its siblings,
    so its container-qualified id (`container::episode`) is what tells it apart.

    Parameters
    ----------
    episode : GradedEpisode
        The episode to name.
    root : UPath
        The dataset root its source paths are shown relative to.

    Returns
    -------
    str
        Its source paths under `root`, comma-separated,
        or its id with `::` shown as `/` when it has no paths of its own.
    """

    if not episode.source_paths or "::" in episode.id:
        return episode.id.replace("::", "/")
    return ", ".join(_under_root(path, root) for path in episode.source_paths)


def _reason_label(reason: EligibilityReason) -> str:
    """Name the condition behind one eligibility reason, in plain words.

    A finding reason's id ends in its metric's name; any other reason's id
    is a capability or binding key.

    Parameters
    ----------
    reason : EligibilityReason
        The reason to name.

    Returns
    -------
    str
        The metric's label for a finding reason,
        else the key with underscores shown as spaces.
    """

    if reason.kind is ReasonKind.FINDING:
        return metric_label(reason.id.rsplit(".", 1)[-1])
    return reason.id.replace("_", " ")


def _finding_source(finding: Finding) -> str:
    """Name the recorded source a finding sits on: instance, stream and channel.

    Parameters
    ----------
    finding : Finding
        The finding to locate.

    Returns
    -------
    str
        `instance:stream/channel`, leaving out whichever parts the finding lacks,
        or `episode` when it sits on no stream.
    """

    if finding.stream is None:
        source = "episode"
    elif finding.instance is not None:
        source = f"{finding.instance}:{finding.stream}"
    else:
        source = finding.stream
    if finding.channel is not None:
        source = f"{source}/{finding.channel}"
    return source


def _distinct(items: list[str]) -> list[str]:
    """Drop repeats, keeping first-seen order.

    Parameters
    ----------
    items : list[str]
        The items to deduplicate.

    Returns
    -------
    list[str]
        Each item once, where it first appeared.
    """

    return list(dict.fromkeys(items))


# SECTIONS


def title(report: Report, ctx: CardContext) -> list[Text]:
    """The card's first line: the product, then the root that was graded.

    Parameters
    ----------
    report : Report
        The report to render.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        One line.
    """

    line = Text()
    line.append("Kalanos", style=ctx.theme.style("heading"))
    line.append(f" {ctx.theme.sep} {report.root}")
    return [ctx.theme.fit(line, ctx.width)]


def _assessment(report: Report, ctx: CardContext) -> Text:
    """The verdict: the first of the run's conditions that holds, worst first.

    Parameters
    ----------
    report : Report
        The report to judge.
    ctx : CardContext
        The theme to style the verdict with.

    Returns
    -------
    Text
        The verdict, styled `good`, `warn` or `bad`.
    """

    counts, sep = report.eligibility_counts, ctx.theme.sep
    if report.operational_errors:
        verdict, role = f"Incomplete {sep} operational errors", "bad"
    elif counts is None:
        verdict, role = "Not graded", "bad"
    elif not counts.inventory_complete:
        verdict, role = f"Incomplete {sep} inventory not fully read", "warn"
    elif counts.blocked:
        verdict, role = "Blocked episodes found", "bad"
    elif counts.review:
        verdict, role = "Review required", "warn"
    elif counts.unknown:
        verdict, role = "Evidence missing", "warn"
    else:
        verdict, role = "Ready", "good"
    return Text(verdict, style=ctx.theme.style(role))


def _readiness(report: Report, ctx: CardContext) -> str:
    """The readiness index, or why it is undefined.

    Parameters
    ----------
    report : Report
        The report to read the index from.
    ctx : CardContext
        The theme whose separator joins the reasons.

    Returns
    -------
    str
        The index out of 100, falling back to the score when no index was recorded,
        `undefined` and its reasons, or `not graded`.
    """

    readiness = report.readiness
    if readiness is None:
        score = report.score.score
        return "not graded" if score is None else f"{score:.0f}/100"
    if readiness.score is None:
        return f"undefined {ctx.theme.sep} {'; '.join(readiness.reasons)}"
    return f"{readiness.score:.0f}/100"


def headline(report: Report, ctx: CardContext) -> list[Text]:
    """The verdict, readiness and episode counts, and what they are relative to.

    Parameters
    ----------
    report : Report
        The report to render.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        One row per label that applies to this report.
    """

    sep = ctx.theme.sep
    pairs: list[tuple[Cell, Cell]] = [
        ("Assessment", _assessment(report, ctx)),
        ("Readiness", _readiness(report, ctx)),
    ]
    counts = report.eligibility_counts
    if counts is not None:
        pairs.append(
            (
                "Episodes",
                f"{counts.total} {sep} eligible {counts.pass_count} {sep} "
                f"review {counts.review} {sep} blocked {counts.blocked} {sep} "
                f"unknown {counts.unknown}",
            )
        )
    if report.scope is not None:
        pairs.append(("Scope", report.scope.requirements_id))
    sufficiency = report.sufficiency
    if sufficiency is not None and sufficiency.status is not SufficiencyStatus.UNKNOWN:
        status = sufficiency.status.value
        detail = next(
            (
                check.detail
                for check in sufficiency.checks
                if check.status is not SufficiencyStatus.SUFFICIENT and check.detail
            ),
            None,
        )
        pairs.append(("Sufficiency", f"{status} {sep} {detail}" if detail else status))
    return _rows(pairs, ctx)


def needs_attention(report: Report, ctx: CardContext) -> list[Text]:
    """The episodes that did not pass, worst first, and why each did not.

    Parameters
    ----------
    report : Report
        The report to render.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        Up to `_MAX_ROWS` rows and a count of the rest,
        or nothing when every episode passed.
    """

    flagged = [
        (episode, episode.eligibility)
        for episode in report.episodes
        if episode.eligibility is not None
        and episode.eligibility.status is not EligibilityStatus.PASS
    ]
    if not flagged:
        return []
    # A score of None sorts after every real score instead of reading as zero.
    flagged.sort(
        key=lambda pair: (
            _PRECEDENCE.index(pair[1].status),
            pair[0].score.score is None,
            pair[0].score.score or 0.0,
        )
    )

    pairs: list[tuple[Cell, Cell]] = []
    for episode, eligibility in flagged[:_MAX_ROWS]:
        status = eligibility.status
        score = "n/a" if episode.score.score is None else f"{episode.score.score:.1f}"
        reasons = _distinct(
            [_reason_label(r) for r in eligibility.reasons if r.status is status]
        )
        role = "bad" if status is EligibilityStatus.BLOCKED else "warn"
        right = Text()
        right.append(f"{status.value:<7}", style=ctx.theme.style(role))
        right.append(f" {score:>5}  {', '.join(reasons)}")
        pairs.append((_episode_label(episode, report.root), right))

    lines = _rows(pairs, ctx)
    remaining = len(flagged) - _MAX_ROWS
    if remaining > 0:
        lines.append(_note(f"+ {remaining} more in report", "muted", ctx))
    return lines


def findings(report: Report, ctx: CardContext) -> list[Text]:
    """Findings grouped by the source they sit on and what they do to eligibility.

    Parameters
    ----------
    report : Report
        The report to render.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        Up to `_MAX_ROWS` rows, each counting distinct episodes,
        then how many sources are left out; `none` when nothing was found.
    """

    if not report.findings:
        return [_note("none", "good", ctx)]

    groups: dict[tuple[str, Consequence], list[Finding]] = {}
    for finding in report.findings:
        groups.setdefault((_finding_source(finding), finding.consequence), []).append(
            finding
        )
    rows = sorted(
        (
            (
                source,
                consequence,
                len({f.episode_id for f in members}),
                _distinct(
                    [
                        metric_label(f.metric_id.removeprefix(f"{f.family}."))
                        for f in members
                    ]
                ),
            )
            for (source, consequence), members in groups.items()
        ),
        key=lambda row: (_CONSEQUENCE_ORDER.index(row[1]), -row[2], row[0]),
    )

    sep, pairs = ctx.theme.sep, []
    for source, consequence, n_episodes, labels in rows[:_MAX_ROWS]:
        noun = "episode" if n_episodes == 1 else "episodes"
        right = Text(f"{n_episodes} {noun} {sep} {', '.join(labels)}  ")
        right.append(
            consequence.value.replace("_", "-"),
            style=ctx.theme.style(_CONSEQUENCE_ROLE[consequence]),
        )
        pairs.append((source, right))

    lines = _rows(pairs, ctx)
    hidden = len({row[0] for row in rows[_MAX_ROWS:]})
    if hidden:
        lines.append(_note(f"+ {hidden} more sources in report", "muted", ctx))
    return lines


def coverage(report: Report, ctx: CardContext) -> list[Text]:
    """How much of each applicable capability was computed, and the inventory.

    Parameters
    ----------
    report : Report
        The report to render.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        One row per capability with eligible subjects, in model order,
        a `mapping` row when any stream type went unmapped, then an `inventory` row.
    """

    theme, sep = ctx.theme, ctx.theme.sep
    recorded = report.coverage
    if recorded is None:
        return [_note("not recorded", "warn", ctx)]

    pairs: list[tuple[Cell, Cell]] = []
    for row in recorded.capabilities:
        if not row.eligible:
            continue
        parts: list[Cell] = [f"{row.computed}/{row.eligible} {row.unit}s"]
        for count, word, role in (
            (row.unavailable, "unavailable", "warn"),
            (row.skipped, "skipped", "warn"),
            (row.error, "errors", "bad"),
        ):
            if count:
                parts.append(Text(f"{count} {word}", style=theme.style(role)))
        if "video" in row.key and recorded.decoded_frames_examined:
            parts.append(f"{recorded.decoded_frames_examined} frames examined")
        # A gap carries its reason, so it never reads as an absence of problems.
        if row.unavailable and row.reasons:
            parts.append(Counter(row.reasons).most_common(1)[0][0])
        pairs.append((row.key.replace("_", " "), _parts(*parts, sep=sep)))

    # Counted by type, since `--map` types a source field once for every episode.
    unmapped = {
        stream.taxonomy_type
        for episode in report.episodes
        for stream in episode.streams
        if stream.taxonomy_type.startswith(f"{UNMAPPED_TAXONOMY_PREFIX}.")
    }
    if unmapped:
        noun = "stream type" if len(unmapped) == 1 else "stream types"
        pairs.append(
            (
                "mapping",
                _parts(
                    Text(f"{len(unmapped)} {noun} unmapped", style=theme.style("warn")),
                    "type them with --map",
                    sep=sep,
                ),
            )
        )

    if recorded.inventory_complete:
        inventory: Cell = "complete"
    else:
        inventory = Text(
            f"incomplete {sep} {recorded.unassessed_episodes} episodes unassessed "
            f"{sep} {recorded.refused_sources} sources refused",
            style=theme.style("warn"),
        )
    pairs.append(("inventory", inventory))
    return _rows(pairs, ctx)


def not_analysed(report: Report, ctx: CardContext) -> list[Text]:
    """The source files that never reached grading, and why.

    Parameters
    ----------
    report : Report
        The report to render.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        Up to `_MAX_ROWS` rows and a count of the rest,
        or nothing when every file was graded.
    """

    sources: list[tuple[Cell, Cell]] = [
        (_under_root(item.path, report.root), item.reason.value.replace("_", " "))
        for item in report.skipped
    ]
    sources += [
        (_under_root(item.path, report.root), f"no schema: {item.reason}")
        for item in report.unresolved
    ]
    if not sources:
        return []
    lines = _rows(sources[:_MAX_ROWS], ctx)
    remaining = len(sources) - _MAX_ROWS
    if remaining > 0:
        lines.append(_note(f"+ {remaining} more in report", "muted", ctx))
    return lines


def footer(report: Report, ctx: CardContext) -> list[Text]:
    """Where the full report went, then the versions and duration of the run.

    Parameters
    ----------
    report : Report
        The report to render.
    ctx : CardContext
        The theme and width to lay out against.

    Returns
    -------
    list[Text]
        A `Report` row when one was written, then the meta line.
    """

    # The path is left whole even past the width: it is there to be copied.
    lines = [Text(f"  Report  {ctx.report_path}")] if ctx.report_path else []
    parts = [f"schema {report.schema_version}"]
    if report.policy_version is not None:
        parts.append(f"policy v{report.policy_version}")
    if report.duration_s is not None:
        parts.append(f"{report.duration_s:.2f}s")
    meta = Text(f" {ctx.theme.sep} ".join(parts), style=ctx.theme.style("muted"))
    lines.append(ctx.theme.fit(meta, ctx.width))
    return lines
