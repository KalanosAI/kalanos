"""Render a Report as JSON, YAML or HTML."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections import Counter

# External
import yaml
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

# Internal
from kalanos.analysis.models.metrics import MetricResult, MetricStatus
from kalanos.analysis.models.report import Report
from kalanos.analysis.models.scoring import Finding, ScoreResult
from kalanos.analysis.reporting.registry import reporter


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# PackageLoader resolves through the package's own loader, so this works
# whether `kalanos` is an editable checkout or unzipped from a wheel —
# the same reasoning `assets/policy.py` follows for its own resource path.
_TEMPLATE_ENVIRONMENT = Environment(
    loader=PackageLoader("kalanos.analysis.reporting", "templates"),
    # select_autoescape's own extension list doesn't include ".j2" — this
    # template is named report.html.j2, so autoescaping needs telling explicitly
    # or it silently turns itself off for the one file that renders user paths.
    autoescape=select_autoescape(enabled_extensions=("html", "j2", "xml")),
    undefined=StrictUndefined,
)

# How many findings the page shows before the rest go behind a disclosure.
_FINDINGS_PREVIEW = 8


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


def _score_attr(score: ScoreResult) -> str:
    """Format a ScoreResult's raw number for a `data-score` attribute.

    Parameters
    ----------
    score : ScoreResult
        The score to format.

    Returns
    -------
    str
        The score with full precision, or an empty string when `score.score` is `None`.
    """

    return "" if score.score is None else repr(score.score)


def _score_text(score: ScoreResult) -> str:
    """Format a ScoreResult for display: a grade and a rounded number.

    Parameters
    ----------
    score : ScoreResult
        The score to format.

    Returns
    -------
    str
        `"{grade} {score:.1f}"`, or `"not graded"` when nothing rolled up to
        this level — never a bare `0`, which would read as a real, low score.
    """

    if score.score is None:
        return "not graded"
    grade = score.grade.value if score.grade is not None else "?"
    return f"{grade} {score.score:.1f}"


def _metric_text(metric: MetricResult) -> str:
    """Format one metric's value for display, honouring its graded status.

    Parameters
    ----------
    metric : MetricResult
        The metric result to format.

    Returns
    -------
    str
        `"—"` when the metric is `not_applicable` or carries no value —
        a metric with no opinion must never be mistaken for a measured zero.
        Otherwise the value, rounded, with its unit when it has one.
    """

    if metric.status == MetricStatus.NOT_APPLICABLE or metric.value is None:
        return "—"
    unit = f" {metric.unit}" if metric.unit else ""
    return f"{metric.value:.4g}{unit}"


_TEMPLATE_ENVIRONMENT.filters["score_attr"] = _score_attr
_TEMPLATE_ENVIRONMENT.filters["score_text"] = _score_text
_TEMPLATE_ENVIRONMENT.filters["metric_text"] = _metric_text


def _episode_finding_counts(findings: list[Finding]) -> dict[str, int]:
    """Count findings per episode id, so the page can open the episodes that have one.

    Parameters
    ----------
    findings : list[Finding]
        The report's flat findings list.

    Returns
    -------
    dict[str, int]
        How many findings each episode id carries.
    """

    return Counter(finding.episode_id for finding in findings)


def _stream_finding_counts(
    findings: list[Finding],
) -> dict[tuple[str, str, str | None], int]:
    """Count findings per (episode id, stream taxonomy type, instance).

    Two streams in one episode can share a taxonomy type and differ only by instance,
    so the instance has to be part of the key.

    Parameters
    ----------
    findings : list[Finding]
        The report's flat findings list.

    Returns
    -------
    dict[tuple[str, str, str or None], int]
        How many findings each stream carries. A finding raised above stream
        level is not counted here; `_episode_finding_counts` already counts it.
    """

    return Counter(
        (finding.episode_id, finding.stream, finding.instance)
        for finding in findings
        if finding.stream is not None
    )


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@reporter(name="json", extensions=(".json",))
def render_json(report: Report) -> str:
    """Render a Report as indented, parseable JSON.

    Parameters
    ----------
    report : Report
        The report to render.

    Returns
    -------
    str
        The report as JSON text, matching `Report`'s own field names.
    """

    return report.model_dump_json(indent=2)


@reporter(name="yaml", extensions=(".yaml", ".yml"))
def render_yaml(report: Report) -> str:
    """Render a Report as YAML, in the model's own field order.

    Parameters
    ----------
    report : Report
        The report to render.

    Returns
    -------
    str
        The report as YAML text. Parses back into the model with
        `Report.model_validate(yaml.safe_load(...))`.
    """

    # mode="json" turns Path, Grade, Severity, Level, MetricStatus and SkipReason into
    # strings yaml.safe_dump accepts and Report.model_validate can read back.
    return yaml.safe_dump(
        report.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


@reporter(name="html", extensions=(".html",))
def render_html(report: Report) -> str:
    """Render a Report as a self-contained page, collapsed to a summary.

    Parameters
    ----------
    report : Report
        The report to render.

    Returns
    -------
    str
        The rendered page.
        Every score-bearing element carries `data-level`,
        `data-name` and `data-score`; every metric carries `data-status` —
        so a test can walk the markup and check it against the model directly,
        rather than parsing rendered prose back into numbers.
    """

    template = _TEMPLATE_ENVIRONMENT.get_template("report.html.j2")
    return template.render(
        report=report,
        episode_findings=_episode_finding_counts(report.findings),
        stream_findings=_stream_finding_counts(report.findings),
        findings_preview=_FINDINGS_PREVIEW,
    )
