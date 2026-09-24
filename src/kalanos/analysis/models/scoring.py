"""What one level's grade looks like: a 0-100 score, a letter, train-readiness.

`ScoreResult` is the output of every rollup step. Channel, Stream, Episode
and Dataset all produce one through this same shape, so `scoring`
never needs a level-specific result type — see `kalanos.analysis.scoring.score`
for what actually builds one.

`Finding` is scoring's other output: a flat, located record of one metric
that graded as a defect. It's addressed down to the channel only if the
metric itself ran at channel level, and carries enough evidence to check
the claim without rerunning anything.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum
from typing import Any

# External
from pydantic import BaseModel, Field

# Internal
from kalanos.analysis.models.metrics import Level


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Grade(str, Enum):
    """The letter a score maps to, per docs/METRICS.md's scoring table."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class ScoreResult(BaseModel):
    """The rolled-up outcome at one level of the Dataset tree.

    Attributes
    ----------
    level : Level
        Where in the rollup this result attaches.
    score : float or None
        The weighted 0-100 score, or `None` when nothing at or below this
        level had a graded, applicable result to roll up — a level with
        nothing to say stays silent rather than defaulting to zero.
    grade : Grade or None
        `score` mapped through docs/METRICS.md's table,
        or `None` alongside a `None` score.
    train_ready : bool or None
        Whether `score` clears the training-readiness bar,
        or `None` alongside a `None` score.
    n_contributing : int
        How many results fed `score` — graded metrics for `score_metrics`,
        scored children for `rollup`.
        Excludes `report_only`, `not_applicable`, and `score=None` results.
    families : dict[str, float]
        Each family's own 0-100 score, keyed by family name. Set from
        `score_metrics`' per-family fold, before the fail penalty, which
        applies only to `score`. `rollup` carries a family forward as the
        mean over the children that measured it, missing from a family
        no child below measured.
    """

    level: Level
    score: float | None
    grade: Grade | None
    train_ready: bool | None
    n_contributing: int
    families: dict[str, float] = Field(default_factory=dict)


class Severity(str, Enum):
    """How bad a graded metric turned out, assigned by the policy in scoring.

    A metric returns a measurement; scoring is the only place that turns it
    into a verdict.
    """

    # fmt: off
    CRITICAL = "critical"
    WARNING  = "warning"
    # fmt: on


class FindingLocation(BaseModel):
    """Where one stream or channel sits in the graded tree, for addressing its findings.

    Attributes
    ----------
    episode_id : str
        The recording a finding belongs to.
    stream : str or None
        The stream's taxonomy type, or `None` above stream level.
    instance : str or None
        Which subject the stream belongs to,
        or `None` for a single-subject recording — mirrors `domain.Stream.instance`.
    channel : str or None
        The channel a finding was raised on, or `None` above channel level.
    """

    episode_id: str
    stream: str | None = None
    instance: str | None = None
    channel: str | None = None


class Finding(BaseModel):
    """One graded metric that came back a defect, addressed and evidenced.

    Attributes
    ----------
    metric_id : str
        The metric's policy key, `"family.name"`.
    family : str
        The family this metric belongs to, e.g. `"timing"`.
    severity : Severity
        The verdict the policy assigned.
    value : float
        The metric's raw computed value.
    unit : str or None
        The unit `value` is expressed in, or `None` when there is none.
    points : float
        The 0-100 this metric scored — its own contribution to the rollup.
    episode_id : str
        The recording this finding was raised in.
    stream : str or None
        The stream's taxonomy type, or `None` for a finding raised above stream level —
        an episode-level metric belongs to no one stream.
    instance : str or None
        Which subject the stream belongs to,
        or `None` for a single-subject recording.
    channel : str or None
        The channel this finding was raised on, or `None` above channel level.
    evidence : dict[str, Any]
        Supporting detail behind the verdict, carried over from the metric's
        own `MetricResult.evidence`.
    """

    metric_id: str
    family: str
    severity: Severity
    value: float
    unit: str | None
    points: float
    episode_id: str
    stream: str | None = None
    instance: str | None = None
    channel: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
