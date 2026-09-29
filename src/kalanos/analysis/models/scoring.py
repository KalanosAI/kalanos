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
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal
from kalanos.analysis.models.eligibility import BlockingRoute, Consequence
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
        Compatibility only. Since schema 7 no score threshold sets this: at
        episode level it mirrors `GradedEpisode.eligibility` (`True` pass,
        `False` blocked, `None` review/unknown) and is filled in by report
        assembly; at every other level it is `None`. Do not read it as a
        decision; read the eligibility.
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


class SupportKind(str, Enum):
    """Whether a finding's evidence is about the whole episode or located samples."""

    # fmt: off
    WHOLE_EPISODE = "whole_episode"
    INTERVALS     = "intervals"
    # fmt: on


class SampleInterval(BaseModel):
    """A half-open run of samples, zero-based, in a named index space.

    Attributes
    ----------
    start : int
        First affected sample.
    end_exclusive : int
        One past the last affected sample.
    support_start, support_end_exclusive : int or None
        The wider run a filter or derivative depended on, when it is wider
        than the defect itself. Never narrower than `[start, end_exclusive)`.
    """

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end_exclusive: int = Field(ge=0)
    support_start: int | None = Field(default=None, ge=0)
    support_end_exclusive: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "SampleInterval":
        """An interval runs forward, and its support contains it."""

        if self.end_exclusive < self.start:
            raise ValueError("interval ends before it starts")
        if self.support_start is not None and self.support_start > self.start:
            raise ValueError("support starts after the defect")
        if (
            self.support_end_exclusive is not None
            and self.support_end_exclusive < self.end_exclusive
        ):
            raise ValueError("support ends before the defect")
        return self


class TemporalSupport(BaseModel):
    """Where in time a finding's evidence lives, separate from what it is about.

    A channel-level spectrum result is about one channel and supports the
    whole episode; a stream-level dropout is about the stream and supports an
    interval. The subject (episode/stream/channel) and the support are
    different axes and are recorded separately.

    Attributes
    ----------
    kind : SupportKind
    index_space : str or None
        Which stream's sample index the intervals count in; `None` for
        `WHOLE_EPISODE`. Source order unless `SourceOrder` says otherwise.
    intervals : list[SampleInterval]
        Empty for `WHOLE_EPISODE`. A metric that measured the whole episode
        never invents an interval.
    """

    kind: SupportKind = SupportKind.WHOLE_EPISODE
    index_space: str | None = None
    intervals: list[SampleInterval] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "TemporalSupport":
        """Intervals need an index space; whole-episode support has none."""

        if self.kind == SupportKind.INTERVALS and not self.intervals:
            raise ValueError("interval support without intervals")
        if self.kind == SupportKind.WHOLE_EPISODE and self.intervals:
            raise ValueError("whole-episode support carries intervals")
        if self.kind == SupportKind.INTERVALS and self.index_space is None:
            raise ValueError("intervals without an index space")
        return self


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
    consequence : Consequence
        What the policy says this finding does to eligibility. Severity is
        the assessment; this is the decision about it.
    route : BlockingRoute or None
        For `BLOCK`, which route justified it. A statistical block without
        an accepted calibration manifest is downgraded to review before it
        reaches here.
    support : TemporalSupport
        Where in time the evidence lives. Whole-episode unless the metric
        located it.
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
    consequence: Consequence = Consequence.REPORT_ONLY
    route: BlockingRoute | None = None
    support: TemporalSupport = Field(default_factory=TemporalSupport)
