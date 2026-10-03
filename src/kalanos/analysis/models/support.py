"""Sample support shared by measurements and findings."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum

# External
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


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

        if self.end_exclusive <= self.start:
            raise ValueError("interval ends before it starts")
        if (self.support_start is None) != (self.support_end_exclusive is None):
            raise ValueError("support bounds must be supplied together")
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

    A channel-level spectrum result is about one channel and supports the whole episode;
    a stream-level dropout is about the stream and supports an interval.
    The subject (episode/stream/channel) and the support are different axes
    and are recorded separately.

    Attributes
    ----------
    kind : SupportKind
    index_space : str or None
        Which stream's sample index the intervals count in; `None` for `WHOLE_EPISODE`.
        Source order unless `SourceOrder` says otherwise.
    intervals : list[SampleInterval]
        Empty for `WHOLE_EPISODE`.
        A metric that measured the whole episode never invents an interval.
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
