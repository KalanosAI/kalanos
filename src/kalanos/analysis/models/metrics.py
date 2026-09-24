"""What one metric computed, and at which level of the rollup it attaches.

`MetricResult.status` is a closed enum, so scoring can rely on the five statuses
exhaustively rather than guessing at what a stage might have written.

`Requires`, `StreamContext` and `ChannelContext` are the other half of the boundary:
what a metric function declares it needs, and what the registry hands it once that
declaration is satisfied.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# External
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal
from kalanos.analysis.models.domain import Channel, Episode, Payload, Stream


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class MetricStatus(str, Enum):
    """The closed set of outcomes a metric can report."""

    # fmt: off
    GOOD            = "good"
    WARNING         = "warning"
    CRITICAL        = "critical"
    REPORT_ONLY     = "report_only"     # Metric is shown but never graded
    NOT_APPLICABLE  = "not_applicable"  # Requirements were not met on this channel
    # fmt: on


class Level(str, Enum):
    """The levels a score rolls up through, Channel first."""

    # fmt: off
    CHANNEL = "channel"
    STREAM  = "stream"
    EPISODE = "episode"
    DATASET = "dataset"
    # fmt: on


class Family(str, Enum):
    """The question a metric asks, and the weight key its policy entry falls under."""

    # fmt: off
    TIMING      = "timing"
    INTEGRITY   = "integrity"
    MOTION      = "motion"
    CONSISTENCY = "consistency"
    VISION      = "vision"
    COVERAGE    = "coverage"
    CALIBRATION = "calibration"
    ANNOTATION  = "annotation"
    SCHEMA      = "schema"
    # fmt: on


class MetricResult(BaseModel):
    """What one metric computed for one channel, stream, episode or dataset.

    Attributes
    ----------
    value : float or None
        The computed value, or `None` when `status` is `not_applicable`.
    unit : str or None
        The unit `value` is expressed in, or `None` when there is none.
    status : MetricStatus
        The graded outcome.
    evidence : dict[str, Any]
        Supporting detail behind `status` and `value`,
        e.g. sample counts or the thresholds compared against.
    """

    value: float | None
    unit: str | None
    status: MetricStatus
    evidence: dict[str, Any] = Field(default_factory=dict)


class Requires(BaseModel):
    """What a context must satisfy before the registry will call a metric.

    The registry checks these once, so a metric function never has to guard
    against a precondition its own declaration already ruled out.

    Attributes
    ----------
    regular_sampling : bool
        Whether the context's source must have classified as regularly sampled.
    min_samples : int
        The fewest data points the context must carry.
    taxonomy : list[str]
        The taxonomy types this metric needs, read one of two deliberately
        different ways depending on level. At CHANNEL and STREAM, matched
        any-of: the context's own type must be one of these. At EPISODE,
        matched all-of: every listed type must be present among the
        episode's streams. An empty list gates on nothing.
    """

    regular_sampling: bool = False
    min_samples: int = 0
    taxonomy: list[str] = Field(default_factory=list)


@runtime_checkable
class MetricContext(Protocol):
    """The structural contract every metric context satisfies, whatever its level.

    A Protocol, because `StreamContext` stores `is_regular` as a field
    while `ChannelContext` computes it from its stream, and a Pydantic base class
    cannot declare a field that a subclass overrides with a property.
    """

    @property
    def is_regular(self) -> bool:
        """Whether inference classified this source's sampling as regular."""

        ...

    @property
    def n_samples(self) -> int:
        """Count the samples this context's own level ranges over."""

        ...

    @property
    def taxonomy_type(self) -> str:
        """The taxonomy type of the signal this context's metrics read."""

        ...


class StreamContext(BaseModel):
    """Everything a stream-level metric function is handed, gathered in one place.

    `timestamps` and `payload` read straight through to `stream`, rather than
    carrying a second copy that the alignment invariant between them — already
    validated once, on `Stream` itself — could then drift out of step with.

    Attributes
    ----------
    stream : Stream
        The stream this context was built for.
    is_regular : bool
        Whether inference classified this source's sampling as regular —
        read from that verdict, never recomputed here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stream: Stream
    is_regular: bool

    @property
    def timestamps(self) -> pl.Series:
        """The stream's own time column.

        Returns
        -------
        pl.Series
            `self.stream.timestamps`.
        """

        return self.stream.timestamps

    @property
    def payload(self) -> Payload | None:
        """The stream's data, fetched by the caller if a metric needs it.

        Returns
        -------
        Payload or None
            `self.stream.payload`.
        """

        return self.stream.payload

    @property
    def n_samples(self) -> int:
        """Count the stream's timestamps.

        Returns
        -------
        int
            `len(timestamps)`.
        """

        return len(self.timestamps)

    @property
    def taxonomy_type(self) -> str:
        """The stream's own taxonomy type.

        Returns
        -------
        str
            `self.stream.taxonomy_type`.
        """

        return self.stream.taxonomy_type


class ChannelContext(BaseModel):
    """Everything a channel-level metric function is handed, gathered in one place.

    Attributes
    ----------
    channel : Channel
        The channel this context was built for.
    values : pl.Series
        The channel's own data, in its stream's row order.
    stream : StreamContext
        The stream this channel belongs to, for its shared timestamps and clock.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    channel: Channel
    values: pl.Series
    stream: StreamContext

    @property
    def is_regular(self) -> bool:
        """Read the regularity verdict from this channel's own stream.

        Returns
        -------
        bool
            `self.stream.is_regular`.
        """

        return self.stream.is_regular

    @property
    def n_samples(self) -> int:
        """Count the channel's own values.

        Returns
        -------
        int
            `len(values)`.
        """

        return len(self.values)

    @property
    def taxonomy_type(self) -> str:
        """Read the taxonomy type from this channel's own stream.

        Returns
        -------
        str
            `self.stream.taxonomy_type`.
        """

        return self.stream.taxonomy_type

    @model_validator(mode="after")
    def _values_and_timestamps_stay_aligned(self) -> "ChannelContext":
        """Refuse a context whose values cannot be read index-for-index against time.

        Returns
        -------
        ChannelContext
            `self`, unchanged, once the lengths agree.

        Raises
        ------
        ValueError
            If `values` and the stream's `timestamps` carry a different number of rows.
        """

        if len(self.values) != len(self.stream.timestamps):
            raise ValueError(
                f"values has {len(self.values)} rows but the stream's "
                f"timestamps has {len(self.stream.timestamps)}; a ChannelContext "
                "must carry one timestamp per value"
            )
        return self


class EpisodeContext(BaseModel):
    """Everything an episode-level metric function is handed: the whole recording.

    Deliberately not a `MetricContext`. A recording has no single regularity
    verdict and no single sample count, so it implements neither — an episode
    metric gates on taxonomy alone and checks anything structural itself, on the
    streams it actually reads.

    Attributes
    ----------
    episode : Episode
        The recording this context was built for.
    """

    episode: Episode

    @property
    def streams(self) -> list[Stream]:
        """The episode's own streams.

        Returns
        -------
        list[Stream]
            `self.episode.streams`.
        """

        return self.episode.streams

    @property
    def taxonomy_types(self) -> frozenset[str]:
        """Every distinct taxonomy type carried by the episode's streams.

        Returns
        -------
        frozenset[str]
            What the all-of taxonomy gate tests membership against.
        """

        return frozenset(stream.taxonomy_type for stream in self.streams)

    def streams_of(self, taxonomy_type: str) -> list[Stream]:
        """List the episode's streams carrying exactly one taxonomy type.

        Parameters
        ----------
        taxonomy_type : str
            The taxonomy type to filter by.

        Returns
        -------
        list[Stream]
            The matching streams, in episode order. Possibly empty.
        """

        return [
            stream for stream in self.streams if stream.taxonomy_type == taxonomy_type
        ]


# The contexts a metric function can receive.
MetricInput = ChannelContext | StreamContext | EpisodeContext
