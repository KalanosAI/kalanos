"""The canonical model: the recordings one run analysed.

`Dataset → Episode → Stream → Channel` mirrors `docs/ARCHITECTURE.md`.
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
from upath import UPath

# Internal
from kalanos.analysis.models.paths import AnyPath


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The canonical stream time axis's name and unit, once an adapter has normalised it.
CANONICAL_TIME_COLUMN = "time_s"
CANONICAL_TIME_UNIT = "s"

# The taxonomy_type prefix an adapter falls back to when nothing in the
# dictionary matched, so an unrecognised field reaches the report instead of
# disappearing.
UNMAPPED_TAXONOMY_PREFIX = "unmapped"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Kind(str, Enum):
    """The shape of a Stream's payload."""

    # fmt: off
    SERIES  = "series"
    IMAGE   = "image"
    VIDEO   = "video"
    POINTS  = "points"
    EVENTS  = "events"
    TEXT    = "text"
    AUDIO   = "audio"
    # fmt: on


class Clock(str, Enum):
    """Where a Stream's timestamps came from.

    A format that records no per-sample time and only declares a nominal rate
    yields a synthesised timebase. Latency measured against one of those is a
    restatement of the rate, so a metric that depends on real timing needs to be
    able to tell the two apart.
    """

    # fmt: off
    CAPTURE = "capture"  # Stamped when the sample was taken
    RECEIVE = "receive"  # Stamped when it arrived
    LOG     = "log"      # Stamped when it was written
    UNKNOWN = "unknown"  # Recorded but unlabelled, absent, or synthesised from a rate
    # fmt: on


@runtime_checkable
class Payload(Protocol):
    """A Stream's data, behind a handle that need not have fetched it yet.

    Decoding a camera stream is expensive and most of a grade can be computed
    without it, so a payload is fetched only when a metric asks for one.
    `len()` answers how many samples there are without paying that cost.

    The payload does not declare what shape it holds; a Stream's `kind` already
    names what `fetch` returns.
    """

    def __len__(self) -> int:
        """Count the payload's samples without fetching them."""

        ...

    def fetch(self) -> Any:
        """Return the data itself, reading or decoding it if that has not happened."""

        ...


class FramePayload(BaseModel):
    """A series payload that is already in memory: one column per Channel.

    What a table-shaped source produces, where parsing has already read every
    row and there is nothing left to defer.

    Attributes
    ----------
    frame : pl.DataFrame
        The Stream's channels, in the row order of its timestamps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame: pl.DataFrame

    def __len__(self) -> int:
        """Count the rows in the frame.

        Returns
        -------
        int
            The frame's height.
        """

        return self.frame.height

    def fetch(self) -> pl.DataFrame:
        """Return the frame.

        Returns
        -------
        pl.DataFrame
            The frame this payload was built around, unchanged.
        """

        return self.frame


class Attribution(str, Enum):
    """Where a Stream's `instance` came from, so that `instance is None` is unambiguous.

    A recording with one subject and a recording whose instance key was empty for
    these rows both leave `instance` unset. They are not the same thing: the second
    is a defect in the data, and a grade that cannot tell them apart hides it.
    """

    # fmt: off
    SINGLE       = "single"        # One subject; there was no key to split on
    KEYED        = "keyed"         # `instance` is the value the key held for these rows
    UNATTRIBUTED = "unattributed"  # There was a key, but these rows had no value for it
    # fmt: on


class Channel(BaseModel):
    """One scalar series within a Stream.

    Attributes
    ----------
    name : str
        The column or field name as the source had it.
    axis : str or None
        The axis letter or numeric index the column name ended in, as
        `inference.roles` resolved it, or `None` when it carried neither.
        Orders channels within a stream and tells an axes-shaped stream
        which member is which.
    """

    name: str
    axis: str | None = None


class Stream(BaseModel):
    """One taxonomy-typed timestamped signal within an Episode.

    Attributes
    ----------
    taxonomy_type : str
        What this signal is, as a key in `dictionary.yaml` — or
        `UNMAPPED_TAXONOMY_PREFIX.<name>` when no entry matched, so an
        unrecognised field still reaches the report instead of disappearing.
    instance : str or None
        Which subject this signal belongs to when a recording holds several,
        such as one arm of a bimanual robot or one machine on a line.
        `None` when the recording has a single subject
        or when `attribution` is `UNATTRIBUTED`.
    attribution : Attribution
        Where `instance` came from.
        Defaults to `KEYED` when the input carries an `instance`, `SINGLE` otherwise.
    kind : Kind
        The payload's shape.
    timestamps : pl.Series
        Sample times in seconds, aligned index-for-index with the payload.
        Eager, because every timing metric needs them and they are cheap.
    payload : Payload or None
        The data, fetched on demand. `None` when the structure is known but
        nothing has been attached to it yet.
    source_path : UPath
        The file this stream was read from. A stream comes from exactly one file.
    source_field : str or None
        What the stream was called in that file,
        or `None` when it had no name of its own.
    clock : Clock
        Which timebase `timestamps` are on.
    is_regular : bool
        Whether the sampling behind this stream classified as regular, which is
        what gates a metric declaring `requires.regular_sampling`. Defaults
        `False` so a format that cannot say does not claim regularity it has
        not shown.
    channels : list[Channel]
        The scalar series inside this stream.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    taxonomy_type: str
    instance: str | None = None
    # A default so a caller who omits `attribution` type-checks:
    # the before-validator below always fills the real value first.
    attribution: Attribution = Attribution.SINGLE
    kind: Kind
    timestamps: pl.Series
    payload: Payload | None = None
    source_path: AnyPath
    source_field: str | None = None
    clock: Clock = Clock.UNKNOWN
    is_regular: bool = False
    channels: list[Channel] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _default_attribution_from_instance(cls, data: Any) -> Any:
        """Fill in `attribution` from `instance` when a caller does not name it.

        A caller that sets `instance` gets `KEYED`;
        one that leaves it `None` gets `SINGLE`.

        Parameters
        ----------
        data : Any
            The raw input to the model.

        Returns
        -------
        Any
            `data`, with `attribution` filled in when it was a dict missing one.
        """

        if isinstance(data, dict) and "attribution" not in data:
            data = {
                **data,
                "attribution": (
                    Attribution.KEYED
                    if data.get("instance") is not None
                    else Attribution.SINGLE
                ),
            }
        return data

    @model_validator(mode="after")
    def _payload_and_timestamps_stay_aligned(self) -> "Stream":
        """Refuse a stream whose payload cannot be read index-for-index against time.

        Catches a length mismatch here, before it surfaces later as a `StatisticsError`
        or an off-by-one inside whichever metric first pairs payload and timestamps.
        A `None` payload has nothing to check.

        Returns
        -------
        Stream
            `self`, unchanged, once the lengths agree or there is no payload.

        Raises
        ------
        ValueError
            If `payload` and `timestamps` carry a different number of rows.
        """

        if self.payload is not None and len(self.payload) != len(self.timestamps):
            raise ValueError(
                f"payload has {len(self.payload)} rows but timestamps has "
                f"{len(self.timestamps)}; a Stream must carry one timestamp "
                "per sample"
            )
        return self

    @model_validator(mode="after")
    def _attribution_and_instance_agree(self) -> "Stream":
        """Refuse an `attribution` that contradicts whether `instance` is set.

        `KEYED` without an `instance` and `SINGLE` or `UNATTRIBUTED` with one
        are both the third state `Attribution` exists to rule out.

        Returns
        -------
        Stream
            `self`, unchanged, once `attribution` and `instance` agree.

        Raises
        ------
        ValueError
            If `attribution` is `KEYED` and `instance` is `None`, or if
            `attribution` is `SINGLE` or `UNATTRIBUTED` and `instance` is not `None`.
        """

        if self.attribution is Attribution.KEYED and self.instance is None:
            raise ValueError("attribution is KEYED but instance is None")
        if (
            self.attribution in (Attribution.SINGLE, Attribution.UNATTRIBUTED)
            and self.instance is not None
        ):
            raise ValueError(
                f"attribution is {self.attribution.value} but instance is "
                f"{self.instance!r}"
            )
        return self


class Episode(BaseModel):
    """One recording, which may draw on several files at once.

    Attributes
    ----------
    id : str
        The recording's identifier. An adapter names it after the file it read,
        and `pipeline.run` re-mints it to that file's place under the walked root.
    streams : list[Stream]
        The signals captured during it.
    """

    id: str
    streams: list[Stream] = Field(default_factory=list)

    @property
    def source_paths(self) -> list[UPath]:
        """List the files this episode's streams were read from, first seen first.

        Derived rather than stored, so it cannot drift from the streams it describes.

        Returns
        -------
        list[UPath]
            Each file once, in the order its first stream appears.
        """

        seen: dict[UPath, None] = {}
        for stream in self.streams:
            seen.setdefault(stream.source_path, None)
        return list(seen)


class Dataset(BaseModel):
    """Everything one run analysed.

    Attributes
    ----------
    root : UPath
        The path the user pointed at — a folder, or a single recording.
    episodes : list[Episode]
        The recordings found within it.
    """

    root: AnyPath
    episodes: list[Episode] = Field(default_factory=list)
