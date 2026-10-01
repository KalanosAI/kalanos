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
from kalanos.analysis.models.binding import ChannelBinding
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
    CAPTURE       = "capture"        # Stamped when the sample was taken
    RECEIVE       = "receive"        # Stamped when it arrived
    LOG           = "log"            # Stamped when it was written
    RECONSTRUCTED = "reconstructed"  # From frame numbers or a rate; measured nothing
    UNKNOWN       = "unknown"        # Recorded but unlabelled, or absent
    # fmt: on


class TimestampDtype(str, Enum):
    """The float format a Stream's timestamps were stored in at the source.

    Read before any cast, it bounds how far rounding can have moved each stamp.
    Integer and text sources are `FLOAT64`,
    since the conversion to float64 seconds is their only rounding.
    """

    # fmt: off
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    # fmt: on

    @property
    def epsilon(self) -> float:
        """The machine epsilon: the gap between 1.0 and the next float above it."""

        match self:
            case TimestampDtype.FLOAT16:
                return 2**-10
            case TimestampDtype.FLOAT32:
                return 2**-23
            case TimestampDtype.FLOAT64:
                return 2**-52


class ClockOrigin(str, Enum):
    """Where a Stream's timestamps come from, at the resolution schema 7 needs.

    `Clock` collapses "the adapter built this grid" and "the file held a
    perfectly uniform series" into `RECONSTRUCTED`. They differ: the first is
    known generation, the second is an inference. `ClockInfo.origin_evidence`
    says which, and `Clock` is derived from this for compatibility.
    """

    # fmt: off
    CAPTURE      = "capture"       # Producer evidence: stamped at acquisition
    RECEIVE      = "receive"       # Stamped on arrival at the recorder
    PUBLISH      = "publish"       # Stamped when published (MCAP publish_time)
    LOG          = "log"           # Stamped when written (MCAP log_time)
    PRESENTATION = "presentation"  # Media PTS; relates to capture only with evidence
    GENERATED    = "generated"     # Built by an adapter from index and rate
    SIMULATION   = "simulation"    # A simulator's step clock
    UNKNOWN      = "unknown"       # Recorded but unlabelled, or absent
    # fmt: on

    @property
    def compatibility_clock(self) -> "Clock":
        """The pre-7 `Clock` value that best describes this origin."""

        match self:
            case ClockOrigin.CAPTURE:
                return Clock.CAPTURE
            case ClockOrigin.RECEIVE:
                return Clock.RECEIVE
            case ClockOrigin.PUBLISH | ClockOrigin.LOG:
                return Clock.LOG
            case ClockOrigin.GENERATED:
                return Clock.RECONSTRUCTED
            case _:
                return Clock.UNKNOWN


class OriginEvidence(str, Enum):
    """How confidently `ClockInfo.origin` is known."""

    # fmt: off
    PRODUCER = "producer"  # The source or its schema declares it
    ADAPTER  = "adapter"   # The adapter did it itself (it built the grid)
    INFERRED = "inferred"  # Matched a pattern, e.g. frame_index / fps
    NONE     = "none"      # Nothing says
    # fmt: on


class ClockInfo(BaseModel):
    """A Stream's timebase, with its origin and history stated.

    Attributes
    ----------
    origin : ClockOrigin
    origin_evidence : OriginEvidence
        Whether `origin` is declared, the adapter's own doing, or inferred.
        Inferred generation never certifies acquisition timing; neither does
        adapter generation.
    domain : str or None
        A clock-domain identifier shared by streams on one clock.
    source_field : str or None
        The source column the timestamps were read from.
    native_unit : str or None
        The unit at the source (`ns`, `s`, `frame`), before conversion.
    native_dtype : str or None
        The source dtype (`int64`, `float32`), before any cast.
    epoch : str or None
        What zero means at the source, when declared.
    transforms : list[str]
        Every conversion applied, in order, e.g. `ns->s`, `sorted`.
    """

    origin: ClockOrigin = ClockOrigin.UNKNOWN
    origin_evidence: OriginEvidence = OriginEvidence.NONE
    domain: str | None = None
    source_field: str | None = None
    native_unit: str | None = None
    native_dtype: str | None = None
    epoch: str | None = None
    transforms: list[str] = Field(default_factory=list)
    tick_period_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @classmethod
    def from_legacy(cls, clock: "Clock") -> "ClockInfo":
        """The best `ClockInfo` a pre-7 `Clock` supports, uncertainty preserved."""

        match clock:
            case Clock.CAPTURE:
                return cls(
                    origin=ClockOrigin.CAPTURE, origin_evidence=OriginEvidence.INFERRED
                )
            case Clock.RECEIVE:
                return cls(
                    origin=ClockOrigin.RECEIVE, origin_evidence=OriginEvidence.PRODUCER
                )
            case Clock.LOG:
                return cls(
                    origin=ClockOrigin.LOG, origin_evidence=OriginEvidence.PRODUCER
                )
            case Clock.RECONSTRUCTED:
                # The old value did not say whether the adapter built the grid
                # or merely recognised one; the honest migration is inferred.
                return cls(
                    origin=ClockOrigin.GENERATED,
                    origin_evidence=OriginEvidence.INFERRED,
                )
            case _:
                return cls()

    @property
    def certifies_acquisition(self) -> bool:
        """Whether timing metrics may treat these stamps as measured capture time."""

        return (
            self.origin == ClockOrigin.CAPTURE
            and self.origin_evidence == OriginEvidence.PRODUCER
        )


class SourceOrder(BaseModel):
    """Whether a Stream's rows are still in source order, and how to get back.

    Attributes
    ----------
    preserved : bool
        `True` when row `i` of the payload is row `i` of the source.
    original_index : list[int] or None
        When `preserved` is `False`, the source row index of each current row,
        so a finding can be mapped back to the sample it came from.
    transform : str or None
        What reordered the rows, e.g. `sorted_by_timestamp`.
    """

    preserved: bool = True
    original_index: list[int] | None = None
    transform: str | None = None


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


class MappingSource(str, Enum):
    """How a Stream's `taxonomy_type` was decided."""

    # fmt: off
    DICTIONARY     = "dictionary"      # The dictionary matched the field's own name
    DECLARED_NAMES = "declared_names"  # The declared channel names agreed on one type
    OVERRIDE       = "override"        # The user mapped the field for this run
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
    source_index: int | None = Field(default=None, ge=0)
    declared_name: str | None = None
    binding: ChannelBinding | None = None


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
    mapping_source : MappingSource or None
        How `taxonomy_type` was decided. `None` when the stream is unmapped.
        Defaults to `DICTIONARY` for a typed stream when a caller does not name it.
    clock : Clock
        Which timebase `timestamps` are on — the compatibility view.
    clock_info : ClockInfo or None
        The timebase's origin, evidence and transform history. `None` when the
        adapter has not been taught to say; consumers then fall back to
        `ClockInfo.from_legacy(clock)`, which preserves the uncertainty.
    source_order : SourceOrder
        Whether the rows are still in source order, and the index map back
        when they are not. An ordering check needs the source order; a
        sorted view is a transformation, not the recording.
    timestamp_dtype : TimestampDtype
        The float format `timestamps` were stored in at the source,
        which bounds their rounding once cast to float64.
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
    native_timestamps: pl.Series | None = None
    payload: Payload | None = None
    source_path: AnyPath
    source_field: str | None = None
    source_identity: str | None = None
    mapping_source: MappingSource | None = None
    clock: Clock = Clock.UNKNOWN
    clock_info: ClockInfo | None = None
    source_order: SourceOrder = Field(default_factory=SourceOrder)
    timestamp_dtype: TimestampDtype
    is_regular: bool = False
    channels: list[Channel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clock_and_order_contract(self) -> "Stream":
        """Keep legacy clock labels derived and native rows aligned.

        A legacy capture label is insufficient evidence for acquisition timing.
        Explicit ClockInfo is authoritative when both representations are given.
        A transformed stream without an index map remains unobservable in source
        order; consumers must not silently inspect the sorted view instead.
        """

        if self.clock_info is None:
            self.clock_info = ClockInfo.from_legacy(self.clock)
        self.clock = self.clock_info.origin.compatibility_clock
        if self.native_timestamps is not None and len(self.native_timestamps) != len(
            self.timestamps
        ):
            raise ValueError("native timestamps must align with canonical timestamps")
        indices = self.source_order.original_index
        if indices is not None and (
            len(indices) != len(self.timestamps)
            or len(set(indices)) != len(indices)
            or any(i < 0 for i in indices)
        ):
            raise ValueError(
                "source row indices must be unique, nonnegative and aligned"
            )
        if (
            self.source_order.preserved
            and indices is not None
            and indices != sorted(indices)
        ):
            raise ValueError("preserved source order requires increasing row indices")
        return self

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

    @model_validator(mode="before")
    @classmethod
    def _default_mapping_source_from_taxonomy(cls, data: Any) -> Any:
        """Fill in `mapping_source` from `taxonomy_type` when a caller does not name it.

        An `unmapped.*` stream gets `None`; a typed one gets `DICTIONARY`.

        Parameters
        ----------
        data : Any
            The raw input to the model.

        Returns
        -------
        Any
            `data`, with `mapping_source` filled in when it was a dict missing one.
        """

        if isinstance(data, dict) and "mapping_source" not in data:
            unmapped = str(data.get("taxonomy_type", "")).startswith(
                f"{UNMAPPED_TAXONOMY_PREFIX}."
            )
            data = {
                **data,
                "mapping_source": None if unmapped else MappingSource.DICTIONARY,
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

    @model_validator(mode="after")
    def _mapping_source_and_taxonomy_agree(self) -> "Stream":
        """Refuse a `mapping_source` that contradicts whether the stream is typed.

        Returns
        -------
        Stream
            `self`, unchanged, once `mapping_source` and `taxonomy_type` agree.

        Raises
        ------
        ValueError
            If an `unmapped.*` stream names a `mapping_source`,
            or a typed stream has none.
        """

        unmapped = self.taxonomy_type.startswith(f"{UNMAPPED_TAXONOMY_PREFIX}.")
        if unmapped and self.mapping_source is not None:
            raise ValueError(
                f"taxonomy_type {self.taxonomy_type!r} is unmapped but "
                f"mapping_source is {self.mapping_source.value}"
            )
        if not unmapped and self.mapping_source is None:
            raise ValueError(
                f"taxonomy_type {self.taxonomy_type!r} is typed but "
                "mapping_source is None"
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
    tasks : list[str] or None
        The natural-language instructions the episode was recorded under, exactly
        as the source stores them, blank strings included. `None` when the dataset
        carries no instruction for any episode, so an absent one is not a defect;
        an empty list when the dataset carries them but this episode has none.
    """

    id: str
    streams: list[Stream] = Field(default_factory=list)
    tasks: list[str] | None = None

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
