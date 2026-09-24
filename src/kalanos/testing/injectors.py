"""A clean synthetic Stream, and defects injected into it one at a time.

Every injector is `Stream -> Stream`: it returns a new object and leaves its
argument untouched, so a test can build one clean recording and derive
several defective variants from it without them interfering.

`apply_defect` is the derivation a contract test drives through `fires_on`:
it picks the injector for a named `Defect` and rebuilds whichever context
type it was handed.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NoReturn, cast

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.models.domain import Channel, Clock, FramePayload, Kind, Stream
from kalanos.analysis.models.metrics import ChannelContext, MetricInput, StreamContext
from kalanos.analysis.optional import load_numpy


if TYPE_CHECKING:
    import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


class Defect(str, Enum):
    """A fault a contract test injects into clean data to prove a metric catches it."""

    # fmt: off
    STUCK_CHANNEL = "stuck_channel"
    DROPOUT       = "dropout"
    JITTER        = "jitter"
    CLOCK_DRIFT   = "clock_drift"
    NULLS         = "nulls"
    FROZEN_FRAMES = "frozen_frames"
    SATURATION    = "saturation"
    DRIFT         = "drift"
    SPIKE         = "spike"
    NOISE         = "noise"
    DEAD_TAXEL    = "dead_taxel"
    HYSTERESIS    = "hysteresis"
    # fmt: on


CLEAN_CHANNELS = ("tcp_pose_x_mm", "tcp_pose_y_mm", "tcp_pose_z_mm")


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class SyntheticFrames:
    """An in-memory sequence of image frames, for a `Kind.IMAGE` test stream.

    Attributes
    ----------
    frames : list
        One `numpy.ndarray` per frame, `(height, width, 3)` uint8.
    """

    frames: "list[np.ndarray]"

    def __len__(self) -> int:
        """Count the frames.

        Returns
        -------
        int
            `len(self.frames)`.
        """

        return len(self.frames)

    def fetch(self) -> "list[np.ndarray]":
        """Return the frames.

        Returns
        -------
        list
            The frames this payload was built around, unchanged.
        """

        return self.frames


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def clean_recording(
    *,
    hz: float = 100.0,
    samples: int = 100,
    channels: Sequence[str] = CLEAN_CHANNELS,
    taxonomy_type: str = "unmapped.tcp_pose",
    instance: str | None = None,
    settle: int = 0,
) -> Stream:
    """Build a regularly sampled Stream with deterministic, non-constant channels.

    Each channel is a sine wave at its own frequency, so a stuck-channel injection reads
    as visibly wrong against it rather than against a flat baseline that would hide the
    defect.
    No randomness: a flaky injector would make every metric test that uses it flaky.

    Parameters
    ----------
    hz : float
        The sampling rate to build timestamps at.
    samples : int
        How many rows the stream carries.
    channels : Sequence[str]
        The channel names to give the payload, one column each.
    settle : int
        When positive, the last `settle` rows of every channel are held at
        the value the channel had at row `samples - settle - 1`, so the
        recording ends static. `0` leaves every channel a pure sine wave.

    Returns
    -------
    Stream
        `kind=Kind.SERIES`, `clock=Clock.CAPTURE`, evenly spaced timestamps,
        and a `FramePayload` with one sine-wave column per channel.
    """

    timestamps = pl.Series("time_s", [index / hz for index in range(samples)])
    columns = {
        name: [
            math.sin(2 * math.pi * (position + 1) * index / samples)
            for index in range(samples)
        ]
        for position, name in enumerate(channels)
    }

    if settle > 0:
        hold_row = samples - settle - 1
        for values in columns.values():
            hold_value = values[hold_row]
            values[samples - settle :] = [hold_value] * settle

    frame = pl.DataFrame(columns)

    return Stream(
        taxonomy_type=taxonomy_type,
        instance=instance,
        kind=Kind.SERIES,
        timestamps=timestamps,
        payload=FramePayload(frame=frame),
        source_path=UPath("synthetic"),
        clock=Clock.CAPTURE,
        channels=[Channel(name=name) for name in channels],
    )


def clean_taxels(
    *,
    hz: float = 50.0,
    samples: int = 100,
    cells: int = 16,
    taxonomy_type: str = "extero.taxel_pressure",
    instance: str | None = None,
) -> Stream:
    """Build a tactile array Stream where every cell traces one load-unload cycle.

    Each cell follows a half-sine envelope that rises to a peak at the
    recording's midpoint and falls back, scaled by a fixed per-cell factor so
    no two cells are identical. No randomness.

    Parameters
    ----------
    hz : float
        The sampling rate to build timestamps at.
    samples : int
        How many rows the stream carries.
    cells : int
        How many taxel channels the array carries.
    taxonomy_type : str
        The stream's taxonomy type.
    instance : str or None
        Which subject this stream belongs to.

    Returns
    -------
    Stream
        `kind=Kind.SERIES`, `clock=Clock.CAPTURE`, evenly spaced timestamps,
        and a `FramePayload` with one `taxel_NN` column per cell.
    """

    timestamps = pl.Series("time_s", [index / hz for index in range(samples)])
    names = [f"taxel_{cell:02d}" for cell in range(cells)]
    envelope = [math.sin(math.pi * index / (samples - 1)) for index in range(samples)]
    frame = pl.DataFrame(
        {
            name: [(1.0 + 0.05 * cell) * value for value in envelope]
            for cell, name in enumerate(names)
        }
    )

    return Stream(
        taxonomy_type=taxonomy_type,
        instance=instance,
        kind=Kind.SERIES,
        timestamps=timestamps,
        payload=FramePayload(frame=frame),
        source_path=UPath("synthetic"),
        clock=Clock.CAPTURE,
        channels=[Channel(name=name) for name in names],
    )


def stream_context(stream: Stream, *, is_regular: bool = True) -> StreamContext:
    """Wrap a Stream in the context a stream-level metric function is handed."""

    return StreamContext(stream=stream, is_regular=is_regular)


def _channel_frame(stream: Stream) -> pl.DataFrame:
    """Fetch a stream's payload as a DataFrame, refusing anything else.

    Every injector below needs to edit a frame column, and only
    `clean_recording`'s `FramePayload` supports that.
    """

    if not isinstance(stream.payload, FramePayload):
        raise ValueError(
            f"stream must carry a FramePayload; got {type(stream.payload).__name__}"
        )
    return stream.payload.frame


def _require_channel(stream: Stream, channel: str) -> None:
    """Raise `ValueError` unless `channel` is one of the stream's own channels."""

    if channel not in {c.name for c in stream.channels}:
        raise ValueError(f"{channel!r} is not one of this stream's channels")


def _window(frame: pl.DataFrame, start: int, length: int) -> int:
    """Validate `[start, start + length)` against a frame's row count.

    Factored out because every injector below needs the same check, worded the same way,
    before it touches a frame.

    Parameters
    ----------
    frame : pl.DataFrame
        The frame the window must fit inside.
    start : int
        The window's first row index.
    length : int
        How many rows the window covers.

    Returns
    -------
    int
        `start + length`, once the window is confirmed to fit.

    Raises
    ------
    ValueError
        If the window falls outside `frame`.
    """

    end = start + length
    if start < 0 or end > frame.height:
        raise ValueError(
            f"window [{start}, {end}) falls outside a frame of height {frame.height}"
        )
    return end


def stick_channel(
    stream: Stream, channel: str, *, start: int = 20, length: int = 40
) -> Stream:
    """Hold one channel at a fixed value across a window, unchanged elsewhere.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    channel : str
        The channel to stick.
    start : int
        The first row index of the stuck window.
    length : int
        How many rows the window covers.

    Returns
    -------
    Stream
        A copy of `stream` whose `channel` holds the value it had just before `start`
        (or at `start`, when `start == 0`) across `[start, start + length)`.
        Every other channel and the timestamps are unchanged.

    Raises
    ------
    ValueError
        If `channel` is not one of the stream's channels, or the window
        falls outside the frame.
    """

    _require_channel(stream, channel)
    frame = _channel_frame(stream)
    end = _window(frame, start, length)

    stuck_value = frame[channel][start if start == 0 else start - 1]
    values = frame[channel].to_list()
    values[start:end] = [stuck_value] * length
    new_frame = frame.with_columns(pl.Series(channel, values))

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def drop_samples(stream: Stream, *, start: int = 40, count: int = 10) -> Stream:
    """Remove a run of rows from the frame and the matching timestamps together.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    start : int
        The first row index to remove.
    count : int
        How many consecutive rows to remove.

    Returns
    -------
    Stream
        A copy of `stream` with rows `[start, start + count)` removed from both
        the frame and the timestamps, keeping the two aligned.

    Raises
    ------
    ValueError
        If the window falls outside the frame.
    """

    frame = _channel_frame(stream)
    end = _window(frame, start, count)

    keep = [row for row in range(frame.height) if not (start <= row < end)]
    new_frame = frame[keep]
    new_timestamps = stream.timestamps.gather(keep)

    return stream.model_copy(
        update={
            "payload": FramePayload(frame=new_frame),
            "timestamps": new_timestamps,
        }
    )


def jitter_clock(stream: Stream, *, milliseconds: float = 2.0) -> Stream:
    """Push alternating timestamps late and early, from index 1 onward.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged. Its payload is untouched.
    milliseconds : float
        The offset to apply, added to odd-indexed timestamps and subtracted
        from even-indexed ones.

    Returns
    -------
    Stream
        A copy of `stream` whose timestamps carry the offset from index 1
        onward; index 0 is unchanged.

    Raises
    ------
    ValueError
        If the offset is large enough to make the timestamps non-monotonic
        — a non-monotonic clock is a different defect than jitter.
    """

    ordered = stream.timestamps.to_list()
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    if gaps:
        # The compressed gap at an odd-to-even boundary shrinks by 2 * offset;
        # a guard against the median would miss a stream with one short gap.
        tightest_gap = min(gaps)
        if 2 * milliseconds / 1000 >= tightest_gap:
            raise ValueError(
                f"{milliseconds} ms is large enough to break monotonicity "
                f"against a smallest gap of {tightest_gap * 1000:.3f} ms"
            )

    offset = milliseconds / 1000
    jittered = list(ordered)
    for index in range(1, len(jittered)):
        jittered[index] += offset if index % 2 == 1 else -offset

    return stream.model_copy(
        update={"timestamps": pl.Series(stream.timestamps.name, jittered)}
    )


def null_run(
    stream: Stream, channel: str, *, start: int = 30, length: int = 10
) -> Stream:
    """Null out a window of one channel, leaving row count and other channels alone.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    channel : str
        The channel to null.
    start : int
        The first row index to null.
    length : int
        How many consecutive rows to null.

    Returns
    -------
    Stream
        A copy of `stream` whose `channel` is null across `[start, start + length)`.
        Row count, timestamps and every other channel are unchanged.

    Raises
    ------
    ValueError
        If `channel` is not one of the stream's channels,
        or the window falls outside the frame.
    """

    _require_channel(stream, channel)
    frame = _channel_frame(stream)
    end = _window(frame, start, length)

    values: list[float | None] = frame[channel].to_list()
    values[start:end] = [None] * length
    new_frame = frame.with_columns(pl.Series(channel, values))

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def stretch_clock(stream: Stream, *, factor: float = 1.4) -> Stream:
    """Ramp the gaps between timestamps linearly from `1.0x` to `factor x`.

    The payload is untouched, so row counts still agree;
    only the clock's own pace drifts.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged. Its payload is untouched.
    factor : float
        How much the last gap is stretched relative to the first.
        Ramps linearly in between.

    Returns
    -------
    Stream
        A copy of `stream` whose timestamps are rebuilt as a cumulative sum
        of the stretched gaps, from the original first timestamp.

    Raises
    ------
    ValueError
        If `factor <= 0`, which would run the clock backwards —
        a different defect than drift.
    """

    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")

    ordered = stream.timestamps.to_list()
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    n_gaps = len(gaps)

    stretched = ordered[:1]
    running = ordered[0] if ordered else 0.0
    for index, gap in enumerate(gaps):
        ramp = 1.0 if n_gaps <= 1 else 1.0 + (factor - 1.0) * index / (n_gaps - 1)
        running += gap * ramp
        stretched.append(running)

    return stream.model_copy(
        update={"timestamps": pl.Series(stream.timestamps.name, stretched)}
    )


def saturate_channel(stream: Stream, channel: str, *, limit: float = 0.5) -> Stream:
    """Clip one channel into `[-limit, limit]`, the way an actuator hits its rail.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    channel : str
        The channel to clip.
    limit : float
        The clamp applied symmetrically around zero.

    Returns
    -------
    Stream
        A copy of `stream` whose `channel` is clipped to `[-limit, limit]`.
        Every other channel and the timestamps are unchanged.

    Raises
    ------
    ValueError
        If `channel` is not one of the stream's channels.
    """

    _require_channel(stream, channel)
    frame = _channel_frame(stream)

    clipped = [max(-limit, min(limit, value)) for value in frame[channel].to_list()]
    new_frame = frame.with_columns(pl.Series(channel, clipped))

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def drift_channel(stream: Stream, channel: str, *, total: float = 1.0) -> Stream:
    """Add a linear ramp from `0.0` to `total` across one channel's rows.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    channel : str
        The channel to drift.
    total : float
        The ramp's value at the last row.

    Returns
    -------
    Stream
        A copy of `stream` whose `channel` carries a linear ramp added on top
        of its original shape. Every other channel and the timestamps are unchanged.

    Raises
    ------
    ValueError
        If `channel` is not one of the stream's channels.
    """

    _require_channel(stream, channel)
    frame = _channel_frame(stream)

    height = frame.height
    step = 0.0 if height <= 1 else total / (height - 1)
    drifted = [value + step * index for index, value in enumerate(frame[channel])]
    new_frame = frame.with_columns(pl.Series(channel, drifted))

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def spike_channel(
    stream: Stream, channel: str, *, index: int | None = None, magnitude: float = 20.0
) -> Stream:
    """Add a single-sample discontinuity to one channel.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    channel : str
        The channel to spike.
    index : int or None
        The row to spike. Defaults to the channel's midpoint.
    magnitude : float
        How many of the channel's own standard deviations to add.

    Returns
    -------
    Stream
        A copy of `stream` whose `channel` carries `magnitude * std` added
        to one sample. Every other channel and the timestamps are unchanged.

    Raises
    ------
    ValueError
        If `channel` is not one of the stream's channels,
        or `index` falls outside the frame.
    """

    _require_channel(stream, channel)
    frame = _channel_frame(stream)
    spike_index = frame.height // 2 if index is None else index
    _window(frame, spike_index, 1)

    std = cast(float, frame[channel].std())
    values = frame[channel].to_list()
    values[spike_index] = values[spike_index] + magnitude * std
    new_frame = frame.with_columns(pl.Series(channel, values))

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def add_noise(stream: Stream, channel: str, *, amplitude: float = 0.2) -> Stream:
    """Add a deterministic alternating offset to one channel, at the Nyquist frequency.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    channel : str
        The channel to add noise to.
    amplitude : float
        The offset added to even-indexed rows and subtracted from odd-indexed ones.

    Returns
    -------
    Stream
        A copy of `stream` whose `channel` carries `+amplitude, -amplitude` alternating
        across every row. Every other channel and the timestamps are unchanged.

    Raises
    ------
    ValueError
        If `channel` is not one of the stream's channels.
    """

    _require_channel(stream, channel)
    frame = _channel_frame(stream)

    noisy = [
        value + (amplitude if index % 2 == 0 else -amplitude)
        for index, value in enumerate(frame[channel].to_list())
    ]
    new_frame = frame.with_columns(pl.Series(channel, noisy))

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def kill_taxels(stream: Stream, *, count: int = 3) -> Stream:
    """Hold the first `count` channels at their row-0 value for the whole recording.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    count : int
        How many of the stream's channels, in declared order, to kill.

    Returns
    -------
    Stream
        A copy of `stream` whose first `count` channels are constant at their original
        row-0 value. Every other channel and the timestamps are unchanged.
    """

    frame = _channel_frame(stream)
    dead_names = [channel.name for channel in stream.channels[:count]]

    dead_columns = [
        pl.Series(name, [frame[name][0]] * frame.height) for name in dead_names
    ]
    new_frame = frame.with_columns(dead_columns)

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def skew_unloading(stream: Stream, *, factor: float = 1.5) -> Stream:
    """Scale every channel's second half by `factor`, breaking a load-unload retrace.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    factor : float
        The scale applied to every channel's second half.

    Returns
    -------
    Stream
        A copy of `stream` whose every channel is scaled by `factor` across
        its second half. The first half and the timestamps are unchanged.
    """

    frame = _channel_frame(stream)
    midpoint = frame.height // 2

    scaled_columns = [
        pl.Series(
            name,
            [
                value * factor if index >= midpoint else value
                for index, value in enumerate(frame[name].to_list())
            ],
        )
        for name in frame.columns
    ]
    new_frame = frame.with_columns(scaled_columns)

    return stream.model_copy(update={"payload": FramePayload(frame=new_frame)})


def clean_frames(
    *,
    hz: float = 30.0,
    frames: int = 30,
    height: int = 8,
    width: int = 8,
    taxonomy_type: str = "unmapped.camera",
    instance: str | None = None,
) -> Stream:
    """Build a regularly sampled `Kind.IMAGE` Stream of distinct synthetic frames.

    Each frame's fill value is derived from its index, so no two consecutive
    frames are identical and a freeze reads as a real change against them.

    Parameters
    ----------
    hz : float
        The sampling rate to build timestamps at.
    frames : int
        How many frames the stream carries.
    height : int
        Each frame's height in pixels.
    width : int
        Each frame's width in pixels.
    taxonomy_type : str
        The stream's taxonomy type.
    instance : str or None
        Which subject this stream belongs to.

    Returns
    -------
    Stream
        `kind=Kind.IMAGE`, `clock=Clock.CAPTURE`, evenly spaced timestamps,
        and a `SyntheticFrames` payload.

    Raises
    ------
    RuntimeError
        If numpy is not installed.
    """

    numpy = load_numpy()
    if numpy is None:
        raise RuntimeError(
            "clean_frames needs numpy; install it with pip install 'kalanos[video]'"
        )

    timestamps = pl.Series("time_s", [index / hz for index in range(frames)])
    payload = SyntheticFrames(
        frames=[
            numpy.full((height, width, 3), fill_value=index % 256, dtype=numpy.uint8)
            for index in range(frames)
        ]
    )

    return Stream(
        taxonomy_type=taxonomy_type,
        instance=instance,
        kind=Kind.IMAGE,
        timestamps=timestamps,
        payload=payload,
        source_path=UPath("synthetic"),
        clock=Clock.CAPTURE,
        channels=[],
    )


def freeze_frames(stream: Stream, *, start: int = 10, length: int = 10) -> Stream:
    """Replace a window of frames with copies of the frame just before it.

    Parameters
    ----------
    stream : Stream
        The stream to inject into; left unchanged.
    start : int
        The first frame index to freeze.
    length : int
        How many consecutive frames to freeze.

    Returns
    -------
    Stream
        A copy of `stream` whose frames across `[start, start + length)` are
        all copies of the frame at `start - 1` (or at `start`, when `start == 0`).
        Frame count and timestamps are unchanged.

    Raises
    ------
    ValueError
        If the payload is not a `SyntheticFrames`,
        or the window falls outside it.
    """

    if not isinstance(stream.payload, SyntheticFrames):
        raise ValueError(
            f"stream must carry a SyntheticFrames payload; "
            f"got {type(stream.payload).__name__}"
        )

    frames = list(stream.payload.frames)
    end = start + length
    if start < 0 or end > len(frames):
        raise ValueError(
            f"window [{start}, {end}) falls outside a payload of {len(frames)} frames"
        )

    frozen_frame = frames[start if start == 0 else start - 1]
    for index in range(start, end):
        frames[index] = frozen_frame.copy()

    return stream.model_copy(update={"payload": SyntheticFrames(frames=frames)})


def _unhandled_defect(defect: NoReturn) -> NoReturn:
    """Raise for a `Defect` no case above handles.

    Typed on `NoReturn` rather than `Defect` so that a `Defect` member with
    no matching `case` in `apply_defect` fails `pyright`, not just a test:
    the argument at the call site no longer narrows to `Never`.
    """

    raise AssertionError(f"no injector builds a {defect!r} defect")


def _resolved_channel(channel: str | None, defect: Defect) -> str:
    """Assert a channel was resolved, naming the defect that needed one."""

    if channel is None:
        raise ValueError(f"a {defect.value} defect needs a channel; stream has none")
    return channel


def apply_defect(ctx: MetricInput, defect: Defect) -> MetricInput:
    """Inject `defect` into a clean context, returning a context of the same type.

    The one place that knows which injector builds which defect, so
    `check_metric` never has to reason about levels.

    Parameters
    ----------
    ctx : MetricInput
        A clean context to inject the defect into.
    defect : Defect
        The fault to inject.

    Returns
    -------
    MetricInput
        A context of the same type as `ctx`, carrying the injected defect.

    Raises
    ------
    ValueError
        If `ctx` is a `StreamContext` with no channels and `defect` needs one,
        if `ctx` is an `EpisodeContext` (no injector builds an episode-shaped defect;
        pass `defective=` to `check_metric` instead),
        or if the stream is too short for the window the defect needs.
    """

    if isinstance(ctx, ChannelContext):
        stream = ctx.stream.stream
        channel = ctx.channel.name
    elif isinstance(ctx, StreamContext):
        stream = ctx.stream
        channel = stream.channels[0].name if stream.channels else None
    else:
        raise ValueError(
            "no injector builds an episode-shaped defect; pass defective= to "
            "check_metric instead"
        )

    n = len(stream.timestamps)
    if n < 8:
        raise ValueError(f"stream has only {n} samples; apply_defect needs at least 8")
    start = n // 4
    length = max(1, n // 4)
    count = max(1, n // 10)

    match defect:
        case Defect.STUCK_CHANNEL:
            injected = stick_channel(
                stream, _resolved_channel(channel, defect), start=start, length=length
            )
        case Defect.NULLS:
            injected = null_run(
                stream, _resolved_channel(channel, defect), start=start, length=length
            )
        case Defect.SATURATION:
            injected = saturate_channel(stream, _resolved_channel(channel, defect))
        case Defect.DRIFT:
            injected = drift_channel(stream, _resolved_channel(channel, defect))
        case Defect.DROPOUT:
            injected = drop_samples(stream, start=start, count=count)
        case Defect.JITTER:
            injected = jitter_clock(stream)
        case Defect.CLOCK_DRIFT:
            injected = stretch_clock(stream)
        case Defect.FROZEN_FRAMES:
            injected = freeze_frames(stream, start=start, length=length)
        case Defect.SPIKE:
            injected = spike_channel(stream, _resolved_channel(channel, defect))
        case Defect.NOISE:
            injected = add_noise(stream, _resolved_channel(channel, defect))
        case Defect.DEAD_TAXEL:
            injected = kill_taxels(stream)
        case Defect.HYSTERESIS:
            injected = skew_unloading(stream)
        case _:
            _unhandled_defect(defect)

    if isinstance(ctx, ChannelContext):
        assert channel is not None
        new_frame = _channel_frame(injected)
        return ChannelContext(
            channel=ctx.channel,
            values=new_frame[channel],
            stream=StreamContext(stream=injected, is_regular=ctx.stream.is_regular),
        )
    return StreamContext(stream=injected, is_regular=ctx.is_regular)
