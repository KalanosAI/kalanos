"""Source sample locations, without manufacturing event times or repair windows."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import polars as pl

# Internal
from kalanos.analysis.models.metrics import ChannelContext
from kalanos.analysis.models.support import SampleInterval, SupportKind, TemporalSupport


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def source_values(
    ctx: ChannelContext,
) -> tuple[pl.Series, list[int], list[float]] | None:
    stream = ctx.stream.stream
    order = stream.source_order
    if not order.preserved and order.original_index is None:
        return None
    indices = order.original_index or list(range(ctx.n_samples))
    positions = list(range(ctx.n_samples))
    if not order.preserved:
        positions.sort(key=lambda i: indices[i])
    return (
        pl.Series([ctx.values[i] for i in positions], dtype=ctx.values.dtype),
        [indices[i] for i in positions],
        [stream.timestamps[i] for i in positions],
    )


def support_for(
    indices: list[int], ranges: list[tuple[int, int]], *, window: int = 0
) -> TemporalSupport:
    """Map measured positions to source rows; split gaps instead of filling them."""
    intervals = []
    for start, end in ranges:
        positions = indices[start:end]
        if not positions:
            continue
        lo = previous = positions[0]
        for current in positions[1:] + [None]:
            # only the trailing sentinel sets these to None, and it ends the loop
            assert lo is not None
            assert previous is not None
            if current is not None and current == previous + 1:
                previous = current
                continue
            intervals.append(
                SampleInterval(
                    start=lo,
                    end_exclusive=previous + 1,
                    support_start=indices[max(0, start - window)] if window else None,
                    support_end_exclusive=indices[
                        min(len(indices) - 1, end - 1 + window)
                    ]
                    + 1
                    if window
                    else None,
                )
            )
            lo = previous = current
    return (
        TemporalSupport(
            kind=SupportKind.INTERVALS, index_space="source_rows", intervals=intervals
        )
        if intervals
        else TemporalSupport()
    )


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)
