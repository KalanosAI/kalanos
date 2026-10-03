"""Clock evidence and adjacent source-row gaps, shared by timing metrics."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from dataclasses import dataclass

# Internal
from kalanos.analysis.models.domain import ClockInfo, Stream


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_SECONDS_PER_UNIT: dict[str | None, float] = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class ClockSamples:
    """Finite adjacent pairs, retaining their source-row addresses."""

    gaps: list[float]
    end_rows: list[int]
    n_samples: int
    invalid_rows: list[int]
    reason: str | None = None
    seconds: bool = True


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def samples(stream: Stream) -> ClockSamples:
    """Subtract native ticks before converting, never bridge an invalid row.

    A transformed view is restored through its explicit source row map.
    Without that map, there is no defensible source-order measurement.
    Integer subtraction happens in Python,
    avoiding both epoch float rounding and unsigned underflow.
    """

    info = stream.clock_info or ClockInfo.from_legacy(stream.clock)
    indices = stream.source_order.original_index
    n = len(stream.timestamps)
    if not stream.source_order.preserved and indices is None:
        return ClockSamples([], [], n, [], "source row order is unavailable")
    indices = list(range(n)) if indices is None else indices
    factor = info.tick_period_s or _SECONDS_PER_UNIT.get(info.native_unit)
    native = stream.native_timestamps
    values = native.to_list() if native is not None else stream.timestamps.to_list()
    scale = factor if native is not None and factor is not None else 1.0
    rows = list(zip(indices, values, strict=True))
    if not stream.source_order.preserved:
        rows.sort(key=lambda row: row[0])

    def valid(value: float | None) -> bool:
        return value is not None and math.isfinite(value)

    invalid = [i for i, value in rows if not valid(value)]
    gaps, end_rows = [], []
    for (_, left), (right_index, right) in zip(rows, rows[1:], strict=False):
        if valid(left) and valid(right):
            gap = (right - left) * scale
            if not math.isfinite(gap):
                invalid.append(right_index)
                continue
            gaps.append(gap)
            end_rows.append(right_index)
    return ClockSamples(
        gaps,
        end_rows,
        n,
        sorted(set(invalid)),
        seconds=native is None or factor is not None,
    )


def clock_evidence(stream: Stream, data: ClockSamples) -> dict:
    """Small report evidence; native arrays stay in the stream, not JSON."""

    info = stream.clock_info or ClockInfo.from_legacy(stream.clock)
    return {
        "clock_origin": info.origin.value,
        "clock_origin_evidence": info.origin_evidence.value,
        "acquisition_observable": info.certifies_acquisition,
        "source_order_preserved": stream.source_order.preserved,
        "n_samples": data.n_samples,
        "n_gaps": len(data.gaps),
        "n_invalid_timestamps": len(data.invalid_rows),
        "first_invalid_sample": data.invalid_rows[0] if data.invalid_rows else None,
    }
