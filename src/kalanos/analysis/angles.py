"""Undo the wraps of an angle channel, so crossing its bound reads as motion."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import polars as pl

# Internal
from kalanos.analysis.models.binding import ChannelBinding, Representation


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The period an angle wraps by, in each unit a binding may declare for it.
ANGLE_PERIODS = {"rad": 2 * math.pi, "deg": 360.0, "mrad": 2000 * math.pi}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def angle_period(binding: ChannelBinding | None) -> float | str | None:
    """Find the period a channel wraps by.

    Parameters
    ----------
    binding : ChannelBinding or None
        The channel's binding, if it has one.

    Returns
    -------
    float or str or None
        `None` for a channel not bound as `angle`, the period in the channel's unit
        for one that is, or the reason its period is unknown when its unit is
        missing or has no known period.
    """

    if binding is None or binding.representation != Representation.ANGLE:
        return None
    if binding.unit is None:
        return "angle channel declares no unit, so its wrap period is unknown"
    period = ANGLE_PERIODS.get(binding.unit)
    if period is None:
        return (
            f"angle channel unit {binding.unit!r} has no known wrap period; "
            "use rad, deg or mrad"
        )
    return period


def unwrap(values: pl.Series, period: float) -> tuple[pl.Series, int]:
    """Undo an angle's wraps, so a crossing of its bound reads as continuous motion.

    A step of more than half a period between consecutive finite samples is read as
    a wrap, the same rule as `np.unwrap`. Null and non-finite samples are skipped and
    kept as they are, so a wrap that falls next to one is still compared across it.

    Parameters
    ----------
    values : pl.Series
        Numeric samples in the order they were recorded.
    period : float
        The span the angle wraps by, in the channel's unit.

    Returns
    -------
    tuple[pl.Series, int]
        The unwrapped samples as Float64, the same length with the same samples
        skipped, and the number of steps corrected.
    """

    values = values.cast(pl.Float64)
    valid = values.is_finite().fill_null(False)
    finite = values.filter(valid)
    turns = (finite.diff() / period).round()
    unwrapped = finite + (-period * turns).fill_null(0.0).cum_sum()
    return values.clone().scatter(valid.arg_true(), unwrapped), int((turns != 0).sum())


def angle_evidence(period: float | None, n_unwrapped: int) -> dict[str, float | int]:
    """Name the period an angle channel was unwrapped by and the steps corrected.

    Parameters
    ----------
    period : float or None
        The channel's wrap period, or `None` for a channel that is not an angle.
    n_unwrapped : int
        The number of steps `unwrap` corrected.

    Returns
    -------
    dict[str, float | int]
        `angle_period` and `n_unwrapped_steps`, or nothing for a channel that is not
        an angle, so its evidence gains no key.
    """

    if period is None:
        return {}
    return {"angle_period": period, "n_unwrapped_steps": n_unwrapped}
