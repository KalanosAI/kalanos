"""The two time grids a fixture can be walked on."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import math
import random
from collections.abc import Iterator
from typing import Annotated, Literal

# External
from pydantic import BaseModel, Field


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class RegularGrid(BaseModel):
    """A fixed-step time grid, walked without needing the draw stream."""

    kind: Literal["regular"] = "regular"
    start: float
    step: float
    count: int
    is_int: bool = True
    digits: int = 0

    @property
    def period(self) -> float:
        """One full lap of the grid, in the grid's time units."""

        return self.step * self.count

    def times(self, rng: random.Random) -> Iterator[tuple[int, int | float]]:
        """Yield `(grid_index, nominal_time)` pairs.

        Parameters
        ----------
        rng : random.Random
            Unused; accepted so `TimeGrid` has one call shape.

        Yields
        ------
        tuple of (int, int or float)
            The grid index and the nominal (pre-jitter) time at it.
        """

        logger.debug(
            "Walking regular grid: count=%d, start=%s, step=%s",
            self.count,
            self.start,
            self.step,
        )
        for i in range(self.count):
            t = self.start + i * self.step
            yield i, (int(t) if self.is_int else round(t, self.digits))

    def phase_at(self, t: int | float, phase0: float) -> float:
        """Compute the phase angle at a nominal time.

        Parameters
        ----------
        t : int or float
            A nominal time from `times`.
        phase0 : float
            The entity's phase offset.

        Returns
        -------
        float
            `phase0 + 2*pi * (t - start) / period`, one full lap per period.
        """

        return phase0 + 2 * math.pi * (t - self.start) / self.period


class IrregularGrid(BaseModel):
    """A grid whose gaps are themselves drawn from the fixture's RNG.

    Entities on an irregular grid are expected to carry only `UniformNoise` channels
    — nothing here defines a meaningful phase.
    """

    kind: Literal["irregular"] = "irregular"
    start: int
    min_gap: int
    max_gap: int
    count: int

    def times(self, rng: random.Random) -> Iterator[tuple[int, int]]:
        """Yield `(grid_index, time)` pairs, drawing each gap as it goes.

        Parameters
        ----------
        rng : random.Random
            Shared draw stream for the fixture being built.
            Each gap is drawn immediately after its record's time is yielded,
            so that a caller drawing channel values in between sees the
            same draw order as the original hand-written generator.

        Yields
        ------
        tuple of (int, int)
            The record index and its time.
        """

        logger.debug(
            "Walking irregular grid: count=%d, start=%d, gap=[%d, %d]",
            self.count,
            self.start,
            self.min_gap,
            self.max_gap,
        )
        t = self.start
        for i in range(self.count):
            yield i, t
            t += rng.randint(self.min_gap, self.max_gap)

    def phase_at(self, t: int | float, phase0: float) -> float:
        """Return a placeholder phase.

        Parameters
        ----------
        t : int or float
            Unused.
        phase0 : float
            Unused.

        Returns
        -------
        float
            Always `0.0` — no entity on an irregular grid reads this.
        """

        return 0.0


TimeGrid = Annotated[RegularGrid | IrregularGrid, Field(discriminator="kind")]
