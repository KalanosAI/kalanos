"""How one channel's raw value is computed and rounded before it is written."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import random
from typing import Annotated, Literal

# External
from pydantic import BaseModel, Field


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Rounding(BaseModel):
    """How a computed channel value is rounded before it is written.

    `as_string=True` formats the value with trailing zeros (`75.000`);
    The arm CSV fixture depends on the string form to keep its flatlined channel
    looking like every other formatted float in the file.
    """

    digits: int
    as_string: bool = False

    def format(self, value: float) -> float | str:
        """Round or format a raw computed value.

        Parameters
        ----------
        value : float
            The unrounded value.

        Returns
        -------
        float or str
            `round(value, digits)`, or an f-string of that precision if
            `as_string` is set.
        """

        if self.as_string:
            return f"{value:.{self.digits}f}"
        return round(value, self.digits)


class Sinusoid(BaseModel):
    """`offset + amplitude * wave(harmonic * theta)`.

    Every channel in the arm, IMU and pose fixtures fits this form.
    """

    kind: Literal["sinusoid"] = "sinusoid"
    offset: float = 0.0
    amplitude: float
    wave: Literal["sin", "cos"] = "sin"
    harmonic: int = 1

    def evaluate(self, theta: float, rng: random.Random) -> float:
        """Evaluate the sinusoid at a phase angle.

        Parameters
        ----------
        theta : float
            Phase angle in radians, before the harmonic multiplier.
        rng : random.Random
            Unused; accepted so `ChannelValue` has one call shape.

        Returns
        -------
        float
            The unrounded channel value.
        """

        f = math.sin if self.wave == "sin" else math.cos
        return self.offset + self.amplitude * f(self.harmonic * theta)


class UniformNoise(BaseModel):
    """A value drawn uniformly at random from `[low, high]`."""

    kind: Literal["uniform"] = "uniform"
    low: float
    high: float

    def evaluate(self, theta: float, rng: random.Random) -> float:
        """Draw a uniform random value.

        Parameters
        ----------
        theta : float
            Unused; accepted so `ChannelValue` has one call shape.
        rng : random.Random
            Shared draw stream for the fixture being built.

        Returns
        -------
        float
            `rng.uniform(low, high)`.
        """

        return rng.uniform(self.low, self.high)


ChannelValue = Annotated[Sinusoid | UniformNoise, Field(discriminator="kind")]
