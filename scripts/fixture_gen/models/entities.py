"""Channels, groups of channels, and the entities that carry them."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import random

# External
from pydantic import BaseModel, Field

# Internal
from fixture_gen.models.pathologies import ClockJitter, DropBurst, Flatline, Pathology
from fixture_gen.models.values import ChannelValue, Rounding


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class ChannelSpec(BaseModel):
    """One numeric series: how to compute it, and how to round it."""

    name: str
    value: ChannelValue
    rounding: Rounding

    def sample(self, theta: float, rng: random.Random) -> float | str:
        """Compute and round one value for this channel.

        Parameters
        ----------
        theta : float
            Phase angle, for `Sinusoid` channels.
        rng : random.Random
            Shared draw stream, for `UniformNoise` channels.

        Returns
        -------
        float or str
            The rounded value, ready to be written.
        """

        return self.rounding.format(self.value.evaluate(theta, rng))


class GroupSpec(BaseModel):
    """A set of channels that belong together.

    `name=None` means the channels are flat fields on the entity
    (the arm CSV's `tcp_pose_x_mm`/`tcp_pose_y_mm`/`tcp_pose_z_mm`);
    a `name` means they are written together as one ordered array (IMU's `acc`/`gyro`).
    """

    name: str | None = None
    channels: list[ChannelSpec]


class EntitySpec(BaseModel):
    """One subject on the grid: an `id` column value, or the whole file."""

    id: str | None = None
    phase0: float = 0.0
    groups: list[GroupSpec]
    pathologies: list[Pathology] = Field(default_factory=list)

    def flatline_for(self, channel: str) -> Flatline | None:
        """Find this entity's flatline pathology for a channel, if any.

        Parameters
        ----------
        channel : str
            The channel name to look up.

        Returns
        -------
        Flatline or None
            The matching pathology, or `None`.
        """

        return next(
            (
                p
                for p in self.pathologies
                if isinstance(p, Flatline) and p.channel == channel
            ),
            None,
        )

    def drop_burst(self) -> DropBurst | None:
        """Find this entity's drop-burst pathology, if any.

        Returns
        -------
        DropBurst or None
            The pathology, or `None`.
        """

        return next((p for p in self.pathologies if isinstance(p, DropBurst)), None)

    def clock_jitter(self) -> ClockJitter | None:
        """Find this entity's clock-jitter pathology, if any.

        Returns
        -------
        ClockJitter or None
            The pathology, or `None`.
        """

        return next((p for p in self.pathologies if isinstance(p, ClockJitter)), None)
