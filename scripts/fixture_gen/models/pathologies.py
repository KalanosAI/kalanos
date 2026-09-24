"""The three signal pathologies a fixture entity can carry."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
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


class Flatline(BaseModel):
    """One named channel is pinned to a constant value for every sample."""

    kind: Literal["flatline"] = "flatline"
    channel: str
    value: float


class DropBurst(BaseModel):
    """A contiguous run of grid indices is dropped for this entity."""

    kind: Literal["drop_burst"] = "drop_burst"
    start_index: int
    count: int

    def covers(self, index: int) -> bool:
        """Check whether a grid index falls inside the dropped run.

        Parameters
        ----------
        index : int
            A grid index, shared across every entity on the grid.

        Returns
        -------
        bool
            Whether `index` should be skipped.
        """

        dropped = self.start_index <= index < self.start_index + self.count
        if dropped:
            logger.debug(
                "Drop-burst covers grid index %d (start=%d, count=%d)",
                index,
                self.start_index,
                self.count,
            )
        return dropped


class ClockJitter(BaseModel):
    """The emitted timestamp wobbles by up to `max_offset` around nominal."""

    kind: Literal["clock_jitter"] = "clock_jitter"
    max_offset: int


Pathology = Annotated[Flatline | DropBurst | ClockJitter, Field(discriminator="kind")]
