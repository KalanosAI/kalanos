"""The two kinds of fixture spec, and the sample type they're built from."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from typing import Annotated, Literal, NamedTuple

# External
from pydantic import BaseModel, Field

# Internal
from fixture_gen.models.entities import EntitySpec
from fixture_gen.models.grids import TimeGrid
from fixture_gen.models.outputs import StaticJsonOutput, TimeSeriesOutput


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# Just any number
SEED = 42


# A sample's non-time fields: one value per flat channel, or one ordered
# list of values per named group.
Fields = dict[str, float | str | list[float | str]]


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class TimeSeriesSpec(BaseModel):
    """A fixture built by walking a time grid and sampling entities on it."""

    kind: Literal["timeseries"] = "timeseries"
    filename: str
    seed: int = SEED
    grid: TimeGrid
    entities: list[EntitySpec]
    output: TimeSeriesOutput


class StaticSpec(BaseModel):
    """A fixture with no time axis: one flat JSON payload, written as-is."""

    kind: Literal["static"] = "static"
    filename: str
    payload: dict[str, float | int | str]
    output: StaticJsonOutput = Field(default_factory=StaticJsonOutput)


class Sample(NamedTuple):
    """One entity's record at one grid index, values already rounded."""

    t: int | float
    entity_id: str | None
    fields: Fields


FlatFixtureSpec = Annotated[TimeSeriesSpec | StaticSpec, Field(discriminator="kind")]
