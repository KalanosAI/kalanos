"""Pydantic schema for declarative fixture specs.

`fixture_gen.corpus` builds `FlatFixtureSpec` instances from these models;
`fixture_gen.sampling` and `fixture_gen.writers` turn a spec into samples
and then into bytes on disk. Nothing here does I/O or holds a fixture's
actual constants.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from fixture_gen.models.entities import ChannelSpec, EntitySpec, GroupSpec
from fixture_gen.models.grids import IrregularGrid, RegularGrid, TimeGrid
from fixture_gen.models.outputs import (
    CsvOutput,
    DelimitedTextOutput,
    JsonLinesOutput,
    KeyedJsonOutput,
    StaticJsonOutput,
    TimeSeriesOutput,
)
from fixture_gen.models.pathologies import ClockJitter, DropBurst, Flatline, Pathology
from fixture_gen.models.specs import (
    SEED,
    Fields,
    FlatFixtureSpec,
    Sample,
    StaticSpec,
    TimeSeriesSpec,
)
from fixture_gen.models.values import ChannelValue, Rounding, Sinusoid, UniformNoise


__all__ = [
    "SEED",
    "ChannelSpec",
    "ChannelValue",
    "ClockJitter",
    "CsvOutput",
    "DelimitedTextOutput",
    "DropBurst",
    "EntitySpec",
    "Fields",
    "FlatFixtureSpec",
    "Flatline",
    "GroupSpec",
    "IrregularGrid",
    "JsonLinesOutput",
    "KeyedJsonOutput",
    "Pathology",
    "RegularGrid",
    "Rounding",
    "Sample",
    "Sinusoid",
    "StaticJsonOutput",
    "StaticSpec",
    "TimeGrid",
    "TimeSeriesOutput",
    "TimeSeriesSpec",
    "UniformNoise",
]
