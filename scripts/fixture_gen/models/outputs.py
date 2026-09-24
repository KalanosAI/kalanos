"""The container formats a fixture can be written as."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from typing import Annotated, Literal

# External
from pydantic import BaseModel, Field


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class CsvOutput(BaseModel):
    """One row per (time, entity), sorted by `(time_field, entity_field)`."""

    kind: Literal["csv"] = "csv"
    time_field: str
    entity_field: str = "id"


class JsonLinesOutput(BaseModel):
    """One JSON object per line, keyed by `time_field` plus each channel."""

    kind: Literal["jsonlines"] = "jsonlines"
    time_field: str


class DelimitedTextOutput(BaseModel):
    """Whitespace-separated columns, with the header in a comment line."""

    kind: Literal["delimited_text"] = "delimited_text"
    comment_header: str


class KeyedJsonOutput(BaseModel):
    """A dict keyed by a per-record template, nested one level by entity."""

    kind: Literal["keyed_json"] = "keyed_json"
    key_template: str
    indent: int = 1


class StaticJsonOutput(BaseModel):
    """A single flat JSON object with no time axis."""

    kind: Literal["static_json"] = "static_json"
    indent: int = 1


TimeSeriesOutput = Annotated[
    CsvOutput | JsonLinesOutput | DelimitedTextOutput | KeyedJsonOutput,
    Field(discriminator="kind"),
]
