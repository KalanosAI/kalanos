"""JSONL: one JSON object per line, detected by content rather than extension."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from collections.abc import Iterator

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.adapters.tabular import ParsedTable, TabularAdapter
from kalanos.analysis.models.dictionary import normalise_name


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# docs/ADAPTERS.md: readable as a table and nothing contradicts that is 0.3.
_GENERIC_TABLE_CONFIDENCE = 0.3

# Enough to see two lines of a real JSONL file without reading the whole thing.
_SNIFF_BYTES = 8192

_SCALAR_TYPES = (int, float, str, bool, type(None))


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _flatten(
    value: object, prefix: str, array_stems: set[str]
) -> Iterator[tuple[str, object]]:
    """Reduce one JSON value to flat `(column, value)` pairs, recording array stems.

    Parameters
    ----------
    value : object
        The JSON value to flatten, as `json.loads` produced it.
    prefix : str
        The column name built so far from the enclosing keys.
    array_stems : set[str]
        Mutated in place: every stem whose member columns came from a
        positional array is added here as it is found.

    Yields
    ------
    tuple[str, object]
        One `(column, value)` pair per scalar the value bottoms out to.
        A `dict` recurses with its keys joined onto `prefix` by `.`;
        a `list` of scalars expands to `prefix_0 … prefix_n` and adds
        `prefix`'s normalised stem to `array_stems` — the same stem
        `TabularAdapter` groups the expanded columns under, since a dotted
        prefix does not survive into a column's own normalised stem;
        anything else — a nested list or a list of dicts — becomes one
        column holding its JSON text, so the sensor it names survives
        rather than vanishing silently.
    """

    if isinstance(value, dict):
        for key, sub_value in value.items():
            sub_prefix = f"{prefix}.{key}" if prefix else key
            yield from _flatten(sub_value, sub_prefix, array_stems)
    elif isinstance(value, list) and all(
        isinstance(item, _SCALAR_TYPES) for item in value
    ):
        array_stems.add(normalise_name(prefix).stem or prefix)
        for index, item in enumerate(value):
            yield f"{prefix}_{index}", item
    elif isinstance(value, _SCALAR_TYPES):
        yield prefix, value
    else:
        yield prefix, json.dumps(value)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@adapter
class JsonlAdapter(TabularAdapter):
    """Reads line-delimited JSON, one record per line, detected by content."""

    name = "jsonl"

    def detect(self, path: UPath) -> float:
        """Bid when the first two non-blank lines each parse as a JSON object.

        A one-record file is left to `JsonAdapter` — requiring two object lines is
        also what keeps this adapter off a pretty-printed JSON file such as
        `capture_index.json`, whose first line is a bare `{`.
        """

        try:
            with path.open("rb") as handle:
                chunk = handle.read(_SNIFF_BYTES)
            text = chunk.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return 0.0

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return 0.0

        try:
            first, second = json.loads(lines[0]), json.loads(lines[1])
        except json.JSONDecodeError:
            return 0.0

        if isinstance(first, dict) and isinstance(second, dict):
            return _GENERIC_TABLE_CONFIDENCE
        return 0.0

    def parse(self, path: UPath) -> ParsedTable:
        """Flatten every line's record and stack them into one frame.

        A key missing from one record but present in another becomes null
        rather than an error — `infer_schema_length=None` scans every
        record's keys before building the frame.
        """

        array_stems: set[str] = set()
        records: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = dict(_flatten(json.loads(line), "", array_stems))
                records.append(record)

        frame = pl.from_dicts(records, infer_schema_length=None)
        return ParsedTable(frame=frame, array_stems=frozenset(array_stems))
