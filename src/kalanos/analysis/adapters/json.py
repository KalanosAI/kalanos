"""Nested JSON: a dict keyed by record, sometimes nested once more by instance."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import re
from datetime import datetime, timezone

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.jsonl import _flatten
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.adapters.tabular import ParsedTable, TabularAdapter


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# docs/ADAPTERS.md: readable as a table and nothing contradicts that is 0.3.
_GENERIC_TABLE_CONFIDENCE = 0.3

# Enough to see whether the first line alone is a complete JSON value.
_SNIFF_BYTES = 8192

# A trailing datetime in a record key, e.g. `...+2023-07-07-10h-00m-34s` or
# an ISO-style `...T10:00:34`. The rest of the key is ignored.
_KEY_TIMESTAMP = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[-T](\d{2})(?:h-|:)(\d{2})(?:m-|:)(\d{2})s?$"
)

# `inference/entity.py`'s `_ENTITY_NAME_PATTERN` recognises `device` on the
# name signal alone, which is what lets the second nesting level split into
# instances without depending on repetition to carry the whole score.
_DEVICE_COLUMN = "device_id"

_TIMESTAMP_COLUMN = "timestamp_s"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _timestamp_from_key(key: str) -> float | None:
    """Recover the epoch seconds a record key's trailing datetime carries.

    Returns
    -------
    float or None
        Epoch seconds in UTC, or `None` when the key carries no recognisable trailing
        datetime, or its digits name no real calendar date or time.
    """

    match = _KEY_TIMESTAMP.search(key)
    if match is None:
        return None

    year, month, day, hour, minute, second = (int(group) for group in match.groups())
    try:
        when = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None
    return when.timestamp()


def _records_from_keyed_dict(
    payload: dict[str, object],
) -> tuple[list[dict[str, object]], frozenset[str]]:
    """Turn a dict keyed by record into rows, splitting a nested instance dimension.

    Parameters
    ----------
    payload : dict[str, object]
        A top-level dict whose every value is itself a dict — one record per key.

    Returns
    -------
    tuple[list[dict[str, object]], frozenset[str]]
        One row per record, or one row per `(record key, sub-key)` pair when every value
        under a record is itself a dict — that second level names an instrument,
        carried in a `device_id` column since nothing in the file names it directly —
        alongside the stems any positional array inside a record expanded,
        found the same way `JsonlAdapter` finds them.
    """

    records: list[dict[str, object]] = []
    array_stems: set[str] = set()

    for key, value in payload.items():
        assert isinstance(value, dict)
        timestamp = _timestamp_from_key(key)
        base: dict[str, object] = (
            {_TIMESTAMP_COLUMN: timestamp} if timestamp is not None else {}
        )

        if value and all(isinstance(sub, dict) for sub in value.values()):
            for device_id, inner in value.items():
                record = {**base, _DEVICE_COLUMN: device_id}
                record.update(_flatten(inner, "", array_stems))
                records.append(record)
        else:
            record = dict(base)
            record.update(_flatten(value, "", array_stems))
            records.append(record)

    return records, frozenset(array_stems)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@adapter
class JsonAdapter(TabularAdapter):
    """Reads a JSON document too structured for `JsonlAdapter`'s one-line records."""

    name = "json"

    def detect(self, path: UPath) -> float:
        """Bid when the file opens with `{` or `[` but isn't a JSONL stream.

        A file whose first line does not parse as a complete JSON value is
        pretty-printed, so no line-oriented reader can claim it.
        A file whose first line does parse is either the whole document —
        a single-record file, read here — or the first of several such lines,
        which is `JsonlAdapter`'s territory instead.
        """

        try:
            with path.open("rb") as handle:
                chunk = handle.read(_SNIFF_BYTES)
            text = chunk.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return 0.0

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or lines[0][0] not in "{[":
            return 0.0

        try:
            json.loads(lines[0])
        except json.JSONDecodeError:
            return _GENERIC_TABLE_CONFIDENCE
        return _GENERIC_TABLE_CONFIDENCE if len(lines) == 1 else 0.0

    def parse(self, path: UPath) -> ParsedTable:
        """Normalise the document to a long frame, by whichever shape it holds.

        - A top-level dict whose values are dicts: one row per record,
          or one row per `(record key, sub-key)` pair when there is a second
          instance dimension.
        - A top-level dict of scalars: one row, one column per key.
          No time column results, so `TabularAdapter` refuses with
          `RefusalCode.NO_TIME_INDEX` — nothing adapter-specific is needed
          for that case.
        - A top-level list of dicts: one row per element, flattened the same
          way `JsonlAdapter` flattens a JSONL record.
        """

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if (
            isinstance(payload, dict)
            and payload
            and all(isinstance(value, dict) for value in payload.values())
        ):
            records, array_stems = _records_from_keyed_dict(payload)
            frame = pl.from_dicts(records, infer_schema_length=None)
            return ParsedTable(frame=frame, array_stems=array_stems)

        if isinstance(payload, list):
            list_array_stems: set[str] = set()
            records = [dict(_flatten(item, "", list_array_stems)) for item in payload]
            frame = pl.from_dicts(records, infer_schema_length=None)
            return ParsedTable(frame=frame, array_stems=frozenset(list_array_stems))

        return ParsedTable(frame=pl.DataFrame([payload]))
