"""Whitespace-delimited text, with column names carried in a comment line.

The header, when there is one, sits in a `#`-prefixed comment line above the
data rather than in a data row of its own — there is no delimiter to sniff.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.adapters.tabular import ParsedTable, TabularAdapter


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# docs/ADAPTERS.md: readable as a table and nothing contradicts that is 0.3.
_GENERIC_TABLE_CONFIDENCE = 0.3

# Enough to see the header comment and the first data line without reading
# the whole file.
_SNIFF_BYTES = 8192


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _as_float(value: str) -> float | None:
    """Parse `value` as a float, or return `None` when it does not parse."""

    try:
        return float(value)
    except ValueError:
        return None


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@adapter
class DelimitedTextAdapter(TabularAdapter):
    """Reads whitespace-delimited text, with an optional `#`-commented header."""

    name = "delimited"

    def detect(self, path: UPath) -> float:
        """Bid when the first two data lines are same-width, part-numeric fields."""

        if path.suffix.lower() == ".csv":
            return 0.0

        try:
            with path.open("rb") as handle:
                chunk = handle.read(_SNIFF_BYTES)
            text = chunk.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return 0.0

        data_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped[0] in "{[":
                return 0.0
            data_lines.append(stripped)
            if len(data_lines) == 2:
                break

        if not data_lines:
            return 0.0

        first_fields = data_lines[0].split()
        if len(first_fields) <= 1:
            return 0.0
        if not any(_as_float(field) is not None for field in first_fields):
            return 0.0
        if len(data_lines) > 1 and len(data_lines[1].split()) != len(first_fields):
            return 0.0
        return _GENERIC_TABLE_CONFIDENCE

    def parse(self, path: UPath) -> ParsedTable:
        """Read every data line, naming columns from the last header comment seen.

        The header comment only counts when it appears before the first data line,
        and only when its token count agrees with that line's field count —
        otherwise the columns fall back to `column_0 … column_n`.
        """

        header_tokens: list[str] | None = None
        data_lines: list[str] = []
        seen_data = False

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    if not seen_data:
                        header_tokens = stripped.lstrip("#").split()
                    continue
                data_lines.append(stripped)
                seen_data = True

        rows = [line.split() for line in data_lines]
        field_count = len(rows[0]) if rows else 0
        if header_tokens is not None and len(header_tokens) == field_count:
            column_names = header_tokens
        else:
            column_names = [f"column_{index}" for index in range(field_count)]

        # A row shorter than the first — a log truncated mid-write, say —
        # contributes null for the fields it never reached rather than raising,
        # so one bad line doesn't refuse the whole file.
        data: dict[str, list[float | None] | list[str | None]] = {}
        for index, name in enumerate(column_names):
            raw_values = [row[index] if index < len(row) else None for row in rows]
            present = [value for value in raw_values if value is not None]
            if present and all(_as_float(value) is not None for value in present):
                data[name] = [
                    _as_float(value) if value is not None else None
                    for value in raw_values
                ]
            else:
                data[name] = raw_values

        return ParsedTable(frame=pl.DataFrame(data))
