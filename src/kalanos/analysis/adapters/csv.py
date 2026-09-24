"""CSV is the generic table format — it sniffs its dialect and reads,
and everything after that is `TabularAdapter`'s.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.registry import adapter
from kalanos.analysis.adapters.tabular import ParsedTable, TabularAdapter
from kalanos.analysis.inference.dialect import sniff_dialect


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# A file being readable as a table says very little — a LeRobot parquet directory
# and a ROS bag are tables too — so the generic reader must never out-rank a
# specialised one. docs/ADAPTERS.md sets 0.3 as that band.
_GENERIC_TABLE_CONFIDENCE = 0.3

_SUFFIX = ".csv"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _looks_delimited(path: UPath) -> bool:
    """Sniff whether a file's first line is delimited like CSV.

    Reads only the first line:
    - Rejects an empty first line
    - Line opening with `{` or `[`
    - Line with no comma-separated fields

    Parameters
    ----------
    path : UPath
        File to sniff.

    Returns
    -------
    bool
        Whether the first line parses as more than one comma-delimited field,
        or `False` if the file could not be opened, decoded, or parsed —
        `detect` must not raise.
    """

    try:
        with path.open(newline="") as handle:
            first_line = handle.readline()

        stripped = first_line.strip()
        if not stripped or stripped[0] in "{[":
            return False

        fields = next(csv.reader([first_line]))
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    return len(fields) > 1


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@adapter
class CsvAdapter(TabularAdapter):
    """Reads any `.csv` file, sniffing its delimiter and header."""

    name = "csv"

    def detect(self, path: UPath) -> float:
        """Bid on the extension and a first-line content sniff."""

        if path.suffix.lower() != _SUFFIX:
            return 0.0
        return _GENERIC_TABLE_CONFIDENCE if _looks_delimited(path) else 0.0

    def parse(self, path: UPath) -> ParsedTable:
        """Sniff the dialect and read the file with it."""

        dialect = sniff_dialect(path)
        with path.open("rb") as handle:
            frame = pl.read_csv(
                source=handle,
                separator=dialect.delimiter,
                has_header=dialect.has_header,
            )
        return ParsedTable(frame=frame)
