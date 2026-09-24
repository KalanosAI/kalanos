"""Detect a CSV's delimiter and header, rather than assume one."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import logging

# External
from upath import UPath

# Internal
from kalanos.analysis.models.schema import CsvDialect


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_SAMPLE_BYTES = 8192
_CANDIDATE_DELIMITERS = ",;\t|"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def sniff_dialect(path: UPath) -> CsvDialect:
    """Detect `path`'s delimiter and whether its first row is a header.

    Parameters
    ----------
    path : UPath
        CSV file to sniff.

    Returns
    -------
    CsvDialect
        The detected delimiter and header presence, with evidence and a confidence
        - 0.9 when `csv.Sniffer` resolves the sample cleanly
        - 0.3 when it fails and a comma-delimited, headered fallback is used

    Raises
    ------
    OSError
        If `path` cannot be opened or read.
    """

    # Step 1: read a sample of the file to sniff against.
    with path.open(newline="") as handle:
        sample = handle.read(_SAMPLE_BYTES)

    # Step 2: let csv.Sniffer resolve delimiter and header from the sample.
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=_CANDIDATE_DELIMITERS)
        has_header = csv.Sniffer().has_header(sample)
        logger.debug("%s: dialect %r, header=%s", path, dialect.delimiter, has_header)
        return CsvDialect(
            delimiter=dialect.delimiter,
            has_header=has_header,
            confidence=0.9,
            evidence=[
                f"csv.Sniffer detected delimiter {dialect.delimiter!r} "
                f"from a {len(sample)}-byte sample",
                f"csv.Sniffer.has_header() returned {has_header}",
            ],
        )
    # Step 3: fall back to a comma-delimited, headered guess if sniffing fails.
    except csv.Error as exc:
        logger.debug("%s: csv.Sniffer failed (%s); falling back to ','", path, exc)
        return CsvDialect(
            delimiter=",",
            has_header=True,
            confidence=0.3,
            evidence=[
                f"csv.Sniffer could not resolve a dialect ({exc}); fell back to ','"
            ],
        )
