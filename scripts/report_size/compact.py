"""Measure a report's compact encoding: size, encode time, load time and peak memory."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# External
from pydantic_core import to_json

# Internal
from kalanos.analysis.models.compact_keys import encode_keys

from . import config


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

COLUMNS = (
    "Dataset",
    "Full MB",
    "Compact MB",
    "Saved",
    "Encode s",
    "Load full s",
    "Load compact s",
    "Peak full MB",
    "Peak compact MB",
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class CompactMeasurement:
    """One report measured in its full and its compact encoding.

    Attributes
    ----------
    full_bytes : int
        The full report file's size.
    compact_bytes : int
        The compact file's size.
    encode_seconds : float
        The time to parse the full file, encode its keys and serialize the result.
    load_full_seconds : float
        The time `load_any` took on the full file.
    load_compact_seconds : float
        The time `load_any` took on the compact file.
    peak_full_bytes : int
        The peak resident memory of the process that loaded the full file.
    peak_compact_bytes : int
        The peak resident memory of the process that loaded the compact file.
    """

    full_bytes: int
    compact_bytes: int
    encode_seconds: float
    load_full_seconds: float
    load_compact_seconds: float
    peak_full_bytes: int
    peak_compact_bytes: int

    @property
    def saving(self) -> float:
        """Return the share of the full size the compact encoding saves.

        Returns
        -------
        float
            `1 - compact_bytes / full_bytes`.
        """

        return 1 - self.compact_bytes / self.full_bytes


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _probe(report: Path) -> tuple[float, int]:
    """Load a report in a fresh process and return its load time and peak memory.

    Parameters
    ----------
    report : Path
        A report JSON, full or compact.

    Returns
    -------
    tuple[float, int]
        The seconds `load_any` took, and the process's peak resident bytes.
    """

    result = subprocess.run(
        [sys.executable, "-m", "scripts.report_size.probe", str(report.resolve())],
        cwd=config.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    measured = json.loads(result.stdout)
    return measured["seconds"], measured["peak_rss_bytes"]


def measure_compact(report: Path) -> CompactMeasurement:
    """Write a report's compact encoding beside it and measure both files.

    Parameters
    ----------
    report : Path
        A full report JSON. The compact file is written as `<stem>.compact.json`.

    Returns
    -------
    CompactMeasurement
        Both files' sizes, the encode time, and each file's load time and peak memory.
    """

    compact = report.with_suffix(".compact.json")
    start = time.perf_counter()
    encoded = to_json(encode_keys(json.loads(report.read_bytes())))
    encode_seconds = time.perf_counter() - start
    compact.write_bytes(encoded)

    load_full, peak_full = _probe(report)
    load_compact, peak_compact = _probe(compact)
    return CompactMeasurement(
        full_bytes=report.stat().st_size,
        compact_bytes=compact.stat().st_size,
        encode_seconds=encode_seconds,
        load_full_seconds=load_full,
        load_compact_seconds=load_compact,
        peak_full_bytes=peak_full,
        peak_compact_bytes=peak_compact,
    )


def to_markdown(rows: list[tuple[str, CompactMeasurement]]) -> str:
    """Render compact measurements as a Markdown table.

    Parameters
    ----------
    rows : list[tuple[str, CompactMeasurement]]
        Each dataset with its measurement.

    Returns
    -------
    str
        A heading line, then one table row per dataset.
    """

    def line(cells: list[str] | tuple[str, ...]) -> str:
        """Join cells into one Markdown table row."""

        return "| " + " | ".join(cells) + " |"

    def mb(size: int) -> str:
        """Format a byte count in megabytes."""

        return f"{size / 1e6:.2f}"

    lines = [
        "Compact encoding of each --after report, by this checkout's codec:",
        "",
        line(COLUMNS),
        "|" + "|".join(["---"] * len(COLUMNS)) + "|",
    ]
    for dataset, m in rows:
        cells = [
            dataset,
            mb(m.full_bytes),
            mb(m.compact_bytes),
            f"{m.saving:.1%}",
            f"{m.encode_seconds:.2f}",
            f"{m.load_full_seconds:.2f}",
            f"{m.load_compact_seconds:.2f}",
            mb(m.peak_full_bytes),
            mb(m.peak_compact_bytes),
        ]
        lines.append(line(cells))
    return "\n".join(lines)
