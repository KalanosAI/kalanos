"""Shared test-only helpers: the fixture corpus location, and HTML-vs-model checks."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from html.parser import HTMLParser

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.inference.dialect import sniff_dialect
from kalanos.analysis.inference.infer import infer_schema
from kalanos.analysis.models.schema import SourceSchema
from kalanos.analysis.models.scoring import ScoreResult


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FIXTURES_DIR = UPath(__file__).parent / "fixtures"
CSV_FIXTURE = FIXTURES_DIR / "arm_multi_device.csv"
IMU_FIXTURE = FIXTURES_DIR / "imu_stream.txt"
POSE_FIXTURE = FIXTURES_DIR / "pose_log.txt"
CAPTURE_INDEX_FIXTURE = FIXTURES_DIR / "capture_index.json"
LEROBOT_FIXTURE = FIXTURES_DIR / "lerobot_v3_tiny"
LEROBOT_V2_0_FIXTURE = FIXTURES_DIR / "lerobot_v2_0_tiny"
LEROBOT_V2_1_FIXTURE = FIXTURES_DIR / "lerobot_v2_1_tiny"
HDF5_FIXTURE = FIXTURES_DIR / "hdf5_tiny.hdf5"
MCAP_FIXTURE = FIXTURES_DIR / "mcap_tiny.mcap"

# A flat metadata blob with no time index, expected to be skipped rather than analysed.
NO_TIMESERIES_FIXTURE = FIXTURES_DIR / "video_meta.json"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


def csv_schema(path: UPath) -> SourceSchema:
    """Sniff, read, and infer a CSV file's schema.

    Parameters
    ----------
    path : UPath
        The CSV file to resolve.

    Returns
    -------
    SourceSchema
        The resolved schema, dialect included.
    """

    dialect = sniff_dialect(path)
    with path.open("rb") as handle:
        frame = pl.read_csv(
            source=handle, separator=dialect.delimiter, has_header=dialect.has_header
        )
    return infer_schema(frame).model_copy(update={"dialect": dialect})


class ScoreAttributeCollector(HTMLParser):
    """Collects every `data-level`/`data-name`/`data-score` triple, in document order.

    Attributes
    ----------
    rows : list[tuple[str, str, str]]
        Every `(level, name, score)` triple seen so far, in document order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one element's score attributes, if it carries any.

        Parameters
        ----------
        tag : str
            The element's tag name; unused, every level uses a different tag.
        attrs : list[tuple[str, str or None]]
            The element's attributes, as `HTMLParser` hands them over.
        """

        attr = dict(attrs)
        if "data-level" in attr:
            self.rows.append(
                (
                    attr["data-level"] or "",
                    attr.get("data-name") or "",
                    attr.get("data-score") or "",
                )
            )


class DisclosureStateCollector(HTMLParser):
    """Collects every tree node's `(level, name, instance, open)` state, in order.

    Attributes
    ----------
    rows : list[tuple[str, str, str, bool]]
        Every `(level, name, instance, open)` state seen so far, in document order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str, str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one tree node's disclosure state, if it carries one.

        Parameters
        ----------
        tag : str
            The element's tag name; unused, every node uses `<details>`.
        attrs : list[tuple[str, str or None]]
            The element's attributes, as `HTMLParser` hands them over.
        """

        attr = dict(attrs)
        if "data-level" in attr and "data-node" in attr:
            self.rows.append(
                (
                    attr["data-level"] or "",
                    attr.get("data-name") or "",
                    attr.get("data-instance") or "",
                    "open" in attr,
                )
            )


def score_attr(score: ScoreResult) -> str:
    """Reproduce the HTML renderer's own `data-score` formatting rule.

    Parameters
    ----------
    score : ScoreResult
        The score an element in the page carries.

    Returns
    -------
    str
        `repr(score.score)`, or an empty string when `score.score` is `None` —
        the same rule `render_html` uses, kept here independently so a test
        does not import the renderer's private formatting helper.
    """

    return "" if score.score is None else repr(score.score)
