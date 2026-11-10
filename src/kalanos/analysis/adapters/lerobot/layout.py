"""Where a LeRobot episode's rows are stored, compared with where its metadata says.

The data files are read for two columns only, `episode_index` and `index`.
A file with a direct location is fetched for those columns' byte ranges alone,
and any other file is fetched whole through its fsspec handle.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Mapping, Sequence
from typing import Any

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.hosts import direct_location
from kalanos.analysis.models.domain import LayoutCheck, RangeMismatch


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_RANGE_FIELDS = ("dataset_from_index", "dataset_to_index", "length")

# What lerobot does with an episode whose metadata names a data file without its rows
_POINTER_EFFECT = (
    "loading only this episode reads the file its metadata names and finds no rows; "
    "delete_episodes and split_dataset raise KeyError"
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _columns(with_index: bool) -> list[str]:
    """Name the data columns a scan reads.

    Parameters
    ----------
    with_index : bool
        Whether to read the `index` column too.

    Returns
    -------
    list[str]
        `episode_index`, and `index` when asked for.
    """

    return ["episode_index", "index"] if with_index else ["episode_index"]


def _located(columns: list[str], source: int) -> list[pl.Expr]:
    """Select the located columns as Int64 and tag them with their file.

    Data files can store `episode_index` or `index` in different integer widths,
    and `pl.concat` refuses frames whose schemas differ.

    Parameters
    ----------
    columns : list[str]
        The columns `_columns` names.
    source : int
        The data file's position.

    Returns
    -------
    list[pl.Expr]
        One expression per column, then `source`.
    """

    return [pl.col(c).cast(pl.Int64) for c in columns] + [
        pl.lit(source, pl.UInt32).alias("source")
    ]


def _empty(with_index: bool) -> pl.DataFrame:
    """Build the output of a scan over no file.

    Parameters
    ----------
    with_index : bool
        Whether the output carries the index range columns.

    Returns
    -------
    pl.DataFrame
        No rows, with the columns `_aggregate` produces.
    """

    schema: dict[str, pl.DataType] = {
        "source": pl.UInt32(),
        "episode_index": pl.Int64(),
        "rows": pl.UInt32(),
    }
    if with_index:
        schema |= {"index_from": pl.Int64(), "index_to": pl.Int64()}
    return pl.DataFrame(schema=schema)


def _aggregate(rows: pl.LazyFrame, with_index: bool) -> pl.LazyFrame:
    """Reduce data rows tagged by `source` to one row per source and episode.

    Parameters
    ----------
    rows : pl.LazyFrame
        A `source` column, `episode_index`, and `index` when `with_index`,
        both Int64.
    with_index : bool
        Whether to aggregate the index range.

    Returns
    -------
    pl.LazyFrame
        `source`, `episode_index` and `rows`, plus `index_from` (inclusive)
        and `index_to` (exclusive) when `with_index`, sorted by source then episode.
    """

    aggregations = [pl.len().alias("rows")]
    if with_index:
        aggregations += [
            pl.col("index").min().alias("index_from"),
            (pl.col("index").max() + 1).alias("index_to"),
        ]
    return (
        rows.group_by("source", "episode_index")
        .agg(aggregations)
        .sort("source", "episode_index")
    )


def aggregate_episode_rows(frame: pl.DataFrame, *, with_index: bool) -> pl.DataFrame:
    """Reduce one data file's rows, already in memory, to one row per episode.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows of a single data file, with `episode_index`,
        and `index` when `with_index`.
    with_index : bool
        Whether to aggregate the index range.

    Returns
    -------
    pl.DataFrame
        The shape `read_episode_rows` returns, every row with `source` 0.
    """

    return _aggregate(
        frame.lazy().select(_located(_columns(with_index), 0)), with_index
    ).collect()


def scan_episode_rows(
    sources: Sequence[str],
    storage_options: dict[str, str] | None,
    *,
    with_index: bool,
) -> pl.DataFrame:
    """Read where each episode's rows are, through polars' native parquet reader.

    Parameters
    ----------
    sources : Sequence[str]
        Data file paths or URIs polars can read.
    storage_options : dict[str, str] or None
        Passed to every `pl.scan_parquet`.
    with_index : bool
        Whether to read the `index` column and aggregate its range.

    Returns
    -------
    pl.DataFrame
        One row per data file and episode: `source`, the file's position in
        `sources`, `episode_index` and `rows`, plus `index_from` and `index_to`
        when `with_index`.
    """

    if not sources:
        return _empty(with_index)
    # Each file is tagged by its position: `include_file_paths` returns
    # an `hf://` input as an `https://` URL, which matches no input string.
    frames = [
        pl.scan_parquet(source, storage_options=storage_options).select(
            _located(_columns(with_index), i)
        )
        for i, source in enumerate(sources)
    ]
    return _aggregate(pl.concat(frames), with_index).collect()


def read_episode_rows(paths: Sequence[UPath], *, with_index: bool) -> pl.DataFrame:
    """Read where each episode's rows are, across `paths`.

    Paths with a direct location are scanned natively, for the two columns' byte
    ranges alone. Any other path is read through its fsspec handle, whole.

    Parameters
    ----------
    paths : Sequence[UPath]
        The data files.
    with_index : bool
        Whether to read the `index` column and aggregate its range.

    Returns
    -------
    pl.DataFrame
        The shape `scan_episode_rows` returns, `source` indexing `paths`.
    """

    mapped = [direct_location(p) for p in paths]
    if all(m is not None for m in mapped):
        sources = [m[0] for m in mapped if m is not None]
        # Every path shares one root, so they share its storage options.
        storage_options = mapped[0][1] if mapped and mapped[0] is not None else None
        return scan_episode_rows(sources, storage_options, with_index=with_index)

    frames = []
    for i, path in enumerate(paths):
        with path.open("rb") as handle:
            frame = pl.read_parquet(handle, columns=_columns(with_index))
        frames.append(frame.lazy().select(_located(_columns(with_index), i)))
    return _aggregate(pl.concat(frames), with_index).collect()


def layout_check(
    row: Mapping[str, Any],
    pointed_path: UPath,
    holding: pl.DataFrame,
    paths: Sequence[UPath],
) -> LayoutCheck:
    """Compare an episode's metadata row with where its rows were found.

    Parameters
    ----------
    row : Mapping[str, Any]
        The episode's row of the episode index.
    pointed_path : UPath
        The data file `row` names.
    holding : pl.DataFrame
        This episode's rows of `read_episode_rows` output.
    paths : Sequence[UPath]
        The data files `holding["source"]` indexes.

    Returns
    -------
    LayoutCheck
        The holding files, the range fields that disagree with the rows found,
        and what lerobot does with the episode when the pointer misses.
        The range is not compared when `row` lacks a range field or holds a null
        in one, when `holding` has no `index_from`, or when it is empty.
    """

    holding_paths = [paths[source] for source in holding["source"].to_list()]
    range_mismatches = None
    if (
        holding.height
        and "index_from" in holding.columns
        and all(row.get(field) is not None for field in _RANGE_FIELDS)
    ):
        found_from, found_to = holding.select(
            pl.col("index_from").min(), pl.col("index_to").max()
        ).row(0)
        # `length` is compared with the rows found, since a declared triple can
        # agree with itself and still be wrong.
        found = {
            "dataset_from_index": found_from,
            "dataset_to_index": found_to,
            "length": found_to - found_from,
        }
        range_mismatches = [
            RangeMismatch(field=field, declared=int(row[field]), found=found[field])
            for field in _RANGE_FIELDS
            if int(row[field]) != found[field]
        ]
    return LayoutCheck(
        pointed_path=pointed_path,
        holding_paths=holding_paths,
        range_mismatches=range_mismatches,
        pointer_effect=_POINTER_EFFECT,
    )
