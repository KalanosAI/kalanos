"""Verifies how a LeRobot episode's stored per-feature stats are compared with
its data, and the `stats_mismatch_count` metric that reports it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.common import check_stats
from kalanos.analysis.metrics.schema import stats_mismatch_count
from kalanos.analysis.models.domain import Episode, StatsCheck, StatsMismatch
from kalanos.analysis.models.metrics import EpisodeContext, MetricStatus

# Local
from helpers import lerobot_stats


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_SOURCE = UPath("/meta/episodes_stats.jsonl")

_NUMERIC = ("observation.state", "action", "timestamp", "frame_index", "gripper_closed")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _frame():
    """Build one episode's rows: a fixed-size vector, a list vector and scalars.

    Returns
    -------
    pl.DataFrame
        Eight rows with float32 values that do not round-trip exactly.
    """

    n = 8
    return pl.DataFrame(
        {
            "observation.state": pl.Series(
                [[0.1 * i + 0.37 * j for j in range(3)] for i in range(n)],
                dtype=pl.Array(pl.Float32, 3),
            ),
            "action": pl.Series(
                [[213.7 + 1.3 * i, -0.01 * i] for i in range(n)],
                dtype=pl.List(pl.Float32),
            ),
            "timestamp": pl.Series([i / 30 for i in range(n)], dtype=pl.Float32),
            "frame_index": pl.Series(range(n), dtype=pl.Int64),
            "gripper_closed": pl.Series([i % 3 == 0 for i in range(n)]),
            "language": pl.Series([f"step {i}" for i in range(n)]),
        }
    )


def _info():
    """Build an `info.json` declaring the frame's features and two cameras.

    Returns
    -------
    dict
        The parsed `info.json`.
    """

    return {
        "features": {
            "observation.state": {"dtype": "float32", "shape": [3]},
            "action": {"dtype": "float32", "shape": [2]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "gripper_closed": {"dtype": "bool", "shape": [1]},
            "language": {"dtype": "string", "shape": [1]},
            "observation.images.top": {"dtype": "video", "shape": [3, 4, 4]},
            "observation.images.wrist": {"dtype": "image", "shape": [3, 4, 4]},
        }
    }


def _found(check):
    """List the (feature, stat) pairs a check reports.

    Parameters
    ----------
    check : StatsCheck
        The comparison.

    Returns
    -------
    list[tuple[str, str]]
        Each mismatch's feature and stat, in report order.
    """

    return [(m.feature, m.stat) for m in check.mismatches]


def _context(check):
    """Wrap a stats check in the context a metric reads.

    Parameters
    ----------
    check : StatsCheck or None
        The episode's stats check.

    Returns
    -------
    EpisodeContext
        The context.
    """

    return EpisodeContext(episode=Episode(id="e", stats_check=check))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_stats_written_from_the_data_raise_no_mismatch():
    """Verify float32-stored stats of the same rows are within tolerance."""

    frame = _frame()

    check = check_stats(_info(), lerobot_stats(frame, _NUMERIC), frame, _SOURCE)

    assert check.mismatches == []
    assert check.checked == list(_NUMERIC)
    assert check.source_path == _SOURCE


def test_an_edited_element_is_reported_on_its_feature_only():
    """Verify a constant added to one element flags that feature's min, max and mean."""

    frame = _frame()
    stored = lerobot_stats(frame, _NUMERIC)
    edited = frame.with_columns(
        pl.Series(
            "action",
            [[a + 50.0, b] for a, b in frame["action"].to_list()],
            dtype=pl.List(pl.Float32),
        )
    )

    check = check_stats(_info(), stored, edited, _SOURCE)

    assert _found(check) == [("action", "min"), ("action", "max"), ("action", "mean")]


def test_a_dropped_row_is_reported_as_a_count_mismatch():
    """Verify stats of more rows than the data holds flag `count`."""

    frame = _frame()
    stored = lerobot_stats(frame, ["frame_index"])

    check = check_stats(_info(), stored, frame.slice(0, frame.height - 1), _SOURCE)

    assert ("frame_index", "count") in _found(check)
    count = next(m for m in check.mismatches if m.stat == "count")
    assert (count.stored, count.found) == ([8.0], [7.0])


def test_a_stored_stat_of_the_wrong_length_is_reported():
    """Verify a stat that does not flatten to the element count is a mismatch."""

    frame = _frame()
    stored = lerobot_stats(frame, ["observation.state"])
    stored["observation.state"]["std"] = stored["observation.state"]["std"][:2]

    check = check_stats(_info(), stored, frame, _SOURCE)

    assert _found(check) == [("observation.state", "std")]


def test_a_non_numeric_stored_stat_is_reported_without_raising():
    """Verify a stat holding text is a mismatch with no stored value."""

    frame = _frame()
    stored = lerobot_stats(frame, ["timestamp"])
    stored["timestamp"]["mean"] = ["n/a"]

    check = check_stats(_info(), stored, frame, _SOURCE)

    assert check.mismatches == [
        StatsMismatch(
            feature="timestamp",
            stat="mean",
            stored=[],
            found=check.mismatches[0].found,
        )
    ]


def test_cameras_missing_columns_and_text_are_not_compared():
    """Verify image and video features, absent columns and strings are skipped."""

    frame = _frame().with_columns(
        pl.Series("observation.images.wrist", [[1.0, 2.0]] * 8),
        pl.Series("observation.images.top", [[0.5]] * 8),
    )
    stored = lerobot_stats(
        frame, ["timestamp", "observation.images.wrist", "observation.images.top"]
    )
    stored["observation.effort"] = {"mean": [1.0]}
    stored["language"] = {"count": [8]}

    check = check_stats(_info(), stored, frame, _SOURCE)

    assert check.checked == ["timestamp"]
    assert check.mismatches == []


def test_a_nan_in_the_data_matches_nan_stats():
    """Verify stats written from data holding a NaN, all NaN as numpy gives, match."""

    frame = pl.DataFrame({"timestamp": [0.0, float("nan"), 2.0]})
    nan = float("nan")
    stored = {
        "timestamp": {
            "min": [nan],
            "max": [nan],
            "mean": [nan],
            "std": [nan],
            "count": [3],
        }
    }

    check = check_stats(_info(), stored, frame, _SOURCE)

    assert check.checked == ["timestamp"]
    assert check.mismatches == []


def test_an_infinite_stored_stat_is_reported():
    """Verify a stored inf does not absorb a finite recomputation in its tolerance."""

    frame = _frame()
    stored = lerobot_stats(frame, ["timestamp"])
    stored["timestamp"]["max"] = [float("inf")]

    check = check_stats(_info(), stored, frame, _SOURCE)

    assert _found(check) == [("timestamp", "max")]


def test_an_integer_too_wide_for_a_float_is_reported_without_raising():
    """Verify a stored integer beyond float range is a mismatch with no stored value."""

    frame = _frame()
    stored = lerobot_stats(frame, ["frame_index"])
    stored["frame_index"]["min"] = [10**400]

    check = check_stats(_info(), stored, frame, _SOURCE)

    assert [(m.stat, m.stored) for m in check.mismatches] == [("min", [])]


def test_an_empty_frame_checks_nothing():
    """Verify an episode with no rows compares no feature."""

    frame = _frame()
    stored = lerobot_stats(frame, _NUMERIC)

    check = check_stats(_info(), stored, frame.clear(), _SOURCE)

    assert check == StatsCheck(source_path=_SOURCE)


def test_the_metric_is_not_applicable_without_a_check():
    """Verify an episode with no stored stats is not applicable."""

    result = stats_mismatch_count(_context(None))

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_the_metric_counts_distinct_features():
    """Verify two stats wrong on one feature and one on another count as 2."""

    check = StatsCheck(
        source_path=_SOURCE,
        checked=["action", "observation.state", "timestamp"],
        mismatches=[
            StatsMismatch(feature="action", stat="mean", stored=[1.0], found=[2.0]),
            StatsMismatch(feature="action", stat="max", stored=[1.0], found=[2.0]),
            StatsMismatch(feature="timestamp", stat="count", stored=[3], found=[2]),
        ],
    )

    result = stats_mismatch_count(_context(check))

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 2.0
    assert result.evidence["path"] == str(_SOURCE)
    assert result.evidence["n_checked"] == 3
    assert [m["stat"] for m in result.evidence["mismatches"]] == [
        "mean",
        "max",
        "count",
    ]
