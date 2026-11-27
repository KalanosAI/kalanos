"""Verifies how a LeRobot episode's rows are compared with the number of values
each feature's shape declares, and the `element_count_mismatch_count` metric
that reports it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.common import (
    FeaturePlan,
    channels_for,
    check_element_counts,
)
from kalanos.analysis.metrics.schema import element_count_mismatch_count
from kalanos.analysis.models.domain import (
    ElementCountCheck,
    ElementCountMismatch,
    Episode,
)
from kalanos.analysis.models.metrics import EpisodeContext, MetricStatus
from kalanos.assets.dictionary import load_default_dictionary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_SOURCE = UPath("data/chunk-000/file-000.parquet")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _plan(shape):
    """Build a feature plan holding one series feature, `f`, of `shape`.

    Parameters
    ----------
    shape : list[int]
        The feature's declared shape.

    Returns
    -------
    FeaturePlan
        The plan.
    """

    spec = {"dtype": "float32", "shape": shape, "names": None}
    channels = channels_for("f", spec, load_default_dictionary())
    return FeaturePlan(series={"f": ("unmapped.f", None, channels)}, video={})


def _context(check):
    """Wrap an element count check in the context a metric reads.

    Parameters
    ----------
    check : ElementCountCheck or None
        The episode's element count check.

    Returns
    -------
    EpisodeContext
        The context.
    """

    return EpisodeContext(episode=Episode(id="e", element_count_check=check))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_short_list_row_is_a_mismatch():
    """Verify frame 1 holding 2 of 3 declared values is reported with its counts."""

    frame = pl.DataFrame(
        {"f": [[1.0, 2.0, 3.0], [4.0, 5.0], [7.0, 8.0, 9.0]]},
        schema={"f": pl.List(pl.Float32)},
    )

    check = check_element_counts(_plan([3]), frame, _SOURCE)

    assert check.checked == ["f"]
    assert check.mismatches == [
        ElementCountMismatch(field="f", expected=3, found=2, frame=1)
    ]
    assert check.mismatches[0].message == "f frame 1: expected 3, found 2"


def test_a_short_nested_row_is_a_mismatch():
    """Verify a `[1, 3]` feature counts the values inside its nested list."""

    frame = pl.DataFrame(
        {"f": [[[1.0, 2.0, 3.0]], [[4.0, 5.0]]]},
        schema={"f": pl.List(pl.List(pl.Float32))},
    )

    check = check_element_counts(_plan([1, 3]), frame, _SOURCE)

    assert check.mismatches == [
        ElementCountMismatch(field="f", expected=3, found=2, frame=1)
    ]


def test_a_frame_is_named_by_its_frame_index():
    """Verify the mismatch names the row's `frame_index`, not its position."""

    frame = pl.DataFrame(
        {"frame_index": [40, 41], "f": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]},
        schema={"frame_index": pl.Int64, "f": pl.List(pl.Float32)},
    )

    check = check_element_counts(_plan([3]), frame, _SOURCE)

    assert [(m.frame, m.found) for m in check.mismatches] == [(41, 4)]


def test_a_null_row_is_no_mismatch():
    """Verify a null row is left to the missing-value metrics."""

    frame = pl.DataFrame(
        {"f": [[1.0, 2.0, 3.0], None]}, schema={"f": pl.List(pl.Float32)}
    )

    check = check_element_counts(_plan([3]), frame, _SOURCE)

    assert check.checked == ["f"]
    assert check.mismatches == []


def test_a_scalar_feature_is_not_compared():
    """Verify a column holding one number per row is not a list to count."""

    frame = pl.DataFrame({"f": [1.0, 2.0]}, schema={"f": pl.Float32})

    check = check_element_counts(_plan([1]), frame, _SOURCE)

    assert check == ElementCountCheck(source_path=_SOURCE)


def test_the_metric_is_not_applicable_without_a_check():
    """Verify an episode whose data was not read is not applicable."""

    result = element_count_mismatch_count(_context(None))

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_the_metric_counts_every_frame_and_lists_the_first_ten():
    """Verify 12 malformed frames count 12, with the first 10 listed in frame order."""

    check = ElementCountCheck(
        source_path=_SOURCE,
        checked=["f"],
        mismatches=[
            ElementCountMismatch(field="f", expected=3, found=2, frame=frame)
            for frame in reversed(range(12))
        ],
    )

    result = element_count_mismatch_count(_context(check))

    assert result.value == 12.0
    assert result.evidence["path"] == str(_SOURCE)
    assert result.evidence["n_checked"] == 1
    [field] = result.evidence["fields"]
    assert (field["field"], field["expected"], field["n_frames"]) == ("f", 3, 12)
    assert [f["frame"] for f in field["frames"]] == list(range(10))
    assert field["frames"][0] == {
        "frame": 0,
        "found": 2,
        "message": "f frame 0: expected 3, found 2",
    }
