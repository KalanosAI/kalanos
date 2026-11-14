"""Verifies how a LeRobot data parquet's schema is compared with the features
`info.json` declares, and the `feature_mismatch_count` metric that reports it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters.lerobot.common import check_schema
from kalanos.analysis.metrics.schema import feature_mismatch_count
from kalanos.analysis.models.domain import (
    Episode,
    SchemaCheck,
    SchemaMismatch,
    SchemaProblem,
)
from kalanos.analysis.models.metrics import EpisodeContext, MetricStatus


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_SOURCE = UPath("/data/chunk-000/file-000.parquet")

_LANGUAGE_ITEM = pl.Struct({"role": pl.String, "content": pl.String})


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _check(features, schema):
    """Compare `schema` with an `info.json` declaring `features`.

    Parameters
    ----------
    features : dict[str, dict]
        The `info.json` feature entries, by key.
    schema : dict[str, pl.DataType]
        The data parquet's schema.

    Returns
    -------
    SchemaCheck
        The comparison.
    """

    return check_schema({"features": features}, schema, _SOURCE)


def _feature(dtype, shape=None):
    """Build one `info.json` feature entry.

    Parameters
    ----------
    dtype : str
        The declared dtype.
    shape : list[int] or None
        The declared shape, `[1]` when not given.

    Returns
    -------
    dict
        The feature entry.
    """

    return {"dtype": dtype, "shape": [1] if shape is None else shape, "names": None}


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


@pytest.mark.parametrize(
    ("declared", "stored"),
    [
        ("float16", pl.Float32),
        ("float32", pl.Float32),
        ("float64", pl.Float64),
        ("float64", pl.Float32),
        ("float32", pl.List(pl.Float32)),
        ("int8", pl.Int8),
        ("int64", pl.Int64),
        ("int64", pl.Int32),
        ("uint8", pl.UInt8),
        ("uint64", pl.UInt64),
        ("bool", pl.Boolean),
        ("string", pl.String),
        ("language", pl.List(_LANGUAGE_ITEM)),
        ("language", pl.String),
        ("language", pl.List(pl.Null)),
    ],
)
def test_a_column_in_its_declared_family_matches(declared, stored):
    """Verify every dtype family accepts its stored types, at any width."""

    check = _check({"feature": _feature(declared)}, {"feature": stored})

    assert check.checked == ["feature"]
    assert check.mismatches == []


def test_a_missing_column_is_reported():
    """Verify a declared feature with no column is reported as missing."""

    check = _check({"observation.state": _feature("float32", [6])}, {})

    assert check.mismatches == [
        SchemaMismatch(
            column="observation.state",
            problem=SchemaProblem.MISSING,
            declared_dtype="float32",
            declared_shape=[6],
            found_dtype=None,
        )
    ]


@pytest.mark.parametrize(
    ("declared", "stored"),
    [
        ("int64", pl.Float64),
        ("float32", pl.List(pl.Null)),
        ("float32", pl.String),
        ("bool", pl.UInt8),
        ("string", pl.Int64),
    ],
)
def test_a_column_in_another_family_is_reported(declared, stored):
    """Verify a column stored in another dtype family names both dtypes."""

    check = _check({"feature": _feature(declared)}, {"feature": stored})

    assert check.mismatches == [
        SchemaMismatch(
            column="feature",
            problem=SchemaProblem.DTYPE,
            declared_dtype=declared,
            declared_shape=[1],
            found_dtype=str(stored),
        )
    ]


def test_a_fixed_size_array_of_another_shape_is_reported():
    """Verify a fixed-size array is checked against the declared shape."""

    stored = pl.Array(pl.Float32, 6)

    check = _check({"feature": _feature("float32", [7])}, {"feature": stored})

    assert check.mismatches == [
        SchemaMismatch(
            column="feature",
            problem=SchemaProblem.SHAPE,
            declared_dtype="float32",
            declared_shape=[7],
            found_dtype=str(stored),
        )
    ]


@pytest.mark.parametrize(
    ("declared", "shape", "stored"),
    [
        ("float32", [6], pl.Array(pl.Float32, 6)),
        ("int16", [2, 2], pl.Array(pl.Int16, (2, 2))),
        ("float32", [7], pl.List(pl.Float32)),
        ("float32", [1], pl.Float32),
    ],
)
def test_a_column_whose_shape_agrees_or_is_not_fixed_matches(declared, shape, stored):
    """Verify a matching array, a variable-length list and a scalar all pass."""

    check = _check({"feature": _feature(declared, shape)}, {"feature": stored})

    assert check.mismatches == []


def test_video_and_image_features_are_not_compared():
    """Verify video and image features, stored outside the parquet, are skipped."""

    check = _check(
        {
            "observation.images.top": _feature("video", [32, 32, 3]),
            "observation.images.wrist": _feature("image", [32, 32, 3]),
            "timestamp": _feature("float32"),
        },
        {"timestamp": pl.Float32},
    )

    assert check.checked == ["timestamp"]
    assert check.mismatches == []


def test_an_unknown_dtype_is_checked_for_presence_only():
    """Verify a dtype outside the families table matches any stored type."""

    features = {"microphone": _feature("audio", [16000])}

    present = _check(features, {"microphone": pl.Array(pl.Int16, 3)})
    absent = _check(features, {})

    assert present.mismatches == []
    assert [m.problem for m in absent.mismatches] == [SchemaProblem.MISSING]


def test_the_metric_is_not_applicable_without_a_declared_schema():
    """Verify an episode no adapter compared reads as not applicable."""

    result = feature_mismatch_count(EpisodeContext(episode=Episode(id="e")))

    assert result.status == MetricStatus.NOT_APPLICABLE


def test_the_metric_counts_mismatches_and_names_them():
    """Verify the value counts mismatched features and the evidence names each."""

    mismatch = SchemaMismatch(
        column="observation.state",
        problem=SchemaProblem.MISSING,
        declared_dtype="float32",
        declared_shape=[6],
    )
    check = SchemaCheck(
        source_path=_SOURCE,
        checked=["observation.state", "timestamp"],
        mismatches=[mismatch],
    )

    result = feature_mismatch_count(
        EpisodeContext(episode=Episode(id="e", schema_check=check))
    )

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 1.0
    assert result.unit == "features"
    assert result.evidence == {
        "path": str(_SOURCE),
        "n_checked": 2,
        "mismatches": [
            {
                "column": "observation.state",
                "problem": "missing",
                "declared_dtype": "float32",
                "declared_shape": [6],
                "found_dtype": None,
            }
        ],
    }
