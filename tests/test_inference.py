"""Verifies inference: resolving a frame's dialect, time column, entity key, regularity."""  # noqa: E501

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
from pathlib import Path

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.inference.dialect import sniff_dialect
from kalanos.analysis.inference.entity import score_entity_candidate
from kalanos.analysis.inference.infer import infer_schema
from kalanos.analysis.inference.regularity import entity_split_gaps, regularity
from kalanos.analysis.inference.timestamp import (
    infer_unit,
    reconcile_unit,
    score_time_candidate,
    time_axis,
    unit_from_name,
)
from kalanos.analysis.models.schema import SourceSchema

# Local
from helpers import csv_schema


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FIXTURE = UPath(__file__).parent / "fixtures" / "arm_multi_device.csv"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _csv(tmp_path: Path, name: str, text: str) -> UPath:
    """Write `text` to `tmp_path / name` and return the path.

    Parameters
    ----------
    tmp_path : Path
        Pytest's per-test temp directory.
    name : str
        File name to write.
    text : str
        File contents.

    Returns
    -------
    UPath
        The written file's path.
    """

    path = tmp_path / name
    path.write_text(text)
    return UPath(path)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_dialect_detects_comma_delimiter_and_header():
    """Verify the delimiter and header are detected, not configured."""

    dialect = sniff_dialect(FIXTURE)
    assert dialect.delimiter == ","
    assert dialect.has_header is True
    assert dialect.evidence


def test_infer_schema_runs_on_a_bare_frame_with_no_file_anywhere():
    """Verify the library resolves a schema from a frame built in memory.

    This is the point of turning inference into a library: nothing here
    reads a path, sniffs a dialect, or knows CSV exists.
    """

    frame = pl.DataFrame(
        {
            "t_ms": [t for t in range(0, 500, 10)],
            "id": ["armA", "armB"] * 25,
            "x": [float(t) for t in range(50)],
        }
    )

    schema = infer_schema(frame)

    assert isinstance(schema, SourceSchema)
    assert schema.dialect is None
    assert schema.time is not None
    assert schema.time.column == "t_ms"
    assert schema.time.unit == "ms"
    assert schema.entity_key is not None
    assert schema.entity_key.column == "id"


def test_t_ms_is_identified_as_the_time_column():
    """Verify t_ms wins, with its unit resolved to ms.

    The fixture's `_ms` suffix and its 10 ms grid agree, so this alone
    can't show which signal decided `unit`. The disagreement tests below
    cover that.
    """

    result = csv_schema(FIXTURE)
    assert result.time is not None
    assert result.time.column == "t_ms"
    assert result.time.unit == "ms"


def test_a_signal_column_is_ruled_out_as_the_time_column():
    """Verify a signal column scores below t_ms, and the evidence says why.

    `tcp_pose_z_mm` is used because it has the lowest monotonicity of the three
    signal columns — about a third of its steps are non-decreasing, against
    t_ms's 100%.
    """

    with FIXTURE.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], rows[1:]
    z_idx = header.index("tcp_pose_z_mm")
    t_idx = header.index("t_ms")

    z_score = score_time_candidate("tcp_pose_z_mm", [float(r[z_idx]) for r in data])
    t_score = score_time_candidate("t_ms", [float(r[t_idx]) for r in data])

    assert z_score.confidence < t_score.confidence
    assert any("does not match" in line for line in z_score.evidence)
    assert not any("100%" in line for line in z_score.evidence)


def test_a_decrease_hidden_behind_a_null_still_counts_against_a_column():
    """Verify compacting nulls, not skipping null-adjacent pairs, is what scores.

    Skipping null-adjacent pairs instead of compacting would score both
    columns at full monotonicity, since the pair straddling the null would
    never be compared.
    """

    with_null = score_time_candidate("t_ms", [0.0, 10.0, None, 5.0])
    clean = score_time_candidate("t_ms", [0.0, 10.0, 20.0, 30.0])

    assert with_null.confidence < clean.confidence
    assert any("1 of 4 values are null" in line for line in with_null.evidence)
    assert not any("null" in line for line in clean.evidence)


def test_id_is_identified_as_the_entity_key():
    """Verify the id column is identified as the entity key."""

    result = csv_schema(FIXTURE)
    assert result.entity_key is not None
    assert result.entity_key.column == "id"
    assert result.entity_key.evidence


def test_entity_key_scoring_carries_evidence_and_confidence():
    """Verify the entity-key sniffer returns evidence, not only an answer."""

    confidence, evidence = score_entity_candidate(
        "id", ["armA", "armB", "armA", "armC", "armA"]
    )
    assert confidence > 0
    assert evidence


def test_the_fixture_series_classifies_as_regular():
    """Verify the fixture's dominant grid reads as regular sampling."""

    result = csv_schema(FIXTURE)
    assert result.time is not None
    assert result.time.regularity.is_regular is True
    assert result.time.regularity.expected_dt == 10.0


def test_regularity_is_computed_after_the_entity_split(tmp_path):
    """Verify splitting by entity first avoids reading interleaved rows as zero-gap.

    Four entities share a subset of timestamps in the fixture; naive
    consecutive-row gaps on the raw file order would include many
    near-zero gaps between different entities' rows. Splitting by entity
    first and pooling only within-entity gaps must not do that.
    """

    # Two entities alternate every row at a shared clock, so raw
    # consecutive-row gaps would be ~0 half the time.
    lines = ["t_ms,id,x"]
    for t in range(0, 500, 10):
        lines.append(f"{t},armA,{t}")
        lines.append(f"{t},armB,{-t}")
    interleaved = _csv(tmp_path, "interleaved.csv", "\n".join(lines) + "\n")

    result = csv_schema(interleaved)
    assert result.time is not None
    assert result.time.regularity.is_regular is True
    assert result.time.regularity.expected_dt == 10.0


def test_entity_split_gaps_pools_only_within_entity_differences():
    """Verify entity_split_gaps never differences across entities."""

    gaps = entity_split_gaps([[0.0, 10.0, 20.0], [0.0, 5.0, 10.0]])
    assert sorted(gaps) == [5.0, 5.0, 10.0, 10.0]


def test_entity_split_gaps_drops_null_timestamps():
    """Verify a null timestamp is bridged into one wide gap, not a break or a raise."""

    gaps = entity_split_gaps([[0.0, None, 20.0], [0.0, 5.0]])
    assert sorted(gaps) == [5.0, 20.0]


def test_an_irregular_series_classifies_as_irregular():
    """Verify a series with no consistent nominal gap classifies as irregular."""

    gaps = [1.0, 9.0, 2.0, 40.0, 3.0, 25.0, 0.5, 60.0]
    result = regularity(gaps)
    assert result.is_regular is False
    assert result.expected_dt is None


def test_a_file_with_no_entity_key_still_resolves(tmp_path):
    """Verify a file with no categorical column resolves with entity_key=None."""

    single_entity = _csv(
        tmp_path,
        "single_entity.csv",
        "t_ms,x\n" + "\n".join(f"{t},{t * 0.1}" for t in range(0, 200, 10)) + "\n",
    )

    result = csv_schema(single_entity)

    assert result.entity_key is None
    assert result.time is not None
    assert result.time.column == "t_ms"


def test_a_time_column_with_a_dropped_sample_still_resolves(tmp_path):
    """Verify a null cell in the time column resolves instead of raising.

    A null timestamp reaches inference as a Python `None`, which the
    monotonicity and gap paths must both tolerate. The single 20 ms gap
    where the dropped sample was stays inside the 80% on-grid tolerance,
    so the column still reads as regular.
    """

    rows = ["t_ms,x"]
    for i in range(50):
        t = "" if i == 25 else str(i * 10)
        rows.append(f"{t},{i}")
    dropped_sample = _csv(tmp_path, "dropped_sample.csv", "\n".join(rows) + "\n")

    result = csv_schema(dropped_sample)

    assert result.time is not None
    assert result.time.column == "t_ms"
    assert result.time.unit == "ms"
    assert result.time.regularity.is_regular is True
    assert result.time.regularity.expected_dt == 10.0


def test_an_all_null_time_candidate_is_refused_not_raised():
    """Verify an all-null numeric column is refused as the time axis, not raised on."""

    frame = pl.DataFrame(
        {"t_ms": [None, None, None], "x": [1.0, 2.0, 3.0]},
        schema={"t_ms": pl.Float64, "x": pl.Float64},
    )

    result = time_axis(frame)

    assert result.column is None


@pytest.mark.parametrize(
    ("nominal_interval", "expected"),
    [
        (0.0, "unknown"),  # a column that never advances has no unit to claim
        (0.999, "s"),
        (1.0, "ms"),
        (999, "ms"),
        (1000, "us"),
        (999_999, "us"),
        (1_000_000, "ns"),
        (-0.5, "s"),  # abs() applied before banding
    ],
)
def test_infer_unit_band_boundaries(nominal_interval, expected):
    """Verify infer_unit's magnitude bands at each boundary, including a negative gap."""  # noqa: E501

    assert infer_unit(nominal_interval) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("t_ms", "ms"),
        ("t_us", "us"),
        ("timestamp_ns", "ns"),
        ("time_sec", "s"),
        ("time (ms)", "ms"),
        ("t [ s ]", "s"),
        ("T_MS", "ms"),
        ("unix_ms", "ms"),  # not caught by _TIME_NAME_PATTERN at all
        ("epoch_us", "us"),
    ],
)
def test_unit_from_name_reads_recognised_suffixes(name, expected):
    """Verify every supported suffix and bracket form resolves to its unit."""

    assert unit_from_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "timestamp",  # no unit suffix at all
        "arm_x_mm",  # a signal column's unit, not a time unit
        "ts",  # trailing bare 's' must not read as seconds
        "times",  # same trap, plural
        "t_abs",  # ends in 's'-shaped text but is not the unit token
        "t_mm",  # one character from '_ms' - must not collide
    ],
)
def test_unit_from_name_rejects_non_unit_and_near_miss_names(name):
    """Verify names with no unit suffix, or only a near-miss, return None.

    False positives are the danger here, not misses: a wrong reading would
    feed straight into the confidence penalty as if it were real evidence.
    """

    assert unit_from_name(name) is None


def test_unit_from_name_does_not_read_an_infix_suffix():
    """Verify a suffix that isn't trailing is not read as a unit.

    `t_ms_raw` plausibly means milliseconds too, but this only reads a
    trailing suffix or a bracketed form. Might be worth widening if names
    like this show up in real data.
    """

    assert unit_from_name("t_ms_raw") is None


def test_reconcile_unit_agreement_keeps_full_confidence():
    """Verify a name and a magnitude that agree cost nothing."""

    resolution = reconcile_unit("t_ms", 10.0)
    assert resolution.unit == "ms"
    assert resolution.confidence_factor == 1.0
    assert any("agrees" in line for line in resolution.evidence)


def test_reconcile_unit_disagreement_keeps_magnitude_and_costs_confidence():
    """Verify magnitude still wins on disagreement, but the disagreement is priced in.

    A `_s` column whose data is really milliseconds is exactly what the
    original magnitude-only rule was meant to catch. If this regresses,
    the change trades one silent error for another.
    """

    resolution = reconcile_unit("t_s", 2.0)
    assert resolution.unit == "ms"
    assert resolution.confidence_factor < 1.0
    assert any("disagrees" in line for line in resolution.evidence)


def test_reconcile_unit_mirrors_the_disagreement_in_the_other_direction():
    """Verify the penalty also fires when the name overclaims and magnitude underclaims.

    Epoch milliseconds sampled at 1 Hz: a 1000-unit gap reads as `us` by
    magnitude alone, while the name claims `ms`. Same failure mode as the
    test above, mirrored.
    """

    resolution = reconcile_unit("t_ms", 1000.0)
    assert resolution.unit == "us"
    assert resolution.confidence_factor < 1.0


def test_reconcile_unit_has_no_cross_check_without_a_name_suffix():
    """Verify a name with no unit suffix defers entirely to magnitude, undocked."""

    resolution = reconcile_unit("timestamp", 0.02)
    assert resolution.unit == "s"
    assert resolution.confidence_factor == 1.0
    assert any("no unit suffix" in line for line in resolution.evidence)


def test_reconcile_unit_falls_back_to_the_name_when_there_is_no_magnitude():
    """Verify an irregular source resolves a unit from the name, at a discount.

    A header-only reading is the only signal available, and it used to carry
    the same full confidence as a magnitude-corroborated one — indistinguishable
    downstream from a column that had actually been cross-checked.
    """

    resolution = reconcile_unit("t_ms", None)
    assert resolution.unit == "ms"
    assert resolution.confidence_factor < 1.0
    assert any("uncorroborated" in line for line in resolution.evidence)


def test_reconcile_unit_is_unknown_with_neither_signal():
    """Verify no magnitude and no name suffix still resolves, honestly, to unknown."""

    resolution = reconcile_unit("x", None)
    assert resolution.unit == "unknown"
    assert resolution.confidence_factor == 1.0


def test_a_time_s_column_with_second_scale_data_resolves_to_seconds(tmp_path):
    """Verify a mislabelled column still gets the right unit, end to end.

    `t_ms` named but sampled on a 0.02 s grid: magnitude must win, the
    disagreement must show up in evidence, and confidence must drop but
    still clear the gate (name matches, fully monotonic).
    """

    lines = ["t_ms,x"]
    for i in range(50):
        t = round(i * 0.02, 2)
        lines.append(f"{t},{i}")
    mislabelled = _csv(tmp_path, "mislabelled.csv", "\n".join(lines) + "\n")

    result = csv_schema(mislabelled)

    assert result.time is not None
    assert result.time.column == "t_ms"
    assert result.time.unit == "s"
    assert result.time.confidence == pytest.approx(0.8)
    assert any("disagrees" in line for line in result.time.evidence)


def test_a_unit_disagreement_can_still_sink_a_column_that_cleared_the_score_gate(
    tmp_path,
):
    """Verify the gate applies to the combined confidence, not the column score alone.

    `t_ms` matches the time-like name pattern and is non-decreasing on half its
    steps, for a raw score of `0.45 + 0.55 * 0.5 = 0.725` — clear of the 0.6
    gate on its own. Its values are evenly spaced 0.05 apart once sorted, so
    the magnitude reads as seconds while the name claims milliseconds; the
    resulting 0.8 mismatch penalty drops the combined confidence to 0.58,
    below the gate. The column score alone would have accepted this column;
    only the unit cross-check refuses it.
    """

    values = [0.00, 0.30, 0.05, 0.35, 0.10, 0.40, 0.15, 0.45, 0.20, 0.50, 0.25]
    lines = ["t_ms", *(str(v) for v in values)]
    weakly_named = _csv(tmp_path, "weakly_named.csv", "\n".join(lines) + "\n")

    result = csv_schema(weakly_named)

    assert result.time is not None
    assert result.time.column is None
    assert any("t_ms" in line for line in result.time.evidence)


def test_a_monotonic_column_with_no_time_name_and_no_time_unit_is_refused(tmp_path):
    """Verify monotonicity alone no longer wins the time axis.

    `distance_mm` has no time-like name and no recognisable time-unit
    suffix (`_mm` is a length unit, not a time one), so nothing corroborates
    monotonicity as evidence of a clock. Neither signal alone reaches the
    acceptance threshold, so a column like this is refused rather than
    silently becoming the time axis.
    """

    lines = ["distance_mm"]
    for i in range(50):
        lines.append(str(i))
    cumulative = _csv(tmp_path, "cumulative.csv", "\n".join(lines) + "\n")

    result = csv_schema(cumulative)

    assert result.time is not None
    assert result.time.column is None
    assert any("distance_mm" in line for line in result.time.evidence)


def test_time_axis_reports_the_regularity_it_computed():
    """Verify time_axis carries the regularity of the column it resolved.

    Regularity is a property of the winning time column, derived from the
    same gaps that resolve its unit — this checks it travels with the
    `TimeSpec` rather than being recomputed, or lost, downstream.
    """

    frame = pl.DataFrame({"t_ms": list(range(0, 500, 10))})

    spec = time_axis(frame)

    assert spec.column == "t_ms"
    assert spec.regularity.is_regular is True
    assert spec.regularity.expected_dt == 10.0
    assert spec.regularity.confidence > 0
