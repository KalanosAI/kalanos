"""Regression controls for window identity, exact clocks and report counts."""

import builtins
from copy import deepcopy

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from test_deeper_diagnostics import (
    audit,
    episode,
    relation,
    signal,
    window_spec,
)

from kalanos.analysis.diagnostics.windows import windows
from kalanos.analysis.localization import support_for
from kalanos.analysis.models.binding import RequirementsSection
from kalanos.analysis.models.diagnostics import (
    DiagnosticPlan,
    DiagnosticResult,
    ModalitySpec,
    MotionSpec,
    Selector,
)
from kalanos.analysis.models.domain import ClockInfo, FramePayload
from kalanos.analysis.models.provenance import Inventory
from kalanos.analysis.models.report import Report
from kalanos.analysis.models.scoring import Finding
from kalanos.analysis.scoring.eligibility import counts_of


def two_channels(*, missing=False):
    """Create two original indices sharing one feature and source clock."""
    s = signal(values=list(np.arange(10, dtype=float)), times=np.arange(10) / 10)
    other = s.channels[0].model_copy(deep=True)
    other.name = "other"
    other.source_index = 1
    other.binding.index = 1
    s.channels.append(other)
    values = list(np.arange(10, dtype=float))
    if missing:
        values[4] = None
    s.payload = FramePayload(
        frame=pl.DataFrame({"value": list(np.arange(10, dtype=float)), "other": values})
    )
    return episode(s)


def finding(**updates):
    """Construct a scoped review with ordinary source identity fields."""
    return Finding(
        **{
            "id": "fixture-finding",
            "metric_id": "integrity.fixture",
            "family": "integrity",
            "severity": "critical",
            "value": 1,
            "unit": None,
            "points": 0,
            "episode_id": "e",
            "source_path": "fixture.csv",
            "source_field": "state",
            "source_index": 1,
            "channel": "other",
            "consequence": "review",
            **updates,
        }
    )


def test_unused_channel_review_keeps_selected_windows_pass_but_episode_review():
    e = two_channels(missing=True)
    r = audit([e], DiagnosticPlan(windows=[window_spec()]))
    assert r.episodes[0].eligibility.status.value == "review"
    assert r.diagnostics.results[0].measurements["counts"]["pass"] == 7
    assert Report.model_validate_json(r.model_dump_json()) == r


@pytest.mark.parametrize("consequence", ["review", "block"])
@pytest.mark.parametrize("by_name", [False, True])
def test_channel_findings_only_affect_consumed_indices(consequence, by_name):
    e = two_channels()
    f = finding(consequence=consequence, source_index=None if by_name else 1)
    spec = window_spec()
    assert windows(e, spec, [f]).measurements["counts"]["pass"] == 7
    spec.modalities[0].selector.index = 1
    expected = "blocked" if consequence == "block" else "review"
    assert windows(e, spec, [f]).measurements["counts"][expected] == 7
    spec.modalities[0].selector.index = None
    assert windows(e, spec, [f]).measurements["counts"][expected] == 7


@pytest.mark.parametrize(
    "updates",
    [
        {"episode_id": "other"},
        {"source_path": "elsewhere.csv"},
        {"source_field": "different"},
        {"instance": "other-arm"},
    ],
)
def test_findings_from_other_subjects_do_not_leak(updates):
    f = finding(source_index=0, channel="value", **updates)
    assert (
        windows(two_channels(), window_spec(), [f]).measurements["counts"]["pass"] == 7
    )


@pytest.mark.parametrize("episode_wide", [False, True])
def test_broad_findings_still_propagate(episode_wide):
    f = finding(
        source_index=None,
        channel=None,
        source_path=None if episode_wide else "fixture.csv",
        source_field=None if episode_wide else "state",
    )
    assert (
        windows(two_channels(), window_spec(), [f]).measurements["counts"]["review"]
        == 7
    )


def test_selected_channel_intervals_only_mark_overlapping_windows():
    f = finding(
        source_index=0,
        channel="value",
        support=support_for(list(range(10)), [(4, 5)]),
    )
    counts = windows(two_channels(), window_spec(), [f]).measurements["counts"]
    assert counts == {"pass": 3, "blocked": 0, "review": 4, "unknown": 0}


def native_signal(unit="ns", origin=0, *, feature="state", rate=10):
    """Record exact integer native ticks, independently of float display times."""
    units_per_second = {"ns": 10**9, "us": 10**6, "ms": 1000}[unit]
    step = int(units_per_second / rate)
    s = signal(
        feature, values=list(np.arange(10, dtype=float)), times=np.arange(10) / rate
    )
    s.native_timestamps = pl.Series(
        [origin + i * step for i in range(10)], dtype=pl.Int64
    )
    s.clock_info = ClockInfo(
        native_unit=unit, origin="capture", origin_evidence="producer", domain="fixture"
    )
    return s


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("epoch", [False, True])
@pytest.mark.parametrize("rate", [10, 12.5])
def test_causal_windows_match_exact_native_grid_without_roundoff(unit, epoch, rate):
    origin = {"ns": 10**18, "us": 10**15, "ms": 10**12}[unit] if epoch else 0
    e = episode(native_signal(unit, origin, rate=rate))
    spec = window_spec()
    spec.sample_rate_hz = rate
    spec.max_gap_s = 1 / rate
    spec.modalities[0].matching = "previous"
    spec.modalities[0].max_age_s = 0
    r = windows(e, spec)
    assert r.measurements["counts"] == {
        "pass": 7,
        "blocked": 0,
        "review": 0,
        "unknown": 0,
    }
    assert r.evidence["windows"][0]["consumed"][0]["source_rows"] == [0, 1, 2, 3]
    assert DiagnosticResult.model_validate_json(r.model_dump_json()) == r


def test_float_native_origin_is_subtracted_without_manufacturing_an_age():
    s = signal(
        values=list(np.arange(10, dtype=float)), times=[1 + i / 10 for i in range(10)]
    )
    spec = window_spec()
    spec.modalities[0].matching = "previous"
    spec.modalities[0].max_age_s = 0
    assert windows(episode(s), spec).measurements["counts"]["pass"] == 7


@pytest.mark.parametrize("future", [1, 100])
def test_causal_matching_does_not_accept_a_genuinely_future_native_tick(future):
    left, right = native_signal(), native_signal(feature="camera-state")
    right.native_timestamps = right.native_timestamps + future
    spec = window_spec()
    spec.modalities = [
        ModalitySpec(
            selector=Selector(feature="camera-state", index=0),
            relation=relation(),
            matching="previous",
            max_age_s=0.00001,
        )
    ]
    assert windows(episode(left, right), spec).measurements["counts"]["blocked"] == 7


def test_causal_matching_preserves_a_distinct_future_float_timestamp():
    left = signal(values=list(np.arange(10, dtype=float)), times=np.arange(10) / 10)
    times = list(np.arange(10) / 10)
    times[3] = float(np.nextafter(0.3, np.inf))
    right = signal("camera-state", values=list(np.arange(10, dtype=float)), times=times)
    spec = window_spec()
    spec.modalities = [
        ModalitySpec(
            selector=Selector(feature="camera-state", index=0),
            relation=relation(),
            matching="previous",
            max_age_s=0.00001,
        )
    ]
    counts = windows(episode(left, right), spec).measurements["counts"]
    assert counts == {"pass": 3, "blocked": 4, "review": 0, "unknown": 0}


@pytest.mark.parametrize("late", [False, True])
def test_exact_max_age_boundary_is_inclusive_without_widening_it(late):
    left, right = native_signal(), native_signal(feature="camera-state")
    right.native_timestamps = right.native_timestamps - 10_000_000 - int(late)
    spec = window_spec()
    spec.modalities = [
        ModalitySpec(
            selector=Selector(feature="camera-state", index=0),
            relation=relation(),
            matching="previous",
            max_age_s=0.01,
        )
    ]
    status = "blocked" if late else "pass"
    assert windows(episode(left, right), spec).measurements["counts"][status] == 7


def test_mixed_units_and_declared_affine_transform_match_exactly():
    left = native_signal("ns", 10**18)
    right = native_signal("us", 10**15, feature="camera-state")
    # Right runs at half the left rate; the declared relation maps it exactly.
    right.native_timestamps = pl.Series([10**15 + i * 50_000 for i in range(10)])
    spec = window_spec()
    spec.modalities = [
        ModalitySpec(
            selector=Selector(feature="camera-state", index=0),
            relation=relation(drift_ppm=1_000_000, anchor_s=1_000_000_000),
            matching="previous",
            max_age_s=0,
        )
    ]
    assert windows(episode(left, right), spec).measurements["counts"]["pass"] == 7


def mixed_report():
    """Two passing episodes and one review, all produced by normal assembly."""
    return audit(
        [
            episode(signal(), identifier="a"),
            episode(signal(), identifier="b"),
            two_channels(missing=True),
        ]
    )


@pytest.mark.parametrize(
    "source,target",
    [("pass_count", "review"), ("review", "pass_count"), ("review", "unknown")],
)
def test_report_rejects_balanced_but_false_episode_status_counts(source, target):
    report = mixed_report()
    data = report.model_dump(mode="json")
    counts = data["eligibility_counts"]
    counts[source] -= 1
    counts[target] += 1
    counts["confirmed_eligible_share"] = counts["pass_count"] / counts["total"]
    with pytest.raises(ValidationError, match="eligibility_counts"):
        Report.model_validate(data)


def test_report_rejects_duplicate_episode_identity():
    data = mixed_report().model_dump(mode="json")
    data["episodes"][1]["id"] = data["episodes"][0]["id"]
    with pytest.raises(ValidationError, match="duplicate episode"):
        Report.model_validate(data)


def test_report_count_reconciliation_keeps_failed_and_unresolved_unknown():
    r = mixed_report()
    inventory = Inventory(
        loaded=3,
        failed=[{"id": "failed", "reason": "read error"}],
        unresolved=2,
        complete=False,
    )
    counts = counts_of([e.eligibility for e in r.episodes], inventory)
    data = r.model_dump(mode="json")
    # The fixture exercises the status boundary independently of optional ledgers.
    data["coverage"] = None
    data["inventory"] = inventory.model_dump(mode="json")
    data["eligibility_counts"] = counts.model_dump(mode="json")
    parsed = Report.model_validate(data)
    assert parsed.eligibility_counts.unknown == 3
    assert parsed.eligibility_counts.confirmed_eligible_share is None
    assert Report.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize("share", [0.99, float("nan")])
def test_report_rejects_false_eligible_share(share):
    data = mixed_report().model_dump(mode="json")
    data["eligibility_counts"]["confirmed_eligible_share"] = share
    with pytest.raises(ValidationError, match="share"):
        Report.model_validate(data)


def test_window_summary_must_match_the_recorded_statuses():
    r = windows(
        two_channels(), window_spec(), [finding(source_index=0, channel="value")]
    )
    data = r.model_dump(mode="json")
    data["measurements"]["counts"] = {
        "pass": 7,
        "blocked": 0,
        "review": 0,
        "unknown": 0,
    }
    with pytest.raises(ValidationError, match="window"):
        DiagnosticResult.model_validate(data)


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "invalid_status", "invalid_bounds", "absent_records"],
)
def test_window_evidence_requires_complete_unique_valid_records(change):
    data = windows(two_channels(), window_spec()).model_dump(mode="json")
    records = data["evidence"]["windows"]
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[1] = deepcopy(records[0])
    elif change == "invalid_status":
        records[0]["status"] = "approved"
    elif change == "invalid_bounds":
        records[0]["grid_end_exclusive"] = records[0]["grid_start"]
    else:
        del data["evidence"]["windows"]
    with pytest.raises(ValidationError, match="window"):
        DiagnosticResult.model_validate(data)


def test_budget_unknowns_reconcile_with_examined_records_on_roundtrip():
    r = windows(two_channels(), window_spec(max_windows=2))
    assert r.measurements["counts"] == {
        "pass": 2,
        "blocked": 0,
        "review": 0,
        "unknown": 5,
    }
    assert DiagnosticResult.model_validate_json(r.model_dump_json()) == r


def test_missing_extra_only_changes_eligibility_when_capability_required(monkeypatch):
    e = episode(signal())
    plan = DiagnosticPlan(
        motion=[
            MotionSpec(
                id="shape", channel=Selector(feature="state", index=0), max_gap_s=0.11
            )
        ]
    )
    original = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        """Simulate a plain installation without altering the environment."""
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("missing optional NumPy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    optional = audit([e], plan)
    required = audit(
        [e], plan, RequirementsSection(required_capabilities=["motion_shape"])
    )
    assert optional.eligibility_counts.pass_count == 1
    assert required.eligibility_counts.unknown == 1
    for report in (optional, required):
        assert report.diagnostics.results[0].availability.value == "unavailable"
        assert "kalanos[numeric]" in report.diagnostics.results[0].reason
