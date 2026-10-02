"""Controls for explicit 0.7.0 diagnostics and their decision boundaries."""

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from upath import UPath

from kalanos.analysis.diagnostics.common import Unavailable
from kalanos.analysis.diagnostics.dataset import cohort
from kalanos.analysis.diagnostics.runner import execute
from kalanos.analysis.diagnostics.signals import motion, timing, tracking
from kalanos.analysis.diagnostics.validation import summarize_study
from kalanos.analysis.diagnostics.windows import windows
from kalanos.analysis.models.binding import (
    Bundle,
    ChannelBinding,
    EvaluationScope,
    RequirementsSection,
    Validation,
)
from kalanos.analysis.models.diagnostics import (
    ClockRelation,
    CohortSpec,
    DiagnosticPlan,
    DiagnosticReviewPolicy,
    MotionSpec,
    Selector,
    TimingSpec,
    TrackingSpec,
    VisionSpec,
    WindowSpec,
)
from kalanos.analysis.models.domain import (
    Channel,
    ClockInfo,
    ClockOrigin,
    Episode,
    FramePayload,
    SourceOrder,
    Stream,
)
from kalanos.analysis.models.report import AnalysedEpisode, Report
from kalanos.analysis.reporting.assemble import assemble_report
from kalanos.analysis.reporting.render import render_html
from kalanos.assets.policy import load_default_policy
from kalanos.cli import app, failing_statuses, parse_fail_on
from kalanos.testing import clean_recording
from kalanos.testing.injectors import SyntheticFrames


def signal(feature="state", values=None, times=None, command="none"):
    """Create an explicitly validated synthetic channel, never production evidence."""
    values = list(values if values is not None else np.linspace(0, 1, 101))
    times = list(times if times is not None else np.arange(len(values)) / 100)
    b = ChannelBinding(
        feature=feature,
        index=0,
        taxonomy_type="proprio.joint_position",
        quantity="position",
        representation="continuous",
        unit="rad",
        frame="joint",
        command=command,
        source_identity="fixture-session",
    )
    current = b.model_dump(mode="json")
    current["identity"] = f"{feature}[0]"
    b.validations = [
        Validation(
            property=k,
            value=current[k],
            scope="fixture-session",
            validator="test",
            evidence="synthetic only",
        )
        for k in ("identity", "quantity", "unit", "frame", "representation", "command")
    ]
    return Stream(
        taxonomy_type="proprio.joint_position",
        mapping_source="declared_names",
        kind="series",
        source_path=UPath("fixture.csv"),
        source_field=feature,
        timestamps=pl.Series(times),
        timestamp_dtype="float64",
        clock_info=ClockInfo(
            origin="capture",
            origin_evidence="producer",
            native_unit="s",
            domain="fixture",
        ),
        is_regular=True,
        payload=FramePayload(frame=pl.DataFrame({"value": values})),
        channels=[Channel(name="value", source_index=0, binding=b)],
    )


def relation(**updates):
    """Return a fixture-only reviewed relation between two source scopes."""
    return ClockRelation(
        left_scope="fixture-session",
        right_scope="fixture-session",
        evidence="fixture-only clock wiring",
        domain="fixture",
        **updates,
    )


def pair(**updates):
    """Return a basic stream-pair request."""
    return TimingSpec(
        id="pair",
        left=Selector(feature="command", index=0),
        right=Selector(feature="state", index=0),
        relation=relation(),
        tolerance_s=0.004,
        **updates,
    )


def episode(*streams, identifier="e"):
    """Build one canonical episode without passing through an adapter."""
    return Episode(id=identifier, streams=list(streams), tasks=["fixture task"])


def audit(
    episodes, plan=None, requirements=None, tier="standard", review=None, vision=None
):
    """Exercise the real assembly, coverage and single eligibility calculation."""
    policy = load_default_policy()
    if review:
        policy.diagnostic_reviews = review
    vision = vision or VisionSpec()
    return assemble_report(
        root=UPath("fixture"),
        analysed=[
            AnalysedEpisode(
                episode=e, adapter="csv", policy=policy, adapter_confidence=1
            )
            for e in episodes
        ],
        policy=policy,
        diagnostics_plan=plan,
        vision_samples=vision.sample_frames,
        full_frame_scan=vision.full_frame_scan,
        vision=vision,
        requirements=requirements,
        scope=EvaluationScope(requirements_id="test", policy_id="test", tier=tier),
    )


def window_spec(**updates):
    """Declare a ten-hertz, two-history plus two-future window contract."""
    return WindowSpec(
        id="train",
        anchor=Selector(feature="state", index=0),
        modalities=[
            {
                "selector": {"feature": "state", "index": 0},
                "max_age_s": 0.00001,
                "matching": "nearest",
            }
        ],
        sample_rate_hz=10,
        history_steps=2,
        prediction_steps=2,
        max_gap_s=0.11,
        **updates,
    )


def test_timing_offset_is_explicit_and_unmatched_rows_are_retained():
    a = signal("command")
    b = signal(times=np.arange(101) / 100 + 0.025)
    e = episode(a, b)
    spec = pair()
    r = timing(
        e, spec, DiagnosticReviewPolicy(timing_max_unmatched_fraction={"pair": 0.1})
    )
    assert r.measurements["unmatched_fraction"] == 1
    assert r.consequence == "review"
    spec.relation.offset_s = -0.025
    fixed = timing(e, spec)
    assert fixed.measurements["matched_samples"] == 101
    assert fixed.measurements["absolute_skew_p95_s"] < 1e-12
    assert b.timestamps[0] == 0.025


@pytest.mark.parametrize(
    "change",
    ["source_scope", "capture_origin", "unknown_units", "source_order", "clock_reset"],
)
def test_pair_abstains_when_clock_evidence_is_missing(change):
    a, b = signal("command"), signal()
    spec = pair()
    spec.relation.claim = "capture_alignment"
    if change == "source_scope":
        spec.relation.right_scope = "another-session"
    if change == "capture_origin":
        b.clock_info.origin = ClockOrigin.GENERATED
    if change == "unknown_units":
        b.native_timestamps = b.timestamps
        b.clock_info.native_unit = "unknown"
    if change == "source_order":
        b.source_order = SourceOrder(preserved=False)
    if change == "clock_reset":
        b.timestamps = b.timestamps.scatter([5], [0.0])
    with pytest.raises(Unavailable):
        timing(episode(a, b), spec)


def test_large_integer_epoch_preserves_small_cross_stream_differences():
    a, b = signal("command"), signal()
    for s, offset in [(a, 0), (b, 1)]:
        s.native_timestamps = pl.Series(
            [10**18 + i * 1000 + offset for i in range(101)], dtype=pl.Int64
        )
        s.clock_info.native_unit = "ns"
    spec = pair()
    spec.tolerance_s = 2e-9
    r = timing(episode(a, b), spec)
    assert r.measurements["signed_skew_median_s"] == pytest.approx(1e-9, abs=1e-15)


def test_event_fit_requires_explicit_correspondence_and_is_not_applied():
    a = signal("command")
    b = signal(times=np.arange(101) / 100 * 1.0001 + 0.02)
    spec = pair(events=[(0, 0), (50, 50), (100, 100)])
    fit = timing(episode(a, b), spec).measurements["event_fit"]
    assert fit["remaining_drift_ppm"] == pytest.approx(-100 / 1.0001)
    assert not fit["applied_to_data"]
    assert timing(episode(a, b), pair()).measurements["event_fit"] is None


def test_tracking_preserves_raw_error_and_declared_delay_comparison():
    times = np.arange(101) / 100
    command = signal("command", values=times, command="absolute")
    state = signal(values=times - 0.02)
    spec = TrackingSpec(
        **pair().model_dump(), semantics="absolute", response_delay_s=0.02
    )
    r = tracking(episode(command, state), spec)
    assert r.measurements["raw_rmse"] == pytest.approx(0.02)
    assert r.measurements["adjusted_rmse"] < 1e-12
    assert r.measurements["unmatched_samples"] == 2


@pytest.mark.parametrize(
    "prop", ["identity", "unit", "quantity", "frame", "representation", "command"]
)
def test_tracking_requires_each_scoped_property(prop):
    command = signal("command", command="absolute")
    command.channels[0].binding.validations = [
        v for v in command.channels[0].binding.validations if v.property != prop
    ]
    with pytest.raises(Unavailable):
        tracking(
            episode(command, signal()),
            TrackingSpec(**pair().model_dump(), semantics="absolute"),
        )


def test_delta_is_relative_to_state_at_command_time():
    values = np.arange(101) / 100
    command = signal("command", values=[0.02] * 101, command="delta")
    spec = TrackingSpec(**pair().model_dump(), semantics="delta", response_delay_s=0.02)
    r = tracking(episode(command, signal(values=values)), spec)
    assert r.measurements["adjusted_rmse"] < 1e-12
    spec.response_delay_s = 0
    with pytest.raises(Unavailable):
        tracking(episode(command, signal()), spec)


def test_tracking_review_enters_existing_gate_without_score_penalty_or_block():
    e = episode(
        signal("command", command="absolute"), signal(values=np.linspace(2, 3, 101))
    )
    plan = DiagnosticPlan(
        tracking=[TrackingSpec(**pair().model_dump(), semantics="absolute")]
    )
    base, new = (
        audit([e]),
        audit(
            [e],
            plan,
            review=DiagnosticReviewPolicy(tracking_max_abs_error={"pair": 0.1}),
        ),
    )
    assert new.eligibility_counts.review == 1 and new.eligibility_counts.blocked == 0
    assert new.score.score == base.score.score
    assert any(
        f.metric_id.startswith("diagnostics.tracking")
        and f.consequence.value == "review"
        for f in new.findings
    )
    assert failing_statuses(new, parse_fail_on("blocked,review,unknown")) == 1
    assert new.readiness.score is None


def test_dimensionless_motion_invariant_to_uniform_time_and_amplitude_scaling():
    x = np.sin(np.linspace(0, 2, 101))
    spec = MotionSpec(
        id="smooth", channel=Selector(feature="state", index=0), max_gap_s=0.1
    )
    values = []
    for duration, amplitude in [(1, 1), (3, 1), (1, 5)]:
        r = motion(
            episode(signal(values=x * amplitude, times=np.linspace(0, duration, 101))),
            spec,
        )
        values.append(r.measurements["segments"][0]["dimensionless_jerk"])
    assert values == pytest.approx([values[0]] * 3, rel=1e-8)


def test_stationary_motion_retains_evidence_without_infinite_smoothness():
    spec = MotionSpec(
        id="smooth", channel=Selector(feature="state", index=0), max_gap_s=0.1
    )
    r = motion(episode(signal(values=[0.0] * 101)), spec)
    assert r.measurements["segments"][0]["stationary"]
    assert r.measurements["segments"][0]["dimensionless_jerk"] is None


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_motion_segments_break_at_invalid_samples(bad):
    values = np.sin(np.linspace(0, 3, 101)).tolist()
    values[50] = bad
    r = motion(
        episode(signal(values=values)),
        MotionSpec(
            id="smooth", channel=Selector(feature="state", index=0), max_gap_s=0.1
        ),
    )
    assert len(r.measurements["segments"]) == 2
    assert r.measurements["examined_samples"] == 100


def test_motion_limits_require_evidence_and_localize_review():
    with pytest.raises(ValidationError):
        MotionSpec(
            id="m",
            channel=Selector(feature="state"),
            max_gap_s=0.1,
            max_abs_velocity=0.5,
        )
    r = motion(
        episode(signal()),
        MotionSpec(
            id="m",
            channel=Selector(feature="state", index=0),
            max_gap_s=0.1,
            max_abs_velocity=0.5,
            limits_evidence="fixture-only specification",
        ),
        DiagnosticReviewPolicy(review_motion_limits=True),
    )
    assert r.consequence == "review" and r.support.intervals


def test_windows_count_boundary_and_invalid_context():
    values = list(np.arange(10, dtype=float))
    values[4] = None
    r = windows(episode(signal(values=values, times=np.arange(10) / 10)), window_spec())
    assert r.measurements["candidate_windows"] == 7
    assert r.measurements["counts"] == {
        "pass": 3,
        "blocked": 4,
        "review": 0,
        "unknown": 0,
    }
    assert r.evidence["windows"][0]["consumed"][0]["source_rows"] == [0, 1, 2, 3]


def test_window_budgets_keep_unexamined_in_denominator():
    r = windows(
        episode(
            signal(values=list(np.arange(10, dtype=float)), times=np.arange(10) / 10)
        ),
        window_spec(max_windows=2),
    )
    assert r.availability.value == "skipped"
    assert r.measurements["counts"]["unknown"] == 5
    assert sum(r.measurements["counts"].values()) == 7


def test_window_non_anchor_modality_needs_relation():
    spec = window_spec()
    spec.modalities.append(
        spec.modalities[0].model_copy(
            update={"selector": Selector(feature="command", index=0)}
        )
    )
    with pytest.raises(Unavailable):
        windows(episode(signal(), signal("command")), spec)


def test_window_sufficiency_never_counts_review_episodes():
    e = episode(
        signal(values=list(np.arange(10, dtype=float)), times=np.arange(10) / 10)
    )
    req = RequirementsSection(min_pass_windows={"train": 8})
    r = audit([e], DiagnosticPlan(windows=[window_spec()]), req)
    assert r.sufficiency.checks[0].observed == 7
    assert r.sufficiency.status.value == "insufficient"
    req.min_pass_windows = {"absent": 1}
    assert audit([e], None, req).sufficiency.status.value == "unknown"


def camera(count=10):
    """Return a deterministic nonuniform static image sequence."""
    image = np.indices((16, 16)).sum(axis=0).astype(np.uint8) * 10
    rgb = np.stack([image] * 3, axis=2)
    return Stream(
        taxonomy_type="extero.camera_rgb",
        kind="video",
        mapping_source="declared_names",
        source_path=UPath("camera.mp4"),
        source_field="camera",
        timestamps=pl.Series(np.arange(count) / 10),
        timestamp_dtype="float64",
        payload=SyntheticFrames(frames=[rgb.copy() for _ in range(count)]),
    )


def camera_read(report):
    """The vision diagnostic result of a report graded with one camera."""
    return next(x for x in report.diagnostics.results if x.kind == "vision")


def test_sampled_video_does_not_certify_every_frame_or_block_static_scene():
    cam = camera()
    e = episode(signal(), cam)
    r = audit(
        [e],
        DiagnosticPlan(vision=True),
        RequirementsSection(required_capabilities=["video_quality"]),
        vision=VisionSpec(sample_frames=4),
    )
    assert r.coverage.decoded_frames_examined == 4
    assert r.eligibility_counts.unknown == 1
    assert camera_read(r).consequence == "report_only"
    sampled = audit(
        [e],
        DiagnosticPlan(vision=True),
        RequirementsSection(required_capabilities=["sampled_video_quality"]),
        vision=VisionSpec(sample_frames=4),
    )
    assert sampled.eligibility_counts.unknown == 0


def test_full_visual_evaluation_satisfies_required_capability():
    r = audit(
        [episode(signal(), camera())],
        DiagnosticPlan(vision=True),
        RequirementsSection(required_capabilities=["video_quality"]),
        vision=VisionSpec(sample_frames=10),
    )
    assert r.eligibility_counts.unknown == 0
    assert r.coverage.decoded_frames_examined == 10
    # A static scene repeats every frame, which stays a candidate, not a fault.
    vr = camera_read(r)
    assert vr.measurements["identical_adjacent_pairs"]
    assert vr.consequence == "report_only"


def test_dark_frame_threshold_is_configured_and_review_only():
    cam = camera()
    cam.payload.frames[3] = np.zeros((16, 16, 3), dtype=np.uint8)
    r = camera_read(
        audit(
            [episode(cam)],
            DiagnosticPlan(vision=True),
            review=DiagnosticReviewPolicy(vision_max_clipped_fraction=0.9),
            vision=VisionSpec(sample_frames=10),
        )
    )
    assert r.consequence == "review"
    assert any(x.start == 3 for x in r.support.intervals)


def test_video_budget_never_claims_complete_scan():
    # Moving actions make the frozen-frame read take every frame, past the budget.
    actions = clean_recording(
        hz=10.0, samples=10, taxonomy_type="action.joint_position_command"
    )
    r = camera_read(
        audit(
            [episode(camera(), actions)],
            DiagnosticPlan(vision=True),
            vision=VisionSpec(sample_frames=2, max_decode_frames=3),
        )
    )
    assert r.availability.value == "unavailable"
    assert r.measurements["segment_frames"] is None
    assert r.evidence["missing_rows"]
    assert not r.evidence["scan"]["complete"]


def cohort_spec():
    """Declare a two-recording comparable fixture cohort."""
    return CohortSpec(
        id="cohort",
        episode_ids=["a", "b"],
        robot_configuration="fixture robot",
        task="fixture task",
        evidence="fixture scope",
        features=[
            {
                "channel": {"feature": "state", "index": 0},
                "unit": "rad",
                "lower": 0,
                "upper": 1,
            }
        ],
        similarity_rmse=0.01,
    )


def test_dataset_runner_keeps_raw_and_passing_summaries_separate():
    spec = cohort_spec()
    episodes = [episode(signal(), identifier=x) for x in spec.episode_ids]
    r = cohort(episodes, spec, {"a": "pass", "b": "review"})
    assert r.measurements["raw_finite"]["episodes"] == 2
    assert r.measurements["passing_episodes_only"]["episodes"] == 1
    assert r.measurements["raw_finite"]["effective_dimensionality"] == pytest.approx(1)
    assert r.measurements["exact_feature_trajectory_duplicates"] == [["a", "b"]]
    assert r.measurements["phase_balance"]["fractions_of_labelled"] is None
    integrated = audit(episodes, DiagnosticPlan(cohorts=[spec]))
    assert integrated.diagnostics.results[0].kind == "cohort"


def test_dataset_invalid_samples_do_not_become_diversity():
    spec = cohort_spec()
    values = list(np.linspace(0, 1, 101))
    values[50] = float("inf")
    r = cohort(
        [
            episode(signal(), identifier="a"),
            episode(signal(values=values), identifier="b"),
        ],
        spec,
        {},
    )
    assert r.measurements["finite_source_rows"] == 201
    assert not r.measurements["similar_trajectories"]
    assert r.evidence["excluded_trajectories"][0]["invalid_rows"] == 1


def test_phase_balance_requires_reviewed_nonoverlapping_labels():
    data = cohort_spec().model_dump()
    data["phase_labels"] = [
        {
            "episode_id": "a",
            "phase": "approach",
            "start": 0,
            "end_exclusive": 10,
            "evidence": "reviewed",
        }
    ]
    spec = CohortSpec(**data)
    r = cohort(
        [episode(signal(), identifier="a"), episode(signal(), identifier="b")], spec, {}
    )
    assert r.measurements["phase_balance"]["labelled_samples"] == 10
    data["phase_labels"] *= 2
    with pytest.raises(ValidationError):
        CohortSpec(**data)


@pytest.mark.parametrize("kind", ["timing", "tracking", "motion", "windows"])
def test_metadata_tier_records_skipped_required_diagnostics(kind):
    specs = {
        "timing": pair(),
        "tracking": TrackingSpec(**pair().model_dump(), semantics="absolute"),
        "motion": MotionSpec(
            id="m", channel=Selector(feature="state", index=0), max_gap_s=0.1
        ),
        "windows": window_spec(),
    }
    spec = specs[kind]
    r = audit(
        [episode(signal("command", command="absolute"), signal())],
        DiagnosticPlan(**{kind: [spec]}),
        RequirementsSection(required_metrics=[f"diagnostics.{kind}.{spec.id}"]),
        tier="metadata",
    )
    assert r.diagnostics.results[0].availability.value == "skipped"
    assert r.eligibility_counts.unknown == 1


def test_operational_errors_are_retained_and_fail_with_exit_two(monkeypatch):
    def broken(*args):
        raise RuntimeError("test decoder failed")

    monkeypatch.setattr("kalanos.analysis.diagnostics.runner.timing", broken)
    r = audit([episode(signal("command"), signal())], DiagnosticPlan(timing=[pair()]))
    assert r.operational_errors
    monkeypatch.setattr("kalanos.cli.api.grade", lambda *a, **k: r)
    invoked = CliRunner().invoke(app, ["grade", "fixture.csv", "--json"])
    assert invoked.exit_code == 2
    assert r.eligibility_counts.unknown == 1


def test_report_roundtrip_html_and_inspect_preserve_diagnostic_evidence(tmp_path):
    r = audit([episode(signal("command"), signal())], DiagnosticPlan(timing=[pair()]))
    assert Report.model_validate_json(r.model_dump_json()) == r
    assert "DIAGNOSTIC EVIDENCE" in render_html(r)
    path = tmp_path / "report.json"
    path.write_text(r.model_dump_json())
    out = CliRunner().invoke(app, ["inspect", str(path)])
    assert out.exit_code == 0 and "matched_samples" in out.stdout
    data = r.model_dump(mode="json")
    data["diagnostics"]["plan_digest"] = "bad"
    with pytest.raises(ValidationError):
        Report.model_validate(data)


def test_plan_changes_execution_identity(tmp_path):
    from kalanos.assets.bundle import prepare_configuration

    path = UPath(tmp_path)
    a = prepare_configuration(path, bundle=Bundle(), sidecar=False)[2]
    b = prepare_configuration(
        path, bundle=Bundle(diagnostics=DiagnosticPlan(timing=[pair()])), sidecar=False
    )[2]
    assert a.execution_id.digest != b.execution_id.digest


def test_study_summary_never_creates_accepted_manifest_and_rejects_leakage():
    rows = [
        {
            "episode_id": "a",
            "session_id": "s1",
            "split": "validation",
            "truth": "valid",
            "decision": "blocked",
            "evidence": "reviewed",
        },
        {
            "episode_id": "b",
            "session_id": "s2",
            "split": "validation",
            "truth": "fault",
            "decision": "review",
            "evidence": "reviewed",
            "real_fault": True,
        },
    ]
    r = summarize_study(rows)
    assert r["false_blocks"] == 1 and r["detected_faults"] == 0
    assert not r["can_authorize_blocking"] and r["status"] == "draft"
    rows.append({**rows[0], "episode_id": "c", "split": "tuning"})
    with pytest.raises(ValueError, match="overlap"):
        summarize_study(rows)


def test_configuration_rejects_nonfinite_numbers_and_duplicate_ids():
    with pytest.raises(ValidationError):
        relation(offset_s=float("nan"))
    with pytest.raises(ValidationError):
        DiagnosticPlan(timing=[pair(), pair()])
    with pytest.raises(ValidationError):
        Bundle(diagnostics={"auto_align": True})


def test_nonfinite_computed_output_is_an_error():
    from kalanos.analysis.models.diagnostics import DiagnosticResult

    r = execute(
        "timing",
        pair(),
        episode(),
        lambda: DiagnosticResult(
            id="pair",
            kind="timing",
            availability="computed",
            measurements={"bad": float("inf")},
        ),
    )
    assert r.availability.value == "error"


def test_video_decoder_failure_preserves_already_examined_frames(monkeypatch):
    from kalanos.analysis.adapters.video import DecodeFailed

    gray_windows = SyntheticFrames.gray_windows

    def broken(self, windows, size, native=frozenset(), **caps):
        yield next(gray_windows(self, windows, size, native, **caps))
        raise DecodeFailed("corrupt packet")

    monkeypatch.setattr(SyntheticFrames, "gray_windows", broken)
    r = camera_read(audit([episode(camera())], DiagnosticPlan(vision=True)))
    assert r.availability.value == "error"
    assert r.measurements["examined_frames"] == 1
    assert r.evidence["frames"][0]["source_row"] == 0


def test_video_shape_change_requests_review_and_records_shapes():
    cam = camera()
    cam.payload.frames[2] = np.zeros((8, 8, 3), dtype=np.uint8)
    r = camera_read(
        audit(
            [episode(cam)],
            DiagnosticPlan(vision=True),
            review=DiagnosticReviewPolicy(review_video_integrity=True),
        )
    )
    assert r.measurements["distinct_image_shapes"] == 2
    assert r.consequence == "review"


def test_video_missing_decoder_is_unavailable_not_pass_or_operational_failure(
    monkeypatch,
):
    from kalanos.analysis.adapters.video import VideoPayload

    monkeypatch.setattr("kalanos.analysis.adapters.video.av", None)
    cam = camera()
    cam.payload = VideoPayload(UPath("camera.mp4"), 10, 0, 1)
    r = audit(
        [episode(signal(), cam)],
        DiagnosticPlan(vision=True),
        RequirementsSection(required_capabilities=["video_quality"]),
    )
    assert r.eligibility_counts.unknown == 1
    assert r.diagnostics.results[0].availability.value == "unavailable"
    assert not r.operational_errors


def test_session_bootstrap_is_reproducible_and_requires_declaration():
    spec = cohort_spec()
    spec.episode_ids = ["a", "b", "c"]
    spec.sessions = {"a": "s1", "b": "s2", "c": "s3"}
    episodes = [
        episode(signal(values=np.linspace(0, 0.3 * (i + 1), 101)), identifier=x)
        for i, x in enumerate(spec.episode_ids)
    ]
    assert (
        cohort(episodes, spec, {}).measurements["uncertainty"]["status"]
        == "unavailable"
    )
    spec.independent_sessions = True
    a, b = cohort(episodes, spec, {}), cohort(episodes, spec, {})
    assert a.measurements["uncertainty"] == b.measurements["uncertainty"]
    assert a.measurements["uncertainty"]["session_count"] == 3


def test_dataset_comparison_budget_is_explicit():
    spec = cohort_spec()
    spec.episode_ids = ["a", "b", "c"]
    spec.max_pair_comparisons = 1
    r = cohort([episode(signal(), identifier=x) for x in spec.episode_ids], spec, {})
    assert r.measurements["pair_comparisons"] == 1
    assert r.measurements["unexamined_pairs"] == 2


def test_configured_error_and_motion_period_controls():
    s = signal(values=np.sin(np.linspace(0, 2, 101)))
    b = s.channels[0].binding
    from kalanos.analysis.models.binding import Representation

    b.representation = Representation.ANGLE
    for v in b.validations:
        if v.property == "representation":
            v.value = "angle"
    spec = MotionSpec(id="m", channel=Selector(feature="state", index=0), max_gap_s=0.1)
    with pytest.raises(Unavailable, match="period"):
        motion(episode(s), spec)
    spec.angle_period = 2 * np.pi
    assert motion(episode(s), spec).availability.value == "computed"


def test_window_visual_sampling_leaves_unsampled_training_windows_unknown():
    cam = camera()
    s = signal(values=np.arange(10, dtype=float), times=np.arange(10) / 10)
    e = episode(s, cam)
    vr = camera_read(
        audit([e], DiagnosticPlan(vision=True), vision=VisionSpec(sample_frames=2))
    )
    spec = window_spec()
    from kalanos.analysis.models.diagnostics import ModalitySpec

    spec.modalities.append(
        ModalitySpec(
            selector=Selector(feature="camera"),
            relation=ClockRelation(
                left_scope="fixture-session",
                right_scope="camera.mp4",
                domain="fixture",
                evidence="fixture",
            ),
            max_age_s=0.001,
            matching="nearest",
            require_decoded_frames=True,
        )
    )
    r = windows(e, spec, visual=[vr])
    assert r.measurements["counts"]["unknown"] == 7
    assert r.measurements["counts"]["pass"] == 0


def test_diagnostic_comparison_lists_changes_and_rejects_changed_plan():
    from kalanos.analysis.compare import compare_reports

    e = episode(signal("command"), signal())
    first = audit([e], DiagnosticPlan(timing=[pair()]))
    spec = pair()
    spec.tolerance_s = 0.001
    second = audit([e], DiagnosticPlan(timing=[spec]))
    comparison = compare_reports(first, second)
    assert "diagnostics.plan_digest" in comparison.identity_changes
    assert comparison.diagnostic_changes


def test_camera_source_scope_survives_channel_free_binding_resolution():
    from kalanos.analysis.bindings import resolve_stream
    from kalanos.analysis.diagnostics.common import scope_of
    from kalanos.assets.dictionary import load_default_dictionary

    cam = resolve_stream(
        camera(), dictionary=load_default_dictionary(), source_identity="dataset-root"
    )
    assert cam.channels == [] and scope_of(cam) == "dataset-root"
    r = audit([episode(signal(), cam)], DiagnosticPlan(vision=True))
    assert (
        next(s for s in r.episodes[0].streams if s.kind == "video").source_identity
        == "dataset-root"
    )


def test_mixed_native_units_at_large_epoch_remain_precise():
    a, b = signal("command"), signal()
    a.native_timestamps = pl.Series(
        [10**18 + i * 1000 for i in range(101)], dtype=pl.Int64
    )
    b.native_timestamps = pl.Series([10**15 + i for i in range(101)], dtype=pl.Int64)
    a.clock_info.native_unit = "ns"
    b.clock_info.native_unit = "us"
    r = timing(episode(a, b), pair())
    assert r.measurements["absolute_skew_p95_s"] < 1e-18


def test_rate_tracking_uses_velocity_quantity_and_native_units():
    from kalanos.analysis.models.binding import Quantity

    a, b = signal("command", command="rate"), signal()
    for s in (a, b):
        binding = s.channels[0].binding
        binding.taxonomy_type = "proprio.joint_velocity"
        binding.quantity = Quantity.VELOCITY
        binding.unit = "rad/s"
        for v in binding.validations:
            if v.property == "quantity":
                v.value = "velocity"
            if v.property == "unit":
                v.value = "rad/s"
    spec = TrackingSpec(**pair().model_dump(), semantics="rate")
    assert tracking(episode(a, b), spec).measurements["raw_rmse"] == 0
    spec.semantics = "absolute"
    with pytest.raises(Unavailable):
        tracking(episode(a, b), spec)


def test_unmatched_review_rule_is_a_configuration_error():
    from kalanos.analysis.models.errors import MappingOverrideError

    with pytest.raises(MappingOverrideError, match="unknown check"):
        audit(
            [episode(signal())],
            DiagnosticPlan(),
            review=DiagnosticReviewPolicy(timing_max_unmatched_fraction={"typo": 0.1}),
        )


def test_review_threshold_changes_policy_identity_without_changing_plan(tmp_path):
    from kalanos.assets.bundle import prepare_configuration

    bundle = Bundle(diagnostics=DiagnosticPlan(timing=[pair()]))
    first = load_default_policy()
    second = first.model_copy(deep=True)
    second.diagnostic_reviews.timing_max_unmatched_fraction = {"pair": 0.1}
    a = prepare_configuration(
        UPath(tmp_path), bundle=bundle, policy=first, sidecar=False
    )[2]
    b = prepare_configuration(
        UPath(tmp_path), bundle=bundle, policy=second, sidecar=False
    )[2]
    assert a.policy_id.digest != b.policy_id.digest
    assert a.execution_id.digest == b.execution_id.digest


def test_counts_cannot_be_forged_on_report_loading():
    e = episode(
        signal(values=list(np.arange(10, dtype=float)), times=np.arange(10) / 10)
    )
    r = audit([e], DiagnosticPlan(windows=[window_spec()]))
    data = r.model_dump(mode="json")
    data["diagnostics"]["results"][0]["measurements"]["counts"]["pass"] += 1
    with pytest.raises(ValidationError, match="reconcile"):
        Report.model_validate(data)


def test_example_bundle_loads_through_cli():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "docs/examples/diagnostics.yaml"
    from kalanos.assets.bundle import load_bundle

    bundle = load_bundle(UPath(path))
    assert bundle.diagnostics.windows[0].id == "train"
    result = CliRunner().invoke(app, ["profiles", "validate", str(path)])
    assert result.exit_code == 0


def test_visual_previews_are_bounded_and_can_be_disabled():
    import base64
    import struct

    cam = camera()
    r = camera_read(
        audit(
            [episode(cam)],
            DiagnosticPlan(vision=True),
            vision=VisionSpec(preview_frames=2, preview_size=16),
        )
    )
    previews = [f for f in r.evidence["frames"] if "thumbnail_png_base64" in f]
    assert len(previews) == 2
    png = base64.b64decode(previews[0]["thumbnail_png_base64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (16, 16)
    off = camera_read(
        audit(
            [episode(cam)],
            DiagnosticPlan(vision=True),
            vision=VisionSpec(preview_frames=0),
        )
    )
    assert not any("thumbnail_png_base64" in f for f in off.evidence["frames"])
    page = render_html(audit([episode(signal(), cam)], DiagnosticPlan(vision=True)))
    assert "data:image/png;base64," in page


def test_tracking_preview_is_bounded_and_retains_command_response():
    from kalanos.analysis.diagnostics.previews import tracking_preview

    r = tracking(
        episode(signal("command", command="absolute"), signal()),
        TrackingSpec(**pair().model_dump(), semantics="absolute"),
    )
    preview = tracking_preview(r)
    assert preview["expected"] == preview["observed"]
    assert preview["samples"] <= 200
    assert r.evidence["samples"][0]["expected_response"] == 0
