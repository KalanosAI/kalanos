"""Verifies scoring: interpolation, family and level rollups, grading."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import ast
from pathlib import Path

# External
import pytest

# Internal
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.policy import Band, MetricPolicy, Policy, ScoreMode
from kalanos.analysis.models.scoring import (
    Finding,
    FindingLocation,
    Grade,
    ScoreResult,
    Severity,
)
from kalanos.analysis.scoring.score import (
    grade_for,
    resolve_status,
    rollup,
    score_metrics,
    sort_findings,
)
from kalanos.assets.policy import load_default_policy


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


ANALYSIS_ROOT = Path(__file__).parent.parent / "src" / "kalanos" / "analysis"
SCORING_SOURCES = [
    ANALYSIS_ROOT / "scoring" / "score.py",
    ANALYSIS_ROOT / "models" / "scoring.py",
]

# mapping's structural-only pass leaves every stream typed like this
# until it consults dictionary.yaml to resolve a real taxonomy type.
_TAXONOMY_TYPE = "unmapped.tcp_pose"

# A location stand-in for tests that only care about a channel's score,
# not about which recording or channel raised it.
_LOCATION = FindingLocation(
    episode_id="episode_0",
    stream=_TAXONOMY_TYPE,
    instance="armA",
    channel="tcp_pose_x",
)

_LETTERS = {
    "A": 90.0,
    "B": 80.0,
    "C": 70.0,
    "D": 60.0,
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _report_only(value: float, *, unit: str = "fraction") -> MetricResult:
    """Build a report_only MetricResult, the shape `metrics` hands to scoring.

    Parameters
    ----------
    value : float
        The computed value `metrics` produced.
    unit : str
        The value's unit.

    Returns
    -------
    MetricResult
        `report_only`, carrying the `ungraded_reason` `metrics` itself writes.
    """

    return MetricResult(
        value=value,
        unit=unit,
        status=MetricStatus.REPORT_ONLY,
        evidence={"ungraded_reason": "grading needs policy thresholds"},
    )


def _not_applicable(reason: str = "sampling is not regular") -> MetricResult:
    """Build a not_applicable MetricResult, the shape an unmet Requires produces.

    Parameters
    ----------
    reason : str
        The reason the registry declined to run the metric.

    Returns
    -------
    MetricResult
        `not_applicable`, with `reason` in its evidence.
    """

    return MetricResult(
        value=None,
        unit=None,
        status=MetricStatus.NOT_APPLICABLE,
        evidence={"reason": reason},
    )


def _score(
    policy: Policy,
    level: Level,
    score: float | None,
    *,
    families: dict[str, float] | None = None,
) -> ScoreResult:
    """Build a ScoreResult with just enough fields set to feed `rollup`.

    Parameters
    ----------
    policy : Policy
        The policy `grade_for` reads its letters table from.
    level : Level
        The level this stand-in result attaches to.
    score : float or None
        The score to carry; `grade` and `train_ready` follow from it the
        same way `score_metrics` and `rollup` derive them.
    families : dict[str, float] or None
        Per-family scores to carry, defaulting to none.

    Returns
    -------
    ScoreResult
        A result usable as one of `rollup`'s children.
    """

    return ScoreResult(
        level=level,
        score=score,
        grade=grade_for(score, policy) if score is not None else None,
        train_ready=score >= 70.0 if score is not None else None,
        n_contributing=1 if score is not None else 0,
        families=families or {},
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_grade_for_matches_every_boundary_in_the_documented_table():
    """Verify each grade band's own edge, per docs/METRICS.md's scoring table."""

    policy = load_default_policy()

    # fmt: off
    assert grade_for(100.0,     policy) == Grade.A
    assert grade_for(90.0,      policy) == Grade.A
    assert grade_for(89.999,    policy) == Grade.B
    assert grade_for(80.0,      policy) == Grade.B
    assert grade_for(79.999,    policy) == Grade.C
    assert grade_for(70.0,      policy) == Grade.C
    assert grade_for(69.999,    policy) == Grade.D
    assert grade_for(60.0,      policy) == Grade.D
    assert grade_for(59.999,    policy) == Grade.F
    assert grade_for(0.0,       policy) == Grade.F
    # fmt: on


def test_a_shifted_letters_table_grades_the_same_score_differently():
    """Verify grade_for reads its bands from the policy, not a module constant."""

    default = load_default_policy()
    shifted = Policy(
        schema_version=1,
        metrics={},
        letters={"A": 95.0, "B": 90.0, "C": 80.0, "D": 70.0},
    )

    assert grade_for(85.0, default) == Grade.B
    assert grade_for(85.0, shifted) == Grade.C


def test_a_not_applicable_result_passes_through_resolve_status_untouched():
    """Verify a metric that could not run is never handed to a threshold."""

    result = _not_applicable("fewer than two valid timestamps")
    policy = load_default_policy()

    resolved = resolve_status(
        result, metric_name="drop_rate", taxonomy_type=_TAXONOMY_TYPE, policy=policy
    )

    assert resolved == result


def test_drop_rate_scores_100_at_exactly_its_good_bound():
    """Verify a value at or better than good scores full points, per the policy."""

    policy = load_default_policy()

    resolved = resolve_status(
        _report_only(0.01),
        metric_name="drop_rate",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.GOOD
    assert "ungraded_reason" not in resolved.evidence


def test_drop_rate_scores_0_at_exactly_its_bad_bound():
    """Verify a value at or worse than bad scores no points."""

    policy = load_default_policy()

    resolved = resolve_status(
        _report_only(0.05),
        metric_name="drop_rate",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.CRITICAL


def test_drop_rate_interpolates_to_the_midpoint_between_its_bounds():
    """Verify a value halfway between good and bad scores halfway between 0 and 100."""

    policy = load_default_policy()

    resolved = resolve_status(
        _report_only(0.03),
        metric_name="drop_rate",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.WARNING


def test_a_higher_is_better_band_scores_from_the_opposite_direction():
    """Verify snr_db's shape: a larger value is the good one, not the small one."""

    policy = Policy(
        schema_version=1,
        metrics={
            "integrity.snr_db": MetricPolicy(
                higher_is_better=True,
                thresholds={"default": Band(good=20.0, bad=6.0)},
            )
        },
        letters=_LETTERS,
    )

    at_good = resolve_status(
        _report_only(20.0, unit="dB"),
        metric_name="snr_db",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )
    at_bad = resolve_status(
        _report_only(6.0, unit="dB"),
        metric_name="snr_db",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )
    at_midpoint = resolve_status(
        _report_only(13.0, unit="dB"),
        metric_name="snr_db",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert at_good.status == MetricStatus.GOOD
    assert at_bad.status == MetricStatus.CRITICAL
    assert at_midpoint.status == MetricStatus.WARNING


def test_abs_dev_scores_full_points_when_the_value_equals_its_declared_target():
    """Verify effective_hz-shaped grading: a value at its nominal rate scores 100."""

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.effective_hz": MetricPolicy(
                mode=ScoreMode.ABS_DEV,
                target="nominal_hz",
                thresholds={"default": Band(good=0.02, bad=0.25)},
                limits={"nominal_hz": 100.0},
            )
        },
        letters=_LETTERS,
    )

    resolved = resolve_status(
        _report_only(100.0, unit="Hz"),
        metric_name="effective_hz",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.GOOD


def test_abs_dev_scores_no_points_once_deviation_reaches_the_bad_bound():
    """Verify a rate 25% off its nominal, the policy's own bad bound, scores 0."""

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.effective_hz": MetricPolicy(
                mode=ScoreMode.ABS_DEV,
                target="nominal_hz",
                thresholds={"default": Band(good=0.02, bad=0.25)},
                limits={"nominal_hz": 100.0},
            )
        },
        letters=_LETTERS,
    )

    resolved = resolve_status(
        _report_only(125.0, unit="Hz"),
        metric_name="effective_hz",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.CRITICAL


def test_abs_dev_with_no_target_declared_under_limits_stays_report_only():
    """Verify effective_hz stays ungraded until a deployment declares nominal_hz.

    This is docs/METRICS.md's own "ungraded where the policy declares no
    nominal rate" — the band and target are decided, but no deployment has
    supplied the number the deviation is measured against.
    """

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.effective_hz": MetricPolicy(
                mode=ScoreMode.ABS_DEV,
                target="nominal_hz",
                thresholds={"default": Band(good=0.02, bad=0.25)},
            )
        },
        letters=_LETTERS,
    )

    resolved = resolve_status(
        _report_only(100.0, unit="Hz"),
        metric_name="effective_hz",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.REPORT_ONLY
    assert "nominal_hz" in resolved.evidence["ungraded_reason"]


def test_dt_jitter_and_effective_hz_stay_report_only_with_a_fresh_reason():
    """Verify the two undecided timing metrics never grade in the default policy.

    Each carries a metrics-stage `ungraded_reason` on the way in;
    the resolved result must carry scoring's own reason instead,
    not the stale one `metrics` wrote before a policy was in the picture.
    """

    policy = load_default_policy()

    for name in ("dt_jitter_ms", "effective_hz"):
        resolved = resolve_status(
            _report_only(1.5, unit="ms"),
            metric_name=name,
            taxonomy_type=_TAXONOMY_TYPE,
            policy=policy,
        )
        assert resolved.status == MetricStatus.REPORT_ONLY
        assert resolved.evidence["ungraded_reason"] != "grading needs policy thresholds"


def test_a_metric_absent_from_the_policy_stays_report_only():
    """Verify a metric the policy never mentions fails safe rather than raising."""

    policy = load_default_policy()

    resolved = resolve_status(
        _report_only(1.0),
        metric_name="not_a_real_metric",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.REPORT_ONLY
    assert "not_a_real_metric" in resolved.evidence["ungraded_reason"]


def test_a_metric_with_no_band_for_the_type_or_default_stays_report_only():
    """Verify a type matching neither its own key nor 'default' fails safe."""

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.drop_rate": MetricPolicy(
                thresholds={"proprio.joint_velocity": Band(good=0.01, bad=0.05)},
            )
        },
        letters=_LETTERS,
    )

    resolved = resolve_status(
        _report_only(0.02),
        metric_name="drop_rate",
        taxonomy_type="proprio.joint_torque",
        policy=policy,
    )

    assert resolved.status == MetricStatus.REPORT_ONLY


def test_a_missing_value_on_an_applicable_result_becomes_not_applicable():
    """Verify a malformed upstream result fails safe rather than crashing on None."""

    broken = MetricResult(value=None, unit=None, status=MetricStatus.REPORT_ONLY)
    policy = load_default_policy()

    resolved = resolve_status(
        broken, metric_name="drop_rate", taxonomy_type=_TAXONOMY_TYPE, policy=policy
    )

    assert resolved.status == MetricStatus.NOT_APPLICABLE


def test_a_channel_graded_only_on_drop_rate_scores_from_that_alone():
    """Verify report_only and not_applicable siblings never enter the average."""

    policy = load_default_policy()
    results = {
        "drop_rate": _report_only(0.005),
        "dt_jitter_ms": _report_only(1.2, unit="ms"),
        "effective_hz": _not_applicable("fewer than two valid timestamps"),
    }

    graded, score, findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert graded["drop_rate"].status == MetricStatus.GOOD
    assert findings == []
    assert score.level == Level.CHANNEL
    assert score.score == pytest.approx(100.0)
    assert score.grade == Grade.A
    assert score.train_ready is True
    assert score.n_contributing == 1


def test_a_channel_with_nothing_graded_scores_none_rather_than_zero():
    """Verify a report_only or not_applicable metric never costs a channel points.

    The denominator excludes it entirely, rather than counting it as a zero.
    """

    policy = load_default_policy()
    results = {
        "dt_jitter_ms": _report_only(1.2, unit="ms"),
        "effective_hz": _not_applicable("fewer than two valid timestamps"),
    }

    _, score, _findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert score.score is None
    assert score.grade is None
    assert score.train_ready is None
    assert score.n_contributing == 0


def test_a_critical_drop_rate_alone_scores_the_channel_zero():
    """Verify a single graded metric decides the whole channel's score."""

    policy = load_default_policy()
    results = {"drop_rate": _report_only(0.20)}

    _, score, findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert score.score == pytest.approx(0.0)
    assert score.grade == Grade.F
    assert score.train_ready is False
    [finding] = findings
    assert finding.severity == Severity.CRITICAL
    assert finding.metric_id == "timing.drop_rate"
    assert finding.points == pytest.approx(0.0)
    assert finding.episode_id == _LOCATION.episode_id
    assert finding.instance == _LOCATION.instance
    assert finding.channel == _LOCATION.channel
    assert finding.stream == _TAXONOMY_TYPE


def test_a_drop_rate_at_the_midpoint_scores_the_channel_fifty():
    """Verify a warning-status metric costs points proportional to its distance
    past threshold, rather than the old fixed 50-point placeholder.
    """

    policy = load_default_policy()
    results = {"drop_rate": _report_only(0.03)}

    _, score, findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert score.score == pytest.approx(50.0)
    assert score.grade == Grade.F
    assert score.train_ready is False
    [finding] = findings
    assert finding.severity == Severity.WARNING
    assert finding.points == pytest.approx(50.0)


def test_a_good_result_raises_no_finding():
    """Verify a passing metric produces a score but no finding.

    Findings exist to report a defect — a good result is not one, and
    should not show up in a list a reader scans for what is wrong.
    """

    policy = load_default_policy()
    results = {"drop_rate": _report_only(0.005)}

    _, _score, findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert findings == []


def test_a_not_applicable_result_raises_no_finding():
    """Verify an unmeasurable metric produces no finding, per its own rule.

    `not_applicable` means "this could not be measured", which is a
    different fact from "this measured badly" — the second is a defect,
    the first is not.
    """

    policy = load_default_policy()
    results = {"effective_hz": _not_applicable()}

    _, _score, findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert findings == []


def test_a_report_only_result_raises_no_finding():
    """Verify a metric that never grades raises no finding either.

    `p99_torque` and similar metrics are report_only by policy design —
    always shown, never graded — so there is no verdict to turn into one.
    """

    policy = Policy(
        schema_version=1,
        metrics={"timing.dt_jitter_ms": MetricPolicy(report_only=True)},
        letters=_LETTERS,
    )
    results = {"dt_jitter_ms": _report_only(1.2, unit="ms")}

    _, _score, findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert findings == []


def test_an_episode_level_finding_names_no_stream():
    """Verify a finding raised above stream level carries no stream or channel.

    `taxonomy_type` still has to reach `_resolve` as `Level.EPISODE.value`
    for the band lookup to succeed, even though the location it is graded
    against names no stream.
    """

    policy = load_default_policy()
    results = {"drop_rate": _report_only(0.20)}

    _, score, findings = score_metrics(
        results,
        level=Level.EPISODE,
        taxonomy_type=Level.EPISODE.value,
        policy=policy,
        location=FindingLocation(episode_id="episode_0"),
    )

    assert score.n_contributing == 1
    [finding] = findings
    assert finding.stream is None
    assert finding.channel is None
    assert finding.points == pytest.approx(0.0)


def test_sort_findings_puts_critical_before_warning():
    """Verify severity is the primary sort key, so the worst thing sorts first."""

    warning = Finding(
        metric_id="timing.drop_rate",
        family="timing",
        severity=Severity.WARNING,
        value=0.03,
        unit="fraction",
        points=50.0,
        episode_id="episode_0",
        stream=_TAXONOMY_TYPE,
        evidence={},
    )
    critical = warning.model_copy(update={"severity": Severity.CRITICAL, "points": 0.0})

    ordered = sort_findings([warning, critical])

    assert ordered == [critical, warning]


def test_sort_findings_breaks_ties_by_ascending_points():
    """Verify that among same-severity findings, the lower score sorts first.

    Two critical findings both scored 0, so a real tiebreaker is needed:
    a critical one point off the bad bound is a smaller miss than one an
    order of magnitude past it, and the worse one should lead.
    """

    worse = Finding(
        metric_id="timing.drop_rate",
        family="timing",
        severity=Severity.CRITICAL,
        value=0.20,
        unit="fraction",
        points=0.0,
        episode_id="episode_0",
        stream=_TAXONOMY_TYPE,
        evidence={},
    )
    less_bad = worse.model_copy(update={"points": 20.0})

    ordered = sort_findings([less_bad, worse])

    assert ordered == [worse, less_bad]


def test_mixing_a_decided_weight_with_an_undecided_one_refuses_to_guess():
    """Verify a channel cannot silently fix a ratio nobody chose.

    `drop_rate` carries an explicit weight; `dt_jitter_ms` here is graded
    with none. Falling back to equal weighting for the second metric alone
    would decide a ratio the policy never actually set.
    """

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.drop_rate": MetricPolicy(
                weight=3.0, thresholds={"default": Band(good=0.01, bad=0.05)}
            ),
            "timing.dt_jitter_ms": MetricPolicy(
                thresholds={"default": Band(good=2.0, bad=5.0)}
            ),
        },
        letters=_LETTERS,
    )
    results = {
        "drop_rate": _report_only(0.005),
        "dt_jitter_ms": _report_only(1.0, unit="ms"),
    }

    with pytest.raises(
        ValueError, match="mixes a decided weight with an undecided one"
    ):
        score_metrics(
            results,
            level=Level.CHANNEL,
            taxonomy_type=_TAXONOMY_TYPE,
            policy=policy,
            location=_LOCATION,
        )


def test_mixing_a_decided_family_weight_with_an_undecided_one_refuses_to_guess():
    """Verify the same refusal applies one level up, across families."""

    policy = Policy(
        schema_version=1,
        metrics={
            "family_a.metric_x": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
            "family_b.metric_y": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
        },
        family_weights={"family_a": 2.0},
        letters=_LETTERS,
    )
    results = {"metric_x": _report_only(2.0), "metric_y": _report_only(2.0)}

    with pytest.raises(
        ValueError, match="mixes a decided weight with an undecided one"
    ):
        score_metrics(
            results,
            level=Level.CHANNEL,
            taxonomy_type=_TAXONOMY_TYPE,
            policy=policy,
            location=_LOCATION,
        )


def test_a_negative_weight_is_rejected_when_the_policy_loads():
    """Verify MetricPolicy refuses a weight that could never be a real share."""

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        MetricPolicy(weight=-1.0, thresholds={"default": Band(good=0.01, bad=0.05)})


def test_families_the_recording_never_produced_are_skipped_not_penalised():
    """Verify a declared-but-absent family renormalises away, per missing_family: skip.

    A recording with no cameras loses nothing for having no vision metrics —
    the same score results whether the policy declares that family or not.
    """

    metrics = {
        "family_a.metric_x": MetricPolicy(
            thresholds={"default": Band(good=0.0, bad=10.0)}
        ),
        "family_b.metric_y": MetricPolicy(
            thresholds={"default": Band(good=0.0, bad=10.0)}
        ),
    }
    results = {"metric_x": _report_only(2.0), "metric_y": _report_only(8.0)}

    without_absent_family = Policy(
        schema_version=1,
        metrics=metrics,
        family_weights={"family_a": 1.0, "family_b": 1.0},
        letters=_LETTERS,
    )
    with_absent_family = Policy(
        schema_version=1,
        metrics=metrics,
        family_weights={"family_a": 1.0, "family_b": 1.0, "family_c": 5.0},
        letters=_LETTERS,
    )

    _, score_a, _findings_a = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=without_absent_family,
        location=_LOCATION,
    )
    _, score_b, _findings_b = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=with_absent_family,
        location=_LOCATION,
    )

    assert score_a.score == pytest.approx(50.0)
    assert score_b.score == pytest.approx(score_a.score)


def test_score_metrics_carries_each_families_score_on_the_result():
    """Verify a channel's per-family scores survive onto its ScoreResult.

    `family_scores` was already computed to fold into the node's own score;
    this checks the per-family numbers themselves reach the caller too,
    keyed by family, rather than being discarded once folded together.
    """

    metrics = {
        "family_a.metric_x": MetricPolicy(
            thresholds={"default": Band(good=0.0, bad=10.0)}
        ),
        "family_b.metric_y": MetricPolicy(
            thresholds={"default": Band(good=0.0, bad=10.0)}
        ),
    }
    policy = Policy(
        schema_version=1,
        metrics=metrics,
        family_weights={"family_a": 1.0, "family_b": 1.0},
        letters=_LETTERS,
    )
    results = {"metric_x": _report_only(2.0), "metric_y": _report_only(8.0)}

    _, score, _findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert score.families == {
        "family_a": pytest.approx(80.0),
        "family_b": pytest.approx(20.0),
    }


def test_score_metrics_family_scores_are_unaffected_by_the_fail_penalty():
    """Verify the fail penalty lowers the node's total but not its family scores.

    The penalty applies once, to the rolled-up node total, per docs/METRICS.md.
    A per-family score is what that one family actually measured, and should
    read the same whether or not some other family also failed.
    """

    metrics = {
        "family_a.metric_fail": MetricPolicy(
            thresholds={"default": Band(good=0.0, bad=10.0)}
        ),
    }
    policy = Policy(
        schema_version=1,
        metrics=metrics,
        fail_penalty=5.0,
        letters=_LETTERS,
    )
    results = {"metric_fail": _report_only(10.0)}

    _, score, _findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    assert score.families == {"family_a": pytest.approx(0.0)}
    assert score.score == pytest.approx(0.0)


def test_two_failing_metrics_subtract_twice_the_fail_penalty():
    """Verify each distinct metric scoring 0 costs its own penalty."""

    policy = Policy(
        schema_version=1,
        metrics={
            "family_a.metric_pass": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
            "family_a.metric_fail1": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
            "family_a.metric_fail2": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
        },
        fail_penalty=5.0,
        letters=_LETTERS,
    )
    results = {
        "metric_pass": _report_only(0.0),
        "metric_fail1": _report_only(10.0),
        "metric_fail2": _report_only(10.0),
    }

    _, score, _findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    # One metric at 100 points, two at 0: (100 + 0 + 0) / 3, minus two penalties.
    assert score.score == pytest.approx(100.0 / 3.0 - 10.0)


def test_the_fail_penalty_clamps_at_its_cap():
    """Verify enough failures stop costing more once the cap is reached."""

    policy = Policy(
        schema_version=1,
        metrics={
            "family_a.metric_pass": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
            "family_a.metric_fail1": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
            "family_a.metric_fail2": MetricPolicy(
                thresholds={"default": Band(good=0.0, bad=10.0)}
            ),
        },
        fail_penalty=50.0,
        fail_penalty_cap=10.0,
        letters=_LETTERS,
    )
    results = {
        "metric_pass": _report_only(0.0),
        "metric_fail1": _report_only(10.0),
        "metric_fail2": _report_only(10.0),
    }

    _, score, _findings = score_metrics(
        results,
        level=Level.CHANNEL,
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
        location=_LOCATION,
    )

    # Uncapped, two failures at 50 each would subtract 100 and floor the
    # score at 0; the cap holds the penalty to 10, so the result is only
    # reachable if the cap is actually being applied rather than ignored.
    assert score.score == pytest.approx(100.0 / 3.0 - 10.0)


def test_rollup_averages_scored_children_and_excludes_none_ones():
    """Verify a child with no score of its own does not drag the average toward zero."""

    policy = load_default_policy()
    children = [
        _score(policy, Level.CHANNEL, 100.0),
        _score(policy, Level.CHANNEL, 60.0),
        _score(policy, Level.CHANNEL, None),
    ]

    stream_score = rollup(Level.STREAM, children, policy=policy)

    assert stream_score.level == Level.STREAM
    assert stream_score.score == pytest.approx(80.0)
    assert stream_score.grade == Grade.B
    assert stream_score.n_contributing == 2


def test_rollup_of_nothing_but_none_children_stays_none():
    """Verify a level with nothing graded beneath it stays silent, not zero."""

    policy = load_default_policy()
    children = [
        _score(policy, Level.CHANNEL, None),
        _score(policy, Level.CHANNEL, None),
    ]

    stream_score = rollup(Level.STREAM, children, policy=policy)

    assert stream_score.score is None
    assert stream_score.grade is None
    assert stream_score.train_ready is None


def test_rollup_averages_a_family_score_across_children_that_measured_it():
    """Verify rollup folds each family the same equal-weight way it folds score."""

    policy = load_default_policy()
    children = [
        _score(policy, Level.CHANNEL, 100.0, families={"timing": 100.0}),
        _score(policy, Level.CHANNEL, 60.0, families={"timing": 60.0}),
    ]

    stream_score = rollup(Level.STREAM, children, policy=policy)

    assert stream_score.families == {"timing": pytest.approx(80.0)}


def test_rollup_of_a_missing_family_averages_only_over_children_that_measured_it():
    """Verify a family absent from one child does not drag its average down.

    One channel measured only `timing`, the other only `vision`; each family's
    rolled-up score should reflect only the children that actually measured it,
    the same "missing family: skip" rule `score_metrics` already applies.
    """

    policy = load_default_policy()
    children = [
        _score(policy, Level.CHANNEL, 100.0, families={"timing": 100.0}),
        _score(policy, Level.CHANNEL, 40.0, families={"vision": 40.0}),
    ]

    stream_score = rollup(Level.STREAM, children, policy=policy)

    assert stream_score.families == {
        "timing": pytest.approx(100.0),
        "vision": pytest.approx(40.0),
    }


def test_rollup_composes_family_scores_unchanged_from_stream_through_dataset():
    """Verify a family score chains all the way from Stream to Dataset unchanged.

    Mirrors test_rollup_composes_unchanged_from_stream_through_dataset for
    the family-score channel, since rollup folds both the same way.
    """

    policy = load_default_policy()
    channels = [
        _score(policy, Level.CHANNEL, 100.0, families={"timing": 100.0}),
        _score(policy, Level.CHANNEL, 40.0, families={"timing": 40.0}),
    ]
    stream = rollup(Level.STREAM, channels, policy=policy)
    episode = rollup(Level.EPISODE, [stream], policy=policy)
    dataset = rollup(Level.DATASET, [episode], policy=policy)

    for result in (stream, episode, dataset):
        assert result.families == {"timing": pytest.approx(70.0)}


def test_rollup_composes_unchanged_from_stream_through_dataset():
    """Verify the same function chains all the way from Stream to Dataset.

    Two channels score 100 and 40; their stream averages to 70, which is
    this episode's only stream and this dataset's only episode, so the same
    70 should surface unchanged at every level above it.
    """

    policy = load_default_policy()
    channels = [
        _score(policy, Level.CHANNEL, 100.0),
        _score(policy, Level.CHANNEL, 40.0),
    ]
    stream = rollup(Level.STREAM, channels, policy=policy)
    episode = rollup(Level.EPISODE, [stream], policy=policy)
    dataset = rollup(Level.DATASET, [episode], policy=policy)

    for result, level in (
        (stream, Level.STREAM),
        (episode, Level.EPISODE),
        (dataset, Level.DATASET),
    ):
        assert result.level == level
        assert result.score == pytest.approx(70.0)
        assert result.grade == Grade.C
        assert result.train_ready is True


def test_flatline_pct_grades_a_mystery_channel_as_critical():
    """Verify the default band fails a fully flatlined, unmapped channel loudly.

    `integrity` gates on nothing but the value, so a mystery column at
    100% flatline must resolve to a graded failure, not stay ungraded.
    """

    policy = load_default_policy()

    resolved = resolve_status(
        _report_only(100.0, unit="%"),
        metric_name="flatline_pct",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.CRITICAL
    assert "ungraded_reason" not in resolved.evidence


@pytest.mark.parametrize(
    "taxonomy_type",
    [
        "reward.frame_reward",
        "reward.success_label",
        "reward.value_score",
        "reward.discount_flag",
    ],
)
def test_flatline_pct_stays_report_only_for_a_reward_channel(taxonomy_type: str):
    """Verify the reward exemption's null band neutralises flatline_pct.

    A reward channel flatlines legitimately until success, so its band
    is `None`/`None` rather than the default, and grading abstains —
    for all four numeric reward types the policy exempts, not just one.
    """

    policy = load_default_policy()

    resolved = resolve_status(
        _report_only(100.0, unit="%"),
        metric_name="flatline_pct",
        taxonomy_type=taxonomy_type,
        policy=policy,
    )

    assert resolved.status == MetricStatus.REPORT_ONLY


def test_drift_stays_report_only_by_policy_design():
    """Verify drift's ungraded reason names its design, not a missing band.

    Report-only by design and report-only for want of a band are
    different facts, and the report should keep them distinguishable.
    """

    policy = load_default_policy()

    resolved = resolve_status(
        _report_only(1.0, unit="unit/min"),
        metric_name="drift",
        taxonomy_type=_TAXONOMY_TYPE,
        policy=policy,
    )

    assert resolved.status == MetricStatus.REPORT_ONLY
    assert resolved.evidence["ungraded_reason"] == "report_only by policy design"


def test_no_metric_threshold_value_is_hardcoded_in_scoring_source():
    """Verify every policy-configured band lives only in policy.yaml.

    Walks the scoring package's own numeric literals and checks none of
    them match a band value the default policy actually carries. The
    grade table is settled, system-wide machinery read from the policy
    itself, not a per-metric domain threshold — this rules out any
    metric's band leaking back into code as a copy-pasted number.
    """

    policy = load_default_policy()
    band_values = {
        value
        for metric_policy in policy.metrics.values()
        for band in metric_policy.thresholds.values()
        for value in (band.good, band.bad)
        if value is not None
    }

    # `_weighted_mean`'s unweighted-contribution default and
    # `score_metrics`'s failing-metric increment are both a literal `1`,
    # generic scoring machinery that happens to coincide with a real band
    # value below. Skip their bodies rather than the value itself, so a
    # genuine hardcoded `1.0` anywhere else in the package still trips this.
    _GENERIC_MACHINERY_FUNCTIONS = {"_weighted_mean", "score_metrics"}

    def _constants(node: ast.AST) -> set[int | float]:
        found: set[int | float] = set()
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.FunctionDef)
                and child.name in _GENERIC_MACHINERY_FUNCTIONS
            ):
                continue
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, int | float)
                and not isinstance(child.value, bool)
            ):
                found.add(child.value)
            found |= _constants(child)
        return found

    literals: set[int | float] = set()
    for source in SCORING_SOURCES:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        literals |= _constants(tree)

    assert not (literals & band_values)
