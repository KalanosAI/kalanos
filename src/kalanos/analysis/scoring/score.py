"""Regrade against the policy, then roll the result up to a grade.

`metrics` cannot grade anything — it has no policy to read —
so every result it produces that carries a value comes back `report_only`,
with the reason parked in `evidence["ungraded_reason"]`.
This is where that gets resolved: each metric's value interpolates between
its policy's `good` and `bad` bound into a 0-100 score, metrics fold into a
family score by weight, families fold into a level score the same way,
and the result becomes a letter grade at every level from Channel up to Dataset.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections.abc import Sequence

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


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_TRAIN_READY_MINIMUM = 70.0  # "train_ready is score >= 70", docs/METRICS.md's own line.

# A metric that lands exactly on the bad bound scores 0 points,
# which this module treats as a failure for `fail_penalty` purposes.
_FAILING_POINTS = 0.0


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def grade_for(score: float, policy: Policy) -> Grade:
    """Map a 0-100 score to its letter, per the policy's own table.

    Parameters
    ----------
    score : float
        The rolled-up score.
    policy : Policy
        The loaded grading policy, carrying the letter minimums.

    Returns
    -------
    Grade
        The highest grade whose minimum `score` clears, or `F` if it clears none.
    """

    bands = sorted(
        ((minimum, Grade(letter)) for letter, minimum in policy.letters.items()),
        key=lambda pair: pair[0],
        reverse=True,
    )
    for minimum, grade in bands:
        if score >= minimum:
            return grade
    return Grade.F


def _band_for(metric_policy: MetricPolicy, taxonomy_type: str) -> Band | None:
    """Pick one metric's band for one taxonomy type, falling back to its default.

    Parameters
    ----------
    metric_policy : MetricPolicy
        The metric's policy, as `policy.yaml` declared it.
    taxonomy_type : str
        The stream's taxonomy type, e.g. `"unmapped.arm"` until `mapping`
        grows type matching.

    Returns
    -------
    Band or None
        The band keyed by `taxonomy_type`, the band keyed `"default"`,
        or `None` if neither exists.
    """

    return metric_policy.thresholds.get(
        taxonomy_type, metric_policy.thresholds.get("default")
    )


def _reduce_value(
    value: float, *, metric_policy: MetricPolicy
) -> tuple[float, str | None]:
    """Turn a metric's raw value into what its band actually grades.

    `direct` mode grades the value as-is.
    `abs_dev` mode grades the relative deviation from a named target,
    resolved against the metric's own `limits`.

    Parameters
    ----------
    value : float
        The metric's computed value.
    metric_policy : MetricPolicy
        The metric's policy, carrying its mode, target and limits.

    Returns
    -------
    tuple[float, str or None]
        The value to grade, and `None` on success — or `0.0` alongside a reason
        when `abs_dev` needs a target the deployment never declared.
    """

    if metric_policy.mode == ScoreMode.DIRECT:
        return value, None

    # mode is ABS_DEV: metric_policy.target is guaranteed set by the model's
    # own validator, but the deployment still has to supply the number itself.
    assert metric_policy.target is not None
    target = metric_policy.limits.get(metric_policy.target)
    if target is None or target == 0:
        return 0.0, f"no {metric_policy.target!r} declared under limits"
    return abs(value - target) / abs(target), None


def _points_for(value: float, band: Band, *, higher_is_better: bool) -> float | None:
    """Interpolate one value between a band's good and bad bounds.

    Parameters
    ----------
    value : float
        The value to grade, already reduced by `_reduce_value`.
    band : Band
        The good/bad boundaries to interpolate between.
    higher_is_better : bool
        Whether a larger value scores higher.

    Returns
    -------
    float or None
        A 0-100 score, or `None` when the band is missing either bound —
        there is nothing to interpolate across yet.
    """

    if band.good is None or band.bad is None:
        return None

    good, bad = band.good, band.bad
    if higher_is_better:
        if value >= good:
            return 100.0
        if value <= bad:
            return 0.0
        return 100.0 * (value - bad) / (good - bad)

    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return 100.0 * (bad - value) / (bad - good)


def _status_for_points(points: float) -> MetricStatus:
    """Map a metric's 0-100 points to its status.

    Parameters
    ----------
    points : float
        The interpolated score `_points_for` produced.

    Returns
    -------
    MetricStatus
        `good` at 100, `critical` at 0, `warning` in between —
        so a value at exactly its `good` bound scores 100 and grades `good`
        by construction, rather than by a second, separately-tuned rule.
    """

    if points >= 100.0:
        return MetricStatus.GOOD
    if points <= 0.0:
        return MetricStatus.CRITICAL
    return MetricStatus.WARNING


def _severity_for_status(status: MetricStatus) -> Severity | None:
    """Map a regraded status to a Finding severity, or None for a non-defect.

    Parameters
    ----------
    status : MetricStatus
        The status a metric result was regraded to.

    Returns
    -------
    Severity or None
        `critical` or `warning`, or `None` for `good`, `report_only` and
        `not_applicable` — a Finding exists to report a defect, and none
        of those three is one.
    """

    if status == MetricStatus.CRITICAL:
        return Severity.CRITICAL
    if status == MetricStatus.WARNING:
        return Severity.WARNING
    return None


def _finding_for(
    result: MetricResult,
    *,
    metric_name: str,
    family: str,
    points: float,
    location: FindingLocation,
) -> Finding | None:
    """Build one Finding from a regraded metric result, if it graded a defect.

    Parameters
    ----------
    result : MetricResult
        The metric's result, already regraded by `_resolve`.
    metric_name : str
        The metric's own bare name.
    family : str
        The family this metric belongs to, from its policy entry.
    points : float
        The 0-100 this metric scored — its own contribution to the rollup.
    location : FindingLocation
        Where the node this metric ran on sits in the graded tree.

    Returns
    -------
    Finding or None
        The finding, or `None` when `result.status` is not `warning` or `critical`.
    """

    severity = _severity_for_status(result.status)
    if severity is None:
        return None

    # result.value is guaranteed set: _resolve demotes any value-less or
    # non-finite result to not_applicable before a status can grade.
    assert result.value is not None
    return Finding(
        metric_id=f"{family}.{metric_name}",
        family=family,
        severity=severity,
        value=result.value,
        unit=result.unit,
        points=points,
        episode_id=location.episode_id,
        stream=location.stream,
        instance=location.instance,
        channel=location.channel,
        evidence=result.evidence,
    )


def sort_findings(findings: Sequence[Finding]) -> list[Finding]:
    """Order findings worst first: critical before warning, lowest points first.

    Parameters
    ----------
    findings : Sequence[Finding]
        The findings to order.

    Returns
    -------
    list[Finding]
        `findings` sorted so the first element is the worst thing found —
        the finding a CI gate or a report would read first.
    """

    return sorted(
        findings,
        key=lambda finding: (finding.severity != Severity.CRITICAL, finding.points),
    )


def _resolve(
    result: MetricResult, *, metric_name: str, taxonomy_type: str, policy: Policy
) -> tuple[MetricResult, float | None]:
    """Recompute one metric result against the policy, and its points if graded.

    Parameters
    ----------
    result : MetricResult
        One metric's result, exactly as `metrics` produced it.
    metric_name : str
        The metric's own bare name, looked up in `policy.metrics` via
        `Policy.entry_for`.
    taxonomy_type : str
        The stream's taxonomy type, for a type-keyed threshold lookup.
    policy : Policy
        The loaded grading policy.

    Returns
    -------
    tuple[MetricResult, float or None]
        The regraded result, alongside its 0-100 points when it graded —
        `None` when the result stayed `not_applicable` or `report_only`.
    """

    # Step 1: nothing to grade — pass the not_applicable verdict through untouched.
    if result.status == MetricStatus.NOT_APPLICABLE:
        return result, None

    evidence = dict(result.evidence)

    # Step 2: a result claiming a status but carrying no value, or a value
    # that is NaN or infinite, would leave every step below with nothing
    # real to compare — treat it the same as not_applicable rather than
    # propagating a NaN into a score no reader could then explain.
    if result.value is None or not math.isfinite(result.value):
        evidence["ungraded_reason"] = (
            "status is not not_applicable but value is missing or non-finite"
        )
        return (
            result.model_copy(
                update={"status": MetricStatus.NOT_APPLICABLE, "evidence": evidence}
            ),
            None,
        )

    def _stay_report_only(reason: str) -> tuple[MetricResult, float | None]:
        evidence["ungraded_reason"] = reason
        return (
            result.model_copy(
                update={"status": MetricStatus.REPORT_ONLY, "evidence": evidence}
            ),
            None,
        )

    # Step 3: no policy entry, permanently report-only, or graded with
    # nothing to grade against — several different reasons to stay report_only,
    # each recorded as its own evidence.
    entry = policy.entry_for(metric_name)
    if entry is None:
        return _stay_report_only(f"policy carries no entry for {metric_name!r}")
    _family, metric_policy = entry
    if metric_policy.report_only:
        return _stay_report_only("report_only by policy design")

    band = _band_for(metric_policy, taxonomy_type)
    if band is None:
        return _stay_report_only(
            f"no threshold band for {taxonomy_type!r} or 'default'"
        )

    reduced, reduction_reason = _reduce_value(
        value=result.value, metric_policy=metric_policy
    )
    if reduction_reason is not None:
        return _stay_report_only(reduction_reason)

    points = _points_for(reduced, band, higher_is_better=metric_policy.higher_is_better)
    if points is None:
        missing = ", ".join(
            name
            for name, bound in (("good", band.good), ("bad", band.bad))
            if bound is None
        )
        return _stay_report_only(
            f"band for {taxonomy_type!r} has no {missing} bound to interpolate against"
        )

    # Step 4: graded, and there is a full band to grade against.
    evidence.pop("ungraded_reason", None)
    status = _status_for_points(points)
    return result.model_copy(update={"status": status, "evidence": evidence}), points


def resolve_status(
    result: MetricResult, *, metric_name: str, taxonomy_type: str, policy: Policy
) -> MetricResult:
    """Recompute one metric result's status against the policy.

    Parameters
    ----------
    result : MetricResult
        One metric's result, exactly as `metrics` produced it.
    metric_name : str
        The metric's own bare name, looked up in `policy.metrics`.
    taxonomy_type : str
        The stream's taxonomy type, for a type-keyed threshold lookup.
    policy : Policy
        The loaded grading policy.

    Returns
    -------
    MetricResult
        The regraded result. See `_resolve` for the full set of outcomes.
    """

    return _resolve(
        result,
        metric_name=metric_name,
        taxonomy_type=taxonomy_type,
        policy=policy,
    )[0]


def _weighted_mean(
    contributions: Sequence[tuple[str, float, float | None]],
) -> float | None:
    """Weighted-average a list of (name, points, weight) contributions.

    An undecided weight (`None`) only stands in for equal weighting when
    every contribution in the list is equally undecided — mixed with a
    decided one, it would silently fix a ratio nobody chose.

    Parameters
    ----------
    contributions : Sequence[tuple[str, float, float or None]]
        Each contribution's name (for the error message), its points, and its weight.

    Returns
    -------
    float or None
        The weighted mean, or `None` when `contributions` is empty.

    Raises
    ------
    ValueError
        If some contributions carry a decided weight and others do not.
    """

    if not contributions:
        return None

    weights = [weight for _, _, weight in contributions]
    if any(weight is not None for weight in weights) and any(
        weight is None for weight in weights
    ):
        undecided = [name for name, _, weight in contributions if weight is None]
        raise ValueError(
            "mixes a decided weight with an undecided one among its "
            f"contributions: {undecided} carry no weight"
        )

    resolved = [
        (points, 1.0 if weight is None else weight)
        for _, points, weight in contributions
    ]
    total_weight = sum(weight for _, weight in resolved)
    if total_weight <= 0:
        return None
    return sum(points * weight for points, weight in resolved) / total_weight


def score_metrics(
    results: dict[str, MetricResult],
    *,
    level: Level,
    taxonomy_type: str,
    policy: Policy,
    location: FindingLocation,
) -> tuple[dict[str, MetricResult], ScoreResult, list[Finding]]:
    """Regrade one node's metric results, roll them up, and find its defects.

    The same function grades a channel or a stream — whichever level's
    metrics `results` holds — since nothing below differs by level beyond
    which `ScoreResult.level` the answer carries.

    Parameters
    ----------
    results : dict[str, MetricResult]
        One node's metric results, keyed by metric name,
        exactly as `run_channel_metrics` or `run_stream_metrics` produced them.
    level : Level
        Which level `results` was computed at, and the level the returned
        `ScoreResult` attaches to.
    taxonomy_type : str
        The key a type-keyed threshold band is looked up under: the stream's
        taxonomy type at channel and stream level, `Level.EPISODE.value` at
        episode level.
    policy : Policy
        The loaded grading policy.
    location : FindingLocation
        Where this node sits in the graded tree, for addressing
        any finding it raises.

    Returns
    -------
    tuple[dict[str, MetricResult], ScoreResult, list[Finding]]
        The same metrics, each regraded against `policy`; the score at
        `level` rolled up from the graded ones, with `report_only` and
        `not_applicable` results excluded from that rollup's denominator;
        and one Finding per metric that graded `warning` or `critical`,
        unsorted — a caller flattening several nodes sorts once, over
        the whole list.

    Raises
    ------
    ValueError
        If the node's graded metrics mix a policy-decided weight with
        an undecided one, at either the per-family or the cross-family step.
    """

    # Step 1: recompute every metric's status against the policy, keeping
    # its points and family alongside — never trust what `metrics` wrote,
    # since it had no policy to grade against. A finding is raised here too,
    # since this is the only place a metric's regraded status, its points
    # and its family are all in scope together.
    graded: dict[str, MetricResult] = {}
    findings: list[Finding] = []
    by_family: dict[str, list[tuple[str, float, float | None]]] = {}

    for name, result in results.items():
        regraded, points = _resolve(
            result,
            metric_name=name,
            taxonomy_type=taxonomy_type,
            policy=policy,
        )
        graded[name] = regraded
        if points is None:
            continue

        # entry_for can't miss here: _resolve only returns points once
        # entry_for already found the entry.
        entry = policy.entry_for(name)
        assert entry is not None
        family, metric_policy = entry
        by_family.setdefault(family, []).append((name, points, metric_policy.weight))

        finding = _finding_for(
            regraded,
            metric_name=name,
            family=family,
            points=points,
            location=location,
        )
        if finding is not None:
            findings.append(finding)

    # Step 2: fold each family's graded metrics into that family's score.
    # A failing metric only counts toward the penalty once its family
    # actually contributed — a metric whose own family's weights cancel
    # out to zero never reaches this node's score, so it should not cost
    # its points either.
    family_scores: list[tuple[str, float, float | None]] = []
    families: dict[str, float] = {}
    contributing_metrics = 0
    failing = 0
    for family, contributions in by_family.items():
        family_points = _weighted_mean(contributions)
        if family_points is not None:
            family_scores.append(
                (family, family_points, policy.family_weights.get(family))
            )
            families[family] = family_points
            contributing_metrics += len(contributions)
            failing += sum(
                1 for _, points, _ in contributions if points <= _FAILING_POINTS
            )

    # Step 3: fold family scores into this node's own — this is missing_family:
    # skip, since a family that graded nothing simply never enters `family_scores`
    # in the first place.
    score = _weighted_mean(family_scores)
    if score is None:
        return (
            graded,
            ScoreResult(
                level=level,
                score=None,
                grade=None,
                train_ready=None,
                n_contributing=0,
                families=families,
            ),
            findings,
        )

    # Step 4: apply the capped per-failure penalty, if the policy declares one.
    if policy.fail_penalty is not None:
        penalty = failing * policy.fail_penalty
        if policy.fail_penalty_cap is not None:
            penalty = min(penalty, policy.fail_penalty_cap)
        score = max(0.0, score - penalty)

    return (
        graded,
        ScoreResult(
            level=level,
            score=score,
            grade=grade_for(score, policy),
            train_ready=score >= _TRAIN_READY_MINIMUM,
            n_contributing=contributing_metrics,
            families=families,
        ),
        findings,
    )


def rollup(
    level: Level, children: Sequence[ScoreResult], *, policy: Policy
) -> ScoreResult:
    """Combine one level's child ScoreResults into this level's own.

    The same function serves every step from Stream up through Dataset —
    Channel to Stream, Stream to Episode, Episode to Dataset — since nothing
    decides a different weight between children at any of those levels;
    each child with a score counts once.

    Parameters
    ----------
    level : Level
        The level this result attaches to — normally one step above every child's level.
        Exception: a Stream's or an Episode's `children` can include that node's own
        `score_metrics` result alongside its channels' or streams', so one child may sit
        at `level` itself rather than one below it.
    children : Sequence[ScoreResult]
        The level below's results, exactly as `score_metrics`
        or a prior `rollup` call produced them.
    policy : Policy
        The loaded grading policy, carrying the letter minimums.

    Returns
    -------
    ScoreResult
        `score=None` when every child was itself `None` —
        nothing graded anywhere beneath this level.
    """

    family_values: dict[str, list[float]] = {}
    for child in children:
        for family, value in child.families.items():
            family_values.setdefault(family, []).append(value)
    families = {
        family: sum(values) / len(values) for family, values in family_values.items()
    }

    scores = [child.score for child in children if child.score is not None]
    if not scores:
        return ScoreResult(
            level=level,
            score=None,
            grade=None,
            train_ready=None,
            n_contributing=0,
            families=families,
        )
    score = sum(scores) / len(scores)

    return ScoreResult(
        level=level,
        score=score,
        grade=grade_for(score, policy),
        train_ready=score >= _TRAIN_READY_MINIMUM,
        n_contributing=len(scores),
        families=families,
    )
