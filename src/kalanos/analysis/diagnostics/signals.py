"""Declared clock comparisons, command response and dimensionless motion."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import statistics
from collections.abc import Sequence
from typing import cast

# Internal
from kalanos.analysis.diagnostics.common import (
    Unavailable,
    axis,
    finite,
    nearest,
    numeric,
    paired_axes,
    result,
    select,
    validated,
)
from kalanos.analysis.localization import support_for
from kalanos.analysis.models.diagnostics import (
    DiagnosticResult,
    DiagnosticReviewPolicy,
    MotionSpec,
    TimingSpec,
    TrackingSpec,
)
from kalanos.analysis.models.domain import Episode


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a linearly interpolated empirical quantile, or None for no data."""
    if not values:
        return None
    xs = sorted(values)
    p = (len(xs) - 1) * fraction
    lo = int(p)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (p - lo)


def timing(
    episode: Episode, spec: TimingSpec, review: DiagnosticReviewPolicy | None = None
) -> DiagnosticResult:
    """Measure pair coverage/skew and optionally fit explicitly corresponding events."""
    review = review or DiagnosticReviewPolicy()
    limit = review.timing_max_unmatched_fraction.get(spec.id)
    left, _ = select(episode, spec.left)
    right, _ = select(episode, spec.right)
    (rows, _, lt), (rr, _, rt) = paired_axes(left, right, spec.relation)
    skew, ages, misses = [], [], []
    matches = []
    for i, time in enumerate(lt):
        # ordered() requires two finite timestamps per axis, so nearest always finds one
        j = cast(int, nearest(rt, time))
        delta = rt[j] - time
        if abs(delta) <= spec.tolerance_s:
            matches.append([rows[i], rr[j]])
            skew.append(delta)
        else:
            misses.append((i, i + 1))
        j = nearest(rt, time, "previous")
        if j is not None:
            ages.append(time - rt[j])
    fit = None
    if spec.events:
        if len(spec.events) < 3:
            raise Unavailable(
                "offset/drift fitting requires at least three declared event pairs"
            )
        lmap, rmap = dict(zip(rows, lt, strict=False)), dict(zip(rr, rt, strict=False))
        if any(a not in lmap or b not in rmap for a, b in spec.events):
            raise Unavailable("event correspondence references an absent source row")
        x = [rmap[b] for a, b in spec.events]
        y = [lmap[a] - rmap[b] for a, b in spec.events]
        xm, ym = statistics.mean(x), statistics.mean(y)
        ss = sum((v - xm) ** 2 for v in x)
        slope = sum((a - xm) * (b - ym) for a, b in zip(x, y, strict=False)) / ss
        intercept = ym - slope * xm
        residuals = [b - intercept - slope * a for a, b in zip(x, y, strict=False)]
        fit = {
            "offset_at_left_origin_s": intercept,
            "remaining_drift_ppm": slope * 1e6,
            "residual_rms_s": math.sqrt(
                statistics.mean(float(v * v) for v in residuals)
            ),
            "slope_standard_error_ppm": math.sqrt(
                sum(v * v for v in residuals) / (len(x) - 2) / ss
            )
            * 1e6,
            "event_count": len(x),
            "applied_to_data": False,
            "interpretation": (
                "fit to declared corresponding events after the configured transform"
            ),
        }
    fraction = len(misses) / len(lt)
    return result(
        "timing",
        spec,
        episode,
        subject=spec.left.model_dump(),
        measurements={
            "left_samples": len(lt),
            "right_samples": len(rt),
            "matched_samples": len(matches),
            "unmatched_fraction": fraction,
            "overlap_s": max(0, min(lt[-1], rt[-1]) - max(lt[0], rt[0])),
            "signed_skew_median_s": percentile(skew, 0.5),
            "absolute_skew_p95_s": percentile([abs(x) for x in skew], 0.95),
            "causal_age_p95_s": percentile(ages, 0.95),
            "event_fit": fit,
        },
        evidence={
            "relation": spec.relation.model_dump(),
            "tolerance_s": spec.tolerance_s,
            "review_max_unmatched_fraction": limit,
            "matching": "nearest, earlier on ties, many-to-one allowed",
            "matched_source_rows": matches,
            "interpretation": "timestamp alignment, not measured transport latency",
        },
        support=support_for(rows, misses),
        consequence="review"
        if limit is not None and fraction > limit
        else "report_only",
    )


def tracking(
    episode: Episode, spec: TrackingSpec, review: DiagnosticReviewPolicy | None = None
) -> DiagnosticResult:
    """Compare absolute/rate commands, or delta commands relative to state at issue."""
    review = review or DiagnosticReviewPolicy()
    limit = review.tracking_max_abs_error.get(spec.id)
    left, lc = select(episode, spec.left)
    right, rc = select(episode, spec.right)
    props = ["identity", "quantity", "unit", "frame", "representation"]
    lb = validated(lc, props + ["command"], "tracking")
    rb = validated(rc, props, "tracking")
    if (
        lb.command.value != spec.semantics
        or lb.quantity != rb.quantity
        or lb.unit != rb.unit
        or lb.frame != rb.frame
    ):
        raise Unavailable(
            "command semantics, quantities, native units or frames do not match"
        )
    expected_quantity = "velocity" if spec.semantics == "rate" else "position"
    if (
        lb.quantity.value != expected_quantity
        or lb.representation.value != "continuous"
        or rb.representation.value != "continuous"
    ):
        raise Unavailable(
            "tracking needs continuous unwrapped position or velocity channels"
        )
    if spec.semantics == "delta" and spec.response_delay_s <= 0:
        raise Unavailable("delta tracking needs a positive declared response horizon")
    (rows, lp, lt), (rrows, rp, rt) = paired_axes(left, right, spec.relation)
    command, state = numeric(left, lc, lp), numeric(right, rc, rp)
    raw, adjusted, samples, bad = [], [], [], []
    for i, (time, value) in enumerate(zip(lt, command, strict=False)):
        # ordered() requires two finite timestamps per axis, so nearest always finds one
        base = cast(int, nearest(rt, time))
        target = cast(int, nearest(rt, time + spec.response_delay_s))
        if not finite(value) or any(
            abs(rt[j] - t) > spec.tolerance_s or not finite(state[j])
            for j, t in ((base, time), (target, time + spec.response_delay_s))
        ):
            continue
        expected = value + state[base] if spec.semantics == "delta" else value
        re, ae = state[base] - expected, state[target] - expected
        raw.append(re)
        adjusted.append(ae)
        samples.append(
            {
                "command_row": rows[i],
                "state_row": rrows[target],
                "baseline_state_row": rrows[base],
                "expected_response": expected,
                "observed_response": state[target],
                "command_value": value,
                "error": ae,
            }
        )
        if limit is not None and abs(ae) > limit:
            bad.append((i, i + 1))
    if not adjusted:
        raise Unavailable(
            "no finite command/response samples meet the matching tolerance"
        )
    return result(
        "tracking",
        spec,
        episode,
        subject=spec.left.model_dump(),
        measurements={
            "native_unit": lb.unit,
            "command_samples": len(command),
            "matched_samples": len(adjusted),
            "unmatched_samples": len(command) - len(adjusted),
            "raw_rmse": math.sqrt(statistics.mean(v * v for v in raw)),
            "adjusted_rmse": math.sqrt(statistics.mean(v * v for v in adjusted)),
            "abs_error_p95": percentile([abs(v) for v in adjusted], 0.95),
            "signed_error_mean": statistics.mean(adjusted),
        },
        evidence={
            "response_delay_s": spec.response_delay_s,
            "review_max_abs_error": limit,
            "semantics": spec.semantics,
            "relation": spec.relation.model_dump(),
            "samples": samples,
            "interpretation": (
                "tracking discrepancy candidate; contact and controller "
                "dynamics are not inferred"
            ),
        },
        support=support_for(rows, bad),
        consequence="review" if bad else "report_only",
    )


def motion(
    episode: Episode, spec: MotionSpec, review: DiagnosticReviewPolicy | None = None
) -> DiagnosticResult:
    """Compute segment-wise dimensionless jerk and physical-limit observations."""
    review = review or DiagnosticReviewPolicy()
    try:
        import numpy as np
    except ImportError as exc:
        raise Unavailable("motion diagnostics require kalanos[numeric]") from exc
    stream, channel = select(episode, spec.channel)
    binding = validated(
        channel, ["identity", "quantity", "unit", "representation"], "derivatives"
    )
    if binding.quantity.value != "position" or binding.representation.value not in (
        "continuous",
        "angle",
    ):
        raise Unavailable("dimensionless position smoothness needs continuous position")
    if binding.representation.value == "angle" and spec.angle_period is None:
        raise Unavailable("wrapped angles require a declared native-unit period")
    rows, positions, times = axis(stream)
    values = numeric(stream, channel, positions)
    segments, start = [], None
    for i in range(len(values) + 1):
        valid = i < len(values) and finite(values[i])
        boundary = (
            i == len(values)
            or not valid
            or (
                i > 0
                and (
                    times[i] - times[i - 1] > spec.max_gap_s
                    or rows[i] != rows[i - 1] + 1
                )
            )
        )
        if boundary and start is not None:
            segments.append((start, i))
            start = None
        if valid and start is None:
            start = i
    details, bad, measured = [], [], 0
    for a, b in segments:
        if b - a < spec.min_segment_samples:
            continue
        t, x = np.array(times[a:b]), np.array(values[a:b], dtype=float)
        if spec.angle_period is not None:
            x = np.unwrap(x, period=spec.angle_period)
        gaps = np.diff(t)
        if not np.allclose(gaps, np.median(gaps), rtol=1e-3, atol=1e-12):
            continue  # no resampling or implicit interpolation
        duration, amplitude = float(t[-1] - t[0]), float(np.ptp(x))
        velocity = np.diff(x) / gaps
        jerk = np.diff(x, n=3) / float(np.median(gaps)) ** 3
        dim = (
            duration**5 * float(np.sum(jerk * jerk) * np.median(gaps)) / amplitude**2
            if amplitude
            else None
        )
        if dim is not None and not math.isfinite(dim):
            raise Unavailable("dimensionless jerk overflowed")
        details.append(
            {
                "source_start": rows[a],
                "source_end_exclusive": rows[b - 1] + 1,
                "duration_s": duration,
                "amplitude": amplitude,
                "dimensionless_jerk": dim,
                "stationary": amplitude == 0,
                "max_abs_velocity": float(np.max(np.abs(velocity))),
                "max_abs_step": float(np.max(np.abs(np.diff(x)))),
                "velocity_sign_changes": int(np.sum(velocity[:-1] * velocity[1:] < 0)),
            }
        )
        measured += b - a
        if spec.max_abs_velocity is not None:
            bad.extend(
                (a + i + 1, a + i + 2)
                for i, v in enumerate(velocity)
                if abs(v) > spec.max_abs_velocity
            )
    if not details:
        raise Unavailable(
            "no sufficiently long finite contiguous regular motion segments"
        )
    limits = None
    if binding.limits is not None:
        try:
            validated(channel, ["identity", "quantity", "unit", "limits"], "limits")
        except Unavailable:
            limits = {
                "status": "unavailable",
                "reason": "limits lack matching scoped validation",
            }
        else:
            outside = [
                i
                for i, v in enumerate(values)
                if finite(v) and not binding.limits[0] <= v <= binding.limits[1]
            ]
            limits = {
                "status": "computed",
                "native_limits": binding.limits,
                "outside_samples": len(outside),
            }
            bad.extend((i, i + 1) for i in outside)
    return result(
        "motion",
        spec,
        episode,
        subject=spec.channel.model_dump(),
        measurements={
            "segments": details,
            "examined_samples": measured,
            "excluded_samples": len(values) - measured,
            "limits": limits,
        },
        evidence={
            "estimator": "position_dimensionless_jerk_v1",
            "formula": "duration^5 * integral(jerk^2 dt) / peak_to_peak^2",
            "amplitude_unit": binding.unit,
            "angle_period": spec.angle_period,
            "max_gap_s": spec.max_gap_s,
            "limits_evidence": spec.limits_evidence,
            "interpretation": (
                "descriptive smoothness; segmentation and sample rate remain material"
            ),
        },
        support=support_for(rows, sorted(set(bad))),
        consequence="review" if bad and review.review_motion_limits else "report_only",
    )
