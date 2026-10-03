"""Fail-closed statistical promotion, matched to the exact recorded run context."""

import math
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from kalanos.analysis.models.eligibility import BlockingRoute, Consequence
from kalanos.analysis.models.provenance import content_digest
from kalanos.analysis.models.scoring import Severity


@lru_cache(maxsize=128)
def binomial_upper(successes, trials, alpha=0.05):
    """One-sided exact upper bound; independent Bernoulli trials are required."""
    if not trials or successes == trials:
        return 1.0
    if successes == 0:
        return -math.expm1(math.log(alpha) / trials)

    def cdf(p):
        terms = [
            math.lgamma(trials + 1)
            - math.lgamma(i + 1)
            - math.lgamma(trials - i + 1)
            + i * math.log(p)
            + (trials - i) * math.log1p(-p)
            for i in range(successes + 1)
        ]
        top = max(terms)
        return math.exp(top) * sum(math.exp(t - top) for t in terms)

    lo, hi = successes / trials, 1.0
    for _ in range(70):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def validation_reason(m, now):
    if m.status != "accepted" or not m.accepted_by or not m.accepted_at:
        return "manifest has no recorded acceptance"
    if m.accepted_at > now or (m.valid_until and m.valid_until <= now):
        return "manifest acceptance is future-dated or expired"
    if not (
        m.independent_episodes
        and m.real_fault_validation
        and m.combined_policy_validated
        and m.validation_sessions
    ):
        return (
            "independence, real-fault, session and combined-policy validation required"
        )
    if m.valid_episodes < 600 or m.fault_episodes < 100:
        return "validation requires at least 600 valid and 100 fault episodes"
    if (
        m.false_blocks / m.valid_episodes > 0.005
        or binomial_upper(m.false_blocks, m.valid_episodes) > 0.005
    ):
        return "false-block upper confidence bound exceeds 0.5%"
    if m.detected_faults / m.fault_episodes < 0.90:
        return "observed recall is below 90%"
    lower = 1 - binomial_upper(m.fault_episodes - m.detected_faults, m.fault_episodes)
    if lower < 0.90:
        return "recall lower confidence bound is below 90%"
    return None


def context_for(run, scope, producer, metric_id, metric_policy, taxonomy_type, policy):
    known_adapters = (
        producer is not None
        and bool(producer.adapters)
        and all(v != "unknown" for v in producer.adapters.values())
    )
    return {
        "metric_id": metric_id,
        "taxonomy_type": taxonomy_type,
        "detector_digest": producer.metrics.get(metric_id) if producer else None,
        "thresholds_digest": content_digest(metric_policy.model_dump(mode="json")),
        "binding_digest": run.binding.digest if run and run.binding else None,
        "scope_digest": content_digest(
            {
                "requirements": run.requirements.digest,
                "execution": run.execution.digest,
                "dictionary": run.dictionary.digest,
                "scope": scope.model_dump(mode="json"),
                "adapters": producer.adapters if producer else None,
                "decisions": policy.model_dump(
                    mode="json", exclude={"calibration_manifests", "calibrated_metrics"}
                ),
            }
        )
        if known_adapters
        and run
        and scope
        and all(
            x and x.digest for x in (run.requirements, run.execution, run.dictionary)
        )
        else None,
    }


def evaluate(policy, context, now=None):
    now = now or datetime.now(timezone.utc)
    if any(v is None or v == "unknown" for v in context.values()):
        return {
            "accepted": False,
            "reason": "runtime identity is incomplete",
            "context": context,
        }
    rejections = []
    for manifest in policy.calibration_manifests:
        if manifest.metric_id != context["metric_id"]:
            continue
        differences = [k for k, v in context.items() if getattr(manifest, k) != v]
        reason = (
            "context mismatch: " + ", ".join(differences)
            if differences
            else validation_reason(manifest, now)
        )
        if reason is None:
            return {
                "accepted": True,
                "manifest_id": manifest.id,
                "manifest_digest": content_digest(manifest.model_dump(mode="json")),
                "context": context,
            }
        rejections.append({"manifest_id": manifest.id, "reason": reason})
    return {
        "accepted": False,
        "reason": "no accepted matching calibration",
        "rejections": rejections,
        "context": context,
    }


def apply_calibration(findings, policies, run, scope, producer):
    """Called before the one eligibility calculation; no surface can promote later."""
    result = []
    for f in findings:
        policy = policies[f.episode_id]
        entry = policy.metrics[f.metric_id]
        key = {
            "episode": f.episode_id,
            "source": f.source_path,
            "field": f.source_field,
            "stream": f.stream if f.source_field is None or f.channel is None else None,
            "instance": f.instance,
            "channel": f.source_index if f.source_index is not None else f.channel,
            "metric": f.metric_id,
        }
        updates: dict[str, Any] = {"id": content_digest(key)}
        if (
            f.severity == Severity.CRITICAL
            and (entry.consequence or Consequence.BLOCK) == Consequence.BLOCK
            and (
                f.metric_id == "integrity.snr_db"
                or (entry.route or BlockingRoute.STATISTICAL)
                == BlockingRoute.STATISTICAL
            )
        ):
            outcome = evaluate(
                policy,
                context_for(
                    run,
                    scope,
                    producer,
                    f.metric_id,
                    entry,
                    f.stream or "episode",
                    policy,
                ),
            )
            if (
                f.metric_id == "integrity.snr_db"
                and f.evidence.get("noise_assessment", {}).get("status")
                != "above_reference"
            ):
                outcome = {
                    **outcome,
                    "accepted": False,
                    "reason": "SNR physical noise assessment is unavailable",
                }
            updates.update(
                calibration=outcome,
                evidence_strength="statistical" if outcome["accepted"] else "heuristic",
                consequence=Consequence.BLOCK
                if outcome["accepted"]
                else Consequence.REVIEW,
                route=BlockingRoute.STATISTICAL if outcome["accepted"] else None,
            )
        result.append(f.model_copy(update=updates))
    return result
