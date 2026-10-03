"""Read-only comparison: identify differences without inventing causal attribution."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
from typing import Any

# External
from pydantic import BaseModel, Field
from upath import UPath

# Internal
from kalanos.analysis.models.legacy import LegacyReport, load_any
from kalanos.analysis.models.provenance import content_digest
from kalanos.analysis.models.report import Report
from kalanos.analysis.source_identity import CANONICALIZATION


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Comparison(BaseModel):
    """What differs between two saved reports, and whether their decisions compare.

    Attributes
    ----------
    comparable : bool
        Whether no reason against comparing was found:
        identities present and matching, inventories complete and equal, runs complete.
    reasons : list[str]
        Every reason the reports are not comparable, sorted and deduplicated.
    identity_changes : list[str]
        Identities that differ between the reports:
        source, binding, dictionary, policy, requirements, execution, metrics, adapters,
        scope, diagnostics or calibration outcome.
    added_episodes : list[str]
        Episode ids only the new report has.
    removed_episodes : list[str]
        Episode ids only the old report has.
    episode_changes : list[dict[str, Any]]
        Episodes in both whose eligibility or quality score changed,
        with the old and new status and score.
    metric_changes : list[dict[str, Any]]
        Metric results that differ between shared episodes,
        keyed by episode and metric address.
    attribution : str
        A fixed caveat that an identity change is only a possible cause.
    readiness_delta : float or None
        New readiness minus old,
        set only when the reports are comparable and both scores are defined.
    diagnostic_changes : list[dict[str, Any]]
        Diagnostic results that differ, keyed by episode, kind and id.
    """

    comparable: bool
    reasons: list[str] = Field(default_factory=list)
    identity_changes: list[str] = Field(default_factory=list)
    added_episodes: list[str] = Field(default_factory=list)
    removed_episodes: list[str] = Field(default_factory=list)
    episode_changes: list[dict[str, Any]] = Field(default_factory=list)
    metric_changes: list[dict[str, Any]] = Field(default_factory=list)
    attribution: str = (
        "Identity changes are possible causes, not proven causal attribution."
    )
    readiness_delta: float | None = None
    diagnostic_changes: list[dict[str, Any]] = Field(default_factory=list)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _source(report):
    source = report.run.source if report.run else None
    if (
        source
        and source.digest
        and source.complete
        and source.scope.value in ("consumed", "complete")
        and source.algorithm == "sha256"
        and source.canonicalization == CANONICALIZATION
    ):
        return (
            source.algorithm,
            source.canonicalization,
            source.scope.value,
            source.digest,
        )
    return None


def _identity(report, name):
    field = getattr(report.run, name) if report.run else None
    return field.digest if field else None


def _metrics(episode):
    items = {}

    def add(address, metrics):
        for name, value in metrics.items():
            key = (*address, name)
            if key in items:
                raise ValueError(f"ambiguous metric address: {key}")
            items[key] = value

    add(("episode", None, None, None, None), episode.metrics)
    for s in episode.streams:
        address = (s.source_path, s.source_field, s.taxonomy_type, s.instance)
        add((*address, None), s.metrics)
        for c in s.channels:
            add(
                (
                    *address,
                    c.channel.source_index
                    if c.channel.source_index is not None
                    else c.channel.name,
                ),
                c.metrics,
            )
    return items


def compare_reports(
    old: Report | LegacyReport | str | os.PathLike[str] | UPath,
    new: Report | LegacyReport | str | os.PathLike[str] | UPath,
) -> Comparison:
    """List what differs between two reports, and whether their decisions compare.

    Parameters
    ----------
    old : Report, LegacyReport, str, PathLike or UPath
        The earlier report, or where its JSON is.
    new : Report, LegacyReport, str, PathLike or UPath
        The later report, or where its JSON is.

    Returns
    -------
    Comparison
        Never comparable when either report is legacy.
        Otherwise every difference in identities, scope, inventories,
        diagnostics, episodes and metrics,
        with `comparable` set only when no reason against it was found.

    Raises
    ------
    OSError
        If a report path cannot be read.
    ValueError
        If a report file does not load,
        a report repeats an episode id,
        or one episode holds two metrics at the same address.
    """

    old = load_any(old) if not isinstance(old, (Report, LegacyReport)) else old
    new = load_any(new) if not isinstance(new, (Report, LegacyReport)) else new
    if isinstance(old, LegacyReport) or isinstance(new, LegacyReport):
        return Comparison(
            comparable=False,
            reasons=[
                "legacy reports lack compatible decision and implementation "
                "identities; "
                "inspect their recorded values separately"
            ],
        )
    reasons, changes = [], []
    for label, left, right in [
        ("source", _source(old), _source(new)),
        *[
            (k, _identity(old, k), _identity(new, k))
            for k in ("binding", "dictionary", "policy", "requirements", "execution")
        ],
        (
            "metrics",
            old.producer.metrics if old.producer else None,
            new.producer.metrics if new.producer else None,
        ),
        (
            "adapters",
            old.producer.adapters if old.producer else None,
            new.producer.adapters if new.producer else None,
        ),
    ]:
        if (
            not left
            or not right
            or (
                label in ("metrics", "adapters")
                and any(
                    v == "unknown"
                    for identities in (left, right)
                    for v in identities.values()
                )
            )
        ):
            reasons.append(f"{label} identity is missing or insufficient")
        elif left != right:
            changes.append(label)
            reasons.append(f"{label} identity changed")
    if old.scope != new.scope:
        changes.append("scope")
        reasons.append("declared scopes differ")
    for report in (old, new):
        if not report.inventory or not report.inventory.complete:
            reasons.append("inventory incomplete or unrecorded")
        if report.run and report.run.completion.value != "complete":
            reasons.append("run did not complete")
    diagnostic_changes = []
    if old.diagnostics or new.diagnostics:
        for name in ("plan_digest", "implementation_digest"):
            left = getattr(old.diagnostics, name, None)
            right = getattr(new.diagnostics, name, None)
            if left != right or not left or not right:
                changes.append("diagnostics." + name)
                reasons.append("diagnostic " + name + " differs or is missing")
        left = (
            {(r.episode_id, r.kind, r.id): r for r in old.diagnostics.results}
            if old.diagnostics
            else {}
        )
        right = (
            {(r.episode_id, r.kind, r.id): r for r in new.diagnostics.results}
            if new.diagnostics
            else {}
        )
        for address in sorted(left.keys() | right.keys(), key=str):
            a, b = left.get(address), right.get(address)
            if a != b:
                diagnostic_changes.append(
                    {
                        "address": list(address),
                        "old": a.model_dump(mode="json") if a else None,
                        "new": b.model_dump(mode="json") if b else None,
                    }
                )
    before = {e.id: e for e in old.episodes}
    after = {e.id: e for e in new.episodes}
    if len(before) != len(old.episodes) or len(after) != len(new.episodes):
        raise ValueError("duplicate episode ids prevent unambiguous comparison")
    episode_changes, metric_changes = [], []
    for key in sorted(before.keys() & after.keys()):
        a, b = before[key], after[key]
        if a.eligibility != b.eligibility or a.score.score != b.score.score:
            # Report._decisions_agree refuses episodes without eligibility
            assert a.eligibility is not None
            assert b.eligibility is not None
            episode_changes.append(
                {
                    "episode_id": key,
                    "old_status": a.eligibility.status.value,
                    "new_status": b.eligibility.status.value,
                    "old_quality": a.score.score,
                    "new_quality": b.score.score,
                }
            )
        left, right = _metrics(a), _metrics(b)
        for address in sorted(left.keys() | right.keys(), key=str):
            av, bv = left.get(address), right.get(address)
            if av != bv:
                metric_changes.append(
                    {
                        "episode_id": key,
                        "address": list(address),
                        "old": av.model_dump(mode="json") if av else None,
                        "new": bv.model_dump(mode="json") if bv else None,
                    }
                )
    if before.keys() != after.keys():
        reasons.append("episode inventories differ")

    # Expiry/revocation can change a decision even when files/configuration agree.
    def decisions(report):
        return {f.id: f.calibration for f in report.findings if f.calibration}

    if content_digest(decisions(old)) != content_digest(decisions(new)):
        changes.append("calibration_outcome")
        reasons.append("calibration decisions changed")
    delta = None
    if (
        not reasons
        and old.readiness
        and new.readiness
        and old.readiness.score is not None
        and new.readiness.score is not None
    ):
        delta = new.readiness.score - old.readiness.score
    return Comparison(
        comparable=not reasons,
        reasons=sorted(set(reasons)),
        identity_changes=sorted(set(changes)),
        added_episodes=sorted(after.keys() - before.keys()),
        removed_episodes=sorted(before.keys() - after.keys()),
        episode_changes=episode_changes,
        metric_changes=metric_changes,
        readiness_delta=delta,
        diagnostic_changes=diagnostic_changes,
    )
