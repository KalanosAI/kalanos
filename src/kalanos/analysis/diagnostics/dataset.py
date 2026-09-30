"""Dataset-level execution over explicitly comparable, named cohorts."""

import hashlib
import json
from collections import Counter

from kalanos.analysis.diagnostics.common import (
    Unavailable,
    axis,
    numeric,
    result,
    select,
    validated,
)


def cohort(episodes, spec, decisions):
    """Measure occupancy, dimension, trajectory repetition and labelled phase balance.

    No inferred task labels, fitted transition model or universal good/bad
    threshold is used. Raw finite observations and passing-only summaries remain
    separate; passing alone does not establish a representative training set.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise Unavailable("dataset diagnostics require kalanos[numeric]") from exc
    lookup = {e.id: e for e in episodes}
    missing = sorted(set(spec.episode_ids) - lookup.keys())
    if missing:
        raise Unavailable("cohort episodes absent: " + ", ".join(missing))
    items, excluded, total_rows, finite_rows, raw_rows = [], [], 0, 0, {}
    for identifier in spec.episode_ids:
        episode = lookup[identifier]
        if episode.tasks is not None and spec.task not in episode.tasks:
            raise Unavailable(
                f"cohort task does not match declared task of {identifier}"
            )
        vectors, reference_times, reference_rows = [], None, None
        first_stream = None
        for feature in spec.features:
            stream, channel = select(episode, feature.channel)
            b = validated(
                channel, ["identity", "quantity", "unit", "representation"], "cohort"
            )
            if b.unit != feature.unit or b.representation.value != "continuous":
                raise Unavailable(
                    "cohort requires matching native units and continuous "
                    "representations"
                )
            rows, positions, times = axis(stream)
            if reference_times is not None and (
                reference_times != times
                or reference_rows != rows
                or stream is not first_stream
            ):
                raise Unavailable(
                    "cohort features must share one resolved stream and source grid"
                )
            first_stream = stream
            reference_times, reference_rows = times, rows
            values = numeric(stream, channel, positions)
            vectors.append(
                [
                    (v - feature.lower) / (feature.upper - feature.lower)
                    if v is not None
                    else float("nan")
                    for v in values
                ]
            )
        x = np.array(vectors, dtype=float).T
        total_rows += len(x)
        valid = np.isfinite(x).all(axis=1)
        finite_rows += int(valid.sum())
        raw_rows[identifier] = (reference_rows, valid.tolist())
        if not valid.all():
            excluded.append(
                {
                    "episode_id": identifier,
                    "reason": "nonfinite rows; excluded from trajectory comparisons",
                    "invalid_rows": int((~valid).sum()),
                }
            )
        if not valid.any():
            continue
        t = np.array(reference_times)
        trajectory = None
        if valid.all():
            trajectory = np.stack(
                [
                    np.interp(
                        np.linspace(0, 1, spec.trajectory_points), t / t[-1], x[:, i]
                    )
                    for i in range(x.shape[1])
                ],
                axis=1,
            )
        digest = (
            hashlib.sha256(
                json.dumps(
                    [reference_times, x.tolist()],
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if valid.all()
            else None
        )
        items.append(
            {
                "id": identifier,
                "x": x[valid],
                "trajectory": trajectory,
                "duration": float(t[-1]),
                "digest": digest,
            }
        )
    if not items:
        raise Unavailable("cohort has no finite observations")

    def summarize(selected):
        """Describe a named subset using fixed normalization and histogram edges."""
        if not selected:
            return {
                "episodes": 0,
                "samples": 0,
                "effective_dimensionality": None,
                "features": [],
            }
        values = np.concatenate([item["x"] for item in selected])
        features = []
        for column in values.T:
            hist, _ = np.histogram(column, bins=np.linspace(0, 1, spec.bins + 1))
            probabilities = hist[hist > 0] / max(1, hist.sum())
            features.append(
                {
                    "bin_counts": hist.tolist(),
                    "outside_range": int(np.sum((column < 0) | (column > 1))),
                    "occupied_bins": int(np.sum(hist > 0)),
                    "entropy_nats": float(
                        -np.sum(probabilities * np.log(probabilities))
                    ),
                    "constant": bool(np.ptp(column) == 0),
                }
            )
        eig = (
            np.linalg.eigvalsh(np.atleast_2d(np.cov(values, rowvar=False)))
            if len(values) > 1
            else np.array([0.0])
        )
        eig = np.maximum(eig, 0)
        dimension = (
            float(eig.sum() ** 2 / (eig * eig).sum()) if (eig * eig).sum() > 0 else None
        )
        return {
            "episodes": len(selected),
            "samples": len(values),
            "effective_dimensionality": dimension,
            "features": features,
        }

    similar, exact, comparisons, total_pairs = [], [], 0, 0
    trajectories = [i for i in items if i["trajectory"] is not None]
    for a, left in enumerate(trajectories):
        for right in trajectories[a + 1 :]:
            total_pairs += 1
            if comparisons >= spec.max_pair_comparisons:
                continue
            comparisons += 1
            ids = [left["id"], right["id"]]
            if left["digest"] == right["digest"]:
                exact.append(ids)
            duration_difference = abs(left["duration"] - right["duration"]) / max(
                left["duration"], right["duration"]
            )
            distance = float(
                np.sqrt(np.mean((left["trajectory"] - right["trajectory"]) ** 2))
            )
            if (
                distance <= spec.similarity_rmse
                and duration_difference <= spec.duration_tolerance_fraction
            ):
                similar.append(
                    {
                        "episodes": ids,
                        "normalized_rmse": distance,
                        "relative_duration_difference": duration_difference,
                    }
                )
    features = np.stack([np.mean(item["x"], axis=0) for item in items])
    med = np.median(features, axis=0)
    mad = np.median(np.abs(features - med), axis=0)
    usable = mad > 0
    outliers = [
        {
            "episode_id": item["id"],
            "robust_mean_distance": float(
                np.sqrt(
                    np.mean(((features[i, usable] - med[usable]) / mad[usable]) ** 2)
                )
            )
            if len(items) >= 5 and usable.any()
            else None,
        }
        for i, item in enumerate(items)
    ]
    uncertainty = {
        "status": "unavailable",
        "reason": "independent session identities not established",
    }
    if spec.independent_sessions and spec.sessions:
        names = sorted(set(spec.sessions.values()))
        groups = [
            np.mean(
                [
                    features[i]
                    for i, item in enumerate(items)
                    if spec.sessions[item["id"]] == name
                ],
                axis=0,
            )
            for name in names
            if any(spec.sessions[item["id"]] == name for item in items)
        ]
        if len(groups) >= 3:
            group_means = np.stack(groups)
            rng = np.random.default_rng(spec.bootstrap_seed)
            draws = np.stack(
                [
                    group_means[rng.integers(0, len(groups), len(groups))].mean(axis=0)
                    for _ in range(spec.bootstrap_resamples)
                ]
            )
            uncertainty = {
                "status": "computed",
                "method": "session_mean_percentile_bootstrap_v1",
                "session_count": len(groups),
                "resamples": spec.bootstrap_resamples,
                "seed": spec.bootstrap_seed,
                "normalized_mean": group_means.mean(axis=0).tolist(),
                "interval_95_percentile": np.quantile(
                    draws, [0.025, 0.975], axis=0
                ).T.tolist(),
                "interpretation": (
                    "declared independent sessions, equal session weight; "
                    "descriptive approximate intervals, not detector calibration"
                ),
            }
        else:
            uncertainty["reason"] = "fewer than three sessions with finite observations"
    phases = Counter()
    labelled = 0
    for label in spec.phase_labels:
        rows, valid = raw_rows[label.episode_id]
        addressed = [
            i for i, row in enumerate(rows) if label.start <= row < label.end_exclusive
        ]
        if len(addressed) != label.end_exclusive - label.start:
            raise Unavailable("phase label contains absent source rows")
        count = sum(valid[i] for i in addressed)
        labelled += count
        phases[label.phase] += count
    measured = result(
        "cohort",
        spec,
        measurements={
            "raw_finite": summarize(items),
            "passing_episodes_only": summarize(
                [i for i in items if decisions.get(i["id"]) == "pass"]
            ),
            "total_source_rows": total_rows,
            "finite_source_rows": finite_rows,
            "exact_feature_trajectory_duplicates": exact,
            "similar_trajectories": similar,
            "cohort_outliers": outliers,
            "uncertainty": uncertainty,
            "pair_comparisons": comparisons,
            "unexamined_pairs": total_pairs - comparisons,
            "phase_balance": {
                "labelled_samples": labelled,
                "unlabelled_finite_samples": finite_rows - labelled,
                "fractions_of_labelled": {k: v / labelled for k, v in phases.items()}
                if labelled
                else None,
            },
        },
        evidence={
            "specification": spec.model_dump(),
            "excluded_trajectories": excluded,
            "trajectory_digest_scope": (
                "selected normalized feature values and relative "
                "timestamps, not source bytes"
            ),
            "phase_index_space": "source rows of the cohort feature stream",
            "interpretation": (
                "similarity is not semantic equivalence; rare behavior "
                "and constrained tasks are not defects; no automatic "
                "deletion"
            ),
        },
    )

    if total_pairs > comparisons:
        measured.availability = "skipped"
        measured.reason = (
            "pair-comparison budget reached; partial descriptive results retained"
        )
    return measured


def run_dataset_diagnostics(episodes, specs, decisions, execute):
    """Execute dataset-level checks independently of the episode score rollup."""
    return [
        execute(
            "cohort", spec, None, lambda spec=spec: cohort(episodes, spec, decisions)
        )
        for spec in specs
    ]
