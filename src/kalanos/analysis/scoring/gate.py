"""The dataset gate: derive the blocked list and the coverage from decided episodes.

Since schema 7 this module decides nothing. `scoring.eligibility` decides each
episode once; the gate lists the blocked ones, caps the compatibility letter by
their share, reports task/dataset-wide patterns as descriptive traits, and
states what the grade rests on. Prevalence exempts nothing: a blocking finding
on every episode is a blocking finding on every episode.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from kalanos.analysis.coverage import state_of
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.eligibility import (
    Consequence,
    EligibilityStatus,
    EpisodeEligibility,
)

# Internal
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.policy import GatePolicy, Policy
from kalanos.analysis.models.report import (
    Coverage,
    DatasetTrait,
    FailingEpisode,
    Gate,
    GradedEpisode,
    NotObservable,
    TaskTrait,
)
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult
from kalanos.analysis.scoring.score import rollup


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Letters from best to worst, so "worse of two" is a max over this order.
_ORDER = [Grade.A, Grade.B, Grade.C, Grade.D, Grade.F]

_GRADED = {MetricStatus.GOOD, MetricStatus.WARNING, MetricStatus.CRITICAL}

# A metric that could not grade at least this share of its results is listed as
# not observable, so a reader sees what a grade does not rest on.
_NOT_OBSERVABLE_SHARE = 0.5


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def finding_key(finding: Finding) -> str:
    """Name a finding by where it sits and what fired: `stream[/channel].metric`."""

    where = finding.stream or "episode"
    if finding.instance:
        where = f"{where}[{finding.instance}]"
    if finding.channel:
        where = f"{where}/{finding.channel}"
    return f"{where}.{finding.metric_id}"


def critical_keys_by_episode(findings: Iterable[Finding]) -> dict[str, set[str]]:
    """Each episode's blocking findings, keyed by `finding_key`.

    Blocking means `consequence == BLOCK`; severity alone does not block.
    """

    keys: dict[str, set[str]] = defaultdict(set)
    for finding in findings:
        if finding.consequence == Consequence.BLOCK:
            keys[finding.episode_id].add(finding_key(finding))
    return dict(keys)


def task_traits(
    critical: Mapping[str, set[str]],
    task_of: Mapping[str, str | None],
    min_episodes: int,
) -> dict[str, tuple[str, int]]:
    """Critical findings on every episode of one task and on none of any other.

    Parameters
    ----------
    critical : Mapping[str, set[str]]
        Each episode's critical finding keys.
    task_of : Mapping[str, str or None]
        Each graded episode's task, or `None` when it has none.
    min_episodes : int
        The fewest episodes a task needs for a shared finding to count.

    Returns
    -------
    dict[str, tuple[str, int]]
        Each trait's finding key, mapped to its task and that task's size.
    """

    size = Counter(task for task in task_of.values() if task is not None)
    on: dict[str, set[str]] = defaultdict(set)
    for episode, keys in critical.items():
        for key in keys:
            on[key].add(episode)
    traits = {}
    for key, episodes in on.items():
        tasks = {task_of.get(episode) for episode in episodes}
        if len(tasks) != 1:
            continue
        [task] = tasks
        if task is None:
            continue
        if len(episodes) == size[task] >= min_episodes:
            traits[key] = (task, size[task])
    return traits


def dataset_traits(
    critical: Mapping[str, set[str]],
    episodes: Iterable[str],
    min_episodes: int,
    min_share: float = 1.0,
) -> set[str]:
    """Critical findings on (nearly) every graded episode of the dataset.

    Such a finding describes the recording setup, not some episodes being worse
    than others: dropping episodes cannot remove it, so the gate, which exists to
    catch the episodes worth dropping, does not count it.

    Parameters
    ----------
    critical : Mapping[str, set[str]]
        Each episode's critical finding keys.
    episodes : Iterable[str]
        Every graded episode, whether or not it has a finding.
    min_episodes : int
        The fewest episodes a dataset needs for "every episode" to mean something.
    min_share : float
        The share of episodes, from 0 to 1, a finding must be on. The gate passes
        one minus its own no-cap allowance: a finding missing from no more
        episodes than the gate would let fail is on every episode that matters.

    Returns
    -------
    set[str]
        The finding keys present in at least `min_share` of the graded episodes.
    """

    ids = list(episodes)
    if len(ids) < min_episodes:
        return set()
    needed = math.ceil(min_share * len(ids) - 1e-9)
    counts = Counter(key for episode in ids for key in critical.get(episode, set()))
    return {key for key, count in counts.items() if count >= needed}


def cap_for(share: float, gate: GatePolicy) -> Grade | None:
    """The best letter a dataset with this share of failing episodes may get."""

    for row in gate.caps:
        if share <= row.max_failing_share:
            return Grade(row.letter) if row.letter else None
    return Grade.F  # unreachable: the last row reaches 1.0


def worse(first: Grade | None, second: Grade | None) -> Grade | None:
    """The worse of two letters, treating `None` as no constraint."""

    if first is None or second is None:
        return first or second
    return max(first, second, key=_ORDER.index)


def _results(episode: GradedEpisode) -> Iterable[tuple[str, MetricResult]]:
    """Every metric result in one episode, at every level, with its metric name."""

    yield from episode.metrics.items()
    for stream in episode.streams:
        yield from stream.metrics.items()
        for channel in stream.channels:
            yield from channel.metrics.items()


def coverage(episodes: Sequence[GradedEpisode], score: ScoreResult) -> Coverage:
    """What a dataset's grade rests on."""

    graded_per_episode = []
    total: Counter[str] = Counter()
    not_applicable: Counter[str] = Counter()
    reason: dict[str, str] = {}
    for episode in episodes:
        graded = 0
        for name, result in _results(episode):
            if state_of(result) == Availability.NOT_APPLICABLE:
                continue
            total[name] += 1
            if result.status in _GRADED:
                graded += 1
            elif result.status == MetricStatus.NOT_APPLICABLE:
                not_applicable[name] += 1
                reason.setdefault(name, str(result.evidence.get("reason", "")))
        graded_per_episode.append(graded)
    unobservable = [
        NotObservable(
            metric=name,
            reason=reason[name],
            share=not_applicable[name] / total[name],
        )
        for name in sorted(not_applicable)
        if not_applicable[name] / total[name] >= _NOT_OBSERVABLE_SHARE and reason[name]
    ]
    return Coverage(
        families_graded=sorted(score.families),
        graded_checks_per_episode=(
            statistics.median(graded_per_episode) if graded_per_episode else None
        ),
        not_observable=unobservable,
    )


def _summary(
    readiness: float | None,
    passing_quality: float | None,
    failing: int,
    total: int,
    cover: Coverage,
    n_dataset_traits: int = 0,
) -> str:
    """One plain-language line: readiness, why, and what it rests on."""

    if readiness is None:
        head = "Readiness undefined"
    elif failing:
        head = (
            f"Readiness {readiness:.0f}/100: {failing} of {total} episodes "
            f"({failing / total:.0%}) have blocking findings"
            + (
                f"; {passing_quality:.0f}/100 after excluding them"
                if passing_quality is not None
                else ""
            )
        )
    else:
        head = f"Readiness {readiness:.0f}/100: no blocking episodes"
    families = " and ".join(cover.families_graded) or "nothing"
    checks = cover.graded_checks_per_episode
    rests = f"graded on {families}" + (
        f", {checks:.0f} checks per episode" if checks is not None else ""
    )
    gaps = "; ".join(f"{item.metric}: {item.reason}" for item in cover.not_observable)
    traits = (
        f" {n_dataset_traits} finding{'s' if n_dataset_traits != 1 else ''} present in"
        " (nearly) every episode reported as dataset traits, not failures."
        if n_dataset_traits
        else ""
    )
    return f"{head}.{traits} {rests[0].upper()}{rests[1:]}." + (
        f" Not observable — {gaps}." if gaps else ""
    )


def apply_gate(
    episodes: Sequence[GradedEpisode],
    findings: Sequence[Finding],
    score: ScoreResult,
    policy: Policy,
    *,
    decisions: Mapping[str, EpisodeEligibility],
    readiness_score: float | None = None,
    passing_quality: float | None = None,
) -> tuple[ScoreResult, Gate | None]:
    """Cap the dataset's letter by its share of failing episodes.

    Parameters
    ----------
    episodes : Sequence[GradedEpisode]
        Every graded episode.
    findings : Sequence[Finding]
        Every finding across them.
    score : ScoreResult
        The dataset's rolled-up score: the mean of its episodes.
    policy : Policy
        The policy; without a `gate`, `score` is returned unchanged.
    decisions : Mapping[str, EpisodeEligibility]
        Every episode's decided eligibility, keyed by id. The gate derives
        its blocked list from these.
    readiness_score, passing_quality : float or None
        Already computed by `scoring.eligibility`, for the summary line.

    Returns
    -------
    tuple[ScoreResult, Gate or None]
        The dataset score with the compatibility letter and train-readiness,
        and the gate; `(score, None)` without a gate.
    """

    if policy.gate is None:
        return score, None

    graded = [episode for episode in episodes if episode.score.score is not None]
    task_of = {
        episode.id: next((t for t in (episode.tasks or []) if t.strip()), None)
        for episode in graded
    }
    critical = critical_keys_by_episode(findings)
    min_episodes = policy.gate.task_trait_min_episodes
    allowance = next(
        (row.max_failing_share for row in policy.gate.caps if row.letter is None), 0.0
    )
    # Traits are reported, not set aside: `everywhere` and `traits` describe
    # patterns, and every episode in `critical` stays blocked.
    everywhere = dataset_traits(critical, task_of, min_episodes, 1.0 - allowance)
    traits = task_traits(critical, task_of, min_episodes)

    # The blocked list is the eligibility's, not the gate's own reading of
    # the findings, so the two can never disagree.
    failing = [
        FailingEpisode(
            episode_id=episode.id,
            reasons=sorted(
                r.id
                for r in decisions[episode.id].reasons
                if r.status == EligibilityStatus.BLOCKED
            ),
        )
        for episode in episodes
        if decisions[episode.id].status == EligibilityStatus.BLOCKED
    ]
    total = len(episodes)
    share = len(failing) / total if total else 0.0
    cap = cap_for(share, policy.gate)
    grade = worse(score.grade, cap)

    pruned_score = None
    if failing:
        failing_ids = {item.episode_id for item in failing}
        kept = [e.score for e in graded if e.id not in failing_ids]
        if kept:
            pruned_score = rollup(Level.DATASET, kept, policy=policy).score

    cover = coverage(graded, score)
    gate = Gate(
        n_episodes=total,
        failing_episodes=failing,
        failing_share=share,
        task_traits=[
            TaskTrait(task=task, finding=key, n_episodes=size)
            for key, (task, size) in sorted(traits.items())
        ],
        dataset_traits=[
            DatasetTrait(
                finding=key,
                n_episodes=total,
                n_with_finding=sum(key in critical.get(e, set()) for e in task_of),
            )
            for key in sorted(everywhere)
        ],
        uncapped_grade=score.grade,
        cap=cap,
        pruned_score=pruned_score,
        coverage=cover,
        summary=_summary(
            readiness_score,
            passing_quality,
            len(failing),
            total,
            cover,
            len(everywhere),
        ),
    )
    # Dataset-level train_ready is not the gate's to decide: assembly sets it
    # from the counts and the inventory, gate or no gate.
    gated = score.model_copy(update={"grade": grade})
    return gated, gate


__all__ = [
    "apply_gate",
    "cap_for",
    "coverage",
    "dataset_traits",
    "task_traits",
    "worse",
]
