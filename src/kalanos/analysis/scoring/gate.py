"""The dataset gate: failing episodes cap the dataset's letter.

Averaging lets a minority of bad episodes hide: eight glitched episodes in fifty
still average to an A. The gate counts the episodes that fail — any metric graded
`critical`, per docs/METRICS.md's train-ready rule — and caps the dataset's letter
by their share, reporting the grade the dataset would get without them.

A critical finding shared by every episode of one task, and by no episode of any
other, is a task trait rather than a fault (a sweeping task never closes the
gripper, so its gripper state never moves) and does not fail those episodes.

Every gated report also states what its grade rests on — the families graded,
the checks per episode, and which metrics could not observe the data and why —
so an A on thin evidence reads differently from an A on thick.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

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
    Readiness,
    TaskTrait,
)
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult, Severity
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
    """Each episode's critical findings, keyed by `finding_key`."""

    keys: dict[str, set[str]] = defaultdict(set)
    for finding in findings:
        if finding.severity == Severity.CRITICAL:
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
        if len(tasks) != 1 or None in tasks:
            continue
        [task] = tasks
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


def readiness_of(
    episodes: Sequence[GradedEpisode], gate: Gate | None
) -> Readiness | None:
    """Readiness: blocking episodes contribute 0, the rest their quality score.

    Parameters
    ----------
    episodes : Sequence[GradedEpisode]
        Every graded episode.
    gate : Gate or None
        The gate's verdict, which names the blocking episodes; without a gate
        there is no blocking rule, and so no readiness.

    Returns
    -------
    Readiness or None
        The readiness summary, or `None` without a gate.
    """

    if gate is None:
        return None
    blocking = {item.episode_id for item in gate.failing_episodes}
    scores = [(e.id, e.score.score) for e in episodes if e.score.score is not None]
    passing = [score for episode, score in scores if episode not in blocking]
    evaluated = len(scores)
    return Readiness(
        score=sum(passing) / evaluated if evaluated else None,
        evaluated_episodes=evaluated,
        passing_episodes=len(passing),
        blocking_episodes=evaluated - len(passing),
        passing_quality=statistics.fmean(passing) if passing else None,
    )


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
            total[name] += 1
            if result.status in _GRADED:
                graded += 1
            elif result.status == MetricStatus.NOT_APPLICABLE:
                not_applicable[name] += 1
                reason.setdefault(name, str(result.evidence.get("reason", "")))
        graded_per_episode.append(graded)
    graded_names = {
        name
        for episode in episodes
        for name, result in _results(episode)
        if result.status in _GRADED
    }
    unobservable = [
        NotObservable(
            metric=name,
            reason=reason[name],
            share=not_applicable[name] / total[name],
        )
        for name in sorted(not_applicable)
        if name not in graded_names
        and not_applicable[name] / total[name] >= _NOT_OBSERVABLE_SHARE
        and reason[name]
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
        head = "Not graded: no episode could be evaluated"
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

    Returns
    -------
    tuple[ScoreResult, Gate or None]
        The dataset score with the gate's letter and train-readiness (its number
        stays the mean), and the gate's verdict; `(score, None)` without a gate.
    """

    if policy.gate is None:
        return score, None

    graded = [episode for episode in episodes if episode.score.score is not None]
    task_of = {
        episode.id: next((t for t in (episode.tasks or []) if t.strip()), None)
        for episode in graded
    }
    critical = {
        episode: keys
        for episode, keys in critical_keys_by_episode(findings).items()
        if episode in task_of
    }
    min_episodes = policy.gate.task_trait_min_episodes
    # The share of episodes a letter may lose for free; a finding missing from no
    # more than that share of episodes is on every episode that matters.
    allowance = next(
        (row.max_failing_share for row in policy.gate.caps if row.letter is None), 0.0
    )
    everywhere = dataset_traits(critical, task_of, min_episodes, 1.0 - allowance)
    remaining = {episode: keys - everywhere for episode, keys in critical.items()}
    traits = task_traits(remaining, task_of, min_episodes)
    set_aside = everywhere | traits.keys()
    failing = [
        FailingEpisode(episode_id=episode, reasons=sorted(keys - set_aside))
        for episode, keys in sorted(critical.items())
        if keys - set_aside
    ]
    total = len(graded)
    share = len(failing) / total if total else 0.0
    cap = cap_for(share, policy.gate)
    grade = worse(score.grade, cap)

    pruned_score = pruned_grade = pruned_ready = None
    if failing:
        failing_ids = {item.episode_id for item in failing}
        kept = [e.score for e in graded if e.id not in failing_ids]
        if kept:
            pruned = rollup(Level.DATASET, kept, policy=policy)
            pruned_score, pruned_grade = pruned.score, pruned.grade
            pruned_ready = pruned.train_ready

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
        pruned_grade=pruned_grade,
        train_ready_after_pruning=pruned_ready,
        coverage=cover,
        summary="",
    )
    ready = readiness_of(graded, gate)
    gate = gate.model_copy(
        update={
            "summary": _summary(
                ready.score if ready else None,
                ready.passing_quality if ready else None,
                len(failing),
                total,
                cover,
                len(everywhere),
            )
        }
    )
    gated = score.model_copy(
        update={
            "grade": grade,
            "train_ready": (
                None
                if score.train_ready is None
                else bool(score.train_ready and cap is None)
            ),
        }
    )
    return gated, gate


__all__ = [
    "apply_gate",
    "readiness_of",
    "cap_for",
    "coverage",
    "dataset_traits",
    "task_traits",
    "worse",
]
