"""The metrics that flagged channels or streams, ready to display.

The card and `kalanos inspect` both build their flagged-channel rows here,
so the two never disagree on which metrics show, in what order, or how a count reads.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass

# Internal
from kalanos.analysis.metrics.registry import metric_label
from kalanos.analysis.models.metrics import Level
from kalanos.analysis.models.scoring import MetricRollup
from kalanos.analysis.reporting.families import FAMILY_LABELS


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class FlaggedRow:
    """One metric that flagged at least one channel or stream.

    Attributes
    ----------
    label : str
        The metric's display name.
    family : str
        The metric's family display name, or its id when it has none.
    subjects : str
        How many subjects it flagged, out of how many it graded,
        such as `"1 of 12 channels"`.
    episodes : str or None
        How many episodes it flagged a subject in, such as `"3 episodes"`;
        `None` for one episode's rollup.
    worst : float or None
        The worst subject's points in one episode's rollup; `None` at dataset level.
    """

    label: str
    family: str
    subjects: str
    episodes: str | None
    worst: float | None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _counted(n: int, noun: str) -> str:
    """Pluralise `noun` after `n`, which counts it.

    Parameters
    ----------
    n : int
        The count.
    noun : str
        The singular noun.

    Returns
    -------
    str
        `"1 channel"` or `"3 channels"`.
    """

    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def flagged_rows(metrics: dict[str, MetricRollup]) -> list[FlaggedRow]:
    """Pick and order the metrics that flagged a channel or stream.

    Parameters
    ----------
    metrics : dict[str, MetricRollup]
        A dataset's or an episode's `ScoreResult.metrics`, keyed `family.metric`.

    Returns
    -------
    list[FlaggedRow]
        One row per channel or stream metric with a flagged subject, the
        metrics flagged in the most episodes first, then by label. Episode
        metrics have no channels to count and are left out.
    """

    rows: list[tuple[int, FlaggedRow]] = []
    for key, rollup in metrics.items():
        if rollup.n_flagged == 0 or rollup.level not in (Level.CHANNEL, Level.STREAM):
            continue
        family, _, name = key.partition(".")
        noun = "channel" if rollup.level == Level.CHANNEL else "stream"
        n_episodes = rollup.n_episodes_flagged
        row = FlaggedRow(
            label=metric_label(name),
            family=FAMILY_LABELS.get(family, family),
            subjects=f"{rollup.n_flagged} of {_counted(rollup.n_graded, noun)}",
            episodes=None if n_episodes is None else _counted(n_episodes, "episode"),
            worst=rollup.points,
        )
        rows.append((n_episodes or 0, row))
    rows.sort(key=lambda pair: (-pair[0], pair[1].label))
    return [row for _, row in rows]
