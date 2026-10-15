"""Display names and number formatting for the dataset's family scores.

The card, the HTML report and `kalanos inspect` all show family scores,
and each family's count of low episodes and lowest episode score,
through this module, so the three never disagree on a name, an order or a rounding.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.models.scoring import FamilyTail


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Display order follows insertion order; a family missing here shows under its id.
# fmt: off
FAMILY_LABELS: dict[str, str] = {
    "timing":     "Temporal",
    "integrity":  "Integrity",
    "motion":     "Motion",
    "vision":     "Vision",
    "annotation": "Annotations",
}
# fmt: on


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def family_scores(families: dict[str, float]) -> list[tuple[str, str, float]]:
    """Order a score's families for display and name each one.

    Parameters
    ----------
    families : dict[str, float]
        Family id to 0-100 score, as `ScoreResult.families` holds them.

    Returns
    -------
    list[tuple[str, str, float]]
        One `(family id, display name, score)` per family:
        the named families in `FAMILY_LABELS` order,
        then every other family by id, displayed as its id.
        Empty when no family scored.
    """

    named = [
        (family, label, families[family])
        for family, label in FAMILY_LABELS.items()
        if family in families
    ]
    others = [
        (family, family, families[family])
        for family in sorted(families)
        if family not in FAMILY_LABELS
    ]
    return named + others


def format_family_score(score: float) -> str:
    """Format a family score as a whole number.

    Parameters
    ----------
    score : float
        The 0-100 score.

    Returns
    -------
    str
        The score rounded to the nearest integer, such as `"33"`.
    """

    return f"{score:.0f}"


def format_low_episodes(tail: FamilyTail) -> str:
    """Format how many of a family's episodes scored below the cutoff.

    Parameters
    ----------
    tail : FamilyTail
        The family's tail on the dataset score.

    Returns
    -------
    str
        The count out of the episodes that scored the family, such as `"1 of 13"`.
    """

    return f"{tail.n_below} of {tail.n_scored}"


def format_lowest(tail: FamilyTail) -> str:
    """Format a family's lowest episode score to one decimal.

    One decimal, so a lowest of 69.6 never reads as 70 beside a count of
    episodes below 70.

    Parameters
    ----------
    tail : FamilyTail
        The family's tail on the dataset score.

    Returns
    -------
    str
        The lowest episode score, such as `"61.4"`.
    """

    return f"{tail.lowest:.1f}"
