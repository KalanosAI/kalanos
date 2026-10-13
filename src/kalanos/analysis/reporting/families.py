"""Display names and number formatting for the dataset's family scores.

The card, the HTML report and `kalanos inspect` all show family scores
through this module, so the three never disagree on a name, an order or a rounding.
"""

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
