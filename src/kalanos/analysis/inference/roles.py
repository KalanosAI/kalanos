"""Resolve column names to taxonomy types through the dictionary.

Knowing a field is called `observation.state` is not the same as knowing it
means joint position — only `dictionary.yaml` carries that mapping,
and this module performs the lookup.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Iterable

# Internal
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.schema import ColumnRole


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def roles(columns: Iterable[str], dictionary: Dictionary) -> list[ColumnRole]:
    """Resolve every column name against the dictionary, in order.

    Parameters
    ----------
    columns : Iterable[str]
        Column names as their source spelled them.
    dictionary : Dictionary
        The taxonomy to resolve names against.

    Returns
    -------
    list[ColumnRole]
        One entry per name in `columns`, in the same order. A name that
        matches no entry, or more than one, resolves with `taxonomy_type=None`
        rather than being dropped, so the caller can still report it as
        `unmapped.<name>` instead of losing it silently.
    """

    results = []
    for name in columns:
        match = dictionary.resolve(name)

        if match.taxonomy_type is not None:
            confidence = 1.0
            evidence = [
                f"{name!r} normalised to {match.normalised.stem!r}, "
                f"matched {match.taxonomy_type!r}"
            ]
            group_hint = dictionary.entries[match.taxonomy_type].group_hint
        elif match.candidates:
            confidence = 0.0
            evidence = [
                f"{name!r} normalised to {match.normalised.stem!r}, "
                f"ambiguous between {', '.join(match.candidates)}"
            ]
            group_hint = None
        else:
            confidence = 0.0
            evidence = [
                f"{name!r} normalised to {match.normalised.stem!r}, "
                "matched nothing in the dictionary"
            ]
            group_hint = None

        results.append(
            ColumnRole(
                column=name,
                taxonomy_type=match.taxonomy_type,
                axis=match.normalised.axis,
                unit=match.normalised.unit,
                candidates=match.candidates,
                confidence=confidence,
                evidence=evidence,
                group_hint=group_hint,
            )
        )
    return results
