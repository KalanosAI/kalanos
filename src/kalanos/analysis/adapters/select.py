"""Pick the adapter that reads one path: every adapter bids, the highest wins.

A library beside the stages, like `discover.py`. A tie at the maximum is a
bug in one of the tied `detect` methods, not a coin flip, so it raises.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from collections.abc import Sequence
from dataclasses import dataclass

# External
from upath import UPath

# Internal
from kalanos.analysis.adapters.discover import LoadedAdapter
from kalanos.analysis.models.adapters import Adapter, AdapterTie


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Selection:
    """The adapter chosen to read one path.

    Attributes
    ----------
    path : UPath
        The path this selection was made for.
    name : str
        The winning adapter's name.
    adapter : Adapter
        The winning adapter instance.
    confidence : float
        The bid it won with.
    """

    path: UPath
    name: str
    adapter: Adapter
    confidence: float


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def select_adapter(path: UPath, adapters: Sequence[LoadedAdapter]) -> Selection | None:
    """Ask every adapter to bid on `path`, and pick the highest bidder.

    Parameters
    ----------
    path : UPath
        The path to select an adapter for.
    adapters : Sequence[LoadedAdapter]
        Every adapter discovery found.

    Returns
    -------
    Selection or None
        The winning adapter, or `None` when every bid was zero or negative.

    Raises
    ------
    AdapterTie
        If more than one adapter shares the maximum bid.
    """

    # Keyed by LoadedAdapter: discovery deliberately returns duplicate names,
    # and a dict keyed by name would collapse two distinct adapters sharing
    # one into a single, wrong bid.
    bids: list[tuple[LoadedAdapter, float]] = []
    for loaded in adapters:
        try:
            bid = loaded.adapter.detect(path)
        except Exception as exc:
            logger.warning(
                "%s: %s.detect() raised %s: %s",
                path,
                loaded.name,
                type(exc).__name__,
                exc,
            )
            bid = 0.0
        if bid > 0.0:
            bids.append((loaded, bid))

    if not bids:
        return None

    best = max(bid for _, bid in bids)
    winners = [loaded for loaded, bid in bids if bid == best]
    if len(winners) > 1:
        raise AdapterTie(path, {loaded.name: best for loaded in winners})

    winner = winners[0]
    return Selection(
        path=path, name=winner.name, adapter=winner.adapter, confidence=best
    )
