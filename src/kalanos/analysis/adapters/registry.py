"""The @adapter decorator and the registry it populates for one process.

An adapter registers itself once, at import time, by decorating its class.
This is the local-registration half of discovery —
the path a single plugin file uses, as opposed to an entry point.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.models.adapters import Adapter


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# Populated by every @adapter-decorated class as its module is imported.
# Holds ready instances, not classes, so locally_registered() needs no
# construction step.
_LOCAL: list[Adapter] = []


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def adapter(cls: type) -> type:
    """Register an adapter class for this process, and return it unchanged.

    Parameters
    ----------
    cls : type
        The class to instantiate and register. Must satisfy `Adapter`.

    Returns
    -------
    type
        `cls`, unchanged.

    Raises
    ------
    TypeError
        If instantiating `cls` with no arguments does not satisfy `Adapter`.
    """

    instance = cls()
    if not isinstance(instance, Adapter):
        raise TypeError(f"{cls.__name__} does not satisfy the Adapter protocol")
    _LOCAL.append(instance)
    return cls


def locally_registered() -> list[Adapter]:
    """List every adapter registered through the decorator, in registration order.

    Returns
    -------
    list[Adapter]
        Every locally registered adapter instance.
    """

    # Copy so callers can't mutate the live registry.
    return list(_LOCAL)


def clear_local_registry() -> None:
    """Forget every locally registered adapter.

    Loading a plugin file has a process-wide side effect through `@adapter`;
    this undoes it, which a test needs between plugin loads.
    """

    _LOCAL.clear()
