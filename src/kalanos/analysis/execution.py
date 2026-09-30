"""What the current run intends to execute, visible to adapters.

The pipeline sets the execution tier for the duration of a run; an adapter
that knows how to read less at a lower tier asks `current_tier()` before it
reads. A metadata tier is a promise that numeric payloads are not read, and
that promise has to be kept at the storage boundary, not after the values are
already in memory.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Internal
from kalanos.analysis.models.provenance import ExecutionTier


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

_TIER: ContextVar[ExecutionTier] = ContextVar(
    "kalanos_tier", default=ExecutionTier.STANDARD
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def current_tier() -> ExecutionTier:
    """The tier the current run is executing at; standard outside any run."""

    return _TIER.get()


def reads_numeric_payloads() -> bool:
    """Whether the current tier reads numeric payloads at all."""

    return current_tier() != ExecutionTier.METADATA


@contextmanager
def use_tier(tier: ExecutionTier) -> Iterator[None]:
    """Run the body at `tier`; restored on exit, even on error."""

    token = _TIER.set(tier)
    try:
        yield
    finally:
        _TIER.reset(token)


__all__ = ["current_tier", "reads_numeric_payloads", "use_tier"]
