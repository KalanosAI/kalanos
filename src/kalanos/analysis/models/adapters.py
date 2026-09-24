"""What an adapter is asked for: the facts it can declare and the contract it satisfies.

This module holds the contract only. A format's adapter lives in its own package
and is found through the discovery mechanism in `kalanos.analysis.adapters`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

# External
from pydantic import BaseModel
from upath import UPath

# Internal
from kalanos.analysis.models.domain import Episode
from kalanos.analysis.models.errors import KalanosError
from kalanos.analysis.models.paths import AnyPath
from kalanos.analysis.models.schema import RefusalCode, SourceSchema, UnresolvedSource


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class DatasetInfo(BaseModel):
    """Facts an adapter can state about a path without reading its episodes.

    Attributes
    ----------
    adapter : str
        The adapter's name.
    path : UPath
        The path this info describes.
    episode_count : int or None
        How many episodes the path holds, when the format declares it.
    nominal_rate_hz : float or None
        The format's declared sampling rate, when it declares one.
    robot_type : str or None
        The robot or platform the format declares, when it declares one.
    """

    adapter: str
    path: AnyPath
    episode_count: int | None = None
    nominal_rate_hz: float | None = None
    robot_type: str | None = None


class AdapterRefusal(Exception):
    """Raised by an adapter that bid on a path and then could not read it.

    Parameters
    ----------
    path : UPath
        The file the adapter could not read.
    reason : str
        Why the file was refused.
    schema_so_far : SourceSchema or None
        Whatever `inference` had resolved before the refusal, if any.
    code : RefusalCode
        The machine-readable counterpart to `reason`.
    """

    def __init__(
        self,
        path: UPath,
        reason: str,
        *,
        schema_so_far: SourceSchema | None = None,
        code: RefusalCode = RefusalCode.UNSPECIFIED,
    ) -> None:
        self.path = path
        self.reason = reason
        self.schema_so_far = schema_so_far
        self.code = code
        super().__init__(f"{path}: {reason}")

    def as_unresolved(self) -> UnresolvedSource:
        """Convert this refusal into an `UnresolvedSource` for the report.

        The evidence `inference` gathered before the refusal would otherwise be
        thrown away by the caller that eventually drives adapters.

        Returns
        -------
        UnresolvedSource
            Carries `path`, `reason`, `code`, and `schema_so_far` —
            or an empty `SourceSchema` when nothing had been resolved yet.
        """

        return UnresolvedSource(
            path=self.path,
            schema_so_far=self.schema_so_far or SourceSchema(),
            reason=self.reason,
            code=self.code,
        )


class AdapterTie(KalanosError):
    """Raised when two or more adapters bid identically on one path.

    Parameters
    ----------
    path : UPath
        The file the tied adapters bid on.
    bids : dict[str, float]
        Every adapter tied at the maximum, name to bid.
    """

    def __init__(self, path: UPath, bids: dict[str, float]) -> None:
        self.path = path
        self.bids = bids
        names = list(bids)
        joined = (
            f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]
        )
        bid = next(iter(bids.values()))
        super().__init__(
            f"{path}: {joined} bid {bid:.2f} each; one of their detect() "
            "methods is wrong"
        )


@runtime_checkable
class Adapter(Protocol):
    """What Kalanos asks of anything that reads a format.

    An adapter reads one family of formats and produces episodes; it never
    applies a threshold or decides whether the data is good.
    """

    name: str

    def detect(self, path: UPath) -> float:
        """Say how confident this adapter is that it can read the path.

        Parameters
        ----------
        path : UPath
            The path to consider.

        Returns
        -------
        float
            A confidence clamped to 0.0-1.0 by contract. 0.0 means
            "I cannot read this".
        """

        ...

    def describe(self, path: UPath) -> DatasetInfo:
        """Report what is known about the path without reading its episodes.

        Parameters
        ----------
        path : UPath
            The path to describe.

        Returns
        -------
        DatasetInfo
            The facts the format declares. `describe` must not read the
            episodes to fill these in.
        """

        ...

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield the path's episodes, lazily.

        Parameters
        ----------
        path : UPath
            The path to read.
        sample : int or None
            When given, stop after this many episodes rather than reading
            the whole path.

        Returns
        -------
        Iterator[Episode]
            The path's episodes. Must be re-iterable: calling `episodes`
            twice yields the same episodes in the same order.
        """

        ...
