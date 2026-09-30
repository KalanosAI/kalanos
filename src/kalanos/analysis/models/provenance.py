"""Provenance: who produced a report, from what, under which configuration.

An installed package version cannot distinguish a release branch from its
ancestor, and a dataset name cannot say which bytes were read. These models
record enough identity that two reports on the same data can be told apart by
what changed — source, binding, metric, policy, requirements or execution —
rather than by guessing.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

# External
from pydantic import BaseModel, Field, model_validator


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Producer(BaseModel):
    """The software that wrote a report.

    Attributes
    ----------
    package : str
        The distribution name, `kalanos`.
    version : str
        The installed package version.
    revision : str or None
        The git commit or build revision, when the build recorded one.
    adapters : dict[str, str]
        Each adapter used, mapped to the version of the package it came from.
    metrics : dict[str, str]
        Each metric implementation used, mapped to its declared version.
    """

    package: str = "kalanos"
    version: str
    revision: str | None = None
    adapters: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, str] = Field(default_factory=dict)


class HashScope(str, Enum):
    """What a source hash covers, so a cheap hash is never mistaken for a deep one."""

    # fmt: off
    METADATA = "metadata"  # Manifest/metadata files only; media bytes not read
    CONSUMED = "consumed"  # Every input byte this run actually read
    COMPLETE = "complete"  # Every file under the source, read or not
    # fmt: on


class SourceEvidence(BaseModel):
    """Identity of the inputs, with the scope and method stated.

    Attributes
    ----------
    algorithm : str
        The hash algorithm, e.g. `sha256`.
    canonicalization : str
        How inputs were ordered and encoded before hashing, versioned.
    scope : HashScope
        What the digest covers.
    digest : str or None
        The hex digest, or `None` when nothing was hashed.
    covered_inputs : int
        How many inputs contributed to the digest.
    complete : bool
        Whether the digest covers everything `scope` claims. A run cut short
        records `False` rather than a partial digest posing as a full one.
    revision : str or None
        A source-declared revision (a Hub commit, say), recorded as declared.
    """

    algorithm: str = "sha256"
    canonicalization: str = "sorted-relative-path-size-v1"
    scope: HashScope = HashScope.METADATA
    digest: str | None = None
    covered_inputs: int = 0
    complete: bool = True
    revision: str | None = None


class ConfigIdentity(BaseModel):
    """One configuration input's identity: what it was called and what it contained.

    Attributes
    ----------
    id : str
        The declared identifier, e.g. `numeric-core-v1`.
    digest : str or None
        A content digest of the resolved values, or `None` when not computed.
    origin : str or None
        Where it came from: a bundle path, `builtin`, an argument.
    """

    id: str
    digest: str | None = None
    origin: str | None = None


class RunCompletion(str, Enum):
    """Whether the run saw everything it set out to see."""

    # fmt: off
    COMPLETE = "complete"  # Every enumerated input reached a decision
    PARTIAL  = "partial"   # Some inputs were not attempted or not finished
    FAILED   = "failed"    # The run stopped before producing usable decisions
    # fmt: on


class ExecutionTier(str, Enum):
    """Which capabilities a run attempts. A tier never changes the requirements."""

    # fmt: off
    METADATA = "metadata"  # Manifest and prerequisite inspection only
    STANDARD = "standard"  # The implemented standard capability set
    FULL     = "full"      # Every installed supported capability, within limits
    # fmt: on


class RunInfo(BaseModel):
    """The identity of one analysis run.

    Attributes
    ----------
    id : str
        A run identifier unique to this execution.
    started_at, finished_at : datetime or None
        Wall-clock bounds, when timed.
    completion : RunCompletion
        Whether every input reached a decision.
    tier : ExecutionTier
        The execution tier attempted.
    source : SourceEvidence
        What was read and how it was identified.
    requirements, policy, binding, dictionary, execution : ConfigIdentity or None
        The configuration identities, when each was resolved. `binding` is
        the *effective* mapping after precedence; `bundle` is the declared
        bundle file, when one was given.
    sampling : dict[str, Any]
        Any sampling specification (seeds, strides) that affected results.
    """

    id: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    completion: RunCompletion = RunCompletion.COMPLETE
    tier: ExecutionTier = ExecutionTier.STANDARD
    source: SourceEvidence = Field(default_factory=SourceEvidence)
    requirements: ConfigIdentity | None = None
    policy: ConfigIdentity | None = None
    binding: ConfigIdentity | None = None
    dictionary: ConfigIdentity | None = None
    execution: ConfigIdentity | None = None
    bundle: ConfigIdentity | None = None
    sampling: dict[str, Any] = Field(default_factory=dict)


class FailedEpisode(BaseModel):
    """An episode the run knew of but could not grade.

    Attributes
    ----------
    id : str
        The episode, as the source named it.
    reason : str
        Why, in one line.
    """

    id: str
    reason: str


class Inventory(BaseModel):
    """Which episodes the run expected, loaded and failed.

    Attributes
    ----------
    expected : int or None
        Episodes the source declared, or `None` when it declared no count.
    loaded : int
        Episodes that reached grading.
    failed : list[FailedEpisode]
        Episodes that did not, with reasons.
    unresolved : int
        Episodes the source declared but the run never enumerated: the gap
        between a declared count and what was loaded or failed. They have no
        identities, so none are invented; they are counted as unknown.
    complete : bool
        Whether `loaded + failed` covers everything the source declared.
        `False` whenever `unresolved > 0`. A run that finished its walk has
        not thereby proved the inventory complete.
    notes : list[str]
        Anything odd about the accounting, e.g. more episodes loaded than
        declared.
    """

    expected: int | None = None
    loaded: int = 0
    failed: list[FailedEpisode] = Field(default_factory=list)
    unresolved: int = Field(default=0, ge=0)
    complete: bool = True
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _gap_means_incomplete(self) -> "Inventory":
        """An unresolved gap and a complete inventory cannot both be claimed."""

        if self.unresolved and self.complete:
            raise ValueError("inventory with unresolved episodes cannot be complete")
        return self


def content_digest(value: Any) -> str:
    """A deterministic sha256 over a JSON-serialisable configuration value.

    Parameters
    ----------
    value : Any
        Resolved configuration content: dictionaries, lists, scalars.

    Returns
    -------
    str
        The hex digest of the canonical JSON encoding (sorted keys, no spaces).
    """

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "ConfigIdentity",
    "ExecutionTier",
    "FailedEpisode",
    "HashScope",
    "Inventory",
    "Producer",
    "RunCompletion",
    "RunInfo",
    "SourceEvidence",
    "content_digest",
]
