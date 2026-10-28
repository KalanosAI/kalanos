"""Read reports written under schema 6.3, 6.4 and 6.5 without rewriting them.

A historical report is evidence of what an older build decided.
This loader returns its values untouched and names the schema.
It points out where the old fields contradict each other,
such as an episode `train_ready: true` that the gate also lists as failing,
and labels anything a current report would record but the old file lacks as unknown.
It never invents producer identity, clock provenance, defect intervals or eligibility.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import json
import os
from typing import Any

# External
from pydantic import BaseModel, Field
from upath import UPath

# Internal
from kalanos.analysis.models.compact_keys import decode_keys, is_compact
from kalanos.analysis.models.report import CURRENT_SCHEMA_VERSION, Report


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

LEGACY_SCHEMAS = ("6.3.0", "6.4.0", "6.5.0")

# Every version from this one up to `CURRENT_SCHEMA_VERSION` must validate as `Report`,
# through the converting validators in `report.py`.
# Raising it is how a later major stops reading 7.x,
# and `NEWEST_PRIOR_SCHEMA` then names the last version of the new prior major.
OLDEST_CURRENT_SCHEMA = (7, 0, 0)

# The last version an earlier major reached.
# A file of an earlier major newer than this was written by no build,
# so it is refused rather than read as current.
NEWEST_PRIOR_SCHEMA = (7, 3, 0)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Contradiction(BaseModel):
    """Two fields in one legacy report that answer the same question differently.

    Attributes
    ----------
    episode_id : str
    fields : list[str]
        The fields that disagree.
    detail : str
    """

    episode_id: str
    fields: list[str]
    detail: str


class DerivedSummary(BaseModel):
    """What a current report's counts *would* say, derived from legacy fields.

    Attributes
    ----------
    derived : bool
        Always `True`: a reader must not mistake this for a recorded decision.
    n_episodes : int
    n_gate_failing : int
        Episodes the legacy gate listed as failing.
    n_score_train_ready : int
        Episodes whose legacy score boolean said train-ready.
    legacy_readiness : float or None
        The legacy readiness number, under its own formula.
    """

    derived: bool = True
    n_episodes: int
    n_gate_failing: int
    n_score_train_ready: int
    legacy_readiness: float | None = None


class LegacyReport(BaseModel):
    """A schema-6 report, loaded losslessly.

    Attributes
    ----------
    schema_version : str
    sha256 : str
        Digest of the bytes read, so the evidence can be pinned.
    data : dict[str, Any]
        The report exactly as stored.
    contradictions : list[Contradiction]
    unknown : list[str]
        Schema-7 fields this report cannot supply.
    summary : DerivedSummary
    """

    schema_version: str
    sha256: str
    data: dict[str, Any]
    contradictions: list[Contradiction] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)
    summary: DerivedSummary


class UnsupportedSchema(ValueError):
    """The file declares a schema this loader does not read."""


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _contradictions(data: dict[str, Any]) -> list[Contradiction]:
    """Episodes the gate lists as failing whose own score says train-ready."""

    gate = data.get("gate") or {}
    failing = {item["episode_id"] for item in gate.get("failing_episodes", [])}
    found = []
    for episode in data.get("episodes", []):
        ready = (episode.get("score") or {}).get("train_ready")
        if episode.get("id") in failing and ready is True:
            found.append(
                Contradiction(
                    episode_id=episode["id"],
                    fields=["episodes[].score.train_ready", "gate.failing_episodes"],
                    detail=(
                        "score.train_ready is true while the gate lists the "
                        "episode as failing; schema 7 records one eligibility"
                    ),
                )
            )
    return found


def _unknown(data: dict[str, Any]) -> list[str]:
    """Schema-7 fields a legacy report lacks, so a reader does not assume them."""

    missing = ["producer", "run", "scope", "inventory", "eligibility_counts"]
    missing.append("episodes[].eligibility")
    missing.append("findings[].consequence")
    missing.append("findings[].support (no defect intervals recorded)")
    missing.append("streams[].clock_info (no clock provenance recorded)")
    if data.get("readiness") is None:
        missing.append("readiness")
    return missing


def _reads_as_current(version: str) -> bool:
    """Whether a file declaring `version` validates as the current `Report`.

    A file from `OLDEST_CURRENT_SCHEMA` up to the current version
    validates as the current `Report`,
    whose validators convert the older shapes.
    A newer one would validate too, dropping the fields this build does not know,
    so it is refused,
    and so is a file of an earlier major newer than `NEWEST_PRIOR_SCHEMA`.

    Parameters
    ----------
    version : str
        The file's `schema_version`.

    Returns
    -------
    bool
        True when `version` is three integers,
        not older than `OLDEST_CURRENT_SCHEMA`,
        not newer than `CURRENT_SCHEMA_VERSION`,
        and, below the current major, not newer than `NEWEST_PRIOR_SCHEMA`.
    """

    current = tuple(int(part) for part in CURRENT_SCHEMA_VERSION.split("."))
    try:
        found = tuple(int(part) for part in version.split("."))
    except ValueError:
        return False
    if len(found) != 3 or not OLDEST_CURRENT_SCHEMA <= found <= current:
        return False
    return found[0] == current[0] or found <= NEWEST_PRIOR_SCHEMA


def load_legacy(source: UPath | str | bytes | dict[str, Any]) -> LegacyReport:
    """Load a schema-6 report from a path, bytes, or an already-parsed dict.

    Parameters
    ----------
    source : UPath, str, bytes or dict
        Where the report is, or its content.

    Returns
    -------
    LegacyReport

    Raises
    ------
    UnsupportedSchema
        If the file is not one of `LEGACY_SCHEMAS`.
        A 7.x or 8.x file is not legacy: read it with `load_any`.
    """

    if isinstance(source, dict):
        raw = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        data = source
    else:
        raw = source if isinstance(source, bytes) else UPath(source).read_bytes()
        data = json.loads(raw)
    version = str(data.get("schema_version", ""))
    if _reads_as_current(version):
        raise UnsupportedSchema(
            f"schema {version} reads as a current Report, not a legacy file"
        )
    if version not in LEGACY_SCHEMAS:
        raise UnsupportedSchema(
            f"schema {version!r} is not one of the legacy schemas {LEGACY_SCHEMAS}"
        )
    episodes = data.get("episodes", [])
    gate = data.get("gate") or {}
    readiness = data.get("readiness") or {}
    return LegacyReport(
        schema_version=version,
        sha256=hashlib.sha256(raw).hexdigest(),
        data=data,
        contradictions=_contradictions(data),
        unknown=_unknown(data),
        summary=DerivedSummary(
            n_episodes=len(episodes),
            n_gate_failing=len(gate.get("failing_episodes", [])),
            n_score_train_ready=sum(
                1 for e in episodes if (e.get("score") or {}).get("train_ready") is True
            ),
            legacy_readiness=readiness.get("score"),
        ),
    )


def load_any(source: UPath | str | os.PathLike[str]) -> Report | LegacyReport:
    """Load a saved report of any supported schema.

    Parameters
    ----------
    source : UPath, str or PathLike
        Where the report JSON is, parsed as a `UPath`.

    Returns
    -------
    Report or LegacyReport
        A `Report` when the file declares a schema from 7.0.0 up to the current version,
        otherwise the file loaded as a `LegacyReport`.

    Raises
    ------
    OSError
        If `source` cannot be read.
    ValueError
        If the file is not JSON,
        a compact file's key table is malformed,
        or a file declaring a schema from 7.0.0 up to the current version
        does not validate as a `Report`.
    UnsupportedEncoding
        If a compact file declares an encoding version this kalanos can't read.
    UnsupportedSchema
        If the file declares neither a schema from 7.0.0 up to the current version,
        nor one of `LEGACY_SCHEMAS`.
    """

    raw = UPath(source).read_bytes()
    data = json.loads(raw)
    compact = is_compact(data)
    if compact:
        # A decoded report holds one string per distinct key, shared by every use.
        # Serializing it again would write each key out in full at every use,
        # so a small file repeating a long key would grow to gigabytes.
        data = decode_keys(data)
    # The file is already parsed, so it is validated as a dict, in pydantic's
    # Python mode: a strict field must accept the type `json.loads` gives it.
    if _reads_as_current(str(data.get("schema_version", ""))):
        return Report.model_validate(data)
    # A full legacy report's digest is of the bytes read.
    return load_legacy(data if compact else raw)


__all__ = [
    "LEGACY_SCHEMAS",
    "Contradiction",
    "DerivedSummary",
    "LegacyReport",
    "UnsupportedSchema",
    "load_any",
    "load_legacy",
]
