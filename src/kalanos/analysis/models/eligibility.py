"""The decision contract: one authoritative eligibility per episode.

Schema 7 separates four answers that older reports folded into one boolean:

- **Quality** — what the evaluated diagnostics measured, 0-100, descriptive.
- **Eligibility** — does this episode satisfy the declared requirements of an
  explicit evaluation scope? `pass | blocked | review | unknown`.
- **Sufficiency** — does the eligible set meet explicit dataset-level
  requirements (counts, and later windows/coverage)?
- **Readiness** — a policy-specific index over eligibility and quality that is
  `null` whenever the evidence cannot support it.

Every dataset count, CI gate, badge and compatibility boolean derives from the
episode eligibility recorded here. Nothing else may answer the same question.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum

# External
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The readiness formula this release computes. Bumped whenever the arithmetic
# or the null rules change, so a stored number stays interpretable.
READINESS_FORMULA_ID = "pass-quality-over-known-inventory-v1"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class EligibilityStatus(str, Enum):
    """The one authoritative answer to "may this episode enter the eligible pool?".

    Precedence when several reasons apply is `BLOCKED > UNKNOWN > REVIEW > PASS`;
    every reason is preserved regardless of which state wins.
    """

    # fmt: off
    BLOCKED = "blocked"  # A justified blocking rule applies
    UNKNOWN = "unknown"  # Required evidence, applicability or binding is unresolved
    REVIEW  = "review"   # Required checks ran; a contextual reason awaits review
    PASS    = "pass"     # Required checks ran and passed; no blocking/review reason
    # fmt: on


# Lower index wins.
_PRECEDENCE = [
    EligibilityStatus.BLOCKED,
    EligibilityStatus.UNKNOWN,
    EligibilityStatus.REVIEW,
    EligibilityStatus.PASS,
]


class Consequence(str, Enum):
    """What a policy says a finding does to eligibility.

    Severity is an assessment of a measurement;
    consequence is the policy's decision about it.
    The two are recorded separately so a contextual rule can be critical-severity
    yet review-only until its blocking evidence exists.
    """

    # fmt: off
    BLOCK       = "block"        # Excludes the episode from the eligible pool
    REVIEW      = "review"       # Holds the episode for a decision
    REPORT_ONLY = "report_only"  # Informational; no effect on eligibility
    # fmt: on


class BlockingRoute(str, Enum):
    """Which of the two routes justified a `BLOCK` consequence.

    A contract violation needs no calibration:
    a declared invariant was broken with direct evidence.
    A statistical rule may block only when an accepted calibration manifest
    covers its scope; otherwise it resolves to review.
    """

    # fmt: off
    CONTRACT    = "contract"     # Declared invariant violated with direct evidence
    STATISTICAL = "statistical"  # Calibrated detector, policy-authorised
    # fmt: on


class ReasonKind(str, Enum):
    """Why a reason contributes to an eligibility status."""

    # fmt: off
    FINDING     = "finding"      # A graded metric result with a consequence
    REQUIREMENT = "requirement"  # A required capability could not be evaluated
    BINDING     = "binding"      # A required channel binding is unresolved
    INVENTORY   = "inventory"    # The episode could not be loaded or graded
    # fmt: on


class EligibilityReason(BaseModel):
    """One reason behind an eligibility status, kept even when outranked.

    Attributes
    ----------
    id : str
        A stable key for the reason: a finding key such as
        `stream[/channel].metric`, a capability name, or a binding reference.
    kind : ReasonKind
        What sort of evidence this is.
    status : EligibilityStatus
        The status this reason argues for on its own.
    consequence : Consequence or None
        The policy consequence, for `FINDING` reasons.
    route : BlockingRoute or None
        Which route justified a `BLOCK`, when it did.
    detail : str
        A plain-language line a reader can act on.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    kind: ReasonKind
    status: EligibilityStatus
    consequence: Consequence | None = None
    route: BlockingRoute | None = None
    detail: str = ""


class EpisodeEligibility(BaseModel):
    """The authoritative decision for one episode under one evaluation scope.

    Attributes
    ----------
    status : EligibilityStatus
        The decision, by precedence over `reasons`.
    scope_id : str
        The evaluation requirements the decision is relative to.
    policy_id : str
        The decision policy that assigned consequences.
    reasons : list[EligibilityReason]
        Every reason, including those outranked by `status`.
    """

    status: EligibilityStatus
    scope_id: str
    policy_id: str
    reasons: list[EligibilityReason] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_reasons(self) -> "EpisodeEligibility":
        """Reject a status the recorded reasons do not justify."""

        expected = worst_status([r.status for r in self.reasons])
        if expected != self.status:
            raise ValueError(
                f"eligibility status {self.status.value!r} does not follow from "
                f"its reasons (expected {expected.value!r})"
            )
        return self

    @property
    def compatibility_train_ready(self) -> bool | None:
        """What a legacy `train_ready` boolean may say: mirrors this status only.

        `True` for pass, `False` for blocked, `None` for review and unknown —
        a boolean cannot carry an unresolved state without lying.
        """

        match self.status:
            case EligibilityStatus.PASS:
                return True
            case EligibilityStatus.BLOCKED:
                return False
            case _:
                return None


class EligibilityCounts(BaseModel):
    """How the known inventory partitions by eligibility.

    Attributes
    ----------
    total : int
        Every episode the run knew of, loaded or not.
    pass_count, blocked, review, unknown : int
        One count per status. They sum to `total`.
    inventory_complete : bool
        Whether `total` is believed to be the whole dataset. When enumeration
        was cut short, whole-dataset fractions are not published.
    confirmed_eligible_share : float or None
        `pass_count / total` when the inventory is complete and non-empty;
        a confirmed fraction, never an estimate of the unknown episodes.
    """

    total: int = Field(ge=0)
    pass_count: int = Field(ge=0)
    blocked: int = Field(ge=0)
    review: int = Field(ge=0)
    unknown: int = Field(ge=0)
    inventory_complete: bool = True
    confirmed_eligible_share: float | None = None

    @model_validator(mode="after")
    def _partition(self) -> "EligibilityCounts":
        """The four counts must partition the inventory."""

        parts = self.pass_count + self.blocked + self.review + self.unknown
        if parts != self.total:
            raise ValueError(
                f"eligibility counts {parts} do not partition total {self.total}"
            )
        return self


class Readiness(BaseModel):
    """The readiness index and the conditions under which it is defined.

    `score` is `sum(quality of pass episodes) / total known episodes`,
    on the 0-100 scale, and only when every episode is pass or blocked,
    every pass has a quality score, and the inventory is complete and non-empty.
    Otherwise it is `None` and `reasons` says why.
    It summarises eligibility and quality;
    it never establishes sufficient training data.

    Attributes
    ----------
    formula_id : str
        Which formula produced `score`.
    score : float or None
        The index, or `None` with `reasons`.
    reasons : list[str]
        Why `score` is undefined; empty when it is defined.
    passing_quality : float or None
        Mean quality of the pass episodes; `None` when none pass or none scored.
    """

    formula_id: str = READINESS_FORMULA_ID
    score: float | None
    reasons: list[str] = Field(default_factory=list)
    passing_quality: float | None = None

    @model_validator(mode="after")
    def _null_has_reasons(self) -> "Readiness":
        """An undefined index must say why; a defined one has nothing to explain."""

        if self.score is None and not self.reasons:
            raise ValueError("readiness is null without a reason")
        if self.score is not None and self.reasons:
            raise ValueError("readiness carries reasons alongside a defined score")
        return self


class SufficiencyStatus(str, Enum):
    """Whether the eligible set meets explicit dataset-level requirements."""

    # fmt: off
    SUFFICIENT   = "sufficient"    # Every explicit supported requirement passed
    INSUFFICIENT = "insufficient"  # A known required constraint failed
    UNKNOWN      = "unknown"       # No requirements, or a required result unavailable
    # fmt: on


class SufficiencyCheck(BaseModel):
    """One explicit requirement and its outcome.

    Attributes
    ----------
    requirement : str
        The requirement's key, e.g. `min_pass_episodes`.
    required : float or None
        The declared value.
    observed : float or None
        What the run measured, or `None` when unavailable.
    status : SufficiencyStatus
        This check's own verdict.
    detail : str
        Why, in one line.
    """

    requirement: str
    required: float | None = None
    observed: float | None = None
    status: SufficiencyStatus
    detail: str = ""


class Sufficiency(BaseModel):
    """The dataset-level sufficiency verdict, relative to declared requirements.

    Attributes
    ----------
    status : SufficiencyStatus
        Insufficient if any check failed; else unknown if any check was
        unavailable or none was declared; else sufficient for those checks.
    checks : list[SufficiencyCheck]
        Every declared requirement and its outcome.
    """

    status: SufficiencyStatus
    checks: list[SufficiencyCheck] = Field(default_factory=list)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def worst_status(statuses: list[EligibilityStatus]) -> EligibilityStatus:
    """The status that takes precedence among several; `PASS` for an empty list."""

    if not statuses:
        return EligibilityStatus.PASS
    return min(statuses, key=_PRECEDENCE.index)


__all__ = [
    "READINESS_FORMULA_ID",
    "BlockingRoute",
    "Consequence",
    "EligibilityCounts",
    "EligibilityReason",
    "EligibilityStatus",
    "EpisodeEligibility",
    "Readiness",
    "ReasonKind",
    "Sufficiency",
    "SufficiencyCheck",
    "SufficiencyStatus",
    "worst_status",
]
