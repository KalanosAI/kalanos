"""Coverage counts measures attempted work, independently of graded severity."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum

# External
from pydantic import BaseModel, Field, model_validator


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Availability(str, Enum):
    COMPUTED = "computed"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    ERROR = "error"
    NOT_REQUIRED = "not_required"


class CoverageRow(BaseModel):
    key: str
    unit: str
    computed: int = Field(default=0, ge=0)
    not_applicable: int = Field(default=0, ge=0)
    unavailable: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    error: int = Field(default=0, ge=0)
    not_required: int = Field(default=0, ge=0)
    eligible: int = Field(default=0, ge=0)
    reasons: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reconciles(self) -> "CoverageRow":
        if (
            self.eligible
            != self.computed + self.unavailable + self.skipped + self.error
        ):
            raise ValueError("eligible coverage must reconcile with outcome counts")
        return self


class Coverage(BaseModel):
    metrics: list[CoverageRow] = Field(default_factory=list)
    capabilities: list[CoverageRow] = Field(default_factory=list)
    dimensions: list[CoverageRow] = Field(default_factory=list)
    inventory_complete: bool = True
    unassessed_episodes: int = Field(default=0, ge=0)
    refused_sources: int = Field(default=0, ge=0)
    # Counts cover loaded subjects only; no unknown stream/frame counts invented.
    count_scope: str = "loaded_subjects"
    decoded_frames_examined: int = Field(default=0, ge=0)
    eligible_visual_frames: int | None = Field(default=None, ge=0)
    training_windows_examined: int | None = Field(default=None, ge=0)
