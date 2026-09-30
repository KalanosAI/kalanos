"""Accepted validation evidence; no benchmark automatically creates acceptance."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Digest = str


class CalibrationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    metric_id: str = Field(pattern=r"^[^.]+\.[^.]+$")
    detector_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    thresholds_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    binding_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    taxonomy_type: str = Field(min_length=1)
    status: Literal["accepted", "draft", "revoked"] = "draft"
    accepted_by: str | None = None
    accepted_at: datetime | None = None
    valid_until: datetime | None = None
    validation_report_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_report: str = Field(min_length=1)
    operating_scope: str = Field(min_length=1)
    independent_episodes: bool = False
    real_fault_validation: bool = False
    combined_policy_validated: bool = False
    tuning_sessions: list[str] = Field(default_factory=list)
    validation_sessions: list[str] = Field(default_factory=list)
    valid_episodes: int = Field(ge=0)
    false_blocks: int = Field(ge=0)
    fault_episodes: int = Field(ge=0)
    detected_faults: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_counts(self):
        if (
            self.false_blocks > self.valid_episodes
            or self.detected_faults > self.fault_episodes
        ):
            raise ValueError("validation outcomes exceed episode counts")
        for stamp in (self.accepted_at, self.valid_until):
            if stamp is not None and stamp.utcoffset() is None:
                raise ValueError("calibration dates require a timezone")
        if (
            self.accepted_at
            and self.valid_until
            and self.valid_until <= self.accepted_at
        ):
            raise ValueError("calibration expires before acceptance")
        if len(self.tuning_sessions) != len(set(self.tuning_sessions)) or len(
            self.validation_sessions
        ) != len(set(self.validation_sessions)):
            raise ValueError("duplicate session identifiers")
        if set(self.tuning_sessions) & set(self.validation_sessions):
            raise ValueError("tuning and validation sessions overlap")
        return self
