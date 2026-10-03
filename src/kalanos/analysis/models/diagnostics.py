"""Strict configuration and evidence for optional 0.7.0 diagnostics."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from collections import Counter
from typing import Annotated, Any, Literal

# External
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.support import TemporalSupport


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class StrictModel(BaseModel):
    """Reject misspelled options and nonfinite configuration numbers."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class DiagnosticReviewPolicy(StrictModel):
    """Review thresholds belong to decision policy, separate from execution facts."""

    timing_max_unmatched_fraction: dict[str, Annotated[float, Field(ge=0, le=1)]] = (
        Field(default_factory=dict)
    )
    tracking_max_abs_error: dict[str, Annotated[float, Field(gt=0)]] = Field(
        default_factory=dict
    )
    vision_max_clipped_fraction: float | None = Field(default=None, gt=0, le=1)
    vision_min_blur_score: float | None = Field(default=None, ge=0)
    review_video_integrity: bool = False
    review_motion_limits: bool = False


class Selector(StrictModel):
    """Identify one source stream, optionally one original channel index."""

    feature: str = Field(min_length=1)
    instance: str | None = None
    index: int | None = Field(default=None, ge=0)
    source_path: str | None = None


class ClockRelation(StrictModel):
    """Attest a right-to-left clock transform for exact source scopes.

    A declaration records externally reviewed evidence;
    Kalanos does not authenticate its author
    or infer a clock relationship from equal units.
    """

    left_scope: str = Field(min_length=1)
    right_scope: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    offset_s: float = 0
    drift_ppm: float = Field(default=0, gt=-1e6)
    anchor_s: float = 0
    claim: Literal["recorded_alignment", "capture_alignment"] = "recorded_alignment"


class TimingSpec(StrictModel):
    """Compare a declared stream pair in the left stream's time domain."""

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    left: Selector
    right: Selector
    relation: ClockRelation
    tolerance_s: float = Field(ge=0)
    # Explicit corresponding events enable fitting; nearest neighbours do not.
    events: list[tuple[int, int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_events(self) -> "TimingSpec":
        """Require unique, nonnegative source-row event correspondences."""
        if any(a < 0 or b < 0 for a, b in self.events):
            raise ValueError("event rows must be nonnegative")
        if len({a for a, _ in self.events}) != len(self.events) or len(
            {b for _, b in self.events}
        ) != len(self.events):
            raise ValueError("event correspondences must be one-to-one")
        return self


class TrackingSpec(TimingSpec):
    """Compare declared command/state channels without fitting away their error."""

    response_delay_s: float = Field(default=0, ge=0)
    semantics: Literal["absolute", "delta", "rate"]


class MotionSpec(StrictModel):
    """Select a validated position channel and optional physical limits."""

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    channel: Selector
    # Wrapped angles require an explicitly declared period in native units.
    angle_period: float | None = Field(default=None, gt=0)
    max_gap_s: float = Field(gt=0)
    min_segment_samples: int = Field(default=8, ge=5)
    max_abs_velocity: float | None = Field(default=None, gt=0)
    limits_evidence: str | None = None

    @model_validator(mode="after")
    def limits_attested(self) -> "MotionSpec":
        """Require an evidence reference for a configured velocity limit."""
        if self.max_abs_velocity is not None and not self.limits_evidence:
            raise ValueError("max_abs_velocity requires limits_evidence")
        return self


class VisionSpec(StrictModel):
    """How camera footage is read.

    Sample size, full scan, decode caps, previews and exposure levels.
    """

    sample_frames: int = Field(default=10, ge=1, le=10000)
    full_frame_scan: bool = False
    max_decode_frames: int = Field(default=20000, ge=2, le=1000000)
    max_pixels: int = Field(default=2097152, ge=64)
    preview_frames: int = Field(default=8, ge=0, le=32)
    preview_size: int = Field(default=96, ge=16, le=256)
    dark_level: int = Field(default=5, ge=0, le=255)
    bright_level: int = Field(default=250, ge=0, le=255)

    @model_validator(mode="after")
    def ordered_levels(self) -> "VisionSpec":
        """Reject reversed exposure thresholds and impossible sampling budgets."""
        if self.dark_level >= self.bright_level:
            raise ValueError("dark_level must be below bright_level")
        if self.sample_frames > self.max_decode_frames:
            raise ValueError("sample_frames exceeds decoder budget")
        return self


class ModalitySpec(StrictModel):
    """A required training modality mapped into the anchor clock domain."""

    selector: Selector
    relation: ClockRelation | None = None
    max_age_s: float = Field(ge=0)
    matching: Literal["previous", "nearest"] = "previous"
    require_decoded_frames: bool = False


class WindowSpec(StrictModel):
    """Explicit training grid with no implicit interpolation or padding."""

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    anchor: Selector
    modalities: list[ModalitySpec] = Field(min_length=1)
    sample_rate_hz: float = Field(gt=0)
    history_steps: int = Field(ge=1, le=10000)
    prediction_steps: int = Field(ge=1, le=10000)
    stride: int = Field(default=1, ge=1)
    max_gap_s: float = Field(gt=0)
    max_windows: int = Field(default=10000, ge=1, le=100000)
    max_probes: int = Field(default=2000000, ge=1, le=100000000)
    padding: Literal["reject"] = "reject"
    interpolation: Literal["none"] = "none"


class CohortFeature(StrictModel):
    """Fixed native-unit range shared across an explicitly comparable cohort."""

    channel: Selector
    unit: str = Field(min_length=1)
    lower: float
    upper: float

    @model_validator(mode="after")
    def increasing_range(self) -> "CohortFeature":
        """Reject degenerate normalization ranges."""
        if self.upper <= self.lower:
            raise ValueError("cohort range must increase")
        return self


class PhaseLabel(StrictModel):
    """Externally reviewed phase intervals in a named stream's source rows."""

    episode_id: str
    phase: str = Field(min_length=1)
    start: int = Field(ge=0)
    end_exclusive: int = Field(gt=0)
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def forward(self) -> "PhaseLabel":
        """Reject empty or reversed phase intervals."""
        if self.end_exclusive <= self.start:
            raise ValueError("phase interval must be nonempty")
        return self


class CohortSpec(StrictModel):
    """An explicit comparable task/robot cohort, not automatic task discovery."""

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    episode_ids: list[str] = Field(min_length=2)
    robot_configuration: str = Field(min_length=1)
    task: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    features: list[CohortFeature] = Field(min_length=1)
    bins: int = Field(default=10, ge=2, le=100)
    trajectory_points: int = Field(default=32, ge=4, le=256)
    similarity_rmse: float = Field(ge=0)
    duration_tolerance_fraction: float = Field(default=0.1, ge=0, le=1)
    max_episodes: int = Field(default=1000, ge=2, le=10000)
    max_pair_comparisons: int = Field(default=50000, ge=1, le=1000000)
    sessions: dict[str, str] = Field(default_factory=dict)
    independent_sessions: bool = False
    bootstrap_resamples: int = Field(default=1000, ge=100, le=10000)
    bootstrap_seed: int = Field(default=0, ge=0)
    phase_labels: list[PhaseLabel] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_members(self) -> "CohortSpec":
        """Prevent duplicate members, overlapping labels and unbounded comparisons."""
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("cohort episode ids must be unique")
        if len(self.episode_ids) > self.max_episodes:
            raise ValueError("cohort exceeds max_episodes")
        if self.sessions and set(self.sessions) != set(self.episode_ids):
            raise ValueError("session mapping must cover every cohort member exactly")
        if any(not name.strip() for name in self.sessions.values()):
            raise ValueError("session identities must be nonempty")
        for episode in self.episode_ids:
            labels = sorted(
                (x for x in self.phase_labels if x.episode_id == episode),
                key=lambda x: x.start,
            )
            if any(
                a.end_exclusive > b.start
                for a, b in zip(labels, labels[1:], strict=False)
            ):
                raise ValueError("phase intervals overlap")
        if any(x.episode_id not in self.episode_ids for x in self.phase_labels):
            raise ValueError("phase label references a nonmember episode")
        return self


class DiagnosticPlan(StrictModel):
    """Optional requested diagnostics; requirements remain a separate contract.

    Attributes
    ----------
    timing : list[TimingSpec]
        The stream pairs whose timing is compared.
    tracking : list[TrackingSpec]
        The command/state channel pairs whose tracking error is measured.
    motion : list[MotionSpec]
        The position channels whose motion is checked against physical limits.
    vision : bool
        Whether to publish each camera's read as a `vision` diagnostic result.
    windows : list[WindowSpec]
        The training grids whose windows are built and checked.
    cohorts : list[CohortSpec]
        The comparable task/robot cohorts compared at dataset level.
    """

    timing: list[TimingSpec] = Field(default_factory=list)
    tracking: list[TrackingSpec] = Field(default_factory=list)
    motion: list[MotionSpec] = Field(default_factory=list)
    vision: bool = False
    windows: list[WindowSpec] = Field(default_factory=list)
    cohorts: list[CohortSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> "DiagnosticPlan":
        """Keep every configured check address stable and unambiguous."""
        ids = [
            x.id
            for xs in (
                self.timing,
                self.tracking,
                self.motion,
                self.windows,
                self.cohorts,
            )
            for x in xs
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("diagnostic ids must be unique across the plan")
        return self


class DiagnosticResult(StrictModel):
    """One measurement with availability, explicit evidence and source support."""

    id: str
    kind: Literal["timing", "tracking", "motion", "vision", "windows", "cohort"]
    episode_id: str | None = None
    subject: dict[str, Any] = Field(default_factory=dict)
    availability: Availability
    reason: str | None = None
    measurements: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    support: TemporalSupport = Field(default_factory=TemporalSupport)
    consequence: Literal["report_only", "review"] = "report_only"

    @model_validator(mode="after")
    def reconciles(self) -> "DiagnosticResult":
        """Reject nonfinite evidence and contradictory visual/window counts."""
        json.dumps(
            {"measurements": self.measurements, "evidence": self.evidence},
            allow_nan=False,
        )
        m = self.measurements
        if self.kind == "windows" and "candidate_windows" in m:
            keys = ("candidate_windows", "examined_windows", "budget_unexamined")
            if any(type(m.get(k)) is not int or m[k] < 0 for k in keys):
                raise ValueError("window counts must be nonnegative integers")
            counts = m.get("counts", {})
            if (
                not isinstance(counts, dict)
                or set(counts) != {"pass", "blocked", "review", "unknown"}
                or any(type(v) is not int or v < 0 for v in counts.values())
            ):
                raise ValueError("window decision counts must partition the candidates")
            if (
                sum(counts.values()) != m["candidate_windows"]
                or m["examined_windows"] + m["budget_unexamined"]
                != m["candidate_windows"]
                or counts["unknown"] < m["budget_unexamined"]
            ):
                raise ValueError("window counts do not reconcile")
            records = self.evidence.get("windows")
            if not isinstance(records, list) or len(records) != m["examined_windows"]:
                raise ValueError("window records do not reconcile with examined count")
            starts, recorded_counts = set(), Counter()
            for record in records:
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("status"), str)
                    or record["status"] not in counts
                ):
                    raise ValueError("window record has an invalid status")
                bounds = [
                    record.get(k)
                    for k in ("grid_start", "anchor_grid_index", "grid_end_exclusive")
                ]
                if any(type(v) is not int or v < 0 for v in bounds):
                    raise ValueError("window record has invalid grid bounds")
                assert (
                    isinstance(bounds[0], int)
                    and isinstance(bounds[1], int)
                    and isinstance(bounds[2], int)
                )
                if not (bounds[0] <= bounds[1] < bounds[2]):
                    raise ValueError("window record has invalid grid bounds")
                if bounds[0] in starts:
                    raise ValueError("duplicate window record address")
                starts.add(bounds[0])
                recorded_counts[record["status"]] += 1
            recorded_counts["unknown"] += m["budget_unexamined"]
            if any(recorded_counts[k] != v for k, v in counts.items()):
                raise ValueError("window counts do not reconcile with window records")
        if self.kind == "vision" and "examined_frames" in m:
            keys = ("examined_frames", "selected_frames", "declared_frames")
            if any(type(m.get(k)) is not int or m[k] < 0 for k in keys):
                raise ValueError("visual counts must be nonnegative integers")
            if (
                not len(self.evidence.get("frames", []))
                == m["examined_frames"]
                <= m["selected_frames"]
                <= m["declared_frames"]
            ):
                raise ValueError("visual counts do not reconcile")
        return self


class DiagnosticsReport(StrictModel):
    """Versioned, identified results; dataset work is explicit, not a score rollup."""

    schema_version: Literal[1] = 1
    plan_digest: str
    implementation_digest: str
    plan: DiagnosticPlan
    results: list[DiagnosticResult] = Field(default_factory=list)
