"""Domain models shared by every pipeline stage.

These describe the data itself: Dataset, Episode, Stream, Channel.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.models.adapters import Adapter, AdapterTie, DatasetInfo
from kalanos.analysis.models.discovery import SkippedSource, SkipReason, SourceCandidate
from kalanos.analysis.models.domain import (
    CANONICAL_TIME_COLUMN,
    CANONICAL_TIME_UNIT,
    Channel,
    Clock,
    Dataset,
    Episode,
    FramePayload,
    Kind,
    Payload,
    Stream,
)
from kalanos.analysis.models.metrics import (
    ChannelContext,
    EpisodeContext,
    Level,
    MetricContext,
    MetricResult,
    MetricStatus,
    Requires,
    StreamContext,
)
from kalanos.analysis.models.policy import Band, MetricPolicy, Policy, ScoreMode
from kalanos.analysis.models.report import (
    CURRENT_SCHEMA_VERSION,
    GradedChannel,
    GradedEpisode,
    GradedStream,
    Report,
)
from kalanos.analysis.models.schema import (
    ColumnRole,
    ColumnSpec,
    SamplingRegularity,
    SourceSchema,
    TimeSpec,
    UnresolvedSource,
)
from kalanos.analysis.models.scoring import (
    Finding,
    FindingLocation,
    Grade,
    ScoreResult,
    Severity,
)


__all__ = [
    "Adapter",
    "AdapterTie",
    "CANONICAL_TIME_COLUMN",
    "CANONICAL_TIME_UNIT",
    "CURRENT_SCHEMA_VERSION",
    "Band",
    "Channel",
    "ChannelContext",
    "Clock",
    "ColumnRole",
    "ColumnSpec",
    "Dataset",
    "DatasetInfo",
    "Episode",
    "EpisodeContext",
    "Finding",
    "FindingLocation",
    "FramePayload",
    "Grade",
    "GradedChannel",
    "GradedEpisode",
    "GradedStream",
    "Kind",
    "Level",
    "MetricContext",
    "MetricPolicy",
    "MetricResult",
    "MetricStatus",
    "Payload",
    "Policy",
    "Report",
    "Requires",
    "SamplingRegularity",
    "ScoreMode",
    "ScoreResult",
    "Severity",
    "SkipReason",
    "SkippedSource",
    "SourceCandidate",
    "SourceSchema",
    "Stream",
    "StreamContext",
    "TimeSpec",
    "UnresolvedSource",
]
