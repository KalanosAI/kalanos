"""Kalanos: data quality scoring for robotics and industrial sensor data.

The names in `__all__` are the supported library surface;
everything under the subpackages may move.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.models.adapters import AdapterTie, DatasetInfo
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.discovery import (
    SkippedSource,
    SkipReason,
    SourceInfo,
    SourceLimits,
)
from kalanos.analysis.models.errors import (
    KalanosError,
    NothingToGrade,
    SourceTooLarge,
    SourceUnavailable,
)
from kalanos.analysis.models.metrics import Level, MetricResult
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.report import (
    GradedChannel,
    GradedEpisode,
    GradedStream,
    Report,
)
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.models.scoring import Finding, Grade, ScoreResult, Severity
from kalanos.api import grade
from kalanos.assets.dictionary import load_dictionary
from kalanos.assets.policy import load_policy


__all__ = [
    "AdapterTie",
    "DatasetInfo",
    "Dictionary",
    "Finding",
    "Grade",
    "GradedChannel",
    "GradedEpisode",
    "GradedStream",
    "KalanosError",
    "Level",
    "MetricResult",
    "NothingToGrade",
    "Policy",
    "Report",
    "ScoreResult",
    "Severity",
    "SkipReason",
    "SkippedSource",
    "SourceInfo",
    "SourceLimits",
    "SourceTooLarge",
    "SourceUnavailable",
    "UnresolvedSource",
    "grade",
    "load_dictionary",
    "load_policy",
]
