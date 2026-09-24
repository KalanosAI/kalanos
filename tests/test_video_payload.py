"""Verifies VideoPayload: the missing-decoder path degrades, and decoding works."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import polars as pl
import pytest

# Internal
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.video import DecoderUnavailable, VideoPayload
from kalanos.analysis.metrics import registry
from kalanos.analysis.models.domain import Clock, Kind, Stream
from kalanos.analysis.models.metrics import (
    Family,
    Level,
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.analysis.reporting.assemble import grade_stream
from kalanos.assets.policy import load_default_policy

# Local
from helpers import LEROBOT_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


LEROBOT_VIDEO = (
    LEROBOT_FIXTURE / "videos" / "observation.images.up" / "chunk-000" / "file-000.mp4"
)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_fetch_without_the_decoder_raises_naming_the_extra(monkeypatch):
    """Verify a missing decoder raises DecoderUnavailable naming the install extra."""

    monkeypatch.setattr(video, "_load_av", lambda: None)
    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=0.0, end_s=8 / 30)

    with pytest.raises(DecoderUnavailable, match=r"kalanos\[video\]"):
        payload.fetch()


def test_a_frame_metric_degrades_to_not_applicable(monkeypatch):
    """Verify a frame metric degrades rather than fails when the decoder is missing.

    The stream's own timing metrics must still run.
    """

    monkeypatch.setattr(video, "_load_av", lambda: None)
    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.STREAM, family=Family.VISION)
    def stub_frame_metric(ctx: StreamContext) -> MetricResult:
        assert ctx.payload is not None
        try:
            ctx.payload.fetch()
        except DecoderUnavailable as exc:
            return MetricResult(
                value=None,
                unit=None,
                status=MetricStatus.NOT_APPLICABLE,
                evidence={"reason": str(exc)},
            )
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    stream = Stream(
        taxonomy_type="unmapped.observation.images.up",
        kind=Kind.VIDEO,
        timestamps=pl.Series("time_s", [0.0, 1 / 30]),
        payload=VideoPayload(
            path=LEROBOT_VIDEO, frame_count=2, start_s=0.0, end_s=2 / 30
        ),
        source_path=LEROBOT_VIDEO,
        clock=Clock.UNKNOWN,
        is_regular=True,
        channels=[],
    )

    graded, _findings = grade_stream(
        stream, policy=load_default_policy(), is_regular=True, episode_id="episode_0"
    )

    assert graded.metrics["stub_frame_metric"].status == MetricStatus.NOT_APPLICABLE
    assert graded.metrics["effective_hz"].value is not None


def test_fetch_decodes_the_fixture():
    """Verify fetch() decodes the fixture's mp4, real PyAV installed."""

    payload = VideoPayload(path=LEROBOT_VIDEO, frame_count=8, start_s=0.0, end_s=8 / 30)

    frames = payload.fetch()

    assert len(frames) == payload.frame_count


def test_fetch_seeks_past_the_first_segment_for_a_later_one():
    """Verify a payload starting mid-file seeks correctly rather than reading from 0.

    The fixture's second episode is the one real case that exercises this:
    its segment starts at frame 8 of a 14-frame mp4 shared with episode 1.
    """

    payload = VideoPayload(
        path=LEROBOT_VIDEO, frame_count=6, start_s=8 / 30, end_s=14 / 30
    )

    frames = payload.fetch()

    assert len(frames) == payload.frame_count
