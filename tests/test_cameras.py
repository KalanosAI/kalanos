"""Verifies the cross-episode camera comparison flags the episode unlike its camera."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import pytest

# Internal
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.report import GradedEpisode, GradedStream
from kalanos.analysis.models.scoring import ScoreResult
from kalanos.analysis.scoring.cameras import compare_cameras


pytest.importorskip("numpy")


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

CAMERA = "extero.exterior_rgb"

# Ten clean episodes spanning lerobot/aloha_static_towel cam_high's measured range
# (docs/METRICS.md, "Across episodes").
CLEAN_VARIANCES = [88.7, 92.0, 95.5, 98.0, 100.8, 101.5, 104.0, 108.3, 113.6, 119.9]
CLEAN_LEVELS = [83.0, 85.6, 88.1, 90.4, 92.4, 93.0, 95.2, 97.5, 99.8, 101.9]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _stream(
    variance: float,
    level: float,
    *,
    bright_share: float = 0.0,
    dark_share: float = 0.0,
    taxonomy_type: str = CAMERA,
    instance: str | None = None,
    source_field: str | None = "observation.images.cam_high",
) -> GradedStream:
    """A graded camera stream with the given blur and exposure results."""

    return GradedStream(
        taxonomy_type=taxonomy_type,
        instance=instance,
        kind="video",
        source_field=source_field,
        score=ScoreResult(
            level=Level.STREAM,
            score=None,
            grade=None,
            train_ready=None,
            n_contributing=0,
        ),
        metrics={
            "sharpness_score": MetricResult(
                value=0.8,
                unit="fraction",
                status=MetricStatus.REPORT_ONLY,
                evidence={"laplacian_var": variance},
            ),
            "exposure_level": MetricResult(
                value=level,
                unit="gray level",
                status=MetricStatus.REPORT_ONLY,
                evidence={"dark_share": dark_share, "bright_share": bright_share},
            ),
        },
    )


def _episode(index: int, *streams: GradedStream) -> GradedEpisode:
    """A graded episode holding `streams`."""

    return GradedEpisode(
        id=f"episode_{index}",
        adapter="test",
        adapter_confidence=1.0,
        score=ScoreResult(
            level=Level.EPISODE,
            score=None,
            grade=None,
            train_ready=None,
            n_contributing=0,
        ),
        streams=list(streams),
    )


def _camera(
    variances: list[float], levels: list[float], **kwargs
) -> list[GradedEpisode]:
    """One episode per value pair, each holding one camera stream."""

    return [
        _episode(index, _stream(variance, level, **kwargs))
        for index, (variance, level) in enumerate(zip(variances, levels, strict=True))
    ]


def test_a_blurred_episode_is_flagged_against_its_camera():
    """A 3x3-blurred episode's variance falls below its camera's and is flagged."""

    variances = [*CLEAN_VARIANCES]
    variances[3] = 20.3

    _summaries, findings = compare_cameras(
        _camera(variances, CLEAN_LEVELS), grades_vision=True
    )

    assert [(f.metric_id, f.episode_id) for f in findings] == [
        ("vision.blur_vs_camera", "episode_3")
    ]


def test_a_dimmed_episode_carries_both_findings():
    """Dimming lowers both the level and the variance, so both rules flag it."""

    variances, levels = [*CLEAN_VARIANCES], [*CLEAN_LEVELS]
    variances[6], levels[6] = 26.4, 46.2

    _summaries, findings = compare_cameras(
        _camera(variances, levels), grades_vision=True
    )

    assert sorted((f.metric_id, f.episode_id) for f in findings) == [
        ("vision.blur_vs_camera", "episode_6"),
        ("vision.exposure_vs_camera", "episode_6"),
    ]


def test_clean_episodes_raise_nothing():
    """A camera's natural spread across episodes is no outlier."""

    summaries, findings = compare_cameras(
        _camera(CLEAN_VARIANCES, CLEAN_LEVELS), grades_vision=True
    )

    assert findings == []
    assert [s.compared for s in summaries] == [True]


def test_fewer_than_five_episodes_are_not_compared():
    """Four episodes are too few to place quartiles, so even a blur goes unflagged."""

    variances = [*CLEAN_VARIANCES[:4]]
    variances[0] = 20.3

    summaries, findings = compare_cameras(
        _camera(variances, CLEAN_LEVELS[:4]), grades_vision=True
    )

    assert findings == []
    assert not summaries[0].compared
    assert summaries[0].reason


@pytest.mark.parametrize(
    ("level", "shares", "trait"),
    [(249.5, {"bright_share": 0.8}, "white"), (2.0, {"dark_share": 0.9}, "dark")],
)
def test_a_camera_exposed_one_way_in_every_episode_is_described_not_flagged(
    level, shares, trait
):
    """A simulator rendering on white, or a camera always dark, is a trait."""

    summaries, findings = compare_cameras(
        _camera(CLEAN_VARIANCES, [level] * 10, **shares), grades_vision=True
    )

    assert findings == []
    assert summaries[0].trait == trait


def test_a_value_measured_in_too_few_episodes_is_named():
    """Blur compares on its own; exposure, measured in four episodes, says why not."""

    episodes = _camera(CLEAN_VARIANCES[:5], CLEAN_LEVELS[:5])
    for episode in episodes[:1]:
        del episode.streams[0].metrics["exposure_level"]

    summaries, _findings = compare_cameras(episodes, grades_vision=True)

    assert summaries[0].compared
    assert summaries[0].exposure_level is not None
    assert "exposure" in (summaries[0].reason or "")


def test_the_consequence_follows_the_scope():
    """The outlier is a critical review only where the scope grades vision."""

    variances = [*CLEAN_VARIANCES]
    variances[3] = 20.3
    episodes = _camera(variances, CLEAN_LEVELS)

    _s, graded = compare_cameras(episodes, grades_vision=True)
    _s, reported = compare_cameras(episodes, grades_vision=False)

    assert [(f.severity.value, f.consequence.value) for f in graded] == [
        ("critical", "review")
    ]
    assert [(f.severity.value, f.consequence.value) for f in reported] == [
        ("warning", "report_only")
    ]


def test_cameras_are_keyed_by_source_field_before_taxonomy_type():
    """Two features of one type are two cameras; else type and instance key it."""

    episodes = [
        _episode(
            index,
            _stream(variance, level, source_field="observation.images.cam_left"),
            _stream(variance, level, source_field="observation.images.cam_right"),
            _stream(variance, level, source_field=None, instance="left"),
        )
        for index, (variance, level) in enumerate(
            zip(CLEAN_VARIANCES, CLEAN_LEVELS, strict=True)
        )
    ]

    summaries, _findings = compare_cameras(episodes, grades_vision=True)

    assert sorted(s.camera for s in summaries) == [
        f"{CAMERA}[left]",
        "observation.images.cam_left",
        "observation.images.cam_right",
    ]
    assert all(s.n_episodes == 10 for s in summaries)
