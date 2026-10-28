"""Verifies the cross-episode camera comparison flags the episode unlike its camera."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import pytest

# Internal
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.report import (
    CameraFrame,
    CameraFrames,
    CameraSummary,
    GradedEpisode,
    GradedStream,
    round_nrmse,
)
from kalanos.analysis.models.scoring import Finding, ScoreResult
from kalanos.analysis.scoring.cameras import (
    _TOO_FEW_FOOTAGE,
    FootageVariation,
    camera_samples,
    compare_cameras,
)


numpy = pytest.importorskip("numpy")


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
    luma: list[str] | None = None,
    missing_rows: list[int] | None = None,
) -> GradedStream:
    """A graded camera stream with the given blur and exposure results,
    and frame hashes when `luma` is given."""

    frames = None
    if luma is not None:
        rows = list(range(len(luma)))
        frames = CameraFrames(
            requested_rows=rows,
            missing_rows=missing_rows or [],
            frames=[
                CameraFrame(
                    source_row=row,
                    shape=[2, 2],
                    blur_score=1.0,
                    clipped_fraction=0.0,
                    luma_sha256=digest,
                )
                for row, digest in zip(rows, luma, strict=True)
            ],
            decoded_frames=len(luma),
            adjacent_pairs_examined=0,
            identical_adjacent_pairs=[],
            sample_plan="evenly_spaced_v1",
            parameters={},
            availability=Availability.COMPUTED,
        )
    return GradedStream(
        frames=frames,
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


def _compare(
    episodes: list[GradedEpisode], **options
) -> tuple[list[CameraSummary], list[Finding]]:
    """Compare the cameras of graded episodes, through their camera samples.

    Parameters
    ----------
    episodes : list[GradedEpisode]
        The graded episodes.
    **options
        Passed on to `compare_cameras`.

    Returns
    -------
    tuple of (list of CameraSummary, list of Finding)
        What `compare_cameras` returns.
    """

    return compare_cameras(
        [sample for e in episodes for sample in camera_samples(e)], **options
    )


def _vectors(count: int, length: int, seed: int = 42) -> list:
    """`count` seeded uniform vectors in [0, 1] of `length` float32 values."""

    rng = numpy.random.default_rng(seed)
    return list(rng.random((count, length), dtype=numpy.float32))


def _pairwise(vectors: list) -> float:
    """The RMS over every pair of `vectors` of their RMS difference, pair by pair."""

    squares = [
        float(numpy.mean((a.astype(numpy.float64) - b) ** 2))
        for i, a in enumerate(vectors)
        for b in vectors[i + 1 :]
    ]
    return round_nrmse(math.sqrt(sum(squares) / len(squares)))


def _footage(*groups: list) -> FootageVariation:
    """An accumulator holding each vector of `groups` as one episode of one camera."""

    footage = FootageVariation()
    vectors = [vector for group in groups for vector in group]
    for index, vector in enumerate(vectors):
        footage.add(f"episode_{index}", CAMERA, vector)
    return footage


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_blurred_episode_is_flagged_against_its_camera():
    """A 3x3-blurred episode's variance falls below its camera's and is flagged."""

    variances = [*CLEAN_VARIANCES]
    variances[3] = 20.3

    _summaries, findings = _compare(
        _camera(variances, CLEAN_LEVELS), grades_vision=True
    )

    assert [(f.metric_id, f.episode_id) for f in findings] == [
        ("vision.blur_vs_camera", "episode_3")
    ]


def test_a_dimmed_episode_carries_both_findings():
    """Dimming lowers both the level and the variance, so both rules flag it."""

    variances, levels = [*CLEAN_VARIANCES], [*CLEAN_LEVELS]
    variances[6], levels[6] = 26.4, 46.2

    _summaries, findings = _compare(_camera(variances, levels), grades_vision=True)

    assert sorted((f.metric_id, f.episode_id) for f in findings) == [
        ("vision.blur_vs_camera", "episode_6"),
        ("vision.exposure_vs_camera", "episode_6"),
    ]


def test_clean_episodes_raise_nothing():
    """A camera's natural spread across episodes is no outlier."""

    summaries, findings = _compare(
        _camera(CLEAN_VARIANCES, CLEAN_LEVELS), grades_vision=True
    )

    assert findings == []
    assert [s.compared for s in summaries] == [True]


def test_fewer_than_five_episodes_are_not_compared():
    """Four episodes are too few to place quartiles, so even a blur goes unflagged."""

    variances = [*CLEAN_VARIANCES[:4]]
    variances[0] = 20.3

    summaries, findings = _compare(
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

    summaries, findings = _compare(
        _camera(CLEAN_VARIANCES, [level] * 10, **shares), grades_vision=True
    )

    assert findings == []
    assert summaries[0].trait == trait


def test_a_value_measured_in_too_few_episodes_is_named():
    """Blur compares on its own; exposure, measured in four episodes, says why not."""

    episodes = _camera(CLEAN_VARIANCES[:5], CLEAN_LEVELS[:5])
    for episode in episodes[:1]:
        del episode.streams[0].metrics["exposure_level"]

    summaries, _findings = _compare(episodes, grades_vision=True)

    assert summaries[0].compared
    assert summaries[0].exposure_level is not None
    assert "exposure" in (summaries[0].reason or "")


def test_the_consequence_follows_the_scope():
    """The outlier is a critical review only where the scope grades vision."""

    variances = [*CLEAN_VARIANCES]
    variances[3] = 20.3
    episodes = _camera(variances, CLEAN_LEVELS)

    _s, graded = _compare(episodes, grades_vision=True)
    _s, reported = _compare(episodes, grades_vision=False)

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

    summaries, _findings = _compare(episodes, grades_vision=True)

    assert sorted(s.camera for s in summaries) == [
        f"{CAMERA}[left]",
        "observation.images.cam_left",
        "observation.images.cam_right",
    ]
    assert all(s.n_episodes == 10 for s in summaries)


def _hashed(*sequences: list[str]) -> list[GradedEpisode]:
    """One clean episode per hash sequence, each holding one camera stream."""

    return [
        _episode(
            index,
            _stream(CLEAN_VARIANCES[index], CLEAN_LEVELS[index], luma=luma),
        )
        for index, luma in enumerate(sequences)
    ]


def test_only_the_identical_pair_is_flagged():
    """Two episodes hashing alike repeat each other's footage;
    a third sharing a prefix does not."""

    summaries, findings = _compare(
        _hashed(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "d"]),
        grades_vision=True,
    )

    assert [(f.metric_id, f.episode_id) for f in findings] == [
        ("vision.repeated_footage_vs_camera", "episode_0"),
        ("vision.repeated_footage_vs_camera", "episode_1"),
    ]
    assert summaries[0].repeated_footage == [["episode_0", "episode_1"]]


def test_a_constant_camera_is_not_repeated_footage():
    """A camera showing one frame throughout cannot tell its episodes apart."""

    summaries, findings = _compare(
        _hashed(["a", "a", "a"], ["a", "a", "a"]), grades_vision=True
    )

    assert findings == []
    assert summaries[0].repeated_footage == []


def test_an_incomplete_read_is_not_compared():
    """A read that missed rows hashes only part of the episode."""

    episodes = _hashed(["a", "b", "c"], ["a", "b", "c"])
    episodes[1].streams[0] = _stream(
        CLEAN_VARIANCES[1], CLEAN_LEVELS[1], luma=["a", "b", "c"], missing_rows=[2]
    )

    _summaries, findings = _compare(episodes, grades_vision=True)

    assert findings == []


def test_footage_variation_matches_every_pair():
    """The running sums give the RMS over every pair of the pairs' RMS difference."""

    vectors = _vectors(30, 2560)

    variation = _footage(vectors).result(CAMERA)

    assert variation.episodes == 30
    assert variation.value == _pairwise(vectors)


def test_footage_variation_covers_the_largest_group():
    """Episodes sampling fewer frames form their own group; the larger counts."""

    full, short = _vectors(3, 2560), _vectors(2, 1280, seed=7)

    variation = _footage(short, full).result(CAMERA)

    assert variation.episodes == 3
    assert variation.value == _pairwise(full)


def test_a_footage_tie_keeps_the_earlier_group():
    """Two groups of two: the one seen first counts."""

    first, second = _vectors(2, 2560), _vectors(2, 1280, seed=7)

    variation = _footage(first, second).result(CAMERA)

    assert variation.episodes == 2
    assert variation.value == _pairwise(first)


def test_one_usable_episode_has_no_visual_variation():
    """A camera with one usable episode cannot compare its footage, and says why."""

    footage = FootageVariation()
    footage.add("episode_0", CAMERA, _vectors(1, 2560)[0])
    footage.add("episode_1", CAMERA, None)

    variation = footage.result(CAMERA)

    assert variation.value is None
    assert variation.episodes == 1
    assert variation.reason == _TOO_FEW_FOOTAGE


def test_a_camera_counts_once_per_episode():
    """A camera's second stream in an episode is ignored, as in the camera pass."""

    first, other, second = _vectors(3, 2560)
    footage = FootageVariation()
    footage.add("episode_0", CAMERA, first)
    footage.add("episode_0", CAMERA, second)
    footage.add("episode_1", CAMERA, other)

    variation = footage.result(CAMERA)

    assert variation.episodes == 2
    assert variation.value == _pairwise([first, other])


def test_without_footage_there_is_no_visual_variation():
    """The camera pass sets no visual variation when nothing gathered the footage."""

    summaries, _findings = _compare(
        _camera(CLEAN_VARIANCES, CLEAN_LEVELS), grades_vision=True
    )

    assert summaries[0].visual_variation is None
