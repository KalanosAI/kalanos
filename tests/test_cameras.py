"""Verifies the cross-episode camera comparison flags the episode unlike its camera."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import pytest
from upath import UPath

# Internal
from kalanos.analysis.metrics.vision import _noise_score
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.domain import Episode, Stream
from kalanos.analysis.models.eligibility import Consequence
from kalanos.analysis.models.metrics import Level, MetricResult, MetricStatus
from kalanos.analysis.models.report import (
    AnalysedEpisode,
    CameraFrame,
    CameraFrames,
    CameraSummary,
    FootageStream,
    GradedEpisode,
    GradedStream,
    Report,
    round_nrmse,
)
from kalanos.analysis.models.scoring import Finding, ScoreResult, Severity
from kalanos.analysis.reporting.assemble import ReportBuilder
from kalanos.analysis.scoring.cameras import (
    _SWAP_SCORE,
    _SWAP_SEPARATION,
    _TOO_FEW_FOOTAGE,
    FootageVariation,
    _swap_scores,
    camera_samples,
    compare_cameras,
    find_camera_swaps,
    find_shared_footage,
    find_viewpoint_changes,
)
from kalanos.assets.policy import load_default_policy

# Local
from helpers import noise_frame, replaced_frames, scene_frames


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


LEFT = "observation.images.cam_high"
RIGHT = "observation.images.cam_left_wrist"


def _pair_world(
    count: int,
    swapped: set[int],
    *,
    stereo: bool = False,
    missing: frozenset[int] = frozenset(),
) -> tuple[list[GradedEpisode], FootageVariation, list[str]]:
    """Two cameras over `count` episodes, with the cameras exchanged in `swapped`.

    Each camera is a fixed 16x16 image seen through 3 frames of small noise.
    With `stereo`, the second camera is the first one shifted by a column,
    a smooth picture seen through more noise, so the two views stay alike.
    The first camera has no vector in the `missing` episodes.
    """

    rng = numpy.random.default_rng(42)
    noise = 0.02
    if stereo:
        axis = numpy.linspace(0, 1, 16)
        base_a = numpy.outer(axis, axis)
        base_b = base_a[:, [0, *range(15)]]
        noise = 0.08
    else:
        base_a, base_b = rng.random((16, 16)), rng.random((16, 16))

    def vector(base):
        frames = base.reshape(1, -1) + noise * rng.standard_normal((3, 256))
        return frames.astype(numpy.float32).reshape(-1)

    footage = FootageVariation()
    episodes = []
    for index in range(count):
        base_left, base_right = (
            (base_b, base_a) if index in swapped else (base_a, base_b)
        )
        left = None if index in missing else vector(base_left)
        footage.add(f"episode_{index}", LEFT, left)
        footage.add(f"episode_{index}", RIGHT, vector(base_right))
        episodes.append(
            _episode(
                index,
                _stream(1.0, 1.0, source_field=LEFT),
                _stream(1.0, 1.0, source_field=RIGHT),
            )
        )
    return episodes, footage, [e.id for e in episodes]


def _swaps(world):
    """What `find_camera_swaps` reports for a `_pair_world`."""

    episodes, footage, ids = world
    return find_camera_swaps(
        [sample for e in episodes for sample in camera_samples(e)], footage, ids
    )


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


ACROSS = "vision.repeated_footage_across_cameras"
HIGH = "observation.images.cam_high"
LOW = "observation.images.cam_low"


def _two_cameras(index: int, high: list[str], low: list[str], **kwargs):
    """One episode holding a `cam_high` and a `cam_low` stream with the given hashes."""

    return _episode(
        index,
        _stream(1.0, 100.0, source_field=HIGH, luma=high),
        _stream(1.0, 100.0, source_field=LOW, luma=low, **kwargs),
    )


def _shared(
    episodes: list[GradedEpisode], **options
) -> tuple[list[list[FootageStream]], list[Finding]]:
    """The footage shared across camera keys in graded episodes.

    Parameters
    ----------
    episodes : list[GradedEpisode]
        The graded episodes.
    **options
        Passed on to `find_shared_footage`.

    Returns
    -------
    tuple of (list of list of FootageStream, list of Finding)
        What `find_shared_footage` returns.
    """

    return find_shared_footage(
        [sample for e in episodes for sample in camera_samples(e)],
        [e.id for e in episodes],
        **options,
    )


def test_the_same_footage_under_two_cameras_is_flagged():
    """One recording stored under two camera keys is reported for both."""

    episodes = [_two_cameras(0, ["a", "b", "c"], ["a", "b", "c"])]

    groups, findings = _shared(episodes, grades_vision=True)

    assert groups == [
        [FootageStream(episode=0, camera=HIGH), FootageStream(episode=0, camera=LOW)]
    ]
    assert [(f.metric_id, f.episode_id, f.evidence["camera"]) for f in findings] == [
        (ACROSS, "episode_0", HIGH),
        (ACROSS, "episode_0", LOW),
    ]
    assert [f.value for f in findings] == [1.0, 1.0]
    assert all(
        f.evidence == {"camera": f.evidence["camera"], "n_frames": 3} for f in findings
    )
    assert all(f.severity.value == "critical" for f in findings)
    assert all(f.consequence is Consequence.REVIEW for f in findings)

    _groups, findings = _shared(episodes, grades_vision=False)

    assert len(findings) == 2
    assert all(f.severity.value == "warning" for f in findings)
    assert all(f.consequence is Consequence.REPORT_ONLY for f in findings)


def test_the_same_footage_across_episodes_and_cameras_is_flagged():
    """A camera's footage found under another key in another episode is reported."""

    episodes = [
        _two_cameras(0, ["a", "b", "c"], ["x", "y", "z"]),
        _two_cameras(1, ["p", "q", "r"], ["a", "b", "c"]),
    ]

    groups, findings = _shared(episodes, grades_vision=True)

    assert groups == [
        [FootageStream(episode=0, camera=HIGH), FootageStream(episode=1, camera=LOW)]
    ]
    assert [(f.episode_id, f.evidence["camera"]) for f in findings] == [
        ("episode_0", HIGH),
        ("episode_1", LOW),
    ]


def test_a_group_is_stored_once_and_counted_in_each_finding():
    """Three streams of one footage make one group,
    and each finding counts the streams under other camera keys."""

    episodes = [_two_cameras(i, ["a", "b", "c"], ["x", "y", str(i)]) for i in range(2)]
    episodes.append(_two_cameras(2, ["p", "q", "r"], ["a", "b", "c"]))

    groups, findings = _shared(episodes, grades_vision=True)

    assert groups == [
        [
            FootageStream(episode=0, camera=HIGH),
            FootageStream(episode=1, camera=HIGH),
            FootageStream(episode=2, camera=LOW),
        ]
    ]
    assert [f.value for f in findings] == [1.0, 1.0, 2.0]
    assert all("same_as" not in f.evidence for f in findings)


def test_groups_and_members_come_in_order_of_first_appearance():
    """Two interleaved groups are each ordered by their earliest stream."""

    episodes = [
        _two_cameras(0, ["a", "b", "c"], ["p", "q", "r"]),
        _two_cameras(1, ["x", "y", "z"], ["a", "b", "c"]),
        _two_cameras(2, ["p", "q", "r"], ["s", "t", "u"]),
    ]

    groups, findings = _shared(episodes, grades_vision=True)

    assert groups == [
        [FootageStream(episode=0, camera=HIGH), FootageStream(episode=1, camera=LOW)],
        [FootageStream(episode=0, camera=LOW), FootageStream(episode=2, camera=HIGH)],
    ]
    assert {f.unit for f in findings} == {"streams"}
    assert [(f.episode_id, f.evidence["camera"]) for f in findings] == [
        ("episode_0", HIGH),
        ("episode_1", LOW),
        ("episode_0", LOW),
        ("episode_2", HIGH),
    ]


def test_cameras_with_different_footage_are_not_flagged():
    """Two views hash differently, and a constant camera cannot match."""

    for episode in (
        _two_cameras(0, ["a", "b", "c"], ["a", "b", "d"]),
        _two_cameras(0, ["a", "a", "a"], ["a", "a", "a"]),
    ):
        assert _shared([episode], grades_vision=True) == ([], [])


def test_an_incomplete_read_is_not_compared_across_cameras():
    """A read that missed rows hashes only part of the stream."""

    episodes = [_two_cameras(0, ["a", "b", "c"], ["a", "b", "c"], missing_rows=[2])]

    assert _shared(episodes, grades_vision=True) == ([], [])


def test_footage_repeated_under_one_key_alone_is_not_shared():
    """Same-key repeats are `repeated_footage_vs_camera`'s, not this pass's."""

    episodes = _hashed(["a", "b", "c"], ["a", "b", "c"])

    assert _shared(episodes, grades_vision=True) == ([], [])


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


def _streak_frame(seed: int, size: int = 64) -> "numpy.ndarray":
    """A frame of horizontal bands, grey and white, with two noise rows at each edge.

    Its noise statistic reads between the metric's floor and its absolute bound,
    the way decoder streaks that look like texture do.
    """

    rng = numpy.random.default_rng(seed)
    plane = numpy.zeros((size, size), numpy.uint8)
    row = 0
    for band in range(size):
        width = (4, 5)[band % 2]
        plane[row : row + width] = 240 if band % 2 else int(rng.integers(110, 141))
        row += width
        if row >= size:
            break
    plane[:2] = rng.integers(0, 256, (2, size))
    plane[-2:] = rng.integers(0, 256, (2, size))
    return numpy.repeat(plane[:, :, None], 3, axis=2)


def _textured_frame(seed: int, size: int = 64) -> "numpy.ndarray":
    """A frame of random 4x4 blocks, which reads 2-4 on the noise statistic."""

    rng = numpy.random.default_rng(seed)
    blocks = rng.integers(60, 200, (size // 4, size // 4)).astype(numpy.uint8)
    plane = numpy.kron(blocks, numpy.ones((4, 4), numpy.uint8))
    return numpy.repeat(plane[:, :, None], 3, axis=2)


def _stream_with(stream: Stream, frames: "dict[int, numpy.ndarray]") -> Stream:
    """`stream` with the given frames replaced, naming its camera."""

    return replaced_frames(stream, frames).model_copy(
        update={"source_field": "observation.images.up"}
    )


def _corrupt_findings(streams: list[Stream]):
    """Grade one episode per stream under a full scan, and return the report."""

    builder = ReportBuilder(policy=load_default_policy(), full_frame_scan=True)
    for number, stream in enumerate(streams, start=1):
        builder.add(
            AnalysedEpisode(
                episode=Episode(id=f"episode_{number:06d}", streams=[stream]),
                adapter="test",
                adapter_confidence=1.0,
                policy=load_default_policy(),
            )
        )
    report = builder.finish(root=UPath("."))
    return report, [
        f for f in report.findings if f.metric_id == "vision.corrupt_frame_pct"
    ]


def _cameras(count: int = 6, *, frames: int = 20) -> list[Stream]:
    """`count` episodes' clean scenes of one camera."""

    return [
        _stream_with(scene_frames(frames, seed=number), {}) for number in range(count)
    ]


def test_a_report_names_the_episodes_of_its_shared_footage():
    """A graded report holds the group by episode index, and reloads it."""

    builder = ReportBuilder(policy=load_default_policy(), full_frame_scan=True)
    scenes = [(0, 100), (1, 0), (2, 102)]
    for number, (up, down) in enumerate(scenes, start=1):
        streams = [
            replaced_frames(scene_frames(20, seed=up), {}).model_copy(
                update={"source_field": "observation.images.up"}
            ),
            replaced_frames(scene_frames(20, seed=down), {}).model_copy(
                update={"source_field": "observation.images.down"}
            ),
        ]
        builder.add(
            AnalysedEpisode(
                episode=Episode(id=f"episode_{number:06d}", streams=streams),
                adapter="test",
                adapter_confidence=1.0,
                policy=load_default_policy(),
            )
        )
    report = builder.finish(root=UPath("."))
    assert isinstance(report, Report)

    assert [
        [(report.episodes[s.episode].id, s.camera) for s in group]
        for group in report.shared_footage
    ] == [
        [
            ("episode_000001", "observation.images.up"),
            ("episode_000002", "observation.images.down"),
        ]
    ]
    assert [
        (f.episode_id, f.source_field) for f in report.findings if f.metric_id == ACROSS
    ] == [
        ("episode_000001", "observation.images.up"),
        ("episode_000002", "observation.images.down"),
    ]
    payload = report.model_dump(mode="json")
    assert Report.model_validate(payload).shared_footage == report.shared_footage
    del payload["shared_footage"]
    assert Report.model_validate(payload).shared_footage == []


def test_streak_frames_read_between_the_floor_and_the_bound():
    """The test's streak frame is one only the camera-relative rule can catch."""

    ratio, spread = _noise_score(_streak_frame(0)[:, :, 0])

    assert 2 <= ratio < 5
    assert spread >= 32


def test_streak_garbage_is_caught_against_its_camera():
    """Streaks below the absolute bound block an episode whose camera is smooth."""

    streams = _cameras()
    streams[2] = _stream_with(
        streams[2], {i: _streak_frame(i) for i in range(0, 20, 2)}
    )

    report, findings = _corrupt_findings(streams)

    [finding] = findings
    assert finding.episode_id == "episode_000003"
    assert finding.consequence == Consequence.BLOCK
    assert finding.source_field == "observation.images.up"
    assert finding.evidence["relative_indices"]
    assert finding.value == 50
    assert {e.id for e in report.episodes if e.eligibility.status == "blocked"} == {
        "episode_000003"
    }


def test_a_textured_camera_is_its_own_norm():
    """A camera whose every frame scores 2-4 is not flagged against itself."""

    streams = [
        _stream_with(
            scene_frames(20, seed=n),
            {i: _textured_frame(100 * n + i) for i in range(20)},
        )
        for n in range(6)
    ]

    report, findings = _corrupt_findings(streams)

    assert findings == []
    assert all(e.eligibility.status != "blocked" for e in report.episodes)
    [stream] = report.episodes[0].streams
    assert stream.metrics["corrupt_frame_pct"].evidence["candidates"]


def test_one_stream_one_finding():
    """The camera-relative frames replace the episode's own finding, as one."""

    streams = _cameras(frames=40)
    frames = {i: noise_frame(64, i) for i in (3, 11)}
    frames.update({i: _streak_frame(i) for i in range(20, 38)})
    streams[3] = _stream_with(streams[3], frames)

    _report, findings = _corrupt_findings(streams)

    [finding] = findings
    assert finding.episode_id == "episode_000004"
    assert finding.severity.value == "critical"
    assert finding.value == 50
    assert finding.evidence["episode_share"] == 5.0
    assert len(finding.evidence["relative_indices"]) == 18


def test_too_few_episodes_skip_the_relative_rule():
    """Under five episodes of a camera, the camera has no norm to judge against."""

    streams = _cameras(4)
    streams[1] = _stream_with(
        streams[1], {i: _streak_frame(i) for i in range(0, 20, 2)}
    )

    _report, findings = _corrupt_findings(streams)

    assert findings == []


def test_footage_keeps_each_episodes_mean_frame():
    """`appearance` holds the per-column mean over an episode's frames."""

    frames = _vectors(3, 256)
    footage = FootageVariation()
    footage.add("a", CAMERA, numpy.concatenate(frames))
    footage.add("b", CAMERA, None)
    footage.add("a", CAMERA, numpy.concatenate(_vectors(3, 256, seed=7)))

    assert list(footage.appearance(CAMERA)) == ["a"]
    assert numpy.allclose(
        footage.appearance(CAMERA)["a"], numpy.mean(frames, axis=0), atol=1e-6
    )
    assert footage.appearance("unknown") == {}
    assert footage.cameras() == [CAMERA]


def test_without_numpy_no_swap_is_looked_for(monkeypatch):
    """A base install keeps no signatures, so the pass reports nothing."""

    from kalanos.analysis.scoring import cameras

    monkeypatch.setattr(cameras, "load_numpy", lambda: None)

    assert find_camera_swaps([], FootageVariation(), ["episode_0"]) == ([], [])


def test_footage_keeps_each_episodes_median_frame_and_still_share():
    """`background` holds the per-column median, `still_share` the steady columns."""

    rng = numpy.random.default_rng(3)
    base = rng.random(256, dtype=numpy.float32)
    frames = numpy.stack([base, base + 0.001, base + 0.5 * (numpy.arange(256) < 64)])
    footage = FootageVariation()
    footage.add("a", CAMERA, frames.astype(numpy.float32).reshape(-1))
    footage.add("b", CAMERA, None)

    assert list(footage.background(CAMERA)) == ["a"]
    assert numpy.allclose(
        footage.background(CAMERA)["a"], numpy.median(frames, axis=0), atol=1e-6
    )
    assert footage.still_share(CAMERA) == {"a": pytest.approx(0.75)}
    assert footage.background("unknown") == {}
    assert footage.still_share("unknown") == {}


def test_without_numpy_no_viewpoint_change_is_looked_for(monkeypatch):
    """A base install keeps no pictures, so the pass reports nothing."""

    from kalanos.analysis.scoring import cameras

    monkeypatch.setattr(cameras, "load_numpy", lambda: None)

    assert find_viewpoint_changes(FootageVariation(), ["episode_0"]) == []


def _viewpoint_world(
    count: int,
    change_at: int,
    *,
    moving: bool = False,
    missing: frozenset[int] = frozenset(),
) -> tuple[FootageVariation, list[str]]:
    """One camera over `count` episodes whose picture shifts down from `change_at`.

    The camera is a smooth 16x16 picture plus a fixed texture,
    seen through 3 frames of noise well under the still-pixel SD.
    With `moving`, every frame is a fresh random picture.
    The camera has no vector in the `missing` episodes.
    """

    rng = numpy.random.default_rng(42)
    ramp = numpy.linspace(0.2, 0.8, 16)
    picture = numpy.outer(ramp, ramp[::-1]) + 0.1 * rng.random((16, 16))
    footage = FootageVariation()
    ids = [f"episode_{n}" for n in range(count)]
    for number, episode_id in enumerate(ids):
        if number in missing:
            footage.add(episode_id, CAMERA, None)
            continue
        view = numpy.roll(picture, 4, axis=0) if number >= change_at else picture
        if moving:
            frames = rng.random((3, 256))
        else:
            frames = view.reshape(1, 256) + rng.normal(0, 0.005, (3, 256))
        footage.add(episode_id, CAMERA, frames.astype(numpy.float32).reshape(-1))
    return footage, ids


def test_a_fixed_camera_moved_partway_is_reported():
    """A camera whose picture shifts after episode 6 is named with both sides."""

    footage, ids = _viewpoint_world(14, 7)

    [change] = find_viewpoint_changes(footage, ids)

    assert change.camera == CAMERA
    assert change.before == ids[:7]
    assert change.after == ids[7:]
    assert change.score >= 1.5


def test_a_moving_camera_is_not_judged():
    """A camera whose picture changes within each episode is left out."""

    footage, ids = _viewpoint_world(14, 7, moving=True)

    assert find_viewpoint_changes(footage, ids) == []


def test_identical_renders_on_each_side_still_give_a_finite_change():
    """A simulator camera with no spread inside a side is floored, not divided by."""

    footage = FootageVariation()
    ids = [f"episode_{n}" for n in range(10)]
    for number, episode_id in enumerate(ids):
        level = 0.2 if number < 5 else 0.6
        footage.add(episode_id, CAMERA, numpy.full(3 * 256, level, dtype=numpy.float32))

    [change] = find_viewpoint_changes(footage, ids)

    assert change.before == ids[:5]
    assert math.isfinite(change.score)


def test_a_steady_camera_raises_nothing():
    """A camera that never moves has no change."""

    footage, ids = _viewpoint_world(14, 99)

    assert find_viewpoint_changes(footage, ids) == []


def test_a_camera_needs_five_episodes_each_side():
    """A change after episode 4 of 9 has too few episodes to compare."""

    footage, ids = _viewpoint_world(9, 5)

    assert find_viewpoint_changes(footage, ids) == []


def test_an_episode_without_a_picture_is_skipped():
    """The episode with no vector is on neither side."""

    footage, ids = _viewpoint_world(14, 7, missing=frozenset({3}))

    [change] = find_viewpoint_changes(footage, ids)

    assert "episode_3" not in change.before
    assert change.after == ids[7:]


def test_a_large_camera_is_split_coarse_then_fine():
    """With 450 episodes the coarse pass skips even splits; the fine pass finds 300."""

    footage, ids = _viewpoint_world(450, 300)

    [change] = find_viewpoint_changes(footage, ids)

    assert change.after[0] == "episode_300"


def test_a_swapped_run_is_reported():
    """Cameras exchanged in three consecutive episodes are named with that run."""

    swaps, findings = _swaps(_pair_world(12, {5, 6, 7}))

    [swap] = swaps
    assert swap.cameras == (LEFT, RIGHT)
    assert swap.episodes == ["episode_5", "episode_6", "episode_7"]
    assert len(findings) == 6
    for finding in findings:
        assert finding.metric_id == "vision.swapped_vs_camera"
        assert finding.consequence == Consequence.REPORT_ONLY
        assert finding.severity == Severity.WARNING
        other = {LEFT: RIGHT, RIGHT: LEFT}
        assert finding.evidence["swapped_with"] == other[finding.evidence["camera"]]
        assert finding.evidence["run"] == ["episode_5", "episode_7"]


def test_a_swap_shorter_than_three_episodes_is_not_reported():
    """Two swapped episodes are under the minimum run."""

    assert _swaps(_pair_world(12, {5, 6})) == ([], [])


def test_a_stereo_pair_is_not_reported():
    """Two views of one scene fall under the separation gate.

    The swapped episodes score under `_SWAP_SCORE` here,
    so only the gate keeps them out.
    """

    episodes, footage, ids = _pair_world(12, {5, 6, 7}, stereo=True)
    a = numpy.stack(list(footage.appearance(LEFT).values()))
    b = numpy.stack(list(footage.appearance(RIGHT).values()))
    scores = _swap_scores(a, b)

    assert all(scores[5:8] <= _SWAP_SCORE)
    assert numpy.median(scores) < _SWAP_SEPARATION
    assert _swaps((episodes, footage, ids)) == ([], [])


def test_clean_cameras_raise_nothing():
    """Cameras that never swap raise nothing."""

    assert _swaps(_pair_world(12, set())) == ([], [])


def test_too_few_shared_episodes_are_not_compared():
    """Four shared episodes are under `_MIN_EPISODES`, even all swapped."""

    assert _swaps(_pair_world(4, {0, 1, 2, 3})) == ([], [])


def test_a_missing_signature_breaks_the_run():
    """An episode without a signature ends a run, so the two halves do not join."""

    world = _pair_world(12, {4, 5, 6, 7, 8}, missing=frozenset({6}))

    assert _swaps(world) == ([], [])
