"""Compare each camera across the dataset's episodes.

Every vision metric judges one camera stream inside one episode,
so a camera exposed one way throughout is its own norm,
and an episode blurrier or darker than the same camera elsewhere,
or repeating another episode's footage, goes unnoticed.
This pass reads each camera's per-episode results after grading,
describes the camera, and flags the episodes unlike it or repeating another's footage.
Its findings request review or are reported; they never move a score.
`FootageVariation` gathers each camera's pooled frames during grading,
to say how much its footage changes between episodes.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

# Internal
from kalanos.analysis.coverage import state_of
from kalanos.analysis.metrics.vision import _SHIFT_FLOOR, is_camera_footage
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.eligibility import Consequence
from kalanos.analysis.models.metrics import Family, Level, MetricResult
from kalanos.analysis.models.provenance import content_digest
from kalanos.analysis.models.report import (
    CameraSummary,
    GradedEpisode,
    GradedStream,
    Variation,
    round_nrmse,
)
from kalanos.analysis.models.scoring import Finding, Severity
from kalanos.analysis.optional import load_numpy


if TYPE_CHECKING:
    import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The rules below are candidates, measured on lerobot/aloha_static_towel
# (docs/METRICS.md, "Across episodes"), and not calibrated.
# Below 5 episodes the quartiles miss injected blurs and dims on aloha.
_MIN_EPISODES = 5
# Tukey's fence; on its own it held a false positive in about one aloha window in eight.
_IQR_FENCE = 1.5
# A blur outlier must also sit at or below this share of the camera's median variance,
# which removed every false positive on aloha and kept every injected 3x3 blur.
_BLUR_GAP = 0.5
# A camera is dark or white when at least this share of its pixels is crushed or clipped
# in every compared episode.
_TRAIT_SHARE = 0.5
# Raw Laplacian variance of a flat frame is 0, which has no logarithm.
_LOG_FLOOR = 1e-6

# TODO: an absolute blur floor is not set;
# placing one needs data from more than one dataset.

_UNCALIBRATED = {
    "accepted": False,
    "reason": "candidate cross-episode rule; not calibrated",
}
_TOO_FEW_FOOTAGE = (
    "fewer than two episodes of this camera had every sampled frame read, "
    "with the same number of frames"
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class _Sample:
    """One camera stream in one episode, with what the comparison reads of it."""

    episode_id: str
    stream: GradedStream
    laplacian_var: float | None
    exposure_level: float | None
    dark_share: float | None
    bright_share: float | None
    luma: tuple[str, ...] | None


@dataclass
class _Running:
    """Welford's running mean and sum of squared deviations, per column.

    Attributes
    ----------
    n : int
        The vectors added.
    mean : numpy.ndarray
        Their float64 mean, per column.
    m2 : numpy.ndarray
        Their float64 sum of squared deviations from `mean`, per column.
    """

    n: int
    mean: "np.ndarray"
    m2: "np.ndarray"

    def add(self, vector: "np.ndarray") -> None:
        """Fold one vector into the running sums.

        Parameters
        ----------
        vector : numpy.ndarray
            A vector as long as `mean`.
        """

        self.n += 1
        delta = vector - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (vector - self.mean)


@dataclass
class FootageVariation:
    """How much each camera's footage changes between episodes, gathered while grading.

    It keeps no episode's frames: per camera and vector length,
    only the running mean and sum of squared deviations of each column.
    """

    _seen: set[tuple[str, str]] = field(default_factory=set, init=False)
    _groups: "dict[str, dict[int, _Running]]" = field(default_factory=dict, init=False)

    def add(self, episode_id: str, camera: str, vector: "np.ndarray | None") -> None:
        """Add one episode's pooled frames of one camera.

        Only the camera's first stream in an episode counts, as in `compare_cameras`,
        even when that stream has no vector.

        Parameters
        ----------
        episode_id : str
            The episode the stream belongs to.
        camera : str
            The camera, as `camera_key` names it.
        vector : numpy.ndarray or None
            The stream's `footage_vector`, `None` when its read missed a sampled frame.
        """

        if (camera, episode_id) in self._seen:
            return
        self._seen.add((camera, episode_id))
        if vector is None:
            return

        numpy = load_numpy()
        assert numpy is not None

        groups = self._groups.setdefault(camera, {})
        if len(vector) not in groups:
            groups[len(vector)] = _Running(
                n=0,
                mean=numpy.zeros(len(vector), dtype=numpy.float64),
                m2=numpy.zeros(len(vector), dtype=numpy.float64),
            )
        groups[len(vector)].add(vector.astype(numpy.float64))

    def result(self, camera: str) -> Variation:
        """The camera's visual variation, over its largest group of episodes.

        Episodes are grouped by vector length, that is by frames sampled;
        the group with the most episodes counts, the first seen on a tie.
        Over all pairs of a group's `n` episodes, the mean squared distance
        between two vectors is `2 / (n - 1)` times the summed per-column `m2`,
        so the value is the square root of that, divided by the vector length.

        Parameters
        ----------
        camera : str
            The camera, as `camera_key` names it.

        Returns
        -------
        Variation
            The value and the episodes it covers,
            or no value and `_TOO_FEW_FOOTAGE` below two episodes.
        """

        best: _Running | None = None
        for group in self._groups.get(camera, {}).values():
            if best is None or group.n > best.n:
                best = group
        if best is None or best.n < 2:
            return Variation(
                episodes=best.n if best is not None else 0, reason=_TOO_FEW_FOOTAGE
            )
        mean_square = 2 * float(best.m2.sum()) / ((best.n - 1) * len(best.mean))
        return Variation(value=round_nrmse(math.sqrt(mean_square)), episodes=best.n)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def camera_key(stream: GradedStream) -> str:
    """The camera a stream belongs to: its source field, else its type and instance."""

    if stream.source_field:
        return stream.source_field
    if stream.instance:
        return f"{stream.taxonomy_type}[{stream.instance}]"
    return stream.taxonomy_type


def _computed(result: MetricResult | None) -> MetricResult | None:
    """`result` if it computed a value, else `None`."""

    if result is None or state_of(result) != Availability.COMPUTED:
        return None
    return result


def _luma(stream: GradedStream) -> tuple[str, ...] | None:
    """The luminance hashes of a complete read, in order,
    or `None` when they cannot identify the footage."""

    frames = stream.frames
    if (
        frames is None
        or frames.availability != Availability.COMPUTED
        or frames.missing_rows
    ):
        return None
    hashes = tuple(f.luma_sha256 for f in frames.frames if f.luma_sha256 is not None)
    # A constant camera, black throughout for one, would pair unrelated episodes.
    if len(set(hashes)) < 2:
        return None
    return hashes


def _sample(episode_id: str, stream: GradedStream) -> _Sample:
    """Read the blur, exposure and luminance-hash values of one camera stream."""

    sharpness = _computed(stream.metrics.get("sharpness_score"))
    level = _computed(stream.metrics.get("exposure_level"))
    variance = sharpness.evidence.get("laplacian_var") if sharpness else None
    return _Sample(
        episode_id=episode_id,
        stream=stream,
        laplacian_var=float(variance) if variance is not None else None,
        exposure_level=level.value if level else None,
        dark_share=level.evidence.get("dark_share") if level else None,
        bright_share=level.evidence.get("bright_share") if level else None,
        luma=_luma(stream),
    )


def _median(values: list[float]) -> float:
    """The median of `values`."""

    numpy = load_numpy()
    assert numpy is not None

    return float(numpy.median(values))


def _fences(values: list[float]) -> tuple[float, float, float]:
    """The median and Tukey's low and high fences of `values`."""

    numpy = load_numpy()
    assert numpy is not None

    q1, median, q3 = (float(q) for q in numpy.percentile(values, [25, 50, 75]))
    spread = _IQR_FENCE * (q3 - q1)
    return median, q1 - spread, q3 + spread


def _finding(
    sample: _Sample,
    key: str,
    *,
    metric_id: str,
    value: float,
    unit: str,
    evidence: dict,
    grades_vision: bool,
    evidence_strength: Literal["deterministic", "statistical", "heuristic"] = (
        "statistical"
    ),
) -> Finding:
    """An episode-scoped finding that `sample` is unlike its camera,
    or repeating another episode's footage."""

    stream = sample.stream
    return Finding(
        id=content_digest(
            {"metric": metric_id, "episode": sample.episode_id, "camera": key}
        ),
        metric_id=metric_id,
        family=Family.VISION.value,
        # A critical finding reads FAIL on the card and sorts before graded ones,
        # which would misstate a scope that does not grade vision.
        severity=Severity.CRITICAL if grades_vision else Severity.WARNING,
        value=value,
        unit=unit,
        points=0,
        episode_id=sample.episode_id,
        stream=stream.taxonomy_type,
        instance=stream.instance,
        source_field=stream.source_field,
        source_path=stream.source_path,
        subject_level=Level.STREAM,
        evidence_strength=evidence_strength,
        evidence={"camera": key, **evidence},
        consequence=Consequence.REVIEW if grades_vision else Consequence.REPORT_ONLY,
        route=None,
        calibration=dict(_UNCALIBRATED),
    )


def compare_cameras(
    episodes: Sequence[GradedEpisode],
    *,
    grades_vision: bool,
    footage: FootageVariation | None = None,
) -> tuple[list[CameraSummary], list[Finding]]:
    """Compare each camera across the episodes it appears in.

    An episode is flagged when its camera is markedly blurrier,
    or darker or brighter, than the same camera in the other episodes.
    A camera is compared only once at least `_MIN_EPISODES` of its episodes
    have the value.
    An episode is also flagged when its camera's sampled frames hash identically
    to another episode's, for which two episodes suffice.

    Parameters
    ----------
    episodes : Sequence[GradedEpisode]
        The graded episodes, with their camera streams' vision results.
    grades_vision : bool
        Whether the scope grades the vision family:
        the findings then request review, and are only reported otherwise.
    footage : FootageVariation or None
        Each camera's pooled frames, gathered while grading `episodes`,
        for its visual variation; without it no camera has one.

    Returns
    -------
    tuple of (list of CameraSummary, list of Finding)
        One summary per camera,
        and one finding per episode unlike its camera or repeating another's footage.
    """

    # Step 1: group every camera stream by the camera it belongs to,
    # one per episode, so a camera repeated within an episode is not counted twice.
    cameras: dict[str, dict[str, _Sample]] = {}
    for episode in episodes:
        for stream in episode.streams:
            if not is_camera_footage(stream.kind or "", stream.taxonomy_type):
                continue
            cameras.setdefault(camera_key(stream), {}).setdefault(
                episode.id, _sample(episode.id, stream)
            )

    summaries: list[CameraSummary] = []
    findings: list[Finding] = []
    for key, by_episode in cameras.items():
        samples = list(by_episode.values())
        blurred = [(s, s.laplacian_var) for s in samples if s.laplacian_var is not None]
        exposed = [
            (s, s.exposure_level) for s in samples if s.exposure_level is not None
        ]
        short = [
            name
            for name, measured in (("blur", blurred), ("exposure", exposed))
            if len(measured) < _MIN_EPISODES
        ]
        blur_median = _median([v for _s, v in blurred]) if blurred else None
        level_median = _median([v for _s, v in exposed]) if exposed else None
        blur_outliers: list[str] = []
        exposure_outliers: list[str] = []
        trait: Literal["dark", "white"] | None = None

        # Step 2: blur, on the low side only.
        if blur_median is not None and len(blurred) >= _MIN_EPISODES:
            # Raw variance is skewed upwards; on aloha its fences raised 2 false
            # positives over 50 clean episodes, and the log scale none.
            _center, low, high = _fences(
                [math.log(max(v, _LOG_FLOOR)) for _s, v in blurred]
            )
            ceiling = _BLUR_GAP * blur_median
            for sample, variance in blurred:
                if math.log(max(variance, _LOG_FLOOR)) >= low or variance > ceiling:
                    continue
                blur_outliers.append(sample.episode_id)
                findings.append(
                    _finding(
                        sample,
                        key,
                        metric_id="vision.blur_vs_camera",
                        value=variance,
                        unit="variance",
                        evidence={
                            "n_episodes": len(blurred),
                            "camera_median": blur_median,
                            "fences": [math.exp(low), math.exp(high)],
                            "gap": _BLUR_GAP,
                        },
                        grades_vision=grades_vision,
                    )
                )

        # Step 3: exposure, on both sides, and the camera's trait.
        if len(exposed) >= _MIN_EPISODES:
            median, low, high = _fences([v for _s, v in exposed])
            for sample, level in exposed:
                if low <= level <= high or abs(level - median) < _SHIFT_FLOOR:
                    continue
                exposure_outliers.append(sample.episode_id)
                findings.append(
                    _finding(
                        sample,
                        key,
                        metric_id="vision.exposure_vs_camera",
                        value=level,
                        unit="gray level",
                        evidence={
                            "n_episodes": len(exposed),
                            "camera_median": median,
                            "fences": [low, high],
                            "gap": _SHIFT_FLOOR,
                        },
                        grades_vision=grades_vision,
                    )
                )
            if all((s.dark_share or 0) >= _TRAIT_SHARE for s, _v in exposed):
                trait = "dark"
            elif all((s.bright_share or 0) >= _TRAIT_SHARE for s, _v in exposed):
                trait = "white"

        # Step 4: footage repeated across episodes, with no minimum episode count.
        # A match says the footage repeats, not that the whole episode was copied:
        # a video linked to the wrong episode matches,
        # and so does a camera replaying its output.
        by_luma: dict[tuple[str, ...], list[_Sample]] = {}
        for sample in samples:
            if sample.luma is not None:
                by_luma.setdefault(sample.luma, []).append(sample)
        repeated_footage: list[list[str]] = []
        for luma, group in by_luma.items():
            if len(group) < 2:
                continue
            ids = [s.episode_id for s in group]
            repeated_footage.append(ids)
            for sample in group:
                findings.append(
                    _finding(
                        sample,
                        key,
                        metric_id="vision.repeated_footage_vs_camera",
                        value=float(len(group)),
                        unit="episodes",
                        evidence={
                            "repeated_in": [i for i in ids if i != sample.episode_id],
                            "n_frames": len(luma),
                        },
                        grades_vision=grades_vision,
                        evidence_strength="deterministic",
                    )
                )

        first = samples[0].stream
        summaries.append(
            CameraSummary(
                camera=key,
                stream=first.taxonomy_type,
                instance=first.instance,
                source_field=first.source_field,
                n_episodes=len(samples),
                compared=len(short) < 2,
                reason=(
                    f"fewer than {_MIN_EPISODES} episodes measured {' or '.join(short)}"
                    if short
                    else None
                ),
                laplacian_var=blur_median,
                exposure_level=level_median,
                trait=trait,
                blur_outliers=blur_outliers,
                exposure_outliers=exposure_outliers,
                repeated_footage=repeated_footage,
                visual_variation=footage.result(key) if footage is not None else None,
            )
        )
    return summaries, findings
