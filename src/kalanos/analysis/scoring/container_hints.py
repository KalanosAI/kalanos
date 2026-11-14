"""Compare each camera's video files across the dataset, and report container hints.

A video file whose bitrate is far from its camera's other files hints at a
re-compressed copy, and rotation metadata, an HDR transfer function or interlacing
hint at footage that loads or looks wrong.
The warnings feed no score, gate or eligibility decision.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Literal

# Internal
from kalanos.analysis.metrics.vision import is_camera_footage
from kalanos.analysis.models.report import DatasetWarning, GradedEpisode
from kalanos.analysis.models.summaries import ContainerSample
from kalanos.analysis.scoring.cameras import camera_key


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Files of one camera, size and rate, before their bitrates are compared.
_MIN_FILES = 5
# Candidate, uncalibrated.
_LOW_BITRATE_SHARE = 0.25
# Candidate, measured on one dataset (docs/METRICS.md).
_HIGH_BITRATE_RATIO = 3.0
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
_FILES_SHOWN = 10


# ░█▀█░█░█░█▀▄░█░░░▀█▀░█▀▀
# ░█▀▀░█░█░█▀▄░█░░░░█░░█░░
# ░▀░░░▀▀▀░▀▀░░▀▀▀░▀▀▀░▀▀▀


def container_samples(episode: GradedEpisode) -> list[ContainerSample]:
    """What `container_hints` reads of one episode: its camera video files.

    Parameters
    ----------
    episode : GradedEpisode
        A graded episode.

    Returns
    -------
    list[ContainerSample]
        One sample per camera stream whose container format was read,
        without repeating a camera and file within the episode.
    """

    samples: dict[tuple[str, str], ContainerSample] = {}
    for stream in episode.streams:
        if not is_camera_footage(stream.kind or "", stream.taxonomy_type):
            continue
        if stream.video_format is None or not stream.source_path:
            continue
        camera = camera_key(stream)
        key = (camera, stream.source_path)
        if key in samples:
            continue
        found = stream.video_format.found
        samples[key] = ContainerSample(
            camera=camera,
            source_path=stream.source_path,
            width=found.width,
            height=found.height,
            fps=found.fps,
            bit_rate=found.bit_rate,
            rotation_deg=found.rotation_deg,
            interlaced=found.interlaced,
            color_transfer=found.color_transfer,
        )
    return list(samples.values())


def container_hints(samples: Iterable[ContainerSample]) -> list[DatasetWarning]:
    """Report the video files whose container settings hint at a problem.

    Parameters
    ----------
    samples : Iterable[ContainerSample]
        Every episode's `container_samples`.

    Returns
    -------
    list[DatasetWarning]
        One warning per hint, in the order `low_bitrate`, `high_bitrate`, `rotation`,
        `hdr_transfer`, `interlaced`, for each hint some file raises.
    """

    # A LeRobot v3 file holds many episodes, so the same file arrives once per episode.
    files: dict[tuple[str, str], ContainerSample] = {}
    for sample in samples:
        files.setdefault((sample.camera, sample.source_path), sample)

    groups: dict[
        tuple[str, int | None, int | None, float | None], list[ContainerSample]
    ]
    groups = defaultdict(list)
    for sample in files.values():
        if sample.bit_rate:
            groups[(sample.camera, sample.width, sample.height, sample.fps)].append(
                sample
            )
    low: list[ContainerSample] = []
    high: list[ContainerSample] = []
    medians: dict[tuple[str, str], float] = {}
    for group in groups.values():
        if len(group) < _MIN_FILES:
            continue
        group.sort(key=lambda s: s.bit_rate or 0)
        rates = [s.bit_rate or 0 for s in group]
        for index, sample in enumerate(group):
            median = _median_without(rates, index)
            if rates[index] < _LOW_BITRATE_SHARE * median:
                low.append(sample)
            elif rates[index] > _HIGH_BITRATE_RATIO * median:
                high.append(sample)
            else:
                continue
            medians[(sample.camera, sample.source_path)] = median

    def bitrate(sample: ContainerSample) -> str:
        """State a flagged file's bitrate beside its camera's median."""

        median = medians[(sample.camera, sample.source_path)]
        kbps = round((sample.bit_rate or 0) / 1000)
        return f"{kbps} kbps (median {round(median / 1000)} kbps)"

    everything = list(files.values())
    warnings = [
        _warning(
            "low_bitrate",
            low,
            "has a bitrate",
            "have a bitrate",
            f"below {_LOW_BITRATE_SHARE:g}x its camera's median",
            f"below {_LOW_BITRATE_SHARE:g}x their camera's median",
            bitrate,
        ),
        _warning(
            "high_bitrate",
            high,
            "has a bitrate",
            "have a bitrate",
            f"above {_HIGH_BITRATE_RATIO:g}x its camera's median",
            f"above {_HIGH_BITRATE_RATIO:g}x their camera's median",
            bitrate,
        ),
        _warning(
            "rotation",
            [s for s in everything if s.rotation_deg],
            "carries",
            "carry",
            "rotation metadata",
            "rotation metadata",
            lambda s: f"{s.rotation_deg:g} deg",
        ),
        _warning(
            "hdr_transfer",
            [s for s in everything if s.color_transfer in _HDR_TRANSFERS],
            "uses",
            "use",
            "an HDR transfer function",
            "an HDR transfer function",
            lambda s: str(s.color_transfer),
        ),
        _warning(
            "interlaced",
            [s for s in everything if s.interlaced is True],
            "is",
            "are",
            "interlaced",
            "interlaced",
            lambda s: "",
        ),
    ]
    return [w for w in warnings if w is not None]


# ░█░█░█▀▀░█░░░█▀█░█▀▀░█▀▄░█▀▀
# ░█▀█░█▀▀░█░░░█▀▀░█▀▀░█▀▄░▀▀█
# ░▀░▀░▀▀▀░▀▀▀░▀░░░▀▀▀░▀░▀░▀▀▀


def _median_without(sorted_values: list[int], index: int) -> float:
    """The median of `sorted_values` with the entry at `index` left out.

    Parameters
    ----------
    sorted_values : list[int]
        At least three values, in ascending order.
    index : int
        The position of the value to leave out.

    Returns
    -------
    float
        The median of the remaining values, read in constant time.
    """

    count = len(sorted_values) - 1

    def other(position: int) -> int:
        """The value at `position` of the list without the entry at `index`."""

        return sorted_values[position + (position >= index)]

    middle = count // 2
    if count % 2:
        return float(other(middle))
    return (other(middle - 1) + other(middle)) / 2


def _warning(
    code: Literal[
        "low_bitrate", "high_bitrate", "rotation", "hdr_transfer", "interlaced"
    ],
    samples: list[ContainerSample],
    one_verb: str,
    many_verb: str,
    one_what: str,
    many_what: str,
    detail: Callable[[ContainerSample], str],
) -> DatasetWarning | None:
    """Build the warning for one hint, `None` when no file raises it.

    Parameters
    ----------
    code : str
        The warning's code.
    samples : list[ContainerSample]
        The files that raise the hint.
    one_verb, many_verb : str
        The message's verb after "1 video file" and after "n video files".
    one_what, many_what : str
        What the file or files have in common, ending the message's first clause.
    detail : Callable[[ContainerSample], str]
        What to state about one file after its camera and path, empty for nothing.

    Returns
    -------
    DatasetWarning or None
        The warning, listing up to `_FILES_SHOWN` files by camera then path.
    """

    if not samples:
        return None
    ordered = sorted(samples, key=lambda s: (s.camera, s.source_path))
    shown = []
    for sample in ordered[:_FILES_SHOWN]:
        text = detail(sample)
        shown.append(f"{sample.camera} {sample.source_path} {text}".rstrip())
    more = len(ordered) - _FILES_SHOWN
    tail = f"; and {more} more" if more > 0 else ""
    if len(ordered) == 1:
        head = f"1 video file {one_verb} {one_what}"
    else:
        head = f"{len(ordered)} video files {many_verb} {many_what}"
    return DatasetWarning(code=code, message=f"{head}: {'; '.join(shown)}{tail}")
