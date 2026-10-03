"""Shared test-only helpers: fixture paths, synthetic inputs, HTML-vs-model checks."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Container, Sequence
from html.parser import HTMLParser
from pathlib import Path

# External
import polars as pl
from upath import UPath

# Internal
from kalanos.analysis.inference.dialect import sniff_dialect
from kalanos.analysis.inference.infer import infer_schema
from kalanos.analysis.models.domain import (
    Channel,
    Clock,
    FramePayload,
    Kind,
    Stream,
    TimestampDtype,
)
from kalanos.analysis.models.metrics import ChannelContext, StreamContext
from kalanos.analysis.models.report import GradedEpisode
from kalanos.analysis.models.schema import SourceSchema
from kalanos.analysis.models.scoring import ScoreResult


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FIXTURES_DIR = UPath(__file__).parent / "fixtures"
CSV_FIXTURE = FIXTURES_DIR / "arm_multi_device.csv"
IMU_FIXTURE = FIXTURES_DIR / "imu_stream.txt"
POSE_FIXTURE = FIXTURES_DIR / "pose_log.txt"
CAPTURE_INDEX_FIXTURE = FIXTURES_DIR / "capture_index.json"
LEROBOT_FIXTURE = FIXTURES_DIR / "lerobot_v3_tiny"
LEROBOT_V2_0_FIXTURE = FIXTURES_DIR / "lerobot_v2_0_tiny"
LEROBOT_V2_1_FIXTURE = FIXTURES_DIR / "lerobot_v2_1_tiny"
HDF5_FIXTURE = FIXTURES_DIR / "hdf5_tiny.hdf5"
MCAP_FIXTURE = FIXTURES_DIR / "mcap_tiny.mcap"

# A flat metadata blob with no time index, expected to be skipped rather than analysed.
NO_TIMESERIES_FIXTURE = FIXTURES_DIR / "video_meta.json"

_CHANNEL_SOURCE_PATH = UPath("test_integrity.csv")


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


def csv_schema(path: UPath) -> SourceSchema:
    """Sniff, read, and infer a CSV file's schema.

    Parameters
    ----------
    path : UPath
        The CSV file to resolve.

    Returns
    -------
    SourceSchema
        The resolved schema, dialect included.
    """

    dialect = sniff_dialect(path)
    with path.open("rb") as handle:
        frame = pl.read_csv(
            source=handle, separator=dialect.delimiter, has_header=dialect.has_header
        )
    return infer_schema(frame).model_copy(update={"dialect": dialect})


class ScoreAttributeCollector(HTMLParser):
    """Collects every `data-level`/`data-name`/`data-score` triple, in document order.

    Attributes
    ----------
    rows : list[tuple[str, str, str]]
        Every `(level, name, score)` triple seen so far, in document order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one element's score attributes, if it carries any.

        Parameters
        ----------
        tag : str
            The element's tag name; unused, every level uses a different tag.
        attrs : list[tuple[str, str or None]]
            The element's attributes, as `HTMLParser` hands them over.
        """

        attr = dict(attrs)
        if "data-level" in attr:
            self.rows.append(
                (
                    attr["data-level"] or "",
                    attr.get("data-name") or "",
                    attr.get("data-score") or "",
                )
            )


class DisclosureStateCollector(HTMLParser):
    """Collects every tree node's `(level, name, instance, open)` state, in order.

    Attributes
    ----------
    rows : list[tuple[str, str, str, bool]]
        Every `(level, name, instance, open)` state seen so far, in document order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str, str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one tree node's disclosure state, if it carries one.

        Parameters
        ----------
        tag : str
            The element's tag name; unused, every node uses `<details>`.
        attrs : list[tuple[str, str or None]]
            The element's attributes, as `HTMLParser` hands them over.
        """

        attr = dict(attrs)
        if "data-level" in attr and "data-node" in attr:
            self.rows.append(
                (
                    attr["data-level"] or "",
                    attr.get("data-name") or "",
                    attr.get("data-instance") or "",
                    "open" in attr,
                )
            )


def score_attr(score: ScoreResult) -> str:
    """Reproduce the HTML renderer's own `data-score` formatting rule.

    Parameters
    ----------
    score : ScoreResult
        The score an element in the page carries.

    Returns
    -------
    str
        `repr(score.score)`, or an empty string when `score.score` is `None` —
        the same rule `render_html` uses, kept here independently so a test
        does not import the renderer's private formatting helper.
    """

    return "" if score.score is None else repr(score.score)


def decided(episode: GradedEpisode) -> GradedEpisode:
    """Give a hand-built GradedEpisode the eligibility its stand-in score implies.

    Test fixtures build graded episodes directly, without running the
    evaluator. Schema 7 refuses an episode without an eligibility and refuses
    a `train_ready` that contradicts it, so the stand-in is derived from the
    score's own `train_ready`: `True` passes, `False` is blocked on a synthetic
    reason, `None` is unknown for want of a graded result.

    Parameters
    ----------
    episode : GradedEpisode
        The episode, `eligibility` unset.

    Returns
    -------
    GradedEpisode
        A copy carrying a consistent eligibility.
    """

    from kalanos.analysis.models.eligibility import (
        Consequence,
        EligibilityReason,
        EligibilityStatus,
        EpisodeEligibility,
        ReasonKind,
    )

    match episode.score.train_ready:
        case True:
            reasons = []
            status = EligibilityStatus.PASS
        case False:
            reasons = [
                EligibilityReason(
                    id="stand-in.blocking",
                    kind=ReasonKind.FINDING,
                    status=EligibilityStatus.BLOCKED,
                    consequence=Consequence.BLOCK,
                    detail="test stand-in: the score's train_ready was False",
                )
            ]
            status = EligibilityStatus.BLOCKED
        case _:
            reasons = [
                EligibilityReason(
                    id="graded_result",
                    kind=ReasonKind.REQUIREMENT,
                    status=EligibilityStatus.UNKNOWN,
                    detail="test stand-in: nothing graded",
                )
            ]
            status = EligibilityStatus.UNKNOWN
    return episode.model_copy(
        update={
            "eligibility": EpisodeEligibility(
                status=status,
                scope_id="test-scope",
                policy_id="test-policy",
                reasons=reasons,
            )
        }
    )


def write_arm(
    path: Path,
    n_episodes: int,
    glitched: set[int],
    tasks=None,
    jittered: Container[int] = frozenset(),
) -> None:
    """Write an HDF5 arm recording of smooth 50 Hz joint commands.

    `glitched` episodes get large command jumps.
    `jittered` episodes get a physical clock's 50 µs timestamp jitter, so they grade.
    The rest are exactly even, so their timing is not observable.

    Parameters
    ----------
    path : Path
        Where to write the file.
    n_episodes : int
        How many episodes to write.
    glitched : set[int]
        The indices of the episodes that get command jumps.
    tasks : list[str or None], optional
        One task instruction per episode, stored when not `None`.
    jittered : Container[int]
        The indices of the episodes whose timestamps get clock jitter.
    """

    # Optional extras: importing the other helpers must not need them.
    import h5py
    import numpy as np

    rng = np.random.default_rng(0)
    t = np.arange(200) * 0.02
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        data.attrs["fps"] = 50.0
        for index in range(n_episodes):
            group = data.create_group(f"demo_{index}")
            actions = np.stack(
                [0.2 * np.sin(2 * np.pi * 0.4 * t + phase) for phase in range(6)], 1
            )
            if index in glitched:
                rows = rng.choice(np.arange(1, 199), size=20, replace=False)
                actions[rows] += rng.choice([-1, 1], (20, 6)) * rng.uniform(
                    2, 3, (20, 6)
                )
            group.create_dataset("actions", data=actions)
            stamps = t
            if index in jittered:
                stamps = t + rng.normal(0, 5e-5, t.shape)
                stamps[0] = 0.0
            group.create_dataset("timestamps", data=stamps)
            if tasks is not None and tasks[index] is not None:
                group.attrs["task"] = tasks[index]


def write_spiked_arm(path: Path, n_episodes: int, glitched: set[int]) -> None:
    """Write an HDF5 recording of one noisy sine joint, spiked in `glitched` episodes.

    Parameters
    ----------
    path : Path
        Where to write the file.
    n_episodes : int
        How many episodes to write.
    glitched : set[int]
        The indices of the episodes that get a ten-sample spike of +40.
    """

    # Optional extras: importing the other helpers must not need them.
    import h5py
    import numpy as np

    rng = np.random.default_rng(0)
    with h5py.File(str(path), "w") as store:
        for index in range(n_episodes):
            group = store.create_group(f"data/demo_{index}")
            t = np.arange(200) / 50.0
            signal = np.sin(t) + rng.normal(0, 0.01, 200)
            if index in glitched:
                signal[50:60] += 40.0
            group.create_dataset("joint_pos", data=signal)
            group.create_dataset("timestamp", data=t)


def channel_ctx(
    values: Sequence[float | None], *, rate_hz: float = 100.0
) -> ChannelContext:
    """Wrap a plain list of values in a ChannelContext, on a regular clock.

    Parameters
    ----------
    values : Sequence[float or None]
        The channel's samples.
    rate_hz : float
        The sample rate of the regular clock the samples sit on.

    Returns
    -------
    ChannelContext
        One channel named `value`, on a capture clock.
    """

    timestamps = pl.Series("time_s", [index / rate_hz for index in range(len(values))])
    payload = FramePayload(frame=pl.DataFrame({"value": values}))
    stream = Stream(
        taxonomy_type="unmapped.test",
        kind=Kind.SERIES,
        timestamps=timestamps,
        payload=payload,
        source_path=_CHANNEL_SOURCE_PATH,
        timestamp_dtype=TimestampDtype.FLOAT64,
        clock=Clock.CAPTURE,
        channels=[Channel(name="value")],
    )
    stream_ctx = StreamContext(stream=stream, is_regular=True)
    return ChannelContext(
        channel=stream.channels[0], values=payload.frame["value"], stream=stream_ctx
    )
