"""Flag episodes whose numeric payloads repeat another episode's.

A dataset assembled from several sources can hold the same episode twice,
and a copy whose video was re-encoded no longer hashes like its original's footage.
Its state and action data are carried over,
while a re-run of the same task never repeats them,
so this pass compares the numeric payloads of every pair of episodes after grading:

- an exact duplicate hashes identically on every payload, and requests review;
- a near duplicate lies within `_NEAR_RMSE` once each trajectory is resampled
  and normalized, and is only reported.

`Report.duplicates` stores each pair once, as episode indices,
and every near pair up to `_MAX_LISTED_PAIRS`.
A finding gives its partner count and lists no partner,
since the pairs are already stored;
a near finding also gives the limit and its nearest distance.
When most pairs are near, the report says so once, in `Report.near_duplicates`
and as one warning in `Report.dataset_warnings`,
in place of a near finding on every episode.

Camera footage takes no part in the comparison,
and `Report.cameras[].repeated_footage` already says which episodes repeat it.
It reads payloads the channel metrics already fetched, so it decodes no video.
A copy trimmed or shifted in time resamples onto different points,
so it is not found (docs/METRICS.md, "Duplicate episodes").
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import json
import math
import statistics
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

# External
import polars as pl
from pydantic import BaseModel

# Internal
from kalanos.analysis.models.eligibility import Consequence
from kalanos.analysis.models.metrics import Family, Level
from kalanos.analysis.models.provenance import content_digest
from kalanos.analysis.models.report import (
    DatasetWarning,
    DuplicateEpisodes,
    GradedEpisode,
    NearDuplicateSummary,
    NearPairs,
    round_nrmse,
)
from kalanos.analysis.models.scoring import Finding, Severity
from kalanos.analysis.optional import load_numpy


if TYPE_CHECKING:
    import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Points each channel is resampled to, over its relative row position.
FINGERPRINT_POINTS = 32
# The largest normalized RMSE at which two episodes are near duplicates,
# a candidate measured on five LeRobot datasets (docs/METRICS.md, "Duplicate episodes").
_NEAR_RMSE = 0.02
# The most a near pair's streams may each differ in row count,
# as a share of the longer one.
_DURATION_TOLERANCE = 0.1
# Near partners each episode keeps past `_MAX_LISTED_PAIRS`.
_MAX_PARTNERS = 5
# The most near pairs `Report.duplicates` lists in full.
# Past it, each episode keeps only its `_MAX_PARTNERS` nearest,
# so the list stops growing with the square of the episode count
# (docs/METRICS.md, "Duplicate episodes").
_MAX_LISTED_PAIRS = 1_500_000
# Above this share of compared pairs the near tier is saturated
# (docs/METRICS.md, "Duplicate episodes").
_SATURATED_SHARE = 0.5
# The most entries in one block of the pairwise distance matrix,
# which bounds its memory.
_CHUNK = 2_000_000
# Below this many episodes comparing every pair takes well under a second,
# and data with no main directions makes pruning slower than that.
_PRUNE_FROM = 10_000
# The main directions episodes are projected on, and how many rows estimate them.
_PRUNE_AXES = 8
_PRUNE_SAMPLE = 4_000
# Above this share of every pair left to compare, pruning is dropped.
_PRUNE_GIVE_UP = 0.25

_UNCALIBRATED = {
    "accepted": False,
    "reason": "candidate cross-episode rule; not calibrated",
}

# Where a stream sits in its episode: taxonomy type, instance, source field,
# and which of the streams sharing those three it is.
StreamKey = tuple[str, str | None, str | None, int]
# Receives a block of confirmed pairs: row indices `i < j` and their distances.
Emit = Callable[["np.ndarray", "np.ndarray", "np.ndarray"], None]


# ░█▄█░█▀█░█▀▄░█▀▀░█░░░█▀▀
# ░█░█░█░█░█░█░█▀▀░█░░░▀▀█
# ░▀░▀░▀▀▀░▀▀░░▀▀▀░▀▀▀░▀▀▀


class PayloadFingerprint(BaseModel):
    """A stream's numeric payload, reduced to what the duplicate check compares.

    It lives only for the run: grading collects it and this pass reads it,
    and no report carries it.

    Attributes
    ----------
    sha256 : str
        SHA-256 over the channel names and every value as a float64, in channel order.
    n_rows : int
        How many rows the payload holds.
    channels : list[str]
        The channels hashed and resampled, in order.
    points : bytes or None
        A little-endian float32 array of `FINGERPRINT_POINTS` rows by one column
        per channel, in C order: each channel linearly resampled over its relative
        row position, less its mean. The mean is taken off before rounding,
        since float32 keeps about 7 significant digits,
        and a small motion around a large offset would round away with it.
        Packed bytes keep a fingerprint near 4 bytes a number,
        where a list of Python floats takes about 32, and every episode's
        fingerprint is held until the pass runs.
        `None` when a value is not finite, the payload has fewer than 2 rows,
        or numpy is not installed.
    means, variances : list of float or None
        Each channel's mean and population variance over every row,
        which the pass adds back to `points`,
        and pools across episodes to scale each channel;
        `None` exactly when `points` is.
    varies : bool
        Whether any channel takes more than one value.
    """

    sha256: str
    n_rows: int
    channels: list[str]
    points: bytes | None = None
    means: list[float] | None = None
    variances: list[float] | None = None
    varies: bool


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def payload_fingerprint(
    frame: pl.DataFrame, channels: Sequence[str]
) -> PayloadFingerprint | None:
    """Reduce a stream's numeric payload to what the duplicate check compares.

    Parameters
    ----------
    frame : polars.DataFrame
        The fetched payload, one column per channel.
    channels : Sequence[str]
        The channels to fingerprint, in order.

    Returns
    -------
    PayloadFingerprint or None
        The fingerprint, or `None` when there is no channel
        or a channel does not cast to float64.
    """

    if not channels:
        return None
    try:
        cast = [frame[name].cast(pl.Float64) for name in channels]
    except (pl.exceptions.PolarsError, KeyError):
        return None
    # A nested column can cast without error and still hold no plain numbers.
    if any(series.dtype != pl.Float64 for series in cast):
        return None
    columns = [series.fill_null(math.nan).to_list() for series in cast]

    digest = hashlib.sha256(json.dumps(list(channels)).encode())
    for values in columns:
        digest.update(struct.pack(f"<{len(values)}d", *values))
    n_rows = frame.height
    varies = any(frame[name].n_unique() > 1 for name in channels)

    points = means = variances = None
    numpy = load_numpy()
    if numpy is not None and n_rows >= 2:
        data = numpy.array(columns, dtype=numpy.float64)
        if numpy.isfinite(data).all():
            rows = numpy.linspace(0.0, 1.0, n_rows)
            grid = numpy.linspace(0.0, 1.0, FINGERPRINT_POINTS)
            centre = data.mean(axis=1)
            points = (
                (
                    numpy.stack(
                        [numpy.interp(grid, rows, column) for column in data], axis=1
                    )
                    - centre
                )
                .astype("<f4")
                .tobytes()
            )
            means = centre.tolist()
            variances = data.var(axis=1).tolist()

    return PayloadFingerprint(
        sha256=digest.hexdigest(),
        n_rows=n_rows,
        channels=list(channels),
        points=points,
        means=means,
        variances=variances,
        varies=varies,
    )


def _vectors(
    members: list[dict[StreamKey, PayloadFingerprint]], keys: list[StreamKey]
) -> "np.ndarray":
    """One row per episode: its resampled trajectories,
    z-scored per channel and concatenated.

    Each channel is scaled by its standard deviation over every row of every member,
    pooled from the per-episode means and variances.
    The spread of the resampled points would do for a smooth channel,
    but a sparse one, such as a reward that is 1 on a few rows,
    is 0 at nearly every point, and dividing by that spread inflates any difference.

    Parameters
    ----------
    members : list of dict
        Each episode's fingerprints, all holding `points` on every key.
    keys : list of tuple
        The stream keys every member shares, in order.

    Returns
    -------
    numpy.ndarray
        An `(len(members), FINGERPRINT_POINTS * n_channels)` float32 array.
        After scaling, float32 rounding moves a squared distance by far less
        than the near limit, and halves the memory of float64.
    """

    numpy = load_numpy()
    assert numpy is not None

    n = len(members)
    widths = [len(members[0][key].channels) for key in keys]
    out = numpy.empty((n, FINGERPRINT_POINTS * sum(widths)), dtype=numpy.float32)
    column = 0
    for key, width in zip(keys, widths, strict=True):
        stacked = numpy.frombuffer(
            b"".join(m[key].points or b"" for m in members), dtype="<f4"
        ).reshape(n, FINGERPRINT_POINTS, width)
        weights = numpy.array([m[key].n_rows for m in members], dtype=numpy.float64)
        means = numpy.array([m[key].means for m in members], dtype=numpy.float64)
        variances = numpy.array(
            [m[key].variances for m in members], dtype=numpy.float64
        )
        total = weights.sum()
        mean = (weights[:, None] * means).sum(axis=0) / total
        pooled = (weights[:, None] * (variances + (means - mean) ** 2)).sum(
            axis=0
        ) / total
        std = numpy.sqrt(pooled)
        # A channel constant across every episode carries no distance.
        std[std == 0] = 1.0
        # Written straight into its slice of `out`, so no full-size copy is made.
        # The points are stored less each episode's own mean,
        # so each episode's offset from the pooled mean is added back.
        target = out[:, column : column + FINGERPRINT_POINTS * width].reshape(
            n, FINGERPRINT_POINTS, width
        )
        numpy.add(stacked, (means - mean).astype(numpy.float32)[:, None, :], out=target)
        target /= std.astype(numpy.float32)
        column += FINGERPRINT_POINTS * width
    return out


def _confirm(
    vectors: "np.ndarray",
    left: "np.ndarray",
    right: "np.ndarray",
    limit: float,
    emit: Emit,
) -> None:
    """Pass on the candidate pairs `(left[k], right[k])` within `limit` normalized RMSE.

    Every search path ends here, so they agree on a pair that sits on the limit.

    Parameters
    ----------
    vectors : numpy.ndarray
        One row per episode, from `_vectors`.
    left, right : numpy.ndarray
        Row indices of the candidate pairs, `left[k] < right[k]`.
    limit : float
        The largest normalized RMSE kept.
    emit : Emit
        Called once per block of candidates that holds a pair within `limit`,
        with those pairs' int64 rows and float64 distances, in candidate order.
    """

    numpy = load_numpy()
    assert numpy is not None

    dim = vectors.shape[1]
    step = max(1, _CHUNK // dim)
    for start in range(0, len(left), step):
        i, j = left[start : start + step], right[start : start + step]
        difference = vectors[i] - vectors[j]
        distance = numpy.sqrt(
            numpy.einsum("ij,ij->i", difference, difference, dtype=numpy.float64) / dim
        )
        keep = numpy.nonzero(distance <= limit)[0]
        if len(keep):
            emit(
                i[keep].astype(numpy.int64),
                j[keep].astype(numpy.int64),
                distance[keep],
            )


def _all_close_pairs(vectors: "np.ndarray", limit: float, emit: Emit) -> None:
    """`_search` by comparing every pair, a block of rows at a time.

    Parameters
    ----------
    vectors : numpy.ndarray
        One row per episode, from `_vectors`.
    limit : float
        The largest normalized RMSE kept.
    emit : Emit
        Receives the pairs within `limit`, a block at a time, from `_confirm`.
    """

    numpy = load_numpy()
    assert numpy is not None

    n, dim = vectors.shape
    squares = (vectors**2).sum(axis=1)
    # The expanded square loses a little to rounding, so it only shortlists,
    # with some slack, and `_confirm` decides.
    bound = (limit * 1.01) ** 2 * dim + 1e-3
    # Its rounding grows with the rows' squared lengths, not with their distance,
    # so a pair of copies far from every other episode needs slack in proportion:
    # summing `dim` products, the squares and the dot product each lose
    # at most `dim` units of rounding on the squared lengths.
    tolerance = 2 * (dim + 1) * float(numpy.finfo(vectors.dtype).eps)
    rows = max(1, _CHUNK // max(1, n))
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        # Only columns after the block's first row can hold a pair above the diagonal.
        block = squares[start:stop, None] + squares[None, start:]
        cut = block * tolerance + bound
        block -= 2 * vectors[start:stop] @ vectors[start:].T
        rows_i, cols = numpy.nonzero(block <= cut)
        above = start + cols > start + rows_i
        _confirm(vectors, start + rows_i[above], start + cols[above], limit, emit)


def close_pairs(vectors: "np.ndarray", limit: float) -> list[tuple[int, int, float]]:
    """Every pair of rows of `vectors` whose normalized RMSE is at most `limit`.

    Parameters
    ----------
    vectors : numpy.ndarray
        One row per episode, from `_vectors`.
    limit : float
        The largest normalized RMSE kept.

    Returns
    -------
    list of tuple of (int, int, float)
        `(i, j, distance)` with `i < j`, ordered by `i` then `j`.
    """

    found: list[tuple[int, int, float]] = []

    def collect(i: "np.ndarray", j: "np.ndarray", distance: "np.ndarray") -> None:
        """Keep a block of pairs as tuples.

        Parameters
        ----------
        i, j : numpy.ndarray
            The pairs' rows, `i < j`.
        distance : numpy.ndarray
            Their normalized RMSE.
        """

        found.extend(zip(i.tolist(), j.tolist(), distance.tolist(), strict=True))

    _search(vectors, limit, collect)
    found.sort()
    return found


def _search(vectors: "np.ndarray", limit: float, emit: Emit) -> None:
    """Pass on every pair of rows of `vectors` whose normalized RMSE is at most `limit`.

    Comparing every pair grows with the square of the episode count,
    while a near duplicate lies far closer than distinct episodes do.
    So the rows are projected on their `_PRUNE_AXES` main directions,
    sorted along the first, and only rows within `limit * sqrt(dim)` of each other
    on every projection are compared.
    A projection never lengthens a distance, so no pair within `limit` is skipped,
    and the result equals comparing every pair.
    When the rows sit too close together for the projections to separate them,
    as in a dataset of near-identical demonstrations, every pair is compared.

    Parameters
    ----------
    vectors : numpy.ndarray
        One row per episode, from `_vectors`.
    limit : float
        The largest normalized RMSE kept.
    emit : Emit
        Receives the pairs within `limit`, a block at a time, from `_confirm`,
        each pair once and in no set order.
    """

    numpy = load_numpy()
    assert numpy is not None

    n, dim = vectors.shape
    if n < _PRUNE_FROM:
        _all_close_pairs(vectors, limit, emit)
        return

    # Step 1: the main directions, from evenly spaced rows, and every row's projection.
    sample = vectors[numpy.linspace(0, n - 1, min(n, _PRUNE_SAMPLE)).astype(int)]
    centre = sample.mean(axis=0, dtype=numpy.float64)
    covariance = numpy.cov(sample - centre, rowvar=False, dtype=numpy.float64)
    _values, axes = numpy.linalg.eigh(covariance)
    axes = axes[:, ::-1][:, : min(_PRUNE_AXES, dim)]
    coords = (vectors - centre.astype(vectors.dtype)) @ axes.astype(vectors.dtype)
    coords = coords.astype(numpy.float64)

    # Step 2: along the first projection, each row's window of later rows.
    radius = limit * math.sqrt(dim) * 1.01 + 1e-6
    order = numpy.argsort(coords[:, 0], kind="stable")
    first = coords[order, 0]
    ends = numpy.searchsorted(first, first + radius, side="right")
    counts = ends - numpy.arange(n) - 1
    if counts.sum() > _PRUNE_GIVE_UP * n * (n - 1) / 2:
        _all_close_pairs(vectors, limit, emit)
        return

    # Step 3: the window's pairs, kept where every other projection is close too,
    # a slice of rows at a time so the candidate arrays stay bounded.
    position = 0
    while position < n:
        stop = position + 1
        budget = counts[position]
        while stop < n and budget + counts[stop] <= _CHUNK:
            budget += counts[stop]
            stop += 1
        rows = numpy.repeat(numpy.arange(position, stop), counts[position:stop])
        offsets = numpy.arange(len(rows)) - numpy.repeat(
            numpy.cumsum(counts[position:stop]) - counts[position:stop],
            counts[position:stop],
        )
        partners = rows + 1 + offsets
        a, b = order[rows], order[partners]
        near = (numpy.abs(coords[a, 1:] - coords[b, 1:]) <= radius).all(axis=1)
        a, b = a[near], b[near]
        swap = a > b
        a[swap], b[swap] = b[swap], a[swap]
        _confirm(vectors, a, b, limit, emit)
        position = stop


class _Nearest:
    """Each row's near partner count and nearest partners, kept as pairs stream in,
    so the full list of near pairs is never held.

    Attributes
    ----------
    counts : numpy.ndarray
        How many near partners each row has, as int64.
    weighted : numpy.ndarray
        The same, each partner counting the exact copies it stands for, as float64.
    partner : numpy.ndarray
        An `(n, _MAX_PARTNERS)` int64 array of each row's nearest partners,
        nearest first, ties broken on the lower row; -1 past the last one.
    distance : numpy.ndarray
        An `(n, _MAX_PARTNERS)` float64 array of their distances,
        `inf` past the last one.
    total : int
        How many pairs were added.
    """

    def __init__(self, n: int, weights: "Sequence[int] | np.ndarray") -> None:
        """Start with no pair.

        Parameters
        ----------
        n : int
            How many rows the pairs index.
        weights : sequence of int or numpy.ndarray
            How many exact copies each row stands for, itself included.
        """

        numpy = load_numpy()
        assert numpy is not None

        self._weights = numpy.asarray(weights, dtype=numpy.float64)
        self.counts = numpy.zeros(n, dtype=numpy.int64)
        self.weighted = numpy.zeros(n, dtype=numpy.float64)
        self.partner = numpy.full((n, _MAX_PARTNERS), -1, dtype=numpy.int64)
        self.distance = numpy.full((n, _MAX_PARTNERS), numpy.inf)
        self.total = 0

    def add(self, i: "np.ndarray", j: "np.ndarray", distance: "np.ndarray") -> None:
        """Count a block of pairs on both their rows, and keep the nearest.

        Parameters
        ----------
        i, j : numpy.ndarray
            The pairs' rows, `i < j`, each pair added once.
        distance : numpy.ndarray
            Their normalized RMSE.
        """

        numpy = load_numpy()
        assert numpy is not None

        if not len(i):
            return
        n, k = self.partner.shape
        self.total += len(i)
        rows = numpy.concatenate([i, j])
        others = numpy.concatenate([j, i])
        distances = numpy.concatenate([distance, distance])
        self.counts += numpy.bincount(rows, minlength=n)
        self.weighted += numpy.bincount(
            rows, weights=self._weights[others], minlength=n
        )

        # Step 1: each row's k nearest in this block.
        order = numpy.lexsort((others, distances, rows))
        rows, others, distances = rows[order], others[order], distances[order]
        starts = numpy.flatnonzero(numpy.r_[True, rows[1:] != rows[:-1]])
        rank = numpy.arange(len(rows)) - numpy.repeat(
            starts, numpy.diff(numpy.r_[starts, len(rows)])
        )
        keep = rank < k
        touched = rows[starts]
        slot = numpy.searchsorted(touched, rows[keep])

        # Step 2: merged with the k each touched row already holds.
        partner = numpy.full((len(touched), 2 * k), -1, dtype=numpy.int64)
        merged = numpy.full((len(touched), 2 * k), numpy.inf)
        partner[:, :k] = self.partner[touched]
        merged[:, :k] = self.distance[touched]
        partner[slot, k + rank[keep]] = others[keep]
        merged[slot, k + rank[keep]] = distances[keep]
        best = numpy.lexsort((partner, merged))[:, :k]
        self.partner[touched] = numpy.take_along_axis(partner, best, axis=1)
        self.distance[touched] = numpy.take_along_axis(merged, best, axis=1)

    def pairs(self) -> list[tuple[int, int, float]]:
        """The pairs among the nearest of either of their rows.

        Returns
        -------
        list of tuple of (int, int, float)
            `(i, j, distance)` with `i < j`, each pair once, ordered by `i` then `j`.
        """

        numpy = load_numpy()
        assert numpy is not None

        n, k = self.partner.shape
        rows = numpy.repeat(numpy.arange(n), k)
        others = self.partner.ravel()
        valid = others >= 0
        a = numpy.minimum(rows, others)[valid]
        b = numpy.maximum(rows, others)[valid]
        distance = self.distance.ravel()[valid]
        _keys, first = numpy.unique(a * n + b, return_index=True)
        return list(
            zip(
                a[first].tolist(),
                b[first].tolist(),
                distance[first].tolist(),
                strict=True,
            )
        )


class _Listed:
    """Every near pair streamed in, until there are more than `_MAX_LISTED_PAIRS`.

    Attributes
    ----------
    blocks : list of tuple or None
        Each block's group index, rows `i < j` and distances, in the order added.
        `None` once the pairs added outnumber `_MAX_LISTED_PAIRS`,
        and the blocks are dropped.
    group : int
        The group index the next blocks are recorded under.
    total : int
        How many pairs were added.
    """

    def __init__(self) -> None:
        """Start with no pair."""

        self.blocks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] | None = []
        self.group = 0
        self.total = 0

    def add(self, i: "np.ndarray", j: "np.ndarray", distance: "np.ndarray") -> None:
        """Keep a block of pairs, or drop every block once there are too many.

        Parameters
        ----------
        i, j : numpy.ndarray
            The pairs' rows, `i < j`.
        distance : numpy.ndarray
            Their normalized RMSE.
        """

        self.total += len(i)
        if self.blocks is None:
            return
        if self.total > _MAX_LISTED_PAIRS:
            self.blocks = None
            return
        if len(i):
            self.blocks.append((self.group, i, j, distance))


def _fan_out(*emits: Emit) -> Emit:
    """Pass each block of pairs to every one of `emits`.

    Parameters
    ----------
    *emits : Emit
        The receivers, called in order.

    Returns
    -------
    Emit
        The combined emit.
    """

    def emit(i: "np.ndarray", j: "np.ndarray", distance: "np.ndarray") -> None:
        """Pass the block on.

        Parameters
        ----------
        i, j : numpy.ndarray
            The pairs' rows, `i < j`.
        distance : numpy.ndarray
            Their normalized RMSE.
        """

        for receiver in emits:
            receiver(i, j, distance)

    return emit


def _within_duration(n_rows: "np.ndarray", emit: Emit) -> Emit:
    """Wrap `emit` so it receives only the pairs whose row counts agree.

    Parameters
    ----------
    n_rows : numpy.ndarray
        An `(n, n_keys)` array of each row's row count on every stream key.
    emit : Emit
        Receives the pairs whose streams each differ in row count
        by at most `_DURATION_TOLERANCE` of the longer one.

    Returns
    -------
    Emit
        The filtering emit.
    """

    numpy = load_numpy()
    assert numpy is not None

    def filtered(i: "np.ndarray", j: "np.ndarray", distance: "np.ndarray") -> None:
        """Pass on the block's pairs whose row counts agree.

        Parameters
        ----------
        i, j : numpy.ndarray
            The pairs' rows, `i < j`.
        distance : numpy.ndarray
            Their normalized RMSE.
        """

        left, right = n_rows[i], n_rows[j]
        mismatch = (numpy.abs(left - right) / numpy.maximum(left, right)).max(axis=1)
        keep = mismatch <= _DURATION_TOLERANCE
        emit(i[keep], j[keep], distance[keep])

    return filtered


def _finding(
    episode_id: str,
    tier: Literal["exact", "near"],
    count: int,
    evidence: dict,
) -> Finding:
    """An episode-level finding that `episode_id` duplicates `count` episodes.

    Parameters
    ----------
    episode_id : str
        The episode the finding is about.
    tier : {"exact", "near"}
        `exact` requests review, `near` is only reported.
    count : int
        How many episodes `episode_id` duplicates,
        counting the exact copies a partner stands for.
    evidence : dict
        The finding's evidence.

    Returns
    -------
    Finding
        The finding.
    """

    exact = tier == "exact"
    metric_id = (
        "integrity.duplicate_episode" if exact else "integrity.near_duplicate_episode"
    )
    return Finding(
        id=content_digest({"metric": metric_id, "episode": episode_id}),
        metric_id=metric_id,
        family=Family.INTEGRITY.value,
        severity=Severity.CRITICAL if exact else Severity.WARNING,
        value=float(count),
        unit="episodes",
        points=0,
        episode_id=episode_id,
        subject_level=Level.EPISODE,
        evidence_strength="deterministic" if exact else "statistical",
        evidence=evidence,
        consequence=Consequence.REVIEW if exact else Consequence.REPORT_ONLY,
        route=None,
        calibration=dict(_UNCALIBRATED),
    )


def find_duplicate_episodes(
    episodes: Sequence[GradedEpisode],
    fingerprints: Mapping[str, Mapping[StreamKey, PayloadFingerprint]],
) -> tuple[DuplicateEpisodes, list[Finding], NearDuplicateSummary | None]:
    """Pair the episodes whose numeric payloads are identical or nearly so.

    Only episodes whose fingerprinted streams sit at the same keys
    with the same channels are compared,
    and an episode whose every channel is constant takes no part,
    since two such episodes would pair whatever they recorded.

    Episodes that hash identically form a group, which the earliest of them stands for,
    so that k copies give k - 1 pairs rather than one for every two of them:

    - each later copy pairs with the earliest, exactly;
    - a near pair joins the earliest episodes of two groups,
      and every copy in either group is near the other group's earliest episode.

    Every copy still carries each finding the pair implies.
    A finding lists no partner, only how many there are,
    and a near finding also the distance to the nearest.
    Every near pair is kept while there are at most `_MAX_LISTED_PAIRS`;
    past that, a near pair is kept only when it is among the `_MAX_PARTNERS` nearest
    of either episode.
    When the near tier is saturated, no near finding is made.

    Parameters
    ----------
    episodes : Sequence[GradedEpisode]
        Every graded episode, which sets the order of pairs and findings,
        and whose positions the pairs' indices are.
        `Report.episodes` must keep this order.
    fingerprints : Mapping
        Each episode's fingerprints by episode id, as grading collected them.

    Returns
    -------
    tuple of (DuplicateEpisodes, list of Finding, NearDuplicateSummary or None)
        Every exact group and kept near pair, one finding per episode and tier
        with at least one partner, and the near search's counts over the dataset,
        `None` when no group reached the near search.
    """

    # Step 1: each taking part episode's fingerprints, grouped by what they hold.
    order = {episode.id: index for index, episode in enumerate(episodes)}
    groups: dict[tuple, list[tuple[str, dict[StreamKey, PayloadFingerprint]]]] = {}
    for episode in episodes:
        prints = dict(fingerprints.get(episode.id, {}))
        if not any(p.varies for p in prints.values()):
            continue
        signature = tuple(sorted((key, tuple(p.channels)) for key, p in prints.items()))
        groups.setdefault(signature, []).append((episode.id, prints))

    # Each group's episodes by its earliest one, which comes first.
    copies: dict[str, list[str]] = {}
    # Each representative's distance to its nearest partner,
    # and how many episodes it is near, copies included.
    near: dict[str, tuple[float, int]] = {}
    # Each searched group's representatives and their nearest partners.
    searched: list[tuple[list[str], _Nearest]] = []
    listed = _Listed()
    episodes_compared = pairs_compared = pairs_near = 0
    nearest_distances: list[float] = []
    for signature, members in groups.items():
        if len(members) < 2:
            continue
        keys = [key for key, _channels in signature]

        # Step 2: exact copies hash identically, row count included, on every key.
        by_hash: dict[tuple, list[str]] = {}
        for episode_id, prints in members:
            identity = tuple((prints[k].sha256, prints[k].n_rows) for k in keys)
            by_hash.setdefault(identity, []).append(episode_id)
        for ids in by_hash.values():
            copies[ids[0]] = ids

        # Step 3: near pairs, among the earliest episodes of the groups
        # resampled on every key. Copies resample to the same points,
        # so the other copies would only repeat their earliest one's pairs.
        numpy = load_numpy()
        if numpy is None:
            continue
        resampled = [
            (episode_id, prints)
            for episode_id, prints in members
            if all(prints[k].points is not None for k in keys)
        ]
        if len(resampled) < 2:
            continue
        # Scaled over every copy, as each copy's rows weigh in the pooled spread.
        vectors = _vectors([p for _id, p in resampled], keys)
        firsts = [
            index
            for index, (episode_id, _p) in enumerate(resampled)
            if episode_id in copies
        ]
        if len(firsts) < 2:
            continue
        if len(firsts) < len(resampled):
            vectors = vectors[firsts]
        ids = [resampled[index][0] for index in firsts]
        n_rows = numpy.array(
            [[resampled[index][1][k].n_rows for k in keys] for index in firsts],
            dtype=numpy.float64,
        )
        nearest = _Nearest(len(ids), [len(copies[episode_id]) for episode_id in ids])
        listed.group = len(searched)
        _search(
            vectors,
            _NEAR_RMSE,
            _within_duration(n_rows, _fan_out(nearest.add, listed.add)),
        )

        searched.append((ids, nearest))
        episodes_compared += len(ids)
        pairs_compared += len(ids) * (len(ids) - 1) // 2
        pairs_near += nearest.total
        for row in numpy.flatnonzero(nearest.counts).tolist():
            nearest_distance = float(nearest.distance[row, 0])
            near[ids[row]] = (nearest_distance, round(float(nearest.weighted[row])))
            nearest_distances.append(nearest_distance)

    # Step 4: every near pair while they fit, else each episode's nearest.
    if listed.blocks is not None:
        found = [
            (searched[group][0], i, j, d)
            for group, rows_i, rows_j, distances in listed.blocks
            for i, j, d in zip(
                rows_i.tolist(), rows_j.tolist(), distances.tolist(), strict=True
            )
        ]
    else:
        found = [
            (ids, i, j, d) for ids, nearest in searched for i, j, d in nearest.pairs()
        ]
    later: dict[int, list[tuple[int, float]]] = {}
    for ids, i, j, distance in found:
        a, b = order[ids[i]], order[ids[j]]
        lo, hi = min(a, b), max(a, b)
        later.setdefault(lo, []).append((hi, round_nrmse(distance)))
    rows = [(lo, sorted(later[lo])) for lo in sorted(later)]
    groups_exact = sorted(
        (
            sorted(order[episode_id] for episode_id in ids)
            for ids in copies.values()
            if len(ids) > 1
        ),
        key=lambda group: group[0],
    )
    pairs = DuplicateEpisodes(
        exact=groups_exact,
        near=NearPairs(
            episode=[lo for lo, _kept in rows],
            partners=[[hi for hi, _d in kept] for _lo, kept in rows],
            nrmse=[[d for _hi, d in kept] for _lo, kept in rows],
        ),
    )

    summary = None
    if searched:
        summary = NearDuplicateSummary(
            limit=_NEAR_RMSE,
            max_partners=_MAX_PARTNERS,
            episodes_compared=episodes_compared,
            pairs_compared=pairs_compared,
            pairs_near=pairs_near,
            pairs_listed=len(found),
            episodes_with_partner=sum(len(copies[rep]) for rep in near),
            median_nearest_nrmse=(
                statistics.median(nearest_distances) if nearest_distances else None
            ),
            saturated=(
                pairs_near > _SATURATED_SHARE * pairs_compared
                and pairs_near > episodes_compared
            ),
        )

    # Step 5: one finding per episode and tier, in episode order.
    made: dict[tuple[str, Literal["exact", "near"]], tuple[int, dict]] = {}
    for ids in copies.values():
        if len(ids) > 1:
            for episode_id in ids:
                made[(episode_id, "exact")] = (len(ids) - 1, {})
    if summary is not None and not summary.saturated:
        for rep, (nearest_distance, count) in near.items():
            evidence = {
                "max_nrmse": _NEAR_RMSE,
                "nearest_nrmse": round_nrmse(nearest_distance),
            }
            for copy in copies[rep]:
                made[(copy, "near")] = (count, evidence)
    findings = [
        _finding(episode_id, tier, count, dict(evidence))
        for (episode_id, tier), (count, evidence) in sorted(
            made.items(), key=lambda item: (order[item[0][0]], item[0][1])
        )
    ]
    return pairs, findings, summary


def near_duplicate_warning(
    summary: NearDuplicateSummary | None, n_episodes: int
) -> DatasetWarning | None:
    """The dataset warning a saturated near tier writes in place of near findings.

    Parameters
    ----------
    summary : NearDuplicateSummary or None
        The near search's counts, as `find_duplicate_episodes` returned them.
    n_episodes : int
        Every graded episode, whether or not it took part in the search.

    Returns
    -------
    DatasetWarning or None
        The warning, stating how many episodes have a near partner
        and the share of compared pairs that are near.
        `None` when the search did not run or the near tier is not saturated.
    """

    if summary is None or not summary.saturated:
        return None
    share = summary.pairs_near / summary.pairs_compared
    return DatasetWarning(
        code="near_duplicates_saturated",
        message=(
            f"{summary.episodes_with_partner} of {n_episodes} episodes"
            " have a near-identical partner,"
            f" and {summary.pairs_near:,} of {summary.pairs_compared:,}"
            f" compared pairs ({share:.1%}) are near:"
            " the near-duplicate check cannot tell copies from re-runs here,"
            " so no episode carries a near-duplicate finding."
        ),
    )
