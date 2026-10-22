"""Verifies the duplicate-episode pass pairs copied payloads,
and leaves re-runs alone."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# External
import polars as pl
import pytest

# Internal
from kalanos.analysis.models.eligibility import Consequence
from kalanos.analysis.models.metrics import Level
from kalanos.analysis.models.report import CameraSummary, GradedEpisode
from kalanos.analysis.models.scoring import ScoreResult
from kalanos.analysis.scoring import duplicates
from kalanos.analysis.scoring.duplicates import (
    _MAX_PARTNERS,
    _NEAR_RMSE,
    _Nearest,
    _search,
    close_pairs,
    find_duplicate_episodes,
    payload_fingerprint,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

CHANNELS = ["joint_0", "joint_1", "joint_2"]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _arm(
    phase: float = 0.0, *, rows: int = 200, amplitude: float = 1.0
) -> pl.DataFrame:
    """Three smooth joint trajectories, each a sine at its own frequency,
    spanning the same motion from first row to last whatever `rows` is."""

    return pl.DataFrame(
        {
            name: [
                amplitude * math.sin(2 * math.pi * (j + 1) * i / (rows - 1) + phase)
                for i in range(rows)
            ]
            for j, name in enumerate(CHANNELS)
        }
    )


def _run(*frames: pl.DataFrame, cameras: list[CameraSummary] | None = None):
    """Run the pass over one episode per frame, each a single numeric stream,
    and return its pairs, findings and near-duplicate summary."""

    score = ScoreResult(
        level=Level.EPISODE, score=None, grade=None, train_ready=None, n_contributing=0
    )
    episodes = [
        GradedEpisode(
            id=f"episode_{index}", adapter="test", adapter_confidence=1.0, score=score
        )
        for index in range(len(frames))
    ]
    key = ("proprio.joint_position", None, "observation.state", 0)
    fingerprints = {}
    for episode, frame in zip(episodes, frames, strict=True):
        fingerprint = payload_fingerprint(frame, list(frame.columns))
        fingerprints[episode.id] = {key: fingerprint} if fingerprint else {}
    return find_duplicate_episodes(episodes, fingerprints, cameras or [])


def _find(*frames: pl.DataFrame, cameras: list[CameraSummary] | None = None):
    """Run the pass over one episode per frame, and return its pairs and findings."""

    pairs, findings, _summary = _run(*frames, cameras=cameras)
    return pairs, findings


def _alike(n: int) -> list[pl.DataFrame]:
    """`n` re-runs of one motion, each within the near limit of every other."""

    return [_arm(1e-4 * index) for index in range(n)]


def _ids(findings, metric_id: str) -> list[str]:
    """The episodes carrying a finding of `metric_id`, in order."""

    return [f.episode_id for f in findings if f.metric_id == metric_id]


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_identical_payloads_are_exact_duplicates():
    """A copied payload pairs with its original, and both are held for review."""

    episodes = [_arm(), _arm(1.0), _arm()]

    pairs, findings = _find(*episodes)

    assert [(p.episodes, p.tier, p.nrmse) for p in pairs] == [
        (["episode_0", "episode_2"], "exact", 0.0)
    ]
    assert _ids(findings, "integrity.duplicate_episode") == ["episode_0", "episode_2"]
    assert {f.consequence for f in findings} == {Consequence.REVIEW}


def test_exact_copies_pair_with_the_first_copy_alone():
    """Four copies give three pairs, each against the earliest copy,
    while every copy is still flagged and held for review."""

    episodes = [_arm(), _arm(1.0), _arm(), _arm(), _arm()]

    pairs, findings = _find(*episodes)

    assert [(p.episodes, p.tier) for p in pairs] == [
        (["episode_0", "episode_2"], "exact"),
        (["episode_0", "episode_3"], "exact"),
        (["episode_0", "episode_4"], "exact"),
    ]
    flagged = {
        f.episode_id: (f.value, f.evidence["partners"], f.consequence)
        for f in findings
        if f.metric_id == "integrity.duplicate_episode"
    }
    assert flagged == {
        "episode_0": (3.0, ["episode_2", "episode_3", "episode_4"], Consequence.REVIEW),
        "episode_2": (3.0, ["episode_0"], Consequence.REVIEW),
        "episode_3": (3.0, ["episode_0"], Consequence.REVIEW),
        "episode_4": (3.0, ["episode_0"], Consequence.REVIEW),
    }


def test_a_near_copy_of_one_copy_is_near_every_copy():
    """A float32 copy of a group of exact copies pairs once, with the earliest copy,
    and every copy in the group still carries the near finding."""

    pytest.importorskip("numpy")

    copy = _arm().with_columns(pl.all().cast(pl.Float32))

    pairs, findings = _find(_arm(), _arm(1.0), _arm(), _arm(), copy)

    assert [(p.episodes, p.tier) for p in pairs] == [
        (["episode_0", "episode_2"], "exact"),
        (["episode_0", "episode_3"], "exact"),
        (["episode_0", "episode_4"], "near"),
    ]
    near = {
        f.episode_id: (f.value, f.evidence["partners"])
        for f in findings
        if f.metric_id == "integrity.near_duplicate_episode"
    }
    assert near == {
        "episode_0": (1.0, ["episode_4"]),
        "episode_2": (1.0, ["episode_4"]),
        "episode_3": (1.0, ["episode_4"]),
        "episode_4": (3.0, ["episode_0"]),
    }


def test_a_float32_copy_is_a_near_duplicate():
    """A copy re-saved as float32 no longer hashes alike, but still pairs."""

    pytest.importorskip("numpy")

    original = _arm()
    copy = original.with_columns(pl.all().cast(pl.Float32))

    pairs, findings = _find(original, _arm(1.0), copy)

    assert [(p.episodes, p.tier) for p in pairs] == [
        (["episode_0", "episode_2"], "near")
    ]
    assert _ids(findings, "integrity.near_duplicate_episode") == [
        "episode_0",
        "episode_2",
    ]
    assert {f.consequence for f in findings} == {Consequence.REPORT_ONLY}


def test_a_noisy_copy_is_a_near_duplicate():
    """Noise at 1% of each channel's spread keeps a copy within the near limit."""

    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(42)
    original = _arm()
    copy = original.with_columns(
        pl.col(name)
        + rng.normal(
            0.0, 0.01 * float(np.std(original[name].to_numpy())), original.height
        )
        for name in CHANNELS
    )

    pairs, _findings = _find(original, _arm(1.0), copy)

    assert [(p.episodes, p.tier) for p in pairs] == [
        (["episode_0", "episode_2"], "near")
    ]
    assert pairs[0].nrmse <= _NEAR_RMSE


def test_different_trajectories_are_not_paired():
    """Re-runs of one motion, shifted in phase or scaled, stay apart."""

    pytest.importorskip("numpy")

    episodes = [
        _arm(),
        _arm(0.3),
        _arm(amplitude=0.8),
        _arm(2.0),
    ]

    pairs, findings = _find(*episodes)

    assert pairs == []
    assert findings == []


def test_a_large_offset_does_not_hide_a_small_motion():
    """Re-runs that move by a thousandth around a large offset stay apart,
    though float32 cannot hold the motion on top of the offset."""

    pytest.importorskip("numpy")

    episodes = [
        _arm(phase, amplitude=1e-3).with_columns(pl.all() + 1e5)
        for phase in (0.0, 1.0, 2.0)
    ]

    pairs, findings = _find(*episodes)

    assert pairs == []
    assert findings == []


def test_constant_payloads_never_pair():
    """Episodes that log only constants cannot be told apart, so take no part."""

    pytest.importorskip("numpy")

    flat = pl.DataFrame({name: [0.0] * 50 for name in CHANNELS})

    pairs, findings = _find(flat, flat)

    assert pairs == []
    assert findings == []


def test_episodes_with_different_channels_are_not_compared():
    """A payload is compared only against payloads holding the same channels."""

    original = _arm()
    renamed = original.rename({"joint_2": "gripper"})

    pairs, _findings = _find(original, renamed)

    assert pairs == []


def test_a_duration_mismatch_is_not_near():
    """The same motion over half as many rows again is not a copy;
    over 5% more rows, it is."""

    pytest.importorskip("numpy")

    longer, _findings = _find(_arm(rows=100), _arm(rows=150))
    close, _findings = _find(_arm(rows=100), _arm(rows=105))

    assert longer == []
    assert [(p.episodes, p.tier) for p in close] == [
        (["episode_0", "episode_1"], "near")
    ]


def test_identical_footage_is_noted():
    """A pair that also repeats a camera's footage exactly names that camera."""

    pytest.importorskip("numpy")

    camera = CameraSummary(
        camera="observation.images.top",
        stream="extero.exterior_rgb",
        n_episodes=2,
        compared=False,
        repeated_footage=[["episode_0", "episode_1"]],
    )

    pairs, findings = _find(_arm(), _arm(), cameras=[camera])

    assert pairs[0].footage_identical
    assert pairs[0].cameras == ["observation.images.top"]
    assert findings[0].evidence["footage_identical"] == [True]


def test_footage_is_noted_on_each_pair_against_the_first_copy():
    """Each pair against the earliest copy names the footage that pair repeats,
    not footage the later copies repeat among themselves."""

    camera = CameraSummary(
        camera="observation.images.top",
        stream="extero.exterior_rgb",
        n_episodes=3,
        compared=False,
        repeated_footage=[["episode_0", "episode_2"]],
    )
    wrist = camera.model_copy(
        update={
            "camera": "observation.images.wrist",
            "repeated_footage": [["episode_1", "episode_2"]],
        }
    )

    pairs, findings = _find(_arm(), _arm(), _arm(), cameras=[camera, wrist])

    assert [(p.episodes, p.footage_identical, p.cameras) for p in pairs] == [
        (["episode_0", "episode_1"], False, []),
        (["episode_0", "episode_2"], True, ["observation.images.top"]),
    ]
    assert {f.episode_id: f.evidence["footage_identical"] for f in findings} == {
        "episode_0": [False, True],
        "episode_1": [False],
        "episode_2": [True],
    }


def test_a_non_numeric_channel_has_no_fingerprint():
    """A payload that does not cast to float64 is left out rather than guessed at."""

    frame = pl.DataFrame({"label": ["a", "b", "c"]})

    assert payload_fingerprint(frame, ["label"]) is None


def test_a_noisy_copy_with_a_sparse_reward_still_pairs():
    """A reward that is 1 on its last row alone does not inflate a copy's distance."""

    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(42)
    frames = []
    for phase in (0.0, 0.5, 1.0, 1.5):
        frame = _arm(phase)
        frames.append(
            frame.with_columns(reward=pl.Series([0.0] * (frame.height - 1) + [1.0]))
        )
    original = frames[0]
    copy = original.with_columns(
        pl.col(name)
        + rng.normal(
            0.0, 0.01 * float(np.std(original[name].to_numpy())), original.height
        )
        for name in original.columns
    )

    pairs, _findings = _find(*frames, copy)

    assert [(p.episodes, p.tier) for p in pairs] == [
        (["episode_0", "episode_4"], "near")
    ]


def test_close_pairs_agree_with_every_pair_across_blocks(monkeypatch):
    """Row blocks smaller than the episode count
    find the same pairs as a search over every pair."""

    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(40, 6))
    vectors[25] = vectors[3] + 0.001
    vectors[39] = vectors[25]
    limit = 0.3
    expected = [
        (i, j)
        for i in range(40)
        for j in range(i + 1, 40)
        if np.sqrt(np.mean((vectors[i] - vectors[j]) ** 2)) <= limit
    ]
    monkeypatch.setattr(duplicates, "_CHUNK", 7 * 40)

    found = close_pairs(vectors, limit)

    assert [(i, j) for i, j, _d in found] == expected
    assert (3, 25) in expected and (25, 39) in expected


def test_a_struct_channel_has_no_fingerprint():
    """A nested column is left out rather than read as numbers."""

    frame = pl.DataFrame({"pose": [{"x": 1.0}, {"x": 2.0}]})

    assert payload_fingerprint(frame, ["pose"]) is None


def _every_close_pair(np, vectors, limit):
    """Every pair within `limit`, by comparing each pair directly."""

    n = len(vectors)
    return [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if np.sqrt(np.mean((vectors[i] - vectors[j]) ** 2)) <= limit
    ]


def test_the_pruned_search_finds_every_pair_the_full_search_finds(monkeypatch):
    """Projecting and sorting skips only pairs that cannot be within the limit."""

    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(42)
    # Spread along a few directions, as distinct motions are, plus small noise.
    vectors = (
        rng.normal(size=(600, 4)) @ rng.normal(size=(4, 24))
        + rng.normal(0, 0.05, (600, 24))
    ).astype(np.float32)
    for copy, original in ((590, 7), (591, 300), (592, 590)):
        vectors[copy] = vectors[original] + rng.normal(0, 0.002, 24)
    monkeypatch.setattr(duplicates, "_PRUNE_FROM", 10)
    monkeypatch.setattr(
        duplicates,
        "_all_close_pairs",
        lambda *_args: pytest.fail("the pruned search fell back to every pair"),
    )

    for limit in (0.02, 0.15):
        found = close_pairs(vectors, limit)

        assert [(i, j) for i, j, _d in found] == _every_close_pair(np, vectors, limit)
    assert (7, 590) in [(i, j) for i, j, _d in close_pairs(vectors, 0.02)]


def test_rows_too_close_to_prune_are_all_compared(monkeypatch):
    """Near-identical rows defeat the projections, so every pair is compared."""

    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(42)
    vectors = (np.ones((300, 16)) + rng.normal(0, 0.01, (300, 16))).astype(np.float32)
    calls = []
    every = duplicates._all_close_pairs
    monkeypatch.setattr(duplicates, "_PRUNE_FROM", 10)
    monkeypatch.setattr(
        duplicates,
        "_all_close_pairs",
        lambda *args: calls.append(1) or every(*args),
    )

    found = close_pairs(vectors, 0.012)

    assert calls
    assert [(i, j) for i, j, _d in found] == _every_close_pair(np, vectors, 0.012)


def test_a_copy_far_from_every_other_episode_still_pairs():
    """Rounding in a far-out row's length does not drop its copy from the shortlist."""

    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(50, 448)).astype(np.float32)
    vectors[0] = 1000.0 + rng.normal(size=448)
    vectors[1] = vectors[0] + rng.normal(0, 0.005, 448)

    found = close_pairs(vectors, _NEAR_RMSE)

    assert [(i, j) for i, j, _d in found] == [(0, 1)]


def test_alike_episodes_saturate_the_near_tier():
    """When every pair is near, the summary says so once,
    and no episode carries a near finding."""

    pytest.importorskip("numpy")

    pairs, findings, summary = _run(*_alike(30))

    assert summary is not None
    assert summary.saturated
    assert summary.pairs_near == summary.pairs_compared == 435
    assert summary.episodes_compared == summary.episodes_with_partner == 30
    assert _ids(findings, "integrity.near_duplicate_episode") == []
    assert 0 < sum(p.tier == "near" for p in pairs) <= _MAX_PARTNERS * 30


def test_near_output_grows_linearly():
    """The near pairs listed stay within a fixed number per episode,
    however many pairs are near."""

    pytest.importorskip("numpy")

    for n, near in ((20, 190), (60, 1_770)):
        pairs, _findings, summary = _run(*_alike(n))

        assert summary is not None
        assert summary.pairs_near == near
        assert sum(p.tier == "near" for p in pairs) <= _MAX_PARTNERS * n


def test_a_cluster_lists_its_five_nearest_partners():
    """In a cluster of 8 alike episodes among distinct ones,
    each lists its 5 nearest partners and counts all 7."""

    pytest.importorskip("numpy")

    distinct = [_arm(0.5 * (index + 1)) for index in range(12)]

    pairs, findings, summary = _run(*_alike(8), *distinct)

    cluster = {f"episode_{index}" for index in range(8)}
    assert summary is not None
    assert not summary.saturated
    assert (summary.pairs_near, summary.pairs_compared) == (28, 190)
    assert all(set(p.episodes) <= cluster for p in pairs)
    near = {
        f.episode_id: f
        for f in findings
        if f.metric_id == "integrity.near_duplicate_episode"
    }
    assert set(near) == cluster
    for finding in near.values():
        assert finding.value == 7.0
        assert len(finding.evidence["partners"]) == _MAX_PARTNERS
        assert finding.evidence["nrmse"] == sorted(finding.evidence["nrmse"])
        assert finding.evidence["partners_omitted"] == 2


def test_two_near_episodes_never_saturate():
    """A single near pair is every pair compared, and is still reported."""

    pytest.importorskip("numpy")

    copy = _arm().with_columns(pl.all().cast(pl.Float32))

    _pairs, findings, summary = _run(_arm(), copy)

    assert summary is not None
    assert summary.pairs_near == summary.pairs_compared == 1
    assert not summary.saturated
    assert _ids(findings, "integrity.near_duplicate_episode") == [
        "episode_0",
        "episode_1",
    ]


def test_exact_copies_are_unchanged_when_saturated():
    """Exact copies are paired and held for review whether or not
    the near tier is saturated."""

    pytest.importorskip("numpy")

    pairs, findings, summary = _run(*_alike(30), _arm(), _arm())

    assert summary is not None
    assert summary.saturated
    assert [(p.episodes, p.tier) for p in pairs if p.tier == "exact"] == [
        (["episode_0", "episode_30"], "exact"),
        (["episode_0", "episode_31"], "exact"),
    ]
    assert {
        f.episode_id: (f.value, f.evidence["partners"], f.consequence) for f in findings
    } == {
        "episode_0": (2.0, ["episode_30", "episode_31"], Consequence.REVIEW),
        "episode_30": (2.0, ["episode_0"], Consequence.REVIEW),
        "episode_31": (2.0, ["episode_0"], Consequence.REVIEW),
    }


def _nearest_from_pairs(found, n):
    """Each row's near partner count and nearest partners, from every close pair."""

    neighbours: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for i, j, distance in found:
        neighbours[i].append((distance, j))
        neighbours[j].append((distance, i))
    return [len(row) for row in neighbours], [
        sorted(row)[:_MAX_PARTNERS] for row in neighbours
    ]


def test_nearest_partners_agree_across_blocks(monkeypatch):
    """Pairs streamed a block at a time keep the same nearest partners per row
    as sorting every close pair, on both search paths."""

    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(42)
    # Spread along a few directions, so the pruned search does not give up,
    # with a cluster of 9 near copies, each with more partners than are kept.
    vectors = (
        rng.normal(size=(400, 4)) @ rng.normal(size=(4, 24))
        + rng.normal(0, 0.05, (400, 24))
    ).astype(np.float32)
    for copy in range(390, 398):
        vectors[copy] = vectors[7] + rng.normal(0, 0.002, 24)
    vectors[398] = vectors[300] + rng.normal(0, 0.002, 24)
    limit = 0.3
    monkeypatch.setattr(duplicates, "_CHUNK", 7 * 24)

    for pruned in (False, True):
        if pruned:
            monkeypatch.setattr(duplicates, "_PRUNE_FROM", 10)
            monkeypatch.setattr(
                duplicates,
                "_all_close_pairs",
                lambda *_args: pytest.fail("the pruned search fell back to every pair"),
            )
        nearest = _Nearest(len(vectors), np.ones(len(vectors), dtype=np.int64))

        _search(vectors, limit, nearest.add)

        found = close_pairs(vectors, limit)
        counts, expected = _nearest_from_pairs(found, len(vectors))
        assert max(counts) > _MAX_PARTNERS
        assert nearest.counts.tolist() == counts
        assert nearest.total == len(found)
        for row, partners in enumerate(expected):
            kept = [
                (float(d), int(p))
                for d, p in zip(
                    nearest.distance[row], nearest.partner[row], strict=True
                )
                if p >= 0
            ]
            assert [p for _d, p in kept] == [p for _d, p in partners]
            assert [d for d, _p in kept] == pytest.approx([d for d, _p in partners])


def test_without_numpy_there_is_no_summary(monkeypatch):
    """Without numpy only exact copies are paired, and no summary is made."""

    monkeypatch.setattr(duplicates, "load_numpy", lambda: None)

    pairs, _findings, summary = _run(_arm(), _arm(1.0), _arm())

    assert summary is None
    assert [(p.episodes, p.tier) for p in pairs] == [
        (["episode_0", "episode_2"], "exact")
    ]
