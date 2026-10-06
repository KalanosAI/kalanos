"""Verifies the dataset gate: failing episodes, task traits, the cap, pruning,
coverage, and the packaged policies that switch it on, off, or extend it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import re
from pathlib import Path
from typing import cast

# External
import h5py
import numpy as np
import pytest
from calibration_helpers import grade_with_test_calibration

# Internal
from kalanos.analysis.models.policy import GatePolicy
from kalanos.analysis.models.scoring import Grade
from kalanos.analysis.scoring.gate import cap_for, dataset_traits, task_traits, worse
from kalanos.api import grade
from kalanos.assets.policy import load_default_policy, load_policy

# Local
from helpers import write_arm as _write_arm


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_cap_table_follows_the_share_of_failing_episodes():
    """0-5% no cap, to 15% B, to 30% C, beyond D (docs/METRICS.md)."""

    gate = load_default_policy().gate
    assert gate is not None
    assert cap_for(0.0, gate) is None
    assert cap_for(0.05, gate) is None
    assert cap_for(0.06, gate) == Grade.B
    assert cap_for(0.16, gate) == Grade.C
    assert cap_for(0.5, gate) == Grade.D


def test_a_cap_table_that_does_not_reach_every_share_is_refused():
    """The last row must cover a dataset where every episode fails."""

    with pytest.raises(ValueError, match="end at 1.0"):
        GatePolicy.model_validate(
            {"caps": [{"max_failing_share": 0.5}], "task_trait_min_episodes": 5}
        )


def test_worse_keeps_the_lower_letter_and_ignores_no_cap():
    """A cap only ever lowers a letter."""

    assert worse(Grade.A, Grade.C) == Grade.C
    assert worse(Grade.D, Grade.B) == Grade.D
    assert worse(Grade.A, None) == Grade.A


def test_a_finding_on_every_episode_of_one_task_is_a_trait():
    """A sweeping task never closes the gripper: that is the task, not a fault."""

    task_of = {f"e{i}": ("sweep" if i < 25 else "pick") for i in range(50)}
    critical = {f"e{i}": {"state/gripper.flatline_pct"} for i in range(25)}
    critical["e30"] = {"action/joint_2.spike_pct"}

    traits = task_traits(critical, task_of, min_episodes=20)

    assert traits == {"state/gripper.flatline_pct": ("sweep", 25)}


def test_a_finding_that_misses_one_episode_of_its_task_is_not_a_trait():
    """All of one task's episodes, or it is a fault on the ones it hits."""

    task_of = {f"e{i}": "sweep" for i in range(25)}
    critical = {f"e{i}": {"state/gripper.flatline_pct"} for i in range(24)}

    assert task_traits(critical, task_of, min_episodes=20) == {}


def test_a_small_task_cannot_make_a_trait():
    """Below the minimum, a shared finding may be coincidence: it still fails."""

    task_of = {f"e{i}": "rare" for i in range(3)}
    critical = {f"e{i}": {"x.spike_pct"} for i in range(3)}

    assert task_traits(critical, task_of, min_episodes=20) == {}


def test_glitched_episodes_cap_the_dataset_and_pruning_restores_it(tmp_path):
    """Four glitched episodes in twenty (20%) cap an otherwise clean dataset at C.

    The number stays the mean; the letter carries the gate; the report says which
    episodes fail, why, and the grade without them.
    """

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched={2, 7, 11, 16})

    report = grade_with_test_calibration(path)
    gate = report.gate

    assert gate is not None
    assert {f.episode_id.rsplit("_", 1)[-1] for f in gate.failing_episodes} == {
        "2",
        "7",
        "11",
        "16",
    }
    assert gate.failing_share == pytest.approx(0.2)
    assert gate.cap == Grade.C
    assert report.score.grade == Grade.C
    assert gate.uncapped_grade is not None and gate.uncapped_grade != Grade.C
    assert report.score.train_ready is False
    # `pruned_score` describes the non-blocked candidate set; since schema 7
    # nothing claims that set is train-ready or sufficient.
    assert report.score.score is not None
    assert gate.pruned_score is not None and gate.pruned_score > report.score.score
    assert not hasattr(gate, "train_ready_after_pruning")
    assert not hasattr(gate, "pruned_grade")
    assert (
        report.sufficiency is not None and report.sufficiency.status.value == "unknown"
    )
    assert "4 of 20 episodes" in gate.summary
    assert all(f.reasons for f in gate.failing_episodes)


def test_a_clean_dataset_is_not_capped_and_says_what_it_rests_on(tmp_path):
    """No failing episode, no cap — and the coverage line is still there."""

    path = tmp_path / "clean.hdf5"
    _write_arm(path, 20, glitched=set())

    report = grade(path)

    assert report.gate is not None
    assert report.gate.failing_episodes == []
    assert report.gate.cap is None
    assert report.score.grade == report.gate.uncapped_grade
    assert report.gate.coverage.families_graded
    assert report.gate.coverage.graded_checks_per_episode
    assert "no blocking episodes" in report.gate.summary
    assert "Graded on" in report.gate.summary


def test_the_legacy_policy_grades_by_the_mean_alone(tmp_path):
    """legacy_0_5 reproduces 0.5 grades: no gate, the mean's own letter."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched={2, 7, 11, 16})

    report = grade(path, policy=load_policy(Path("legacy_0_5")))

    assert report.gate is None
    assert report.score.grade != Grade.C


def test_the_language_conditioned_policy_fails_an_episode_with_no_instruction(
    tmp_path,
):
    """For a VLA an episode without its instruction fails; by default it does not."""

    path = tmp_path / "vla.hdf5"
    tasks = ["open the drawer"] * 19 + [None]
    _write_arm(path, 20, glitched=set(), tasks=tasks)

    default = grade(path)
    vla = grade(path, policy=load_policy(Path("language_conditioned")))

    assert default.gate is not None and default.gate.failing_episodes == []
    assert vla.gate is not None
    assert [f.reasons for f in vla.gate.failing_episodes] == [
        ["episode.annotation.task_instruction_missing"]
    ]


def test_an_unknown_base_policy_is_refused(tmp_path):
    """`extends` names only packaged policies."""

    path = tmp_path / "mine.yaml"
    path.write_text("extends: somewhere_else\n")

    with pytest.raises(ValueError, match="packaged policies are"):
        load_policy(path)


def test_a_finding_on_every_episode_is_a_dataset_trait_not_a_failure():
    """A state dimension no episode ever moves describes the recording, not a fault.

    From lerobot/cmu_stretch: two state dimensions never change in any of its
    135 episodes, across all five tasks. Dropping episodes cannot fix that.
    """

    episodes = [f"e{i}" for i in range(135)]
    critical = {
        e: {"state/state_1.flatline_pct", "state/state_3.flatline_pct"}
        for e in episodes
    }
    critical["e7"] |= {"state/state_0.flatline_pct"}

    everywhere = dataset_traits(critical, episodes, min_episodes=20, min_share=0.95)

    assert everywhere == {"state/state_1.flatline_pct", "state/state_3.flatline_pct"}


def test_nearly_every_episode_counts_within_the_gates_own_allowance():
    """Missing from no more episodes than the gate lets fail for free still counts."""

    episodes = [f"e{i}" for i in range(480)]
    critical = {e: {"state/state_14.flatline_pct"} for e in episodes[:479]}

    assert dataset_traits(critical, episodes, 20, min_share=0.95) == {
        "state/state_14.flatline_pct"
    }
    assert dataset_traits(critical, episodes, 20, min_share=1.0) == set()


def test_a_finding_on_most_but_not_nearly_all_episodes_still_fails_them():
    """Below the share, the finding varies between episodes: the gate's business."""

    episodes = [f"e{i}" for i in range(100)]
    critical = {e: {"x.spike_pct"} for e in episodes[:90]}

    assert dataset_traits(critical, episodes, 20, min_share=0.95) == set()


def test_a_small_dataset_has_no_dataset_traits():
    """Below the minimum size, every-episode may be coincidence."""

    episodes = ["a", "b", "c"]
    critical = {e: {"x.flatline_pct"} for e in episodes}

    assert dataset_traits(critical, episodes, 20, min_share=0.95) == set()


def test_a_blocking_finding_on_every_episode_is_reported_as_a_trait_and_still_blocks(
    tmp_path,
):
    """A stuck channel in every episode is a dataset trait *and* blocks every episode.

    Since schema 7 prevalence exempts nothing: the report cannot tell a
    recording convention from corruption in every episode, so it names the
    pattern as a descriptive trait and leaves every episode blocked. A scoped
    policy rule may exempt it explicitly; the gate never does on its own.
    """

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched={2, 7, 11, 16})
    stuck = np.concatenate([np.linspace(0.0, 1.0, 6), np.ones(194)])
    with h5py.File(str(path), "a") as store:
        for index in range(20):
            group = cast(h5py.Group, store[f"data/demo_{index}"])
            group.create_dataset("unused_dim", data=stuck)

    report = grade_with_test_calibration(path)
    gate = report.gate

    assert gate is not None
    assert [t.finding.split("/")[-1] for t in gate.dataset_traits] == [
        "unused_dim.integrity.flatline_pct"
    ]
    assert gate.dataset_traits[0].n_with_finding == 20
    assert len(gate.failing_episodes) == 20
    assert all(
        any(r.endswith("unused_dim.integrity.flatline_pct") for r in f.reasons)
        for f in gate.failing_episodes
    )
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.blocked == 20
    assert report.readiness is not None and report.readiness.score == 0.0
    assert "dataset traits" in gate.summary


def test_a_channel_that_never_changes_is_a_warning_not_a_failure(tmp_path):
    """Constant for the whole episode: unused or disconnected, and the data can't say.

    From lerobot/berkeley_fanuc_manipulation, whose state_7 never changes in 158
    of 415 episodes: it cannot show those episodes are worse than the others.
    """

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched=set())
    with h5py.File(str(path), "a") as store:
        for index in range(0, 20, 3):
            group = cast(h5py.Group, store[f"data/demo_{index}"])
            group.create_dataset("unused_dim", data=np.zeros(200))

    report = grade(path)

    assert report.gate is not None and report.gate.failing_episodes == []
    flags = [f for f in report.findings if f.channel == "unused_dim"]
    assert flags and all(f.severity.value == "warning" for f in flags)


def test_a_channel_that_freezes_partway_is_still_a_stuck_sensor(tmp_path):
    """Moving, then frozen for the rest of the episode: critical, the episode fails."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched=set())
    stuck = np.concatenate([np.linspace(0.0, 1.0, 6), np.ones(194)])
    with h5py.File(str(path), "a") as store:
        group = cast(h5py.Group, store["data/demo_3"])
        group.create_dataset("unused_dim", data=stuck)

    report = grade_with_test_calibration(path)

    assert report.gate is not None
    assert [f.episode_id.rsplit("_", 1)[-1] for f in report.gate.failing_episodes] == [
        "3"
    ]


def test_readiness_counts_blocking_episodes_as_zero(tmp_path):
    """Readiness = sum of passing episodes' quality / evaluated episodes."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched={2, 7, 11, 16})

    report = grade_with_test_calibration(path)
    r = report.readiness

    assert r is not None and r.passing_quality is not None
    assert report.gate is not None
    c = report.eligibility_counts
    assert c is not None
    assert (c.total, c.pass_count, c.blocked, c.review, c.unknown) == (20, 16, 4, 0, 0)
    assert c.confirmed_eligible_share == pytest.approx(0.8)
    assert r.formula_id == "pass-quality-over-known-inventory-v1"
    assert r.reasons == []
    passing = [
        e.score.score
        for e in report.episodes
        if e.id not in {f.episode_id for f in report.gate.failing_episodes}
        and e.score.score is not None
    ]
    assert len(passing) == 16
    assert r.passing_quality == pytest.approx(sum(passing) / 16)
    assert r.score == pytest.approx(sum(passing) / 20)
    assert r.score == pytest.approx(r.passing_quality * 16 / 20)
    assert report.gate.summary.startswith(f"Readiness {r.score:.0f}/100: 4 of 20")


def test_a_clean_dataset_reads_as_its_mean(tmp_path):
    """No blocking episodes: readiness is the mean episode quality."""

    path = tmp_path / "clean.hdf5"
    _write_arm(path, 20, glitched=set())

    report = grade(path)
    r = report.readiness

    assert r is not None
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.blocked == 0
    assert r.score == pytest.approx(r.passing_quality)


def test_readiness_derives_from_eligibility_not_from_the_gate(tmp_path):
    """legacy_0_5 has no gate; eligibility, counts and readiness exist regardless.

    The gate only caps a compatibility letter. The decision lives on each
    episode, so a policy without a gate still decides and still reports.
    """

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched={2})

    report = grade_with_test_calibration(path, policy=load_policy(Path("legacy_0_5")))

    assert report.gate is None
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.blocked == 1
    assert report.readiness is not None and report.readiness.score is not None
    assert all(e.eligibility is not None for e in report.episodes)


def test_the_terminal_card_leads_with_readiness_and_shows_no_letter(tmp_path):
    """The headline shows Readiness n/100 and the blocking count, and no letter."""

    from kalanos.analysis.reporting.card import render_terminal

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched={2, 7, 11, 16})
    report = grade_with_test_calibration(path)

    text = render_terminal(report, width=120)

    assert report.readiness is not None
    assert re.search(rf"^\s+Readiness\s+{report.readiness.score:.0f}/100", text, re.M)
    assert "blocked 4" in text
    assert "Assessment  Blocked episodes found" in text
    header = text.splitlines()[:8]
    assert not any(
        line.split() and line.split()[-1] in {"A", "B", "C", "D", "F"}
        for line in header
    )


def test_jitter_alone_cannot_make_an_unknown_clock_observable(tmp_path):
    """Two jittery time axes provide no producer declaration of capture origin."""

    path = tmp_path / "arm.hdf5"
    _write_arm(path, 20, glitched=set(), jittered={0, 1})

    report = grade(path)

    assert report.gate is not None
    not_observable = {item.metric: item for item in report.gate.coverage.not_observable}
    assert {"effective_hz", "drop_rate"} <= not_observable.keys()
    assert not_observable["drop_rate"].share == pytest.approx(1.0)
