"""Verifies task instructions end to end: how adapters read them, the dataset
rule that decides whether they apply, and the `task_instruction_missing` metric.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import shutil
from pathlib import Path

# External
import h5py
import numpy as np
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters.hdf5 import Hdf5Adapter
from kalanos.analysis.adapters.lerobot.v2 import LeRobotV2Adapter
from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter
from kalanos.analysis.inference.tasks import as_task_list, dataset_tasks
from kalanos.analysis.metrics.annotation import task_instruction_missing
from kalanos.analysis.models.domain import Episode
from kalanos.analysis.models.metrics import EpisodeContext, MetricStatus


_FIXTURES = Path(__file__).parent / "fixtures"


def _flag(tasks):
    """Run the metric on an episode carrying `tasks`."""

    return task_instruction_missing(
        EpisodeContext(episode=Episode(id="e", tasks=tasks))
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_an_episode_with_an_instruction_is_not_flagged():
    """One non-blank instruction is enough."""

    result = _flag(["pick up the red cube"])

    assert result.status == MetricStatus.REPORT_ONLY
    assert result.value == 0.0
    assert result.unit == "flag"


@pytest.mark.parametrize("tasks", [[], [""], ["   "], ["", "\t\n"]])
def test_no_or_blank_instructions_are_flagged(tasks):
    """No instruction, or only empty or whitespace ones, counts as missing."""

    result = _flag(tasks)

    assert result.value == 1.0
    assert result.evidence == {"n_tasks": len(tasks), "n_blank": len(tasks)}


def test_a_dataset_without_instructions_is_not_applicable():
    """`None` means the dataset never carried instructions: nothing is missing."""

    result = _flag(None)

    assert result.status == MetricStatus.NOT_APPLICABLE
    assert "no task instruction for any episode" in result.evidence["reason"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("stack the cups", ["stack the cups"]),
        (b"stack the cups", ["stack the cups"]),
        (["a", "b"], ["a", "b"]),
        (np.array([b"a", b"b"]), ["a", "b"]),
        ([], []),
    ],
)
def test_stored_instructions_normalise_to_strings(raw, expected):
    """Strings, bytes and lists or arrays of either all become a list of str."""

    assert as_task_list(raw) == expected


def test_the_dataset_rule_keeps_instructions_only_when_some_are_non_blank():
    """A dataset with at least one real instruction keeps every episode's list."""

    assert dataset_tasks({0: ["go"], 1: []}) == {0: ["go"], 1: []}
    assert dataset_tasks({0: [], 1: ["  "]}) is None
    assert dataset_tasks({}) is None


@pytest.mark.parametrize(
    ("fixture", "adapter"),
    [
        ("lerobot_v2_0_tiny", LeRobotV2Adapter),
        ("lerobot_v2_1_tiny", LeRobotV2Adapter),
        ("lerobot_v3_tiny", LeRobotV3Adapter),
    ],
)
def test_lerobot_fixtures_carry_their_instructions(fixture, adapter):
    """Every LeRobot fixture reads its per-episode task list."""

    episodes = list(adapter().episodes(UPath(_FIXTURES / fixture)))

    assert [episode.tasks for episode in episodes] == [["pick up the cube"]] * 2


def test_a_lerobot_v2_episode_with_an_empty_task_list_reads_as_empty(tmp_path):
    """One episode losing its instruction becomes `[]`, the rest keep theirs."""

    root = tmp_path / "v2"
    shutil.copytree(_FIXTURES / "lerobot_v2_1_tiny", root)
    path = root / "meta" / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    records[1]["tasks"] = []
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    episodes = list(LeRobotV2Adapter().episodes(UPath(root)))

    assert [episode.tasks for episode in episodes] == [["pick up the cube"], []]


def test_a_lerobot_v3_dataset_with_no_instructions_reads_as_none(tmp_path):
    """Blanking every episode's tasks makes the check not apply, not flag all."""

    root = tmp_path / "v3"
    shutil.copytree(_FIXTURES / "lerobot_v3_tiny", root)
    [meta] = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    frame = pl.read_parquet(meta)
    frame.with_columns(
        pl.Series("tasks", [[""]] * frame.height, dtype=pl.List(pl.String))
    ).write_parquet(meta)

    episodes = list(LeRobotV3Adapter().episodes(UPath(root)))

    assert [episode.tasks for episode in episodes] == [None, None]


def _write_hdf5(path, tasks):
    """Write one demo group per entry of `tasks`, setting its `task` attr if given."""

    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        for index, task in enumerate(tasks):
            group = data.create_group(f"demo_{index}")
            group.create_dataset("actions", data=np.zeros((5, 2)))
            if task is not None:
                group.attrs["task"] = task


def test_hdf5_reads_the_task_attribute_and_flags_the_gap(tmp_path):
    """An episode group without the attribute reads as `[]` when others have one."""

    path = tmp_path / "demos.hdf5"
    _write_hdf5(path, ["open the drawer", None, "close the drawer"])

    episodes = list(Hdf5Adapter().episodes(UPath(path)))

    assert [episode.tasks for episode in episodes] == [
        ["open the drawer"],
        [],
        ["close the drawer"],
    ]


def test_hdf5_without_any_task_attribute_reads_as_none(tmp_path):
    """No group carries an instruction: the dataset is not language-annotated."""

    path = tmp_path / "plain.hdf5"
    _write_hdf5(path, [None, None])

    assert [e.tasks for e in Hdf5Adapter().episodes(UPath(path))] == [None, None]


def test_sampling_still_judges_the_whole_dataset(tmp_path):
    """Grading one episode must not decide the rule from that episode alone.

    Only the second group has an instruction; sampling the first alone must
    still report it missing (`[]`), not declare the dataset un-annotated.
    """

    path = tmp_path / "sampled.hdf5"
    _write_hdf5(path, [None, "wipe the table"])

    [first] = list(Hdf5Adapter().episodes(UPath(path), sample=1))

    assert first.tasks == []


def test_the_report_carries_each_episodes_instructions(tmp_path):
    """The graded report shows what each episode was told, and the flag."""

    from kalanos.api import grade  # local: keeps these tests import-light

    path = tmp_path / "graded.hdf5"
    _write_hdf5(path, ["open the drawer", None])

    report = grade(path)

    by_id = {episode.id: episode for episode in report.episodes}
    tasks = sorted((episode.tasks for episode in by_id.values()), key=len)
    assert tasks == [[], ["open the drawer"]]
    flags = sorted(
        episode.metrics["task_instruction_missing"].value for episode in by_id.values()
    )
    assert flags == [0.0, 1.0]
    assert report.score.score is not None
