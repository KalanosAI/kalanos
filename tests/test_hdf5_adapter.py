"""Verifies the generic HDF5 adapter: structural episode detection,
the cycle-guarded walk, the declared/synthesised rate, and the dictionary lookup —
including that a robomimic-shaped file reads correctly with no code path of its own.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import h5py
import numpy as np
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters.hdf5 import Hdf5Adapter
from kalanos.analysis.models.adapters import AdapterRefusal
from kalanos.analysis.models.domain import Clock, TimestampDtype
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, HDF5_FIXTURE


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _write_demos(
    path,
    timestamps,
    *,
    key="timestamps",
    rate=None,
    dtype: type[np.floating] = np.float64,
):
    """Write a two-demo robomimic-shaped file, each demo carrying `timestamps`."""

    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        if rate is not None:
            data.attrs["fps"] = rate
        for name in ("demo_0", "demo_1"):
            group = data.create_group(name)
            n = len(timestamps)
            group.create_dataset("actions", data=np.zeros((n, 2), dtype=np.float32))
            group.create_dataset(key, data=np.asarray(timestamps, dtype=dtype))


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_contract(tmp_path):
    """Verify the adapter holds every property `check_adapter` requires."""

    decoy_path = tmp_path / "empty.hdf5"
    with h5py.File(str(decoy_path), "w") as store:
        store.create_group("nothing")

    check_adapter(Hdf5Adapter(), HDF5_FIXTURE, [CSV_FIXTURE, decoy_path])


def test_two_sibling_groups_are_read_as_two_episodes():
    """Verify the fixture's two sibling groups yield two episodes."""

    episodes = list(Hdf5Adapter().episodes(HDF5_FIXTURE))

    assert [e.id for e in episodes] == ["runs_session_a", "runs_session_b"]
    lengths = [len(episode.streams[0].timestamps) for episode in episodes]
    assert lengths == [5, 4]


def test_a_robomimic_shaped_file_is_read_without_a_special_case(tmp_path):
    """Verify a robomimic-shaped tree reads correctly through generic detection.

    `data/demo_N` with `actions`/`rewards`/`obs` is robomimic's own layout,
    but nothing in the adapter names it — this proves the sibling-group
    heuristic reads it correctly anyway.
    """

    path = tmp_path / "robomimic.hdf5"
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        data.attrs["control_freq"] = 20.0
        for name, length in (("demo_0", 5), ("demo_1", 4)):
            demo = data.create_group(name)
            demo.create_dataset("actions", data=np.zeros((length, 3), dtype=np.float32))
            demo.create_dataset("rewards", data=np.zeros((length,), dtype=np.float32))
            demo.create_group("obs")

    adapter = Hdf5Adapter()
    upath = UPath(path)

    assert adapter.detect(upath) == 0.3
    info = adapter.describe(upath)
    assert info.nominal_rate_hz == pytest.approx(20.0)
    assert info.episode_count == 2

    episodes = list(adapter.episodes(upath))
    assert [e.id for e in episodes] == ["data_demo_0", "data_demo_1"]


def test_a_single_episode_group_is_read_as_one_episode(tmp_path):
    """Verify a lone episode group with no sibling split reads as one episode."""

    path = tmp_path / "single.hdf5"
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((5, 2), dtype=np.float32))

    adapter = Hdf5Adapter()
    upath = UPath(path)

    assert adapter.describe(upath).episode_count == 1
    [episode] = list(adapter.episodes(upath))
    assert episode.id == "root"
    assert any(s.source_field == "actions" for s in episode.streams)


def test_a_flat_file_with_no_grouping_is_one_episode(tmp_path):
    """Verify datasets sitting directly at the root read as one episode."""

    path = tmp_path / "flat.hdf5"
    with h5py.File(str(path), "w") as store:
        store.create_dataset("actions", data=np.zeros((5, 2), dtype=np.float32))

    adapter = Hdf5Adapter()
    upath = UPath(path)

    assert adapter.describe(upath).episode_count == 1
    [episode] = list(adapter.episodes(upath))
    assert episode.id == "root"


def test_a_file_with_no_dataset_anywhere_is_declined(tmp_path):
    """Verify a file with no dataset at any depth is declined rather than misread."""

    path = tmp_path / "empty.hdf5"
    with h5py.File(str(path), "w") as store:
        store.create_group("nothing")

    adapter = Hdf5Adapter()
    upath = UPath(path)

    assert adapter.detect(upath) == 0.0
    with pytest.raises(AdapterRefusal):
        adapter.describe(upath)


def test_a_cyclic_group_graph_terminates(tmp_path):
    """Verify a hard-linked cycle does not hang the whole-file recursive walk.

    A single group with no sibling split forces `_find_episodes` to descend
    into the whole-file fallback, so `a/loop`'s self-link is actually
    traversed. Without the cycle guard the walk never returns, which is
    itself the failure signal here — there is no timeout machinery.
    """

    path = tmp_path / "cyclic.hdf5"
    with h5py.File(str(path), "w") as store:
        a = store.create_group("a")
        a.create_dataset("x", data=np.zeros((3,), dtype=np.float32))
        store["a/loop"] = store["a"]

    info = Hdf5Adapter().describe(UPath(path))

    assert info.episode_count == 1
    [episode] = list(Hdf5Adapter().episodes(UPath(path)))
    assert any(s.source_field == "x" for s in episode.streams)


def test_the_declared_rate_is_read_and_used_for_timestamps():
    """Verify the container's declared fps is read and used to synthesise timestamps."""

    info = Hdf5Adapter().describe(HDF5_FIXTURE)
    assert info.nominal_rate_hz == pytest.approx(20.0)

    [first, _] = list(Hdf5Adapter().episodes(HDF5_FIXTURE))
    assert first.streams[0].timestamps.to_list() == [i / 20.0 for i in range(5)]


def test_a_rate_attribute_on_the_root_is_found(tmp_path):
    """Verify a rate declared on the file root, not the container, is still read."""

    path = tmp_path / "root_rate.hdf5"
    with h5py.File(str(path), "w") as store:
        store.attrs["rate_hz"] = 10.0
        data = store.create_group("data")
        for name, length in (("a", 3), ("b", 4)):
            group = data.create_group(name)
            group.create_dataset(
                "actions", data=np.zeros((length, 2), dtype=np.float32)
            )

    info = Hdf5Adapter().describe(UPath(path))
    assert info.nominal_rate_hz == pytest.approx(10.0)


def test_an_undeclared_rate_falls_back_to_a_synthesised_timebase(tmp_path):
    """Verify no rate anywhere leaves nominal_rate_hz unset and timestamps at 1 Hz."""

    path = tmp_path / "no_rate.hdf5"
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        for name, length in (("a", 3), ("b", 4)):
            group = data.create_group(name)
            group.create_dataset(
                "actions", data=np.zeros((length, 2), dtype=np.float32)
            )

    adapter = Hdf5Adapter()
    upath = UPath(path)

    assert adapter.describe(upath).nominal_rate_hz is None
    episodes = list(adapter.episodes(upath))
    first_timestamps = (
        next(e for e in episodes if e.id == "data_a").streams[0].timestamps
    )
    assert first_timestamps.to_list() == [0.0, 1.0, 2.0]


def test_dataset_keys_resolve_through_the_dictionary():
    """Verify each fixture dataset key resolves to its expected taxonomy type."""

    [first, _] = list(Hdf5Adapter().episodes(HDF5_FIXTURE))
    by_field = {s.source_field: s for s in first.streams}

    assert by_field["actions"].taxonomy_type == "action.action_vector"
    assert by_field["rewards"].taxonomy_type == "reward.frame_reward"
    assert by_field["eef_pose"].taxonomy_type == "proprio.ee_pose"
    assert by_field["probe_raw"].taxonomy_type == "unmapped.probe_raw"
    assert [c.name for c in by_field["probe_raw"].channels] == [
        "probe_raw_0",
        "probe_raw_1",
    ]


def test_an_image_shaped_dataset_is_skipped(tmp_path):
    """Verify a rank-3+ dataset is skipped rather than read as a channel."""

    path = tmp_path / "with_image.hdf5"
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        for name, length in (("a", 3), ("b", 4)):
            group = data.create_group(name)
            group.create_dataset(
                "actions", data=np.zeros((length, 2), dtype=np.float32)
            )
            group.create_dataset(
                "camera", data=np.zeros((length, 4, 4, 3), dtype=np.uint8)
            )

    [episode] = [e for e in Hdf5Adapter().episodes(UPath(path)) if e.id == "data_a"]

    assert all(s.source_field != "camera" for s in episode.streams)


def test_a_group_with_no_channel_shaped_dataset_is_refused(tmp_path):
    """Verify an episode group holding only an image-shaped dataset is refused."""

    path = tmp_path / "no_channel.hdf5"
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        good = data.create_group("a")
        good.create_dataset("actions", data=np.zeros((3, 2), dtype=np.float32))
        bad = data.create_group("b")
        bad.create_dataset("camera", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))

    with pytest.raises(AdapterRefusal):
        list(Hdf5Adapter().episodes(UPath(path)))


def test_describe_does_not_read_dataset_values(monkeypatch):
    """Verify describe() never reads a dataset's values, only group structure."""

    monkeypatch.setattr(
        h5py.Dataset,
        "__getitem__",
        lambda self, key: pytest.fail("Dataset.__getitem__ was called by describe()"),
    )

    info = Hdf5Adapter().describe(HDF5_FIXTURE)
    assert info.episode_count == 2


def test_an_episodes_own_timestamps_become_its_timebase(tmp_path):
    """Verify a recorded time dataset times the episode and isn't graded as a channel.

    Recorded timestamps describe the capture, so the streams are marked regular
    when their gaps are; a declared rate alone never could.
    """

    path = tmp_path / "timed.hdf5"
    recorded = [0.0, 0.02, 0.0401, 0.06, 0.0799, 0.1]
    _write_demos(path, recorded, rate=5.0)  # the declared rate is deliberately wrong

    [first, _] = list(Hdf5Adapter().episodes(UPath(path)))

    assert [s.source_field for s in first.streams] == ["actions"]
    stream = first.streams[0]
    assert stream.timestamps.to_list() == pytest.approx(recorded)
    assert stream.is_regular is True


def test_millisecond_timestamps_are_read_in_seconds(tmp_path):
    """Verify a time dataset recorded in milliseconds is scaled to seconds."""

    path = tmp_path / "ms.hdf5"
    _write_demos(path, [0.0, 20.0, 40.0, 60.0, 80.0], key="time")

    [first, _] = list(Hdf5Adapter().episodes(UPath(path)))

    assert first.streams[0].timestamps.to_list() == pytest.approx(
        [0.0, 0.02, 0.04, 0.06, 0.08]
    )


def test_a_repeated_timestamp_is_kept_for_the_timing_metrics_to_find(tmp_path):
    """Verify a dropped physics step (a repeated timestamp) still reads as a clock.

    The repeat is a defect for timing metrics to grade, not a reason to throw
    the recorded clock away and fall back to a synthesised one.
    """

    path = tmp_path / "dropout.hdf5"
    recorded = [index * 0.02 for index in range(40)]
    recorded[10] = recorded[9]
    _write_demos(path, recorded)

    [first, _] = list(Hdf5Adapter().episodes(UPath(path)))

    stream = first.streams[0]
    assert stream.timestamps.to_list() == pytest.approx(recorded)
    assert stream.is_regular is True


def test_a_backwards_named_clock_is_preserved_instead_of_regenerated(tmp_path):
    """A broken recorded time axis must remain available for ordering diagnostics."""

    path = tmp_path / "not_a_clock.hdf5"
    _write_demos(path, [3.0, 1.0, 2.0, 0.0, 5.0, 4.0], key="t", rate=10.0)

    [first, _] = list(Hdf5Adapter().episodes(UPath(path)))

    assert sorted(s.source_field for s in first.streams) == ["actions"]
    assert first.streams[0].native_timestamps.to_list() == [
        3.0,
        1.0,
        2.0,
        0.0,
        5.0,
        4.0,
    ]
    assert first.streams[0].clock is Clock.UNKNOWN
    assert all(s.is_regular is False for s in first.streams)


def test_without_a_time_dataset_the_regular_grid_is_labelled_synthesised(tmp_path):
    """A regular generated grid does not certify measured acquisition timing."""

    path = tmp_path / "untimed.hdf5"
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        data.attrs["fps"] = 50.0
        for name in ("demo_0", "demo_1"):
            data.create_group(name).create_dataset(
                "actions", data=np.zeros((4, 2), dtype=np.float32)
            )

    [first, _] = list(Hdf5Adapter().episodes(UPath(path)))

    assert first.streams[0].timestamps.to_list() == pytest.approx(
        [0.0, 0.02, 0.04, 0.06]
    )
    assert first.streams[0].is_regular is True
    assert not first.streams[0].clock_info.certifies_acquisition
    assert first.streams[0].clock is Clock.RECONSTRUCTED
    assert first.streams[0].timestamp_dtype is TimestampDtype.FLOAT64


def test_a_float32_time_dataset_is_recorded_as_float32_on_an_unknown_clock(tmp_path):
    """The stored format is what bounds the stamps' rounding, whatever the cast."""

    path = tmp_path / "float32.hdf5"
    _write_demos(path, [0.0, 0.02, 0.0401, 0.06, 0.0799, 0.1], dtype=np.float32)

    [first, _] = list(Hdf5Adapter().episodes(UPath(path)))

    assert all(s.timestamp_dtype is TimestampDtype.FLOAT32 for s in first.streams)
    assert all(s.clock is Clock.UNKNOWN for s in first.streams)


def test_robomimic_keys_resolve_to_their_types(tmp_path):
    """Verify robomimic's `robot0_*` observations and `dones` get taxonomy types."""

    path = tmp_path / "robomimic_keys.hdf5"
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        for name in ("demo_0", "demo_1"):
            group = data.create_group(name)
            group.create_dataset("actions", data=np.zeros((5, 7), dtype=np.float32))
            group.create_dataset("dones", data=np.zeros(5, dtype=np.uint8))
            obs = group.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=np.zeros((5, 7)))
            obs.create_dataset("robot0_eef_pos", data=np.zeros((5, 3)))
            obs.create_dataset("robot0_eef_quat", data=np.zeros((5, 4)))
            obs.create_dataset("robot0_gripper_qpos", data=np.zeros((5, 2)))

    [first, _] = list(Hdf5Adapter().episodes(UPath(path)))
    types = {s.source_field.rsplit("/", 1)[-1]: s.taxonomy_type for s in first.streams}

    assert types == {
        "actions": "action.action_vector",
        "dones": "reward.discount_flag",
        "robot0_joint_pos": "proprio.joint_position",
        "robot0_eef_pos": "proprio.ee_pose",
        "robot0_eef_quat": "proprio.ee_pose",
        "robot0_gripper_qpos": "proprio.gripper_width",
    }


def test_smooth_joint_motion_raises_no_motion_finding(tmp_path):
    """Verify clean, smooth robot motion in a robomimic file raises no false alarm.

    Joint positions resolve now, so the motion family runs on them. Ordinary
    motion (here 0.5 Hz sines) must not come back as critical jerk, and the two
    values of a gripper command must not come back as a stuck channel.
    """

    from kalanos.api import grade  # local: keeps the adapter tests import-light

    path = tmp_path / "smooth.hdf5"
    t = np.arange(200) * 0.02
    with h5py.File(str(path), "w") as store:
        data = store.create_group("data")
        data.attrs["fps"] = 50.0
        for index in range(3):
            group = data.create_group(f"demo_{index}")
            joints = np.stack(
                [0.2 * np.sin(2 * np.pi * 0.5 * t + phase) for phase in range(6)],
                axis=1,
            )
            gripper = np.where(t > t[100], 1.0, -1.0)[:, None]
            group.create_dataset("actions", data=np.hstack([joints, gripper]))
            group.create_dataset("timestamps", data=t)
            group.create_group("obs").create_dataset("robot0_joint_pos", data=joints)

    report = grade(path)

    findings = [f for f in report.findings if f.severity is not None]
    assert not [f for f in findings if f.family == "motion"]
    assert not [f for f in findings if f.metric_id == "integrity.flatline_pct"]
    assert report.score.grade == "A"
