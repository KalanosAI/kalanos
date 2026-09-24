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
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, HDF5_FIXTURE


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
