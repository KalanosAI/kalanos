"""`hdf5_tiny.hdf5`: a single-file HDF5 dataset with two sibling episode groups.

Two groups under `runs`, each holding per-step datasets directly plus a
`signals` subgroup — the shape the adapter's sibling-group heuristic finds structurally.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

from dataclasses import dataclass


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Hdf5Feature:
    """One subgroup dataset: its own key and its channel width."""

    key: str
    width: int


@dataclass(frozen=True)
class Hdf5TinySpec:
    """Everything `fixture_gen.writers._write_hdf5_tiny` needs to build the file.

    Attributes
    ----------
    filename : str
        The file's name under `tests/fixtures/`.
    container : str
        The group the episode groups sit under, and which carries the rate.
    episode_names : tuple[str, ...]
        One name per episode group, in order.
    episode_lengths : tuple[int, ...]
        Sample count per episode group, positionally matched to the names.
    fps : float
        The declared rate, read by `describe` rather than inferred.
    action_dim : int
        The `actions` dataset's channel width.
    subgroup : str
        The nested group each episode's extra datasets sit in.
    subgroup_features : tuple[Hdf5Feature, ...]
        Two datasets: the first resolves through `dictionary.yaml`, the
        second does not; so the fixture exercises both taxonomy paths.
    """

    filename: str
    container: str
    episode_names: tuple[str, ...]
    episode_lengths: tuple[int, ...]
    fps: float
    action_dim: int
    subgroup: str
    subgroup_features: tuple[Hdf5Feature, ...]


HDF5_TINY = Hdf5TinySpec(
    filename="hdf5_tiny.hdf5",
    container="runs",
    episode_names=("session_a", "session_b"),
    episode_lengths=(5, 4),
    fps=20.0,
    action_dim=4,
    subgroup="signals",
    subgroup_features=(
        Hdf5Feature(key="eef_pose", width=6),
        Hdf5Feature(key="probe_raw", width=2),
    ),
)
