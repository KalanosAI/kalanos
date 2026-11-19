"""Verifies the whole-file video prefetcher: its order, its cap and its threads."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

# External
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters import prefetch
from kalanos.analysis.adapters.prefetch import VideoPrefetcher, prefetching

from helpers import LEROBOT_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

LEROBOT_VIDEO = (
    LEROBOT_FIXTURE / "videos" / "observation.images.up" / "chunk-000" / "file-000.mp4"
)
# Each copy ends in its own index byte, so a mixed-up file is told apart.
SIZE = LEROBOT_VIDEO.stat().st_size + 1


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


def _wait_for(condition: Callable[[], bool]) -> bool:
    """Poll `condition` for up to 5 s.

    Parameters
    ----------
    condition : Callable
        Returns whether the wait is over.

    Returns
    -------
    bool
        Whether `condition` held before the deadline.
    """

    deadline = time.monotonic() + 5
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def _submitted(prefetcher: VideoPrefetcher, path: UPath) -> bool:
    """Return whether `path`'s download has been submitted and not released.

    Parameters
    ----------
    prefetcher : VideoPrefetcher
        The prefetcher to look into.
    path : UPath
        The file.

    Returns
    -------
    bool
        Whether a download future is held for it.
    """

    return isinstance(prefetcher._entries.get(str(path)), Future)


def _prefetch_threads() -> list[threading.Thread]:
    """Return the prefetch threads still alive after up to 2 s of waiting."""

    deadline = time.monotonic() + 2
    while True:
        alive = [
            t for t in threading.enumerate() if t.name.startswith("kalanos-prefetch")
        ]
        if not alive or time.monotonic() > deadline:
            return alive
        time.sleep(0.01)


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture
def remote_files(request):
    """Return a factory copying the fixture mp4 to `count` files in memory.

    Each copy ends in its own index byte. Every file is removed afterwards.
    """

    made: list[UPath] = []

    def make(count: int) -> list[UPath]:
        paths = [
            UPath(f"memory://prefetch/{request.node.name}/episode-{k:03d}.mp4")
            for k in range(count)
        ]
        for index, path in enumerate(paths):
            path.write_bytes(LEROBOT_VIDEO.read_bytes() + bytes([index]))
        made.extend(paths)
        return paths

    yield make
    for path in made:
        path.unlink(missing_ok=True)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_held_bytes_never_exceed_the_cap(monkeypatch, remote_files):
    """Verify the bytes held stay under the cap while every file is fetched in turn.

    Each download blocks until the test lets it finish,
    so files pile up against the cap instead of draining at once.
    """

    paths = remote_files(10)
    gates = {str(path): threading.Event() for path in paths}
    held: list[int] = []
    original = VideoPrefetcher._download

    def gated(self, path):
        held.append(self.held_bytes)
        gates[str(path)].wait(5)
        return original(self, path)

    monkeypatch.setattr(VideoPrefetcher, "_download", gated)
    cap = 2 * SIZE

    with prefetching([[path] for path in paths], cap) as prefetcher:
        assert prefetcher is not None
        for path in paths:
            assert _wait_for(lambda p=path: _submitted(prefetcher, p))
            gates[str(path)].set()
            assert prefetcher.take(path) == path.read_bytes()
            prefetcher.release([path])

    assert len(held) == 10
    assert max(held) <= cap


def test_files_ahead_are_fetched_while_an_episode_is_open(remote_files):
    """Verify the files of later episodes download while the first one is read."""

    paths = remote_files(3)

    with prefetching([[path] for path in paths], 3 * SIZE) as prefetcher:
        assert prefetcher is not None
        assert prefetcher.take(paths[0]) is not None
        for path in paths[1:]:
            assert _wait_for(lambda p=path: _submitted(prefetcher, p))
            future = prefetcher._entries[str(path)]
            assert isinstance(future, Future)
            assert future.result(timeout=5) == path.read_bytes()
        assert prefetcher._used == 1


def test_release_frees_room_for_the_next_file(remote_files):
    """Verify a file waiting for room is submitted once the one before is released."""

    first, second = remote_files(2)

    with prefetching([[first], [second]], SIZE) as prefetcher:
        assert prefetcher is not None
        assert prefetcher.take(first) is not None
        time.sleep(0.05)
        assert not _submitted(prefetcher, second)
        assert prefetcher.held_bytes == SIZE

        prefetcher.release([first])

        assert _wait_for(lambda: _submitted(prefetcher, second))
        assert prefetcher.take(second) is not None


def test_a_file_over_the_cap_is_skipped(monkeypatch, remote_files):
    """Verify a file larger than the cap is never downloaded and is left to stream."""

    (path,) = remote_files(1)
    reads = []
    monkeypatch.setattr(VideoPrefetcher, "_download", lambda self, p: reads.append(p))

    with prefetching([[path]], SIZE - 1) as prefetcher:
        assert prefetcher is not None
        prefetcher._feeder.join(5)
        assert prefetcher.take(path) is None
        assert prefetcher.held_bytes == 0

    assert reads == []


def test_a_failed_download_falls_back(monkeypatch, remote_files):
    """Verify a failed download streams, and the file after it still downloads."""

    broken, healthy = remote_files(2)
    original = type(broken).read_bytes

    def failing(self):
        if str(self) == str(broken):
            raise OSError("connection reset")
        return original(self)

    monkeypatch.setattr(type(broken), "read_bytes", failing)

    with prefetching([[broken], [healthy]], 2 * SIZE) as prefetcher:
        assert prefetcher is not None
        assert _wait_for(lambda: _submitted(prefetcher, broken))
        assert prefetcher.take(broken) is None
        assert _wait_for(lambda: _submitted(prefetcher, healthy))
        assert prefetcher.take(healthy) == healthy.read_bytes()
        assert prefetcher._failed == 1


def test_prefetching_with_a_zero_cap_starts_nothing(remote_files):
    """Verify a zero cap yields no prefetcher and starts no thread."""

    (path,) = remote_files(1)

    with prefetching([[path]], 0) as prefetcher:
        assert prefetcher is None
        assert prefetch.prefetched(path) is None
        assert _prefetch_threads() == []


def test_closing_stops_every_thread(remote_files):
    """Verify no feeder or download thread outlives the prefetcher."""

    paths = remote_files(3)

    with prefetching([[path] for path in paths], SIZE) as prefetcher:
        assert prefetcher is not None
        assert prefetch.prefetched(paths[0]) is not None

    assert prefetch.prefetched(paths[0]) is None
    assert _prefetch_threads() == []


def test_a_download_released_mid_way_keeps_its_room_until_it_ends(
    monkeypatch, remote_files
):
    """Verify releasing a running download frees its room only once it finishes.

    Its bytes are still arriving, so the next file must not take that room yet.
    """

    first, second = remote_files(2)
    started = threading.Event()
    gate = threading.Event()
    original = VideoPrefetcher._download

    def gated(self, path):
        if str(path) == str(first):
            started.set()
            gate.wait(5)
        return original(self, path)

    monkeypatch.setattr(VideoPrefetcher, "_download", gated)

    with prefetching([[first], [second]], SIZE) as prefetcher:
        assert prefetcher is not None
        assert started.wait(5)

        prefetcher.release([first])
        time.sleep(0.05)

        assert prefetcher.held_bytes == SIZE
        assert not _submitted(prefetcher, second)
        gate.set()
        assert _wait_for(lambda: _submitted(prefetcher, second))
        assert prefetcher.take(second) is not None
