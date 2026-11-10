"""Verifies the readahead block cache: its bytes, its fetches and its cap."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import random
import threading
import time

# External
import pytest

# Internal
from kalanos.analysis.adapters.readahead import KEPT_BEHIND, ReadaheadCache


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

BLOCK = 16


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class FakeFetcher:
    """Serve ranges of `data`, recording each call, optionally failing or blocking.

    Attributes
    ----------
    data : bytes
        The file's contents.
    calls : list of tuple of int
        Each `(start, end)` asked for, in the order the calls started.
    fail_once : set of int
        Block starts that raise `OSError` the first time they are fetched.
    gate : threading.Event or None
        When set to an event, every fetch waits for it first.
    """

    def __init__(self, data: bytes) -> None:
        """Serve `data`, failing nothing and blocking on nothing."""

        self.data = data
        self.calls: list[tuple[int, int]] = []
        self.fail_once: set[int] = set()
        self.gate: threading.Event | None = None
        self._lock = threading.Lock()

    def __call__(self, start: int, end: int) -> bytes:
        """Return `data[start:end]`, after recording the call.

        Parameters
        ----------
        start, end : int
            The byte range.

        Returns
        -------
        bytes
            The range.

        Raises
        ------
        OSError
            The first time `start` is in `fail_once`.
        """

        with self._lock:
            self.calls.append((start, end))
            failing = start in self.fail_once
            self.fail_once.discard(start)
        if self.gate is not None:
            self.gate.wait()
        if failing:
            raise OSError(f"fetch of {start} failed")
        return self.data[start:end]

    def blocks(self) -> list[int]:
        """Return the block numbers fetched, in call order.

        Returns
        -------
        list of int
            One entry per call.
        """

        with self._lock:
            return [start // BLOCK for start, _ in self.calls]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _cache(nblocks: float, readahead: int = 5) -> tuple[ReadaheadCache, FakeFetcher]:
    """Return a cache over `nblocks` blocks of seeded random bytes, and its fetcher."""

    size = int(nblocks * BLOCK)
    data = random.Random(size).randbytes(size)
    fetcher = FakeFetcher(data)
    return ReadaheadCache(BLOCK, fetcher, size, readahead=readahead), fetcher


def _read_block(cache: ReadaheadCache, block: int) -> bytes:
    """Read one whole block through the cache."""

    return cache._fetch(block * BLOCK, (block + 1) * BLOCK)


def _wait_for_fetches(fetcher: FakeFetcher, count: int) -> None:
    """Wait up to 2 s for `count` fetch calls to have started."""

    deadline = time.monotonic() + 2
    while len(fetcher.calls) < count and time.monotonic() < deadline:
        time.sleep(0.01)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_reads_return_the_requested_bytes():
    """Verify random reads, across blocks and past the end, return the source slice."""

    cache, fetcher = _cache(10.5)
    rng = random.Random(0)
    try:
        for _ in range(500):
            start = rng.randrange(0, cache.size + BLOCK)
            end = start + rng.randrange(0, 4 * BLOCK)
            assert cache._fetch(start, end) == fetcher.data[start:end]
    finally:
        cache.close()


def test_open_ended_reads_cover_the_file():
    """Verify `None` bounds read from the start and to the end."""

    cache, fetcher = _cache(3.5)
    try:
        assert cache._fetch(None, None) == fetcher.data
        assert cache._fetch(BLOCK + 3, None) == fetcher.data[BLOCK + 3 :]
    finally:
        cache.close()


def test_a_sequential_read_fetches_the_next_blocks_ahead():
    """Verify reading block 0 then block 1 fetches blocks 2 to 6, each once."""

    cache, fetcher = _cache(20)
    try:
        _read_block(cache, 0)
        _read_block(cache, 1)
        _wait_for_fetches(fetcher, 7)
        assert sorted(fetcher.blocks()) == [0, 1, 2, 3, 4, 5, 6]
    finally:
        cache.close()


def test_a_jump_fetches_only_the_blocks_it_needs():
    """Verify a read at block 7 after one at block 0 fetches block 7 alone."""

    cache, fetcher = _cache(20)
    try:
        _read_block(cache, 0)
        _read_block(cache, 7)
        time.sleep(0.05)
        assert fetcher.blocks() == [0, 7]
    finally:
        cache.close()


@pytest.mark.parametrize("readahead", [0, 1, 5])
def test_held_blocks_never_exceed_the_cap(readahead):
    """Verify a long sequential read with periodic rewinds holds at most the cap.

    Every 5 blocks the reader goes back 3, as the decode pass rewinds
    to the start of the segment it just listed.
    """

    cache, fetcher = _cache(2000, readahead=readahead)
    cap = KEPT_BEHIND + 1 + readahead
    try:
        for block in range(2000):
            assert _read_block(cache, block) == fetcher.data[block * BLOCK :][:BLOCK]
            assert cache.held_blocks <= cap
            if block % 5 == 4:
                for back in range(block - 2, block + 1):
                    _read_block(cache, back)
                    assert cache.held_blocks <= cap
    finally:
        cache.close()


def test_a_rewind_inside_the_kept_blocks_fetches_nothing():
    """Verify reading blocks 0 to 4 and then block 2 makes no new fetch."""

    cache, fetcher = _cache(20)
    try:
        for block in range(5):
            _read_block(cache, block)
        _wait_for_fetches(fetcher, 10)
        before = len(fetcher.calls)
        _read_block(cache, 2)
        time.sleep(0.05)
        assert len(fetcher.calls) == before
    finally:
        cache.close()


def test_a_rewind_inside_the_window_keeps_the_blocks_ahead():
    """Verify going back 4 blocks and reading forward again fetches nothing twice.

    The decode pass rewinds to the start of a segment the listing pass read through,
    and the blocks fetched ahead of the listing pass are still needed after it.
    """

    cache, fetcher = _cache(40)
    try:
        for block in range(10):
            _read_block(cache, block)
        for block in range(5, 16):
            _read_block(cache, block)
        _wait_for_fetches(fetcher, 21)
        time.sleep(0.05)
        assert sorted(fetcher.blocks()) == list(range(21))
    finally:
        cache.close()


def test_a_skip_into_the_blocks_ahead_keeps_them():
    """Verify a forward seek into the blocks fetched ahead fetches nothing twice."""

    cache, fetcher = _cache(40)
    try:
        for block in (0, 1, 4, 5):
            _read_block(cache, block)
        _wait_for_fetches(fetcher, 11)
        time.sleep(0.05)
        assert sorted(fetcher.blocks()) == list(range(11))
    finally:
        cache.close()


def test_a_failed_block_is_fetched_again():
    """Verify a failed block raises once, then is fetched again on the next read."""

    cache, fetcher = _cache(20, readahead=0)
    fetcher.fail_once.add(3 * BLOCK)
    try:
        with pytest.raises(OSError, match="fetch of 48 failed"):
            _read_block(cache, 3)
        assert _read_block(cache, 3) == fetcher.data[3 * BLOCK : 4 * BLOCK]
        assert fetcher.blocks() == [3, 3]
    finally:
        cache.close()


def test_close_stops_pending_fetches():
    """Verify `close` returns while fetches are blocked, and no thread outlives them."""

    cache, fetcher = _cache(20)
    _read_block(cache, 0)
    fetcher.gate = threading.Event()
    # Block 0 is held, so this returns at once, leaving blocks 1 to 5 stuck at the gate.
    _read_block(cache, 0)
    _wait_for_fetches(fetcher, 2)
    cache.close()
    fetcher.gate.set()

    deadline = time.monotonic() + 2
    alive = True
    while alive and time.monotonic() < deadline:
        alive = any(
            t.name.startswith("kalanos-readahead") for t in threading.enumerate()
        )
        time.sleep(0.01)
    assert not alive


def test_close_is_idempotent():
    """Verify a second `close` does nothing."""

    cache, _ = _cache(4)
    cache.close()
    cache.close()
    assert cache.held_blocks == 0


def test_readahead_zero_fetches_nothing_ahead():
    """Verify a sequential read with readahead 0 fetches only the blocks it reads."""

    cache, fetcher = _cache(20, readahead=0)
    try:
        for block in range(6):
            _read_block(cache, block)
        time.sleep(0.05)
        assert fetcher.blocks() == [0, 1, 2, 3, 4, 5]
    finally:
        cache.close()
