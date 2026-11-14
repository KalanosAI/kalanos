"""A block cache for remote files that fetches ahead of a sequential reader.

fsspec's caches fetch one block per blocking request, at most one ahead,
so a file read over a high-latency link waits a full round trip per block.
This cache keeps several requests in flight under a fixed count of held blocks.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor

# External
from fsspec.caching import BaseCache, Fetcher, register_cache


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# A video segment is read twice: once to list its packets, then again to decode
# from its start. Keeping blocks behind the furthest read lets the second pass
# find them cached, for segments up to about this many blocks long.
KEPT_BEHIND = 4


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class ReadaheadCache(BaseCache):
    """Cache whole blocks of a file, fetching blocks ahead of a sequential read.

    The cache keeps a window of `KEPT_BEHIND` blocks before the furthest block read
    and `readahead` after it, fetched in parallel, and drops every block outside it.
    A read starting in the window or just past it moves the window forward;
    any other read is a jump, which restarts the window there with no readahead.

    Held blocks are futures, so blocks in flight count against the cap of `capacity`.
    A dropped fetch that has started still finishes, so for about one round trip
    after a jump the file can hold `KEPT_BEHIND + 1 + 2 * readahead` blocks.

    Reads come from one thread at a time, as an fsspec file allows;
    the lock lets `close` and `held_blocks` run from another.

    Parameters
    ----------
    blocksize : int
        The size of each range request, in bytes.
    fetcher : Fetcher
        The file's `_fetch_range(start, end)`, making one blocking request.
    size : int
        The file's size, in bytes.
    readahead : int
        How many blocks to fetch ahead of a sequential read. `0` turns readahead off,
        leaving a cache of the furthest block read and the `KEPT_BEHIND` before it.

    Attributes
    ----------
    nblocks : int
        How many blocks the file spans.
    readahead : int
        As given.
    capacity : int
        The most blocks held at once, `KEPT_BEHIND + 1 + readahead`.
    """

    name = "kalanos-readahead"

    def __init__(
        self, blocksize: int, fetcher: Fetcher, size: int, readahead: int = 5
    ) -> None:
        """Set up an empty cache with its own pool of `readahead` fetching threads."""

        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.readahead = readahead
        self.capacity = KEPT_BEHIND + 1 + readahead
        self._blocks: dict[int, Future[bytes]] = {}
        # Blocks fetched ahead that no read has needed yet.
        self._ahead: set[int] = set()
        # Blocks fetched at least once, to count re-fetches after eviction.
        self._seen: set[int] = set()
        # The furthest block read since the last jump.
        self._furthest: int | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, readahead), thread_name_prefix="kalanos-readahead"
        )
        self._fetched = 0
        self._fetched_ahead = 0
        self._ahead_used = 0
        self._refetched = 0

    @property
    def held_blocks(self) -> int:
        """How many blocks are held, fetched or in flight.

        Returns
        -------
        int
            The count, never over `capacity` after a read no wider than it.
        """

        with self._lock:
            return len(self._blocks)

    def _fetch(self, start: int | None, stop: int | None) -> bytes:
        """Return bytes `start` to `stop`, fetching missing blocks and those ahead.

        Parameters
        ----------
        start : int or None
            The first byte, `None` for the start of the file.
        stop : int or None
            One past the last byte, `None` for the end of the file. Clamped to the size.

        Returns
        -------
        bytes
            The requested bytes, empty when the range is.

        Raises
        ------
        Exception
            Whatever the fetcher raised for a needed block, unchanged.
            That block is dropped, so the next read fetches it again.
        """

        # Clamp the range to the file, as fsspec's own caches do.
        if start is None:
            start = 0
        end = self.size if stop is None else min(stop, self.size)
        if start >= self.size or start >= end:
            return b""

        # Work out the blocks this read needs and the window to keep after it.
        first = start // self.blocksize
        last = (end - 1) // self.blocksize
        needed = range(first, last + 1)
        furthest = self._furthest
        # A rewind to a segment's start and a seek into the blocks ahead both land
        # inside the window, so they keep it and its readahead.
        if (
            furthest is not None
            and furthest - KEPT_BEHIND <= first <= furthest + self.readahead + 1
        ):
            furthest = max(furthest, last)
            stop_ahead = min(furthest + 1 + self.readahead, self.nblocks)
            ahead = range(furthest + 1, stop_ahead)
        else:
            # A jump reveals nothing about the next read, so it fetches nothing ahead.
            furthest = last
            ahead = range(0)
        wanted = set(needed) | set(ahead)
        window = range(furthest - KEPT_BEHIND, furthest + 1 + len(ahead))

        # Start missing fetches and drop blocks outside the window,
        # under the lock so `close` never sees the blocks half updated.
        with self._lock:
            if all(block in self._blocks for block in needed):
                self.hit_count += 1
            else:
                self.miss_count += 1
            for block in sorted(wanted):
                if block in self._blocks:
                    continue
                self._submit(block, ahead=block in ahead)
            for block in needed:
                if block in self._ahead:
                    self._ahead.discard(block)
                    self._ahead_used += 1
            self._evict(wanted, window)
            futures = [(block, self._blocks[block]) for block in needed]

        # Wait for the needed blocks outside the lock, so blocks ahead keep arriving.
        pieces = []
        for block, future in futures:
            try:
                data = future.result()
            except BaseException:
                with self._lock:
                    # `close` may have dropped the block already.
                    if self._blocks.get(block) is future:
                        del self._blocks[block]
                        # So the retry doesn't count as a re-fetch after eviction.
                        self._seen.discard(block)
                raise
            offset = block * self.blocksize
            pieces.append(data[max(start - offset, 0) : end - offset])
        # Set only once every needed block arrived, so a failed read keeps the window.
        self._furthest = furthest
        return pieces[0] if len(pieces) == 1 else b"".join(pieces)

    def _submit(self, block: int, *, ahead: bool) -> None:
        """Start fetching `block` and hold its future.

        Called with the lock held.

        Parameters
        ----------
        block : int
            The block to fetch.
        ahead : bool
            Whether no read needs it yet.
        """

        start = block * self.blocksize
        end = min(start + self.blocksize, self.size)
        self._blocks[block] = self._pool.submit(self.fetcher, start, end)
        self.total_requested_bytes += end - start
        self._fetched += 1
        if block in self._seen:
            self._refetched += 1
        self._seen.add(block)
        if ahead:
            self._fetched_ahead += 1
            self._ahead.add(block)

    def _evict(self, wanted: set[int], window: range) -> None:
        """Drop every block outside `window` that the current read does not need.

        Called with the lock held. A dropped block still in flight is cancelled
        when its fetch has not started.

        Parameters
        ----------
        wanted : set of int
            The blocks the current read needs or fetches ahead, never dropped.
        window : range
            The blocks kept around the furthest block read.
        """

        for block in list(self._blocks):
            if block in wanted or block in window:
                continue
            self._blocks.pop(block).cancel()
            self._ahead.discard(block)

    def close(self) -> None:
        """Cancel pending fetches, drop every block and log the counters.

        Safe to call twice.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            for future in self._blocks.values():
                future.cancel()
            self._blocks.clear()
            self._ahead.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
        logger.debug(
            "readahead over %d bytes in %d-byte blocks: %d fetched, "
            "%d ahead of which %d used, %d re-fetched after eviction",
            self.size,
            self.blocksize,
            self._fetched,
            self._fetched_ahead,
            self._ahead_used,
            self._refetched,
        )


# Re-registering replaces the class, so importing this module twice is harmless.
register_cache(ReadaheadCache, clobber=True)


__all__ = ["KEPT_BEHIND", "ReadaheadCache"]
