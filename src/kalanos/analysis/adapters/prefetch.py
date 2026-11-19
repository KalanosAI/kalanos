"""Whole-file prefetch of the remote video files a dataset reads next.

A remote video file opened cold waits on round trips until its readahead fills,
and a dataset with one file per episode pays that at every file boundary.
An adapter that knows which files its next episodes open hands them over in order,
and they are downloaded whole in the background, under a cap on the bytes held.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import threading
from collections.abc import Generator, Iterable, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager

# External
from upath import UPath


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# How many files download at once.
PREFETCH_WORKERS = 8

# Marks a file that is never fetched: over the cap, or of unknown size.
_SKIPPED = object()

# The prefetcher `prefetched` reads from, set by `prefetching`.
# Not a ContextVar: an adapter's generator runs in its caller's context,
# so a ContextVar set inside it would leak into the caller between `next()` calls.
_ACTIVE: "VideoPrefetcher | None" = None
_ACTIVE_LOCK = threading.Lock()


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class VideoPrefetcher:
    """Download video files whole, in the order given, under a cap on bytes held.

    A feeder thread walks the files in order. A file larger than the cap,
    or whose size cannot be read, is skipped and left to stream.
    Any other file waits until it fits under the cap, then downloads;
    files behind a waiting file wait too, so downloads never overtake the reader.
    A file's bytes count against the cap from the moment its download is submitted
    until `release`, or until the download ends when it is released mid-way,
    since a download that has started cannot be cancelled.

    Parameters
    ----------
    episodes : Sequence of Sequence of UPath
        Each episode's video files, episodes in the order they are read.
    cap_bytes : int
        The most bytes reserved at once, downloading or downloaded.

    Attributes
    ----------
    cap_bytes : int
        As given.
    """

    def __init__(self, episodes: Sequence[Sequence[UPath]], cap_bytes: int) -> None:
        """Start the feeder thread and the download pool."""

        self.cap_bytes = cap_bytes
        self._order = [path for episode in episodes for path in episode]
        # Keyed on `str(path)`, as the pool of open video files is.
        self._entries: dict[str, object] = {}
        self._reserved: dict[str, int] = {}
        # Released before the feeder reached them, so never fetched.
        self._released: set[str] = set()
        self._held = 0
        self._closed = False
        self._cond = threading.Condition()
        self._pool = ThreadPoolExecutor(
            max_workers=PREFETCH_WORKERS, thread_name_prefix="kalanos-prefetch"
        )
        self._fetched = 0
        self._used = 0
        self._streamed = 0
        self._skipped = 0
        self._failed = 0
        self._feeder = threading.Thread(
            target=self._feed, name="kalanos-prefetch-feeder", daemon=True
        )
        self._feeder.start()

    @property
    def held_bytes(self) -> int:
        """How many bytes are reserved, downloading or downloaded and not released.

        Returns
        -------
        int
            The count, never over `cap_bytes`.
        """

        with self._cond:
            return self._held

    def _feed(self) -> None:
        """Submit each file's download in order, waiting for room under the cap."""

        for path in self._order:
            key = str(path)
            with self._cond:
                if self._closed:
                    return
                if key in self._released or key in self._entries:
                    continue
            size = self._size(path)
            with self._cond:
                if size is None or size > self.cap_bytes:
                    self._entries[key] = _SKIPPED
                    self._skipped += 1
                    continue
                while (
                    not self._closed
                    and key not in self._released
                    and self._held + size > self.cap_bytes
                ):
                    self._cond.wait()
                if self._closed:
                    return
                if key in self._released:
                    continue
                self._held += size
                self._reserved[key] = size
                self._entries[key] = self._pool.submit(self._download, path)
                self._fetched += 1

    def _size(self, path: UPath) -> int | None:
        """Return `path`'s size in bytes, `None` when it cannot be read.

        Parameters
        ----------
        path : UPath
            A remote video file.

        Returns
        -------
        int or None
            The size its filesystem reports.
        """

        try:
            size = path.fs.size(path.path)
        except Exception as exc:
            logger.debug("prefetch: no size for %s: %s", path, exc)
            return None
        return None if size is None else int(size)

    def _download(self, path: UPath) -> bytes:
        """Read `path` whole, counting a failure before passing it on.

        Parameters
        ----------
        path : UPath
            The file to read.

        Returns
        -------
        bytes
            Its contents.
        """

        try:
            return path.read_bytes()
        except BaseException:
            with self._cond:
                self._failed += 1
            raise

    def take(self, path: UPath) -> bytes | None:
        """Return `path`'s downloaded bytes, waiting when its download is under way.

        The bytes stay counted against the cap until `release`.

        Parameters
        ----------
        path : UPath
            A video file about to be opened.

        Returns
        -------
        bytes or None
            The file's contents, or `None` when the caller should stream it:
            skipped, not yet submitted, released, failed, or this prefetcher closed.
        """

        key = str(path)
        with self._cond:
            entry = None if self._closed else self._entries.get(key)
        if not isinstance(entry, Future):
            return self._stream()
        try:
            data = entry.result()
        except CancelledError:
            return self._stream()
        except Exception as exc:
            logger.debug("prefetch: download of %s failed: %s", path, exc)
            return self._stream()
        with self._cond:
            self._used += 1
        return data

    def _stream(self) -> None:
        """Count a file the caller streams instead.

        Returns
        -------
        None
            Always, for `take` to return.
        """

        with self._cond:
            self._streamed += 1
        return None

    def release(self, paths: Iterable[UPath]) -> None:
        """Drop `paths`' downloads and reservations, making room for the next files.

        A pending download is cancelled, one under way keeps its room until it ends,
        and a file the feeder has not reached is never fetched.
        Paths this prefetcher was not given are ignored.

        Parameters
        ----------
        paths : Iterable of UPath
            The files of an episode that has been read.
        """

        with self._cond:
            for path in paths:
                key = str(path)
                self._released.add(key)
                entry = self._entries.pop(key, None)
                size = self._reserved.pop(key, 0)
                if isinstance(entry, Future) and not entry.cancel():
                    # A done future runs the callback at once, under this same RLock.
                    entry.add_done_callback(lambda _f, size=size: self._unreserve(size))
                else:
                    self._held -= size
            self._cond.notify_all()

    def _unreserve(self, size: int) -> None:
        """Give back the room of a released download once it has ended.

        Parameters
        ----------
        size : int
            The bytes it reserved.
        """

        with self._cond:
            # `close` has already zeroed the count.
            if not self._closed:
                self._held -= size
                self._cond.notify_all()

    def close(self) -> None:
        """Stop the feeder, cancel pending downloads, drop every file, log the counters.

        Safe to call twice.
        """

        with self._cond:
            if self._closed:
                return
            self._closed = True
            for entry in self._entries.values():
                if isinstance(entry, Future):
                    entry.cancel()
            self._entries.clear()
            self._reserved.clear()
            self._held = 0
            self._cond.notify_all()
        self._pool.shutdown(wait=False, cancel_futures=True)
        logger.debug(
            "prefetch over %d files under a %d-byte cap: %d fetched, %d used, "
            "%d streamed, %d skipped over the cap, %d failed",
            len(self._order),
            self.cap_bytes,
            self._fetched,
            self._used,
            self._streamed,
            self._skipped,
            self._failed,
        )


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@contextmanager
def prefetching(
    episodes: Sequence[Sequence[UPath]], cap_bytes: int
) -> Generator[VideoPrefetcher | None, None, None]:
    """Prefetch `episodes`' video files for the body, for `prefetched` to hand out.

    Parameters
    ----------
    episodes : Sequence of Sequence of UPath
        Each episode's video files, episodes in the order they are read.
    cap_bytes : int
        The most bytes held at once; `0` or less prefetches nothing.

    Yields
    ------
    VideoPrefetcher or None
        The active prefetcher, `None` when there is nothing to prefetch.
        It is closed on exit, even on error or when a generator holding it is closed.
    """

    global _ACTIVE

    if cap_bytes <= 0 or not any(episodes):
        yield None
        return
    prefetcher = VideoPrefetcher(episodes, cap_bytes)
    with _ACTIVE_LOCK:
        _ACTIVE = prefetcher
    try:
        yield prefetcher
    finally:
        with _ACTIVE_LOCK:
            # A later reader may have replaced it while this one sat unfinished.
            if _ACTIVE is prefetcher:
                _ACTIVE = None
        prefetcher.close()


def prefetched(path: UPath) -> bytes | None:
    """Return `path`'s prefetched bytes, `None` when it should be streamed.

    Parameters
    ----------
    path : UPath
        A remote video file about to be opened.

    Returns
    -------
    bytes or None
        Its contents from the active prefetcher, `None` without one.
    """

    with _ACTIVE_LOCK:
        active = _ACTIVE
    return None if active is None else active.take(path)


__all__ = ["PREFETCH_WORKERS", "VideoPrefetcher", "prefetched", "prefetching"]
