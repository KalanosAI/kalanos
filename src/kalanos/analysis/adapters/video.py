"""Lazy video payloads and the optional decoder behind them.

Lives outside any one adapter because more than one format (LeRobot, MCAP, RLDS)
carries frames as an mp4 alongside timestamped metadata, and each wants the same lazy,
optional decoder.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib
import logging
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

# External
from upath import UPath


if TYPE_CHECKING:
    import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

logger = logging.getLogger(__name__)


class DecoderUnavailable(RuntimeError):
    """Raised when a video payload is fetched without the decoder installed."""


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _load_av() -> ModuleType | None:
    """Import PyAV, or return `None` when the video extra is not installed."""

    try:
        return importlib.import_module("av")
    except ImportError:
        return None


def decoder_available() -> bool:
    """Whether a video payload can be decoded in this process.

    Returns
    -------
    bool
        Whether PyAV imports successfully.
    """

    return _load_av() is not None


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class VideoPayload:
    """A video stream's frames, decoded only when `fetch` is called.

    Attributes
    ----------
    path : UPath
        The video file this payload's frames come from.
    frame_count : int
        How many frames the stream declares, without opening `path`.
    start_s : float
        Where in `path` this stream's frames begin.
    end_s : float
        Where in `path` this stream's frames end.
    """

    path: UPath
    frame_count: int
    start_s: float
    end_s: float

    def __len__(self) -> int:
        """Return the declared frame count, reading nothing.

        Returns
        -------
        int
            `frame_count`.
        """

        return self.frame_count

    def fetch(self) -> "list[np.ndarray]":
        """Decode and return every frame between `start_s` and `end_s`.

        Returns
        -------
        list of numpy.ndarray
            One `(height, width, 3)` RGB array per frame, in order.

        Raises
        ------
        DecoderUnavailable
            If PyAV is not installed in this process.
        """

        av = _load_av()
        if av is None:
            raise DecoderUnavailable(
                f"{self.path}: decoding a video payload needs PyAV; add kalanos[video]"
            )

        container = av.open(str(self.path))
        try:
            stream = container.streams.video[0]
            container.seek(int(self.start_s / stream.time_base), stream=stream)
            # end_s is the exclusive upper bound: it is the next segment's own start_s,
            # so a frame landing exactly on it belongs there, not here.
            # The count is also capped at frame_count, so a float-precision mismatch
            # can't pull in an extra frame.
            frames = []
            for frame in container.decode(stream):
                if len(frames) >= self.frame_count:
                    break
                if frame.time is not None and frame.time >= self.end_s:
                    break
                if frame.time is not None and frame.time < self.start_s:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
        finally:
            container.close()

        if len(frames) != self.frame_count:
            logger.warning(
                "%s: declared %d frame(s) but decoded %d between %.6fs and %.6fs",
                self.path,
                self.frame_count,
                len(frames),
                self.start_s,
                self.end_s,
            )
        return frames
