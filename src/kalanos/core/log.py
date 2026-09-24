"""The log level vocabulary and the single handler the CLI installs.

No library module under `kalanos` may configure logging itself — attaching a
handler or calling `logging.basicConfig` is `configure_logging`'s job alone.
Every other module just calls `logging.getLogger(__name__)`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import sys
from enum import Enum
from typing import TextIO


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Verbosity(str, Enum):
    """One of logging's own level names, exposed as a CLI-friendly enum.

    Values are lowercased for a nicer `--verbosity` argument,
    but otherwise name the same levels the standard library does.

    Attributes
    ----------
    DEBUG : str
        Every inference guess, and everything below.
    INFO : str
        Stage boundaries and per-file outcomes, and everything below.
    WARNING : str
        A skip, an unresolved file, or an unmapped stream, and everything below.
    ERROR : str
        A run-ending failure.
    CRITICAL : str
        Nothing this distribution logs at today; silences everything else.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

LOGGER_NAME = "kalanos"

_HANDLER_NAME = "kalanos"

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Second resolution: a CLI run is measured in seconds, not milliseconds,
# and the report already carries the precise `duration_s`.
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def configure_logging(verbosity: Verbosity, *, stream: TextIO | None = None) -> None:
    """Attach the one handler the CLI installs, and set its level.

    The only function in the distribution allowed to attach a handler.

    Parameters
    ----------
    verbosity : Verbosity
        The level to report at for this run.
    stream : TextIO or None
        Where to write records. `None` resolves `sys.stderr` inside the body,
        at call time — a caller may replace `sys.stderr` after this module is imported,
        and a default bound at import time would miss that.
    """

    logger = logging.getLogger(LOGGER_NAME)

    # Drop any handler a previous call installed,
    # so calling this twice in one process doesn't double every record
    # A handler an embedder attached itself carries no such name and survives.
    logger.handlers = [
        handler for handler in logger.handlers if handler.name != _HANDLER_NAME
    ]

    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.name = _HANDLER_NAME
    logger.addHandler(handler)

    logger.setLevel(verbosity.value.upper())
