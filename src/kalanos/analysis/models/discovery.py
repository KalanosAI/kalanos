"""What `discovery` emits: the root it walks, and a candidate or a skip per file.

Pre-inference, pre-loading — not to be confused with `domain.Source`,
which is what a file becomes once `loading` has parsed it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum

# External
from pydantic import BaseModel

# Internal
from kalanos.analysis.models.paths import AnyPath


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class SkipReason(str, Enum):
    """Why a file never reached grading.

    `UNREADABLE` and `EMPTY` come from `discovery`; `NO_ADAPTER` comes from
    selection, once discovery has already let the file through.
    `HIDDEN` comes from `discovery`: a dot-named entry below the root,
    reported once, its contents never walked.
    """

    # fmt: off
    NO_ADAPTER   = "no_adapter"
    UNREADABLE   = "unreadable"
    EMPTY        = "empty"
    HIDDEN       = "hidden"
    # fmt: on


class SourceCandidate(BaseModel):
    """A file or directory `discovery` decided is worth offering to an adapter.

    Attributes
    ----------
    path : UPath
        Location of the file or directory that passed classification.
    """

    path: AnyPath


class SkippedSource(BaseModel):
    """A file `discovery` declined to carry forward, with a machine-readable reason.

    Attributes
    ----------
    path : UPath
        Location of the file that was skipped.
    reason : SkipReason
        Why it was skipped.
    """

    path: AnyPath
    reason: SkipReason


class SourceInfo(BaseModel):
    """What one run graded: where the root came from, and how much it held.

    Attributes
    ----------
    uri : str
        The root as fsspec reads it, pinned to one commit when its host versions it,
        such as `hf://datasets/<org>/<name>@<sha>[/<path>]` or a local path.
    protocol : str
        The fsspec protocol the root is read through, such as `"file"` or `"hf"`.
    repo_id : str or None
        The hosted repository as `<org>/<name>`, or `None` for a root outside one.
    revision : str or None
        The commit the run read, or `None` when the root is not versioned.
    size_bytes : int
        Total size of the files under the root, as listed before any was read.
    file_count : int
        How many files sit under the root.
    """

    uri: str
    protocol: str
    repo_id: str | None = None
    revision: str | None = None
    size_bytes: int
    file_count: int

    @property
    def is_remote(self) -> bool:
        """Whether reading the root fetches its bytes over a network."""

        return self.protocol not in ("file", "local")


class SourceLimits(BaseModel):
    """The largest remote root a run will stream.

    `None` lifts a limit.
    Local roots are never limited: their bytes are already on disk.

    Attributes
    ----------
    max_bytes : int or None
        Refuse a remote root whose listed size is over this many bytes.
    max_files : int or None
        Refuse a remote root listing more files than this.
    """

    max_bytes: int | None = None
    max_files: int | None = None
