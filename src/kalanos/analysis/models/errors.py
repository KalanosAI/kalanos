"""The errors the public API raises on purpose, under one base a caller can catch."""

# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class KalanosError(Exception):
    """Base for every error `kalanos.grade` raises about its input rather than a bug."""


class NothingToGrade(KalanosError):
    """The path held nothing at all: no file analysed, skipped or refused."""


class SourceUnavailable(KalanosError):
    """The path could not be opened.

    It is missing, private, or needs an extra that is not installed.
    """


class SourceTooLarge(KalanosError):
    """A remote root is over the size or file-count limit; nothing was read."""
