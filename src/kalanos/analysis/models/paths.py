"""A path annotation that survives a non-local filesystem intact.

`AnyPath` accepts a string, a `pathlib.Path` or a `UPath` on a Pydantic field
and always stores a `UPath`, serializing through upath so a remote path's protocol
and storage options survive a report round-trip.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import PurePath
from typing import Annotated

# External
from pydantic import BeforeValidator
from upath import UPath


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


def _as_upath(value: object) -> object:
    """Convert a `pathlib.Path` to a `UPath`, leaving everything else to upath's schema.

    Only a local `UPath` is a `pathlib.Path` and reaches this branch; a non-local one
    (`memory://`, `s3://`, ...) is not a `PurePath` instance and falls straight through
    to upath's own schema below, which validates it as-is.
    Rebuilding a local `UPath` here is idempotent and preserves `storage_options`,
    so it needs no separate case from a bare `pathlib.Path`.
    A `str` or upath's serialized dict form is left untouched for the same reason:
    upath's own Pydantic schema already validates both.

    Parameters
    ----------
    value : object
        The raw field input.

    Returns
    -------
    object
        `value` converted to a `UPath` when it was a `pathlib.Path`,
        otherwise unchanged.
    """

    if isinstance(value, PurePath):
        return UPath(value)
    return value


AnyPath = Annotated[UPath, BeforeValidator(_as_upath)]
