"""Per-run mapping overrides: a user's typing of a source field, and its origin."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum

# External
from pydantic import BaseModel, ConfigDict

# Internal
from kalanos.analysis.models.paths import AnyPath


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

MAPPING_SCHEMA_VERSION = 1


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class OverrideOrigin(str, Enum):
    """Which of the three override sources an override came from."""

    # fmt: off
    ARGUMENT = "argument"  # `--map`, or `mapping=` from the library
    FILE     = "file"      # `--map-file`, or `mapping_file=` from the library
    SIDECAR  = "sidecar"   # A `kalanos-map.yaml` in the graded root
    # fmt: on


class MappingOverride(BaseModel):
    """One source field typed by the user for this run.

    Attributes
    ----------
    feature : str
        The `Stream.source_field` this override matches, exactly.
    taxonomy_type : str
        The dictionary key every matching stream is typed as.
    origin : OverrideOrigin
        Which source supplied this override.
    path : UPath or None
        The file it was read from; `None` for `ARGUMENT`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    taxonomy_type: str
    origin: OverrideOrigin
    path: AnyPath | None = None


class MappingFile(BaseModel):
    """The on-disk shape shared by `--map-file` and the sidecar.

    Attributes
    ----------
    schema_version : int
        Must equal `MAPPING_SCHEMA_VERSION`.
    features : dict[str, str]
        Each source field, mapped to the taxonomy type it should be graded as.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    features: dict[str, str]
