"""Loads and merges per-run mapping overrides: `--map`, `--map-file` and the sidecar."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Sequence

# External
import yaml
from pydantic import ValidationError
from upath import UPath

# Internal
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.errors import MappingOverrideError
from kalanos.analysis.models.mapping import (
    MAPPING_SCHEMA_VERSION,
    MappingFile,
    MappingOverride,
    OverrideOrigin,
)
from kalanos.assets.yaml_strict import safe_load_strict


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

SIDECAR_NAME = "kalanos-map.yaml"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def load_mapping_file(path: UPath, *, origin: OverrideOrigin) -> list[MappingOverride]:
    """Read one mapping file into overrides.

    Parameters
    ----------
    path : UPath
        The YAML file to read.
    origin : OverrideOrigin
        Recorded on every override the file yields.

    Returns
    -------
    list[MappingOverride]
        One override per `features` entry, in file order.

    Raises
    ------
    MappingOverrideError
        If the file is missing, is not YAML, does not match `MappingFile`,
        or declares a `schema_version` other than `MAPPING_SCHEMA_VERSION`.
    """

    try:
        raw_yaml = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MappingOverrideError(f"no mapping file at {path}") from exc

    try:
        payload = safe_load_strict(raw_yaml)
    except yaml.YAMLError as exc:
        raise MappingOverrideError(f"{path} is not valid YAML: {exc}") from exc

    try:
        mapping_file = MappingFile.model_validate(payload)
    except ValidationError as exc:
        raise MappingOverrideError(
            f"{path} does not match the mapping file schema: {exc}"
        ) from exc

    if mapping_file.schema_version != MAPPING_SCHEMA_VERSION:
        raise MappingOverrideError(
            f"{path} declares schema_version {mapping_file.schema_version}; "
            f"this version of kalanos reads {MAPPING_SCHEMA_VERSION}"
        )

    return [
        MappingOverride(
            feature=feature, taxonomy_type=taxonomy_type, origin=origin, path=path
        )
        for feature, taxonomy_type in mapping_file.features.items()
    ]


def sidecar_path(root: UPath) -> UPath:
    """Locate a root's sidecar: inside it when a folder, beside it when a file.

    Parameters
    ----------
    root : UPath
        The path being graded.

    Returns
    -------
    UPath
        Where the sidecar would be. It need not exist.
    """

    return (root if root.is_dir() else root.parent) / SIDECAR_NAME


def parse_map_argument(text: str) -> tuple[str, str]:
    """Split one `FEATURE=TYPE` argument.

    Splits on the last `=`, because a taxonomy key never contains one
    and a source field might.

    Parameters
    ----------
    text : str
        The argument as the user typed it.

    Returns
    -------
    tuple[str, str]
        The feature and the taxonomy type, each stripped.

    Raises
    ------
    MappingOverrideError
        If either side is empty.
    """

    feature, _, taxonomy_type = text.rpartition("=")
    feature, taxonomy_type = feature.strip(), taxonomy_type.strip()
    if not feature or not taxonomy_type:
        raise MappingOverrideError(
            f"--map {text!r} is not FEATURE=TYPE, "
            "e.g. observation.state=proprio.joint_position"
        )
    return feature, taxonomy_type


def check_taxonomy_types(
    overrides: Sequence[MappingOverride], dictionary: Dictionary
) -> None:
    """Refuse any override whose taxonomy type the dictionary does not define.

    Parameters
    ----------
    overrides : Sequence[MappingOverride]
        The merged overrides.
    dictionary : Dictionary
        The dictionary this run grades against.

    Raises
    ------
    MappingOverrideError
        Listing every override with an unknown type, its feature and its origin.
    """

    unknown = [o for o in overrides if o.taxonomy_type not in dictionary.entries]
    if unknown:
        listed = "; ".join(
            f"{o.feature}={o.taxonomy_type} (from {o.origin.value})" for o in unknown
        )
        raise MappingOverrideError(
            "mapping override names a taxonomy type the dictionary does not "
            f"define: {listed}"
        )
