"""Loads dictionary.yaml: the packaged taxonomy, or a deployment's own override.

Nothing outside this module may open a dictionary file directly —
see `kalanos.analysis.models.dictionary` for what a loaded dictionary looks like,
and `docs/ARCHITECTURE.md` for why the split from `policy.yaml` exists at all.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from importlib import resources
from pathlib import Path

# External
import yaml
from pydantic import ValidationError

# Internal
from kalanos.analysis.models.dictionary import Dictionary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# `resources.files` resolves through the package's own loader, so this works
# whether `kalanos` is an editable checkout or unzipped from a wheel.
# A `Path(__file__)` walk would break for the wheel case.
_ANCHOR_PACKAGE = "kalanos.assets"
_DEFAULT_DICTIONARY_RESOURCE = ("dictionary.yaml",)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _ActiveDictionary:  # Singleton
    """Process-wide slot for a deployment's dictionary override.

    Adapters are constructed with no arguments during plugin discovery,
    so an override has no other route to a third-party adapter.
    """

    dictionary: Dictionary | None = None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _parse_dictionary(raw_yaml: str, *, source: str) -> Dictionary:
    """Parse dictionary YAML text into a Dictionary, failing at parse time.

    A missing entry surfaces here as a parse error, rather than downstream as a
    stream that quietly typed itself `unmapped` and lost its physics-aware metrics.

    Parameters
    ----------
    raw_yaml : str
        The YAML text to parse.
    source : str
        Where `raw_yaml` came from,
        folded into the error message so a bad override
        names the file a user actually edited.

    Returns
    -------
    Dictionary
        The parsed and validated dictionary.

    Raises
    ------
    ValueError
        If `raw_yaml` is not valid YAML, or does not match the Dictionary schema.
    """

    # Step 1: the file must at least be YAML before its shape is worth checking.
    try:
        payload = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} is not valid YAML: {exc}") from exc

    # Step 2: it must also match the Dictionary schema — which is where a threshold
    # or a format note copied in from the reference tables is refused.
    try:
        return Dictionary.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            f"{source} does not match the dictionary schema: {exc}"
        ) from exc


def use_dictionary(dictionary: Dictionary) -> None:
    """Install the dictionary every adapter resolves names against for this process."""

    _ActiveDictionary.dictionary = dictionary


def clear_active_dictionary() -> None:
    """Forget the installed dictionary, falling back to the packaged default."""

    _ActiveDictionary.dictionary = None


def load_default_dictionary() -> Dictionary:
    """Return the installed dictionary, or load the one shipped inside the package.

    Returns
    -------
    Dictionary
        The dictionary installed with `use_dictionary`, or the default.

    Raises
    ------
    ValueError
        If nothing is installed and the packaged dictionary.yaml is missing
        from the install.
    """

    if _ActiveDictionary.dictionary is not None:
        return _ActiveDictionary.dictionary

    resource = resources.files(_ANCHOR_PACKAGE).joinpath(*_DEFAULT_DICTIONARY_RESOURCE)
    try:
        raw_yaml = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(
            "the packaged dictionary.yaml is missing from this install; "
            "without it every signal is unmapped"
        ) from exc
    return _parse_dictionary(raw_yaml, source="the default dictionary")


def load_dictionary(path: Path | None) -> Dictionary:
    """Load a dictionary: a deployment's own file when given, the default otherwise.

    Parameters
    ----------
    path : Path or None
        A user-supplied dictionary.yaml.
        `None` loads the packaged default instead.

    Returns
    -------
    Dictionary
        The parsed and validated dictionary.

    Raises
    ------
    ValueError
        If `path` does not exist.
    """

    if path is None:
        return load_default_dictionary()

    # Explicit UTF-8: `read_text()` falls back to the locale encoding until PEP 686
    # lands, and a dictionary carries unit strings and labels from every source
    # convention there is.
    try:
        raw_yaml = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"no dictionary file at {path}") from exc
    return _parse_dictionary(raw_yaml, source=str(path))
