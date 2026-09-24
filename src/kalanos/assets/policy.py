"""Loads policy.yaml: the packaged default, or a deployment's own override.

Nothing outside this module may open a policy file directly —
see `kalanos.analysis.models.policy` for what a loaded policy looks like,
and `docs/ARCHITECTURE.md` for why the split from `dictionary.yaml` exists at all.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from importlib import resources
from pathlib import Path

# External
import yaml
from pydantic import ValidationError

# Internal
from kalanos.analysis.models.policy import Policy


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# `resources.files` resolves through the package's own loader, so this works
# whether `kalanos` is an editable checkout or unzipped from a wheel.
# A `Path(__file__)` walk would break for the wheel case.
_ANCHOR_PACKAGE = "kalanos.assets"
_DEFAULT_POLICY_RESOURCE = ("policies", "default.yaml")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _parse_policy(raw_yaml: str, *, source: str) -> Policy:
    """Parse policy YAML text into a Policy, failing at parse time.

    Turns a bad file into an error, instead of a scoring bug
    surfacing later from a metric silently missing its bands.

    Parameters
    ----------
    raw_yaml : str
        The YAML text to parse.
    source : str
        Where `raw_yaml` came from, folded into the error message
        so a bad override names the file a user actually edited.

    Returns
    -------
    Policy
        The parsed and validated policy.

    Raises
    ------
    ValueError
        If `raw_yaml` is not valid YAML, or does not match the Policy schema.
    """

    # Step 1: the file must at least be YAML before its shape is worth checking.
    try:
        payload = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"{source} is not valid YAML: {exc}") from exc

    # Step 2: it must also match the Policy schema.
    try:
        return Policy.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{source} does not match the policy schema: {exc}") from exc


def load_default_policy() -> Policy:
    """Load the policy shipped inside the package.

    Returns
    -------
    Policy
        The default policy.
    """

    raw_yaml = (
        resources.files(_ANCHOR_PACKAGE)
        .joinpath(*_DEFAULT_POLICY_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return _parse_policy(raw_yaml, source="the default policy")


def load_policy(path: Path | None) -> Policy:
    """Load a policy: a deployment's own file when given, the default otherwise.

    Parameters
    ----------
    path : Path or None
        A user-supplied policy.yaml, such as `Settings.policy_path`.
        `None` loads the packaged default instead.

    Returns
    -------
    Policy
        The parsed and validated policy.
    """

    if path is None:
        logger.info("using the default policy")
        return load_default_policy()
    logger.info("using policy override %s", path)
    return _parse_policy(path.read_text(encoding="utf-8"), source=str(path))
