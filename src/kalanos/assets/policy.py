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
from typing import Any

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

# Policies shipped inside the package, loadable by name as well as by path.
PACKAGED_POLICIES = ("default", "language_conditioned", "legacy_0_5")


def _packaged_yaml(name: str) -> str:
    """Read a packaged policy's YAML text by name."""

    return (
        resources.files(_ANCHOR_PACKAGE)
        .joinpath("policies", f"{name}.yaml")
        .read_text(encoding="utf-8")
    )


def _resolve_extends(payload: Any, *, source: str) -> Any:
    """Merge a policy that `extends` a packaged one over its base.

    `metrics` merge entry by entry, each entry replacing the base's whole
    entry; every other top-level key replaces the base's, so `gate: null`
    switches the gate off.
    """

    if not isinstance(payload, dict) or "extends" not in payload:
        return payload
    base_name = payload.pop("extends")
    if base_name not in PACKAGED_POLICIES:
        raise ValueError(
            f"{source} extends {base_name!r}; packaged policies are "
            f"{', '.join(PACKAGED_POLICIES)}"
        )
    base = _resolve_extends(yaml.safe_load(_packaged_yaml(base_name)), source=base_name)
    merged = {**base, **{k: v for k, v in payload.items() if k != "metrics"}}
    merged["metrics"] = {**base.get("metrics", {}), **payload.get("metrics", {})}
    return merged


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

    # Step 2: a policy that extends a packaged one is merged over it.
    payload = _resolve_extends(payload, source=source)

    # Step 3: it must also match the Policy schema.
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
    if not Path(path).exists() and str(path) in PACKAGED_POLICIES:
        logger.info("using the packaged %s policy", path)
        return _parse_policy(_packaged_yaml(str(path)), source=f"the {path} policy")
    logger.info("using policy override %s", path)
    return _parse_policy(path.read_text(encoding="utf-8"), source=str(path))
