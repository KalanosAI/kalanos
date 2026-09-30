"""A YAML loader that refuses duplicate mapping keys.

`yaml.safe_load` silently keeps the last of two identical keys, so a bundle
or mapping file that says `observation.state` twice with different types
would load as whichever came last, and the conflict would never reach the
resolver. Configuration disagreement is an error, not a coin toss.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from typing import Any

# External
import yaml


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class DuplicateKeyError(yaml.YAMLError):
    """Two identical keys in one YAML mapping."""


class _StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise DuplicateKeyError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def safe_load_strict(text: str) -> Any:
    """`yaml.safe_load`, except that a duplicate key raises `DuplicateKeyError`."""

    return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass


__all__ = ["DuplicateKeyError", "safe_load_strict"]
