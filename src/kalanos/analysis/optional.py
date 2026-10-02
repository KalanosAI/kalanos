"""Loaders for dependencies the base install does not require."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import importlib
from types import ModuleType


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def load_numpy() -> ModuleType | None:
    """Import numpy, or return `None` when it is not installed."""

    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


def load_huggingface_hub() -> ModuleType | None:
    """Import huggingface_hub, or return `None` when the `hf` extra is not installed."""

    try:
        return importlib.import_module("huggingface_hub")
    except ImportError:
        return None
