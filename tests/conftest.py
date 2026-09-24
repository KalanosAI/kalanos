"""Shared fixtures every test module in this package can use without importing it."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os

# External
import pytest

# Internal
from kalanos.assets.dictionary import clear_active_dictionary
from kalanos.core.settings import get_settings


# ░█▀▀░▀█▀░█░█░▀█▀░█░█░█▀▄░█▀▀░█▀▀
# ░█▀▀░░█░░▄▀▄░░█░░█░█░█▀▄░█▀▀░▀▀█
# ░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░▀▀▀


@pytest.fixture(autouse=True)
def _clear_settings_env(monkeypatch):
    """Strip any KALANOS_ environment variables and reset the cached Settings.

    `get_settings` is `lru_cache`d, so a value set by one test would
    otherwise leak into the next through the cache alone, even after
    `monkeypatch` undoes the environment variable itself.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's environment-patching fixture.
    """

    for key in list(os.environ):
        if key.startswith("KALANOS_"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_active_dictionary():
    """Forget any dictionary a test installed with `use_dictionary`.

    A process-wide side effect, so a test that installs an override cannot
    leak it into the next test.
    """

    yield
    clear_active_dictionary()
