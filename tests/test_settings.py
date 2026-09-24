"""Verifies the settings module: env-driven config.

Retrofitting these later is expensive, so they are pinned as tests
before any pipeline code reads from Settings.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

from kalanos.core.log import Verbosity
from kalanos.core.settings import Settings


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_defaults_load_with_no_environment_set():
    """Verify Settings constructs with no KALANOS_ environment variables present.

    `data_root`, `policy_path` and `dictionary_path` default to None on purpose —
    the CLI argument supplies the first, and the packaged policy and dictionary supply
    the other two, so a stand-in default for any of them would be silently wrong.
    """

    settings = Settings()
    assert settings.data_root is None
    assert settings.policy_path is None
    assert settings.dictionary_path is None
    assert settings.reports_dir == Path()


def test_data_root_is_overridden_through_the_environment(monkeypatch, tmp_path):
    """Verify KALANOS_DATA_ROOT overrides the default data root."""

    monkeypatch.setenv("KALANOS_DATA_ROOT", str(tmp_path))
    assert str(Settings().data_root) == str(tmp_path)


def test_reports_dir_is_overridden_through_the_environment(monkeypatch, tmp_path):
    """Verify KALANOS_REPORTS_DIR overrides the default reports directory."""

    monkeypatch.setenv("KALANOS_REPORTS_DIR", str(tmp_path))
    assert str(Settings().reports_dir) == str(tmp_path)


def test_policy_path_is_overridden_through_the_environment(monkeypatch, tmp_path):
    """Verify KALANOS_POLICY_PATH overrides the default policy path."""

    override = tmp_path / "policy.yaml"
    monkeypatch.setenv("KALANOS_POLICY_PATH", str(override))
    assert str(Settings().policy_path) == str(override)


def test_dictionary_path_is_overridden_through_the_environment(monkeypatch, tmp_path):
    """Verify KALANOS_DICTIONARY_PATH overrides the default dictionary path."""

    override = tmp_path / "dictionary.yaml"
    monkeypatch.setenv("KALANOS_DICTIONARY_PATH", str(override))
    assert str(Settings().dictionary_path) == str(override)


def test_verbosity_is_overridden_through_the_environment(monkeypatch):
    """Verify KALANOS_VERBOSITY overrides the default verbosity.

    Asserted at the `Settings` level only: `get_settings()` is `lru_cache`d,
    so an env change mid-process does not reliably reach a CLI invocation.
    """

    monkeypatch.setenv("KALANOS_VERBOSITY", "debug")
    assert Settings().verbosity is Verbosity.DEBUG


def test_plugin_dir_defaults_under_the_user_config_directory(monkeypatch):
    """Verify plugin_dir is built from platformdirs' user config dir, not a fixed path.

    Patches `user_config_dir` itself rather than comparing against its
    return value, so the assertion would fail if `plugin_dir` ever hard-coded
    a path instead of calling through to it.
    """

    import kalanos.core.settings as settings_module

    monkeypatch.setattr(
        settings_module, "user_config_dir", lambda app: f"/fake/config/{app}"
    )

    assert Settings().plugin_dir == Path("/fake/config/kalanos/plugins")


def test_plugin_dir_is_overridden_through_the_environment(monkeypatch, tmp_path):
    """Verify KALANOS_PLUGIN_DIR overrides the default plugin directory."""

    monkeypatch.setenv("KALANOS_PLUGIN_DIR", str(tmp_path))
    assert str(Settings().plugin_dir) == str(tmp_path)
