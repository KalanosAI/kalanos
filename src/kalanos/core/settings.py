"""Application settings: where data lives and where reports go.

Every field reads from the environment under the `KALANOS_` prefix, with a
default. Frozen, so nothing can be mutated at a call site after construction.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from functools import lru_cache
from pathlib import Path

# External
from platformdirs import user_config_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Internal
from kalanos.core.log import Verbosity


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Settings(BaseSettings):
    """Process-wide configuration.

    Attributes
    ----------
    data_root : Path or None
        Folder to analyse when no path is given on the command line.
        `None` by default: the CLI argument is the normal way to supply one,
        and a default here would silently analyse the wrong folder.
    reports_dir : Path
        Folder reports are written to. Defaults to the working directory,
        so a report lands where the user ran the command.
    policy_path : Path or None
        Override for the `policy.yaml` grading this run uses. `None` means
        the default policy shipped inside the package.
    dictionary_path : Path or None
        Override for the `dictionary.yaml` this run resolves signal names against.
        `None` means the default dictionary shipped inside the package.
    plugin_dir : Path
        Where single-file adapter plugins are read from.
        Defaults to the platform's user config directory for `kalanos`,
        overridden by `KALANOS_PLUGIN_DIR`.
    verbosity : Verbosity
        Default log level for a run, overridden for one invocation by the
        CLI's `--verbosity` option. Defaults to `WARNING`.
    remote_max_bytes : int or None
        Refuse a remote root whose listed size is over this many bytes,
        overridden for one invocation by the CLI's `--max-remote-gb` option.
        Defaults to 20 GB; `None` lifts the limit.
    remote_max_files : int or None
        Refuse a remote root listing more files than this,
        overridden for one invocation by the CLI's `--max-remote-files` option.
        Defaults to 10,000; `None` lifts the limit.
    """

    model_config = SettingsConfigDict(
        env_prefix="KALANOS_",
        env_nested_delimiter="__",
        frozen=True,
    )

    data_root: Path | None = None
    reports_dir: Path = Path()
    policy_path: Path | None = None
    dictionary_path: Path | None = None
    plugin_dir: Path = Field(
        default_factory=lambda: Path(user_config_dir("kalanos")) / "plugins"
    )
    verbosity: Verbosity = Verbosity.WARNING
    remote_max_bytes: int | None = 20_000_000_000
    remote_max_files: int | None = 10_000


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance, constructed once.

    Returns
    -------
    Settings
        The cached settings instance, shared by every caller in the process.
    """

    return Settings()
