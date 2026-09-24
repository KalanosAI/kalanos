"""The library entry point: grade a path as the `kalanos grade` command does."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import os

# External
from upath import UPath

# Internal
from kalanos.analysis import pipeline
from kalanos.analysis.discovery.source import enforce_limits, resolve_source
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.discovery import SourceLimits
from kalanos.analysis.models.errors import NothingToGrade
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.report import Report
from kalanos.assets.dictionary import load_dictionary, use_dictionary
from kalanos.assets.policy import load_policy
from kalanos.core.settings import get_settings


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def grade(
    path: str | os.PathLike[str] | UPath,
    *,
    policy: Policy | None = None,
    dictionary: Dictionary | None = None,
    limits: SourceLimits | None = None,
) -> Report:
    """Grade a recording, or every recording under a folder.

    Parameters
    ----------
    path : str, PathLike or UPath
        What to grade. Parsed as a `UPath`, so any protocol upath knows works.
    policy : Policy or None
        The grading policy.
        `None` loads `Settings.policy_path`, or the packaged default.
    dictionary : Dictionary or None
        The signal dictionary.
        `None` loads `Settings.dictionary_path`, or the packaged default.
        It becomes the process-wide active dictionary,
        so two concurrent grades must share one.
    limits : SourceLimits or None
        The largest remote root to stream.
        `None` uses `Settings.remote_max_bytes` and `Settings.remote_max_files`.

    Returns
    -------
    Report
        Every recording graded, plus every file skipped or refused, with its reason.

    Raises
    ------
    SourceUnavailable
        If `path` does not exist,
        a Hugging Face dataset is missing, gated or private,
        or the `hf` extra is not installed.
    SourceTooLarge
        If `path` is remote and over a limit; nothing was read.
    NothingToGrade
        If `path` held nothing to analyse, skip or refuse.
    AdapterTie
        If two adapters bid the same top confidence on one file.
    """

    # Step 1: resolve what was typed into a root, and refuse an oversized remote one,
    # before the policy and dictionary load.
    root, source = resolve_source(path)
    settings = get_settings()
    if limits is None:
        limits = SourceLimits(
            max_bytes=settings.remote_max_bytes, max_files=settings.remote_max_files
        )
    enforce_limits(source, limits)

    # Step 2: load whatever configuration the caller did not pass.
    if policy is None:
        policy = load_policy(settings.policy_path)
    if dictionary is None:
        dictionary = load_dictionary(settings.dictionary_path)
    use_dictionary(dictionary)
    logger.info("grading %s", root)

    # Step 3: run the pipeline. Every file ends up analysed, skipped or unresolved,
    # so an empty report means the path itself held nothing.
    report = pipeline.run(root, policy=policy, source=source)
    if not (report.episodes or report.skipped or report.unresolved):
        raise NothingToGrade(f"{root} contains nothing to grade")
    return report
