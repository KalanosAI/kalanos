"""The library entry point: grade a path as the `kalanos grade` command does."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4

# External
from upath import UPath

# Internal
from kalanos.analysis import pipeline
from kalanos.analysis.discovery.source import enforce_limits, resolve_source
from kalanos.analysis.models.binding import Bundle
from kalanos.analysis.models.dictionary import Dictionary
from kalanos.analysis.models.discovery import SourceLimits
from kalanos.analysis.models.errors import NothingToGrade
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.provenance import (
    ExecutionTier,
    HashScope,
    Producer,
    RunInfo,
    SourceEvidence,
    content_digest,
)
from kalanos.analysis.models.report import Report
from kalanos.assets.bundle import load_bundle, resolve_run_configuration
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


def _package_version() -> str:
    """The installed kalanos version, or `unknown` when run from a bare checkout."""

    try:
        return version("kalanos")
    except PackageNotFoundError:
        return "unknown"


def _build_revision() -> str | None:
    """A build revision, when the environment supplies one; never guessed."""

    return os.environ.get("KALANOS_BUILD_REVISION") or None


def grade(
    path: str | os.PathLike[str] | UPath,
    *,
    policy: Policy | None = None,
    dictionary: Dictionary | None = None,
    limits: SourceLimits | None = None,
    mapping: Mapping[str, str] | None = None,
    mapping_file: str | os.PathLike[str] | UPath | None = None,
    sidecar: bool = True,
    bundle: str | os.PathLike[str] | UPath | Bundle | None = None,
    tier: ExecutionTier | None = None,
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
    mapping : Mapping[str, str] or None
        Source fields to type for this run, each a `Stream.source_field`
        mapped to a dictionary key. The same as `--map`.
    mapping_file : str, PathLike, UPath or None
        A YAML mapping file, the same as `--map-file`.
    bundle : str, PathLike, UPath, Bundle or None
        A configuration bundle (`--profile`): binding, requirements, policy
        and execution sections with separate identities. `None` grades under
        the built-in `numeric-core` scope at the standard tier.
    tier : ExecutionTier or None
        Overrides the bundle's execution tier. A tier never changes the
        requirements: skipping a required capability makes episodes unknown.
    sidecar : bool
        Whether to read a `kalanos-map.yaml` in the graded root,
        or beside it when the root is a file.

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
    MappingOverrideError
        If a mapping source is malformed, names a type the dictionary lacks,
        or names a field no stream has.

    Notes
    -----
    The three mapping sources merge per field: `mapping` beats `mapping_file`,
    which beats the sidecar. Each applied override is recorded on
    `Report.mapping_overrides` with where it came from.
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

    # Step 3: resolve every configuration input through the one resolver
    # `benchmark` also uses, so both describe the same interpretation.
    loaded_bundle: Bundle | None
    bundle_path: UPath | None = None
    if isinstance(bundle, Bundle):
        loaded_bundle = bundle
    elif bundle is not None:
        bundle_path = UPath(bundle)
        loaded_bundle = load_bundle(bundle_path)
    else:
        loaded_bundle = None
    if loaded_bundle is not None and loaded_bundle.policy.path is not None:
        policy = load_policy(UPath(loaded_bundle.policy.path))
    config = resolve_run_configuration(
        root,
        dictionary=dictionary,
        policy_digest=content_digest(policy.model_dump(mode="json")),
        bundle=loaded_bundle,
        bundle_path=bundle_path,
        mapping=mapping,
        mapping_file=UPath(mapping_file) if mapping_file is not None else None,
        sidecar=sidecar,
        tier=tier,
    )
    for conflict in config.conflicts:
        logger.info(
            "mapping %s: %s from %s displaced %s",
            conflict.feature,
            conflict.winner.taxonomy_type,
            conflict.winner.origin.value,
            ", ".join(
                f"{d.taxonomy_type} ({d.origin.value})" for d in conflict.displaced
            ),
        )
    logger.info("grading %s under scope %s", root, config.scope.requirements_id)

    producer = Producer(version=_package_version(), revision=_build_revision())
    run_info = RunInfo(
        id=uuid4().hex,
        started_at=datetime.now(timezone.utc),
        tier=config.scope.tier,
        source=SourceEvidence(
            scope=HashScope.METADATA,
            digest=None,
            covered_inputs=0,
            complete=False,
            revision=source.revision,
        ),
        requirements=config.requirements_id,
        policy=config.policy_id,
        binding=config.binding_id,
        dictionary=config.dictionary_id,
    )

    # Step 4: run the pipeline. Every file ends up analysed, skipped or unresolved,
    # so an empty report means the path itself held nothing.
    report = pipeline.run(
        root,
        policy=policy,
        source=source,
        overrides=config.overrides,
        config=config,
        producer=producer,
        run_info=run_info,
    )
    if not (report.episodes or report.skipped or report.unresolved):
        raise NothingToGrade(f"{root} contains nothing to grade")
    return report
