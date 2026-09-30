"""Drive selection over one path and hand back the graded Report.

`discovery` and `reporting` are the pipeline stages this spans, with
`adapters` as the library in between: every discovered adapter bids on each
candidate, the highest bid reads it, and nothing may see a stage after its own.
This module sits beside the stages rather than inside any one, so it can import
several — a `test_boundaries.py` check pins it as the only module that does.
It never writes a file; its caller, such as the CLI, does that with
`reporting.write_report`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter

# External
from upath import UPath

# Internal
from kalanos.analysis.adapters.discover import discover_adapters
from kalanos.analysis.adapters.select import select_adapter
from kalanos.analysis.discovery.walk import walk_folder
from kalanos.analysis.execution import use_tier
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.discovery import (
    SkippedSource,
    SkipReason,
    SourceCandidate,
    SourceInfo,
)
from kalanos.analysis.models.domain import Episode, MappingSource
from kalanos.analysis.models.errors import MappingOverrideError
from kalanos.analysis.models.mapping import MappingOverride
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.provenance import (
    ExecutionTier,
    Inventory,
    Producer,
    RunCompletion,
    RunInfo,
)
from kalanos.analysis.models.report import AnalysedEpisode, Report
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.reporting.assemble import assemble_report
from kalanos.assets.bundle import RunConfiguration


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# How many seen source fields an unmatched-override error lists before it stops.
_SEEN_FIELDS_SHOWN = 20


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _source_id(path: UPath, base: UPath) -> str:
    """Name one source file by its place under the walked root."""

    try:
        relative = path.relative_to(base).as_posix()
    except ValueError:
        return str(path)
    return path.name if relative == "." else relative


def _qualify_ids(episodes: list[Episode], *, path: UPath, base: UPath) -> list[Episode]:
    """Re-mint each episode's id so it stays unique across the whole walk.

    An adapter names an episode with whatever the file it read calls it.
    Two files in different folders can share that name, and their place under the
    walked root tells them apart, so it prefixes the id: `source::episode`.

    A single-recording file keeps the short form, `source` alone. Which files
    those are is the adapter's statement, not a count: an adapter that names
    its episode after the file itself (its stem or name) is saying the file
    *is* the recording. A container that names episodes by their own keys
    (LeRobot's `episode_000000`, an HDF5 group) always gets the qualified
    form, so a recording keeps the same id whether one or all of its
    siblings loaded.

    Parameters
    ----------
    episodes : list[Episode]
        Everything one adapter read from `path`.
    path : UPath
        The candidate the adapter was invoked on.
    base : UPath
        The folder ids are relative to.

    Returns
    -------
    list[Episode]
        The same episodes, in the same order, each with a re-minted id.
    """

    source = _source_id(path, base)
    whole_file = {path.name, path.stem}
    return [
        episode.model_copy(
            update={
                "id": source if episode.id in whole_file else f"{source}::{episode.id}"
            }
        )
        for episode in episodes
    ]


def _apply_overrides(
    episodes: list[Episode],
    by_feature: Mapping[str, MappingOverride],
    matched: set[str],
    seen: set[str],
) -> list[Episode]:
    """Retype every stream whose `source_field` a mapping override names.

    An override replaces a dictionary match as well as an unmapped fallback.

    Parameters
    ----------
    episodes : list[Episode]
        Everything one adapter read from a candidate.
    by_feature : Mapping[str, MappingOverride]
        The run's overrides, keyed by the source field each matches.
    matched : set[str]
        Grown in place with every override feature that matched a stream.
    seen : set[str]
        Grown in place with every source field read.
        An unmatched override's error lists them.

    Returns
    -------
    list[Episode]
        The same episodes, in the same order, with matching streams retyped.
    """

    retyped: list[Episode] = []
    for episode in episodes:
        streams = []
        for stream in episode.streams:
            if stream.source_field is None:
                streams.append(stream)
                continue
            seen.add(stream.source_field)
            override = by_feature.get(stream.source_field)
            if override is None:
                streams.append(stream)
                continue
            matched.add(override.feature)
            streams.append(
                stream.model_copy(
                    update={
                        "taxonomy_type": override.taxonomy_type,
                        "mapping_source": MappingSource.OVERRIDE,
                    }
                )
            )
        retyped.append(episode.model_copy(update={"streams": streams}))
    return retyped


def _unmatched_override_error(
    unmatched: Sequence[MappingOverride], seen: set[str]
) -> MappingOverrideError:
    """Build the error for overrides that matched no stream.

    Parameters
    ----------
    unmatched : Sequence[MappingOverride]
        The overrides no stream's `source_field` equalled.
    seen : set[str]
        Every source field the run read.

    Returns
    -------
    MappingOverrideError
        Naming each unmatched feature with its origin, and the fields that were seen.
    """

    missing = ", ".join(f"{o.feature!r} (from {o.origin.value})" for o in unmatched)
    fields = sorted(seen)
    listed = ", ".join(fields[:_SEEN_FIELDS_SHOWN])
    if len(fields) > _SEEN_FIELDS_SHOWN:
        listed += f", and {len(fields) - _SEEN_FIELDS_SHOWN} more"
    return MappingOverrideError(
        f"mapping override matched no stream: {missing}; "
        f"source fields seen: {listed or 'none'}"
    )


def with_declared_limits(policy: Policy, info: DatasetInfo) -> Policy:
    """Fill in every limit a metric's own policy entry asks `describe()` for.

    Each metric's `target_source`, declared in the policy file, names the
    `DatasetInfo` attribute to read its `target` from.
    That vocabulary lives entirely in the policy, so this function stays generic:
    it never has to know any particular metric or adapter by name.

    Parameters
    ----------
    policy : Policy
        The policy to fill limits into.
    info : DatasetInfo
        What the winning adapter's `describe()` declared about this candidate.

    Returns
    -------
    Policy
        `policy`, with every declared, not-already-set limit filled in.
    """

    filled = policy
    for metric_policy in policy.metrics.values():
        if metric_policy.target is None or metric_policy.target_source is None:
            continue
        value = getattr(info, metric_policy.target_source, None)
        if isinstance(value, int | float) and not isinstance(value, bool):
            filled = filled.with_limit(metric_policy.target, value)
    return filled


def _lies_under(path: UPath, claimed_root: UPath) -> bool:
    """Check whether `path` sits inside `claimed_root`, itself excluded."""

    if path == claimed_root:
        return False
    try:
        path.relative_to(claimed_root)
    except ValueError:
        return False
    return True


def run(
    root: UPath,
    *,
    policy: Policy,
    source: SourceInfo | None = None,
    overrides: Sequence[MappingOverride] = (),
    config: RunConfiguration | None = None,
    producer: Producer | None = None,
    run_info: RunInfo | None = None,
) -> Report:
    """Grade `root` end to end: walk, select an adapter, read, then assemble.

    Parameters
    ----------
    root : UPath
        A recording to grade, or a folder of them. `discovery.walk_folder`
        treats both identically, so nothing here branches on which it is.
    policy : Policy
        The loaded grading policy to score every metric against.
    source : SourceInfo or None
        What `root` was resolved from, recorded on the Report as given.
    overrides : Sequence[MappingOverride]
        Per-run mapping overrides, already merged and checked against the dictionary.
        Each retypes every stream whose `source_field` equals its `feature`.
    config : RunConfiguration or None
        The resolved scope, requirements and binding conflicts. `None` grades
        under the built-in `numeric-core` scope.
    producer, run_info : Producer, RunInfo or None
        Provenance to record on the report, when the caller built it.

    Returns
    -------
    Report
        Every recording an adapter read, graded and rolled up, alongside
        every file skipped or that never made it through to grading.

    Raises
    ------
    AdapterTie
        If more than one discovered adapter bid the same maximum confidence
        on a candidate.
    MappingOverrideError
        If an override matched no stream anywhere in the run.
    """

    start = perf_counter()
    base = root.parent if root.is_file() else root

    # Step 1: split what discovery found into candidates to analyse
    # and files it already skipped with a reason.
    found = walk_folder(root)
    candidates = [item for item in found if isinstance(item, SourceCandidate)]
    skipped: list[SkippedSource] = [
        item for item in found if isinstance(item, SkippedSource)
    ]
    logger.info(
        "discovery found %d candidate(s) and skipped %d file(s) under %s",
        len(candidates),
        len(skipped),
        root,
    )

    discovery = discover_adapters()
    for failure in discovery.failures:
        logger.warning(
            "%s (%s): failed to load: %s", failure.name, failure.origin, failure.reason
        )
    logger.info("discovered %d adapter(s)", len(discovery.adapters))

    analysed: list[AnalysedEpisode] = []
    datasets: list[DatasetInfo] = []
    unresolved: list[UnresolvedSource] = []
    claimed: list[UPath] = []
    by_feature = {override.feature: override for override in overrides}
    matched: set[str] = set()
    seen: set[str] = set()
    unresolved_episodes = 0
    refused_sources: list[str] = []
    inventory_notes: list[str] = []

    for candidate in candidates:
        # Step 2: a candidate a directory adapter already claimed was
        # read as part of that directory; offering it again would double-count it.
        if any(_lies_under(candidate.path, claimed_root) for claimed_root in claimed):
            continue

        # Step 3: every adapter bids on the path; the highest bid reads it.
        # Nothing claiming it is the normal case for most files in a real folder,
        # so it is logged at DEBUG rather than WARNING. An unclaimed folder is
        # not a skipped file: its own contents are still offered individually.
        selection = select_adapter(candidate.path, discovery.adapters)
        if selection is None:
            if candidate.path.is_dir():
                continue
            logger.debug("%s: no adapter bid above zero", candidate.path)
            skipped.append(
                SkippedSource(path=candidate.path, reason=SkipReason.NO_ADAPTER)
            )
            continue

        # Step 4: describe() before reading, so whatever it declares reaches
        # the policy before any of this candidate's episodes are graded.
        # A broken describe() must not lose an otherwise-readable dataset.
        try:
            info = selection.adapter.describe(candidate.path)
        except Exception as exc:
            logger.warning(
                "%s: %s.describe() raised %s: %s",
                candidate.path,
                selection.name,
                type(exc).__name__,
                exc,
            )
            info = DatasetInfo(adapter=selection.name, path=candidate.path)

        episode_policy = with_declared_limits(policy, info)

        # Step 5: read the path's episodes one at a time, keeping every episode
        # the adapter yielded before it refused. Only a refusal the adapter
        # raised about the input file is caught here — anything else, a
        # ValidationError from our own models included, is a bug and propagates.
        # The source's declaration is retained either way, so a refusal
        # part-way through is reconciled against it like a clean read.
        episodes: list[Episode] = []
        refusal: AdapterRefusal | None = None
        with use_tier(config.scope.tier if config else ExecutionTier.STANDARD):
            try:
                for episode in selection.adapter.episodes(candidate.path):
                    episodes.append(episode)
            except AdapterRefusal as exc:
                refusal = exc
        datasets.append(info)
        # A directory an adapter selected belongs to that adapter whatever
        # the read produced — episodes, none, or a refusal. Offering its
        # manifest and data files to other adapters would grade the same
        # source twice.
        if candidate.path.is_dir():
            claimed.append(candidate.path)
        if refusal is not None:
            logger.warning(
                "%s: unresolved after %d episode(s): %s",
                refusal.path,
                len(episodes),
                refusal.reason,
            )
            unresolved.append(refusal.as_unresolved())
            # A refused source is not one failed episode: the episodes it did
            # not yield are a gap of unknown size unless it declared a count.
            refused_sources.append(str(candidate.path))
        # A declared count the adapter did not deliver is an inventory gap:
        # those episodes exist somewhere and have no identities here, so they
        # are counted as unresolved rather than silently dropped from the
        # denominator. More loaded than declared is noted, not a gap.
        if info.episode_count is not None:
            if info.episode_count > len(episodes):
                unresolved_episodes += info.episode_count - len(episodes)
                inventory_notes.append(
                    f"{candidate.path}: declared {info.episode_count} episodes, "
                    f"loaded {len(episodes)}"
                    + (" before the adapter refused" if refusal else "")
                )
            elif info.episode_count < len(episodes):
                inventory_notes.append(
                    f"{candidate.path}: declared {info.episode_count} episodes, "
                    f"loaded {len(episodes)} (more than declared)"
                )
        if not episodes:
            continue

        # Step 6: an adapter's name for a recording is local to the file it read,
        # so re-mint it against the walked root to keep it unique across the run.
        episodes = _qualify_ids(episodes, path=candidate.path, base=base)
        if by_feature:
            episodes = _apply_overrides(episodes, by_feature, matched, seen)
        analysed.extend(
            AnalysedEpisode(
                episode=episode,
                adapter=selection.name,
                adapter_confidence=selection.confidence,
                policy=episode_policy,
            )
            for episode in episodes
        )
        logger.info("%s: analysed by %s", candidate.path, selection.name)

    unmatched = [o for o in overrides if o.feature not in matched]
    if unmatched:
        raise _unmatched_override_error(unmatched, seen)

    # Step 7: grade everything that made it through, and assemble the report.
    logger.info("graded %d episode(s); %d unresolved", len(analysed), len(unresolved))
    expected = (
        sum(d.episode_count for d in datasets if d.episode_count is not None)
        if any(d.episode_count is not None for d in datasets)
        else None
    )
    inventory = Inventory(
        expected=expected,
        loaded=len(analysed),
        # `failed` is for episodes with identities. A refused source is not
        # one episode; it is listed as a source, and whatever it declared but
        # did not yield is in `unresolved`.
        failed=[],
        unresolved=unresolved_episodes,
        refused_sources=refused_sources,
        # Finishing the walk proves nothing about episodes a source declared
        # but never yielded, and a refused source may hold any number of them.
        complete=unresolved_episodes == 0 and not refused_sources,
        notes=inventory_notes,
    )
    if run_info is not None:
        run_info = run_info.model_copy(
            update={
                "finished_at": datetime.now(timezone.utc),
                "completion": (
                    RunCompletion.COMPLETE
                    if inventory.complete
                    else RunCompletion.PARTIAL
                ),
            }
        )
    return assemble_report(
        root=root,
        analysed=analysed,
        policy=policy,
        skipped=skipped,
        unresolved=unresolved,
        duration_s=perf_counter() - start,
        source=source,
        datasets=datasets,
        mapping_overrides=list(overrides),
        requirements=config.requirements if config else None,
        scope=config.scope if config else None,
        producer=producer,
        run=run_info,
        inventory=inventory,
        binding_conflicts=config.conflicts if config else (),
    )
