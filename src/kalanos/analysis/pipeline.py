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
from time import perf_counter

# External
from upath import UPath

# Internal
from kalanos.analysis.adapters.discover import discover_adapters
from kalanos.analysis.adapters.select import select_adapter
from kalanos.analysis.discovery.walk import walk_folder
from kalanos.analysis.models.adapters import AdapterRefusal, DatasetInfo
from kalanos.analysis.models.discovery import (
    SkippedSource,
    SkipReason,
    SourceCandidate,
    SourceInfo,
)
from kalanos.analysis.models.domain import Episode
from kalanos.analysis.models.policy import Policy
from kalanos.analysis.models.report import AnalysedEpisode, Report
from kalanos.analysis.models.schema import UnresolvedSource
from kalanos.analysis.reporting.assemble import assemble_report


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

logger = logging.getLogger(__name__)


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
    walked root tells them apart, so it becomes the id; a file holding several
    recordings keeps the adapter's name for each after it.

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
    if len(episodes) == 1:
        return [episodes[0].model_copy(update={"id": source})]
    return [
        episode.model_copy(update={"id": f"{source}::{episode.id}"})
        for episode in episodes
    ]


def _with_declared_limits(policy: Policy, info: DatasetInfo) -> Policy:
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


def run(root: UPath, *, policy: Policy, source: SourceInfo | None = None) -> Report:
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

        episode_policy = _with_declared_limits(policy, info)

        # Step 5: read the path's episodes. Only a refusal the adapter raised
        # about the input file is caught here — anything else, a ValidationError
        # from our own models included, is a bug and propagates.
        try:
            episodes = list(selection.adapter.episodes(candidate.path))
        except AdapterRefusal as exc:
            logger.warning("%s: unresolved: %s", exc.path, exc.reason)
            unresolved.append(exc.as_unresolved())
            continue
        datasets.append(info)

        # Step 6: an adapter's name for a recording is local to the file it read,
        # so re-mint it against the walked root to keep it unique across the run.
        episodes = _qualify_ids(episodes, path=candidate.path, base=base)
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

        if candidate.path.is_dir():
            claimed.append(candidate.path)

    # Step 7: grade everything that made it through, and assemble the report.
    logger.info("graded %d episode(s); %d unresolved", len(analysed), len(unresolved))
    return assemble_report(
        root=root,
        analysed=analysed,
        policy=policy,
        skipped=skipped,
        unresolved=unresolved,
        duration_s=perf_counter() - start,
        source=source,
        datasets=datasets,
    )
