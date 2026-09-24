"""The @metric decorator and the registry that runs a level's metrics.

A metric registers itself once, at import time, by decorating its function
with its Level, Family and Requires. The registry is the only place that checks
Requires against a context, so a metric function may assume its declared
requirements already hold by the time it runs — through one of two gates,
one for a stream or channel and one for an episode.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

# Internal
from kalanos.analysis.models.metrics import (
    ChannelContext,
    EpisodeContext,
    Family,
    Level,
    MetricInput,
    MetricResult,
    MetricStatus,
    Requires,
    StreamContext,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


MetricFunction = Callable[[MetricInput], MetricResult]

# A metric's actual context type: StreamContext, ChannelContext or EpisodeContext,
# not the shared MetricInput union. Keeps each decorated function's own type.
CtxT = TypeVar("CtxT", bound=MetricInput)


@dataclass(frozen=True)
class RegisteredMetric:
    """One metric as the registry knows it: name, level, family, requirements and body.

    Attributes
    ----------
    name : str
        The metric's name, taken from the decorated function.
    level : Level
        Where in the rollup this metric attaches.
    family : Family
        The question this metric asks, and the policy weight key it falls under.
    requires : Requires
        What a context must satisfy before the registry calls `func`.
    func : MetricFunction
        The decorated function itself.
    module : str
        The module that registered it, so discovery can attribute a metric to
        the entry point whose import put it here.
    """

    name: str
    level: Level
    family: Family
    requires: Requires
    func: MetricFunction
    module: str


# Populated by every @metric-decorated function as its module is imported.
# A plain module list rather than a class, since nothing here needs identity
# beyond the name each entry already carries.
_REGISTRY: list[RegisteredMetric] = []


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def metric(
    *, level: Level, family: Family | str, requires: Requires | None = None
) -> Callable[[Callable[[CtxT], MetricResult]], Callable[[CtxT], MetricResult]]:
    """Register a function as a metric, keyed by its own name.

    Parameters
    ----------
    level : Level
        Where in the rollup the decorated function attaches.
    family : Family or str
        The question this metric asks. A string is coerced,
        so a family outside the nine fails at import rather than at grading time.
    requires : Requires or None
        What a context must satisfy before the registry calls it.
        Defaults to no requirements at all.

    Returns
    -------
    Callable
        A decorator that registers `func` unchanged and returns it.

    Raises
    ------
    ValueError
        1. If `family` names no member of `Family`.
        2. If `level` is `Level.EPISODE` and `requires` declares
           lower-level structural requirements.
    """

    resolved_family = Family(family)
    resolved_requires = requires or Requires()

    def decorator(
        func: Callable[[CtxT], MetricResult],
    ) -> Callable[[CtxT], MetricResult]:
        if level is Level.EPISODE and (
            resolved_requires.regular_sampling or resolved_requires.min_samples
        ):
            raise ValueError(
                f"{func.__name__} is registered at Level.EPISODE with a "
                "structural requirement (regular_sampling or min_samples); an "
                "episode metric gates on taxonomy only — check sampling or "
                "sample counts inside the function, on the streams it "
                "actually reads"
            )

        # Stored as the generic MetricFunction type so all levels share one
        # registry list. Nobody checks that `level` matches func's context
        # type; callers must keep e.g. STREAM metrics taking StreamContext.
        _REGISTRY.append(
            RegisteredMetric(
                name=func.__name__,
                level=level,
                family=resolved_family,
                requires=resolved_requires,
                func=cast(MetricFunction, func),
                module=func.__module__,
            )
        )
        return func

    return decorator


def registered_metrics(level: Level | None = None) -> list[RegisteredMetric]:
    """List registered metrics in registration order, optionally at one level.

    Parameters
    ----------
    level : Level or None
        The rollup level to filter by. `None` returns every registered metric.

    Returns
    -------
    list[RegisteredMetric]
        Every registered metric, or only those at `level` when one is given.
    """

    return [entry for entry in _REGISTRY if level is None or entry.level == level]


def _unmet_node_reason(
    requires: Requires, ctx: StreamContext | ChannelContext
) -> str | None:
    """Say why a stream or channel context fails one metric's requirements.

    Taxonomy is any-of: `ctx.taxonomy_type` must be one of `requires.taxonomy`.

    Parameters
    ----------
    requires : Requires
        The requirements to check.
    ctx : StreamContext or ChannelContext
        The context to check them against.

    Returns
    -------
    str or None
        A human-readable reason the requirements are unmet,
        or `None` when `ctx` satisfies every requirement.
    """

    if requires.taxonomy and ctx.taxonomy_type not in requires.taxonomy:
        return f"needs one of {sorted(requires.taxonomy)}, got {ctx.taxonomy_type!r}"
    if requires.regular_sampling and not ctx.is_regular:
        return "sampling is not regular"
    if ctx.n_samples < requires.min_samples:
        return f"fewer than {requires.min_samples} samples"
    return None


def _unmet_episode_reason(requires: Requires, ctx: EpisodeContext) -> str | None:
    """Say why an episode context fails one metric's requirements.

    Taxonomy is all-of: every type in `requires.taxonomy` must be present
    among the episode's streams. Structural requirements are refused at
    decoration time, so there is nothing else to check here.

    Parameters
    ----------
    requires : Requires
        The requirements to check.
    ctx : EpisodeContext
        The episode to check them against.

    Returns
    -------
    str or None
        A human-readable reason the requirements are unmet,
        or `None` when `ctx` satisfies every requirement.
    """

    missing = [
        wanted for wanted in requires.taxonomy if wanted not in ctx.taxonomy_types
    ]
    if missing:
        return f"episode carries no {', '.join(sorted(missing))} stream"
    return None


def _run(
    level: Level,
    ctx: MetricInput,
    unmet: Callable[[Requires, Any], str | None],
) -> dict[str, MetricResult]:
    """Run every registered metric at one level against one context.

    Parameters
    ----------
    level : Level
        Which registered metrics to run.
    ctx : MetricInput
        The context to run them against.
    unmet : Callable[[Requires, Any], str or None]
        The gate to check `ctx` against — `_unmet_node_reason` for a stream
        or channel context, `_unmet_episode_reason` for an episode.

    Returns
    -------
    dict[str, MetricResult]
        One result per registered metric at `level`, keyed by its name —
        `not_applicable`, with a reason in `evidence`, for anything the
        registry declined to run.
    """

    # Step 1: check requirements first, so a metric function is only ever
    # called on a context it already knows how to handle.
    results: dict[str, MetricResult] = {}
    for entry in registered_metrics(level):
        reason = unmet(entry.requires, ctx)
        if reason is not None:
            results[entry.name] = MetricResult(
                value=None,
                unit=None,
                status=MetricStatus.NOT_APPLICABLE,
                evidence={"reason": reason},
            )
            continue

        # Step 2: requirements hold, so the function runs for real.
        results[entry.name] = entry.func(ctx)

    return results


def run_stream_metrics(ctx: StreamContext) -> dict[str, MetricResult]:
    """Run every registered stream-level metric against one context.

    Parameters
    ----------
    ctx : StreamContext
        The stream's timestamps, payload and regularity verdict.

    Returns
    -------
    dict[str, MetricResult]
        One result per registered stream-level metric, keyed by its name.
    """

    return _run(Level.STREAM, ctx, _unmet_node_reason)


def run_channel_metrics(ctx: ChannelContext) -> dict[str, MetricResult]:
    """Run every registered channel-level metric against one context.

    Parameters
    ----------
    ctx : ChannelContext
        The channel data and its stream's regularity verdict.

    Returns
    -------
    dict[str, MetricResult]
        One result per registered channel-level metric, keyed by its name.
    """

    return _run(Level.CHANNEL, ctx, _unmet_node_reason)


def run_episode_metrics(ctx: EpisodeContext) -> dict[str, MetricResult]:
    """Run every registered episode-level metric against one context.

    Parameters
    ----------
    ctx : EpisodeContext
        The recording, for a metric that compares streams within it.

    Returns
    -------
    dict[str, MetricResult]
        One result per registered episode-level metric, keyed by its name.
    """

    return _run(Level.EPISODE, ctx, _unmet_episode_reason)
