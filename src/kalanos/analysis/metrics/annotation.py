"""task_instruction_missing — the first metric in the annotation family.

`annotation` asks whether an episode's labels are sound. The first label it checks
is the task instruction a language-conditioned policy learns to follow: an episode
recorded without one teaches "act without being told what to do".

The default policy keeps it report-only,
because whether a missing instruction matters depends on the model being trained:
it breaks a vision-language-action model and does not affect behaviour cloning.
The report lists the affected episodes in a dataset warning,
and the `language_conditioned` policy blocks them.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.metrics.registry import metric
from kalanos.analysis.metrics.results import not_applicable
from kalanos.analysis.models.metrics import (
    EpisodeContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


@metric(level=Level.EPISODE, family=Family.ANNOTATION, label="missing task instruction")
def task_instruction_missing(ctx: EpisodeContext) -> MetricResult:
    """Flag an episode that carries no task instruction.

    An instruction that is empty or only whitespace counts as missing. What else
    counts — placeholder strings a conversion pipeline writes — is left for real
    recordings to show; every episode's instructions are in the report meanwhile,
    so a placeholder is visible to a reader.

    Parameters
    ----------
    ctx : EpisodeContext
        The episode to check; only its `tasks` are read.

    Returns
    -------
    MetricResult
        `not_applicable` when the dataset carries no instruction for any episode
        (`tasks` is `None`): a dataset never annotated with language is not
        missing anything. `report_only` otherwise: `1.0` when the episode has no
        non-blank instruction, `0.0` when it has at least one.
    """

    tasks = ctx.episode.tasks
    if tasks is None:
        return not_applicable(
            "the dataset carries no task instruction for any episode, "
            "so none is missing"
        )

    present = [task for task in tasks if task.strip()]
    return MetricResult(
        value=0.0 if present else 1.0,
        unit="flag",
        status=MetricStatus.REPORT_ONLY,
        evidence={"n_tasks": len(tasks), "n_blank": len(tasks) - len(present)},
    )
