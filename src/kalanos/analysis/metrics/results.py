"""Result-building helpers shared across metric families."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.models.metrics import MetricResult, MetricStatus


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def not_applicable(reason: str) -> MetricResult:
    """Build the not_applicable result every check across the families returns.

    Parameters
    ----------
    reason : str
        Why the metric could not be computed, carried in `evidence["reason"]`.

    Returns
    -------
    MetricResult
        `status=MetricStatus.NOT_APPLICABLE`, with no value or unit.
    """

    return MetricResult(
        value=None,
        unit=None,
        status=MetricStatus.NOT_APPLICABLE,
        evidence={"reason": reason},
    )
