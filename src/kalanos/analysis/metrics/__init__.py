"""The metric registry and its families."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
# Imported for its registration side effect:
# decorating each function here is what puts it in the registry,
# so the module must load before anything calls run_channel_metrics.
from kalanos.analysis.metrics import (  # noqa: F401
    annotation,
    integrity,
    motion,
    timing,
    vision,
)
