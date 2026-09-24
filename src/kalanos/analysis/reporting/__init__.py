"""Assemble the report and render it as JSON, YAML, HTML or a terminal card."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
# Imported for its registration side effect:
# the @reporter decorations there are what populate the reporter registry,
# so registered_reporters() is right for a caller that never touches write.
from kalanos.analysis.reporting import render  # noqa: F401
