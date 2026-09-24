"""Contract checks and defect injectors for testing Kalanos plugins.

The public surface a metric or adapter author imports to test their own
plugin.
"""

from kalanos.testing.contracts import check_adapter, check_metric
from kalanos.testing.injectors import (
    Defect,
    add_noise,
    apply_defect,
    clean_frames,
    clean_recording,
    clean_taxels,
    drift_channel,
    drop_samples,
    freeze_frames,
    jitter_clock,
    kill_taxels,
    null_run,
    saturate_channel,
    skew_unloading,
    spike_channel,
    stick_channel,
    stream_context,
    stretch_clock,
)


__all__ = [
    "Defect",
    "add_noise",
    "apply_defect",
    "check_adapter",
    "check_metric",
    "clean_frames",
    "clean_recording",
    "clean_taxels",
    "drift_channel",
    "drop_samples",
    "freeze_frames",
    "jitter_clock",
    "kill_taxels",
    "null_run",
    "saturate_channel",
    "skew_unloading",
    "spike_channel",
    "stick_channel",
    "stream_context",
    "stretch_clock",
]
