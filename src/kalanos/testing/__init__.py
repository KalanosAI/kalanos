"""Contract checks and defect injectors for testing Kalanos plugins.

The public surface a metric or adapter author imports to test their own
plugin.
"""

from kalanos.testing.contracts import check_adapter, check_metric
from kalanos.testing.injectors import (
    Defect,
    add_noise,
    apply_defect,
    blur_frames,
    clean_frames,
    clean_recording,
    clean_taxels,
    clip_frames,
    cut_frames,
    drift_channel,
    drop_chunk,
    drop_samples,
    freeze_frames,
    jitter_clock,
    kill_taxels,
    null_run,
    repeat_timestamps,
    saturate_channel,
    shift_video,
    skew_unloading,
    spike_channel,
    step_channel,
    stick_channel,
    stream_context,
    stretch_clock,
)


__all__ = [
    "Defect",
    "add_noise",
    "apply_defect",
    "blur_frames",
    "check_adapter",
    "check_metric",
    "clean_frames",
    "clip_frames",
    "clean_recording",
    "clean_taxels",
    "cut_frames",
    "drift_channel",
    "drop_chunk",
    "drop_samples",
    "freeze_frames",
    "jitter_clock",
    "kill_taxels",
    "null_run",
    "repeat_timestamps",
    "saturate_channel",
    "shift_video",
    "skew_unloading",
    "spike_channel",
    "step_channel",
    "stick_channel",
    "stream_context",
    "stretch_clock",
]
