"""Verifies `check_metric` itself, and that every built-in metric passes it."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import pytest

# Internal
# The bare package import runs every built-in module's @metric decorators —
# timing, integrity and motion alike — so the parametrised sweep below sees
# the full built-in set the moment this module is collected.
import kalanos.analysis.metrics  # noqa: F401
from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.metrics.timing import dt_jitter_ms
from kalanos.analysis.models.domain import FramePayload
from kalanos.analysis.models.metrics import (
    ChannelContext,
    Family,
    Level,
    MetricResult,
    MetricStatus,
    StreamContext,
)
from kalanos.assets.policy import load_default_policy
from kalanos.testing import (
    Defect,
    check_metric,
    clean_recording,
    clean_taxels,
    stream_context,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Which fault each built-in metric exists to catch. `None` means the metric
# is checked in its own test because no injector builds its defect.
# A metric missing from here has no discrimination test — the sweep below fails on it.
BUILT_IN_DEFECTS: dict[str, Defect | None] = {
    "effective_hz": Defect.CLOCK_DRIFT,
    "dt_jitter_ms": Defect.JITTER,
    "drop_rate": Defect.DROPOUT,
    "missing_pct": Defect.NULLS,
    "flatline_pct": Defect.STUCK_CHANNEL,
    "spike_pct": Defect.SPIKE,
    "drift": Defect.DRIFT,
    "snr_db": Defect.NOISE,
    "dead_taxel_pct": Defect.DEAD_TAXEL,
    "hysteresis": Defect.HYSTERESIS,
    "mean_jerk_norm": Defect.SPIKE,
    "max_abs_jerk": Defect.SPIKE,
    "action_chatter": Defect.SPIKE,
    "still_drift": Defect.DRIFT,
    "vel_saturation_pct": Defect.SATURATION,
    "limit_proximity_pct": Defect.SATURATION,
    "hf_vibration_ratio": Defect.NOISE,
    "p99_torque": Defect.SATURATION,
    "max_torque": Defect.SATURATION,
    "mean_torque": Defect.SATURATION,
    "energy_proxy": None,
}

_BUILT_IN_MODULE_PREFIX = "kalanos.analysis.metrics."

# The one taxonomy type whose clean context is a tactile array rather than
# the default three-channel series.
_TAXEL_PRESSURE = "extero.taxel_pressure"


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def _built_in_entries():
    """List every registered metric whose module lives under the built-in package."""

    return [
        entry
        for level in Level
        for entry in registered_metrics(level)
        if entry.module.startswith(_BUILT_IN_MODULE_PREFIX)
    ]


def test_every_built_in_metric_declares_the_defect_it_fires_on():
    """Every built-in metric is a key of `BUILT_IN_DEFECTS`."""

    for entry in _built_in_entries():
        assert entry.name in BUILT_IN_DEFECTS, (
            f"{entry.name} has no entry in BUILT_IN_DEFECTS; it has no "
            "discrimination test"
        )


@pytest.mark.parametrize("entry", _built_in_entries(), ids=lambda entry: entry.name)
def test_every_built_in_metric_passes_the_contract(entry):
    """Every built-in metric holds the properties `check_metric` demands."""

    fires_on = BUILT_IN_DEFECTS[entry.name]
    if fires_on is None:
        pytest.skip(
            f"{entry.name} has no injector-built defect; checked in its own test"
        )

    taxonomy_type = (
        entry.requires.taxonomy[0] if entry.requires.taxonomy else ("unmapped.tcp_pose")
    )
    samples = max(100, entry.requires.min_samples)

    if taxonomy_type == _TAXEL_PRESSURE:
        stream = clean_taxels(samples=samples)
    elif entry.family is Family.MOTION and entry.level is Level.STREAM:
        # still_drift's clean context must end settled to discriminate a
        # DRIFT injection; a settled tail is harmless to its stream-level
        # siblings, so every motion stream metric gets one rather than
        # special-casing a metric name here.
        stream = clean_recording(
            samples=samples, settle=samples // 5, taxonomy_type=taxonomy_type
        )
    else:
        stream = clean_recording(samples=samples, taxonomy_type=taxonomy_type)

    if entry.level is Level.CHANNEL:
        assert isinstance(stream.payload, FramePayload)
        channel = stream.channels[0]
        ctx = ChannelContext(
            channel=channel,
            values=stream.payload.frame[channel.name],
            stream=stream_context(stream, is_regular=True),
        )
    else:
        ctx = stream_context(stream, is_regular=True)

    check_metric(
        entry.func,
        clean=ctx,
        fires_on=fires_on,
        policy=load_default_policy(),
    )


def test_check_metric_rejects_a_metric_absent_from_the_registry():
    """A plain undecorated function fails the registration check."""

    def unregistered(ctx: StreamContext) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    with pytest.raises(AssertionError, match="registered exactly once"):
        check_metric(unregistered, clean=stream_context(clean_recording()))


def test_check_metric_rejects_a_metric_that_cannot_tell_the_defect_apart(monkeypatch):
    """A metric returning the same value on clean and defective data fails."""

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.STREAM, family=Family.INTEGRITY)
    def constant_metric(ctx: StreamContext) -> MetricResult:
        return MetricResult(value=1.0, unit=None, status=MetricStatus.REPORT_ONLY)

    with pytest.raises(AssertionError, match="could not tell"):
        check_metric(
            constant_metric,
            clean=stream_context(clean_recording()),
            fires_on=Defect.DROPOUT,
        )


def test_check_metric_rejects_a_metric_with_no_policy_entry(monkeypatch):
    """A registered metric absent from the shipped policy fails the policy check."""

    import kalanos.analysis.metrics.registry as registry

    monkeypatch.setattr(registry, "_REGISTRY", list(registry._REGISTRY))

    @registry.metric(level=Level.STREAM, family=Family.INTEGRITY)
    def unpolicied_metric(ctx: StreamContext) -> MetricResult:
        return MetricResult(
            value=float(ctx.n_samples), unit=None, status=MetricStatus.REPORT_ONLY
        )

    with pytest.raises(AssertionError, match="no entry in the given policy"):
        check_metric(
            unpolicied_metric,
            clean=stream_context(clean_recording()),
            policy=load_default_policy(),
        )


def test_naming_both_a_defect_and_a_defective_context_is_refused():
    """Passing `fires_on` and `defective` together is a caller mistake,
    not an override."""

    with pytest.raises(ValueError, match="fires_on and defective"):
        check_metric(
            dt_jitter_ms,
            clean=stream_context(clean_recording()),
            fires_on=Defect.JITTER,
            defective=stream_context(clean_recording()),
        )


def test_a_defect_the_clean_context_cannot_carry_fails_naming_the_metric():
    """A defect no series stream can carry fails with the metric's own name."""

    with pytest.raises(AssertionError, match="dt_jitter_ms"):
        check_metric(
            dt_jitter_ms,
            clean=stream_context(clean_recording()),
            fires_on=Defect.FROZEN_FRAMES,
        )
