# Requirements profiles

A bundle keeps binding, requirements, policy and execution separate. Existing
`kalanos-map.yaml`, `--map-file` and `--map` inputs remain supported under the
existing precedence; this patch does not deprecate or migrate sidecars.

```bash
kalanos profiles list
kalanos profiles show numeric-core-v1
kalanos profiles show vision-imitation-v1
kalanos profiles validate acquisition.yaml
```

Validation checks YAML/model structure and any referenced decision policy. It does
not load a dataset or certify a binding. Unknown capability or metric names cannot
produce a pass: their required evidence will be unavailable at grading time.

## Numeric data with required acquisition timing

Save as `acquisition.yaml`:

```yaml
schema_version: 1
requirements:
  id: numeric-with-capture-v1
  required_families: [integrity]
  require_numeric_payloads: true
  required_capabilities: [numeric, acquisition_timing]
  required_metrics: [integrity.missing_pct]
execution:
  tier: standard
```

```bash
kalanos grade ./recording --profile acquisition.yaml --report audit.json
```

Capture origin alone is insufficient: all streams must compute the acquisition
metrics. Built-in generic readers do not assert producer capture provenance, so
this requirement is expected to remain unknown unless an adapter has justified
producer evidence and usable timestamps. Metadata tier keeps the same requirements
and therefore leaves skipped numeric evaluation unknown.

To require torque measurement as well, add `motion.p99_torque` to
`required_metrics`. An episode must contain eligible torque subjects and evaluate
all of them. The value may remain report-only; this requirement does not define a
safe torque limit or authorize statistical blocking.

## Vision requirements

```yaml
schema_version: 1
requirements:
  id: vision-imitation-v1
execution:
  tier: full
```

This preset requires video quality: every declared frame of every camera examined.
At the full tier the vision metrics read every frame, so a run can satisfy it; at
the standard tier they read a sample, and it remains unknown. Camera metadata and
numeric analysis do not satisfy that requirement. Choosing a weaker profile changes the
question asked and must not be described as completing the original audit.

## A selected decision policy

```yaml
schema_version: 1
requirements:
  id: numeric-core-v1
policy:
  id: reviewed-numeric-policy-v1
  path: policies/reviewed-numeric.yaml
execution:
  tier: standard
```

The policy path resolves relative to the bundle. Use a complete policy with its
`calibration_manifests` list; see the [combined contract](R07-05-07.md). Missing or
invalid files fail with exit 2. Do not add a test manifest to a production policy.
Run a diagnostic grade first to obtain exact candidate calibration contexts, then
validate the final implementation and scope before recording acceptance.

The default gate fails `blocked,unknown`. For unattended training:

```bash
kalanos grade ./recording --profile acquisition.yaml \
  --fail-on blocked,review,unknown --report audit.json
```


## Noise reference evidence

A per-channel binding may include the following `noise_floor` object. These
numbers are a configuration illustration, **not a validated reference for your
robot**:

```yaml
noise_floor:
  standard_deviation: 0.001
  unit: rad
  source: reference_capture
  reference: evidence/session-reference-v1.json
  sensor_configuration: sensor-model-config-revision
  sample_rate_hz: 100
  bandwidth_hz: 50
  estimator: centered_mean_5_residual_std_v1
  scale_transform: null
```

The binding also needs `quantity`, `representation`, `unit`, feature/index and
source scope. Supply separate `validations` for `identity`, `quantity`, `unit`
and `noise_floor`, each with `value` equal to the resolved property, the exact
`scope` recorded as `source_identity`, `validator`, `evidence`, and
`capability: noise`. `identity` is `feature[index]`, including `[None]` for a
scalar with no index. The `noise_floor` validation value is the complete object,
including explicit null `scale_transform`. First inspect the actual resolved
binding; do not guess source identities or copy test attestations.

The reference must use the same native scale and estimator. An override to the
floor, unit, scale transform or source invalidates applicability until it is
validated again. Native sample rate must match. The validator is responsible for
verifying configuration and bandwidth against acquisition records. This evidence
makes the reference usable; statistical blocking still needs a separately
accepted, exactly matching calibration manifest.


## Deeper diagnostic plans

An optional `diagnostics` bundle section selects work without weakening existing
requirements. `sampled_video_quality` requires the requested samples on every
camera; `video_quality` requires every declared frame examined. Reducing the tier
or budget preserves the requirement and records skipped/unavailable work.
`cross_stream_timing`, `action_consistency`, `motion_shape` and `training_windows`
can be required capabilities; individual checks can be required with keys such as
`diagnostics.timing.camera_state` in `required_metrics`.

Use `requirements.min_pass_windows: {train: 10000}` for an explicit dataset
minimum. Configure review thresholds in the referenced policy's
`diagnostic_reviews`, not in the plan. See the complete
[diagnostic contract](DIAGNOSTICS-0.7.0.md) and
[example bundle](examples/diagnostics.yaml).
