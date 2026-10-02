# Requirements profiles

A bundle keeps binding, requirements, policy and execution separate. Existing `kalanos-map.yaml`, `--map-file` and `--map` inputs remain supported under the existing precedence; sidecars are not deprecated or auto-migrated into a bundle.

```bash
kalanos profiles list
kalanos profiles show numeric-core-v1
kalanos profiles show vision-imitation-v1
kalanos profiles validate acquisition.yaml
```

Validation checks YAML/model structure and any referenced decision policy. It does not load a dataset or certify a binding. Unknown capability or metric names cannot produce a pass: their required evidence will be unavailable at grading time.

## Channel selectors and validation

A per-channel binding, in a bundle's `binding.channels` or a `kalanos-map.yaml` entry, selects a channel by `feature` (the adapter's original `source_field`, matched exactly) and an optional `index`, the zero-based source member index — LeRobot records it from manifest order, other multichannel streams from their adapter's own channel order. A scalar channel with no vector index uses `null`; a one-member LeRobot feature uses `0`; negative indices are rejected.

An optional `name` asserts the expected source name; a mismatch is a configuration error, not a replacement display label. An optional `source_identity` restricts the selector to one resolved dataset root or file URI — without it, the selector matches across the whole run. That scope is a URI/path scope, not a content hash: pin remote sources to a revision, since an unchanged local path does not prove unchanged bytes. A duplicate selector, an overlap between a global and a scoped selector on the same input, a selector that matches nothing, and an unknown taxonomy key are all configuration errors.

A channel's `validations` entries each record `property`, `value`, `validator`, `evidence`, `scope` and an optional `capability` (`validator` and `evidence` cannot be blank), and are active only once their `value` matches the resolved property and their `scope` matches the channel's resolved source identity — see [noise reference evidence](#noise-reference-evidence) below for a worked example. A capability such as `derivatives` or `noise` needs its own scoped evidence: validating `unit` alone does not validate `derivatives`, and evidence scoped to `derivatives` does not validate physical limits or a noise reference. Changing a resolved channel property invalidates its prior validation records; they stay in the report as invalidated, and newly supplied evidence can establish the replacement interpretation.

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

Capture origin alone is insufficient: all streams must compute the acquisition metrics. Built-in generic readers do not assert producer capture provenance, so this requirement is expected to remain unknown unless an adapter has justified producer evidence and usable timestamps. Metadata tier keeps the same requirements and therefore leaves skipped numeric evaluation unknown.

To require torque measurement as well, add `motion.p99_torque` to `required_metrics`. An episode must contain eligible torque subjects and evaluate all of them. The value may remain report-only; this requirement does not define a safe torque limit or authorize statistical blocking.

## Vision requirements

```yaml
schema_version: 1
requirements:
  id: vision-imitation-v1
execution:
  tier: full
```

This preset requires video quality: every declared frame of every camera examined. At the full tier the vision metrics read every frame, so a run can satisfy it; at the standard tier they read a sample, and it remains unknown. Camera metadata and numeric analysis do not satisfy that requirement. Choosing a weaker profile changes the question asked and must not be described as completing the original audit.

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

The policy path resolves relative to the bundle. Use a complete policy with its `calibration_manifests` list; see [Calibration](DECISIONS.md#calibration). Missing or invalid files fail with exit 2. Do not add a test manifest to a production policy. Run a diagnostic grade first to obtain exact candidate calibration contexts, then validate the final implementation and scope before recording acceptance.

The default gate fails `blocked,unknown`. For unattended training:

```bash
kalanos grade ./recording --profile acquisition.yaml \
  --fail-on blocked,review,unknown --report audit.json
```

## Noise reference evidence

A per-channel binding may include the following `noise_floor` object. These numbers are a configuration illustration, **not a validated reference for your robot**:

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

The binding also needs `quantity`, `representation`, `unit`, feature/index and source scope. Supply separate `validations` for `identity`, `quantity`, `unit` and `noise_floor`, each with `value` equal to the resolved property, the exact `scope` recorded as `source_identity`, `validator`, `evidence`, and `capability: noise`. `identity` is `feature[index]`, including `[None]` for a scalar with no index. The `noise_floor` validation value is the complete object, including explicit null `scale_transform`. First inspect the actual resolved binding; do not guess source identities or copy test attestations.

The reference must use the same native scale and estimator. An override to the floor, unit, scale transform or source invalidates applicability until it is validated again. Native sample rate must match. The validator is responsible for verifying configuration and bandwidth against acquisition records. This evidence makes the reference usable; statistical blocking still needs a separately accepted, exactly matching calibration manifest.

## Deeper diagnostic plans

An optional `diagnostics` bundle section selects work without weakening existing requirements. `sampled_video_quality` requires the requested samples on every camera; `video_quality` requires every declared frame examined. Reducing the tier or budget preserves the requirement and records skipped/unavailable work. `cross_stream_timing`, `action_consistency`, `motion_shape` and `training_windows` can be required capabilities; individual checks can be required with keys such as `diagnostics.timing.camera_state` in `required_metrics`.

Use `requirements.min_pass_windows: {train: 10000}` for an explicit dataset minimum. Configure review thresholds in the referenced policy's `diagnostic_reviews`, not in the plan. See the complete [diagnostic contract](DIAGNOSTICS.md) and [example bundle](examples/diagnostics.yaml).
