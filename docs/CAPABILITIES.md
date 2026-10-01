# Diagnostic capabilities in 0.7.0

This matrix describes implemented and tested scope, not a promise of detector
accuracy or automatic training-data acceptance. See [configuration and evidence
contracts](DIAGNOSTICS-0.7.0.md). Missing prerequisites remain explicit.

| Capability | Initial supported inputs | Evidence / limits | Control coverage |
|---|---|---|---|
| Stream-pair timing | Canonical streams from supported adapters | Exact pair/scope declarations; valid native axes; capture claim requires producer evidence | Offsets, integer epochs, unknown clocks/units, resets, explicit event drift |
| Command response | Eager numeric channels, explicit absolute/delta/rate semantics | Validated units/frames/identity; declared response horizon; no fitted repair | Delay, delta baseline, property mismatch, review-only gate |
| Dimensionless motion | Validated continuous/angle position | Finite contiguous regular segments; explicit angle period; limit evidence | Time/amplitude scaling, quiet hold, invalid samples, local limit support |
| Visual audit | LeRobot `VideoPayload`, including shared file segments; compatible custom bounded payloads | Optional PyAV; sample and decoder budgets distinct; no eager fallback | Real encoded shared-file segment, budget/pixel limits, missing decoder, partial failure, static scene |
| Training windows | Eager numeric/Boolean modalities; bounded decoded video evidence | Declared grid, pairing, no interpolation/padding; missing media evidence unknown | Boundaries, null context, budgets, required decoded frames, sufficiency |
| Cohort analysis | Explicit compatible task/robot cohorts on one source grid | Fixed normalization/binning, finite-row disclosure, comparison budgets | Duplicates, dimensionality, raw/pass split, corruption, reviewed phases, session uncertainty |
| Validation summary | Externally labelled episode/session outcomes | Draft only, no accepted manifest or independence inference | Duplicate/split leakage, review versus blocking counts |

All new diagnostics are optional. Default measurements are report-only. Explicit
policy thresholds may request review; these diagnostics cannot automatically
block. Existing calibrated core metrics keep their established behavior.

Generic embedded-image decoding, physical FK/Jacobian models, learned transition
residuals, automatic timestamp/channel repair and automatic data removal are not
implemented by this patch. Numerical diagnostics do not certify visual quality.
