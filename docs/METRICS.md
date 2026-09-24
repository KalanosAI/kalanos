# Metrics

The catalogue. This file is the source of truth for what Kalanos computes; `policy.yaml` holds the thresholds and weights that grade it.

This is the design. Today only `timing`, `integrity` and `motion` are implemented; every other metric below is designed and not yet built.

Thresholds marked *to define* are undecided. They must be settled against real recordings rather than filled in with a plausible number: once written down, an invented threshold is indistinguishable from a decided one, and it will be implemented as though it were one.

A threshold marked *candidate* is a proposed value under validation. It may ship in `default.yaml` as the working default, provided the entry there carries a comment marking it unconfirmed; it is not decided until it has been confirmed against recordings. The marker exists so a value being tried out is never mistaken for one that was settled.

## Rules

- A metric declares its requirements and the registry filters. **Unmet requirements produce `not_applicable` with a reason.** Never a raise, never a zero. "This signal is bad" and "this metric could not run here" are different facts and stay different all the way to the report.
- Every result carries `value`, `unit`, `status` and `evidence`: whatever a reader needs to check the claim, such as sample counts, the window where it happened, or a fitted slope.
- `status` is one of `good` / `warning` / `critical` / `report_only` / `not_applicable`.
- Thresholds live in `policy.yaml`, keyed by taxonomy type. Never in code.
- Severity is assigned by scoring, from the policy. A metric returns a measurement.
- Metrics read the canonical seconds `loading` produced. No metric parses timestamps.
- **Rate-based metrics require regular sampling.** A series indexed by event rather than by a steady clock has no meaningful sampling rate, so any rate reported for it would be invented.
- **A metric that would mislead under some condition must detect that condition and return `not_applicable`**, rather than returning a number that happens to be large.

## Levels

| Level | Context |
|---|---|
| `CHANNEL` | one scalar series |
| `STREAM` | one taxonomy-typed signal, which may have several channels |
| `EPISODE` | one recording, including comparisons between its streams |
| `DATASET` | the analysed path |

## Families

A family names the question a metric asks. The taxonomy type already names the signal it reads, so a family named after the signal would say it twice, and a jerk metric applied to a commanded trajectory rather than a measured one would have nowhere to live.

| Family | Asks |
|---|---|
| `timing` | did the clock behave |
| `integrity` | is the signal itself sound |
| `motion` | is the motion physically plausible |
| `consistency` | do signals that should agree, agree |
| `vision` | are the frames usable |
| `coverage` | is the dataset varied enough |
| `calibration` | is what is needed to interpret the data present |
| `annotation` | are the labels sound |
| `schema` | did Kalanos understand the file |

Families are the weight keys in `policy.yaml`, and weights renormalise over the families a recording produced. A dataset with no cameras loses nothing for having no vision metrics.

Note the `coverage` **family** above asks about dataset variety. It is unrelated to the **graded fraction** described under Scoring, which measures how much of a recording was actually graded. The two share a word only.

---

## timing

From the clock alone. Weighted heaviest, because broken timing invalidates everything computed downstream: a model learns that when the robot saw X it did Y from pairs that never co-occurred.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `effective_hz` | STREAM | regular sampling | Hz | Samples per second, from the median gap between timestamps. Median so one long gap cannot move it. | *candidate: within 10% of nominal good, 25% bad*; ungraded where the policy declares no nominal rate |
| `dt_jitter_ms` | STREAM | regular sampling | ms | Standard deviation of the gaps between consecutive timestamps. | *candidate: good < 5 ms at 100 Hz*; one-sided (no bad bound), so it stays report-only |
| `drop_rate` | STREAM | regular sampling | fraction | Fraction of expected samples that never arrived, expected being duration ÷ median gap. | good < 1%, bad > 5% |
| `monotonic_violations` | STREAM | | count | Samples whose timestamp is at or before the previous one. A clock that goes backwards means a reordered or merged log. | *to define* |

---

## integrity

Universal. Computed for every numeric channel whether or not its type was identified, which is why nothing here gates on taxonomy. A mystery column that is flatlined must still fail loudly.

| Metric | Level | Unit | Definition | Threshold |
|---|---|---|---|---|
| `missing_pct` | CHANNEL | % | Share of null or NaN values. | *candidate: good < 0.5%, bad > 5%* |
| `flatline_pct` | CHANNEL | % | Percentage of time the value does not change, which is a stuck sensor. Evidence carries the longest run. | *candidate: good < 1%, bad > 20%*; reward signals exempt (they flatline legitimately until success) |
| `spike_pct` | CHANNEL | % | Percentage of samples beyond 6σ of a local window. | *candidate: good < 0.1%, bad > 2%* (default and joint velocity) |
| `drift` | CHANNEL | unit/min | Slow trend where the signal should be stationary. Evidence carries the fitted slope and r². | motor and joint temperature: Δ < 15 °C over an episode; other types graded only where the policy marks the signal stationary |
| `snr_db` | CHANNEL | dB | Signal against the high-frequency noise floor. | *candidate:* joint velocity good > 20 / bad < 10 dB; joint acceleration good > 12 / bad < 6 dB; other types good > 30 / bad < 15 dB |
| `dead_taxel_pct` | STREAM | % | Share of a tactile array's cells that never respond. | good < 2%, *candidate bad > 10%* |
| `hysteresis` | STREAM | fraction | Difference between a tactile cell's loading and unloading response. | good < 5%, *candidate bad > 20%* |

`drift` is not graded by default. A rising temperature is a fault on a servo and expected behaviour on a heating element, and only the policy knows which. Grading it unconditionally produces warnings users learn to ignore. It ships `report_only` in `default.yaml`, so it is excluded from the graded fraction rather than counted as an ungraded gap. Grading it for a signal the policy marks stationary needs per-taxonomy report-only support, which is tracked separately.

The window `spike_pct` takes its 6σ over is *to define*: the threshold is settled but the window is not, and choosing one silently would make a free parameter look like a decision.

The shipped metric carries the window it actually used in `evidence`, as `window_samples`, so the undecided parameter stays visible in every report rather than buried in code.

---

## motion

Whether the recorded motion is physically plausible. These need the taxonomy: jerk is normalised by the motion's own scale so that two robots can be compared, and 90% of maximum only means something once a value is known to be a joint velocity.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `mean_jerk_norm`, `max_abs_jerk` | STREAM | `proprio.joint_position` | normalised | Jerk is the third derivative of position: squared, averaged, and normalised by the motion's own scale. | jerk < 0.1 m/s³, *candidate bad > 0.5* |
| `action_chatter` | STREAM | `proprio.joint_velocity` | normalised | Average step-to-step change in velocity, normalised by its spread. | *to define* |
| `vel_saturation_pct` | CHANNEL | `proprio.joint_velocity` + declared max | % | How often a joint ran above 90% of maximum velocity. A saturated joint clips, so the recording shows something other than what was commanded. | good < 1%; shipped metric measures against the channel's own observed maximum and stays report-only until a declared limit is supplied |
| `limit_proximity_pct` | CHANNEL | `proprio.joint_position` + declared limits | % | Percentage of time a joint spent above 95% of its allowed range. | range adherence 99.8%; shipped metric measures against the channel's own observed range and stays report-only until declared limits are supplied |
| `still_drift` | STREAM | `proprio.joint_position` | unit | How far joint readings wander when the recording ends with the robot holding still. Skipped automatically when it does not end static, because motion is not drift. | *to define* |
| `hf_vibration_ratio` | CHANNEL | `proprio.joint_torque` | fraction | Share of the torque signal's energy above 20 Hz. Deliberate motion lives below about 5 Hz; energy above that is mechanical. | *candidate: good < 0.1, bad > 0.3* |
| `p99_torque`, `max_torque`, `mean_torque` | CHANNEL | `proprio.joint_torque` | N·m | 99th percentile, peak and mean of absolute torque. | **report-only, always.** What counts as high depends entirely on the robot and the task |
| `energy_proxy` | EPISODE | torque + velocity | J (robot units) | Sum of \|torque × velocity\| over time, roughly the mechanical work done. Two runs of one task should land close; an outlier used the robot differently. | report-only |

Any metric with a precondition should skip the way `still_drift` does, by detecting the condition itself instead of relying on the policy to exclude it.

---

## consistency

Whether signals describing the same physical thing agree with each other. These catch a class of fault nothing else here can see: a miscalibrated model, a mislabelled field, a unit error, a controller not doing what it was told.

Distinct from `cross_stream` sync, which asks whether two clocks agree. These ask whether two signals agree, given that the clocks already do.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `fk_residual` | EPISODE | `proprio.ee_pose` + `proprio.joint_position` | m | Distance between the recorded end-effector pose and the one forward kinematics predicts from the joint angles. | good < 1 mm |
| `current_torque_r` | EPISODE | `proprio.motor_current` + `proprio.joint_torque` | correlation | Correlation between motor current and reported torque, which should track closely. | good > 0.9 |
| `command_tracking_error` | EPISODE | an `action.*` command + its measured counterpart | fraction | Error between what was commanded and what was achieved. | good < 5% |
| `jacobian_residual` | EPISODE | `proprio.ee_twist` + `proprio.joint_velocity` | fraction | Whether the end-effector twist matches what the joint velocities imply. | good < 5% |

---

## vision

Frame-level quality for camera streams. Every metric here runs on a stratified sample of frames and reports how many it looked at, because decoding everything to grade a dataset is not a plausible thing to do.

Without a decoder installed, all of these return `not_applicable` with a reason and the rest of the grade completes. A missing optional dependency must not turn a gradeable dataset into an error.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `frozen_frame_pct` | STREAM | an image or video stream | % | Share of frames identical to their predecessor. A camera that stopped updating looks fine in every other metric. | *to define* |
| `blur_score` | STREAM | decode | variance | Variance of the Laplacian over sampled frames; lower means blurrier. Higher is better. | *to define* |
| `exposure_bad_pct` | STREAM | decode | % | Share of sampled frames clipped at black or white. | *to define* |
| `frame_count_vs_timebase` | STREAM | | fraction | Deviation between the number of frames present and the number the timebase implies. | *to define* |

`frozen_frame_pct` needs no full decode: hashing a strided subsample of each frame finds duplicates cheaply.

---

## cross_stream

Whether streams recorded together agree about when things happened. This family only becomes possible with episodes. While each file was analysed independently there was nothing to compare.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `sync_error_ms` | EPISODE | two streams with real clocks | ms | Mean time gap between each sample of one stream and the nearest sample of another. | good < 20 ms |
| `multi_camera_skew_ms` | EPISODE | two or more image streams | ms | Spread of capture times across cameras that should be simultaneous. | *to define* |
| `finger_sync_frames` | EPISODE | multi-finger joint streams | frames | Skew between fingers of one hand. | good < 1 frame |

A stream whose timestamps were synthesised from a declared rate rather than recorded cannot support a latency claim, since the answer would restate the rate. Such a finding is reported with reduced confidence.

---

## coverage

Whether the dataset is varied enough to train on. Five hundred near-identical demonstrations teach a model less than fifty varied ones. The only family that asks about the dataset rather than about a recording.

| Metric | Level | Unit | Definition | Threshold |
|---|---|---|---|---|
| `anomaly_spike_pct` | EPISODE | % | Compress the state signals and measure how badly each moment reconstructs; report the share that do not fit the recording's own dominant patterns. | *to define* |
| `coverage_entropy` | DATASET | nats | Project states onto their two most informative directions and take the entropy of the histogram. Low means the same motion repeated. | *to define*; report-only meanwhile |

`anomaly_spike_pct` finds candidate incidents without labels: collisions, sensor glitches, dropped objects. It is a cheap linear stand-in. A learned detector is a later question and would run as a background job rather than in the interactive path.

---

## calibration and annotation

Two families the taxonomy supports and no metric yet occupies.

`calibration` asks whether what is needed to interpret the data is present: frame conventions documented, camera intrinsics recorded, filter cutoffs stated, extrinsics available. The reference type catalogue names these repeatedly as quality concerns, and they are checks on metadata rather than on signals.

`annotation` asks whether labels are sound: inter-rater agreement, class balance, temporal boundary precision, leakage across train and test splits. Relevant only to datasets that carry labels at all.

Both are declared so that the family list is stable and the policy schema does not change when they are filled. Neither has a metric yet, and `missing_family: skip` means their absence costs nothing.

---

## Scoring

Each metric is graded against a **band**: the `good` and `bad` threshold pair for that metric, keyed by taxonomy type in `policy.yaml`, with a `default` band as the fallback for a type the policy does not name specifically. A value at or beyond `good` scores 100; at or beyond `bad`, 0; between them it interpolates linearly to a number from 0 to 100.

A band needs **both bounds** to grade. A band missing a bound, a metric with no band at any tier, and a metric the policy marks `report_only` all stay `report_only`: measured, not judged.

### Band values at a glance

The working defaults for the graded metrics, gathered in one place for orientation. **The authoritative, current values live in `src/kalanos/assets/policies/default.yaml` — where this table and that file disagree, the file wins.** Every *candidate* value is unconfirmed against real recordings (see the candidate note at the top of this file); only `drop_rate` is decided.

| Metric | Applies to | good | bad | Status |
|---|---|---|---|---|
| `drop_rate` | default | 1% | 5% | decided |
| `effective_hz` | default (deviation from nominal) | 10% | 25% | candidate |
| `dt_jitter_ms` | default | 5 ms | — | candidate · one-sided, so report-only |
| `missing_pct` | default | 0.5% | 5% | candidate |
| `flatline_pct` | default | 1% | 20% | candidate · reward signals exempt |
| `spike_pct` | default & joint_velocity | 0.1% | 2% | candidate |
| `snr_db` | default | 30 dB | 15 dB | candidate |
| `snr_db` | joint_velocity | 20 dB | 10 dB | candidate |
| `snr_db` | joint_acceleration | 12 dB | 6 dB | candidate |
| `dead_taxel_pct` | default | 2% | 10% | candidate |
| `hysteresis` | default | 0.05 | 0.20 | candidate |
| `mean_jerk_norm` | default | 0.1 | 0.5 | candidate · low confidence |
| `max_abs_jerk` | default | 0.1 | 0.5 | candidate · low confidence |
| `hf_vibration_ratio` | default | 0.1 | 0.3 | candidate · low confidence |

Metrics not listed are either `report_only` (`drift`, `p99_torque`, `max_torque`, `mean_torque`, `energy_proxy`, and `vel_saturation_pct` / `limit_proximity_pct` until a declared limit is wired in) or still *to define* (`action_chatter`, `still_drift`, `monotonic_violations`) — none of them grade.

Family scores are weighted means of their metrics; the overall score is a weighted mean of family scores, renormalised over the families that ran, minus a capped penalty per distinct failing metric.

| Score | Grade | Meaning |
|---|---|---|
| ≥ 90 | A | Collect more of exactly this |
| 80–89 | B | Train on it; note the warnings |
| 70–79 | C | Usable with caveats |
| 60–69 | D | Fix the failures, then re-record or filter |
| < 60 | F | Do not train; debug the pipeline |

Every score also carries a **graded fraction**: of the metrics that could have been graded at a node, how many actually were. A metric that resolved to a grade counts toward it; a metric left `report_only` for want of a band counts against it; `not_applicable` results and metrics that are `report_only` by design (`p99_torque`, `energy_proxy`, `drift`) are excluded from the denominator entirely, since a metric that was never meant to grade here is not a gap. An A over a graded fraction of 0.15 is a different claim from an A over 0.95, and the report carries the number so a reader can tell them apart. (Not to be confused with the `coverage` family; the policy key drafted as `min_coverage` is a candidate for renaming to `min_graded_fraction` to keep the two apart.)

`train_ready` requires all of: `score ≥ 70`; no metric graded `critical`; the graded fraction at or above the policy's floor; and every family the policy names as required having graded at least one metric. A single `critical` channel is enough to withhold it, regardless of the mean. Until the graded-fraction work ships, `train_ready` is `score ≥ 70` alone.

Weights are *to define*. Only their ordering is settled, with timing weighing most. `report_only` and `not_applicable` results are excluded from the denominator, because a metric that could not run must not silently cost points.

## Adding a metric

1. Write the function with its `@metric` decorator — `level=` and `family=`, both required — in the family's module under `src/kalanos/analysis/metrics/`. A `family=` outside the nine above fails at import.
2. Add its `good`, `bad` and weight to `src/kalanos/assets/policies/default.yaml`, keyed `family.metric_name`; that key's prefix is what makes it a member of the family. A bound that is not yet settled stays *to define* here and absent there. A *candidate* bound may ship, but only with a comment marking it unconfirmed — never as though it were decided.
3. Add a contract test naming the defect it fires on, using an injector.
4. Add a row above.

Out of tree, the same function ships as a package declaring a `kalanos.metrics` entry point. That group names the *module*, not the function: importing it is what runs the decoration. `kalanos new metric` writes one.

A metric added without those four steps gives the tool an opinion nobody agreed to. If the catalogue looks like it is missing something, raise it and file an issue rather than closing the gap in a table.
