# Metrics

The catalogue. This file is the source of truth for what Kalanos computes; `policy.yaml` holds the thresholds and weights that grade it.

The registered metric catalogue below includes implemented core metrics and
older design proposals. The explicit **0.7.0 diagnostic plan** now supplies
stream-pair timing, command response, sampled vision, dimensionless motion,
training-window and cohort execution. Its definitions, prerequisites and policy
behavior are authoritative in [DIAGNOSTICS-0.7.0.md](DIAGNOSTICS-0.7.0.md).
Proposed thresholds and formulas in the historical tables do not configure that
runner. FK/Jacobian and learned transition checks remain unimplemented.

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
| `effective_hz` | STREAM | regular sampling + producer capture evidence | Hz | Capture samples per second, from the median gap between timestamps. Median so one long gap cannot move it. | *candidate: within 10% of nominal good, 25% bad*; ungraded where the policy declares no nominal rate |
| `dt_jitter_ms` | STREAM | regular sampling + producer capture evidence | ms | Standard deviation of adjacent capture-time gaps. | *candidate: good < 5 ms at 100 Hz*; one-sided (no bad bound), so it stays report-only |
| `drop_rate` | STREAM | regular sampling + producer capture evidence | fraction | Missing-sample estimate against the observed cadence; expected count is duration ÷ median gap + 1. | good < 1%, bad > 5% |
| `monotonic_violations` | STREAM | | count | Samples whose timestamp is at or before the previous one. A clock that goes backwards means a reordered or merged log; one that repeats usually means a dropped simulation step. | *to define*; ships report-only — measured in every report, graded once a band is settled against real recordings. Evidence carries the fraction of steps, the repeated and backwards counts and the first offending sample |

**Clock provenance (R07-04).** `effective_hz`, `dt_jitter_ms` and `drop_rate` require `ClockInfo.origin=capture` and `origin_evidence=producer`. Unknown, inferred, generated, receive, publish, log, presentation and simulation clocks cannot certify acquisition timing. An evenly spaced producer-declared capture clock is measurable; uniformity alone is no longer used to veto it. LeRobot's match to `frame_index / fps` is explicitly inferred generation, including when indices are skipped or reordered. HDF5's adapter-created grid is known generation.

| Recorded-axis metric | Meaning | Decision behavior |
| --- | --- | --- |
| `recorded_hz` | Reciprocal median adjacent interval | Report-only by default |
| `recorded_dt_spread_ms` | Sample standard deviation of recorded intervals | Report-only by default; not sensor jitter |
| `recorded_drop_estimate` | Estimated holes relative to recorded median cadence | Report-only by default; not verified sensor frame loss |

Recorded cadence requires a regular, strictly increasing axis with known units and no invalid timestamps. Spread needs five valid samples. `monotonic_violations` remains independent of clock origin and regularity. It counts adjacent repeated/backwards steps in source order and includes source row addresses. Null/NaN/Inf values break adjacency; they are never removed and bridged. Native integer ticks are subtracted before conversion to seconds, preserving small intervals at large epochs. These results establish neither cross-stream alignment nor calibration approval. See [R07-04](R07-04.md) for migration and acceptance tests.

---

## integrity

Universal. Computed for every numeric channel whether or not its type was identified, which is why nothing here gates on taxonomy. A mystery column that is flatlined still produces a diagnostic; automatic exclusion requires the appropriate authority.

| Metric | Level | Unit | Definition | Threshold |
|---|---|---|---|---|
| `missing_pct` | CHANNEL | % | Share of null or NaN values. | *candidate: good < 0.5%, bad > 5%* |
| `flatline_pct` | CHANNEL | % | Percentage of finite adjacent pairs that do not change. This alone does not establish a stuck sensor. Evidence carries source-row intervals and the longest run. | *candidate: good < 50%, bad > 90%* (revised against real 50 Hz teleoperation, where joints hold still for 20–45% of an episode; a stuck sensor is unchanged for nearly all of it). A channel that **never changes at all** in an episode grades as a **warning** at the middle of the band, never critical: it is an unused joint or a disconnected sensor, which the data cannot tell apart, and it does not show the episode is worse than the others (on `lerobot/berkeley_fanuc_manipulation` a state dimension constant in 158 of 415 episodes capped the dataset at D). A channel that moves and then **freezes partway** is the stuck sensor, and stays critical; reward signals exempt (they flatline legitimately until success), and action commands exempt (not sensors: a pause, an idle arm or an unused action dimension holds a command still legitimately — on real LeRobot datasets action channels hit this band in nearly every episode of five datasets out of six) |
| `spike_pct` | CHANNEL | % | Percentage of samples beyond 6σ of a local window. | *candidate: good < 0.1%, bad > 2%* (default and joint velocity) |
| `drift` | CHANNEL | unit/min | Slow trend where the signal should be stationary. Evidence carries the fitted slope and r². | motor and joint temperature: Δ < 15 °C over an episode; other types graded only where the policy marks the signal stationary |
| `snr_db` | CHANNEL | dB | Signal against the high-frequency noise floor: the variance of a 5-sample moving average against what it removes. **Not applicable below about 45 Hz**, where 5 samples span more than 0.1 s and smooth away real motion, which would then read as noise (on real 5-15 Hz LeRobot datasets the ratio tracked the sampling rate rather than the robot). At 50 Hz and above the window is unchanged. | *candidate:* joint velocity good > 20 / bad < 10 dB; joint acceleration good > 12 / bad < 6 dB; other types good > 30 / bad < 15 dB |
| `dead_taxel_pct` | STREAM | % | Share of a tactile array's cells that never respond. | good < 2%, *candidate bad > 10%* |
| `hysteresis` | STREAM | fraction | Difference between a tactile cell's loading and unloading response. | good < 5%, *candidate bad > 20%* |

`drift` is not graded by default. A rising temperature is a fault on a servo and expected behaviour on a heating element, and only the policy knows which. Grading it unconditionally produces warnings users learn to ignore. It ships `report_only` in `default.yaml`, so it is excluded from the graded fraction rather than counted as an ungraded gap. Grading it for a signal the policy marks stationary needs per-taxonomy report-only support, which is tracked separately.

The window `spike_pct` takes its 6σ over is *to define*: the threshold is settled but the window is not, and choosing one silently would make a free parameter look like a decision.

The shipped metric carries the window it actually used in `evidence`, as `window_samples`, so the undecided parameter stays visible in every report rather than buried in code.

**Switch channels.** A channel that takes exactly two values over an episode — a gripper open/close command, a done flag, a contact bit — is a switch, not a sampled signal. Holding one value for most of the episode is the switch doing its job, so `flatline_pct`, `spike_pct` and `snr_db` return `not_applicable` for it, with the reason in `evidence`. A channel stuck on a *single* value is still measured, since that can be a dead sensor, and a switch with a glitch on it (a third value) is measured again.

---

## motion

Whether the recorded motion is physically plausible. These need the taxonomy: jerk is normalised by the motion's own scale so that two robots can be compared, and 90% of maximum only means something once a value is known to be a joint velocity.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `mean_jerk_norm`, `max_abs_jerk` | STREAM | `proprio.joint_position` | normalised | Jerk is the third derivative of position: squared, averaged, and normalised by the motion's own scale. | **report-only** until dimensionless — see below; *candidate band kept on file: good < 0.1, bad > 0.5* |
| `action_chatter` | STREAM | `proprio.joint_velocity` | normalised | Average step-to-step change in velocity, normalised by its spread. | *to define* |
| `vel_saturation_pct` | CHANNEL | `proprio.joint_velocity` + declared max | % | How often a joint ran above 90% of maximum velocity. A saturated joint clips, so the recording shows something other than what was commanded. | good < 1%; shipped metric measures against the channel's own observed maximum and stays report-only until a declared limit is supplied |
| `limit_proximity_pct` | CHANNEL | `proprio.joint_position` + declared limits | % | Percentage of time a joint spent above 95% of its allowed range. | range adherence 99.8%; shipped metric measures against the channel's own observed range and stays report-only until declared limits are supplied |
| `still_drift` | STREAM | `proprio.joint_position` | unit | How far joint readings wander when the recording ends with the robot holding still. Skipped automatically when it does not end static, because motion is not drift. | *to define* |
| `hf_vibration_ratio` | CHANNEL | `proprio.joint_torque` | fraction | Share of the torque signal's energy above 20 Hz. Deliberate motion lives below about 5 Hz; energy above that is mechanical. | *candidate: good < 0.1, bad > 0.3* |
| `p99_torque`, `max_torque`, `mean_torque` | CHANNEL | `proprio.joint_torque` | N·m | 99th percentile, peak and mean of absolute torque. | **report-only, always.** What counts as high depends entirely on the robot and the task |
| `energy_proxy` | EPISODE | torque + velocity | J (robot units) | Sum of \|torque × velocity\| over time, roughly the mechanical work done. Two runs of one task should land close; an outlier used the robot differently. | report-only |

### Jerk metrics are report-only

`mean_jerk_norm` and `max_abs_jerk` divide each channel by its own standard deviation before taking the third difference. That removes amplitude but not time: the result still carries units of 1/s³, so it grows with the cube of how fast the robot moves. A perfectly smooth joint following a sine at frequency *f* measures about √2·(2π*f*)³ — roughly 0.35 at 0.1 Hz, 44 at 0.5 Hz and 180 at 0.8 Hz — against a candidate *bad* bound of 0.5. Graded against that band, any ordinary manipulation motion scores 0.

So both metrics ship `report_only`: measured and shown in every report, not graded. The candidate band stays in `default.yaml` so the numbers are not lost. They return to grading once the measure is made dimensionless (normalised by the motion's duration as well as its amplitude, as log dimensionless jerk is) and a band is set against real recordings.

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

Two families the taxonomy supports. `annotation` holds its first metric; `calibration` has none yet.

| Metric | Family | Level | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `task_instruction_missing` | annotation | EPISODE | flag | The episode carries no task instruction, or only an empty or whitespace one. **Not applicable** when no episode in the dataset carries one: a dataset never annotated with language is not missing anything. | *to define* — the share of episodes that may lack one, and whether the default policy grades it at all (a missing instruction is fatal for a vision-language-action model and irrelevant for plain behaviour cloning); ships report-only |

Instructions are read by the adapters — LeRobot v2/v3 from each episode's task list in `meta/`, HDF5 from an episode group's `task`, `language_instruction`, `instruction`, `lang` or `language` attribute — and every episode's instructions are in the report (`tasks`), so a reader sees what each was told. What counts as a placeholder instruction beyond an empty one is left for real recordings to show.

`calibration` asks whether what is needed to interpret the data is present: frame conventions documented, camera intrinsics recorded, filter cutoffs stated, extrinsics available. The reference type catalogue names these repeatedly as quality concerns, and they are checks on metadata rather than on signals.

`annotation` asks whether labels are sound: inter-rater agreement, class balance, temporal boundary precision, leakage across train and test splits. Relevant only to datasets that carry labels at all.

Both are declared so that the family list is stable and the policy schema does not change when they are filled. `missing_family: skip` means a family with no graded metric costs nothing.

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
| `flatline_pct` | default | 50% | 90% | candidate, revised against real 50 Hz teleoperation · reward signals and action commands exempt |
| `spike_pct` | default & joint_velocity | 0.1% | 2% | candidate |
| `snr_db` | default | 30 dB | 15 dB | candidate |
| `snr_db` | joint_velocity | 20 dB | 10 dB | candidate |
| `snr_db` | joint_acceleration | 12 dB | 6 dB | candidate |
| `snr_db` | joint_torque | — | — | report-only: motor-current channels read 8–15 dB on real ALOHA arms in every episode |
| `dead_taxel_pct` | default | 2% | 10% | candidate |
| `hysteresis` | default | 0.05 | 0.20 | candidate |
| `mean_jerk_norm` | default | 0.1 | 0.5 | candidate · low confidence · **report-only** (see *Jerk metrics are report-only*) |
| `max_abs_jerk` | default | 0.1 | 0.5 | candidate · low confidence · **report-only** (see *Jerk metrics are report-only*) |
| `hf_vibration_ratio` | default | 0.1 | 0.3 | candidate · low confidence |

Metrics not listed are either `report_only` (`drift`, `p99_torque`, `max_torque`, `mean_torque`, `energy_proxy`, and `vel_saturation_pct` / `limit_proximity_pct` until a declared limit is wired in) or still *to define* (`action_chatter`, `still_drift`, `monotonic_violations`, `task_instruction_missing`) — none of them grade.

Family scores are weighted means of their metrics; the overall score is a weighted mean of family scores, renormalised over the families that ran, minus a capped penalty per distinct failing metric.

### Readiness

The dataset's headline number is **readiness**, from 0 to 100, in the report's `readiness` section:

```text
Readiness is defined only when the declared inventory is complete,
all episodes are evaluated, and every episode is pass or blocked.
A blocked episode contributes 0; a passing episode contributes its quality.
Review or unknown episodes leave readiness undefined.

readiness = sum of contributions / expected episode count
```

Only an authorized blocking consequence blocks an episode. Critical severity alone
does not supply that authorization. `passing_quality` describes passing episodes;
when none pass it is null. Review and required missing evidence cannot be converted
into a high readiness score by averaging the remaining episodes.

Readiness is an index of technical readiness under the evaluated checks, not a prediction of training success.

**Letter grades are deprecated** since 0.6.5. Reports still carry `score.grade` (from the bands A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F below, in `policy.yaml`'s `letters`), `gate.cap` and `gate.uncapped_grade` for compatibility; nothing presents them and they drive no decision. `gate.pruned_grade` and `gate.train_ready_after_pruning` were removed in 0.7.0.

Every score also carries a **graded fraction**: of the metrics that could have been graded at a node, how many actually were. A metric that resolved to a grade counts toward it; a metric left `report_only` for want of a band counts against it; `not_applicable` results and metrics that are `report_only` by design (`p99_torque`, `energy_proxy`, `drift`) are excluded from the denominator entirely, since a metric that was never meant to grade here is not a gap. An A over a graded fraction of 0.15 is a different claim from an A over 0.95, and the report carries the number so a reader can tell them apart. (Not to be confused with the `coverage` family; the policy key drafted as `min_coverage` is a candidate for renaming to `min_graded_fraction` to keep the two apart.)

**`train_ready` is a compatibility field** since schema 7. No score threshold sets it. At episode level it mirrors the episode's `eligibility` (`true` pass, `false` blocked, `null` review or unknown); at every other level it is `null`. Read the eligibility, not the boolean — see `docs/DECISIONS.md`.

### The dataset gate

A mean lets a minority of bad episodes hide: eight glitched episodes in fifty still average to 98. Since schema 7 the gate decides nothing: each episode carries one authoritative `eligibility`, decided once after every metric has run (`kalanos.analysis.scoring.eligibility`), and the gate lists the `blocked` ones in `failing_episodes`, why each blocks, and the mean quality of the rest (`pruned_score`, a description of a candidate selection, not a sufficiency claim).

**An episode is blocked** when a finding carries the consequence `block`. A finding's *severity* is the assessment its bands assigned; its *consequence* is what the policy says that does to eligibility, and the two are recorded separately. A `critical` statistical result is a candidate for blocking and resolves to review without accepted, matching calibration. Two routes justify a block: a **contract** violation (a declared invariant broken with direct evidence) needs no statistical calibration; a **statistical** rule needs a structured accepted manifest matching its detector, thresholds, binding and scope. Calibration enforcement is always on and cannot be disabled. An explicit `consequence: review` remains review.

**Prevalence exempts nothing.** A blocking finding on every episode of one task is reported in `gate.task_traits`, and one on (nearly) every episode of the dataset in `gate.dataset_traits`, so a reader sees the pattern — but the episodes stay blocked. The report cannot tell a recording convention from corruption in every episode; a scoped policy rule may exempt a finding explicitly, the gate never does on its own. (Before schema 7 both traits exempted; grading `lerobot/cmu_stretch` under the old rule turned 135 blocked episodes into 0, which a defect in every episode would also have done.)

**Gripper channels.** A channel the dataset names as a gripper inside a wider vector (ALOHA's `left_gripper` in `observation.state` or `action`) grades under the gripper type for its stream: `proprio.gripper_width` for state, `action.gripper_command` for actions. A gripper holds open or closed for most of an episode and then snaps, which the stuck-sensor check would call stuck and the noise check would call noise (ALOHA's grippers, carrying a battery or a towel, read 90–99% unchanged and −2 to 15 dB), so for grippers both are reported, not graded; the spike check still grades them, so a glitching gripper produces a finding that reviews by default and blocks only with accepted matching calibration. A gripper's motor current keeps its torque type. Names come from the dataset: LeRobot's `info.json` declares them either as a list or nested under a label (`"names": {"motors": [...]}`), and both are read. An unnamed gripper cannot be told from any other channel and grades as one.

*Historical diagnostic evidence*: the former blocking rule was tested on twelve datasets — six synthetic with planted defects, where it caught every planted glitched episode (41 of 41) and blocked no clean one, and six LeRobot hub datasets — then revised against real 50 Hz teleoperation (ALOHA, Unitree H1), but it is unconfirmed against labelled real-world failures.

*Deprecated*: the gate also still writes a letter cap by the share of blocking episodes (up to 5%: none; 15%: B; 30%: C; more: D), `gate.cap` and the capped `score.grade`, for 0.6 compatibility. The 5% allowance is what sets the dataset-trait share above; the cap itself is removed in 0.7.0.

**Every gated report states what readiness rests on** (`gate.coverage` and `gate.summary`): the families graded, the median checks per episode, and every metric that could not observe most of the data, with its reason. A readiness of 100 graded on integrity alone over 13 checks per episode, with timing not observable, says so beside the number. Required capabilities and metrics must be evaluated before an episode can pass. Optional missing capabilities do not invent a score penalty, but remain visible in the separate coverage ledger. Required acquisition timing, for example, stays unknown without usable producer capture evidence.

A dataset graded with `legacy_0_5` gets no gate, reproducing a grade published before 0.6. `language_conditioned` extends the default for datasets that train a vision-language-action model: there `task_instruction_missing` grades, so an episode without its instruction is blocking.

**Context and reference evidence (0.7.0).** `snr_db` is a smooth/residual
diagnostic, not a sensor-health measurement. Known discrete/reward/annotation
signals do not qualify. A validated native-scale residual reference permits a
within-reference result to remain report-only without an SNR penalty. Missing
reference evidence leaves critical candidates at review even if a manifest
matches. See [the 0.7.0 release contract](RELEASE-0.7.0.md) for applicability, sample/window handling and
physical-reference prerequisites.

Weights are *to define*. Only their ordering is settled, with timing weighing most. `report_only` and `not_applicable` results are excluded from the denominator, because a metric that could not run must not silently cost points.

## Adding a metric

1. Write the function with its `@metric` decorator — `level=` and `family=`, both required — in the family's module under `src/kalanos/analysis/metrics/`. A `family=` outside the nine above fails at import.
2. Add its `good`, `bad` and weight to `src/kalanos/assets/policies/default.yaml`, keyed `family.metric_name`; that key's prefix is what makes it a member of the family. A bound that is not yet settled stays *to define* here and absent there. A *candidate* bound may ship, but only with a comment marking it unconfirmed — never as though it were decided.
3. Add a contract test naming the defect it fires on, using an injector.
4. Add a row above.

Out of tree, the same function ships as a package declaring a `kalanos.metrics` entry point. That group names the *module*, not the function: importing it is what runs the decoration. `kalanos new metric` writes one.

A metric added without those four steps gives the tool an opinion nobody agreed to. If the catalogue looks like it is missing something, raise it and file an issue rather than closing the gap in a table.

## Computation coverage and consequences

Metric `availability` records computation separately from graded `status`. A report-only torque statistic is computed and belongs in its eligible torque denominator. Known taxonomy mismatch is not applicable; unresolved taxonomy, insufficient samples or missing prerequisites remain unavailable. Skipped payloads retain declared channels in the denominator. Unexpected computation errors are recorded and cause exit 2.

Flatline evaluation no longer bridges null, NaN or infinite values; this can change its measured fraction. Spike intervals identify scored source samples and their wider 51-sample support. Neither SNR nor spectrum measurements invent intervals.

Severity thresholds still describe measurements. Statistical critical findings require matching accepted calibration to block; otherwise their consequence is review. Existing SNR applicability and noise-floor improvements are deferred to R07-03. No new calibrated-performance claim accompanies these changes.
