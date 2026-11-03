# Metrics

The catalogue. This file is the source of truth for what Kalanos computes; `policy.yaml` holds the thresholds and weights that grade it.

The registered metric catalogue below includes implemented core metrics and older design proposals. The explicit **0.7.0 diagnostic plan** now supplies stream-pair timing, command response, sampled vision, dimensionless motion, training-window and cohort execution. Its definitions, prerequisites and policy behavior are authoritative in [DIAGNOSTICS.md](DIAGNOSTICS.md). Proposed thresholds and formulas in the historical tables do not configure that runner. FK/Jacobian and learned transition checks remain unimplemented.

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

From the clock alone. Broken timing invalidates everything computed downstream: a model learns that when the robot saw X it did Y from pairs that never co-occurred.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `effective_hz` | STREAM | regular sampling + producer capture evidence | Hz | Capture samples per second, from the median gap between timestamps. Median so one long gap cannot move it. | *candidate: within 10% of nominal good, 25% bad*; ungraded where the policy declares no nominal rate |
| `dt_jitter_ms` | STREAM | regular sampling + producer capture evidence | ms | Standard deviation of adjacent capture-time gaps. | *candidate: good < 5 ms at 100 Hz*; one-sided (no bad bound), so it stays report-only |
| `drop_rate` | STREAM | regular sampling + producer capture evidence | fraction | Missing-sample estimate against the observed cadence; expected count is duration ÷ median gap + 1. | good < 1%, bad > 5% |
| `monotonic_violations` | STREAM | | count | Samples whose timestamp is at or before the previous one. A clock that goes backwards means a reordered or merged log; one that repeats usually means a dropped simulation step. | *to define*; ships report-only — measured in every report, graded once a band is settled against real recordings. Evidence carries the fraction of steps, the repeated and backwards counts and the first offending sample |

**Clock provenance.** `effective_hz`, `dt_jitter_ms` and `drop_rate` require `ClockInfo.origin=capture` and `origin_evidence=producer`. Unknown, inferred, generated, receive, publish, log, presentation and simulation clocks cannot certify acquisition timing. An evenly spaced producer-declared capture clock is measurable; uniformity alone is no longer used to veto it. LeRobot's match to `frame_index / fps` is explicitly inferred generation, including when indices are skipped or reordered; HDF5's adapter-created grid is known generation, and an existing damaged time column is preserved rather than repaired. MCAP uses a complete header time column with unknown origin where one exists, otherwise a distinguishable nonzero publish-time column, otherwise log time, never mixing these fields row by row; a header timestamp alone does not certify capture time. A camera stream backed by LeRobot metadata retains that metadata clock — decoded media PTS are not inspected.

| Recorded-axis metric | Meaning | Decision behavior |
| --- | --- | --- |
| `recorded_hz` | Reciprocal median adjacent interval | Report-only by default |
| `recorded_dt_spread_ms` | Sample standard deviation of recorded intervals | Report-only by default; not sensor jitter |
| `recorded_drop_estimate` | Estimated holes relative to recorded median cadence | Report-only by default; not verified sensor frame loss |

Recorded cadence requires a regular, strictly increasing axis with known units and no invalid timestamps. Spread needs five valid samples. `monotonic_violations` remains independent of clock origin and regularity. It counts adjacent repeated/backwards steps in source order and includes source row addresses. Null/NaN/Inf values break adjacency; they are never removed and bridged. Native integer ticks are subtracted before conversion to seconds, preserving small intervals at large epochs. These results establish neither cross-stream alignment nor calibration approval. A reordered third-party stream needs a source-row index map before any of these checks can run; without one they abstain.

Reports carry `clock_info` and `source_order` for every stream, including metadata-only runs, so the origin, its evidence, the source field, native unit/dtype and any conversions are visible even where nothing was graded. A shared numeric epoch between two streams does not by itself establish clock synchronization.

---

## integrity

Universal. Computed for every numeric channel whether or not its type was identified, which is why nothing here gates on taxonomy. A mystery column that is flatlined still produces a diagnostic; automatic exclusion requires the appropriate authority.

| Metric | Level | Unit | Definition | Threshold |
|---|---|---|---|---|
| `missing_pct` | CHANNEL | % | Share of null or NaN values. Boolean-dtype channels count toward this check as well as numeric ones; Boolean data stays outside SNR regardless. | *candidate: good < 0.5%, bad > 5%* |
| `flatline_pct` | CHANNEL | % | Percentage of finite adjacent pairs that do not change. This alone does not establish a stuck sensor. Evidence carries source-row intervals and the longest run. | *candidate: good < 50%, bad > 90%* (revised against real 50 Hz teleoperation, where joints hold still for 20–45% of an episode; a stuck sensor is unchanged for nearly all of it). A channel that **never changes at all** in an episode grades as a **warning** at the middle of the band, never critical: it is an unused joint or a disconnected sensor, which the data cannot tell apart, and it does not show the episode is worse than the others (on `lerobot/berkeley_fanuc_manipulation` a state dimension constant in 158 of 415 episodes capped the dataset at D). A channel that moves and then **freezes partway** is the stuck sensor, and stays critical; reward signals exempt (they flatline legitimately until success), and action commands exempt (not sensors: a pause, an idle arm or an unused action dimension holds a command still legitimately — on real LeRobot datasets action channels hit this band in nearly every episode of five datasets out of six) |
| `spike_pct` | CHANNEL | % | Percentage of samples beyond 6σ of a local window. | *candidate: good < 0.1%, bad > 2%* (default and joint velocity); frame rewards exempt (a computed reward steps up and down as the task progresses, which reads as spikes) |
| `drift` | CHANNEL | unit/min | Slow trend where the signal should be stationary. Evidence carries the fitted slope and r². | motor and joint temperature: Δ < 15 °C over an episode; other types graded only where the policy marks the signal stationary |
| `snr_db` | CHANNEL | dB | Signal against the high-frequency noise floor: the variance of a 5-sample moving average against what it removes. **Not applicable below about 45 Hz**, where 5 samples span more than 0.1 s and smooth away real motion, which would then read as noise (on real 5-15 Hz LeRobot datasets the ratio tracked the sampling rate rather than the robot). At 50 Hz and above the window is unchanged. | *candidate:* joint velocity good > 20 / bad < 10 dB; joint acceleration good > 12 / bad < 6 dB; other types good > 30 / bad < 15 dB |
| `dead_taxel_pct` | STREAM | % | Share of a tactile array's cells that never respond. | good < 2%, *candidate bad > 10%* |
| `hysteresis` | STREAM | fraction | Difference between a tactile cell's loading and unloading response. | good < 5%, *candidate bad > 20%* |

`drift` is not graded by default. A rising temperature is a fault on a servo and expected behaviour on a heating element, and only the policy knows which. Grading it unconditionally produces warnings users learn to ignore. It ships `report_only` in `default.yaml`, so it is excluded from the graded fraction rather than counted as an ungraded gap. Grading it for a signal the policy marks stationary needs per-taxonomy report-only support, which is tracked separately.

The window `spike_pct` takes its 6σ over is *to define*: the threshold is settled but the window is not, and choosing one silently would make a free parameter look like a decision.

The shipped metric carries the window it actually used in `evidence`, as `window_samples`, so the undecided parameter stays visible in every report rather than buried in code.

**Switch channels.** A channel that takes exactly two values over an episode — a gripper open/close command, a done flag, a contact bit — is a switch, not a sampled signal. Holding one value for most of the episode is the switch doing its job, so `flatline_pct`, `spike_pct` and `snr_db` return `not_applicable` for it, with the reason in `evidence`. A channel stuck on a *single* value is still measured, since that can be a dead sensor, and a switch with a glitch on it (a third value) is measured again.

**Noise-floor reference.** In addition to the switch-channel exemption above, `snr_db` excludes known reward, annotation and metadata signals and channels declared binary or discrete; its evidence retains the component signal and residual standard deviations, the smoothing window, and the valid and invalid sample counts. A channel's `ChannelBinding.noise_floor` can supply a residual standard-deviation reference in the channel's native scale, validated against the current binding's identity, quantity, unit and sample rate (see [noise reference evidence](PROFILES.md#noise-reference-evidence) for the binding shape and validation requirements). Within a validated reference, the result stays report-only with no SNR penalty: a quiet robot hold does not read as a noise fault when its measured residual sits inside the verified reference, while a trace with real corruption keeps its residual evidence visible — a within-reference result is not proof of sensor health, and other checks keep their own findings. Without a usable reference the ratio stays diagnostic; above the reference, a critical result stays at review until an accepted calibration manifest matches the detector, thresholds, binding and operating scope (see *Calibration* in [docs/DECISIONS.md](DECISIONS.md)) — a matching manifest alone cannot promote an unassessed ratio to a statistical block.

### Duplicate episodes

After every episode is graded, the numeric payloads of every pair of episodes are compared, so a dataset assembled from several sources is checked for holding the same episode twice. A copy carries its state and action data over, while a re-run of the same task never repeats them, which camera footage alone cannot tell apart: a re-encoded copy no longer hashes like its original, and a fixed camera over an unchanged scene makes different episodes look alike. The pass reads the payloads the channel metrics already fetched and decodes no video. Its rules are candidates, not calibrated:

- **Exact.** An episode is flagged `integrity.duplicate_episode` when every numeric stream it holds has the same channels, row count and SHA-256 over its values cast to float64 as another episode's. The finding is critical and requests review under every scope, since equal payloads are direct evidence of a copy.
- **Near.** An episode is flagged `integrity.near_duplicate_episode` when, at the same streams and channels, its trajectories lie within a normalized RMSE of 0.02 of another episode's, and its streams' row counts differ by at most 10%. Each channel is resampled to 32 points over its relative row position, scaled by its standard deviation over every row of the compared episodes, and the RMSE is taken over every point and channel. Since that standard deviation is pooled over the dataset, the same two episodes can measure a different distance in another dataset. The finding is a `report_only` warning.

An episode whose every channel is constant takes no part, since two such episodes pair whatever they recorded. Index columns (LeRobot's `index`, `frame_index`, `episode_index`, `task_index` and `timestamp`) are not channels, so a copy's payload is free of the indices that differ between copies; an adapter that exposes a per-episode index or absolute time as a channel breaks both rules for its data. Video takes no part in the comparison; which episodes also repeat a camera's footage exactly (`vision.repeated_footage_vs_camera`) is in `Report.cameras[].repeated_footage`. `Report.duplicates` holds each exact group and each near pair with its distance, naming episodes by their index in `Report.episodes`; whether a pair's footage also repeats is read from `Report.cameras[].repeated_footage`. Episodes that hash identically form a group, stored once, and every exact pair is against the group's first episode, so k copies give k - 1 exact pairs rather than one for every two of them. A near pair joins the first episodes of two groups (a lone episode is a group of one), and stands for every copy in either group. Every copy still carries both findings, and neither names a partner, since `Report.duplicates` holds the pairs: an exact finding's evidence is empty, and a near finding gives the limit and the distance to its nearest partner in `max_nrmse` and `nearest_nrmse`. A finding's value counts every episode it duplicates, the copies a partner stands for included, and only a near finding gives a distance, since an exact partner is always 0 apart. The near tier needs numpy (the `numeric` or `video` extra); without it only exact duplicates are found.

`Report.duplicates` stores each near pair once, under its earlier episode, as the partner's index and the distance to 4 significant digits, and lists every near pair up to 1.5 million. Past that, it keeps a near pair only when it is among the 5 nearest of either episode, so the list stops growing with the square of the episode count, and `Report.near_duplicates.pairs_listed` falls short of `pairs_near`. On `lerobot/usc_cloth_sim` below, its 487,458 near pairs take 6.2 MB, about 13 bytes a pair, so the limit holds about 20 MB.

**Saturation.** When more than half the pairs compared are near, and there are more near pairs than episodes compared, the near tier is saturated: the limit does not separate copies from re-runs in that dataset. The second condition keeps a dataset of two or three episodes holding one copy from qualifying. Episodes and pairs are counted among the first episodes of each exact group, which are what the near search compares. A saturated dataset gets no near finding, and `Report.near_duplicates` says so once, with the pair counts and the median distance from each episode to its nearest partner. It gets one `near_duplicates_saturated` entry in `Report.dataset_warnings` in their place, stating how many episodes have a near-identical partner and the share of compared pairs that are near, which `kalanos inspect` and the HTML report show. For that dataset, treat the near tier as uninformative: the exact tier still applies, and the near pairs in `Report.duplicates` show which episodes lie closest. `lerobot/usc_cloth_sim` at `6c4fb6acf8a6db8c4d0692b318023272c4967b5f`, retrieved 2026-10-22 and graded through `kalanos grade`, is saturated: of 497,503 pairs among 998 compared episodes, 487,458 (98.0%) are near, and the median nearest distance is 0.0000058. Its report still lists all 487,458 near pairs, and none of its 1,000 near findings.

Measured on five LeRobot datasets retrieved 2026-10-13: `lerobot/pusht` at `7628202a2180972f291ba1bc6723834921e72c19`, `lerobot/cmu_stretch` at `d84574233973ea9a292babd008256d2290a3f96e`, `lerobot/aloha_static_towel` at `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff`, `lerobot/aloha_sim_transfer_cube_human` at `6a43d500f101255823a9d2b9dc244eeb01a2cd31` and `lerobot/berkeley_autolab_ur5` at `c4e26a697fc4c04776b0558f83e563d14be0109f`. Every non-index numeric column was read from the dataset's parquet and fingerprinted as one stream; on the first four datasets, a check through `kalanos grade` gave the same clean-pair minimum and median to five decimals. For 10 evenly spaced episodes per dataset, four copies were measured against their original: exact, re-saved as float32, Gaussian noise at 1% of each channel's standard deviation (seed 42), and the first 1% of rows trimmed.

| dataset | episodes | clean pairs | clean min / 1st pct / median | float32 copy, max | 1% noise copy, max | 1% trimmed copy, max |
|---|---|---|---|---|---|---|
| pusht | 206 | 21,115 | 0.273 / 0.573 / 1.086 | 0 | 0.0077 | 0.066 |
| cmu_stretch | 135 | 9,045 | 0.068 / 0.113 / 0.474 | 0 | 0.0069 | 0.105 |
| aloha_static_towel | 50 | 1,225 | 0.543 / 0.612 / 0.964 | 0 | 0.0084 | 0.367 |
| aloha_sim_transfer_cube_human | 50 | 1,225 | 0.406 / 0.486 / 0.914 | 0 | 0.0088 | 0.077 |
| berkeley_autolab_ur5 | 1,000 | 499,500 | 0.267 / 0.544 / 1.151 | 0 | 0.0085 | 0.257 |

Every exact copy hashed equal, and no two clean episodes of any dataset did. All five datasets store their values as float32, so a float32 copy hashed equal too. The near limit is twice the largest noise distance, 0.0088, rounded up to 0.02; no clean pair of any dataset lies within it, the closest (cmu_stretch episodes 90 and 101) at 0.068. Episodes that look alike on camera stay apart: aloha_sim_transfer_cube_human 8 and 10 at 0.455, cmu_stretch 9 and 79 at 0.131. Resampled points are kept as float32 less each channel's mean, so a small motion around a large offset is not rounded away; an exact copy is still found by its hash. From 10,000 episodes the near search projects every trajectory on its 8 main directions and compares only the episodes close on all of them; a projection never lengthens a distance, so it finds the same pairs as comparing every pair. On synthetic joint motions with 50 planted copies it confirmed only those 50 pairs and took 4.1 s at 100,000 episodes. Data with no main directions defeats the projections, and every pair is compared: on random 28-channel vectors the pass took 34 s at 100,000 episodes. Either way it holds about 7 KB per episode of 28 channels until it runs, growing with the channel count (4 KB at 14 channels, 12 KB at 56).

**Known limit.** A copy trimmed or shifted in time resamples onto different points, so it is found by neither rule: trimming 1% of rows moved copies 0.066 to 0.367 away, inside the clean range. The fingerprints live only for the run, so a report loaded from disk cannot be checked again.

---

### Motion variation

`Report.motion_variation` says how much episodes' trajectories differ from each other. Its `value` is the root mean square, over every pair of episodes, of the normalized RMSE the near tier measures, from the same resampled and scaled trajectories, so it reads no payload twice and decodes no video. It is computed from the per-point variance across episodes, so no pair is compared and it costs one pass over the trajectories at any episode count. Two episodes that move the same way read 0 apart, and a value grows as their motions part.

It covers one group of episodes, the largest that holds the same numeric streams and channels and could be resampled (every value finite, at least 2 rows), and `episodes` says how many. Of two groups of equal size, the one met first in episode order is covered. Exact copies are included, since each is an episode a policy trains on, so a dataset of identical episodes reads 0. Episodes whose every channel is constant take no part, as in the duplicate check.

The value follows trajectories, not scenes: episodes in one scene with varied motions read high, and episodes in several rooms that repeat one motion read low. It compares episodes within one dataset. Each channel is scaled by its spread over the covered episodes, so two datasets' values are not comparable. It is descriptive: no score, grade or threshold reads it. It is unavailable, with `value` null and the reason in `reason`, when numpy is missing (`episodes` is then 0) or fewer than two episodes are comparable.

## motion

Whether the recorded motion is physically plausible. These need the taxonomy: jerk is normalised by the motion's own scale so that two robots can be compared, and 90% of maximum only means something once a value is known to be a joint velocity.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `mean_jerk_norm`, `max_abs_jerk` | STREAM | `proprio.joint_position` | normalised | Jerk is the third derivative of position: squared, averaged, and normalised by the motion's own scale. | **report-only** until dimensionless — see below; *candidate band kept on file: good < 0.1, bad > 0.5* |
| `velocity_spike_pct` | CHANNEL | `proprio.joint_position` | % | Share of step-to-step position changes more than 6σ from the channel's own mean change: a jump no smooth motion makes, such as an encoder slip or a re-homed joint. Needs at least 40 samples, since one jump among fewer changes cannot reach 6σ. Evidence carries the source rows. | **report-only**; no band decided, waits on the detector benchmark's benign rate |
| `log_dimensionless_jerk` | CHANNEL | `proprio.joint_position` | dimensionless | −ln(T⁵·∫jerk²dt / range²) over the whole channel. It does not change with playback speed or amplitude, and higher is smoother. Not applicable on a channel with nulls or one that never moves. Same estimator as the diagnostic `position_dimensionless_jerk_v1`, without its segmentation. | **report-only**; no band decided |
| `action_chatter` | STREAM | `proprio.joint_velocity` | normalised | Average step-to-step change in velocity, normalised by its spread. | *to define* |
| `vel_saturation_pct` | CHANNEL | `proprio.joint_velocity` + declared max | % | How often a joint ran above 90% of maximum velocity. A saturated joint clips, so the recording shows something other than what was commanded. | good < 1%; shipped metric measures against the channel's own observed maximum and stays report-only until a declared limit is supplied |
| `limit_proximity_pct` | CHANNEL | `proprio.joint_position` + declared limits | % | Percentage of time a joint spent above 95% of its allowed range. | range adherence 99.8%; shipped metric measures against the channel's own observed range and stays report-only until declared limits are supplied |
| `still_drift` | STREAM | `proprio.joint_position` | unit | How far joint readings wander when the recording ends with the robot holding still. Skipped automatically when it does not end static, because motion is not drift. | *to define* |
| `hf_vibration_ratio` | CHANNEL | `proprio.joint_torque` | fraction | Share of the torque signal's energy above 20 Hz. Deliberate motion lives below about 5 Hz; energy above that is mechanical. | *candidate: good < 0.1, bad > 0.3* |
| `p99_torque`, `max_torque`, `mean_torque` | CHANNEL | `proprio.joint_torque` | N·m | 99th percentile, peak and mean of absolute torque. | **report-only, always.** What counts as high depends entirely on the robot and the task |
| `energy_proxy` | EPISODE | torque + velocity | J (robot units) | Sum of \|torque × velocity\| over time, roughly the mechanical work done. Two runs of one task should land close; an outlier used the robot differently. | report-only |

### Jerk metrics are report-only

`mean_jerk_norm` and `max_abs_jerk` divide each channel by its own standard deviation before taking the third difference. That removes amplitude but not time: the result still carries units of 1/s³, so it grows with the cube of how fast the robot moves. A perfectly smooth joint following a sine at frequency *f* measures about √2·(2π*f*)³ — roughly 0.35 at 0.1 Hz, 44 at 0.5 Hz and 180 at 0.8 Hz — against a candidate *bad* bound of 0.5. Graded against that band, any ordinary manipulation motion scores 0.

So both metrics ship `report_only`: measured and shown in every report, not graded. The candidate band stays in `default.yaml` so the numbers are not lost. They return to grading once the measure is made dimensionless (normalised by the motion's duration as well as its amplitude, as log dimensionless jerk is) and a band is set against real recordings. `log_dimensionless_jerk` is that dimensionless measure, shipped as its own metric so these two keep their meaning. It is report-only for the same reason.

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

Frame-level quality for camera streams. Every metric here reads a sample of frames by default and reports how many it looked at, because decoding everything to grade a dataset is not a plausible thing to do. `sharpness_score`, `exposure_shift_pct` and `exposure_level` measure 10 evenly spaced frames per camera stream by default, `blank_frame_pct`, `contrast_score`, `dark_frame_pct` and `bright_frame_pct` read the statistics kept of those same frames without decoding any more, and `frozen_frame_pct` reads 10 evenly spaced windows of about 1 s of consecutive frames. The bundle's top-level `vision` section sets both counts (`sample_frames`) and whether every metric that reads frames reads all of them instead (`full_frame_scan`); `KALANOS_VISION_SAMPLES` and `KALANOS_FULL_FRAME_SCAN` override it, and `--vision-samples` and `--full-frame-scan` override both. `--tier full` always reads every frame. No frame is decoded twice: when `frozen_frame_pct` runs, the sampled frames are taken from inside its windows, and otherwise only the sampled frames are decoded. The same decode reads `cut_count`'s windows and the frame `camera_stream_present` checks. Each frame is decoded as a 128x128 gray image for exposure and freezes, and each sampled frame also at native resolution for blur; every frame is measured and dropped, so a full scan holds no frames in memory. `frame_count_vs_timebase` counts the container's packets, decoding nothing, under a full scan too, since a decode bounded by the timestamps could never find a frame past them. `camera_stream_present` and `video_format_consistency` share one read of the container header; presence also takes the segment's first frame from the same read, and decodes up to the first frame inside the segment by itself, wherever in the segment it sits, only when that read cannot tell.

These metrics grade only under a scope that requires `sampled_video_quality` or `video_quality`. Under the default `numeric-core-v1` scope they are measured and shown in every report but graded `report_only`. They feed the coverage ledger: a camera stream whose `sharpness_score` and `exposure_shift_pct` computed counts as sampled, and one where both measured every declared frame counts as fully read. `sampled_video_quality` needs every camera sampled; `video_quality`, which `vision-imitation-v1` requires, needs every camera fully read. Depth streams are not cameras here. A critical vision result requests review until a calibration manifest covers the metric, except `frame_count_vs_timebase` and `camera_stream_present`, which block through the contract route, still only under a scope that grades vision.

Without a decoder installed, all of these return `not_applicable` with a reason and the rest of the grade completes. A missing optional dependency must not turn a gradeable dataset into an error.

The section's `max_decode_frames` bounds the frames the one read each camera's vision metrics share decodes, `cut_count`'s windows and seek preroll included; a full scan reads every frame whatever its value. `max_pixels` bounds the size of a frame measured at native resolution. A read that reaches either cap is `unavailable`, and one the decoder fails after its first frame is an `error`; neither is graded, and both keep the frames already read as evidence. A file that fails before any frame decodes, one that is missing, cannot be opened or has no frame rate, is `not_applicable` for every metric but `camera_stream_present`, which reads 0 and blocks the episode under a scope that grades vision. A read that fails outside the file, such as a dropped connection or an HTTP error from a remote dataset, says nothing about the file: presence is `unavailable` there, the episode stays unknown, and the next read tries again. A decode failure after the segment's first frame leaves the episode unknown, not blocked. A stream whose rows the adapter reordered is not decoded, since frame positions would no longer name source rows.

Each camera stream's read is recorded once, as `GradedStream.frames`: the rows requested, examined and missing, the sample plan and its parameters, and per frame its presentation time, luminance shape, blur and clipped share. The frames a sample would examine, and those `exposure_shift_pct` judged bad, also carry `luma_sha256`, a SHA-256 of their native luminance. Each metric carries only its own bad frames, as `support` intervals of source rows: `exposure_shift_pct` its bad sampled frames, `frozen_frame_pct` its counted runs. A `sharpness_score` or `exposure_shift_pct` result names the frame to look at in `worst_index`, and a `frozen_frame_pct` result in `first_run_index`.

Depth streams (`extero.depth`, `extero.depth_confidence`, `extero.disparity`) are not camera footage, and every metric here is `not_applicable` on them.

| Metric | Level | Requires | Unit | Definition | Threshold |
|---|---|---|---|---|---|
| `frozen_frame_pct` | STREAM | an image or video stream + an `action.*` stream | % | Share of frames, among those read, that repeat their predecessor inside a run of at least 0.5 s while an action stream in the same episode moves. A pair repeats when the mean absolute difference of its 64x64 gray thumbnails (2x2 averages of the shared 128x128 decode), scaled to a contrast of 255 by the lower P95 - P5 of the two frames, is below max(0.10, 0.25 x the 10th percentile of the stream's differences); a pair with a blank frame (P95 - P5 below 5) is never frozen. Reads `--vision-samples` evenly spaced windows (10 by default) of 1 s of consecutive frames, or every frame with `--full-frame-scan`. **Not applicable** without an action stream (an idle robot in a static scene would read as frozen) or with under 1 s of frames. | *calibrated: good < 0.01%, bad > 10%* |
| `sharpness_score` | STREAM | decode | fraction | Median, over the sampled frames, of the share of the 3x3 Laplacian variance of the frame's native-resolution luminance that a further 3x3 box blur removes. Higher is sharper; a frame with no edges reads 0. | *calibrated: good > 0.74, bad < 0.55* |
| `exposure_shift_pct` | STREAM | decode | % | Share of sampled frames exposed unlike their camera's typical frame within the episode, or blank. The typical frame is the median, over the sampled frames, of the mean gray level and of the share of pixels at 250 or more. A frame is bad when it is darker or brighter (its mean moves by more than 35% of the typical mean, and at least 20 gray levels), clipped (its share of pixels at 250 or more exceeds the typical share by more than 20 points), or blank (P95 - P5 below 5, on any camera). | *calibrated: good < 10%, bad > 40%* |
| `exposure_level` | STREAM | decode | gray level | Median, over the sampled frames, of each frame's mean gray (0-255); evidence carries the median share of pixels at or below 5 and at or above 250. | reported, no band |
| `blank_frame_pct` | STREAM | decode | % | Share of sampled frames that are blank (P95 - P5 below 5). `exposure_shift_pct` counts these frames too; this metric counts them alone, so a camera blank throughout reads 100. | reported, no band |
| `contrast_score` | STREAM | decode | gray level | Median, over the sampled frames, of each frame's P95 - P5 spread (0-255). Lower is flatter. It follows what is in view as much as the camera. | reported, no band |
| `dark_frame_pct` | STREAM | decode | % | Share of sampled frames that are crushed (at least half their pixels at or below 5, `dark_level`) or dim (P95 at or below 40), each frame judged by itself. A camera dark or dim throughout reads 100 here, and clean on `exposure_shift_pct` unless its frames are blank. | reported, no band |
| `bright_frame_pct` | STREAM | decode | % | Share of sampled frames with at least half their pixels at or above 250 (`bright_level`), each frame judged by itself. A white simulator scene reads 100, as a camera blown out throughout does. | reported, no band |
| `frame_count_vs_timebase` | STREAM | a video file | fraction | `\|packets in the episode's segment - timestamps\| / timestamps`, with the segment's bounds taken half a frame early as tolerance. On a one-file-per-episode layout (LeRobot v2) the count runs to the file's end, so frames past the timestamps show. Blocks through the contract route. | *candidate: good < 1%, bad > 5%* |
| `camera_stream_present` | STREAM | a video file | flag | 1 when the camera's file opens and yields a frame inside the episode's segment, 0 when it is missing, cannot be opened, holds no video stream, or yields no frame there. Reads the container header, then takes the segment's first frame from the shared decode, or decodes up to it alone when that read cannot tell, discarding the seek preroll; a frame dropped at the segment's start does not count as missing. Blocks through the contract route. | good = 1, bad = 0.5 |
| `video_format_consistency` | STREAM | a video file | fields | How many of width, height, frame rate and pixel format in the container header differ from what the dataset declares; evidence names each differing field with its declared and found values. A field the dataset does not declare is listed under `undeclared` and skipped, and one the container header does not state under `unstated`, also skipped; with none declared, the metric is `not_applicable`. Two frame rates differ when further apart than 0.1% of the declared one, which absorbs 30000/1001 against a declared 29.97. Reads the container header only. LeRobot declares the rate and pixel format in the feature's `info` (`video_info` in v3.0 datasets), and the size there or through `shape` labelled by `names`. | reported, no band; *candidate* fps tolerance |
| `pts_defect_pct` | STREAM | a video file | % | Share of the steps between the segment's sorted packet presentation timestamps that are a duplicate (two equal timestamps) or a gap (a step longer than 1.5 declared frame periods, the period being the segment's span over its declared frame count, since the container's average rate falls with every dropped frame). The rule assumes the constant frame rate the format declares; LeRobot's own encoder places frame k at k/fps, so a recording it wrote reads 0. | reported, no band |
| `max_gap_s` | STREAM | a video file | s | Longest step between the segment's sorted packet presentation timestamps; evidence gives its start and end in segment seconds, the index of the packet before it, and its length in declared frame periods. Edges of the segment are not counted, so a stall whose later frames fall past the segment's end does not show here; `frame_count_vs_timebase` reads those frames as missing. | reported, no band |
| `camera_sync_skew_ms` | EPISODE | two or more cameras, each a one-file-per-episode video | ms | Per camera, the median over frames paired by index of (packet time - recorded timestamp); the value is the largest offset minus the smallest. It measures where each file's frames sit against the recorded timestamps; capture synchronisation is not measured, since LeRobot's recorder stamps every camera at k/fps, so a clean LeRobot episode reads 0. Three kinds of camera are listed under `skipped` evidence: a segment of a file holding several episodes (LeRobot v3), whose packets are chosen by its recorded start and so sit on its timestamps whatever happened; a camera whose paired frames hold a duplicate or a gap, which would shift every later pair by a frame period; and a camera whose paired frames hold a null or non-finite recorded timestamp. A LeRobot v3 dataset therefore reads not applicable. | reported, no band |
| `cut_count` | STREAM | decode + a state or `action.*` stream | cuts | Cuts in the footage: stretches removed from video and control alike, which leave both jumping at one step. The 3 largest robust z-scores (median and MAD) of the per-step mean \|delta\| of every state and action stream, where a stream that holds still on most steps takes the median of its nonzero deviations in place of a MAD of 0, are placed on the camera's clock by timestamp, each on the frames whose incoming step overlaps it in time (one frame when camera and control share a clock, more when their rates differ), and the 7 frames before each and its own frames with the 6 after them are read in the shared decode, whatever the tier. A candidate is a cut when the 64x64 gray mean absolute difference of the step into one of its frames exceeds 3.6 times the median of the window's other nonzero steps. Reports at most 3; each cut is a one-frame support interval at the first frame after it. State is every numeric series in the episode that is not an `action.*` or a `reward.*` stream: a reward jumps with the task, not with the robot. The same state serves `dropped_chunk_count` and `video_control_lag_ms`. **Not applicable** without a state or action stream, with fewer than 15 frames, or when no frame of the windows decodes. | reported, no band |
| `dropped_chunk_count` | STREAM | decode + a state stream | chunks | A chunk of footage lost while the timestamps stayed uniform, which makes the footage jump at a step where the state does not. Each step's 64x64 gray mean absolute difference is divided by the median of its 6 nonzero neighbours on each side, and that by the largest same ratio of any state stream's per-step mean \|delta\| at that step, floored at 1. The highest score more than 10 steps from either end is a drop when it exceeds 9.9. Reports at most one drop per episode, as a one-frame support interval at the first frame after it. A cut moves the state too and scores low here; a drop that leaves the footage short of its timestamps is `frame_count_vs_timebase`'s to find. Reads every frame, in the decode the other vision metrics share. **Not applicable** without `--full-frame-scan`, without a state stream, with fewer than 14 moving steps, or with no step more than 10 steps from either end. | reported, no band |
| `video_control_lag_ms` | STREAM | decode + a state or `action.*` stream | ms | How far the footage runs ahead of the control signals; negative when it trails. The footage's motion per step (its 64x64 gray mean absolute difference, the largest of each step and its two neighbours, minus its 25-step moving average) is correlated with every control channel's step size, detrended the same way, at each lag within 10 frames. The lag is the peak of the mean curve of the 3 channels that correlate best, refined on a parabola, converted at the stream's frame rate. It is read against state when a state channel correlates with the footage, otherwise against action, and evidence names which as `reference`; evidence carries both lags, with the peak correlation and its margin over the best correlation 2 or more lags away. Every clean episode reads a lag, so the value describes the rig. One episode's value is noisy: on aloha the peak stands 0.01-0.06 above the rest of the curve. A dataset-level reading is not computed yet, and drift of the lag within an episode is not reported. Reads every frame, in the decode the other vision metrics share. **Not applicable** without `--full-frame-scan`, without a state or action stream, without a frame rate, or when no control channel correlates with the footage. | reported, no band |

`frozen_frame_pct` counts only runs of at least 0.5 s, so isolated duplicates such as toto's every-4th-frame repeats never count. Its windows can miss a freeze that falls between them; `--full-frame-scan` finds it at the cost of time and, on a remote dataset, of downloading every video segment. The comparison, minimum run and moving-robot rules come from rda. Its *good* bound follows rda and Calibra: both flag any freeze, so any counted freeze leaves *good*; its *bad* bound is calibrated below. rda also excludes on three or more freezes or a single one of 3 s or more; neither is graded here, since `longest_run_s` and `n_runs` are evidence, not metrics.

`cut_count`, `dropped_chunk_count` and `video_control_lag_ms` check the footage against the robot's own signals. The two thresholds, and the readings below, were measured on `lerobot/aloha_static_towel` at revision `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff`, retrieved 2026-10-14, on the cameras `cam_left_wrist` and `cam_high`, re-encoded as AV1 with a keyframe every 2 frames, with defects injected at known frames. Each threshold is 1.25 times the largest score on 20 clean calibration episodes (20-39), the larger of the two cameras; the counts below are on 20 evaluation episodes (0-19) per camera. Both thresholds are calibrated on that one dataset only.

- `cut_count`: no false positive on a clean episode; 20 of 20 cuts of 25 or 100 frames found during motion on both cameras, 14 of 20 cuts of 25 frames in the stillest stretch, and 20 of 20 of 100 frames there.
- `dropped_chunk_count`: no false positive on a clean episode; of 15 frames dropped during motion and padded at the end, 7 of 20 found on each camera; of 5 frames, none. The full read takes about 2 s of wall time per 500-frame episode, shared with the other vision metrics.
- `video_control_lag_ms` has no threshold. Against state, the clean median is -2.9 frames on `cam_left_wrist` and -2.7 on `cam_high`: the footage trails the state by about 3 frames at 50 fps. With the footage shifted 5 frames ahead, the median over the 20 episodes moves +5.2 and +4.8 frames from the clean median, and with 2 frames, +2.1 and +1.9. It takes about 2 s of wall time per 500-frame episode, the full read included.

### Calibration

The bands of `sharpness_score`, `exposure_shift_pct` and `frozen_frame_pct` are calibrated on three LeRobot datasets, retrieved from Hugging Face on 2026-10-01 and graded with kalanos 0.7.0, before its release, under a scope requiring `sampled_video_quality`, with the default 10 sampled frames:

| dataset | revision | episodes | cameras | camera streams |
|---|---|---|---|---|
| `lerobot/pusht` | `7628202a2180972f291ba1bc6723834921e72c19` | 206 | `observation.image` (96x96, 10 fps) | 206 |
| `lerobot/cmu_stretch` | `d84574233973ea9a292babd008256d2290a3f96e` | 135 | `observation.images.image` (128x128, 5 fps) | 135 |
| `lerobot/aloha_static_towel` | `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff` | 50 | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` (640x480, 50 fps) | 200 |

Every episode was graded as it is, 541 clean camera streams in all. Defects went into 10 evenly spaced episodes of each dataset, 60 camera streams, each decoded at native resolution and graded clean and then under each variant:

- `blur_5x5`: every frame replaced by its 5x5 box mean (`blur_frames`).
- `clip_40pct`: the middle 40% of frames replaced by the same frames multiplied by 4 and clipped at white (`clip_frames` applied to the whole stream, then spliced).
- `freeze_10pct`: 10% of the frames, from the 25% point on, replaced by the frame before them (`freeze_frames`), read with a full frame scan.

The band rule:

- *good* sits at the worst clean camera stream across all three datasets, its value rounded to two significant figures toward the defect, so that stream still grades good (a value on the bound is inside it).
- *bad* is the value the matching defect reaches on the median injected stream.
- Where *good* would reach or pass *bad*, the candidate band stays.

| metric | band | worst clean stream (n = 541) | median injected stream (n = 60) | benchmark benign rate | benchmark detection rate |
|---|---|---|---|---|---|
| `sharpness_score` | good 0.74, bad 0.55 | 0.743 (aloha `cam_left_wrist`) | 0.549 (`blur_5x5`) | 0 of 206, 135 and 50 episodes | blur: 30/30, 30/30, 120/120 |
| `exposure_shift_pct` | good 10%, bad 40% | 10% (aloha wrist cameras) | 40% (`clip_40pct`) | 0 of 206, 135 and 50 episodes | clipped: 30/30, 0/30, 0/120 |
| `frozen_frame_pct` | good 0.01%, bad 10% | 0% | 10% (`freeze_10pct`) | 0 of 206, 135 and 50 episodes | frozen frames: 30/30, 2/30, 120/120 |

Benchmark rates are per dataset, in the order pusht, cmu_stretch, aloha_static_towel, from `kalanos benchmark` with 30 sampled episodes under these bands. Four things limit them:

- The benign rates are in-sample: *good* was placed beyond the same clean streams they count, so 0% follows from the rule. They show that no clean stream fires, not how often an unseen dataset would.
- `exposure_shift_pct` compares each frame with its own episode's typical frame, so the benchmark's clip, which overexposes every frame alike, moves it only where the clip whitens frames unevenly, as on pusht's white background. A camera exposed one way throughout is the cross-episode comparison's to catch.
- The benchmark freezes a quarter of the frames from the 25% point, wherever the robot is. It catches every freeze on pusht and aloha_static_towel but 2 of 30 on cmu_stretch, and under a full scan 57 of 60 `freeze_10pct` streams fire, the 3 misses cmu_stretch's too. Each missed freeze is found and set aside as idle: in the three cmu_stretch episodes checked, the actions move on 42% to 62% of steps, and on 30% to 45% inside the frozen stretch, below the majority a run needs. A freeze while the robot stands still reads like a still scene, so no signal-derived metric can tell them apart.
- The 40% at which `exposure_shift_pct` reaches *bad* follows from the variant: 4 of the 10 sampled frames fall in the clipped stretch.

*good* for `frozen_frame_pct` stays at 0.01% rather than 0: no clean stream froze, and the shortest counted run (0.5 s) already reads above 0.01% of any episode under 80 minutes.

Left as candidates:

- `frame_count_vs_timebase`: every clean stream read 0 and no variant models a frame-count fault, so rda's (liesliy/rda) working values stand, 1% for review and 5% for exclusion.
- `blur_vs_camera` and `exposure_vs_camera`: neither raised a finding on any of the 541 clean streams; nothing was injected to place their fences.
- `repeated_footage_vs_camera`: not yet run on a calibration dataset.
- `integrity.duplicate_episode` and `integrity.near_duplicate_episode`: measured on five datasets (under integrity, "Duplicate episodes"), with no clean pair flagged; no duplicate occurs in them to confirm the rules find one.

`blank_frame_pct`, `contrast_score`, `dark_frame_pct` and `bright_frame_pct` were measured on the same three datasets at the same revisions, retrieved from Hugging Face on 2026-10-09 and graded with kalanos 0.7.2 plus these metrics, with the default 10 sampled frames: all 541 clean camera streams, and 10 evenly spaced episodes of each dataset (60 camera streams) decoded at native resolution with the whole stream replaced by each variant:

- `black`: every frame zeroed.
- `dim_0p1`: every frame scaled by 0.1.
- `flat`: every frame mapped to `frame * 0.08 + 118`, a spread of about 20 gray levels on a frame that spans 0 to 255. pusht's frames span far less, so there it leaves a spread below 5 and reads blank.
- `white`: every frame at 255.

Clean camera streams, min / median / max:

| dataset | streams | `blank_frame_pct` | `contrast_score` | `dark_frame_pct` | `bright_frame_pct` |
|---|---|---|---|---|---|
| pusht | 206 | 0 / 0 / 0 | 16 / 40.5 / 53 | 0 / 0 / 0 | 100 / 100 / 100 |
| cmu_stretch | 135 | 0 / 0 / 0 | 120 / 148 / 184 | 0 / 0 / 0 | 0 / 0 / 0 |
| aloha_static_towel | 200 | 0 / 0 / 0 | 106.5 / 168 / 197.5 | 0 / 0 / 0 | 0 / 0 / 0 |

Median injected stream, per dataset in the order pusht, cmu_stretch, aloha_static_towel:

| variant | `blank_frame_pct` | `contrast_score` | `dark_frame_pct` | `bright_frame_pct` |
|---|---|---|---|---|
| `black` | 100, 100, 100 | 0, 0, 0 | 100, 100, 100 | 0, 0, 0 |
| `dim_0p1` | 40, 0, 0 | 5, 15, 16.75 | 100, 100, 100 | 0, 0, 0 |
| `flat` | 100, 0, 0 | 4, 12.5, 13.25 | 0, 0, 0 | 0, 0, 0 |
| `white` | 100, 100, 100 | 0, 0, 0 | 0, 0, 0 | 100, 100, 100 |

The band rule above, applied across all three datasets, proposes:

- `blank_frame_pct`: good 0%, bad 100% (`black`). No clean stream held a blank sampled frame.
- `contrast_score`: good 16, bad 13 (`flat`), higher is better. The worst clean stream is a pusht one, and 53 of the 60 `flat` streams fall below 16.
- `dark_frame_pct`: good 0%, bad 100% (`dim_0p1`). Every clean stream reads 0, and every `dim_0p1` and `black` stream reads 100.
- `bright_frame_pct`: no separation. Every clean pusht stream reads 100, as `white` does, because its simulator renders on a white background.

`dark_frame_pct` counts a frame as dim when its P95 is at or below 40. Its column above comes from a second run on the same datasets and revisions, retrieved on 2026-10-14 with the same sample, which kept each sampled frame's statistics. Over the 5,410 sampled frames of the 541 clean streams, the lowest P95 was 141 (aloha_static_towel; 155 on cmu_stretch, 255 on every pusht frame). Over the 600 `dim_0p1` frames, the highest was 25 (pusht). Crushed pixels alone did not separate them: a `dim_0p1` frame keeps few pixels at or below 5, and the median `dim_0p1` stream had no frame with half its pixels there. Dimmer copies sit between: scaled by 0.2, a stream reads 100 on cmu_stretch, a median of 70 on aloha_static_towel and 0 on pusht; scaled by 0.3, 0 on all three.

All four stay reported, with no band, until a band is decided.

A calibrated band sets the grade only. It claims nothing about whether a task succeeded, and blocking on a vision result still needs an accepted calibration manifest, except through the contract route of `frame_count_vs_timebase` and `camera_stream_present` under a scope that grades vision.

The measurements behind each metric's definition, on single datasets:

- **Sharpness band.** Measured on `lerobot/aloha_static_towel` at revision `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff`, retrieved 2026-09-30: episode 0 (the first 500 frames of the AV1, 640x480, 50 fps clip) of `cam_high` and `cam_left_wrist`, as the median re-blur loss of 10 evenly spaced frames. A ratio, not a variance: the raw native Laplacian variance of the two sharp cameras of one robot differed 2.4x (128.5 and 53.3), so no fixed variance band fits every camera, while the ratio stayed within 0.78 to 0.83 for both.

  | | sharp | box 3x3 | box 5x5 | box 9x9 |
  |---|---|---|---|---|
  | cam_high | 0.813 (0.800-0.834) | 0.508 | 0.366 | 0.294 |
  | cam_left_wrist | 0.780 (0.688-0.804) | 0.465 | 0.308 | 0.218 |

  Under the calibrated band, sharp footage grades good and a 3x3 blur or worse critical. The range beside a sharp value is its minimum to maximum over the 10 frames.
- **Dim footage.** Re-encoded with libx264 and read in full on the same `cam_high` episode, a copy darkened to 0.2x read 5.4% frozen under a fixed difference floor and reads 0% with the contrast scaling, while a 10% freeze injected into it reads 9.6% (9.8% on the undarkened clip).
- **Exposure follows each camera.** rda judges exposure by fixed gray levels: dark below a mean of 50, bright above 230, clipped at 20% of pixels at 250 or more, flat below a P95 - P5 of 20. Those rules marked every sampled frame of lerobot/pusht bad, because its simulator renders on a white background, and rda saw the same false blown-out results. No fixed level can separate the two: pusht frames average 249 to 250 with 74% to 84% of pixels at 255, where lerobot/cmu_stretch frames brightened 4x average 229 to 241 with 74% to 78%. So each frame is compared with its own camera instead, and pusht and cmu_stretch both read 0% on every episode. Overexposing, dimming to 0.4x or blacking out 4 of 10 sampled frames reads 40% on both. Contrast is not compared: it follows what is in view, and a third of clean pusht frames hold half the typical spread.
- **Known limit.** Within one episode, a camera exposed the same way in most sampled frames is its own typical frame, so `exposure_shift_pct` reads it clean unless its frames are blank. Across episodes, a camera exposed one way in every episode gets a descriptive trait, and a single episode unlike its camera is flagged once the camera has 5 episodes. Below that, the per-episode limit still holds. A camera dark or dim throughout shows in `dark_frame_pct`, which judges each frame by itself.

### Across episodes

After every episode is graded, each camera is compared with itself across the episodes it appears in. A camera is keyed by its source field (the LeRobot `video_key`) when it has one, and otherwise by its taxonomy type and instance; depth streams take no part. Blur and exposure are compared only when at least 5 of the camera's episodes have the value. The rules are candidates, not calibrated:

- **Blur.** An episode is flagged `vision.blur_vs_camera` when its `laplacian_var` (from `sharpness_score` evidence), on a log scale, lies below Q1 - 1.5 x IQR of the camera's episodes, and is also at most half the camera's median. Only the low side is flagged. Raw variance is skewed upwards: on all 50 aloha episodes its fences raised 2 false positives on `cam_high`, and the log scale none.
- **Exposure.** An episode is flagged `vision.exposure_vs_camera` when its `exposure_level` lies outside [Q1 - 1.5 x IQR, Q3 + 1.5 x IQR] of the camera's episodes, and is also at least 20 gray levels from the camera's median.
- **Repeated footage.** An episode is flagged `vision.repeated_footage_vs_camera` when the SHA-256 sequence of its camera's hashed native-luminance frames equals another episode's. The match says the camera's footage repeats, not that the whole episode was copied: a video copied or linked to the wrong episode, and a camera replaying the same output, both match. Only complete reads take part, a sequence with fewer than two distinct hashes is ignored, and the rule needs no minimum episode count.

Quartiles are linear interpolations over every episode of the camera, the one under test included. A finding is critical and requests review under a scope that grades vision (`video_quality` or `sampled_video_quality`), and is a `report_only` warning otherwise; it never moves the 0-100 score. The blur and exposure rules are independent: raw Laplacian variance scales with contrast, so a dimmed or brightened episode can carry both findings. No absolute blur floor is set, because none separates the calibration datasets: clean `laplacian_var` ranges from 6.0 on aloha's `cam_right_wrist` to 1821 on cmu_stretch, while a 5x5 blur reaches up to 52.8 on cmu_stretch and 11.3 on aloha, and every frame dimmed to 0.2x, measured on the same 60 injected streams, up to 97.2 on cmu_stretch.

A camera with at least half its pixels at or below 5 in every compared episode is described as `dark`, and one with at least half at or above 250 as `white`. The trait requests nothing: lerobot/pusht, whose simulator renders on white, is described, not flagged. `Report.cameras` holds one summary per camera: its key, the episodes it appears in, whether blur and exposure were compared and why not, its median `laplacian_var` and `exposure_level`, its trait, the episodes flagged by each rule, the groups of episodes with repeated footage, and its visual variation.

Measured on `lerobot/aloha_static_towel` at revision `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff`, retrieved 2026-09-30: 50 episodes of 500 frames (AV1, 640x480, 50 fps) of `cam_high` and `cam_left_wrist`, each summarised from 10 evenly spaced frames as the median native-luminance Laplacian variance and the median frame mean. Variants were applied to the decoded luminance before measuring.

| camera | clean laplacian_var, min / median / max | clean mean gray, min / median / max | 3x3 blur, median var | 0.5x dim, median var / mean | 2x bright, median var / mean |
|---|---|---|---|---|---|
| cam_high | 88.7 / 100.8 / 119.9 | 83.0 / 92.4 / 101.9 | 20.3 | 26.4 / 46.2 | 266.9 / 163.5 |
| cam_left_wrist | 27.2 / 35.6 / 44.6 | 100.1 / 105.3 / 113.8 | 8.6 | 9.9 / 52.6 | 120.6 / 180.5 |

Over every run of 4 to 10 consecutive clean episodes, the fence alone held a false positive in 5 to 9 of every 41 to 47 windows, and with the two gaps it held none on either camera. With the gaps, every injected 3x3 blur, 0.5x dim and 2x brightening was flagged at 5, 6, 8 and 10 episodes, while at 4 episodes some were missed (7 of 188 blurs on `cam_left_wrist`, 21 of 188 dims on `cam_high`).

`cameras[].visual_variation` says how much a camera's footage changes between episodes. Each episode's sampled frames, the positions blur and exposure measure, decoded as 128x128 gray, are average-pooled to 16x16 and scaled to 0-1, so it decodes nothing more. Its `value` is the root mean square, over every pair of the camera's episodes, of the RMS difference between their pooled frames. It is computed from running per-pixel sums while episodes are graded, so no pair is compared and no episode's frames are kept. Two episodes that show the same footage read 0 apart.

Motion in view counts as change: an arm crossing a fixed scene makes the episodes read varied, as a change of room or objects does. The value compares episodes within one camera. Two cameras' or two datasets' values do not rank them, since resolution, framing and how much of the view moves all set the scale. It is descriptive: no score, grade or threshold reads it.

It covers the episodes whose read decoded every sampled frame, grouped by how many frames were sampled, and `episodes` says how many the largest group holds. Of two groups of equal size, the one met first in episode order is covered. A full frame scan pools the same frames a sample takes. Only a camera's first stream in an episode counts. It is unavailable, with `value` null and the reason in `reason`, when fewer than two episodes have a complete read with the same number of frames.

Measured on six LeRobot v3.0 datasets, retrieved 2026-10-27, 50 evenly spaced episodes each, 10 evenly spaced frames per episode, against a control per camera of 50 copies of one episode with Gaussian noise of SD 2.55 (1% of 255, `numpy.random.default_rng(42)`): every one of the 11 real cameras read above its control, from about 3x on usc_cloth_sim (0.0055 against 0.0018) to over 100x on cmu_stretch and aloha_static_towel. Pooling the frames cost about 0.19 ms per camera per episode.

| Dataset | Revision |
|---|---|
| lerobot/cmu_stretch | `d84574233973ea9a292babd008256d2290a3f96e` |
| lerobot/berkeley_autolab_ur5 | `c4e26a697fc4c04776b0558f83e563d14be0109f` |
| lerobot/pusht | `7628202a2180972f291ba1bc6723834921e72c19` |
| lerobot/aloha_sim_transfer_cube_human | `6a43d500f101255823a9d2b9dc244eeb01a2cd31` |
| lerobot/usc_cloth_sim | `6c4fb6acf8a6db8c4d0692b318023272c4967b5f` |
| lerobot/aloha_static_towel | `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff` |

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
| `spike_pct` | default & joint_velocity | 0.1% | 2% | candidate · frame rewards exempt |
| `snr_db` | default | 30 dB | 15 dB | candidate |
| `snr_db` | joint_velocity | 20 dB | 10 dB | candidate |
| `snr_db` | joint_acceleration | 12 dB | 6 dB | candidate |
| `snr_db` | joint_torque | — | — | report-only: motor-current channels read 8–15 dB on real ALOHA arms in every episode |
| `dead_taxel_pct` | default | 2% | 10% | candidate |
| `hysteresis` | default | 0.05 | 0.20 | candidate |
| `mean_jerk_norm` | default | 0.1 | 0.5 | candidate · low confidence · **report-only** (see *Jerk metrics are report-only*) |
| `max_abs_jerk` | default | 0.1 | 0.5 | candidate · low confidence · **report-only** (see *Jerk metrics are report-only*) |
| `hf_vibration_ratio` | default | 0.1 | 0.3 | candidate · low confidence |
| `sharpness_score` | default | 0.74 | 0.55 | calibrated (see *Calibration*) · higher is better |
| `frame_count_vs_timebase` | default | 1% | 5% | candidate |
| `camera_stream_present` | default | 1 | 0.5 | contract · higher is better |
| `frozen_frame_pct` | default | 0.01% | 10% | calibrated (see *Calibration*) |
| `exposure_shift_pct` | default | 10% | 40% | calibrated (see *Calibration*) |
| `exposure_level` | default | — | — | reported, no band |
| `blank_frame_pct` | default | — | — | reported, no band |
| `contrast_score` | default | — | — | reported, no band |
| `dark_frame_pct` | default | — | — | reported, no band |
| `bright_frame_pct` | default | — | — | reported, no band |

Metrics not listed are either `report_only` (`drift`, `p99_torque`, `max_torque`, `mean_torque`, `energy_proxy`, `velocity_spike_pct`, `log_dimensionless_jerk`, `pts_defect_pct`, `max_gap_s`, `camera_sync_skew_ms`, `video_format_consistency`, `cut_count`, `dropped_chunk_count`, `video_control_lag_ms`, and `vel_saturation_pct` / `limit_proximity_pct` until a declared limit is wired in) or still *to define* (`action_chatter`, `still_drift`, `monotonic_violations`, `task_instruction_missing`) — none of them grade.

An episode counts each metric once, at its worst channel, stream or camera. Its family scores are weighted means of those metrics, and its score is a weighted mean of its family scores, renormalised over the families that scored, minus a capped penalty per distinct failing metric. The dataset averages its episodes.

### Readiness

The dataset's headline number is **readiness**, from 0 to 100, in the report's `readiness` section:

```text
Readiness is defined only when the declared inventory is complete,
all episodes are evaluated, and every episode is pass or blocked.
A blocked episode contributes 0; a passing episode contributes its quality.
Review or unknown episodes leave readiness undefined.

readiness = sum of contributions / expected episode count
```

Only an authorized blocking consequence blocks an episode. Critical severity alone does not supply that authorization. `passing_quality` describes passing episodes; when none pass it is null. Review and required missing evidence cannot be converted into a high readiness score by averaging the remaining episodes.

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

`language_conditioned`, selected with `KALANOS_POLICY_PATH=language_conditioned`, extends the default for datasets that train a vision-language-action model: there `task_instruction_missing` grades, so an episode without its instruction is blocking.

**Context and reference evidence.** `snr_db` is a smooth/residual diagnostic, not a sensor-health measurement; see *Noise-floor reference* above, under `integrity`, for exclusions, the reference contract and decision behavior.

Weights are relative integers set in `policy.yaml`: ~1 is supporting evidence, ~3 a standard core check, ~5 a defect that directly corrupts training.

The episode score is built from its metrics directly:

- Each metric counts once per episode, at the points of its worst channel, stream or camera. One failing channel among twelve costs the metric's full share, as it would on a single-channel stream. Unmapped streams count like any other.
- A family's score is the weighted mean of those worst-subject points over its metrics that graded.
- A family scores in an episode only when at least one of its *core* metrics graded (`core: true` in `policy.yaml`, marked in the table below); its other graded metrics then renormalise around it. A family whose policy marks no metric core scores whenever any of its metrics graded.
- The episode score is the weighted mean of its family scores, by the family weights, renormalised over the families that scored.
- `fail_penalty`, when a policy sets one, subtracts once per metric whose worst subject scored 0, however many subjects failed it.

Channel and stream scores are for display. They fold only their own node's metrics the same way, and a stream averages its channels and its own metrics equally. The dataset score and the dataset's family scores are the mean over the episodes that scored them. For each family, the dataset also reports how many of those episodes scored below the policy's `low_episode_score` (70 by default) and the lowest of them (`score.family_tails`). Neither changes a score.

The report records, per metric, how many subjects it graded and how many it flagged (`score.metrics`). The dataset count is of distinct channels, streams or cameras, with the number of episodes the metric flagged; the card's *Flagged channels* section and `kalanos inspect` show it.

`report_only` and `not_applicable` results are excluded from the denominator, because a metric that could not run must not silently cost points. A report-only metric's weight takes effect once the metric grades.

| Family | Metric | Weight | Core |
|---|---|---|---|
| timing | `effective_hz` | 5 | true |
| timing | `drop_rate` | 4 | true |
| timing | `monotonic_violations` | 4 | false |
| timing | `dt_jitter_ms` | 3 | false |
| timing | `recorded_drop_estimate` | 2 | false |
| timing | `recorded_hz` | 1 | false |
| timing | `recorded_dt_spread_ms` | 1 | false |
| integrity | `missing_pct` | 4 | true |
| integrity | `flatline_pct` | 4 | true |
| integrity | `spike_pct` | 4 | true |
| integrity | `dead_taxel_pct` | 3 | true |
| integrity | `snr_db` | 2 | false |
| integrity | `hysteresis` | 2 | true |
| integrity | `drift` | 1 | false |
| motion | `log_dimensionless_jerk` | 5 | false |
| motion | `velocity_spike_pct` | 4 | false |
| motion | `action_chatter` | 2 | false |
| motion | `vel_saturation_pct` | 2 | false |
| motion | `hf_vibration_ratio` | 2 | true |
| motion | `mean_jerk_norm` | 1 | false |
| motion | `max_abs_jerk` | 1 | false |
| motion | `still_drift` | 1 | false |
| motion | `limit_proximity_pct` | 1 | false |
| motion | `p99_torque` | 1 | false |
| motion | `max_torque`, `mean_torque`, `energy_proxy` | none, report-only | false |
| vision | `frozen_frame_pct` | 6 | true |
| vision | `sharpness_score` | 5 | true |
| vision | `blank_frame_pct` | 5 | false |
| vision | `frame_count_vs_timebase` | 4 | true |
| vision | `cut_count` | 4 | false |
| vision | `dropped_chunk_count` | 4 | false |
| vision | `exposure_shift_pct` | 3 | true |
| vision | `dark_frame_pct` | 3 | false |
| vision | `bright_frame_pct` | 3 | false |
| vision | `max_gap_s` | 3 | false |
| vision | `pts_defect_pct` | 2 | false |
| vision | `contrast_score` | 2 | false |
| vision | `camera_sync_skew_ms` | 2 | false |
| vision | `video_format_consistency` | 1 | false |
| vision | `camera_stream_present` | 0, a contract gate that never moves the score | false |
| vision | `exposure_level` | none, report-only | false |
| vision | `video_control_lag_ms` | none, report-only | false |
| annotation | `task_instruction_missing` | 5 | true |

Family weights: integrity 3, timing 2, motion 2, vision 2, annotation 1.

## Benchmarking metrics

```bash
kalanos benchmark                                                    # the two pinned reference datasets, 30 sampled episodes each
kalanos benchmark ./recording --profile acquisition.yaml --sample 10 --out benchmark.json
```

Measures, per dataset, the benign rate (how often a metric fires on the data as recorded) and the detection rate (how often it fires once a `kalanos.testing` defect is injected into a sample of episodes). With no `paths` it grades the two pinned reference datasets (`lerobot/pusht`, `lerobot/libero_10`, each pinned to a fixed revision); `--sample` sets how many episodes per dataset receive an injected defect, 30 by default. `--out FILE.md` renders the result table; `--out FILE.json` dumps the model; without `--out` the table prints to stdout.

`benchmark` shares bundle loading, mapping precedence, binding resolution and configuration identities with `grade`, and accepts the same `--profile`, `--map`, `--map-file`, `--no-sidecar` and `--tier`. Metadata-tier benchmarking injects no payload defects. Camera defects (blur, clipping, frozen frames) go into a native-resolution copy of each sampled video stream, graded clean and then with the defect; a camera whose native frames would take more than 1 GiB is reported rather than injected. Vision metrics only grade in the benchmark when the scope requires `sampled_video_quality` or `video_quality`, as in `kalanos grade`; under any other scope no camera is decoded for injection.

See [Calibration](#calibration) above for how benchmark results on the three LeRobot datasets there set the vision metric bands.

## Adding a metric

1. Write the function with its `@metric` decorator — `level=` and `family=`, both required — in the family's module under `src/kalanos/analysis/metrics/`. A `family=` outside the nine above fails at import. `label=` names the condition a finding from this metric describes, in plain words, for the terminal summary; without it the metric's name is used.
2. Add its `good`, `bad` and weight to `src/kalanos/assets/policies/default.yaml`, keyed `family.metric_name`; that key's prefix is what makes it a member of the family. A metric that grades needs a weight: in a weighted family, grading fails and names any graded metric without one. A bound that is not yet settled stays *to define* here and absent there. A *candidate* bound may ship, but only with a comment marking it unconfirmed — never as though it were decided.
3. Add a contract test naming the defect it fires on, using an injector.
4. Add a row above.

Out of tree, the same function ships as a package declaring a `kalanos.metrics` entry point. That group names the *module*, not the function: importing it is what runs the decoration. `kalanos new metric` writes one.

A metric added without those four steps gives the tool an opinion nobody agreed to. If the catalogue looks like it is missing something, raise it and file an issue rather than closing the gap in a table.

## Computation coverage and consequences

Metric `availability` records computation separately from graded `status`. A report-only torque statistic is computed and belongs in its eligible torque denominator. Known taxonomy mismatch is not applicable; unresolved taxonomy, insufficient samples or missing prerequisites remain unavailable. Skipped payloads retain declared channels in the denominator. Unexpected computation errors are recorded and cause exit 2.

Flatline evaluation no longer bridges null, NaN or infinite values; this can change its measured fraction. Spike intervals identify scored source samples and their wider 51-sample support. Neither SNR nor spectrum measurements invent intervals.

Severity thresholds still describe measurements. Statistical critical findings require matching accepted calibration to block; otherwise their consequence is review. No new calibrated-performance claim accompanies these changes.
