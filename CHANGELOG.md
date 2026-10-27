# Changelog

All notable changes to Kalanos are listed here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Kalanos uses [Semantic Versioning](https://semver.org/).
Before 1.0, a minor version can break the report schema or the CLI.

## [Unreleased]

### Added

- `kalanos grade --compact` writes the report JSON with every key replaced by a short alias and one key table first, under `_compact`. `kalanos.load_report`, `kalanos inspect`, `kalanos compare` and `kalanos publish` read both formats. Full JSON stays the default.
- The report states how much episodes' trajectories differ from each other in `motion_variation`: the root mean square, over every pair of episodes, of the distance the near-duplicate check measures, from the trajectories it already resamples, with the number of episodes covered. It is descriptive, feeds no score, and compares episodes within one dataset (report schema 8.0.0).
- When the near-duplicate tier is saturated, the report writes one dataset warning to the new `dataset_warnings`, stating how many episodes have a near-identical partner and the share of compared pairs that are near. `kalanos inspect` and the HTML report show it (report schema 7.3.0).

### Changed

- `kalanos.load_report`, `kalanos inspect` and `kalanos compare` validate a full report from the JSON they already parsed, so loading takes 22–29% less time and 33–46% less peak memory on reports of 9 to 118 MB.
- A vision diagnostic embeds one preview per camera and episode, a WebP of its blurriest sampled frame in `thumbnail_webp_base64`, named by `measurements.preview_row`, in place of up to `preview_frames` PNG previews. A vision metric graded warning or critical no longer embeds a preview; its `worst_index` or `first_run_index` names the frame (report schema 7.3.0).
- Each episode's diagnostic results are stored once, in `diagnostics.results`; `episodes[].diagnostics` is gone (report schema 7.3.0).
- `duplicates` names episodes by their index in `episodes`: `duplicates.exact` holds each group of identical episodes, and `duplicates.near` stores each near pair once, with its distance to 4 significant digits. A pair no longer carries `footage_identical` or `cameras`, which `cameras[].repeated_footage` already holds. On `lerobot/usc_cloth_sim`, its 487,458 near pairs take 6.2 MB. An older 7.x report that lists pair objects is converted on load (report schema 7.3.0).
- A near finding lists no partner: its evidence is `max_nrmse` and `nearest_nrmse`, the distance to its nearest partner, and its `value` counts every episode it is near. `duplicates` lists every near pair up to 1.5 million, and past that each episode's 5 nearest, as `near_duplicates.pairs_listed` reports. When most episode pairs are near, the new `near_duplicates` summary says so once and no near finding is written. An exact finding's evidence is empty: its group is in `duplicates.exact`, and its repeated footage in `cameras[].repeated_footage` (report schema 7.3.0).
- Every reason text is written once, in the new first key `reason_codes`, and each `reason`, `reasons` and `detail` value holds its index there (report schema 8.0.0). `kalanos inspect`, `kalanos compare` and `kalanos.load_report` read the codes back as text.
- `kalanos inspect`, `kalanos compare` and `kalanos.load_report` read any 7.x report up to the current schema, 7.3.0. A 7.2.0 report's per-episode diagnostics are ignored on load.
- A written report leaves out a field whose value is null, an empty list or an empty map when that is the field's default, and loading the report restores it. This affects `--json`, `--report` and the publish payload; `Report.model_dump` is unchanged (report schema 8.0.0).
- Each stream's `coverage` lists only the metrics that computed, skipped or errored on it. The episode and dataset coverage ledgers keep every metric with the same counts (report schema 8.0.0).

## [0.8.2] - 2026-10-21

Report schema 7.2.0.
The report records each camera's format, codec and source metadata, each episode's split and each channel's source dtype.

### Added

- The report records each camera's width, height, frame rate and pixel format, both as the dataset declares it and as the video file's container states it, whether or not they agree, in `streams[].video_format` (report schema 7.2.0). A value the source does not state is null, and the metadata tier, which opens no media, leaves the container's side null. ([#53](https://github.com/KalanosAI/kalanos/pull/53))
- The report records each camera's codec, profile, bitrate, frame count, duration and colour range, primaries, transfer and space as the video file's container states them, the codec, channel count and depth-map and audio flags a LeRobot feature declares, and the camera make, model and encoder the container's tags name, in `streams[].video_format` (report schema 7.2.0). The container's bitrate, frame count and duration describe the whole file, which in a LeRobot v3.0 dataset holds several episodes. Only those three tags are read, so no location is recorded. A value the source does not state is null. ([#54](https://github.com/KalanosAI/kalanos/pull/54))
- The report records each episode's split, read from a LeRobot dataset's `info.json`, in `episodes[].split`, and each channel's dtype as a LeRobot feature, an HDF5 dataset or an MCAP message field declares it, in `source_dtype` (report schema 7.2.0). Both are null where the source states none; a CSV, JSON or tabular column's inferred type is not recorded. ([#53](https://github.com/KalanosAI/kalanos/pull/53))

### Changed

- `kalanos inspect`, `kalanos compare` and `kalanos.load_report` read any 7.x report up to the current schema, 7.2.0, so a report saved by 0.7.x, 0.8.0 or 0.8.1 still loads. A report of a newer schema is refused. ([#53](https://github.com/KalanosAI/kalanos/pull/53))

## [0.8.1] - 2026-10-19

Report schema 7.1.0.
Policy errors and policy names are reported correctly, frame rewards no longer grade as spikes, and the packaged `legacy_0_5` policy is removed.

### Changed

- `spike_pct` no longer grades `reward.frame_reward` channels: a computed reward rises and falls in steps that read as spikes. The value is still reported, as `flatline_pct` already does for rewards. ([#45](https://github.com/KalanosAI/kalanos/pull/45))

### Fixed

- `kalanos grade` and `kalanos benchmark` report a policy that weights some metrics or families and leaves another without a weight as one `kalanos:` line naming it, and exit 2 instead of printing a traceback and exiting 1. The Python API raises `MixedWeights`, which is still a `ValueError`. ([#44](https://github.com/KalanosAI/kalanos/pull/44))
- With `KALANOS_POLICY_PATH` set, the report's `run.policy` and `scope.policy_id` name the policy that graded: a packaged policy by its name, from `builtin`, and a policy file by its file name without the extension, with its path as origin. They named the built-in default before. ([#46](https://github.com/KalanosAI/kalanos/pull/46))

### Removed

- The packaged `legacy_0_5` policy. Its numbers no longer matched 0.5's once the bands changed; `KALANOS_POLICY_PATH=legacy_0_5` now fails like any missing policy file. A policy file with `extends: default` and `gate: null` grades without the dataset gate. ([#50](https://github.com/KalanosAI/kalanos/pull/50))

## [0.8.0] - 2026-10-15

Report schema 7.1.0.
Each metric family gets its own score from shipped default weights, and camera footage gets checks for missing files, cuts, lag and duplicate episodes.

### Added

- `kalanos grade`, the HTML report and `kalanos inspect` show the dataset's 0-100 score for each metric family that scored, under a display name (`timing` as Temporal, `integrity` as Integrity, `motion` as Motion, `vision` as Vision, `annotation` as Annotations); any other family shows under its id. ([#38](https://github.com/KalanosAI/kalanos/pull/38))
- The report records the family weights it graded with in `family_weights` (report schema 7.1.0), and `kalanos grade` shows each family's weight beside its score. ([#39](https://github.com/KalanosAI/kalanos/pull/39))
- `kalanos grade`, the HTML report and `kalanos inspect` show, for each family, how many episodes scored below 70 and the lowest episode score. The report records both in `score.family_tails` (report schema 7.1.0), and the policy's `low_episode_score` sets the cutoff.
- The report records, for each metric in each episode and across the dataset, how many channels, streams or cameras it graded and how many it flagged, and each graded metric's points (report schema 7.1.0). The card's new "Flagged channels" section and `kalanos inspect` show the counts. ([#40](https://github.com/KalanosAI/kalanos/pull/40))
- A `core` flag on a policy's metric entries marks the metrics a family needs graded before it scores in an episode. The default policy marks the core timing, integrity, motion, vision and annotation metrics (listed in `docs/METRICS.md`), and an `extends` override keeps the base's flag unless it sets one. ([#40](https://github.com/KalanosAI/kalanos/pull/40))
- Two report-only vision metrics read from video packet timestamps, decoding nothing: `max_gap_s`, the longest gap between a camera's frames and where it sits, and `camera_sync_skew_ms`, the spread of an episode's cameras' offsets against their recorded timestamps, on one-file-per-episode videos. ([#42](https://github.com/KalanosAI/kalanos/pull/42))
- Report-only vision metrics `blank_frame_pct`, `contrast_score`, `dark_frame_pct` and `bright_frame_pct`, read from the frames the vision metrics already sample. `dark_frame_pct` counts both crushed frames and dim ones, so a camera dimmed throughout reads 100. ([#42](https://github.com/KalanosAI/kalanos/pull/42))
- `camera_stream_present` checks that each declared camera's video file opens and yields a frame inside the episode's segment. ([#42](https://github.com/KalanosAI/kalanos/pull/42))
- `video_format_consistency` reports which of width, height, frame rate and pixel format in a camera's video file differ from what the dataset declares. ([#42](https://github.com/KalanosAI/kalanos/pull/42))
- `integrity.duplicate_episode` flags episodes whose numeric payloads (state, action and any other graded series) are identical to another episode's, and requests review under every scope; `integrity.near_duplicate_episode` reports, without consequence, episodes whose resampled trajectories are nearly identical, such as a copy re-saved as float32. `Report.duplicates` (report schema 7.1.0) lists the pairs, each copy against the first of the episodes it copies, and whether each pair also repeats its camera footage exactly. ([#42](https://github.com/KalanosAI/kalanos/pull/42))
- `cut_count` reports where camera footage was cut: a stretch removed from video and control alike, found at the largest jumps in the state and action streams, reward streams left out, and confirmed in a few decoded frames. It is report-only. ([#42](https://github.com/KalanosAI/kalanos/pull/42))
- `dropped_chunk_count` reports a chunk of camera footage lost while its timestamps stayed uniform: a jump in the footage that the state does not share. It reads every frame, so it runs only with `--full-frame-scan`, and it is report-only. ([#42](https://github.com/KalanosAI/kalanos/pull/42))
- `video_control_lag_ms` reports how far camera footage runs ahead of the state stream, or of the action stream when there is no state or the state does not move with the footage, by correlating the motion of both. It reads every frame, so it runs only with `--full-frame-scan`, and it is report-only. ([#42](https://github.com/KalanosAI/kalanos/pull/42))

### Changed

- The default policy weights each metric and each family (integrity 3, timing 2, motion 2, vision 2, annotation 1; metric weights are listed in `docs/METRICS.md`), so a family, episode or dataset score changes wherever metrics of different weight grade differently. A policy that grades a metric without a `weight` beside weighted ones, such as one of the unweighted report-only metrics given a band, now fails and names it. ([#39](https://github.com/KalanosAI/kalanos/pull/39))
- A policy that `extends` another keeps the base's weight for a metric it overrides without setting `weight`, and merges `family_weights` family by family. ([#39](https://github.com/KalanosAI/kalanos/pull/39))
- An episode's family scores count each metric once, at its worst channel, stream or camera, and a family scores only when one of its core metrics graded. The episode score is the family-weighted mean of those family scores, so family, episode and dataset scores drop where one channel fails among many. Channel and stream scores are shown as before and no longer feed the episode score. ([#40](https://github.com/KalanosAI/kalanos/pull/40))
- Under a scope requiring `sampled_video_quality` or `video_quality`, a camera whose file is missing, unreadable or frameless now blocks its episode; it left the episode unknown before.
  A network error reading a remote camera file still leaves the episode unknown.
  `camera_stream_present` carries weight 0 in the default policy and is not a core metric, so it gates without moving the vision score.
  A custom policy without a `vision.camera_stream_present` entry keeps the metric report-only. ([#42](https://github.com/KalanosAI/kalanos/pull/42))

### Fixed

- A video segment is placed by its own rate, its frame count over its recorded span, instead of the file's average rate. A stall that brought a shared file's average rate to half the real one or below pulled every segment of the file at least one frame early. ([#42](https://github.com/KalanosAI/kalanos/pull/42))

## [0.7.2] - 2026-10-08

Report schema 7.0.0.
Reports can be published to hub.kalanos.ai, from a saved file or straight after grading.

### Added

- `kalanos publish` sends a saved report to your private space on hub.kalanos.ai, and `kalanos grade --publish` sends it straight after grading; both authenticate with `KALANOS_API_KEY` or `--api-key`, and `KALANOS_HUB_URL` sets the hub. ([#36](https://github.com/KalanosAI/kalanos/pull/36))

### Changed

- `certifi` is a direct dependency, and TLS to the hub trusts its CAs alongside the system's. ([#36](https://github.com/KalanosAI/kalanos/pull/36))

## [0.7.1] - 2026-10-07

Report schema 7.0.0.
`kalanos grade` prints a shorter card, and camera footage gets integrity checks read from the video container.

### Added

- `--color/--no-color` on `kalanos grade`; a non-empty `NO_COLOR` acts as `--no-color`, and a spinner shows on stderr in a terminal. ([#33](https://github.com/KalanosAI/kalanos/pull/33))
- `label=` on `@metric`, naming the condition a finding describes on the card; every built-in metric has one. ([#33](https://github.com/KalanosAI/kalanos/pull/33))
- `vision.repeated_footage_vs_camera` flags episodes whose camera's sampled frames hash identically to another episode's, requesting review under a vision scope; `Report.cameras[].repeated_footage` lists the groups. ([#34](https://github.com/KalanosAI/kalanos/pull/34))
- `vision.pts_defect_pct`, the share of a video's packet timestamps that are duplicated or leave a gap longer than 1.5 frame periods, report-only. ([#34](https://github.com/KalanosAI/kalanos/pull/34))

### Changed

- `kalanos grade` prints a shorter card with the same sections for every format: verdict, episodes needing attention, findings by source, coverage and files not analysed; `--json` stays the interface for scripts. ([#33](https://github.com/KalanosAI/kalanos/pull/33))
- The card is plain ASCII when stdout is not a terminal. ([#33](https://github.com/KalanosAI/kalanos/pull/33))
- Skipped or unresolved files, unmapped streams and rows with no instance key log at `info` instead of `warning`, and the card gains a `mapping` row; `warning` now means trouble with the run itself. ([#33](https://github.com/KalanosAI/kalanos/pull/33))
- `vision.frame_count_vs_timebase` blocks through the contract route under a scope that grades vision, instead of requesting review. ([#34](https://github.com/KalanosAI/kalanos/pull/34))
- An episode's `unknown` reason for an unevaluated video capability names each unsampled camera and why. ([#34](https://github.com/KalanosAI/kalanos/pull/34))
- `VideoPayload.packet_times()` replaces `count_frames()`, and both packet metrics share one cached read per segment. ([#34](https://github.com/KalanosAI/kalanos/pull/34))

### Fixed

- LeRobot v2 videos are counted to the end of the episode's file, so frames past its last timestamp are no longer missed. ([#34](https://github.com/KalanosAI/kalanos/pull/34))

## [0.7.0] - 2026-10-03

Report schema 7.0.0.
Every episode now gets one decision, and the readiness score, the gate and the exit code all follow from it.

### Breaking

- Each episode carries one `eligibility` (`pass`, `blocked`, `review` or `unknown`), decided after every metric runs; `train_ready`, the counts, the gate and the exit code derive from it, and `score >= 70` no longer decides anything. ([#17](https://github.com/KalanosAI/kalanos/pull/17), [#19](https://github.com/KalanosAI/kalanos/pull/19))
- `readiness` is `null`, with reasons, whenever an episode is `review` or `unknown` or the inventory is incomplete. ([#17](https://github.com/KalanosAI/kalanos/pull/17))
- An uncalibrated critical statistical finding sends its episode to `review` instead of blocking it; blocking requires a structured calibration manifest, and `enforce_calibration: false` is rejected. ([#24](https://github.com/KalanosAI/kalanos/pull/24))
- Capture timing metrics (`effective_hz`, `dt_jitter_ms`, `drop_rate`) abstain unless the producer proves its timestamps are capture times, which no built-in reader does yet; rows keep their source order and are never sorted. ([#23](https://github.com/KalanosAI/kalanos/pull/23))
- Episodes inside a container are always named `source::episode`, so a single-episode LeRobot export changes id from `root` to `root::episode_000000`. ([#21](https://github.com/KalanosAI/kalanos/pull/21))
- An incomplete inventory (a refused source, or declared episodes that never load) fails the default gate. ([#19](https://github.com/KalanosAI/kalanos/pull/19), [#21](https://github.com/KalanosAI/kalanos/pull/21))
- `kalanos grade` exits `0` for a clean audit, `1` when `--fail-on` trips and `2` for a configuration or operational error. ([#17](https://github.com/KalanosAI/kalanos/pull/17))

### Added

- Vision metrics for camera streams: `sharpness_score`, `exposure_shift_pct`, `exposure_level`, `frozen_frame_pct` and `frame_count_vs_timebase`, with `--vision-samples` and `--full-frame-scan`. ([#29](https://github.com/KalanosAI/kalanos/pull/29))
- `velocity_spike_pct` and `log_dimensionless_jerk` motion metrics on joint positions, report-only. ([#30](https://github.com/KalanosAI/kalanos/pull/30))
- `kalanos benchmark`, which measures how often each metric fires on reference datasets and how often it catches an injected defect. ([#15](https://github.com/KalanosAI/kalanos/pull/15))
- `--map`, `--map-file` and a `kalanos-map.yaml` sidecar to type a source field for one run, recorded on `report.mapping_overrides`. ([#16](https://github.com/KalanosAI/kalanos/pull/16))
- `--profile` bundles with separate binding, requirements, policy and execution sections, per-channel bindings, and `--tier metadata|standard|full`. ([#17](https://github.com/KalanosAI/kalanos/pull/17), [#22](https://github.com/KalanosAI/kalanos/pull/22))
- `kalanos inspect`, `kalanos compare` and `kalanos profiles list|show|validate`, with `kalanos.compare` and `kalanos.load_report` in the library; `inspect` and `compare` also read 6.3, 6.4 and 6.5 reports. ([#24](https://github.com/KalanosAI/kalanos/pull/24))
- `--hash-source` records the byte identity of local sources and refuses a source that changes during the run. ([#24](https://github.com/KalanosAI/kalanos/pull/24))
- Clock provenance (`clock_info`, `source_order`) on every stream, and report-only `recorded_hz`, `recorded_dt_spread_ms` and `recorded_drop_estimate`. ([#23](https://github.com/KalanosAI/kalanos/pull/23))
- A coverage ledger per episode and dataset, and findings localized to source-row intervals. ([#24](https://github.com/KalanosAI/kalanos/pull/24))
- Contextual SNR that separates the smooth signal from the residual and accepts a validated `noise_floor` reference. ([#25](https://github.com/KalanosAI/kalanos/pull/25))
- Optional deeper diagnostics, configured in the bundle: stream-pair timing, command response, sampled video, motion shape, training windows and dataset cohorts, with the new `numeric` extra and `kalanos diagnostics summarize-study`. ([#26](https://github.com/KalanosAI/kalanos/pull/26))
- Dictionary categories: each stream carries its category, and `report.categories` maps every category to its group. ([#28](https://github.com/KalanosAI/kalanos/pull/28))
- `Bundle` and `ExecutionTier` exported from `kalanos`, and `--profile` accepts the preset names that `profiles list` prints.

### Changed

- JSON reports are written compactly. ([#31](https://github.com/KalanosAI/kalanos/pull/31))
- Boolean channels count toward `missing_pct`. ([#25](https://github.com/KalanosAI/kalanos/pull/25))
- Task and dataset traits are descriptive only; a finding on every episode still blocks. ([#17](https://github.com/KalanosAI/kalanos/pull/17))

### Fixed

- Float32 frame-number timestamps are read as a reconstructed clock, so timing metrics no longer grade rounding noise. ([#14](https://github.com/KalanosAI/kalanos/pull/14))
- Training windows ignore defects on channels they do not consume, and match native clocks exactly. ([#27](https://github.com/KalanosAI/kalanos/pull/27))
- Episode counts reconcile with recorded decisions and with failed or unresolved inventory. ([#27](https://github.com/KalanosAI/kalanos/pull/27))

### Removed

- `gate.pruned_grade` and `gate.train_ready_after_pruning`. ([#17](https://github.com/KalanosAI/kalanos/pull/17))

## [0.6.5] - 2026-09-28

### Changed

- `report.readiness` replaces letter grades on the card, the CLI and the README; the letter fields are deprecated. ([#13](https://github.com/KalanosAI/kalanos/pull/13))
- Publishing needs no manual approval; the PyPI environment accepts only `v*` tags. ([#12](https://github.com/KalanosAI/kalanos/pull/12))

## [0.6.4] - 2026-09-28

### Fixed

- LeRobot's nested channel names are read, and channels named as grippers grade as grippers. ([#11](https://github.com/KalanosAI/kalanos/pull/11))

## [0.6.3] - 2026-09-28

### Changed

- A channel that never changes in an episode is a warning (unused or disconnected) instead of a stuck sensor. ([#10](https://github.com/KalanosAI/kalanos/pull/10))

## [0.6.2] - 2026-09-27

### Changed

- Flatline is critical only above 90%, since joints hold still in real teleoperation; torque SNR is report-only. ([#9](https://github.com/KalanosAI/kalanos/pull/9))

## [0.6.1] - 2026-09-27

### Fixed

- A finding on nearly every episode is a dataset trait instead of a failure on each one, which removes false D grades. ([#8](https://github.com/KalanosAI/kalanos/pull/8))

## [0.6.0] - 2026-09-26

### Added

- A dataset grade gate: the share of failing episodes caps the letter, with task traits, pruning and coverage. ([#7](https://github.com/KalanosAI/kalanos/pull/7))
- `language_conditioned` and `legacy_0_5` policies. ([#7](https://github.com/KalanosAI/kalanos/pull/7))

## [0.5.0] - 2026-09-26

### Changed

- Action commands are exempt from flatline, SNR is not applicable below about 45 Hz, and reconstructed clocks report timing as not observable. ([#6](https://github.com/KalanosAI/kalanos/pull/6))

## [0.4.0] - 2026-09-26

### Added

- Task instructions are read per episode, with a report-only `annotation.task_instruction_missing`. ([#5](https://github.com/KalanosAI/kalanos/pull/5))

## [0.3.0] - 2026-09-26

### Added

- `monotonic_violations`, report-only until a band is settled, and a repeated-timestamp test fault. ([#4](https://github.com/KalanosAI/kalanos/pull/4))

## [0.2.1] - 2026-09-26

### Added

- Citation metadata, and a GitHub Release for every tag. ([#2](https://github.com/KalanosAI/kalanos/pull/2))

### Fixed

- Common robot-learning field names are recognised, false switch and jerk criticals are gone, and HDF5 timestamps are read. ([#3](https://github.com/KalanosAI/kalanos/pull/3))

## [0.2.0] - 2026-09-24

### Added

- Published on PyPI with trusted publishing. ([#1](https://github.com/KalanosAI/kalanos/pull/1))

## 0.1.0 - 2026-09-24

Initial public release.

[0.8.2]: https://github.com/KalanosAI/kalanos/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/KalanosAI/kalanos/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/KalanosAI/kalanos/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/KalanosAI/kalanos/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/KalanosAI/kalanos/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/KalanosAI/kalanos/compare/v0.6.5...v0.7.0
[0.6.5]: https://github.com/KalanosAI/kalanos/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/KalanosAI/kalanos/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/KalanosAI/kalanos/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/KalanosAI/kalanos/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/KalanosAI/kalanos/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/KalanosAI/kalanos/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/KalanosAI/kalanos/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/KalanosAI/kalanos/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/KalanosAI/kalanos/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/KalanosAI/kalanos/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/KalanosAI/kalanos/releases/tag/v0.2.0
