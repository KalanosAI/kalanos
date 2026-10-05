# Changelog

All notable changes to Kalanos are listed here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Kalanos uses [Semantic Versioning](https://semver.org/).
Before 1.0, a minor version can break the report schema or the CLI.

## [Unreleased]

### Changed

- `kalanos grade` prints a shorter card with the same sections for every format: the verdict and episode counts, the episodes that need attention, findings grouped by source in plain words, coverage for each capability that applies, and the files not analysed. Values, evidence and every episode stay in `--report` and `kalanos inspect`.
- The card is plain ASCII when stdout is not a terminal.

### Added

- `--color/--no-color` on `kalanos grade`; a non-empty `NO_COLOR` acts as `--no-color`.
- `label=` on `@metric`, naming the condition a finding describes for the terminal card.

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
