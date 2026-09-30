# Changelog

## 0.7.0 — unreleased

The decision-integrity release. Report schema 7.0.0.

### Contract (R07-01, R07-03 interfaces, R07-02/04/05 models)

- One authoritative `episodes[].eligibility` (`pass | blocked | review | unknown`) under a named scope, decided once after every metric runs. Every dataset count, the gate, the CLI exit code and the compatibility `train_ready` derive from it; a report whose surfaces disagree fails validation.
- `score >= 70` no longer decides anything. `train_ready` mirrors eligibility (`true`/`false`/`null`).
- Task and dataset traits are descriptive; prevalence no longer exempts an episode from blocking.
- Findings carry `consequence` and `route` separately from severity, and `support` (whole-episode or explicit half-open sample intervals).
- `readiness` is `null` with reasons whenever any episode is `review` or `unknown`, the inventory is incomplete or empty, or a passing episode lacks a score. Formula id recorded.
- `eligibility_counts` always published; `sufficiency` evaluated against explicit requirements (`min_pass_episodes` in this release).
- `producer`, `run` (configuration identities with content digests, source evidence with explicit hash scope), `scope`, `inventory`, `binding_conflicts` recorded.
- Removed `gate.pruned_grade` and `gate.train_ready_after_pruning`.
- Domain: `ClockInfo` (origin incl. `publish` and `presentation`, origin evidence), `SourceOrder`, `ChannelBinding` with separate actuator kind, quantity, representation, unit and command semantics. Adapters populate these in R07-04/02.
- Policy: `MetricPolicy.consequence`/`route`, `Policy.enforce_calibration` (off) and `calibrated_metrics`.

### Contract corrections (branch review round 2)

- Inventory: a declared episode that never loads is `inventory.unresolved`, counted as `unknown`; `complete` is derived, never assumed; readiness and the eligible share are null while the denominator is unresolved; the default gate fails an incomplete audit; `run.completion` is `partial`.
- Every stream records `evaluation.payload`; under `numeric-core` a stream with channels that was not read (`missing_input`, `skipped`, `error`) makes its episode `unknown`. One stream's result never covers another stream.
- `--tier metadata` is an execution boundary: no numeric payload is fetched, required checks are `unknown`, the default gate fails.
- Dataset `score.train_ready` derives from the counts and inventory in assembly, with or without a letter gate; validation rejects a contradiction.
- `run.binding` digests the effective mapping after precedence (an argument override changes it); `run.bundle` identifies the declared file; the dictionary digest covers full content; `run.execution` records tier and limits.
- Same-priority mapping conflicts are refused wherever they occur: repeated `--map` flags, duplicate YAML keys (strict loader for bundles and mapping files), and disagreements below the winning level.
- A bundle's relative `policy.path` resolves against the bundle's directory; missing or malformed policies are `ConfigurationError` → exit 2.

### Contract corrections (branch review round 3)

- Inventory: an adapter that refuses part-way keeps the episodes it yielded and the count it declared; the gap is `unresolved`; a refused source is listed in `inventory.refused_sources` rather than counted as one failed episode; a refused directory is not re-offered file by file; `run.completion` is `partial`.
- Metadata tier is enforced at the storage boundary for LeRobot v2/v3: parquet schema read, index/clock columns projected, numeric vectors never materialised (`kalanos.analysis.execution` carries the tier to adapters). `StreamEvaluation.n_channels_declared` added.

### Clock provenance and recorded order (R07-04)

- Readers preserve source row order (LeRobot v2/v3, CSV/delimited/JSON/JSONL, HDF5, MCAP); backwards and repeated timestamps are measured in source order with the offending source rows, never hidden by sorting.
- Every stream reports `clock_info` (origin, origin evidence, source field, native unit/dtype, domain/epoch, transforms) and `source_order`, including at metadata tier. Native integer ticks are subtracted before conversion to seconds; null/NaN/inf timestamps break adjacency and keep their source addresses.
- New report-only recorded-timeline checks: `recorded_hz`, `recorded_dt_spread_ms`, `recorded_drop_estimate`. The capture checks `effective_hz`, `dt_jitter_ms` and `drop_rate` require producer evidence that timestamps are capture times, and abstain otherwise.
- **Behaviour change:** no built-in reader records producer capture evidence, so dropped samples, wrong clock rates and jitter no longer block or lower readiness on built-in formats. They are reported, with the affected episodes named. A legacy `Clock.CAPTURE` label migrates as inferred capture, not producer proof. README worked examples re-scored on 0.7.0.
- MCAP reads in file order and selects header, then distinguishable publish, then log time per topic; a header stamp alone is not capture evidence. HDF5 keeps a damaged recorded time axis instead of substituting a generated grid.

### Contract corrections (branch review round 4)

- CLI: an incomplete inventory fails the default `--fail-on blocked,unknown` gate (exit 1) — a refused source alone, passing episodes beside a refused source, and undelivered declared episodes. `--fail-on blocked` permits it with a stderr warning. Operational errors remain exit 2.
- Episode ids are stable across partial and complete runs: containers always qualify `source::episode`; only an adapter that names the episode after the file keeps the short form. A single-episode LeRobot export changes id from `root` to `root::episode_000000`.
- A directory an adapter selected is owned on every read path, including zero-yield; its manifest and data files are no longer offered to other adapters.

### Configuration

- `--profile bundle.yaml`: one file, four sections with separate identities (`binding`, `requirements`, `policy`, `execution`).
- `--tier metadata|standard|full`; a tier never changes requirements.
- One resolver (`assets/bundle.py`) for `--map` > `--map-file` > bundle > sidecar; conflicts recorded, same-priority disagreement is an error. `--map`, `--map-file` and `kalanos-map.yaml` remain supported unchanged.

### CLI

- `--fail-on` (default `blocked,unknown`); exit `0` clean audit, `1` gate tripped, `2` configuration/operational error.
- `kalanos inspect REPORT.json` reads schema 7 and legacy 6.3/6.4/6.5 reports, naming contradictions and unsupplied fields.

### Not in this change (tracked in the release plan)

- Routing `benchmark` through the shared resolver (R07-07); per-channel typed views (R07-02); adapters filling `clock_info`/`source_order` (R07-04); applicable-denominator coverage and interval localisation for existing metrics (R07-05); `compare` (R07-06); publisher and Action migration (R07-09); source content hashing (`run.source.digest` is `null` and `complete: false` until implemented).
