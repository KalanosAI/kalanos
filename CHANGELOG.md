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

### Configuration

- `--profile bundle.yaml`: one file, four sections with separate identities (`binding`, `requirements`, `policy`, `execution`).
- `--tier metadata|standard|full`; a tier never changes requirements.
- One resolver (`assets/bundle.py`) for `--map` > `--map-file` > bundle > sidecar; conflicts recorded, same-priority disagreement is an error. `--map`, `--map-file` and `kalanos-map.yaml` remain supported unchanged.

### CLI

- `--fail-on` (default `blocked,unknown`); exit `0` clean audit, `1` gate tripped, `2` configuration/operational error.
- `kalanos inspect REPORT.json` reads schema 7 and legacy 6.3/6.4/6.5 reports, naming contradictions and unsupplied fields.

### Not in this change (tracked in the release plan)

- Routing `benchmark` through the shared resolver (R07-07); per-channel typed views (R07-02); adapters filling `clock_info`/`source_order` (R07-04); applicable-denominator coverage and interval localisation for existing metrics (R07-05); `compare` (R07-06); publisher and Action migration (R07-09); source content hashing (`run.source.digest` is `null` and `complete: false` until implemented).
