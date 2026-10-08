# Report schema 7.0.0

Written by Kalanos 0.7.0. This page lists what changed from 6.5, what a consumer must do, and what a legacy file can and cannot provide.

Report loading rejects duplicate episode IDs and reconciles each status count against episode decisions, adding identified failures and unresolved inventory to unknown. A published eligible share must match those counts; an omitted share remains readable. Diagnostic window counts also reconcile with the recorded window statuses and explicit unexamined-budget unknowns. Valid report shapes are unchanged; contradictory summaries are rejected rather than silently repaired.

## New top-level fields

| Field | Type | What it is |
|---|---|---|
| `producer` | `Producer` | Package name and version, build revision when the environment supplies `KALANOS_BUILD_REVISION`, adapter and metric implementation versions (populated by later work packages). |
| `run` | `RunInfo` | Run id, start/finish, completion (`partial` when the inventory is incomplete), execution tier, `source` evidence, and configuration identities each with a content digest: `requirements`, `policy`, `binding` (the *effective* mapping after precedence — an argument override changes it), `bundle` (the declared bundle file), `dictionary` (full content, not key list), `execution` (tier and limits, plus the diagnostic plan and the resolved vision settings when they differ from the defaults). |
| `scope` | `EvaluationScope` | `requirements_id`, `policy_id`, `binding_id`, `tier`. Every decision is relative to this. |
| `inventory` | `Inventory` | Episodes expected (when the source declares a count), loaded, failed with reasons, and `unresolved` (declared but never enumerated — no identities are invented). `refused_sources` lists sources an adapter refused before or during enumeration — a refused source is not one failed episode; it holds an unknown number unless it declared a count, so the inventory is incomplete. Episodes an adapter yielded before refusing are kept and graded. `complete` is `false` whenever `unresolved > 0` or any source was refused; finishing a directory walk proves nothing. `notes` records oddities such as more loaded than declared. |
| `eligibility_counts` | `EligibilityCounts` | `total`, `pass_count`, `blocked`, `review`, `unknown`, `inventory_complete`, `confirmed_eligible_share`. The counts partition `total`. |
| `readiness` | `Readiness` | `formula_id`, `score` or `null`, `reasons`, `passing_quality`. Replaces the 6.x shape. |
| `sufficiency` | `Sufficiency` | `status` and per-requirement `checks`. |
| `binding_conflicts` | `list[BindingConflict]` | Every feature where a lower-precedence mapping source disagreed with the winner. |
| `cameras` | `list[CameraSummary]` | One summary per camera compared across its episodes: key, episodes, whether blur and exposure were compared and why not, median `laplacian_var` and `exposure_level`, its trait, the episodes flagged by each rule, and the groups of episodes with repeated footage. See `docs/METRICS.md`. |

## New per-episode and per-finding fields

| Field | What it is |
|---|---|
| `episodes[].eligibility` | `status`, `scope_id`, `policy_id`, `reasons[]` (each with `id`, `kind`, `status`, `consequence`, `route`, `detail`). Required on every episode. |
| `episodes[].streams[].evaluation` | `payload` (`computed`, `not_required`, `missing_input`, `skipped`, `error`), `reason`, `n_channels_declared`, `n_channels_graded`. Under `numeric-core` a stream with channels whose payload is `missing_input`, `skipped` or `error` makes the episode `unknown` — one stream's pass never covers another stream that was not read. |
| `episodes[].streams[].frames` | A camera stream's shared read: rows requested, examined and missing, the sample plan and its parameters, and per frame its presentation time, shape, blur, clipped share and, for evidence frames, `luma_sha256`. `null` for a stream not read as camera footage. |
| `findings[].consequence` | `block`, `review` or `report_only`. |
| `findings[].route` | `contract` or `statistical` when `consequence` is `block`. |
| `findings[].support` | `kind` (`whole_episode` or `intervals`), `index_space`, `intervals[]` with zero-based half-open `start`/`end_exclusive` and optional wider `support_start`/`support_end_exclusive`. Metrics that measure the whole episode carry `whole_episode`; no interval is ever invented. Flatline and spike evidence is localised in source rows; SNR/spectral results retain `whole_episode`. |

## Changed semantics

- Episode ids are `source::episode` for every recording in a container (LeRobot, HDF5), however many of its siblings loaded; the short `source` form is kept only when the adapter names the episode after the file itself (single-recording CSV, JSON, JSONL, MCAP, delimited). A one-episode LeRobot export is now `root::episode_000000`, where 0.6 wrote `root`.

- `episodes[].score.train_ready` mirrors `eligibility` (`true`/`false`/`null`). No score threshold sets it. Dataset `score.train_ready` is a compatibility summary. Validation rejects a contradiction.
- `gate.failing_episodes` is exactly the set of `blocked` episodes; validation rejects any other set.
- Dataset `score.train_ready` is a function of `eligibility_counts` and nothing else: `false` if any blocked, `true` only if the inventory is complete, non-empty and all pass, `null` otherwise — with or without a letter gate. Validation rejects a value that disagrees.
- `eligibility_counts.total = loaded + failed + unresolved`; failed and unresolved episodes count as `unknown`. `confirmed_eligible_share` and `readiness.score` are `null` while the inventory is incomplete.
- `--tier metadata` reads no numeric payloads. The tier is enforced at the storage boundary for the LeRobot v2/v3 adapters: they read the parquet schema and project only `episode_index`, `timestamp` and `frame_index`; observation/action vectors never leave the file, and streams are yielded with their channel declarations and no payload. Camera video is not decoded or demuxed either: every vision metric reports `not_applicable` at this tier. Grading records `payload: skipped`, required numeric checks are `unknown`, and the default gate fails. A tier never lowers the requirements. Other adapters (HDF5, MCAP, tabular) do not yet project at the source; their payloads are read and then skipped at grading.
- `gate.task_traits` and `gate.dataset_traits` are descriptive; they no longer exempt episodes from blocking.
- `gate.pruned_score` is the mean quality of non-blocked episodes: a candidate description, not a sufficiency claim.
- Letter fields remain and drive nothing.

## Removed

- `gate.pruned_grade`
- `gate.train_ready_after_pruning`
- `readiness.evaluated_episodes`, `readiness.passing_episodes`, `readiness.blocking_episodes` (use `eligibility_counts`)

## Domain model additions (not serialised in the report yet)

- `Stream.clock_info: ClockInfo` — `origin` (`capture`, `receive`, `publish`, `log`, `presentation`, `generated`, `simulation`, `unknown`), `origin_evidence` (`producer`, `adapter`, `inferred`, `none`), native unit/dtype, transforms. `ClockInfo.from_legacy(Clock.RECONSTRUCTED)` yields inferred generation, never certified capture. Adapters populate it and expose `clock`, `clock_info` and `source_order` on each graded stream. `tick_period_s` optionally expresses an adapter-generated grid. Legacy capture labels migrate with inferred evidence; they do not supply a producer declaration. Native ticks remain in `Stream.native_timestamps` during analysis and are omitted from the report JSON.
- `Stream.source_order: SourceOrder` — whether rows are in source order and the index map back when not. LeRobot sample order is preserved; no sample timestamp sort remains.
- `ChannelBinding` — `actuator`, `quantity`, `representation`, `unit`, `command`, `device`, `origin`, `status`, `validations[]`. Metrics are routed through typed views built from these.

## Policy additions

- `MetricPolicy.consequence` (`block`/`review`/`report_only`, default block for critical) and `MetricPolicy.route` (`contract`/`statistical`, default statistical).
- `Policy.enforce_calibration` is now `true` and cannot be disabled. `calibration_manifests` contains structured accepted evidence; `calibrated_metrics` remains readable for migration but grants no authority. Unmatched statistical blockers resolve to review.

## Bundle file (`--profile`)

```yaml
schema_version: 1
binding:
  id: my-robot-v1
  features:                      # whole-feature assertions, the legacy shape
    observation.state: proprio.joint_position
  channels: []                   # per-channel semantics, consumed by typed-view metrics
requirements:
  id: numeric-core-v1
  required_families: [integrity]
  require_resolved_bindings: false
  min_pass_episodes: null
policy:
  id: default-decisions-v1
  path: null                     # a policy file (relative to this bundle's directory), or the packaged default
execution:
  tier: standard                 # metadata | standard | full
```

Every section is optional; the defaults are the built-in scope. A duplicate key anywhere in a bundle or mapping file is a configuration error (exit 2), as is the same feature mapped to two types at one precedence level — even when a higher level would have overridden it. A missing or malformed `policy.path` is exit 2, never a traceback.

## Consumers: what to change

1. Read `eligibility_counts` and `readiness.score` (which may be `null`; show `readiness.reasons`), not `score.train_ready` or a letter.
2. Gate CI on the exit code of `kalanos grade --fail-on ...`, or on `eligibility_counts` from `--json`.
3. Show `scope.requirements_id` beside any number: a readiness is meaningless without its scope.
4. To read a 6.x file, use `kalanos.analysis.models.legacy.load_any`; it returns a `LegacyReport` with the original data, its SHA-256, contradictions and the fields it cannot supply. Do not convert legacy values into schema-7 decisions.
5. Keep showing `scope`, `eligibility_counts`, `readiness`, `sufficiency` and `coverage` together wherever readiness appears: a `readiness.score` of `null` means undefined.

## Fixtures

- `tests/legacy_reports/lerobot_v3_tiny-6.5.0.json` — produced by the release branch at `18cf9b6` before this change; real branch output.
- `tests/legacy_reports/aloha_static_towel-048fef2-6.4.0-trimmed.json` — four episodes from the 048fef2 towel report (original SHA-256 `8001d3ce…`), two of them carrying the `train_ready: true` + gate-failing contradiction. Dataset-level fields are the original 50-episode values and intentionally do not reconcile with the four episodes; the fixture tests loading, not arithmetic.
- `tests/legacy_reports/aloha_static_towel-43e0cb1-6.3.0-trimmed.json` — first episode from the supplied schema-6.3 report. See the adjacent README for source digest and trimming boundaries.

## Coverage, findings and calibration additions

- Top-level and episode `coverage` ledgers; stream `coverage` rows. Earlier schema-7 files may omit these and must show missing coverage, not zero coverage. `gate.coverage` remains a legacy grading summary.
- The ledger distinguishes `computed`, `not_applicable`, `unavailable`, `skipped`, `error` and `not_required`; a computed report-only metric still counts as evaluated. Eligible counts include unavailable, skipped and errored subjects but exclude genuinely inapplicable ones and optional capabilities — 700 computed torque channel-episodes out of 700 eligible is complete coverage even when the values are report-only. An unknown binding keeps that uncertainty rather than shrinking the denominator.
- Semantic mapping, capture-origin evidence, visual analysis and behavioural diversity each keep their own coverage count. Capture-origin evidence alone does not establish a usable clock, and visual frame totals or training-window coverage stay unknown where nothing measured them. These counts describe loaded subjects only: a failed episode or a refused source stays explicit. Coverage is not a readiness percentage.
- Metric `availability` is independent of graded `status`; `support` propagates to findings as evidence locations for a reader to check, never as instructions to delete the samples they name.
- Finding `id`, `source_path`, `source_field`, `source_index`, `subject_level`, `evidence_strength` and `calibration` make subjects and promotion decisions inspectable. IDs identify a metric at a source subject; they are not waveform hashes.
- `operational_errors` records computation/payload exceptions. Such a run is partial and exits 2.
- `producer.metrics` and `producer.adapters` identify built-in code and numerical runtime. Third-party implementation identity remains unknown and cannot authorize statistical promotion.
- Additive fields stay under the ongoing schema-7 development version. Historical files are not rewritten.


## Optional deeper diagnostics in 0.7.0

`Report.diagnostics` is an optional version-1 diagnostic envelope with the plan, plan digest, implementation digest and addressed results. Each episode also retains its results in `episodes[].diagnostics`; validation requires agreement. Each result identifies its kind, configured id, episode/source subject, availability, measurements, evidence, support and report-only/review consequence. Window and visual counts reconcile; partial work is not silently complete. Existing schema-7 reports without these fields remain readable.

The optional plan is part of execution identity; diagnostic review thresholds are part of policy identity. `Stream.source_identity` and its graded counterpart carry the adapter source scope even for cameras with no scalar channel bindings. `requirements.min_pass_windows` adds named dataset minimums. Diagnostic comparison changes appear separately in `Comparison.diagnostic_changes`.
