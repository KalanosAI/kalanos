# Report schema 7.0.0

Written by Kalanos 0.7.0. This page lists what changed from 6.5, what a consumer must do, and what a legacy file can and cannot provide.

## New top-level fields

| Field | Type | What it is |
|---|---|---|
| `producer` | `Producer` | Package name and version, build revision when the environment supplies `KALANOS_BUILD_REVISION`, adapter and metric implementation versions (populated by later work packages). |
| `run` | `RunInfo` | Run id, start/finish, completion, execution tier, `source` evidence, and the four configuration identities (`requirements`, `policy`, `binding`, `dictionary`) each with a content digest. |
| `scope` | `EvaluationScope` | `requirements_id`, `policy_id`, `binding_id`, `tier`. Every decision is relative to this. |
| `inventory` | `Inventory` | Episodes expected (when the source declares a count), loaded, and failed with reasons; whether enumeration is believed complete. |
| `eligibility_counts` | `EligibilityCounts` | `total`, `pass_count`, `blocked`, `review`, `unknown`, `inventory_complete`, `confirmed_eligible_share`. The counts partition `total`. |
| `readiness` | `Readiness` | `formula_id`, `score` or `null`, `reasons`, `passing_quality`. Replaces the 6.x shape. |
| `sufficiency` | `Sufficiency` | `status` and per-requirement `checks`. |
| `binding_conflicts` | `list[BindingConflict]` | Every feature where a lower-precedence mapping source disagreed with the winner. |

## New per-episode and per-finding fields

| Field | What it is |
|---|---|
| `episodes[].eligibility` | `status`, `scope_id`, `policy_id`, `reasons[]` (each with `id`, `kind`, `status`, `consequence`, `route`, `detail`). Required on every episode. |
| `findings[].consequence` | `block`, `review` or `report_only`. |
| `findings[].route` | `contract` or `statistical` when `consequence` is `block`. |
| `findings[].support` | `kind` (`whole_episode` or `intervals`), `index_space`, `intervals[]` with zero-based half-open `start`/`end_exclusive` and optional wider `support_start`/`support_end_exclusive`. Metrics that measure the whole episode carry `whole_episode`; no interval is ever invented. Existing metrics emit `whole_episode` until R07-05 localises them. |

## Changed semantics

- `episodes[].score.train_ready` mirrors `eligibility` (`true`/`false`/`null`). No score threshold sets it. Dataset `score.train_ready` is a compatibility summary. Validation rejects a contradiction.
- `gate.failing_episodes` is exactly the set of `blocked` episodes; validation rejects any other set.
- `gate.task_traits` and `gate.dataset_traits` are descriptive; they no longer exempt episodes from blocking.
- `gate.pruned_score` is the mean quality of non-blocked episodes: a candidate description, not a sufficiency claim.
- Letter fields remain and drive nothing.

## Removed

- `gate.pruned_grade`
- `gate.train_ready_after_pruning`
- `readiness.evaluated_episodes`, `readiness.passing_episodes`, `readiness.blocking_episodes` (use `eligibility_counts`)

## Domain model additions (not serialised in the report yet)

- `Stream.clock_info: ClockInfo` — `origin` (`capture`, `receive`, `publish`, `log`, `presentation`, `generated`, `simulation`, `unknown`), `origin_evidence` (`producer`, `adapter`, `inferred`, `none`), native unit/dtype, transforms. `ClockInfo.from_legacy(Clock.RECONSTRUCTED)` yields inferred generation, never certified capture. R07-04 populates it from adapters and exposes it in the graded report.
- `Stream.source_order: SourceOrder` — whether rows are in source order and the index map back when not. R07-04 makes the LeRobot adapters fill it before sorting.
- `ChannelBinding` — `actuator`, `quantity`, `representation`, `unit`, `command`, `device`, `origin`, `status`, `validations[]`. R07-02 routes metrics through typed views built from these.

## Policy additions

- `MetricPolicy.consequence` (`block`/`review`/`report_only`, default block for critical) and `MetricPolicy.route` (`contract`/`statistical`, default statistical).
- `Policy.enforce_calibration` (default `false`) and `Policy.calibrated_metrics`. When on, a statistical block with no manifest resolves to review.

## Bundle file (`--profile`)

```yaml
schema_version: 1
binding:
  id: my-robot-v1
  features:                      # whole-feature assertions, the legacy shape
    observation.state: proprio.joint_position
  channels: []                   # per-channel semantics (R07-02 consumes these)
requirements:
  id: numeric-core-v1
  required_families: [integrity]
  require_resolved_bindings: false
  min_pass_episodes: null
policy:
  id: default-decisions-v1
  path: null                     # a policy file, or the packaged default
execution:
  tier: standard                 # metadata | standard | full
```

Every section is optional; the defaults are the built-in scope.

## Consumers: what to change

1. Read `eligibility_counts` and `readiness.score` (which may be `null`; show `readiness.reasons`), not `score.train_ready` or a letter.
2. Gate CI on the exit code of `kalanos grade --fail-on ...`, or on `eligibility_counts` from `--json`.
3. Show `scope.requirements_id` beside any number: a readiness is meaningless without its scope.
4. To read a 6.x file, use `kalanos.analysis.models.legacy.load_any`; it returns a `LegacyReport` with the original data, its SHA-256, contradictions and the fields it cannot supply. Do not convert legacy values into schema-7 decisions.

## Fixtures

- `tests/legacy_reports/lerobot_v3_tiny-6.5.0.json` — produced by the release branch at `18cf9b6` before this change; real branch output.
- `tests/legacy_reports/aloha_static_towel-048fef2-6.4.0-trimmed.json` — four episodes from the 048fef2 towel report (original SHA-256 `8001d3ce…`), two of them carrying the `train_ready: true` + gate-failing contradiction. Dataset-level fields are the original 50-episode values and intentionally do not reconcile with the four episodes; the fixture tests loading, not arithmetic.
- A schema 6.3 fixture is not in the repository yet: the 43e0cb1 reports were not available when this change was prepared. Add one from the original files before R07-06 ships `compare`.
