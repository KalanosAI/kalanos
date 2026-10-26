# Report schema

This page lists what changed from 6.5 to 7.0.0 (written by Kalanos 0.7.0), what a consumer must do, and what a legacy file can and cannot provide, the 7.1.0 and 7.2.0 additions, and the 7.3.0 changes below.

Report loading rejects duplicate episode IDs and reconciles each status count against episode decisions, adding identified failures and unresolved inventory to unknown. A published eligible share must match those counts; an omitted share remains readable. Diagnostic window counts also reconcile with the recorded window statuses and explicit unexamined-budget unknowns. Valid report shapes are unchanged; contradictory summaries are rejected rather than silently repaired.

## Changed in 8.0.0

- `reason_codes`, the report's first key, lists each distinct reason text once.
- Every value under a key named `reason`, `reasons` or `detail`, at any depth, holds an index into `reason_codes`: a string becomes an integer, a list of strings a list of integers, and the keys of a `reasons` map (the coverage rows) become integers written as strings. `null` stays `null`.
- Codes are per report. The same text can have a different code in another report.
- A field whose value is `null`, `[]` or `{}` is left out when its default is that same value. This includes top-level lists such as `findings`, `skipped` and `unresolved`. A field that is nullable but has no default is still written as `null`, and a value inside a free-form `evidence` or `calibration` object is written as it is.

## Changed in 7.3.0

- `GradedEpisode.diagnostics` is removed. Each diagnostic result is stored once, in `Report.diagnostics.results`, addressed by `episode_id`, `kind` and `id`.
- `thumbnail_png_base64` is no longer written, neither in a vision diagnostic's `evidence.frames[]` nor in a metric's `evidence`. A vision result embeds one WebP preview of its blurriest sampled frame, in that frame's `thumbnail_webp_base64`, and names the frame in `measurements.preview_row` (`null` with `preview_frames: 0` or when the frame cannot be encoded). Each frame keeps its `source_row` and `presentation_time_s`, and the video file is the result's `subject.source_path`.
- `duplicates` is no longer a list of pair objects. It is a `DuplicateEpisodes` object that names episodes by their index in `episodes`: `exact` holds each group of identical episodes, and `near` stores each near pair once, under its earlier episode, with its distance to 4 significant digits. A pair no longer carries `footage_identical` or `cameras`; read shared footage from `cameras[].repeated_footage`. A 7.0.0 to 7.3.0 report that lists pair objects is converted on load, its distances rounded to 4 significant digits.
- `duplicates` lists every near pair up to 1.5 million. Past that, it keeps a near pair only when it is among the 5 nearest of either episode, and `near_duplicates.pairs_listed` falls short of `pairs_near`.
- An `integrity.near_duplicate_episode` finding no longer lists partners: its evidence is `max_nrmse`, the limit, and `nearest_nrmse`, the distance to its nearest partner to 4 significant digits. `partners`, `nrmse`, `footage_identical` and `partners_omitted` are gone from it, and `value` still counts every episode it is near. Its pairs are in `duplicates.near`.
- An `integrity.duplicate_episode` finding's evidence is empty: `partners` and `footage_identical` are gone, since `duplicates.exact` holds the group and `cameras[].repeated_footage` the repeated footage. `value` still counts the other copies.
- When the near tier is saturated (see `NearDuplicateSummary.saturated`), no `integrity.near_duplicate_episode` finding is written.

| Field | Type | What it is |
|---|---|---|
| `near_duplicates` | `NearDuplicateSummary` or `null` | How many pairs the near-duplicate search found, over the whole dataset. `null` when the search did not run: numpy is missing, or no group of comparable episodes holds two that differ in their payloads and could be resampled (every value finite, at least 2 rows). |
| `NearDuplicateSummary.limit` | `float` | The largest normalized RMSE at which a pair counts as near. |
| `NearDuplicateSummary.max_partners` | `int` | The most near partners each episode keeps in `duplicates` when `pairs_listed` is less than `pairs_near`. A near finding lists no partner. |
| `NearDuplicateSummary.episodes_compared` | `int` | The episodes the near search compared: the first of each group of exact copies. |
| `NearDuplicateSummary.pairs_compared` | `int` | The pairs among them, summed over the groups of episodes that hold the same streams and channels. |
| `NearDuplicateSummary.pairs_near` | `int` | Of those, the pairs within `limit` whose row counts also agree. |
| `NearDuplicateSummary.pairs_listed` | `int` | The near pairs in `duplicates`: every one of `pairs_near`, or, when there are too many to list, those among the nearest `max_partners` of either episode. |
| `NearDuplicateSummary.episodes_with_partner` | `int` | The episodes, exact copies included, with at least one near partner. |
| `NearDuplicateSummary.median_nearest_nrmse` | `float` or `null` | The median, over compared episodes with a near partner, of the distance to their nearest one. `null` when `pairs_near` is 0. |
| `NearDuplicateSummary.saturated` | `bool` | Whether more than half the compared pairs are near and there are more near pairs than compared episodes. The near tier then does not separate copies from re-runs in this dataset; the exact tier still applies. |

A 7.2.0 report still loads: its `episodes[].diagnostics` is ignored, any `thumbnail_png_base64` stays in `evidence` as it was written, and `near_duplicates` is null.

## Added in 7.2.0

| Field | Type | What it is |
|---|---|---|
| `GradedStream.video_format` | `CameraFormat` or `null` | The camera's declared and container format, stated whether or not they agree. `null` when the stream is not camera footage backed by a video file. |
| `CameraFormat.declared` | `VideoFormat` | What the dataset's own metadata declares for the camera. |
| `CameraFormat.found` | `VideoFormat` | What the video file's container header and tags state for its first video stream, covering the whole file. Every field `null` at the metadata tier, which opens no media, or when the header cannot be read. |
| `VideoFormat.width`, `VideoFormat.height` | `int` or `null` | The frame size in pixels. `null` where the source states none. |
| `VideoFormat.fps` | `float` or `null` | The frame rate. `null` where the source states none. |
| `VideoFormat.pix_fmt` | `str` or `null` | The pixel format, as FFmpeg names it (`yuv420p`). `null` where the source states none. |
| `VideoFormat.codec` | `str` or `null` | The video codec: FFmpeg's name on `found` (`h264`), a LeRobot feature's `video.codec` on `declared`. `null` where the source states none. |
| `VideoFormat.profile` | `str` or `null` | The codec profile (`High`). `found` only. |
| `VideoFormat.bit_rate` | `int` or `null` | The video stream's bitrate in bits per second. `found` only. |
| `VideoFormat.frame_count` | `int` or `null` | How many frames the video stream holds. `found` only. |
| `VideoFormat.duration_s` | `float` or `null` | The video stream's duration in seconds. `found` only. |
| `VideoFormat.color_range`, `VideoFormat.color_primaries`, `VideoFormat.color_transfer`, `VideoFormat.color_space` | `str` or `null` | The colour range, primaries, transfer characteristic and colour space, as FFmpeg names them (`tv`, `bt709`). `null` when the container leaves a code unspecified. `found` only. |
| `VideoFormat.channels` | `int` or `null` | How many channels each frame holds, from a LeRobot feature's `video.channels`. `declared` only. |
| `VideoFormat.is_depth_map`, `VideoFormat.has_audio` | `bool` or `null` | Whether the camera records depth, and whether its files carry audio, from a LeRobot feature's `video.is_depth_map` and `has_audio`. `null` unless the feature states a JSON boolean. `declared` only. |
| `VideoFormat.camera_make`, `VideoFormat.camera_model` | `str` or `null` | The camera maker and model a container tag names, such as `com.apple.quicktime.make` or `com.android.model`. `found` only. |
| `VideoFormat.encoder` | `str` or `null` | The software that wrote the file, from its `encoder` tag (`Lavf62.12.102`). `found` only. |
| `GradedEpisode.split` | `str` or `null` | The dataset split the source assigns the episode, such as `train`. Read from a LeRobot `info.json`'s `splits` ranges; `null` when the source assigns none, or when the episode falls in no range or in more than one. |
| `Channel.source_dtype` | `str` or `null` | The dtype the source format declares for the channel's values: a LeRobot feature's `dtype`, an HDF5 dataset's dtype, or an MCAP message field's ROS type. `null` for CSV, delimited, JSON, JSONL and tabular sources, whose types are inferred, and wherever the format states none. Under `streams[].channels[].channel` and `streams[].declared_channels`. |

`found`'s `bit_rate`, `frame_count` and `duration_s` describe the whole video file. A LeRobot v3.0 dataset stores several episodes in one file, so each of those episodes reports the same values. Of the container's tags, only the camera make, model and encoder are read, and no location is recorded. A field one side can never state stays `null` on that side.

A 7.1.0 report loads with `video_format`, `split` and every `source_dtype` null, and a `video_format` that lacks the codec, encoding and source fields loads with them null. `kalanos inspect`, `kalanos compare` and `kalanos.load_report` read any 7.x report up to the current version, and refuse a newer one.

## Added in 7.1.0

| Field | Type | What it is |
|---|---|---|
| `family_weights` | `dict[str, float]` | The policy's family weights the run graded with, keyed by family id. Empty when the policy weights no family. |
| `MetricResult.points` | `float` or `null` | The 0-100 the policy graded the result to, on every channel, stream and episode metric. `null` when the result stayed `not_applicable` or `report_only`. |
| `ScoreResult.metrics` | `dict[str, MetricRollup]` | One rollup per metric that graded, keyed `family.metric`. Filled on `episodes[].score` and the dataset `score`; empty on channel and stream scores. |
| `MetricRollup.level` | `channel`, `stream` or `episode` | Where the metric ran. |
| `MetricRollup.points` | `float` or `null` | In an episode, the points of the worst channel, stream or camera it graded. `null` at dataset level. |
| `MetricRollup.n_graded` | `int` | How many subjects it graded. At dataset level, distinct channels, streams or cameras across every episode. |
| `MetricRollup.n_flagged` | `int` | Of those, how many graded `warning` or `critical`. |
| `MetricRollup.n_episodes_flagged` | `int` or `null` | At dataset level, how many episodes it flagged a subject in. `null` in an episode. |
| `ScoreResult.family_tails` | `dict[str, FamilyTail]` | Per family, how many episodes scored it below the policy's cutoff and the lowest of them. Filled on the dataset `score`; empty on every other score. |
| `FamilyTail.below` | `float` | The policy's `low_episode_score` the count compared against, 70 by default. |
| `FamilyTail.n_below` | `int` | How many episodes scored the family strictly below `below`. |
| `FamilyTail.n_scored` | `int` | How many episodes scored the family. An episode with nothing to score in it is left out. |
| `FamilyTail.lowest` | `float` | The lowest episode score in the family. |
| `duplicates` | `DuplicateEpisodes` | The episodes whose numeric payloads are identical or nearly so, as indices into `episodes`. Which pairs also repeat their footage is read from `cameras[].repeated_footage`. |
| `DuplicateEpisodes.exact` | `list[list[int]]` | Each group of episodes whose payloads hash identically, earliest first, the groups ordered by their earliest episode. The earliest stands for its group in `near`. |
| `DuplicateEpisodes.near` | `NearPairs` | Every listed near pair, each stored once. |
| `NearPairs.episode` | `list[int]` | The earlier episode of each entry's pairs, ascending. Only episodes with a later near partner have an entry. |
| `NearPairs.partners` | `list[list[int]]` | The later episodes near `episode[k]`, ascending. |
| `NearPairs.nrmse` | `list[list[float]]` | The normalized RMSE of each pair, aligned with `partners`, to 4 significant digits. |

An episode's `score` and `families` now come from its metrics, each counted once at its worst subject and weighted within its family. A family scores only when one of its `core` metrics (a new `MetricPolicy.core` flag) graded in the episode. Channel and stream scores keep their own fold and are for display.

A 7.0.0 report loads with `family_weights`, `duplicates`, every `metrics` and every `family_tails` empty, and every `points` null.

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
6. Decode every `reason`, `reasons` and `detail` value through `reason_codes` before showing or comparing it.
7. Read a missing key as `null`, or as an empty list or map when the field is a list or map.

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
- Additive fields stay under the ongoing schema-7 development version, and 7.3.0 also removes fields. `kalanos inspect`, `kalanos compare` and `kalanos.load_report` still read every 7.x report up to 7.3.0. Historical files are not rewritten.


## Optional deeper diagnostics in 0.7.0

`Report.diagnostics` is an optional version-1 diagnostic envelope with the plan, plan digest, implementation digest and addressed results. Each result identifies its kind, configured id, episode/source subject, availability, measurements, evidence, support and report-only/review consequence. Window and visual counts reconcile; partial work is not silently complete. Existing schema-7 reports without these fields remain readable.

The optional plan is part of execution identity; diagnostic review thresholds are part of policy identity. `Stream.source_identity` and its graded counterpart carry the adapter source scope even for cameras with no scalar channel bindings. `requirements.min_pass_windows` adds named dataset minimums. Diagnostic comparison changes appear separately in `Comparison.diagnostic_changes`.
