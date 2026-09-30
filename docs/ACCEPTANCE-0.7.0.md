# 0.7.0 acceptance record — 30 September 2026

Core base: `e40467bfd071f22a63d0f7a4b15f0a44ff61ac09`, the verified
`release/0.7.0` head. Measured with the 0.7.0 implementation before the version
number was set; no package was published when these audits ran.

## Pinned public data

| Dataset | Revision | Episodes |
|---|---|---:|
| `lerobot/aloha_static_towel` | `13ad96f5ed0e219e48c72471626a2e3d0eb5aeff` | 50 |
| `lerobot/aloha_static_battery` | `06dc3da83c4fd3d1889b00f1dfd3780da8421f64` | 49 |

The same locally materialized metadata and numeric parquet files were used for
baseline and candidate runs at standard tier, `numeric-core-v1`. Videos were not
materialized or decoded. The companion evidence archive records per-file SHA-256
and byte size for this subset. These are not whole remote-dataset byte digests.
The same implementation was also run from its installed wheel.

| Dataset | Baseline pass / blocked / review / unknown | Candidate pass / blocked / review / unknown | Candidate readiness |
|---|---|---|---|
| Towel | 0 / 0 / 0 / 50 | 39 / 0 / 11 / 0 | Undefined: 11 reviews |
| Battery | 0 / 0 / 0 / 49 | 1 / 0 / 48 / 0 | Undefined: 48 reviews |

Both inventories reconcile. Baseline numeric capability was unavailable because
Boolean `next.done` failed the numeric-only missing-value check. Counting Boolean
nulls makes that check computable and resolves the capability. Review findings
remain; they are not silently exempted. Default `blocked,unknown` gating returns
0, while a gate including review rejects these datasets.

Descriptive quality changed from 97.1904 to 97.6587 (towel) and 92.0936 to 94.7290
(battery), partly because the Boolean channel now contributes a measured missing
value result. These are explanatory before/after observations, not a supported
strict score comparison: detector identities changed. Neither dataset has numeric
readiness, accepted production noise references, video quality evidence or
validated production calibration from this exercise.

## Candidate checks

- Default suite plus targeted applicability, missing Boolean and calibration
  promotion regressions; installed-package integration suite.
- Build wheel and sdist, strict Twine metadata/render checks and matching
  package/CITATION/tag validation.
- Companion publisher: TypeScript tests/typecheck, Python conversion using the
  installed wheel and locked historical dependency, additive SQLite migration,
  active badge and queue routes, and Worker dry-run build.
- Companion Action: actual installed-wheel CLI runs, one-audit assertion,
  canonical counts/null readiness, default/strict review gates, HTML/JSON output,
  stale-output handling and partial operational failure retention.

See the delivered verification Markdown for exact local test counts and tool
versions. Supported-Python GitHub CI, Linux Docker execution, PyPI publication and
live consumer rollout remain external release steps. No production false-positive
rate, recall, training success, cross-stream timing or visual quality claim is
established by these checks.
