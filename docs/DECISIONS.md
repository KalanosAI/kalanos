# Decisions: what a Kalanos report answers, and what it does not

Since schema 7 a report separates four questions that older reports folded into one boolean. Every surface — JSON, terminal card, HTML, the CLI exit code, a badge — reads the same answers, because each is derived from one place.

## The four answers

| Question | Field | Values | Meaning |
|---|---|---|---|
| **Quality** | `episodes[].score.score` | 0–100 or `null` | What the evaluated diagnostics measured. Descriptive; conditional on the metrics, bindings and evidence available. |
| **Eligibility** | `episodes[].eligibility.status` | `pass` `blocked` `review` `unknown` | Does this episode satisfy the declared requirements of the named scope? The one authoritative decision. |
| **Sufficiency** | `sufficiency.status` | `sufficient` `insufficient` `unknown` | Does the eligible set meet the scope's explicit dataset-level requirements? `unknown` when none are declared. |
| **Readiness** | `readiness.score` | 0–100 or `null` with `reasons` | A policy-specific index over eligibility and quality. Undefined whenever the evidence cannot support it. |

None of these predicts task success. Passing integrity checks does not establish sufficient training data; a high readiness is not a training outcome.

## Eligibility

| Status | Assigned when | Default use |
|---|---|---|
| `blocked` | A finding carries the consequence `block` | Excluded from the eligible pool |
| `unknown` | A required family graded nothing, a stream with channels was not read (payload missing, skipped by the tier, or errored), a required binding is unresolved, the episode could not be graded, or the source declared it but never yielded it | Never auto-approved; the missing evidence is named |
| `review` | Required checks ran; a finding carries the consequence `review` | Awaits a recorded decision |
| `pass` | Required checks ran; no blocking or review reason remains | Eligible, subject to sufficiency |

Precedence is `blocked` > `unknown` > `review` > `pass`. Every reason is kept on `eligibility.reasons` even when another status wins, so a blocked episode still shows what it would otherwise have needed reviewed.

A pass is always relative to `scope.requirements_id` and `scope.policy_id`. The same episode can pass a numeric-only audit and be `unknown` under a vision-required one.

### Severity, consequence, route

A metric's bands assign a **severity** (`warning`, `critical`): an assessment of the measurement. The policy assigns a **consequence** (`block`, `review`, `report_only`): what that assessment does to eligibility. They are recorded separately on every finding.

A `block` names its **route**:

- `contract` — a declared invariant was violated with direct evidence (unreadable required payload, impossible shape, forbidden non-finite values). No calibration needed.
- `statistical` — a threshold detector fired. It may block only when an accepted calibration manifest covers the metric's scope; otherwise it resolves to `review`. Calibration enforcement is always on and cannot be disabled. The legacy `calibrated_metrics` list is not authorization; accepted structured manifests must match the actual detector, thresholds, bindings and scope.

### Prevalence exempts nothing

A blocking finding on every episode of a task is reported as a task trait; one on nearly every episode of the dataset as a dataset trait. Both are descriptive. The episodes stay blocked. The report cannot tell a recording convention from corruption in every episode; a scoped policy rule may exempt a finding explicitly.

## Calibration

A `statistical` block needs a manifest in the policy's `calibration_manifests` list, matched against the finding's exact run context: metric id and taxonomy type, the detector implementation and numerical runtime digest, the complete metric-policy thresholds digest, the effective binding digest after configuration resolution, and a scope digest covering requirements, execution, dictionary, the declared evaluation scope and adapter versions, and the decision policy itself (the manifest list and the deprecated `calibrated_metrics` name list are excluded from that digest, so it never matches against itself).

A manifest must also be `accepted` by a named actor, not future-dated or expired, and declare independent episodes, real-fault validation, combined-policy validation and at least one validation session. The initial gate additionally requires at least 600 independent valid episodes and 100 real fault episodes, an exact one-sided 95% upper confidence bound on false blocks of at most 0.5%, and a recall lower confidence bound of at least 90%.

Kalanos checks these declarations and digests; it does not authenticate the referenced validation report, the reviewer, or the labels behind `independent_episodes` and `real_fault_validation`. A matching accepted manifest is trusted configuration, not cryptographic proof. Any mismatch, revocation, expiry or missing acceptance leaves the finding at `review`.

The report records the exact expected context and any mismatch on each candidate finding, and changing a matched input invalidates the old authorization. The initial gate's evidence must also be representative of the named operating scope: these describe what a manifest must demonstrate to be accepted, independent of how any particular release actually performs. `enforce_calibration: false` is rejected outright, and neither a synthetic benchmark result nor the legacy `calibrated_metrics` name list can grant approval on its own.

## Compatibility fields

`episodes[].score.train_ready` mirrors eligibility exactly: `true` pass, `false` blocked, `null` review or unknown. A report whose `train_ready` contradicts its `eligibility` fails validation. Dataset-level `score.train_ready` is `true` only when every episode passed, `false` when any is blocked, `null` otherwise. Letter grades (`score.grade`, `gate.cap`, `gate.uncapped_grade`) drive nothing.

## Readiness

`readiness.score = sum(quality of pass episodes) / total known episodes`, formula `pass-quality-over-known-inventory-v1`, defined only when:

- the inventory is complete and non-empty;
- every episode is `pass` or `blocked`;
- every `pass` episode has a quality score.

Otherwise it is `null` and `readiness.reasons` says why. A known all-blocked inventory scores 0; a fully passing one scores its mean quality. No assumed values are inserted for missing evidence.

`eligibility_counts` is always published: `total` partitions into the four statuses; failed-to-load episodes and episodes the source declared but never yielded (`inventory.unresolved`) count as `unknown`; `confirmed_eligible_share = pass / total` only when the inventory is complete — a confirmed fraction, not an estimate. A source that declares 50 episodes and yields 49 is incomplete: the 50th is unknown, not absent.

## Sufficiency

Evaluated against explicit requirements in the bundle's `requirements` section. In 0.7.0 only `min_pass_episodes` is evaluable. A known unmet requirement is `insufficient`; a declared requirement whose runner does not exist yet is `unknown`; no declared requirement is `unknown`. Nothing is ever sufficient by omission.

## Scope: the four configuration identities

A bundle (`--profile bundle.yaml`) carries four sections, each with its own identity recorded in `run`:

| Section | Answers | Can a sidecar set it? |
|---|---|---|
| `binding` | What each channel is | Mappings only |
| `requirements` | What a pass needs | No |
| `policy` | How evidence becomes consequences | No |
| `execution` | What this run attempts (`tier`) | No |

Reducing the tier never reduces the requirements: `--tier metadata` reads no numeric payloads, so under `numeric-core` every episode is `unknown` and the default gate fails. Use it to check that a source is readable and how it is bound, not to pass it.

Default scope is `numeric-core-v1`: readable numeric input under an explicit missing-value contract, the `integrity` family graded. It does not silently demand physical units, capture timing, video quality or behavioural diversity.

### Mapping precedence

`--map` > `--map-file` > bundle `binding.features` > discovered `kalanos-map.yaml` > source declarations > inferred. Every displaced assertion is recorded in `binding_conflicts`. Two assertions at one priority that disagree are a configuration error (exit 2), including two `--map` flags for one feature, a duplicated key in a YAML file, and a disagreement at a level a higher level would have overridden. Precedence does not validate: an override outranks a sidecar and is still an assertion. `run.binding` identifies the *effective* mapping after precedence; `run.bundle` identifies the declared file.

### Execution limits

`execution.tier` changes what is attempted; it never reduces `requirements`. `execution.limits` adds `max_bytes` and `max_files` for remote sources, which can only tighten the environment or API limits; the effective budgets are recorded in the execution identity, and an unsupported budget key is a configuration error. Local files keep their existing unrestricted behaviour, and choosing the full tier implies no sampling or caching capability.

### Policy resolution

An explicit Python `policy=` argument wins over a bundle's `policy` section. Otherwise `policy.path` resolves relative to the bundle file, falling back to the configured or packaged default when no path is given. A relative `policy.path` inside an in-memory bundle, which has no file of its own to resolve against, is a configuration error.

### Binding readiness

`episodes[].streams[].channels[].channel.binding` carries a resolved channel's binding in the JSON report; `streams[].declared_channels[].binding` holds it for channels that were not graded. A binding records its origin, conflicts, and active and invalidated validation records, plus prerequisite readiness for numeric inspection, derivatives, limits and noise: whether the evidence a detector needs is in place, ahead of and separate from running that detector or granting calibration approval. Registered jerk and chatter calculations abstain when a channel's derivative prerequisites lack the required scoped binding evidence; generic numeric inspection continues regardless.

## CLI gate

`--fail-on` (default `blocked,unknown`) names the statuses that make `kalanos grade` exit 1. An incomplete inventory — a refused source, or declared episodes that never loaded — counts under `unknown` without inventing an episode count, so the default gate fails an incomplete audit even when every loaded episode passed, or when none loaded at all. A training gate adds `review`. `blocked` alone is exploratory: it permits a partial audit, and prints a warning on stderr naming the refused sources and the undelivered episodes. Exit 2 is reserved for invalid configuration and operational failures and takes precedence.

`kalanos compare` exits 1 when the two reports are not comparable and 2 for invalid inputs or an operational failure. `--hash-source` on `kalanos grade` records a complete local byte identity for later comparison: it reads every local file twice even at the metadata tier, refuses symlinks and nonlocal roots, and withholds the report if the two reads disagree, so write any report output outside the hashed directory to keep the next run's identity unchanged.

## Reading older reports

`kalanos inspect REPORT.json` reads schema 6.3, 6.4 and 6.5 reports without modifying them, names the schema and SHA-256, lists episodes whose `score.train_ready` and gate listing contradict each other, and lists the schema-7 fields the file cannot supply (producer, run, clock provenance, defect intervals, eligibility). It never synthesises them.
