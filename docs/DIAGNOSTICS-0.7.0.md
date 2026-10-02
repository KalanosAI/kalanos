# Deeper diagnostics in 0.7.0

This is an opt-in extension to the stable **0.7.0** release. It uses the existing
`release/0.7.0` branch and package version. It is not a separate prerelease.

The extension measures cross-stream timing, command response, sampled visual
quality, dimensionless motion, training-window validity and comparable-cohort
statistics. It does not repair, align, remove or export data. A diagnostic
measurement alone does not establish training success.

## Configuration and decisions

Add an optional `diagnostics` section to the existing bundle. Run it through the
same library or CLI entry point:

```bash
pip install 'kalanos[numeric,video]'
kalanos profiles validate diagnostics.yaml
kalanos grade ./recording --profile diagnostics.yaml --report audit.json
kalanos inspect audit.json --episode 'recording::episode_000000'
kalanos compare previous.json audit.json --report comparison.json
```

`numeric` installs NumPy for motion/cohort analysis; `video` also installs PyAV.
Missing optional dependencies produce unavailable evidence, never a successful
check. They make an episode unknown when the affected capability is required,
unless a blocking finding takes precedence. Requesting a report-only diagnostic
alone does not add that capability to requirements.
The metadata tier records requested payload diagnostics as skipped.
Defaults remain the existing numeric audit when no diagnostic plan is supplied.

The four existing responsibilities remain separate:

- Bindings and scoped validations identify channels, units and command semantics.
- Requirements declare what evidence and minimum data quantities are needed.
- Decision policy holds review thresholds under `diagnostic_reviews`.
- Execution selects the diagnostic plan, tier and enforced work budgets.

The plan is included in the execution identity, and review thresholds in the
policy identity. Changes therefore invalidate comparisons/calibration relying on
the previous configuration. All analysis implementation changes also invalidate
existing detector identities, including otherwise unchanged metrics.

There are **no universal new fault thresholds**. All diagnostic results default
to report-only. An explicit diagnostic review policy can produce a review finding
which enters the existing, single eligibility calculation without changing the
quality score. Version 1 of these new diagnostics has no automatic statistical
blocking path, including when a manifest exists for another detector. Existing
core calibration and deterministic blocking continue to apply normally.

For example, construct a deployment policy from the default policy, then save it
as YAML and reference it from the bundle's `policy.path`:

```python
import yaml
from pathlib import Path
from kalanos.assets.policy import load_default_policy
from kalanos.analysis.models.diagnostics import DiagnosticReviewPolicy

policy = load_default_policy()
policy.diagnostic_reviews = DiagnosticReviewPolicy(
    timing_max_unmatched_fraction={"camera_state": 0.05},
    tracking_max_abs_error={"arm_response": 0.1},
    vision_max_clipped_fraction=0.98,
    review_video_integrity=True,
    review_motion_limits=True,
)
Path("deployment-policy.yaml").write_text(
    yaml.safe_dump(policy.model_dump(mode="json"), sort_keys=False)
)
```

These numbers illustrate configuration only; they are not validated operating
thresholds. Unknown policy keys are rejected. Missing required evidence produces
unknown; operational errors are retained and CLI exit 2 takes precedence. A gate
including review remains `--fail-on blocked,review,unknown`.

## Stream selection and clock evidence

A selector contains the original `feature` field, optional original channel
`index`, exact `instance` (null means the uninstanced stream), and optional exact
`source_path`. Ambiguous or missing selectors abstain. Do not infer channel
identity from a display name or vector length.

Resolved streams, including cameras without scalar channels, retain the adapter
source's `source_identity`. This is the exact source scope to use in clock
relations; it is also present on graded streams. Directly constructed streams
without a resolved identity fall back to their channel scope or source path.

A clock relation declares left/right source scopes, a domain, external evidence,
offset, drift and anchor. The transform maps right native seconds to left native
seconds:

```text
mapped_right = anchor + (right - anchor) * (1 + drift_ppm / 1e6) + offset_s
```

The runner restores source order and subtracts native integer ticks before
floating conversion. It never reorders or changes the source payload. Unknown
units, invalid/reset time axes or absent order maps abstain. A capture-alignment
claim additionally requires producer capture evidence on both streams. A
recorded-alignment claim remains explicitly about the recorded timeline.

Scopes and evidence are trusted declarations, not authenticated external proof.
Equal timestamp units or domains alone do not establish synchronized acquisition.

## Timing and synchronization

Each declared pair reports overlap, matched and unmatched left samples, signed
median skew, absolute p95 skew and p95 causal sample age. Matching is nearest,
with earlier-sample ties; many-to-one matches are explicitly permitted. These
are correspondence diagnostics, **not measured transport latency**.

The tolerance is a matching parameter; policy can request review above an
unmatched fraction. Reports retain matched source-row pairs and unmatched-row
support. Different-rate cameras or action/state streams can use separate pairs.

Offset/drift fitting requires at least three explicit, one-to-one corresponding
event row pairs. The fit reports remaining offset, drift, residual RMS and slope
standard error after the configured transform. It is never applied to the input.
Nearest-neighbour pairs are not silently promoted to synchronization events.

## Command response

Tracking requires validated identity, quantity, native unit, frame and continuous
representation for both channels, plus validated command semantics for the action.
Validation values and source scopes must match the current bindings. Both sides
must share the same quantity/unit/frame.

- `absolute`: compare a position target with measured position.
- `rate`: compare a velocity target with measured velocity.
- `delta`: compare against state at command issue plus the declared delta, using
  a positive response horizon.

Both raw and declared-delay-adjusted RMSE are retained, with p95 absolute error,
mean signed error, matched/unmatched counts and source-row correspondences. No
best-fitting delay, sign change, unit conversion or channel remapping is applied.
Wrapped angles must be provided as an explicitly validated continuous/unwrapped
representation for this first tracking implementation.

A large discrepancy can mean contact, controller dynamics, mapping error or a
recording fault. The configurable native-unit threshold requests review; it does
not identify the cause. Forward-kinematics/model-residual checks remain outside
this first implementation because no robot-model loader is supplied.

## Sampled video

The `vision` diagnostic decodes nothing of its own: it publishes the vision
metrics' shared read of each camera, recorded on the graded stream as
`GradedStream.frames`, so a diagnostic run and a normal run measure the same
frames the same way. Set `diagnostics.vision: true` to publish it. How every
camera is read is configured once, in the bundle's top-level `vision` section:

```yaml
vision:
  sample_frames: 10        # evenly spaced frames per camera
  full_frame_scan: false   # read every frame instead
  max_decode_frames: 20000
  max_pixels: 2097152
diagnostics:
  vision: true
```

`--vision-samples` and `--full-frame-scan` override `KALANOS_VISION_SAMPLES`
and `KALANOS_FULL_FRAME_SCAN`, which override the bundle; `--tier full` always
reads every frame. The resolved settings, command line and environment included,
are part of the execution identity when they differ from the defaults. Depth
streams are not cameras and get no `vision` result.

A sampled read counts decoded frames, including seek preroll, and stops at
`max_decode_frames`; a full scan reads every frame. Any read stops before
measuring a frame larger than `max_pixels`. The pixel limit does not promise a
cap on a decoder's internal codec allocation. A stopped read keeps the frames
it already measured: a cap leaves the result unavailable, and a decoder failure
after the first frame makes it an operational error. A file that fails before
any frame decodes leaves the result unavailable.
A stream whose rows the adapter reordered is not decoded.

Evidence records requested, examined and missing rows, the sample plan
(`evenly_spaced_v1` or `full_scan_v1`), presentation time, native-luminance
Laplacian-variance blur, the share of pixels at or below `dark_level` plus the
share at or above `bright_level`, and the luminance `shape` as
`[height, width]`. Frames the sample would examine, and frames
`exposure_shift_pct` judged bad, carry `luma_sha256`, a SHA-256 of their native
luminance. Identical adjacent pairs come from the frozen-frame read: a pair with
no difference at all is recorded as a candidate; a static scene is not
automatically a camera failure.

`segment_frames` is `frame_count_vs_timebase`'s count of the container's packets,
on every read. It is reconciled against
the declared frame count. Selected-frame PTS interval differences are compared
with recorded intervals; this is not acquisition alignment. Shape/count
anomalies can request review under `review_video_integrity`. Blur/exposure
require explicit policy thresholds. `vision_min_blur_score` compares each frame's
raw Laplacian variance with one fixed value, and that variance differs by two
orders of magnitude between cameras (6 to 1821 on the calibration datasets in
`docs/METRICS.md`), so one value rarely fits more than one camera.

`sampled_video_quality` requires all discovered cameras to finish their requested
sampled audit. Existing `video_quality` requires **every declared frame** to have
been examined. A small sample cannot satisfy the full-visual requirement.
Unsupported embedded-image payloads report unavailable until their adapter
implements the `SampledFrames` interface; built-in support is the video payload
used for separate, shared and remote files.

## Motion shape and limits

`position_dimensionless_jerk_v1` is a new measurement; it does not redefine the
existing report-only jerk metrics. On each sufficiently long, contiguous regular
segment it computes:

```text
duration^5 * integral(jerk^2 dt) / peak_to_peak(position)^2
```

Uniform duration/amplitude scaling is covered by tests. Sampling, segmentation
and derivative discretization still matter. The runner reports each segment,
stationarity, maximum absolute step/velocity and velocity sign changes. Nulls,
NaNs, infinities, missing source rows and excessive gaps split segments. No
interpolation fills gaps. Stationary segments retain evidence with undefined
smoothness rather than infinite or fabricated zero smoothness.

Position semantics and derivative prerequisites need scoped validation. Wrapped
angles require a declared native-unit period. Existing binding limits are only
used with matching scoped validation. Optional maximum velocity needs an external
limit evidence reference. Limit observations become review only when the policy
requests it. No universal smoothness or oscillation fault threshold is introduced.

## Training windows

A window declares an anchor stream, required modalities and any cross-clock
relations, sample rate, history/prediction steps, stride, maximum age and maximum
gap. History includes the anchor step; prediction begins at the next step.
Nearest or causal-previous matching is explicit. The supported padding behavior
is `reject`; interpolation is `none`. No mask or interpolation is manufactured.

Window matching uses rational arithmetic for native timestamps, declared clock
scales/transforms, the grid rate, sample age and gap limits. Float inputs retain
their recorded decimal values; no tolerance is added to causal comparisons. This
avoids a converted 300,000,000 ns sample falling just after a 0.3 s grid step due
to multiplication roundoff, while keeping a genuinely future sample ineligible
for previous matching. JSON evidence still serializes relative times as numbers.

The grid remains within an episode. Each candidate records grid bounds, consumed
source rows and pass/blocked/review/unknown reasons. Null/nonfinite required
payloads, unmatchable samples and source gaps violate the window contract.
Consequential source findings propagate to intersecting windows; whole-episode
findings apply to the corresponding subject throughout. Subject matching includes
the consumed channel: a fault on an unselected vector index does not propagate,
while whole-vector selection still includes findings on every consumed channel.
Required video frames
not actually decoded remain unknown even when timestamps exist.

`max_windows` and `max_probes` bound work. Unevaluated candidates remain in the
unknown denominator. Counts partition every candidate; zero candidates is not a
claim of useful training volume. Window passes are input-contract observations,
not an export selection or a claim that overlapping samples are independent.

When loading results, window records must be complete for the examined count,
carry unique grid-start addresses and valid bounds/statuses, and reconcile with
the summary. Unknown counts include both examined unknown windows and explicitly
unexamined candidates. Contradictory reports are rejected rather than silently
repairing or trusting a summary.

Declare minimums separately in requirements:

```yaml
requirements:
  id: training-input-v1
  min_pass_windows:
    train: 10000
```

Sufficiency counts contract-passing windows only from passing episodes. A known
lower bound meeting the minimum is sufficient for that requirement; a possible
unexamined shortfall is unknown. Known failures dominate unknown checks. These
minimums do not change the CLI's episode-status gate into a sufficiency gate.

## Dataset execution

The dataset runner consumes named cohorts after episode decisions. A cohort
explicitly identifies episodes, robot configuration, task, evidence and native
feature ranges. Features must share one resolved source grid and validated
identity/quantity/unit/continuous representation. Known task labels must match.
It does not automatically establish that the declared physical robot was used.

Measurements include fixed-bin occupancy/entropy, constant features, effective
dimensionality, normalized trajectory similarity with duration tolerance, exact
selected-feature trajectory duplicates and robust distances of episode means.
The trajectory digest covers the selected values/relative times, not source-file
bytes. Similarity is descriptive, not semantic equivalence or permission to delete.

Raw finite and passing-only summaries remain separate. Nonfinite rows are excluded
and counted; affected trajectories are excluded from interpolation-based
similarity. Phase fractions require reviewed, nonoverlapping source-row labels;
unlabelled samples remain counted. Low velocity does not imply contact.

At least three explicitly declared independent sessions enable a reproducible
session-mean percentile bootstrap with recorded seed/resamples. Otherwise
uncertainty is unavailable. These approximate descriptive intervals do not prove
independence or validate detector false-positive rates. Pair-comparison budgets
record the unexamined pair count and partial result status.

This is a purpose-built cohort runner; general third-party dataset-metric plugin
execution is not introduced. The legacy `behavioral_diversity` requirement is not
silently interpreted as satisfied by descriptive cohort statistics. There is no
universal sufficient-diversity threshold.

## Review and validation evidence

Visual evidence can include bounded embedded PNG previews (`preview_frames`,
`preview_size`; set `preview_frames: 0` to omit images). HTML shows these previews
and a bounded expected/observed command-response plot with source references.
These are inspection aids, not interactive synchronized playback.

JSON/YAML retain the complete identified diagnostic report and per-episode
results. HTML contains expandable evidence; terminal output summarizes states;
`inspect` prints the selected diagnostic records. Comparison includes diagnostic
differences and rejects changed/missing plan or implementation identities.

```bash
kalanos diagnostics summarize-study labelled-outcomes.json > study-draft.json
```

Input is a JSON array with `episode_id`, `session_id`, `split` (`tuning` or
`validation`), `truth` (`valid` or `fault`), `decision`, an external `evidence`
reference, and optional `real_fault`. Duplicate episode IDs and shared tuning /
validation sessions are rejected. The summary counts held-out outcomes, treating
only actual blocking decisions as detected blocking faults. It is always a draft
and cannot authorize blocking. Human review must establish labels, independence,
operating scope and final implementation/configuration identities.

The patch adds engineering controls, not a production-labelled validation corpus.
Synthetic positive/negative controls exercise mechanics. Production promotion of
these new detectors, learned transition models, automatic repairs, robot-model
kinematics, rich synchronized playback and selection/export remain separate work.
