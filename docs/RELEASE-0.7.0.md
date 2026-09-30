# 0.7.0 release contract

This release completes the final three 0.7.0 work items: contextual SNR (R07-03),
real-data acceptance and schema-7 consumers, and release packaging. It does not
supply a production calibration corpus or certify training success.

## Contextual SNR

`integrity.snr_db` remains a five-sample smooth/residual ratio, not measured sensor
SNR. Reports retain the ratio, signal/residual variance and standard deviation,
window count, invalid sample count and whole-episode support. Invalid samples
break windows; source ordering and native clock units must be usable. Known
reward, annotation, metadata, Boolean/binary and discrete channels are excluded.

A `ChannelBinding.noise_floor` can supply a residual standard-deviation reference
in the channel's native units. It records reference provenance, sensor
configuration, sample rate, bandwidth, estimator and optional scale transform.
Units alone, inferred semantics and a low-amplitude trace cannot validate it.
Scoped validation records must match identity, quantity, unit and the complete
noise-floor object. Continuous representation, native scale and sample rate must
match. The validator must verify that the referenced sensor configuration and
bandwidth describe this acquisition; Kalanos cannot verify an external document.
A manufacturer's total RMS specification needs a documented conversion to the
named residual estimator before it can serve as this reference.

- Within a validated reference: preserve measurements, report-only, no SNR penalty.
- No usable reference: ratio remains diagnostic; a critical candidate is review.
- Above reference: candidate remains review unless an accepted calibration
  manifest matches detector, thresholds, binding and operating scope.
- A matching manifest alone cannot promote unassessed SNR to a statistical block.

For a quiet robot hold, this prevents a small ratio from being treated as a noise
failure when measured residuals are within a verified reference. For a trace with
large corruptions, the residual evidence remains visible. “Within reference” is
not proof of sensor health; other checks retain their own results.

Boolean payloads now receive missing-value checks. The towel and battery
`next.done` field previously made numeric coverage unknown despite being read:
Boolean was excluded by the numeric dtype predicate. Null Booleans now count as
missing; this does not make Boolean data eligible for SNR.

## Consumer contract

The companion publisher patch preserves scope, counts, readiness, sufficiency,
coverage, episode decisions and finding consequence/support in `decision_summary`.
Undefined readiness stays null across HTML, Markdown, search, sharing, badges
and social previews. Schema-7 consumers never rebuild eligibility from severity,
letters or a quality threshold. Historical cards keep their historical fields;
they are not silently converted to schema 7.

The Action runs one canonical `kalanos grade` call and preserves exit codes:
0 accepted under the requested gate, 1 gate failure, 2 operational/configuration
failure. `--fail-on blocked,unknown` is the diagnostic default;
`blocked,review,unknown` is the explicit unattended-training gate. Available
partial JSON is retained on exit 2. `min-grade` and the old no-op `publish` input
produce migration errors rather than silently changing semantics.

## Release procedure

1. Apply and review the core patch against its recorded base. Run `uv sync --locked`,
   `uv run ruff check .`, `uv run pytest`, and `uv run pytest -m integration`.
2. Run `uv run python scripts/check_release.py --tag v0.7.0`, `uv build` and
   `uvx twine check --strict dist/*`. The release-validation workflow only tests
   and uploads build artifacts; it cannot publish. Run the supported-Python CI
   matrix.
3. After approval, create the `v0.7.0` tag through the normal release process.
   Only the tag workflow publishes, and a PyPI upload cannot be replaced: any
   later fix ships as 0.7.1 or a post-release, never as a re-tag.
4. Install the published 0.7.0 in the publisher's grader environment and Action.
   Deploy the publisher's additive database migration before its Worker, purge
   edge caches and re-ingest schema-7 reports. Existing reports are not backfilled
   from old letters. Follow the companion repository migration notes.
5. Verify the published wheel and deployed consumer surfaces against the saved
   acceptance evidence.

Changing detector code or validated binding evidence changes identities. Existing
calibration manifests must match the new identities; do not copy hashes or set
acceptance flags merely to restore blocking. Synthetic fixtures exercise the
promotion mechanism only and must never be packaged as accepted real validation.

## Planned 0.7.1: work items 4 and 5

The following are deferred, not implemented in 0.7.0:

- Deeper diagnostics: cross-stream clock alignment/latency, visual quality,
  action/state consistency, and training-window/behavioral diversity coverage.
- Review and selection/export: recorded review decisions, reproducible eligible
  selections, export manifests and an end-to-end pilot in a training data reader.

Each diagnostic needs an explicit applicability and coverage contract, support in
source coordinates, and evidence/calibration before it can affect blocking.
The reader pilot should demonstrate that rejected and unknown samples cannot
silently re-enter through window construction, and preserve source/version IDs.
