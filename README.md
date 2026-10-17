# Kalanos

Audit robot recordings before selecting data for training. Kalanos reports technical quality findings at episode, stream and channel level, shows which checks ran, and assigns each episode a decision under a named evaluation scope. Use the HTML report to investigate problems and the JSON report to automate review.

![Kalanos report summary for the CSV fixture: readiness 61.4, one eligible recording and six findings](assets/arm-audit-overall.png)

*User-provided screenshot of the generated report: readiness 61.4 and 1/1 eligible under `numeric-core-v1`. Sufficiency is separately reported as unknown; eligibility does not establish that this dataset is sufficient for training.*

## Try a small audit

Requires Python 3.10+. This example uses the released **Kalanos 0.8.0** package and a small official CSV fixture. It needs no repository clone, account, API key or model service.

Create and activate a virtual environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, use these commands instead (no activation-policy change needed):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` and `.\.venv\Scripts\kalanos.exe` instead of `kalanos` below.

Install the pinned release, download the pinned sample, and write both reports:

```bash
python -m pip install "kalanos==0.8.0"
python -c "from importlib.metadata import version; print(version('kalanos'))"
python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/KalanosAI/kalanos/1ff101067a54357146e140387d7d95d1c190a547/tests/fixtures/arm_multi_device.csv', 'arm_multi_device.csv')"
kalanos grade arm_multi_device.csv --report report.html --no-color
kalanos grade arm_multi_device.csv --report report.json --no-color
```

Open `report.html` in your browser from your file manager. For a terminal summary of the saved JSON:

```bash
kalanos inspect report.json
```

The sample is a **synthetic test fixture**, not a real-world evaluation dataset. Four interleaved arm streams include a flatlined channel, dropped samples and clock jitter; Kalanos treats this file as **one episode**. The fixture demonstrates parsing and findings, not detection accuracy or training performance. Its source is pinned to the [0.8.0 commit](https://github.com/KalanosAI/kalanos/tree/1ff101067a54357146e140387d7d95d1c190a547); see the [fixture description](https://github.com/KalanosAI/kalanos/blob/1ff101067a54357146e140387d7d95d1c190a547/tests/fixtures/README.md).

## See the result

The included [actual HTML report](reports/arm-audit.html) and [JSON report](reports/arm-audit.json) were generated with a clean base installation of 0.8.0 on macOS ARM64, Python 3.14.7. The macOS run is tested; Linux and Windows command variants have not been executed in this review.

Observed output, shortened from [the captured stdout](provenance/html-stdout.txt):

```text
Assessment  Ready
Readiness   61/100
Episodes    1 - eligible 1 - review 0 - blocked 0 - unknown 0
Scope       numeric-core-v1

Flagged channels
  low signal-to-noise  5 of 11 channels  1 episode  Integrity
  repeated values      1 of 12 channels  1 episode  Integrity
```

Both report commands exited **0**. JSON records readiness `61.44832235703186`, complete inventory and one `pass` episode. The terminal labels that passing count “eligible.” The flatlined `armB` z channel and five low signal-to-noise channels are **report-only** findings under this scope. Inspect them before selecting data: a passing episode can still have quality problems. All four streams have unknown clock provenance; this result does not certify capture timing or claim that every planted defect blocks training.

![Expanded CSV recording showing score 61.4 and the armA pose stream with score 97.2 and recorded timing metrics](assets/arm-audit-detail.png)

*User-provided detail view: the recording scores 61.4 while the expanded `armA` stream scores 97.2. Recorded cadence is shown as 100 Hz; the unknown clock provenance still prevents treating this as capture-timing certification. Dashes are not passing results—inspect applicability and evidence in the full report.*

## Decide what to do next

| Episode decision | Meaning under the selected scope | Next action |
|---|---|---|
| `pass` | Required evidence is sufficient and no blocking or review decision remains | Inspect report-only findings and confirm the scope fits your use |
| `blocked` | A finding has an authorized blocking consequence | Fix or exclude the episode, then rerun |
| `review` | A finding needs a decision or supporting validation | Review the evidence and resolve it before selection |
| `unknown` | Required evidence is missing or unavailable | Supply the needed data, mapping or dependency, then rerun |

Readiness combines passing-episode quality with the passing share; blocked episodes contribute zero. It can be **undefined** when review, unknown decisions or incomplete inventory prevent a supported whole-dataset value. Read the reported reason rather than treating undefined as zero or a pass. See [decision semantics](https://github.com/KalanosAI/kalanos/blob/v0.8.0/docs/DECISIONS.md).

## Audit your own recordings

```bash
kalanos grade recording.csv --report recording.html
kalanos grade dataset/ --report dataset.json
kalanos profiles list
kalanos adapters
```

A directory is walked recursively. Start with a small local subset, check inventory and skipped files, and choose a scope appropriate to your intended use. Use this workflow to triage a recording batch, investigate a sensor channel, or review a vendor delivery. Compare runs only when scope and recorded identities are compatible.

The base install reads CSV, JSON, JSONL, delimited text and LeRobot directories. Reader dependencies and optional checks are separate:

| Need | Install |
|---|---|
| HDF5 reader | `python -m pip install "kalanos[hdf5]==0.8.0"` |
| MCAP / ROS 2 reader | `python -m pip install "kalanos[mcap]==0.8.0"` |
| Hugging Face access | `python -m pip install "kalanos[hf]==0.8.0"` |
| Optional numeric diagnostics | `python -m pip install "kalanos[numeric]==0.8.0"` |
| Sampled video checks, including LeRobot video | `python -m pip install "kalanos[video]==0.8.0"` |
| All extras | `python -m pip install "kalanos[all]==0.8.0"` |

Missing reader dependencies can prevent loading a format. A missing optional check is reported as unavailable; if the scope requires its capability, that missing evidence can make the episode unknown. Installing an extra alone does not change the scope's requirements. See [metrics](https://github.com/KalanosAI/kalanos/blob/v0.8.0/docs/METRICS.md), [profiles](https://github.com/KalanosAI/kalanos/blob/v0.8.0/docs/PROFILES.md) and [diagnostics](https://github.com/KalanosAI/kalanos/blob/v0.8.0/docs/DIAGNOSTICS.md).

## Use an explicit CI gate

```bash
kalanos grade dataset/ --report audit.json --fail-on blocked,review,unknown
```

Exit `0` means the configured status gate did not fail; `1` means it did; `2` means invalid configuration or an operational failure. Incomplete inventory also fails when `unknown` is included. The default gate is `blocked,unknown`, so add `review` when reviews must be resolved. This gate is not a minimum quality-score threshold: the supplied fixture still exits **0** with `blocked,review,unknown` because its findings are report-only.

## Local processing and limits

The quickstart uses the network to install packages and download the public fixture. Grading that local CSV and writing reports runs locally without a model API. Remote sources require network access; `--max-remote-gb` and `--max-remote-files` bound remote discovery. Start small rather than scanning a large remote dataset for a first audit.

Uploading is optional: `grade --publish` or the separate `kalanos publish` command sends a report to the authenticated Hub account. Neither is used here. Report publishing needs a Hub API key; local datasets also need a name. Treat report files as data: they can contain source paths, instructions and diagnostic evidence. Review them before sharing. See the [release's publishing documentation](https://github.com/KalanosAI/kalanos/blob/v0.8.0/README.md).

Kalanos checks technical recording quality within the evidence and scope available. It does not establish task success, dataset representativeness, robot safety or a training-performance guarantee. Timing interpretation depends on clock provenance, and sampled video checks do not inspect every frame by default. Statistical findings may require review or accepted validation before they can block an episode.

## Documentation and contributing

[Source](https://github.com/KalanosAI/kalanos) · [PyPI 0.8.0](https://pypi.org/project/kalanos/0.8.0/) · [Report schema](https://github.com/KalanosAI/kalanos/blob/v0.8.0/docs/SCHEMA.md) · [Contributing](https://github.com/KalanosAI/kalanos/blob/v0.8.0/CONTRIBUTING.md) · [Issues](https://github.com/KalanosAI/kalanos/issues) · [Apache-2.0 license](https://github.com/KalanosAI/kalanos/blob/v0.8.0/LICENSE)

Review evidence: [run metadata and exit codes](provenance/run.json), [resolved dependencies](provenance/requirements-resolved.txt), and [verification notes](provenance/verification.txt).
