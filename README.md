# Kalanos

**Grade your robot data before you train on it.**

One command gives the dataset a **readiness score** out of 100, decides every episode (**pass**, **blocked**, **review** or **unknown**) under a named scope, and scores every recording, with the exact episode, stream and channel behind every problem. When the evidence isn't there, it says so instead of guessing. Diagnostics run on your machine without labels; statistical blocking needs accepted validation evidence. It reads LeRobot, HDF5, MCAP, CSV, JSON and the home-grown formats real robots actually log.

```bash
pip install kalanos
kalanos grade path/to/your/dataset/
```

[![PyPI](https://img.shields.io/pypi/v/kalanos)](https://pypi.org/project/kalanos/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/KalanosAI/kalanos/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/kalanos/)

---

## Quick start

**1. Install.** Python 3.10 or newer; a virtual environment is recommended.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install kalanos
kalanos --help
```

If you see `kalanos: command not found`, the virtual environment isn't active in this terminal. Run `source .venv/bin/activate` (each new terminal needs it) and try again. Working from a clone with [uv](https://docs.astral.sh/uv/)? `uv sync` then `uv run kalanos …` works without activating anything.

The plain install reads text formats (CSV, JSON, JSONL, delimited text) and LeRobot directories. Add extras for the rest:

```bash
pip install 'kalanos[hf]'       # stream Hugging Face datasets
pip install 'kalanos[hdf5]'     # HDF5 (robomimic, Isaac Lab)
pip install 'kalanos[mcap]'     # MCAP / ROS 2
pip install 'kalanos[numeric]'  # NumPy, for the optional deeper diagnostics
pip install 'kalanos[video]'    # PyAV, for sampled video diagnostics
pip install 'kalanos[all]'      # everything
```

Missing optional dependencies leave the affected diagnostic unavailable, with the needed extra named in `diagnostics.results[].reason`. If its capability is listed in `requirements.required_capabilities`, the missing evidence makes the episode `unknown` unless a blocking finding takes precedence. Requesting a report-only diagnostic does not make its capability required. Adapter extras are still needed to read their respective formats.

**2. Grade something.**

```bash
kalanos grade recording.csv                    # one file
kalanos grade dataset/                         # a folder, walked recursively, scored as one dataset
kalanos grade hf://datasets/lerobot/pusht      # a Hugging Face dataset, streamed, never fully downloaded
```

**No data handy? Try the samples in this repo.** [`tests/fixtures/`](https://github.com/KalanosAI/kalanos/tree/main/tests/fixtures) holds small files with known, planted problems, so you can see what a finding looks like in seconds. They're not part of the pip install, so clone the repo and install from the checkout:

```bash
git clone https://github.com/KalanosAI/kalanos.git && cd kalanos
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'                              # installs the `kalanos` command from this checkout, with every format
kalanos grade tests/fixtures/arm_multi_device.csv   # 4 arms: one clean, one flatlined, one dropping samples, one with a jittery clock
kalanos grade tests/fixtures/                       # the whole corpus as one dataset, including a file it skips (with the reason)
```

| Sample | Format | What you should see |
|---|---|---|
| [`arm_multi_device.csv`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/arm_multi_device.csv) | CSV, 4 devices in one file | Split into `armA`–`armD`; `armA` clean, `armB` flatlined `tcp_pose_z_mm`, `armC` a burst of dropped samples, `armD` clock jitter |
| [`imu_stream.txt`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/imu_stream.txt) | JSONL hiding in a `.txt` | Real container detected; `acc` and `gyro` arrays expanded into channels |
| [`pose_log.txt`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/pose_log.txt) | Whitespace text, header in a `#` comment | Header read from the comment; 50 Hz timing |
| [`capture_index.json`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/capture_index.json) | JSON keyed by source+hash+timestamp | Keys parsed, per-camera nesting handled; rate metrics marked not applicable |
| [`video_meta.json`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/video_meta.json) | Flat metadata, no time index | Listed under **Not analysed**, with the reason |
| [`lerobot_v3_tiny/`](https://github.com/KalanosAI/kalanos/tree/main/tests/fixtures/lerobot_v3_tiny) · [`lerobot_v2_1_tiny/`](https://github.com/KalanosAI/kalanos/tree/main/tests/fixtures/lerobot_v2_1_tiny) · [`lerobot_v2_0_tiny/`](https://github.com/KalanosAI/kalanos/tree/main/tests/fixtures/lerobot_v2_0_tiny) | LeRobot directories | Episodes and fps read from `meta/info.json` |
| [`hdf5_tiny.hdf5`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/hdf5_tiny.hdf5) | HDF5 (needs `[hdf5]`) | Episodes found from group structure |
| [`mcap_tiny.mcap`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/mcap_tiny.mcap) | MCAP / ROS 2 (needs `[mcap]`) | Three topics, each resolved by message type or topic name |

Full details of each sample are in [`tests/fixtures/README.md`](https://github.com/KalanosAI/kalanos/blob/main/tests/fixtures/README.md).

**3. Keep the report.**

```bash
kalanos grade dataset/ --report report.html    # a shareable page
kalanos grade dataset/ --report report.json    # the full model as compact JSON
```

JSON reports (`--report FILE.json` and `--json` stdout) use compact formatting to reduce file size. Every field, null, metric and evidence value is retained; only layout whitespace is removed. Existing JSON readers, including `kalanos inspect`, accept the compact output.

That's the whole workflow. Everything below is detail.

---

## What you get back

Every run prints a short summary to the terminal (plain ASCII here; colored in a terminal):

```
$ kalanos grade tests/fixtures/ --no-color
Kalanos - tests/fixtures

  Assessment  Incomplete - inventory not fully read
  Readiness   undefined - inventory incomplete: whole-dataset fractions are not published
  Episodes    13 - eligible 13 - review 0 - blocked 0 - unknown 0
  Scope       numeric-core-v1

Findings
  armA:proprio.ee_pose/tcp_pose_z_mm  1 episode - low signal-to-noise  report-only
  armB:proprio.ee_pose/tcp_pose_z_mm  1 episode - repeated values  report-only
  armC:proprio.ee_pose/tcp_pose_x_mm  1 episode - low signal-to-noise  report-only
  + 6 more sources in report

Coverage
  numeric    13/13 episodes
  mapping    10 stream types unmapped - type them with --map
  inventory  incomplete - 0 episodes unassessed - 1 sources refused

Not analysed
  README.md        no adapter
  video_meta.json  no schema: no candidate column resolved as a time axis (name 'fps' does not match a time-like patt...

schema 7.0.0 - policy v1 - 0.20s
```

The same card on a dataset with video: the coverage rows grow with the data.

```
$ kalanos grade tests/fixtures/lerobot_v3_tiny --profile vision-imitation-v1 --no-color
Kalanos - tests/fixtures/lerobot_v3_tiny

  Assessment  Review required
  Readiness   undefined - 2 episode(s) require review
  Episodes    2 - eligible 0 - review 2 - blocked 0 - unknown 0
  Scope       vision-imitation-v1

Needs attention
  lerobot_v3_tiny/episode_000000  review   83.3  low sharpness, brightness changes
  lerobot_v3_tiny/episode_000001  review   83.3  low sharpness, brightness changes

Findings
  unmapped.observation.images.up  2 episodes - low sharpness, brightness changes  review

Coverage
  numeric        2/2 episodes
  video quality  2/2 episodes - 14 frames examined
  mapping        2 stream types unmapped - type them with --map
  inventory      complete

schema 7.0.0 - policy v1 - 0.12s
```

How to read it, top to bottom:

| Part | What it tells you | What to do with it |
|---|---|---|
| **Assessment and episodes** | The verdict, readiness out of 100 (or `undefined` with its reason), and how many episodes are eligible, in review, blocked or unknown | How much of this dataset is confirmed usable as it is, and what needs a decision or more evidence before you can tell |
| **Needs attention** | The episodes that did not pass, worst first, with their status, score and the conditions behind it | Start at the top: these are the episodes to fix, review or drop |
| **Findings** | Each source with findings, how many episodes it affects, what was observed, and whether it blocks, needs review or is only reported | Open that source in the full report for the values and evidence |
| **Coverage** | For each check that applies to this data, how many episodes it was computed on, how many stream types went unmapped, and whether the whole inventory was read | A gap here means less was checked than the counts suggest; read its reason before trusting the verdict. Type unmapped streams with `--map` so their metrics apply |
| **Not analysed** | Files Kalanos declined to grade, each with a reason | Nothing is dropped silently. Fix the file or confirm it's expected |
| **Report** | Where `--report` wrote the full report | Open it for every episode and finding |

`--no-color` prints plain ASCII; `--color` keeps the styling when piping. Detailed values, evidence and every episode are in `--report` and `kalanos inspect`.

`--report report.html` renders the same information as a page you can send to a teammate or a data vendor.

### What it catches

Every metric answers one of four questions, and none of them needs labels:

- **What does the clock say?** Backwards and repeated timestamps, the recorded cadence, interval spread and gaps, and where each timestamp came from (captured, logged, generated from frame numbers, unknown). When timing breaks, everything breaks: the model learns "saw X, did Y" from pairs that never co-occurred. Kalanos reports what the recorded timeline shows, and only grades *capture* timing when the recording carries evidence that its timestamps are capture times; see [docs/METRICS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/METRICS.md#timing).
- **Is the signal intact?** Flatlined or stuck sensors, saturated channels, gaps. A stuck encoder can look statistically normal; Kalanos checks run lengths per channel.
- **Was the motion good?** Jerky, vibrating or saturated movement from a nervous teleoperator, a badly tuned controller, or hardware on its way out.
- **Was every episode told what to do?** Episodes recorded without a task instruction, which a language-conditioned policy (a VLA) cannot learn from. Every episode's instructions appear in the report, so you can see exactly what each was told.

**A few bad episodes can't hide in an average.** A finding with an authorized **block** consequence makes its episode blocking. Critical statistical findings without matching calibration require **review**. Blocking episodes count as zero:

```text
Readiness = sum of passing episodes' quality scores / number of evaluated episodes
          = passing share × passing-episode quality
```

So 42 passing episodes of 50, at quality 99.99, read 84: exactly the share of the dataset you can train on, times how clean it is. The report names every blocked episode with its reasons, and the mean quality of the rest. Readiness is only defined when every episode is either `pass` or `blocked`: an episode that needs `review`, or is `unknown` because a required check couldn't run, makes it `undefined` with the reasons listed, rather than a number that pretends the evidence is in.

Findings shared by every episode of a task, or by nearly every episode of the dataset, are reported as **traits** so you see the pattern, and any authorized blocking consequence remains in force: the report can't tell a recording convention from corruption in every episode, so it doesn't guess. A scoped policy rule can exempt such a finding explicitly. `language_conditioned` makes an episode without its task instruction blocking, for VLA training; `KALANOS_POLICY_PATH=legacy_0_5` grades with no dataset gate and no letter cap, reproducing 0.5's numbers, while every episode still carries its eligibility.

Letter grades are deprecated since 0.6.5: reports still carry the old letter fields for compatibility, but nothing presents them and they drive no decision. Since 0.7.0 every episode carries one `eligibility` (`pass`, `blocked`, `review`, `unknown`) under a named scope; see [docs/DECISIONS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/DECISIONS.md).

### How findings affect readiness

Severity, coverage and the decision are separate. A diagnostic can be computed, reported and critical while its consequence remains review.

| Situation | Report behavior | Eligibility/readiness impact |
| --- | --- | --- |
| Required numeric checks complete without findings | Counts each computed channel check | Can pass within the named numeric scope |
| A spike rule fires without matching accepted calibration | Critical measurement, localized evidence, review consequence | Episode needs review; readiness is undefined |
| The same rule has matching accepted validation | Manifest identity and matching context accompany the finding | May block; blocked episodes count as zero |
| Required camera analysis has no implementation | Camera streams remain in required coverage as unavailable | Unknown; numeric results cannot cover the gap |
| Metadata tier skips numeric payloads | Declared numeric subjects remain counted as skipped | Unknown under numeric-core |
| Timestamps have unknown or generated origin | Recorded rate/spread/gap estimates; acquisition checks abstain | No timing certification; other checks can still affect the result |
| A metric or payload raises an unexpected exception | Error coverage and an operational error record | Unknown; CLI exits 2 even with a permissive gate |

A rule affecting every episode does not become harmless through prevalence. Calibration authorizes a particular detector, thresholds, bindings and operating scope; it is not a universal approval. The default policy ships without accepted statistical calibration manifests. See [Calibration](https://github.com/KalanosAI/kalanos/blob/main/docs/DECISIONS.md#calibration) for what a manifest must declare before a statistical finding can block.

Some checks are **measured but not yet graded**: repeated or backwards timestamps (`monotonic_violations`) and missing task instructions (`task_instruction_missing`). They appear in every report with their evidence but don't change the score, because their thresholds are still to be settled against real recordings rather than guessed. See [docs/METRICS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/METRICS.md).

Run `kalanos metrics` to see every check installed, or `kalanos metrics --family timing` (also `integrity`, `motion`, `annotation`) to see one group.

### What it does not do

Kalanos does not tell you whether the task **succeeded**. A torque spike is either "the arm hit the table" or "a firm, correct grasp", and no signal-derived metric can tell them apart. Use Kalanos as the hardware-and-logging gate, then spend human review time only on the recordings that pass.

Readiness reflects the checks that applied to your data and the policy you graded against. It is an index of technical readiness, not a prediction of training success or a measure of a dataset's research value.

---

## Why run it before training

<!-- ROI section: replace each bracketed item with a measured number once you have one. Do not ship estimates as results. -->

- **Stop burning GPU hours on broken data.** A single out-of-sync stream or stuck joint can poison a whole run. Finding it takes one command; finding it after training takes a failed eval and a day of debugging.
- **Catch bad sessions while the robot is still set up.** Grade each teleop session as it lands, and redo the bad ones before the operator, rig and scene are gone.
- **Accept or reject vendor data with evidence.** Grade a delivery before you sign off, and send the vendor the HTML report showing exactly which episodes failed and why.
- **Point human review where it's needed.** Reviewers look at the recordings that passed the automated gate, not at every file.
- **Keep quality from drifting.** Grade every dataset update in CI and gate on blocked, review or unknown episode decisions.

---

## Common tasks

**Grade a Hugging Face dataset**

```bash
kalanos grade hf://datasets/lerobot/pusht
kalanos grade hf://datasets/lerobot/pusht@<commit>                  # pin a revision
kalanos grade https://huggingface.co/datasets/lerobot/pusht         # the page URL works too
```

The dataset is pinned to one commit and streamed. Anything listing more than 20 GB or 10,000 files is refused before a byte is read; raise the limit for one run with `--max-remote-gb 50` or `--max-remote-files 20000`. `--full-frame-scan` downloads every video segment, where the default reads only a sample of frames. `--tier full` implies it. For private or gated datasets, set `HF_TOKEN`.

**Use it in CI or a script**

`--json` prints the full report model to stdout, so you can gate on it:

```bash
kalanos grade data/                                   # exit 1 if any episode is blocked or unknown
kalanos grade data/ --fail-on blocked,review,unknown  # a training gate: review must be resolved too
kalanos grade data/ --fail-on blocked                 # exploratory: only authorized blockers fail
kalanos grade data/ --json | jq '.eligibility_counts, .readiness'
```

Exit codes: `0` the audit completed and no episode carries a status in `--fail-on`; `1` the audit completed and at least one does; `2` invalid configuration or an operational failure (missing path, over the remote limit, nothing to grade, malformed `--profile`, unwritable report path), with the reason on stderr. Exit 1 is a decision gate, not a crash: `review` and `unknown` mean the data needs a decision or more evidence.

**Grade against your own thresholds**

A servo vendor and a factory team want different tolerances for the same signal. Copy the default policy, edit it, and point Kalanos at it:

```bash
KALANOS_POLICY_PATH=strict.yaml kalanos grade data/
```

**See why Kalanos decided what it did**

```bash
kalanos --verbosity debug grade data/     # every discovery and inference decision, on stderr
```

**Check what's installed**

```bash
kalanos adapters      # formats you can read, and any that are missing an extra
kalanos metrics       # every quality check
kalanos plugins       # summary, plus anything that failed to load
kalanos benchmark     # how often each check fires on reference data, clean and with an injected defect
```

```bash
kalanos inspect report.json                                  # scope, decision counts, readiness and reasons of a saved report
kalanos inspect report.json --episode ID                     # every reason on one episode
kalanos grade ./recording --hash-source --report new.json    # record a byte identity for later comparison
kalanos compare old.json new.json --report comparison.json   # what changed between two graded runs
```

`inspect` also reads reports written by 0.6 (schema 6.3–6.5) without changing them, and points out where their old fields contradict each other. `compare` aligns exact recorded subject identities and refuses a numeric readiness comparison when they're missing or incompatible; see [docs/DECISIONS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/DECISIONS.md#cli-gate) for its exit codes. Audit sampling remains planned.

**Measure a metric's benign and detection rates**

```bash
kalanos benchmark --sample 10 --out benchmark.md    # the two pinned reference datasets, 10 sampled episodes each
```

Grades each reference dataset as recorded, then again with a `kalanos.testing` defect injected into a sample of episodes, and reports how often every metric fired both times. See [docs/METRICS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/METRICS.md#benchmarking-metrics) for its flags and what the rates mean.

---

## Formats it reads

| Adapter | Reads | Extra |
|---|---|---|
| `lerobot_v2` | LeRobot v2.0 and v2.1 dataset directories | — |
| `lerobot_v3` | LeRobot v3.0 dataset directories | — |
| `hdf5` | HDF5 files, episode boundaries inferred from group structure | `hdf5` |
| `mcap` | MCAP recordings (CDR-encoded ROS 2 messages) | `mcap` |
| `csv` | CSV tables | — |
| `jsonl` | line-delimited JSON, including a `.txt` that is really JSONL | — |
| `json` | nested JSON records | — |
| `delimited` | whitespace-delimited text, with the header in a `#` comment | — |

An adapter whose extra isn't installed shows as unavailable under `kalanos adapters` instead of failing the run.

**Task instructions** are read where the format stores them: LeRobot v2 and v3 from each episode's task list in `meta/`, and HDF5 from an episode group's `task`, `language_instruction`, `instruction`, `lang` or `language` attribute. A dataset with no instructions at all is treated as not language-annotated, not as missing them.

**No format name? Still works.** Robots and factory sensors routinely invent their own layouts, so Kalanos infers the schema when nothing declares one:

| What arrives | What Kalanos does |
|---|---|
| CSV with a `t_ms` column and an `id` column | Splits by `id` into one stream per device before scoring |
| A `.txt` that is really JSONL: `{"t_us":…,"acc":[x,y,z]}` | Detects the real container, expands arrays into channels |
| A `.txt` with the header in a `# timestamp ax ay az` comment | Reads the header out of the comment; infers epoch seconds from magnitude |
| A JSON dict keyed by `"AUTOLab+5d05c5aa+2023-07-07-10h-00m-27s"` | Parses the key for source, hash and timestamp; handles per-camera nesting and irregular timing |
| A flat JSON metadata blob (`fps`, `codec`, `duration_sec`) | Reports it as skipped, with the reason |

Field names are matched against a built-in data dictionary covering the naming, units, shapes and plausible ranges used by DROID, LeRobot, MuJoCo, ROS 2 and others. Recognised signals get physics-aware checks; unrecognised ones still get the universal checks, so a mystery column that is flatlined still produces evidence.

Need a format that isn't here? `kalanos new adapter <name>` scaffolds a publishable plugin. See [docs/ADAPTERS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/ADAPTERS.md).

---

## Using it as a library

```python
from pathlib import Path

from kalanos import grade, load_policy

report = grade("recordings/", policy=load_policy(Path("strict.yaml")))
print(report.eligibility_counts, report.readiness)
for finding in report.findings[:5]:
    print(finding.metric_id, finding.severity.value, finding.stream)
```

`grade` accepts the same paths the command does and returns the same `Report` that `--json` prints. When the input can't be graded at all it raises a subclass of `KalanosError`. Only names exported from `kalanos` itself are stable; subpackages may change between releases.

---

## Configuration

Kalanos keeps facts and policy in separate files, so tuning one threshold never means forking the dictionary.

- `dictionary.yaml` answers *what is this signal?* Aliases, units, expected shape, plausible range. Ships in the package and rarely changes.
- `policy.yaml` answers *how strict should grading be here?* Thresholds, weights, declared limits. A default ships; override it per deployment.

Environment variables:

| Variable | Default | What it does |
|---|---|---|
| `KALANOS_POLICY_PATH` | packaged default | Grade with your own `policy.yaml` |
| `KALANOS_DICTIONARY_PATH` | packaged default | Resolve signal names with your own `dictionary.yaml` |
| `KALANOS_REPORTS_DIR` | current directory | Where a relative `--report` path is written |
| `KALANOS_REMOTE_MAX_BYTES` | `20000000000` (20 GB) | Largest remote dataset to stream; `--max-remote-gb` overrides per run |
| `KALANOS_REMOTE_MAX_FILES` | `10000` | Most files in a remote dataset; `--max-remote-files` overrides per run |
| `KALANOS_VISION_SAMPLES` | unset | Frames each camera samples, over the bundle's `vision` section; `--vision-samples` sets it per run |
| `KALANOS_FULL_FRAME_SCAN` | unset | Frame metrics read every frame instead of a sample, over the bundle's `vision` section; `--full-frame-scan` sets it per run |
| `HF_TOKEN` | unset | Opens private or gated Hugging Face datasets |

### Typing a field for one run

Some field names mean different things on different robots, so the dictionary leaves them unmapped on purpose. When you know what a field holds, type it for one run without touching the dictionary:

```bash
kalanos grade hf://datasets/lerobot/toto --map observation.state=proprio.joint_position
```

The same override can come from four places, merged per field:

1. `--map FEATURE=TYPE`, repeatable. From Python, `grade(path, mapping={...})`.
2. `--map-file PATH`. From Python, `grade(path, mapping_file=...)`.
3. The `binding.features` section of a `--profile` bundle. From Python, `grade(path, bundle=...)`.
4. A `kalanos-map.yaml` sidecar in the graded folder, or beside the graded file. `--no-sidecar` (or `sidecar=False`) ignores only this one.

`--map` beats `--map-file`, which beats the bundle, which beats the sidecar. Two inputs at the same level that disagree stop the run with exit code 2; every lower-level assertion that lost is recorded in `report.binding_conflicts`. A map file or sidecar can only assert mappings. A bundle also carries the evaluation scope (`requirements`), the decision `policy` and the execution `tier`, each with its own identity recorded in `report.run`; see [docs/DECISIONS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/DECISIONS.md) and [docs/SCHEMA.md](https://github.com/KalanosAI/kalanos/blob/main/docs/SCHEMA.md). Map files and sidecars use this shape:

```yaml
schema_version: 1
features:
  observation.state: proprio.joint_position
```

`FEATURE` is the stream's source field: the LeRobot feature key, the HDF5 dataset key, the MCAP topic. For CSV and other tables it is the grouped column stem: `q` covers `q_0..q_5`. `TYPE` must be a key in the dictionary. An override naming a field no stream has, or a type the dictionary lacks, stops the run with exit code 2.

Kalanos records the type as given and does not verify it. The report records every override applied, with where it came from, in `report.mapping_overrides`, and each stream's `mapping_source` says whether the dictionary, the format's declared channel names, or an override typed it.

### Binding individual channels

A vector field can bundle more than one kind of signal — joint position next to motor effort — and `binding.channels` assigns each member its own type and validation. See [docs/PROFILES.md](https://github.com/KalanosAI/kalanos/blob/main/docs/PROFILES.md).

### Requirements, evidence and execution

A bundle's `requirements` section sets what a pass needs beyond raw numbers — physical units, capture timing, video quality — and `execution.tier` changes what Kalanos attempts without ever lowering them. See [docs/DECISIONS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/DECISIONS.md).

### Clock provenance and recorded order

Every stream reports where its timestamps came from and whether its rows are still in source order, down to the native tick. See [docs/METRICS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/METRICS.md#timing).

### Deeper diagnostics

Optional diagnostic plans measure stream-pair timing, command response, sampled video quality, dimensionless motion, training windows and cohort diversity, report-only by default:

```bash
pip install 'kalanos[numeric,video]'
```

See [docs/DIAGNOSTICS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/DIAGNOSTICS.md).

---

## Contributing

See [CONTRIBUTING.md](https://github.com/KalanosAI/kalanos/blob/main/CONTRIBUTING.md) for the development gate and conventions, [docs/METRICS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/METRICS.md) for what each metric computes and how it's graded, and [docs/ADAPTERS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/ADAPTERS.md) for adding a format. Built-in adapters, metrics and reporters register through the same entry points as third-party ones, so the plugin API is exercised by the code we maintain.

### Development

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # create .venv and install everything from uv.lock
uv run kalanos --help   # run the CLI from the checkout
```

Before sending a patch, run the same gate CI does:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

<details>
<summary>Repository layout</summary>

```plain
./
├── docs/                       # ADAPTERS.md, ARCHITECTURE.md, DECISIONS.md, DIAGNOSTICS.md, METRICS.md, PROFILES.md, SCHEMA.md
├── src/kalanos/
│   ├── cli.py                  # The `kalanos` command group
│   ├── api.py                  # The library entry point
│   ├── core/                   # Settings and logging
│   ├── analysis/               # Discovery, adapters, inference, metrics, scoring, reporting
│   ├── assets/                 # dictionary.yaml and policies/, shipped in the wheel
│   └── testing/                # Contract suites and defect injectors for plugin authors
├── tests/fixtures/             # Synthetic corpus, one file per parsing pathology
├── scripts/                    # Helper scripts, including the fixture generator
└── pyproject.toml              # Dependencies, extras, entry points
```

</details>

---

## License

[Apache-2.0](https://github.com/KalanosAI/kalanos/blob/main/LICENSE).
