# Kalanos

**Grade your robot data before you train on it.**

One command gives the dataset a **readiness score** out of 100, names the **blocking episodes** to exclude, and scores every recording, with the exact episode, stream and channel behind every problem. It runs on your machine, needs no labels, and reads LeRobot, HDF5, MCAP, CSV, JSON and the home-grown formats real robots actually log.

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
pip install 'kalanos[all]'      # everything
```

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
kalanos grade dataset/ --report report.json    # the full model, for scripts
```

That's the whole workflow. Everything below is detail.

---

## What you get back

Every run prints a report card to the terminal. It looks like this (numbers illustrative):

```
╭──────────────────────────────────────────────────────────────────────────╮
│ KALANOS · DATASET REPORT                                                 │
│ /data/my_demos                             READINESS 80/100  24 blocking │
│ 120 recordings · 38 findings · 2 not analysed · 1 no schema              │
╰──────────────────────────────────────────────────────────────────────────╯
RECORDINGS ─────────────────────────────────────────────────────────────────
       RECORDING                      SCORE                METRICS
                                                  PASS WARN FAIL SKIP
       OVERALL             ████████░░░░  71.4     3104  212   41  880
BLOCK  ep_017.parquet      ███░░░░░░░░░  24.0       18    6    9    7
BLOCK  ep_052.parquet      ███████░░░░░  58.3       26    4    2    7
  …
  +112 more
FINDINGS ───────────────────────────────────────────────────────────────────
FAIL ep_017/proprio.joint_position/joint_3.flatline_pct
    0.62; run_length=410
WARN ep_052/timing.jitter_cv
    0.31
NOT ANALYSED ───────────────────────────────────────────────────────────────
FILE                       REASON
meta/session.json          no_time_index
schema 7.0.0 · policy v1 · 4.12s
```

How to read it, top to bottom:

| Part | What it tells you | What to do with it |
|---|---|---|
| **Readiness, blocking episodes** | Readiness out of 100 (blocking episodes count as zero), how many episodes block, and a one-line reason | How much of this dataset you can train on as it is, and what excluding the blocking episodes would leave |
| **Recordings** | Every recording, worst first, with its own score; `BLOCK` marks the ones with blocking findings | Start at the top: these are the episodes to fix or drop |
| **PASS / WARN / FAIL / SKIP** | How many metric checks landed in each bucket | A high score with a large `SKIP` count means less of the data was actually graded. Check coverage before trusting it |
| **Findings** | The worst problems, addressed down to `episode/stream/channel.metric`, with the measured value and evidence | Open that exact channel; no hunting |
| **Not analysed / No schema** | Files Kalanos declined to grade, each with a reason | Nothing is dropped silently. Fix the file or confirm it's expected |

`--report report.html` renders the same information as a page you can send to a teammate or a data vendor.

### What it catches

Every metric answers one of four questions, and none of them needs labels:

- **Did the clock lie?** Timestamp jitter, dropped samples, streams out of sync. When timing breaks, everything breaks: the model learns "saw X, did Y" from pairs that never co-occurred.
- **Is the signal intact?** Flatlined or stuck sensors, saturated channels, gaps. A stuck encoder can look statistically normal; Kalanos checks run lengths per channel.
- **Was the motion good?** Jerky, vibrating or saturated movement from a nervous teleoperator, a badly tuned controller, or hardware on its way out.
- **Was every episode told what to do?** Episodes recorded without a task instruction, which a language-conditioned policy (a VLA) cannot learn from. Every episode's instructions appear in the report, so you can see exactly what each was told.

**A few bad episodes can't hide in an average.** An episode with a critical finding is **blocking**, and blocking episodes count as zero:

```text
Readiness = sum of passing episodes' quality scores / number of evaluated episodes
          = passing share × passing-episode quality
```

So 42 passing episodes of 50, at quality 99.99, read 84: exactly the share of the dataset you can train on, times how clean it is. The report names every blocking episode with its reasons, and the readiness once they are excluded ("84 as it is, 100 after excluding 8 episodes"). Findings that describe how the data was recorded rather than a fault in some episodes (the same finding on every episode of a task, or on nearly every episode of the dataset) are reported as traits and block nothing. `language_conditioned` makes an episode without its task instruction blocking, for VLA training; `KALANOS_POLICY_PATH=legacy_0_5` scores without blocking episodes, reproducing 0.5, and so reports no readiness.

Letter grades are deprecated since 0.6.5: reports still carry the old letter fields for compatibility, but nothing presents them and they drive no decision. Since 0.7.0 every episode carries one `eligibility` (`pass`, `blocked`, `review`, `unknown`) under a named scope; see `docs/DECISIONS.md`.

### What will my data score? Worked examples

Each row is a 50-episode, 50 Hz arm dataset with one problem dialled in, scored by Kalanos 0.6.5. Everything else is clean, including a realistic 0.25% clock wobble.

| What's wrong | Readiness | Episodes passing | Blocking | After excluding them |
|---|---|---|---|---|
| Nothing (clean control) | **100** | 50/50 | 0 | — |
| 3% of samples dropped in 5 episodes | **100** | 50/50 | 0 (warning only) | — |
| 8% of samples dropped in 2 episodes (4%) | **96** | 48/50 | 2 | 100 |
| 8% of samples dropped in 5 episodes (10%) | **90** | 45/50 | 5 | 100 |
| 8% of samples dropped in 10 episodes (20%) | **80** | 40/50 | 10 | 100 |
| 8% of samples dropped in 20 episodes (40%) | **60** | 30/50 | 20 | 100 |
| Control glitches (sudden command jumps) in 4 episodes (8%) | **92** | 46/50 | 4 | 100 |
| Clock declared 50 Hz, actually 45 Hz (10% slow) | **100** | 50/50 | 0 | — |
| Clock declared 50 Hz, actually 40 Hz (20% slow) | **95** | 50/50 | 0 (warning: lowers timing) | — |
| Clock declared 50 Hz, actually 35 Hz (30% slow), every episode | **93** | 50/50 | 0: a dataset trait (see below) | — |
| Clock jitter of 5% of the sampling period, every episode | **100** | 50/50 | 0: jitter is measured, not yet scored | — |
| 5% jitter **and** 8% of samples dropped in 10 episodes | **80** | 40/50 | 10 (the drops) | 100 |
| 5% repeated timestamps in 10 episodes | **100** | 50/50 | 0: measured, not yet scored | — |

How to read it:

- **Dropped samples**: under 1% of an episode is fine, 1–5% is a warning that lowers its quality, over 5% makes the episode blocking. Readiness then falls with the share of blocking episodes: 10 blocking of 50 is 80.
- **Glitches** make the episodes they hit blocking, through the spike and noise checks.
- **A problem in every episode is a dataset trait, not blocking.** A clock 30% slow in all 50 episodes can't be fixed by excluding episodes, so it blocks nothing; the timing check still lowers every episode's quality (93, not 100), and the report lists the trait. A dataset-wide defect like this should lower readiness further, and a future release will make it do so; until then, read the report's dataset traits alongside the number.
- **Jitter and repeated timestamps** appear in every report with their numbers but don't change readiness yet: their thresholds will be set from real recordings, not guessed ([docs/METRICS.md](https://github.com/KalanosAI/kalanos/blob/main/docs/METRICS.md)).

All of these thresholds are candidates, tested on synthetic and real datasets and open to revision against labelled failures.

Checks that can't observe something say so rather than score it: on converted datasets whose timestamps were reconstructed from frame numbers (most LeRobot hub data), the timing checks report *capture timing is not observable* instead of a perfect score, and the noise check needs at least ~45 Hz to tell sensor noise from motion.

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
- **Keep quality from drifting.** Grade every dataset update in CI and fail the build when the grade drops.

---

## Common tasks

**Grade a Hugging Face dataset**

```bash
kalanos grade hf://datasets/lerobot/pusht
kalanos grade hf://datasets/lerobot/pusht@<commit>                  # pin a revision
kalanos grade https://huggingface.co/datasets/lerobot/pusht         # the page URL works too
```

The dataset is pinned to one commit and streamed. Anything listing more than 20 GB or 10,000 files is refused before a byte is read; raise the limit for one run with `--max-remote-gb 50` or `--max-remote-files 20000`. For private or gated datasets, set `HF_TOKEN`.

**Use it in CI or a script**

`--json` prints the full report model to stdout, so you can gate on it:

```bash
kalanos grade data/                                   # exit 1 if any episode is blocked or unknown
kalanos grade data/ --fail-on blocked,review,unknown  # a training gate: review must be resolved too
kalanos grade data/ --fail-on blocked                 # exploratory: only objective blockers fail
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
```

Planned: `kalanos inspect`, `--fail-under`, `--sample`.

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

Field names are matched against a built-in data dictionary covering the naming, units, shapes and plausible ranges used by DROID, LeRobot, MuJoCo, ROS 2 and others. Recognised signals get physics-aware checks; unrecognised ones still get the universal checks, so a mystery column that's flatlined still fails loudly.

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
| `HF_TOKEN` | unset | Opens private or gated Hugging Face datasets |

### Typing a field for one run

Some field names mean different things on different robots, so the dictionary leaves them unmapped on purpose. When you know what a field holds, type it for one run without touching the dictionary:

```bash
kalanos grade hf://datasets/lerobot/toto --map observation.state=proprio.joint_position
```

The same override can come from three places, merged per field:

1. `--map FEATURE=TYPE`, repeatable. From Python, `grade(path, mapping={...})`.
2. `--map-file PATH`. From Python, `grade(path, mapping_file=...)`.
3. A `kalanos-map.yaml` sidecar in the graded folder, or beside the graded file. `--no-sidecar` (or `sidecar=False`) ignores it.

`--map` beats `--map-file`, which beats the sidecar. Both files use the same shape:

```yaml
schema_version: 1
features:
  observation.state: proprio.joint_position
```

`FEATURE` is the stream's source field: the LeRobot feature key, the HDF5 dataset key, the MCAP topic. For CSV and other tables it is the grouped column stem: `q` covers `q_0..q_5`. `TYPE` must be a key in the dictionary. An override naming a field no stream has, or a type the dictionary lacks, stops the run with exit code 2.

Kalanos records the type as given and does not verify it. The report records every override applied, with where it came from, in `report.mapping_overrides`, and each stream's `mapping_source` says whether the dictionary, the format's declared channel names, or an override typed it.

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
├── docs/                       # ARCHITECTURE.md, METRICS.md, ADAPTERS.md
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
