# Kalanos

**Grade your robot data before you train on it.**

One command gives every recording, and the dataset as a whole, a 0–100 score, an A–F grade and a train-ready verdict, with the exact episode, stream and channel behind every problem. It runs on your machine, needs no labels, and reads LeRobot, HDF5, MCAP, CSV, JSON and the home-grown formats real robots actually log.

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
│ /data/my_demos                              C  71.4  ▌ NOT TRAIN READY ▐ │
│ 120 recordings · 38 findings · 2 not analysed · 1 no schema              │
╰──────────────────────────────────────────────────────────────────────────╯
RECORDINGS ─────────────────────────────────────────────────────────────────
GRADE  RECORDING                      SCORE                METRICS
                                                  PASS WARN FAIL SKIP
  C    OVERALL             ████████░░░░  71.4     3104  212   41  880
  F    ep_017.parquet      ███░░░░░░░░░  24.0       18    6    9    7
  D    ep_052.parquet      ███████░░░░░  58.3       26    4    2    7
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
schema 6.0.0 · policy v1 · 4.12s
```

How to read it, top to bottom:

| Part | What it tells you | What to do with it |
|---|---|---|
| **Grade, score, verdict** | Dataset-wide A–F, 0–100, and `TRAIN READY` / `NOT TRAIN READY` | The go/no-go for this dataset under the current policy |
| **Recordings** | Every recording, worst first, with its own grade | Start at the top: these are the episodes to fix or drop |
| **PASS / WARN / FAIL / SKIP** | How many metric checks landed in each bucket | A high grade with a large `SKIP` count means less of the data was actually graded. Check coverage before trusting it |
| **Findings** | The worst problems, addressed down to `episode/stream/channel.metric`, with the measured value and evidence | Open that exact channel; no hunting |
| **Not analysed / No schema** | Files Kalanos declined to grade, each with a reason | Nothing is dropped silently. Fix the file or confirm it's expected |

`--report report.html` renders the same information as a page you can send to a teammate or a data vendor.

### What it catches

Every metric answers one of three questions, and none of them needs labels:

- **Did the clock lie?** Timestamp jitter, dropped samples, streams out of sync. When timing breaks, everything breaks: the model learns "saw X, did Y" from pairs that never co-occurred.
- **Is the signal intact?** Flatlined or stuck sensors, saturated channels, gaps. A stuck encoder can look statistically normal; Kalanos checks run lengths per channel.
- **Was the motion good?** Jerky, vibrating or saturated movement from a nervous teleoperator, a badly tuned controller, or hardware on its way out.

Run `kalanos metrics` to see every check installed, or `kalanos metrics --family timing` (also `integrity`, `motion`) to see one group.

### What it does not do

Kalanos does not tell you whether the task **succeeded**. A torque spike is either "the arm hit the table" or "a firm, correct grasp", and no signal-derived metric can tell them apart. Use Kalanos as the hardware-and-logging gate, then spend human review time only on the recordings that pass.

A grade reflects the metrics that applied to your data and the policy you graded against. It is a strong signal about data health, not a promise about downstream policy success.

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
kalanos grade data/ --json | jq -e '.score.train_ready == true'    # non-zero exit if not train-ready
kalanos grade data/ --json | jq '.score.score'                      # just the number
```

`kalanos grade` exits with code `2` when it cannot grade at all (missing path, over the remote limit, nothing to grade, unwritable report path), with the reason on stderr.

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
print(report.score.score, report.score.train_ready)
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
