# Kalanos

[![PyPI](https://img.shields.io/pypi/v/kalanos)](https://pypi.org/project/kalanos/) [![Python](https://img.shields.io/pypi/pyversions/kalanos)](https://pypi.org/project/kalanos/) [![CI](https://github.com/KalanosAI/kalanos/actions/workflows/ci.yaml/badge.svg)](https://github.com/KalanosAI/kalanos/actions/workflows/ci.yaml) [![License](https://img.shields.io/github/license/KalanosAI/kalanos)](https://github.com/KalanosAI/kalanos/blob/main/LICENSE)

**Find which robot episodes need a closer look before training.**

Kalanos audits recorded demonstrations for issues such as noisy commands, flatlined signals, and missing data. It runs where your data lives and writes an HTML report for inspection or JSON for automation. Nothing is uploaded unless you publish. No account or API key is needed for local audits.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KalanosAI/kalanos/blob/main/notebooks/kalanos_quickstart.ipynb)

## Get started

With Python 3.10+ in a virtual environment:

```bash
python -m pip install kalanos
kalanos grade ./my_recordings --report audit.html
```

Replace `./my_recordings` with a local dataset folder or recording file. Open `audit.html` in a browser. To use the results in code, set the output to `audit.json` instead.

Kalanos reads local LeRobot datasets, CSV, and JSON out of the box. For HDF5 or MCAP/ROS 2, install `"kalanos[hdf5,mcap]"` and see [adapters](https://github.com/KalanosAI/kalanos/blob/main/docs/ADAPTERS.md). For sampled camera checks, install `"kalanos[video]"` and choose a [vision profile](https://github.com/KalanosAI/kalanos/blob/main/docs/PROFILES.md).

## See what needs review

In the [Colab quickstart](https://github.com/KalanosAI/kalanos/blob/main/notebooks/kalanos_quickstart.ipynb), Kalanos checks 50 simulated ALOHA cube-transfer demonstrations: **39 pass, 11 need review**. Start with **episode 2**: its left-waist command barely moves and has low signal-to-noise. Compare it with **episode 29**, which also moves little but passes. The chart shows where to look, not proof of a fault.

![Scatter plot of left-waist command movement versus signal-to-noise. Orange points identify episodes to review. Episode 2 has about −5 dB and needs review; episode 29 has about 19 dB and passes.](https://raw.githubusercontent.com/KalanosAI/kalanos/main/docs/assets/aloha-episode-review.png)

<sub>One dot per episode with a measured ratio: 47 of 50. This run checks numeric signals, not cameras. An episode decision reflects the checks that ran, not task success or training performance.</sub>

## Go further

[Website](https://kalanos.ai) · [Public reports](https://hub.kalanos.ai/?utm_source=github&utm_medium=referral&utm_campaign=grader_readme&utm_content=public_reports) · [How decisions work](https://github.com/KalanosAI/kalanos/blob/main/docs/DECISIONS.md) · [GitHub Action](https://github.com/marketplace/actions/kalanos-robotics-dataset-audit) · [Experiment tracking](https://github.com/KalanosAI/kalanos-tracking) · [More notebooks](https://github.com/KalanosAI/kalanos/blob/main/notebooks/README.md) · [Apache-2.0 license](https://github.com/KalanosAI/kalanos/blob/main/LICENSE)

[Report a bug](https://github.com/KalanosAI/kalanos/issues/new/choose) · Private feedback: [hello@kalanos.ai](mailto:hello@kalanos.ai)

## Use Kalanos in your editor

Install the [dataset audit skill](skills/audit-robot-dataset/SKILL.md) from your project directory:

```bash
npx skills add KalanosAI/kalanos --skill audit-robot-dataset
```

Select your editor when prompted. Add `-a <agent>` to skip the prompt or `-g` to install for all projects. To refresh the skill, rerun the install command; the installed `metadata.version` shows which copy you have. Prefer a manual install? Copy the entire `skills/audit-robot-dataset/` folder into `.claude/skills/` for Claude Code, `.agents/skills/` for Codex or Cursor, or `.github/skills/` for VS Code with GitHub Copilot. Replace manual copies to update them.

The install adds the skill only. For a fresh audit, install the [tested Grader CLI version](https://pypi.org/project/kalanos/0.10.2/) in your local Python environment (`python -m pip install "kalanos==0.10.2"`). For published reports, connect the read-only Kalanos MCP endpoint `https://mcp.kalanos.ai/mcp` separately using your editor's MCP settings.

- **Local audit:** “Which episodes in `./my_recordings` should I review before training?”
- **Published report:** “Is there a published audit for `lerobot/pusht`? Which checks did it cover?”

Related: [LeRobot episode selection](https://github.com/KalanosAI/kalanos-mcp) and [attaching an audit to a training run](https://github.com/KalanosAI/kalanos-tracking).
