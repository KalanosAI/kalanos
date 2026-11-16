# Kalanos notebooks

## Inspect synthetic robot recordings

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KalanosAI/kalanos/blob/main/notebooks/kalanos_synthetic_quickstart.ipynb)

**Draft / validation pending:** the badge works after this notebook is merged into the linked repository. Do not promote this as a tested tutorial yet.

This notebook installs Kalanos 0.10.0 in an isolated environment, downloads the public `kalanosai/synthetic-action-spikes` fixture, runs grading, displays the CLI's score/decision summary and HTML report, and downloads JSON/HTML plus provenance.

- Use a CPU runtime. No robot or GPU is required by this example.
- Public data requires no Hugging Face token; this notebook does not publish reports.
- The dataset revision resolves once at runtime and is saved. Before public launch, replace the blank revision with the exact SHA from a successful validation run.
- The download cap is 200 MB. Only an unpacked LeRobot layout is accepted.
- The `numeric` and `video` extras are installed; unavailable checks and the actual scope remain visible in the report.
- HTML generation runs the grader a second time on the same local snapshot; reports have separate run metadata.
- Scores are generated, never hard-coded. A missing score remains missing. The synthetic fixture is not evidence of training performance.

See [validation and launch checklist](VALIDATION.md). The existing main README quickstart and its version pin are unchanged.
