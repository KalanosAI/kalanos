"""Compare report sizes between two kalanos versions on a set of datasets.

Grades each dataset with both versions and prints a table of report bytes, bytes per
episode and the change in bytes per episode. A terminal gets a rendered table, and a
pipe or redirect gets Markdown. `--output` also writes the Markdown table to a file.
Run it from the repository root with `uv run python -m scripts.report_size --help`.

A version is a git ref this repository resolves (branch, tag, SHA or remote-tracking
ref), or the path to a built `.whl`. To measure uncommitted work, run `uv build` and
pass the wheel from `dist/`. Without `--before`, the before side is the merge-base of
the after side and `--base`. When `--base` is the branch the after side started from
(`origin/release/X.Y.Z` for work on a release), the table shows what that work changed.

Reports and their sizes are cached under `reports/report-size/`, keyed by version,
grade options and dataset revision, so a rerun grades only what it hasn't seen.
"""
