"""Compare report sizes between two kalanos versions on a set of datasets.

Grades each dataset with both versions and prints a Markdown table of report bytes,
bytes per episode and the change in bytes per episode. Run it from the repository
root with `uv run python -m scripts.report_size --help`.

A version is a git ref this repository resolves (branch, tag, SHA or remote-tracking
ref), or the path to a built `.whl`. To measure uncommitted work, run `uv build` and
pass the wheel from `dist/`. Without `--before`, the before side is the merge-base of
the after side and `--base`. When `--base` is the branch the after side started from
(`origin/release/X.Y.Z` for work on a release), the table shows what that work changed.

Reports and their sizes are cached under `reports/report-size/`, keyed by version,
grade options and dataset revision, so a rerun grades only what it hasn't seen.
"""
