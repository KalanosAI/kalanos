# Synthetic quickstart validation

Prepared 2026-11-16. Status: **validated in a fresh Colab CPU session on 2026-11-16** (all five cells ran unchanged: install, download, grade, HTML report, ZIP download). Observed on the default fixture: 10 episodes, 8 pass, 2 review, quality 94.3, readiness undefined, review gate failed as configured. The sections below record the earlier offline checks.

## Verified
- Public PyPI metadata lists Kalanos 0.10.0, released November 14, 2026; publishing commit `e9fdc8819a465219440f8fbdc8c70230c44b8d7e`.
- Current official CLI documentation exposes `grade PATH --report FILE`, `--fail-on`, and `inspect REPORT.json`.
- Notebook JSON parsing, Python syntax and six offline guard tests passed (see delivered test script).
- Patch application checked on a temporary repository with the matching README title context.

## Verified 2026-11-16 (local run, not Colab)
- Installed the notebook's exact pins (`kalanos[numeric,video]==0.10.0`, `huggingface-hub==1.31.0`) into a fresh venv on Python 3.13 (Colab's Python differs; kalanos requires >=3.10). Resolved: numpy 2.5.5, av 19.0.2, opencv-python-headless 5.0.0.93.
- Ran every code cell in order, unchanged except the Hugging Face download, which was replaced by a generated 3-episode LeRobot v2.1 fixture with injected action spikes (the real dataset was unreachable from the test host). `grade` exited 0, found the injected spikes at rows 60/150/240 on the action channels as report-only findings, and `inspect`, the HTML report (self-contained, one script, no external URLs) and the results ZIP were produced.
- Real CLI exit codes: 0 gate passed; 1 when declared episodes were missing (report still written); 2 for a missing path or `--fail-on pass` (no report). This is what the notebook's error handling assumes.
- `--fail-on blocked,review,unknown`, `grade --report` and `inspect` match `kalanos --help` / `cli.py` for 0.10.0. `HfApi.dataset_info(files_metadata=True)`, `snapshot_download(local_dir, token, max_workers)` and `HF_HUB_DISABLE_IMPLICIT_TOKEN` exist in huggingface-hub 1.31.0.
- `ruff check .` and `ruff format --check .` (the CI jobs) pass with the notebook and test script included.

## Not verified
- Full nbformat schema validation: nbformat is not installed in this workspace.
- Fresh Colab installation, the real `kalanosai/synthetic-action-spikes` download and its layout, declared format, ground-truth contents, live grading, score, HTML browser rendering and browser download.
- The cloud workspace denied direct requests to PyPI and Hugging Face. Web retrieval of the dataset was also unavailable. No broad internet exception was requested.
- Main-repository code was not checked out or changed. No commit, PR, merge or external publication occurred.

## Before launch
1. Apply the patch on a branch and open the notebook in a fresh CPU Colab runtime.
2. Run all cells. Confirm package installation completes and the public snapshot is under the cap.
3. Verify exactly one unpacked LeRobot root; inspect actual rows and ground-truth files separately. Confirm auxiliary ground truth is not graded as recordings.
4. Paste the successful dataset SHA into `DATASET_REVISION`, save, and rerun from a fresh runtime.
5. Record Python, dependencies, dataset SHA, actual format/episode count, grader scope, score (or unavailable reason), decisions and evidence. Do not expect the historical Hub score to match a different engine/scope.
6. Verify exit 1 retains usable reports and exit 2 raises an error. Confirm counts/inventory match the input.
7. Confirm HTML is usable in Colab; if scripts are limited, test the downloaded HTML. Test ZIP download and a second Run all.
8. Add the observed results here, remove draft labels only after validation, and activate promotional badges after merging.

## Sources
- https://pypi.org/project/kalanos/0.10.0/
- https://github.com/KalanosAI/kalanos/blob/main/README.md
- https://github.com/KalanosAI/kalanos/blob/main/src/kalanos/cli.py
- https://huggingface.co/datasets/kalanosai/synthetic-action-spikes
