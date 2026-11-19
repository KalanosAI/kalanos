# Kalanos notebooks

## Find what needs review in a robot dataset

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KalanosAI/kalanos/blob/main/notebooks/kalanos_quickstart.ipynb)

The notebook installs Kalanos 0.10.0 into the Colab kernel, downloads the public `lerobot/aloha_sim_transfer_cube_human` dataset, grades it, and charts the signal Kalanos sends episodes to review for.

- Use a CPU runtime. No robot or GPU is needed.
- Kalanos, the two packages the notebook adds and the dataset revision are pinned.
- No Hugging Face token or Kalanos API key is needed, and the notebook publishes nothing.
- The last cell downloads the JSON and HTML reports as a ZIP.

## Audit Cube to Bowl before training

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KalanosAI/kalanos/blob/main/notebooks/kalanos_cube_to_bowl.ipynb)

The notebook installs Kalanos 0.10.0 in its own virtual environment, downloads a pinned commit of the public [Cube to Bowl](https://huggingface.co/datasets/triton7777/cube_to_bowl_so100) SO-100 dataset, grades it with the `vision-imitation-v1` profile at the full tier, and shows the episode it sends to review: one whose gripper camera is less sharp than that camera's usual baseline.

- Use a CPU runtime. No Hugging Face token or Kalanos API key is needed, and the notebook publishes nothing.
- Kalanos, `huggingface-hub` and the dataset revision are pinned, and the download is capped at 200 MB.
- The last cell previews the audit JSON and downloads the JSON report, the HTML report and a provenance file.
- The [published Hub assessment](https://hub.kalanos.ai/d/@kalanos/hf/triton7777/cube_to_bowl_so100) is a separate run, so its counts and findings can differ between grader releases.
