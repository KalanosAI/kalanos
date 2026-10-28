"""Datasets, grade options and paths the report-size benchmark uses."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "reports" / "report-size"

DEFAULT_BASE = "origin/main"

# Revisions current on 2026-10-23, so numbers stay comparable over time.
DATASETS: tuple[str, ...] = (
    "lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19",
    "lerobot/utokyo_xarm_bimanual@13533b961150faf398ae9b82cf6d6323b56f9070",
    "lerobot/usc_cloth_sim@6c4fb6acf8a6db8c4d0692b318023272c4967b5f",
)

BUNDLE = Path(__file__).resolve().with_name("profile.yaml")
GRADE_FLAGS = ("--tier", "full")
