# /// script
# requires-python = ">=3.10"
# dependencies = ["kalanos[numeric,hf,video]==0.10.2"]
# ///
r"""Grade liftplace and keep the parts of the report that its notebook charts.

The full report is about 186 MB, so `liftplace_findings.json` holds only these parts.
Run on a local copy of the dataset at the notebook's pinned revision:

    hf download zima-robot/liftplace-ee-xyz-500-v2 --repo-type dataset \
        --revision d796a5d6438a2d72f714fde611e7a429650d2c72 \
        --local-dir liftplace-ee-xyz-500-v2
    uv run notebooks/reports/summarize_liftplace.py \
        liftplace-ee-xyz-500-v2 liftplace_findings.json
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import polars as pl

import kalanos


DATASET_ID = "zima-robot/liftplace-ee-xyz-500-v2"
REVISION = "d796a5d6438a2d72f714fde611e7a429650d2c72"
CAMERAS = ("agentview", "wrist")


def summarize(report: kalanos.Report, dataset: Path) -> dict:
    """Keep the parts of a Kalanos report of this dataset that the notebook charts.

    Parameters
    ----------
    report
        The grade of the dataset.
    dataset
        The local copy of the dataset that was graded.

    Returns
    -------
    dict
        Where the grade came from, its decision counts,
        how many camera frames it examined, and one entry per episode.
    """

    # The recording session and the video location are not in the report,
    # so they come from the dataset's own tables.
    info = json.loads((dataset / "meta" / "info.json").read_text())
    meta = pl.read_parquet(dataset / "meta" / "episodes" / "**" / "*.parquet")
    rows = {row["episode_index"]: row for row in meta.iter_rows(named=True)}
    segments = dict(
        pl.read_parquet(
            dataset / "data" / "**" / "*.parquet",
            columns=["episode_index", "collection_segment_index"],
        )
        .unique("episode_index")
        .iter_rows()
    )

    def video_of(row: dict, camera: str) -> tuple:
        """Return the camera, chunk and file index that locate an episode's video."""

        key = f"videos/observation.images.{camera}"
        return camera, row[f"{key}/chunk_index"], row[f"{key}/file_index"]

    # Only an episode alone in its video file owns the frames past the end of its data.
    sharing = Counter(video_of(row, c) for row in rows.values() for c in CAMERAS)
    # Kalanos records a finding for each check that sent an episode to review
    # or blocked it. Its dataset-wide pass can raise an episode's corrupt-frame
    # value above the episode's own, so the finding holds the value that decided.
    findings = {}
    for f in report.findings:
        if f.consequence.value in ("review", "block"):
            findings.setdefault(f.episode_id, []).append(f)

    episodes = []
    for e in report.episodes:
        # Every episode is scored; this only tells the type checker.
        assert e.score.score is not None
        index = int(e.id.rsplit("_", 1)[1])
        row, own = rows[index], findings.get(e.id, [])
        cameras = {}
        for camera in CAMERAS:
            field, key = (
                f"observation.images.{camera}",
                f"videos/observation.images.{camera}",
            )
            metrics = next(s for s in e.streams if s.source_field == field).metrics
            corrupt = next(
                (
                    f
                    for f in own
                    if f.source_field == field
                    and f.metric_id == "vision.corrupt_frame_pct"
                ),
                metrics["corrupt_frame_pct"],
            )
            frame_count = metrics["frame_count_vs_timebase"]
            # Every camera of this dataset has both values;
            # this only tells the type checker.
            assert corrupt.value is not None and frame_count.value is not None
            _, chunk, file = video_of(row, camera)
            cameras[camera] = {
                "corrupt_frame_pct": corrupt.value,
                "frames_examined": corrupt.evidence.get("n_sampled"),
                "flagged_frames": sorted(corrupt.evidence.get("bad_indices") or []),
                "frame_count_vs_timebase": round(frame_count.value, 4),
                "video_frames": frame_count.evidence.get("present"),
                "video_file": info["video_path"].format(
                    video_key=field, chunk_index=chunk, file_index=file
                ),
                "from_s": row[f"{key}/from_timestamp"],
                "to_s": None
                if sharing[video_of(row, camera)] == 1
                else row[f"{key}/to_timestamp"],
            }
        episodes.append(
            {
                "index": index,
                "segment": segments[index],
                "decision": e.eligibility.status.value if e.eligibility else "unknown",
                "score": round(e.score.score, 1),
                "data_frames": row["length"],
                "reasons": sorted(
                    {
                        (
                            f.metric_id.split(".", 1)[1],
                            "/".join(filter(None, [f.source_field, f.channel])),
                        )
                        for f in own
                    }
                ),
                "cameras": cameras,
            }
        )

    counts, coverage = report.eligibility_counts, report.coverage
    # A grade of this dataset always has coverage; this only tells the type checker.
    assert coverage is not None
    return {
        "source": {
            "dataset": DATASET_ID,
            "revision": REVISION,
            "kalanos": report.producer.version if report.producer else None,
            "tier": report.scope.tier.value if report.scope else None,
            "profile": report.scope.requirements_id if report.scope else None,
        },
        "counts": {
            "eligible": counts.pass_count,
            "review": counts.review,
            "blocked": counts.blocked,
            "unknown": counts.unknown,
        }
        if counts
        else None,
        "frames_decoded": coverage.decoded_frames_examined,
        "frames_total": coverage.eligible_visual_frames,
        "episodes": sorted(episodes, key=lambda e: e["index"]),
    }


if __name__ == "__main__":
    dataset, output = Path(sys.argv[1]), Path(sys.argv[2])
    # As the CLI's `--verbosity error`:
    # hides warnings about file formats this dataset doesn't use.
    logging.getLogger("kalanos").setLevel(logging.ERROR)
    # The default tier and profile, as `kalanos grade` without options.
    summary = summarize(kalanos.grade(dataset), dataset)
    output.write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    print(f"Wrote {output}: {summary['counts']}")
