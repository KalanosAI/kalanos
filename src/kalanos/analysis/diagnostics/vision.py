"""Reproducible sampled image diagnostics with bounded decoder work."""

import hashlib
from types import SimpleNamespace

from kalanos.analysis.adapters.video import DecoderUnavailable
from kalanos.analysis.diagnostics.common import Unavailable, result
from kalanos.analysis.diagnostics.previews import png_thumbnail
from kalanos.analysis.localization import support_for
from kalanos.analysis.models.diagnostics import DiagnosticReviewPolicy


def selected_rows(count, budget):
    """Choose deterministic stratified adjacent pairs, including segment edges."""
    if count <= budget:
        return list(range(count))
    if budget == 2:
        return [0, count - 1]
    if budget == 3:
        return [0, 1, count - 1]
    pairs = budget // 2
    starts = [round(i * (count - 2) / max(1, pairs - 1)) for i in range(pairs)]
    return sorted({j for i in starts for j in (i, i + 1)})


def vision(episode, stream, spec, identifier, review=None):
    """Measure sampled frames; duplicate images remain candidates, not faults."""
    review = review or DiagnosticReviewPolicy()
    try:
        import numpy as np
    except ImportError as exc:
        raise Unavailable("vision diagnostics require kalanos[video]") from exc
    if (
        not stream.source_order.preserved
        or stream.source_order.original_index is not None
    ):
        raise Unavailable("sampled video requires the adapter's original frame order")
    payload = stream.payload
    if payload is None or not callable(getattr(payload, "iter_sampled", None)):
        raise Unavailable("payload has no bounded frame-sampling interface")
    requested = selected_rows(len(stream.timestamps), spec.sample_frames)
    if not requested:
        raise Unavailable("no declared frames")
    observations, bad, duplicate_pairs = [], [], []
    prior = None
    summary = {}
    failure = None
    try:
        for item in payload.iter_sampled(
            requested,
            max_decode_frames=spec.max_decode_frames,
            max_pixels=spec.max_pixels,
        ):
            if "summary" in item:
                summary = item["summary"]
                continue
            image = item.pop("image")
            if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 3:
                raise Unavailable("sampled frame is not an RGB image at least 3 by 3")
            gray = image.astype(float).mean(axis=2)
            lap = (
                gray[1:-1, :-2]
                + gray[1:-1, 2:]
                + gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                - 4 * gray[1:-1, 1:-1]
            )
            blur = float(np.var(lap))
            clipped = float(
                np.mean((gray <= spec.dark_level) | (gray >= spec.bright_level))
            )
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            item.update(
                blur_score=blur,
                clipped_fraction=clipped,
                shape=list(image.shape),
                rgb_sha256=digest,
            )
            if len(observations) < spec.preview_frames:
                item["thumbnail_png_base64"] = png_thumbnail(image, spec.preview_size)
            if (
                prior
                and item["source_row"] == prior["source_row"] + 1
                and digest == prior["rgb_sha256"]
            ):
                duplicate_pairs.append([prior["source_row"], item["source_row"]])
            if (
                review.vision_min_blur_score is not None
                and blur < review.vision_min_blur_score
            ) or (
                review.vision_max_clipped_fraction is not None
                and clipped > review.vision_max_clipped_fraction
            ):
                bad.append((item["source_row"], item["source_row"] + 1))
            observations.append(item)
            prior = item
    except DecoderUnavailable as exc:
        raise Unavailable(str(exc)) from exc
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        summary = {"complete": False, "reason": failure, "decoded_frames": None}
    examined = len(observations)
    complete = bool(summary.get("complete"))
    present = {x["source_row"] for x in observations}
    missing = sorted(set(requested) - present)
    interval_errors = []
    for left, right in zip(observations, observations[1:], strict=False):
        a, b = left["source_row"], right["source_row"]
        ta, tb = stream.timestamps[a], stream.timestamps[b]
        if ta is not None and tb is not None and np.isfinite(ta) and np.isfinite(tb):
            interval_errors.append(
                abs(
                    (right["presentation_time_s"] - left["presentation_time_s"])
                    - (tb - ta)
                )
            )
    count = summary.get("segment_frames") if complete else None
    shapes = {tuple(x["shape"]) for x in observations}
    # Duration/count checks are only claims about the media segment, not capture time.
    declared = len(stream.timestamps)
    mismatch = count is not None and count != declared
    measured = result(
        "vision",
        SimpleNamespace(id=identifier),
        episode,
        subject={
            "feature": stream.source_field,
            "instance": stream.instance,
            "source_path": str(stream.source_path),
        },
        measurements={
            "declared_frames": declared,
            "selected_frames": len(requested),
            "examined_frames": examined,
            "decoded_frames": summary.get("decoded_frames", 0),
            "segment_frames": count,
            "presentation_interval_error_max_s": max(interval_errors)
            if interval_errors
            else None,
            "frame_count_difference": count - declared if count is not None else None,
            "adjacent_pairs_examined": sum(
                b["source_row"] == a["source_row"] + 1
                for a, b in zip(observations, observations[1:], strict=False)
            ),
            "identical_adjacent_pairs": duplicate_pairs,
            "distinct_image_shapes": len(shapes),
        },
        evidence={
            "frames": observations,
            "requested_rows": requested,
            "missing_rows": missing,
            "scan": summary,
            "sample_plan": "stratified_adjacent_pairs_v1",
            "parameters": spec.model_dump(),
            "review_policy": review.model_dump(),
            "interpretation": (
                "sampled visual diagnostics; repeated frames can depict "
                "a stationary scene; PTS is not capture time"
            ),
        },
        support=support_for(list(range(declared)), bad),
        consequence="review"
        if bad or review.review_video_integrity and (mismatch or len(shapes) > 1)
        else "report_only",
    )
    if missing or not complete:
        measured.availability = "unavailable"
        measured.reason = (
            summary.get("reason")
            or "not all selected frames and segment boundaries were evaluated"
        )
    if failure:
        measured.availability = "error"
        measured.reason = failure
    return measured
