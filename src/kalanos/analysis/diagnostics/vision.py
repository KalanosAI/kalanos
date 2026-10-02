"""Publish a camera stream's shared read as a sampled image diagnostic.

The frames come from the vision metrics' read; nothing here decodes beyond previews.
"""

import math
from types import SimpleNamespace

from kalanos.analysis.diagnostics.common import Unavailable, result
from kalanos.analysis.localization import support_for
from kalanos.analysis.metrics.vision import thumbnail
from kalanos.analysis.models.coverage import Availability
from kalanos.analysis.models.diagnostics import DiagnosticReviewPolicy


def vision(
    episode,
    stream,
    frames,
    identifier,
    review=None,
    *,
    segment_frames=None,
    reason=None,
):
    """Publish a camera's shared read; duplicate images remain candidates, not faults.

    Parameters
    ----------
    episode : Episode
        The episode the camera stream belongs to.
    stream : Stream
        The camera stream.
    frames : CameraFrames | None
        The stream's shared read, `None` when the stream was not read.
    identifier : str
        The result's id.
    review : DiagnosticReviewPolicy | None
        The review triggers, the defaults when `None`.
    segment_frames : int | None
        The frames frame_count_vs_timebase found in the segment,
        `None` when it did not compute.
    reason : str | None
        Why the stream was not read, when `frames` is `None`.

    Returns
    -------
    DiagnosticResult
        The stream's vision result, unavailable or error when its read was.

    Raises
    ------
    Unavailable
        If the stream's source order was not preserved,
        it was not read or it declares no frames.
    """
    review = review or DiagnosticReviewPolicy()
    if (
        not stream.source_order.preserved
        or stream.source_order.original_index is not None
    ):
        raise Unavailable("sampled video requires the adapter's original frame order")
    if frames is None:
        raise Unavailable(reason or "the camera stream was not read")
    requested = frames.requested_rows
    if not requested:
        raise Unavailable("no declared frames")
    parameters = frames.parameters
    observations, bad = [], []
    for frame in frames.frames:
        item = {
            "source_row": frame.source_row,
            "presentation_time_s": frame.presentation_time_s,
            "blur_score": frame.blur_score,
            "clipped_fraction": frame.clipped_fraction,
            "shape": frame.shape,
            "luma_sha256": frame.luma_sha256,
        }
        if len(observations) < parameters["preview_frames"]:
            preview = thumbnail(
                stream.payload, frame.source_row, parameters["preview_size"]
            )
            if preview is not None:
                item["thumbnail_png_base64"] = preview
        if (
            review.vision_min_blur_score is not None
            and frame.blur_score < review.vision_min_blur_score
        ) or (
            review.vision_max_clipped_fraction is not None
            and frame.clipped_fraction > review.vision_max_clipped_fraction
        ):
            bad.append((frame.source_row, frame.source_row + 1))
        observations.append(item)
    missing = frames.missing_rows
    interval_errors = []
    for left, right in zip(observations, observations[1:], strict=False):
        a, b = left["source_row"], right["source_row"]
        ta, tb = stream.timestamps[a], stream.timestamps[b]
        pa, pb = left["presentation_time_s"], right["presentation_time_s"]
        if pa is None or pb is None:
            continue
        if (
            ta is not None
            and tb is not None
            and math.isfinite(ta)
            and math.isfinite(tb)
        ):
            interval_errors.append(abs((pb - pa) - (tb - ta)))
    count = segment_frames
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
            "examined_frames": len(frames.frames),
            "decoded_frames": frames.decoded_frames,
            "segment_frames": count,
            "presentation_interval_error_max_s": max(interval_errors)
            if interval_errors
            else None,
            "frame_count_difference": count - declared if count is not None else None,
            "adjacent_pairs_examined": frames.adjacent_pairs_examined,
            "identical_adjacent_pairs": frames.identical_adjacent_pairs,
            "distinct_image_shapes": len(shapes),
        },
        evidence={
            "frames": observations,
            "requested_rows": requested,
            "missing_rows": missing,
            "scan": {
                "complete": frames.availability == Availability.COMPUTED
                and not missing,
                "decoded_frames": frames.decoded_frames,
                "segment_frames": count,
                "reason": frames.reason,
            },
            "sample_plan": frames.sample_plan,
            "parameters": parameters,
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
    if missing or frames.availability == Availability.UNAVAILABLE:
        measured.availability = Availability.UNAVAILABLE
        measured.reason = (
            frames.reason
            or "not all selected frames and segment boundaries were evaluated"
        )
    if frames.availability == Availability.ERROR:
        measured.availability = Availability.ERROR
        measured.reason = frames.reason or "the camera read failed"
    return measured
