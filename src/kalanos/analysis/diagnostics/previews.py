"""Bounded evidence previews: a compressed frame image and tracking plot points."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import base64
import math
from typing import TYPE_CHECKING, Any

# Internal
from kalanos.analysis.models.diagnostics import DiagnosticResult


if TYPE_CHECKING:
    import numpy as np


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def webp_thumbnail(image: "np.ndarray", size: int) -> str:
    """Encode a nearest-subsampled RGB preview as a base64 WebP image.

    Parameters
    ----------
    image : numpy.ndarray
        The frame, as an `(height, width, 3)` uint8 RGB array.
    size : int
        The longest side, in pixels, the preview is subsampled to at most.

    Returns
    -------
    str
        The WebP file, base64-encoded.

    Raises
    ------
    ImportError
        If PyAV is not installed.
    """

    import av

    step = max(1, math.ceil(max(image.shape[:2]) / size))
    small = image[::step, ::step].astype("uint8")
    codec = av.CodecContext.create("libwebp", "w")
    codec.height, codec.width = small.shape[:2]
    codec.pix_fmt = "yuv420p"
    frame = av.VideoFrame.from_ndarray(small, format="rgb24").reformat(format="yuv420p")
    data = b"".join(bytes(p) for p in codec.encode(frame))
    data += b"".join(bytes(p) for p in codec.encode(None))
    return base64.b64encode(data).decode("ascii")


def tracking_preview(result: DiagnosticResult) -> dict[str, Any] | None:
    """Return numeric-only SVG points for a bounded expected/observed preview."""
    samples = result.evidence.get("samples", []) if result.kind == "tracking" else []
    if len(samples) < 2:
        return None
    step = max(1, math.ceil(len(samples) / 200))
    samples = samples[::step]
    expected = [s.get("expected_response") for s in samples]
    observed = [s.get("observed_response") for s in samples]
    if any(v is None or not math.isfinite(v) for v in expected + observed):
        return None
    low, high = min(expected + observed), max(expected + observed)
    span = high - low or 1

    def points(values: list[float]) -> str:
        """Scale finite measurements to a fixed preview viewport."""
        return " ".join(
            f"{10 + i * 380 / (len(values) - 1):.2f},{110 - (v - low) * 100 / span:.2f}"
            for i, v in enumerate(values)
        )

    return {
        "expected": points(expected),
        "observed": points(observed),
        "samples": len(samples),
        "unit": result.measurements.get("native_unit", ""),
    }
