"""Bounded evidence previews; no image models or external rendering service."""

import base64
import math
import struct
import zlib


def png_thumbnail(image, size):
    """Encode a nearest-subsampled RGB preview with a standard-library PNG writer."""
    step = max(1, math.ceil(max(image.shape[:2]) / size))
    small = image[::step, ::step].astype("uint8")
    height, width = small.shape[:2]
    raw = b"".join(b"\x00" + small[i].tobytes() for i in range(height))

    def chunk(kind, data):
        """Serialize a PNG chunk including its CRC."""
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def tracking_preview(result):
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

    def points(values):
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
