"""R07-04 controls: source order, clock evidence and native tick precision."""

import json
import shutil
from pathlib import Path

import h5py
import polars as pl
import pytest
from mcap.writer import CompressionType
from mcap_ros2.writer import Writer
from pydantic import ValidationError
from upath import UPath

from kalanos.analysis.adapters.csv import CsvAdapter
from kalanos.analysis.adapters.hdf5 import Hdf5Adapter
from kalanos.analysis.adapters.lerobot.v2 import LeRobotV2Adapter
from kalanos.analysis.adapters.lerobot.v3 import LeRobotV3Adapter
from kalanos.analysis.adapters.mcap import McapAdapter
from kalanos.analysis.execution import use_tier
from kalanos.analysis.metrics.registry import run_stream_metrics
from kalanos.analysis.models.domain import (
    Clock,
    ClockInfo,
    ClockOrigin,
    Kind,
    OriginEvidence,
    SourceOrder,
    Stream,
    TimestampDtype,
)
from kalanos.analysis.models.metrics import MetricStatus, StreamContext
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.models.report import Report
from kalanos.api import grade
from kalanos.benchmark import benchmark_dataset


FIXTURES = Path(__file__).parent / "fixtures"
ACQUISITION = ("effective_hz", "dt_jitter_ms", "drop_rate")


def signal(
    values,
    *,
    origin=ClockOrigin.UNKNOWN,
    evidence=OriginEvidence.NONE,
    native=None,
    native_unit="s",
    order=None,
    regular=True,
):
    return Stream(
        taxonomy_type="unmapped.test",
        kind=Kind.SERIES,
        timestamps=pl.Series(values, dtype=pl.Float64),
        native_timestamps=native,
        clock_info=ClockInfo(
            origin=origin, origin_evidence=evidence, native_unit=native_unit
        ),
        source_order=order or SourceOrder(),
        timestamp_dtype=TimestampDtype.FLOAT64,
        source_path=UPath("clock.csv"),
        is_regular=regular,
    )


def metrics(stream):
    return run_stream_metrics(
        StreamContext(stream=stream, is_regular=stream.is_regular)
    )


@pytest.mark.parametrize("origin", list(ClockOrigin))
def test_non_capture_origins_never_certify_acquisition_even_with_producer_evidence(
    origin,
):
    result = metrics(
        signal(
            [i / 50 for i in range(10)], origin=origin, evidence=OriginEvidence.PRODUCER
        )
    )
    for key in ACQUISITION:
        assert (result[key].value is not None) == (origin == ClockOrigin.CAPTURE)
    assert result["recorded_hz"].value == pytest.approx(50)
    assert result["recorded_hz"].evidence["measurement_scope"] == "recorded_timeline"


@pytest.mark.parametrize(
    "evidence", [OriginEvidence.NONE, OriginEvidence.INFERRED, OriginEvidence.ADAPTER]
)
def test_capture_label_requires_producer_evidence(evidence):
    result = metrics(
        signal(
            [0, 0.02, 0.041, 0.06, 0.081], origin=ClockOrigin.CAPTURE, evidence=evidence
        )
    )
    for key in ACQUISITION:
        assert result[key].status == MetricStatus.NOT_APPLICABLE
        assert "producer" in result[key].evidence["reason"]


def test_declared_uniform_capture_clock_is_measurable_without_generation_heuristic():
    result = metrics(
        signal(
            [0, 0.125, 0.25, 0.375, 0.5],
            origin=ClockOrigin.CAPTURE,
            evidence=OriginEvidence.PRODUCER,
        )
    )
    assert result["effective_hz"].value == 8
    assert result["dt_jitter_ms"].value == 0
    assert result["drop_rate"].value == 0


def test_unknown_jitter_is_not_promoted_to_capture():
    result = metrics(signal([0, 0.0201, 0.04, 0.0601, 0.08]))
    assert result["dt_jitter_ms"].value is None
    assert result["recorded_dt_spread_ms"].value > 0


def test_generated_gap_is_visible_without_acquisition_claim():
    result = metrics(
        signal(
            [0, 0.02, 0.04, 0.08, 0.1],
            origin=ClockOrigin.GENERATED,
            evidence=OriginEvidence.ADAPTER,
        )
    )
    assert result["recorded_drop_estimate"].value == pytest.approx(1 / 6)
    assert result["recorded_drop_estimate"].evidence["estimate"] is True
    assert all(result[key].value is None for key in ACQUISITION)


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), float("-inf")])
def test_invalid_tick_breaks_adjacency_and_does_not_shift_sample_address(invalid):
    result = metrics(signal([0, invalid, 0.03, 0.02, 0.02]))
    order = result["monotonic_violations"]
    assert order.value == 2
    assert order.evidence["first_sample"] == 3
    assert order.evidence["sample_indices"] == [3, 4]
    assert order.evidence["n_gaps"] == 2
    assert order.evidence["n_invalid_timestamps"] == 1
    assert result["recorded_hz"].value is None
    bridged = metrics(signal([1, invalid, 0]))["monotonic_violations"]
    assert bridged.value is None


def test_native_epoch_nanoseconds_survive_float_rounding_and_unsigned_backwards_step():
    base = 1_700_000_000_000_000_000
    native = pl.Series([base + i for i in (0, 1, 2, 1, 4)], dtype=pl.UInt64)
    stream = signal([float(x) / 1e9 for x in native], native=native, native_unit="ns")
    assert len(set(stream.timestamps.to_list())) == 1  # float64 really lost the ticks
    order = metrics(stream)["monotonic_violations"]
    assert order.evidence["n_repeated"] == 0
    assert order.evidence["n_backwards"] == 1
    assert order.evidence["first_sample"] == 3


def test_native_tick_rate_subtracts_before_conversion():
    base = 1_700_000_000_000_000_000
    ticks = pl.Series([base + i for i in range(5)], dtype=pl.UInt64)
    result = metrics(
        signal(
            [float(x) / 1e9 for x in ticks],
            native=ticks,
            native_unit="ns",
            origin=ClockOrigin.CAPTURE,
            evidence=OriginEvidence.PRODUCER,
        )
    )
    assert result["effective_hz"].value == pytest.approx(1e9)
    assert result["dt_jitter_ms"].value == 0


def test_sorted_view_with_index_map_still_detects_original_backwards_step():
    order = SourceOrder(
        preserved=False, original_index=[0, 2, 1, 3], transform="sorted_by_timestamp"
    )
    result = metrics(signal([0, 1, 2, 3], order=order))["monotonic_violations"]
    assert result.value == 1
    assert result.evidence["first_sample"] == 2


def test_sorted_view_without_index_map_abstains():
    result = metrics(
        signal(
            [0, 1, 2, 3],
            order=SourceOrder(preserved=False, transform="sorted_by_timestamp"),
        )
    )
    assert result["monotonic_violations"].value is None
    assert "source row order" in result["monotonic_violations"].evidence["reason"]


@pytest.mark.parametrize("indices", [[0], [0, 0], [-1, 0]])
def test_bad_row_maps_are_rejected(indices):
    with pytest.raises(ValidationError):
        signal([0, 1], order=SourceOrder(preserved=False, original_index=indices))


def test_native_tick_lengths_are_checked():
    with pytest.raises(ValidationError):
        signal([0, 1], native=pl.Series([0]))


def test_legacy_capture_label_does_not_manufacture_producer_evidence():
    stream = Stream(
        taxonomy_type="unmapped.test",
        kind=Kind.SERIES,
        timestamps=pl.Series([0, 1]),
        clock=Clock.CAPTURE,
        timestamp_dtype=TimestampDtype.FLOAT64,
        source_path=UPath("legacy.csv"),
    )
    assert stream.clock_info.origin_evidence == OriginEvidence.INFERRED
    assert not stream.clock_info.certifies_acquisition


def test_explicit_clock_info_controls_compatibility_label():
    stream = signal(
        [0, 1], origin=ClockOrigin.PUBLISH, evidence=OriginEvidence.PRODUCER
    )
    assert stream.clock == Clock.LOG


@pytest.mark.parametrize(
    "version,adapter",
    [("lerobot_v2_0_tiny", LeRobotV2Adapter), ("lerobot_v3_tiny", LeRobotV3Adapter)],
)
@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_lerobot_keeps_bad_recorded_order_and_payload_alignment(
    tmp_path, version, adapter, tier
):
    root = tmp_path / version
    shutil.copytree(FIXTURES / version, root)
    path = sorted((root / "data").rglob("*.parquet"))[0]
    frame = pl.read_parquet(path)
    indices = list(range(frame.height))
    indices[2], indices[3] = indices[3], indices[2]
    frame = frame[indices]
    frame.write_parquet(path)
    expected = frame.filter(pl.col("episode_index") == 0)
    with use_tier(tier):
        episode = next(adapter().episodes(UPath(root)))
    stream = next(s for s in episode.streams if s.source_field == "observation.state")
    assert (
        stream.timestamps.to_list() == expected["timestamp"].cast(pl.Float64).to_list()
    )
    assert stream.native_timestamps.dtype == expected["timestamp"].dtype
    assert stream.clock_info.origin == ClockOrigin.GENERATED
    assert stream.clock_info.origin_evidence == OriginEvidence.INFERRED
    assert stream.source_order.preserved
    assert metrics(stream)["monotonic_violations"].evidence["first_sample"] == 3
    if tier == ExecutionTier.STANDARD:
        assert stream.payload.frame[stream.channels[0].name].to_list() == [
            row[0] for row in expected["observation.state"].to_list()
        ]
    else:
        assert stream.payload is None
    video = next(s for s in episode.streams if s.kind == Kind.VIDEO)
    assert video.clock_info.source_field == "timestamp"  # no pretend decoded PTS


def test_csv_preserves_source_row_order_and_native_ticks(tmp_path):
    path = tmp_path / "recording.csv"
    pl.DataFrame(
        {
            "time_s": [0.0, 0.02, 0.04, 0.03, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18],
            "position": [10 + i for i in range(10)],
        }
    ).write_csv(path)
    [episode] = list(CsvAdapter().episodes(UPath(path)))
    [stream] = episode.streams
    assert stream.timestamps[3] == 0.03
    assert stream.payload.frame["position"][3] == 13
    assert stream.source_order.original_index == list(range(10))
    assert metrics(stream)["monotonic_violations"].value == 1
    assert not stream.clock_info.certifies_acquisition


@pytest.mark.parametrize("recorded", [True, False])
def test_hdf5_distinguishes_adapter_generation_from_recorded_clock(tmp_path, recorded):
    path = tmp_path / "episode.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("qpos", data=[[i] for i in range(10)])
        f.attrs["fps"] = 50
        if recorded:
            f.create_dataset(
                "timestamps",
                data=[0.2, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02],
            )
    [episode] = list(Hdf5Adapter().episodes(UPath(path)))
    stream = episode.streams[0]
    if recorded:
        assert stream.clock_info.origin == ClockOrigin.UNKNOWN
        assert stream.clock_info.native_unit == "unknown"
        assert metrics(stream)["monotonic_violations"].value == 9
        assert stream.timestamps[0] > stream.timestamps[-1]
    else:
        assert stream.clock_info.origin == ClockOrigin.GENERATED
        assert stream.clock_info.origin_evidence == OriginEvidence.ADAPTER
        assert metrics(stream)["recorded_hz"].value == pytest.approx(50)
    assert not stream.clock_info.certifies_acquisition


_STAMPED = """std_msgs/Header header
float64 value
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


@pytest.mark.parametrize(
    "mode,origin",
    [
        ("header", ClockOrigin.UNKNOWN),
        ("publish", ClockOrigin.PUBLISH),
        ("log", ClockOrigin.LOG),
    ],
)
def test_mcap_preserves_message_order_and_distinguishes_stamp_semantics(
    tmp_path, mode, origin
):
    path = tmp_path / "source.mcap"
    with path.open("wb") as f:
        writer = Writer(f, compression=CompressionType.NONE)
        schema = writer.register_msgdef("test_msgs/msg/Stamped", _STAMPED)
        for i, ns in enumerate(
            [1_000_000_000, 1_020_000_000, 1_010_000_000, 1_030_000_000, 1_040_000_000]
        ):
            writer.write_message(
                "/recording",
                schema,
                {
                    "header": {
                        "stamp": {
                            "sec": ns // 1_000_000_000 if mode == "header" else 0,
                            "nanosec": ns % 1_000_000_000 if mode == "header" else 0,
                        },
                        "frame_id": "",
                    },
                    "value": float(i),
                },
                log_time=ns,
                publish_time=ns + 100 if mode == "publish" else ns,
            )
        writer.finish()
    [episode] = list(McapAdapter().episodes(UPath(path)))
    [stream] = episode.streams
    assert stream.clock_info.origin == origin
    assert stream.clock_info.epoch is None
    assert stream.payload.frame["value"].to_list() == list(range(5))
    assert metrics(stream)["monotonic_violations"].evidence["first_sample"] == 2
    assert all(metrics(stream)[key].value is None for key in ACQUISITION)
    assert stream.native_timestamps[2] < stream.native_timestamps[1]


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_report_round_trip_retains_clock_evidence_and_benchmark_agrees(tier):
    root = FIXTURES / "lerobot_v3_tiny"
    report = grade(root, tier=tier)
    restored = Report.model_validate_json(report.model_dump_json())
    for episode in restored.episodes:
        for stream in episode.streams:
            assert stream.clock_info.origin == ClockOrigin.GENERATED
            assert stream.clock_info.origin_evidence == OriginEvidence.INFERRED
            assert stream.source_order.preserved
            assert all(stream.metrics[key].value is None for key in ACQUISITION)
    bench = benchmark_dataset(str(root), sample=0, tier=tier)
    assert report.run.binding == bench.configuration["binding"]
    assert report.scope == bench.scope
    # The JSON records provenance without duplicating entire native arrays.
    assert (
        "native_timestamps"
        not in json.loads(report.model_dump_json())["episodes"][0]["streams"][0]
    )


@pytest.mark.parametrize("operation", ["drop", "repeat", "jitter", "drift"])
def test_injectors_update_native_clock_state_without_promoting_generated_time(
    operation,
):
    from kalanos.analysis.models.domain import Channel, FramePayload
    from kalanos.testing.injectors import (
        drop_samples,
        jitter_clock,
        repeat_timestamps,
        stretch_clock,
    )

    stream = signal(
        [i / 100 for i in range(100)],
        origin=ClockOrigin.GENERATED,
        evidence=OriginEvidence.ADAPTER,
        native=pl.Series([i * 10_000_000 for i in range(100)]),
        native_unit="ns",
    )
    stream = stream.model_copy(
        update={
            "payload": FramePayload(frame=pl.DataFrame({"x": list(range(100))})),
            "channels": [Channel(name="x")],
        }
    )
    changed = {
        "drop": lambda: drop_samples(stream, start=30, count=10),
        "repeat": lambda: repeat_timestamps(stream, start=30, count=3),
        "jitter": lambda: jitter_clock(stream),
        "drift": lambda: stretch_clock(stream),
    }[operation]()
    # Revalidate lengths: a stale native array must never silently mask injection.
    changed = Stream.model_validate(
        {**changed.model_dump(), "payload": changed.payload}
    )
    result = metrics(changed)
    assert all(result[key].value is None for key in ACQUISITION)
    assert changed.clock_info.origin == ClockOrigin.GENERATED
    if operation == "drop":
        assert result["recorded_drop_estimate"].value > 0.09
        assert len(changed.native_timestamps) == 90
    elif operation == "repeat":
        assert result["monotonic_violations"].value == 3
    elif operation == "jitter":
        assert result["recorded_dt_spread_ms"].value > 0
    else:
        assert result["recorded_hz"].value < 100
    assert stream.native_timestamps is not None and len(stream.native_timestamps) == 100


def test_unknown_time_units_abstain_from_seconds_but_still_check_order():
    result = metrics(signal([0, 2, 1, 3, 4], native_unit="unknown"))
    assert result["recorded_hz"].value is None
    assert result["monotonic_violations"].value == 1


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_lerobot_invalid_timestamp_is_retained_for_audit(tmp_path, invalid):
    root = tmp_path / "recording"
    shutil.copytree(FIXTURES / "lerobot_v3_tiny", root)
    path = root / "data/chunk-000/file-000.parquet"
    frame = pl.read_parquet(path)
    ticks = frame["timestamp"].to_list()
    ticks[2] = invalid
    frame.with_columns(pl.Series("timestamp", ticks, dtype=pl.Float32)).write_parquet(
        path
    )
    report = grade(root)
    stream = report.episodes[0].streams[0]
    result = stream.metrics["monotonic_violations"]
    assert result.evidence["n_invalid_timestamps"] == 1
    assert result.evidence["first_invalid_sample"] == 2
    assert stream.metrics["recorded_hz"].value is None
    assert stream.clock_info.origin == ClockOrigin.UNKNOWN


def test_hdf5_nonfinite_named_ticks_are_not_replaced_with_a_generated_grid(tmp_path):
    path = tmp_path / "invalid.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("qpos", data=[[i] for i in range(5)])
        f.create_dataset("timestamps", data=[0.0, 0.02, float("nan"), 0.06, 0.08])
    [episode] = list(Hdf5Adapter().episodes(UPath(path)))
    stream = episode.streams[0]
    assert stream.clock_info.origin == ClockOrigin.UNKNOWN
    result = metrics(stream)["monotonic_violations"]
    assert result.evidence["n_invalid_timestamps"] == 1
    assert result.evidence["n_gaps"] == 2


def test_unknown_native_units_still_preserve_exact_integer_order():
    base = 1_700_000_000_000_000_000
    native = pl.Series([base, base + 1, base + 2, base + 1], dtype=pl.UInt64)
    stream = signal([float(v) for v in native], native=native, native_unit="unknown")
    result = metrics(stream)
    assert result["monotonic_violations"].evidence["n_backwards"] == 1
    assert result["monotonic_violations"].evidence["n_repeated"] == 0
    assert result["recorded_hz"].value is None
