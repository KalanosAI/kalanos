"""Verifies JsonAdapter: nested-by-serial JSON, its own refusal, and disjointness."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from datetime import datetime, timezone

# External
import pytest

# Internal
from kalanos.analysis.adapters.json import JsonAdapter
from kalanos.analysis.models.adapters import AdapterRefusal
from kalanos.analysis.models.schema import RefusalCode
from kalanos.testing import check_adapter

# Local
from helpers import (
    CAPTURE_INDEX_FIXTURE,
    CSV_FIXTURE,
    IMU_FIXTURE,
    NO_TIMESERIES_FIXTURE,
)


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_passes_the_adapter_contract_suite():
    """Verify JsonAdapter satisfies the Adapter protocol on the real fixture."""

    check_adapter(JsonAdapter(), CAPTURE_INDEX_FIXTURE, CSV_FIXTURE)


def test_detect_bids_on_the_file_it_must_read_in_order_to_refuse():
    """Verify detect bids on video_meta.json, which must be read in order to refuse."""

    assert JsonAdapter().detect(NO_TIMESERIES_FIXTURE) == 0.3


def test_detect_declines_the_jsonl_decoy():
    """Verify detect bids 0.0 on a JSONL file, whose first line alone is complete."""

    assert JsonAdapter().detect(IMU_FIXTURE) == 0.0


def test_detect_bids_on_a_single_record_file(tmp_path):
    """Verify a one-line, one-record file bids here rather than falling through.

    Its first line is a complete JSON value too, same as a JSONL decoy's —
    what tells the two apart is that there is only the one line.
    """

    path = tmp_path / "single.json"
    path.write_text('{"fps": 30, "codec": "h264"}\n')

    assert JsonAdapter().detect(path) == 0.3


def test_capture_index_yields_one_instance_per_camera_serial():
    """Verify the nested-by-serial structure yields exactly the two camera instances."""

    episode = next(JsonAdapter().episodes(CAPTURE_INDEX_FIXTURE))

    instances = {stream.instance for stream in episode.streams}
    assert instances == {"CAM7F3A", "CAM9B21"}


def test_capture_index_streams_are_irregular_with_epoch_second_timestamps():
    """Verify is_regular is False and the first timestamp matches the key's datetime."""

    episode = next(JsonAdapter().episodes(CAPTURE_INDEX_FIXTURE))

    assert all(not stream.is_regular for stream in episode.streams)

    expected_first = datetime(2023, 7, 7, 10, 0, 34, tzinfo=timezone.utc).timestamp()
    first_timestamps = {stream.timestamps.to_list()[0] for stream in episode.streams}
    assert first_timestamps == {expected_first}


def test_video_meta_is_refused_for_having_no_time_index():
    """Verify the flat metadata blob raises AdapterRefusal coded no_time_index."""

    with pytest.raises(AdapterRefusal) as excinfo:
        next(JsonAdapter().episodes(NO_TIMESERIES_FIXTURE))

    assert excinfo.value.code is RefusalCode.NO_TIME_INDEX


def test_a_key_with_no_recognisable_timestamp_does_not_fabricate_one(tmp_path):
    """Verify a record key without a trailing datetime contributes no timestamp."""

    path = tmp_path / "no_datetime_key.json"
    path.write_text(
        json.dumps(
            {
                "unrecognisable-key": {"CAM1": {"exposure_ms": 1.0}},
                "AUTOLab+2023-07-07-10h-00m-34s": {"CAM1": {"exposure_ms": 2.0}},
            }
        )
    )

    parsed = JsonAdapter().parse(path)

    assert parsed.frame["timestamp_s"].null_count() == 1


def test_a_key_whose_digits_name_no_real_date_does_not_fabricate_one(tmp_path):
    """Verify a key matching the pattern but naming an impossible date fabricates none.

    `_timestamp_from_key` only knows a key looks like a datetime; whether
    the digits form one is `datetime`'s to decide, and it can refuse.
    """

    path = tmp_path / "impossible_date_key.json"
    path.write_text(
        json.dumps(
            {
                "AUTOLab+2023-13-40-25h-00m-00s": {"CAM1": {"exposure_ms": 1.0}},
                "AUTOLab+2023-07-07-10h-00m-34s": {"CAM1": {"exposure_ms": 2.0}},
            }
        )
    )

    parsed = JsonAdapter().parse(path)

    assert parsed.frame["timestamp_s"].null_count() == 1
