"""Verifies the container hints: bitrate against the same camera, and the flags."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import statistics
from collections.abc import Sequence

# External
import pytest

# Internal
from kalanos.analysis.models.report import DatasetWarning
from kalanos.analysis.models.summaries import ContainerSample
from kalanos.analysis.scoring.container_hints import (
    _median_without,
    container_hints,
)


def _sample(index: int = 0, **overrides) -> ContainerSample:
    """A 640x480, 30 fps file of `cam` at 1000 kbps, with no hint raised."""

    fields = {
        "camera": "cam",
        "source_path": f"cam/file-{index:03d}.mp4",
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "bit_rate": 1_000_000,
        "rotation_deg": 0.0,
        "interlaced": False,
        "color_transfer": "bt709",
    }
    return ContainerSample(**{**fields, **overrides})


def _codes(warnings: Sequence[DatasetWarning]) -> list[str]:
    """The codes of `warnings`, in order."""

    return [w.code for w in warnings]


def _with_odd_one(bit_rate: int, others: int = 5) -> list[ContainerSample]:
    """`others` files at 1000 kbps and one named `odd` at `bit_rate`."""

    return [
        *(_sample(i) for i in range(others)),
        _sample(99, source_path="cam/odd.mp4", bit_rate=bit_rate),
    ]


def test_a_file_far_below_its_cameras_median_is_reported():
    """Verify the warning names the camera, file, bitrate and median."""

    warnings = container_hints(_with_odd_one(200_000))

    assert _codes(warnings) == ["low_bitrate"]
    message = warnings[0].message
    assert message.startswith("1 video file has a bitrate below 0.25x")
    assert "cam cam/odd.mp4 200 kbps (median 1000 kbps)" in message


def test_a_file_at_a_third_of_the_median_is_not_low():
    """Verify 300 kbps against a 1000 kbps median raises nothing."""

    assert container_hints(_with_odd_one(300_000)) == []


def test_a_file_far_above_its_cameras_median_is_reported():
    """Verify the warning names the file, its bitrate and the median."""

    warnings = container_hints(_with_odd_one(3_500_000))

    assert _codes(warnings) == ["high_bitrate"]
    assert "above 3x" in warnings[0].message
    assert "cam cam/odd.mp4 3500 kbps (median 1000 kbps)" in warnings[0].message


def test_a_file_just_under_three_times_the_median_is_not_high():
    """Verify 2900 kbps against a 1000 kbps median raises nothing."""

    assert container_hints(_with_odd_one(2_900_000)) == []


def test_a_group_of_fewer_than_five_files_is_not_compared():
    """Verify four files of one camera, size and rate raise no bitrate warning."""

    assert container_hints(_with_odd_one(10_000, others=3)) == []


def test_files_of_another_size_or_rate_form_their_own_group():
    """Verify a sixth file at another resolution or rate neither joins nor is judged."""

    other_size = _sample(50, width=1280, height=720, bit_rate=10_000)
    other_rate = _sample(51, fps=60.0, bit_rate=10_000)
    files = [*(_sample(i) for i in range(5)), other_size, other_rate]

    assert container_hints(files) == []


def test_a_file_shared_by_many_episodes_counts_once():
    """Verify files repeated across episodes are not counted once per episode."""

    four_files = _with_odd_one(200_000, others=3)
    six_files = _with_odd_one(200_000)

    assert container_hints(four_files * 10) == []
    (warning,) = container_hints(six_files * 10)
    assert warning.message.startswith("1 video file has")


def test_only_ten_files_are_listed():
    """Verify twelve high files list ten and count the other two."""

    files = [
        *(_sample(i) for i in range(40)),
        *(_sample(100 + i, bit_rate=5_000_000) for i in range(12)),
    ]

    (warning,) = container_hints(files)

    assert warning.code == "high_bitrate"
    assert warning.message.startswith("12 video files have a bitrate above 3x")
    assert warning.message.count(" kbps (median") == 10
    assert warning.message.endswith("; and 2 more")


def test_rotation_is_reported_with_the_angle():
    """Verify an angle gives a warning, and 0.0 or an unread value none."""

    rotated = [_sample(0, rotation_deg=90.0)]

    (warning,) = container_hints(rotated)

    assert warning.code == "rotation"
    assert warning.message == (
        "1 video file carries rotation metadata: cam cam/file-000.mp4 90 deg"
    )
    assert container_hints([_sample(0, rotation_deg=0.0)]) == []
    assert container_hints([_sample(0, rotation_deg=None)]) == []


def test_an_hdr_transfer_is_reported_by_name():
    """Verify PQ and HLG each warn and name the transfer, and bt709 does not."""

    for transfer in ("smpte2084", "arib-std-b67"):
        (warning,) = container_hints([_sample(0, color_transfer=transfer)])
        assert warning.code == "hdr_transfer"
        assert warning.message.endswith(f"cam cam/file-000.mp4 {transfer}")
    assert container_hints([_sample(0, color_transfer="bt709")]) == []


def test_interlacing_is_reported():
    """Verify an interlaced file warns, and False or an unread value does not."""

    (warning,) = container_hints([_sample(0, interlaced=True)])

    assert warning.code == "interlaced"
    assert warning.message == "1 video file is interlaced: cam cam/file-000.mp4"
    assert container_hints([_sample(0, interlaced=False)]) == []
    assert container_hints([_sample(0, interlaced=None)]) == []


def test_warnings_come_in_a_fixed_order():
    """Verify the codes follow low, high, rotation, HDR, interlaced."""

    files = [
        *(_sample(i) for i in range(10)),
        _sample(20, bit_rate=100_000),
        _sample(21, bit_rate=9_000_000),
        _sample(22, rotation_deg=90.0, color_transfer="smpte2084", interlaced=True),
    ]

    assert _codes(container_hints(files)) == [
        "low_bitrate",
        "high_bitrate",
        "rotation",
        "hdr_transfer",
        "interlaced",
    ]


@pytest.mark.parametrize("values", [[1, 2, 3, 10, 20], [1, 2, 3, 10, 20, 30]])
def test_the_median_without_a_file_matches_a_re_sort(values):
    """Verify the constant-time median equals the median of the list minus the entry."""

    for index in range(len(values)):
        rest = values[:index] + values[index + 1 :]
        assert _median_without(values, index) == statistics.median(rest)
