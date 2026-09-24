"""`imu_stream.txt`.

200 Hz IMU samples on a regular 5000 us grid, one anonymous entity.
The only pathology this fixture exercises is the container mismatch
(JSONL wearing a .txt extension) — the signal itself is clean.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from fixture_gen.models import (
    ChannelSpec,
    EntitySpec,
    GroupSpec,
    JsonLinesOutput,
    RegularGrid,
    Rounding,
    Sinusoid,
    TimeSeriesSpec,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_IMU_GRID = RegularGrid(start=1_700_000_000_000_000, step=5_000, count=60, is_int=True)

_IMU_ROUNDING = Rounding(digits=4)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀

_IMU_ENTITIES = [
    EntitySpec(
        groups=[
            GroupSpec(
                name="acc",
                channels=[
                    ChannelSpec(
                        name="acc_x",
                        value=Sinusoid(offset=9.81, amplitude=0.3, wave="sin"),
                        rounding=_IMU_ROUNDING,
                    ),
                    ChannelSpec(
                        name="acc_y",
                        value=Sinusoid(amplitude=0.05, wave="cos"),
                        rounding=_IMU_ROUNDING,
                    ),
                    ChannelSpec(
                        name="acc_z",
                        value=Sinusoid(amplitude=0.02, wave="sin", harmonic=2),
                        rounding=_IMU_ROUNDING,
                    ),
                ],
            ),
            GroupSpec(
                name="gyro",
                channels=[
                    ChannelSpec(
                        name="gyro_x",
                        value=Sinusoid(amplitude=0.01, wave="sin"),
                        rounding=_IMU_ROUNDING,
                    ),
                    ChannelSpec(
                        name="gyro_y",
                        value=Sinusoid(amplitude=0.01, wave="cos"),
                        rounding=_IMU_ROUNDING,
                    ),
                    ChannelSpec(
                        name="gyro_z",
                        value=Sinusoid(amplitude=0.005, wave="sin", harmonic=2),
                        rounding=_IMU_ROUNDING,
                    ),
                ],
            ),
        ]
    )
]

IMU_STREAM = TimeSeriesSpec(
    filename="imu_stream.txt",
    grid=_IMU_GRID,
    entities=_IMU_ENTITIES,
    output=JsonLinesOutput(time_field="t_us"),
)
