"""`arm_multi_device.csv`.

Four entities share a nominal 10 ms grid from t=0 to t=490 (50 samples),
each layering one signal pathology onto the same elliptical motion:
armA is the regular-grid negative control, armB flatlines its z channel,
armC drops a 5-sample burst (indices 20-24), and armD wobbles its clock
around the nominal grid instead of landing on it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math

# Internal
from fixture_gen.models import (
    ChannelSpec,
    ClockJitter,
    CsvOutput,
    DropBurst,
    EntitySpec,
    Flatline,
    GroupSpec,
    RegularGrid,
    Rounding,
    Sinusoid,
    TimeSeriesSpec,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# 3dp string formatting on every arm channel, so a flatlined channel
# still reads "75.000" like every other formatted float in the file.
_ARM_ROUNDING = Rounding(digits=3, as_string=True)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _arm_channels(
    cx: float, cy: float, cz: float, r_xy: float, r_y: float, r_z: float
) -> list[ChannelSpec]:
    """Build the x/y/z channel specs for one arm entity.

    Parameters
    ----------
    cx, cy, cz : float
        Ellipse centre.
    r_xy, r_y : float
        xy-plane semi-axes (feed cos for x, sin for y).
    r_z : float
        z-oscillation amplitude, at twice the xy frequency.

    Returns
    -------
    list of ChannelSpec
        `tcp_pose_x_mm`, `tcp_pose_y_mm`, `tcp_pose_z_mm`, in that order.
    """

    return [
        ChannelSpec(
            name="tcp_pose_x_mm",
            value=Sinusoid(offset=cx, amplitude=r_xy, wave="cos"),
            rounding=_ARM_ROUNDING,
        ),
        ChannelSpec(
            name="tcp_pose_y_mm",
            value=Sinusoid(offset=cy, amplitude=r_y, wave="sin"),
            rounding=_ARM_ROUNDING,
        ),
        ChannelSpec(
            name="tcp_pose_z_mm",
            value=Sinusoid(offset=cz, amplitude=r_z, wave="sin", harmonic=2),
            rounding=_ARM_ROUNDING,
        ),
    ]


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


_ARM_GRID = RegularGrid(start=0, step=10, count=50, is_int=True)

_ARM_ENTITIES = [
    EntitySpec(
        id="armA",
        phase0=math.pi / 2,
        groups=[
            GroupSpec(
                channels=_arm_channels(cx=100, cy=50, cz=20, r_xy=10, r_y=5, r_z=2)
            )
        ],
    ),
    EntitySpec(
        id="armB",
        phase0=0.0,
        groups=[
            GroupSpec(channels=_arm_channels(cx=90, cy=40, cz=75, r_xy=8, r_y=6, r_z=0))
        ],
        pathologies=[Flatline(channel="tcp_pose_z_mm", value=75.0)],
    ),
    EntitySpec(
        id="armC",
        phase0=math.pi,
        groups=[
            GroupSpec(
                channels=_arm_channels(cx=110, cy=60, cz=25, r_xy=6, r_y=3.5, r_z=1)
            )
        ],
        pathologies=[DropBurst(start_index=20, count=5)],
    ),
    EntitySpec(
        id="armD",
        phase0=math.pi / 4,
        groups=[
            GroupSpec(
                channels=_arm_channels(cx=95, cy=45, cz=22, r_xy=6, r_y=4, r_z=1.5)
            )
        ],
        pathologies=[ClockJitter(max_offset=4)],
    ),
]

ARM_MULTI_DEVICE = TimeSeriesSpec(
    filename="arm_multi_device.csv",
    grid=_ARM_GRID,
    entities=_ARM_ENTITIES,
    output=CsvOutput(time_field="t_ms", entity_field="id"),
)
