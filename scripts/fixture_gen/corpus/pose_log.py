"""`pose_log.txt`.

Regular 50 Hz, float-seconds timestamps starting at 0.0, with the
column names in a comment line rather than a real header row.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from fixture_gen.models import (
    ChannelSpec,
    DelimitedTextOutput,
    EntitySpec,
    GroupSpec,
    RegularGrid,
    Rounding,
    Sinusoid,
    TimeSeriesSpec,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_POSE_GRID = RegularGrid(start=0.0, step=0.02, count=50, is_int=False, digits=2)

_POSE_ROUNDING = Rounding(digits=4)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀

_POSE_ENTITIES = [
    EntitySpec(
        groups=[
            GroupSpec(
                channels=[
                    ChannelSpec(
                        name="ax",
                        value=Sinusoid(amplitude=0.5, wave="sin"),
                        rounding=_POSE_ROUNDING,
                    ),
                    ChannelSpec(
                        name="ay",
                        value=Sinusoid(amplitude=0.5, wave="cos"),
                        rounding=_POSE_ROUNDING,
                    ),
                    ChannelSpec(
                        name="az",
                        value=Sinusoid(
                            offset=9.81, amplitude=0.1, wave="sin", harmonic=2
                        ),
                        rounding=_POSE_ROUNDING,
                    ),
                ]
            )
        ]
    )
]

POSE_LOG = TimeSeriesSpec(
    filename="pose_log.txt",
    grid=_POSE_GRID,
    entities=_POSE_ENTITIES,
    output=DelimitedTextOutput(comment_header="# timestamp ax ay az"),
)
