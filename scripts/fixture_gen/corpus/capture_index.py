"""`capture_index.json`.

Irregular gaps (3-9 s) between records, so rate-based metrics do not apply.
Two camera entities nest one level deeper than the record key.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from fixture_gen.models import (
    ChannelSpec,
    EntitySpec,
    GroupSpec,
    IrregularGrid,
    KeyedJsonOutput,
    Rounding,
    TimeSeriesSpec,
    UniformNoise,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


_CAPTURE_GRID = IrregularGrid(start=34, min_gap=3, max_gap=9, count=12)

_CAPTURE_ROUNDING = Rounding(digits=2)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _capture_camera(serial: str) -> EntitySpec:
    """Build one camera entity for the capture-index fixture.

    Parameters
    ----------
    serial : str
        The camera serial, used as the entity id and nested JSON key.

    Returns
    -------
    EntitySpec
        An entity with `exposure_ms` and `gain` channels.
    """

    return EntitySpec(
        id=serial,
        groups=[
            GroupSpec(
                channels=[
                    ChannelSpec(
                        name="exposure_ms",
                        value=UniformNoise(low=8.0, high=10.0),
                        rounding=_CAPTURE_ROUNDING,
                    ),
                    ChannelSpec(
                        name="gain",
                        value=UniformNoise(low=1.0, high=2.0),
                        rounding=_CAPTURE_ROUNDING,
                    ),
                ]
            )
        ],
    )


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀

_CAPTURE_ENTITIES = [_capture_camera("CAM7F3A"), _capture_camera("CAM9B21")]

CAPTURE_INDEX = TimeSeriesSpec(
    filename="capture_index.json",
    grid=_CAPTURE_GRID,
    entities=_CAPTURE_ENTITIES,
    output=KeyedJsonOutput(
        key_template="AUTOLab+5d05c5aa+2023-07-07-10h-{minutes:02d}m-{seconds:02d}s"
    ),
)
