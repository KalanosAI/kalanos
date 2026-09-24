"""Compose entity_key and time_axis into one SourceSchema, over a parsed frame.

`infer_schema` never refuses — an unresolved time axis comes back as a
`TimeSpec` with `column=None` and evidence saying why, so the decision to
treat that as unacceptable belongs to a caller, not the library.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# External
import polars as pl

# Internal
from kalanos.analysis.inference.entity import entity_key
from kalanos.analysis.inference.timestamp import time_axis
from kalanos.analysis.models.schema import ColumnSpec, SourceSchema


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# fmt: off
_DTYPE_LABELS = {
    "Int":      "int",
    "UInt":     "int",
    "Float":    "float",
    "String":   "str",
    "Boolean":  "bool",
}
# fmt: on


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _dtype_label(dtype: pl.DataType) -> str:
    """Map a polars dtype to the coarse label `ColumnSpec.dtype` carries.

    Parameters
    ----------
    dtype : pl.DataType
        The column's polars dtype.

    Returns
    -------
    str
        `"int"`, `"float"`, `"str"` or `"bool"` for the dtype families
        `loading` and `mapping` care about; the lowercased polars dtype
        name for anything else, so nothing is silently dropped.
    """

    name = str(dtype)
    for prefix, label in _DTYPE_LABELS.items():
        if name.startswith(prefix):
            return label
    return name.lower()


def infer_schema(frame: pl.DataFrame) -> SourceSchema:
    """Resolve a parsed frame's schema: its columns, entity key and time axis.

    Format-agnostic — operates on a frame that has already been parsed,
    which is what lets a CSV reader and a parquet reader that got
    unfamiliar field names share this one code path.

    Parameters
    ----------
    frame : pl.DataFrame
        An already-parsed frame; any reader that can produce one may call this.

    Returns
    -------
    SourceSchema
        Every column's coarse dtype, the entity key if one was found,
        and the resolved time axis. `dialect` is left `None` —
        a frame has already been parsed by the time it reaches here,
        so there is nothing left to sniff.
    """

    columns = [
        ColumnSpec(name=name, dtype=_dtype_label(dtype))
        for name, dtype in frame.schema.items()
    ]
    key = entity_key(frame)
    time = time_axis(frame, entity_column=key.column if key else None)
    return SourceSchema(columns=columns, entity_key=key, time=time)
