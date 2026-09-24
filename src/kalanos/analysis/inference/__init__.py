"""Work out what a parsed frame's columns mean: time axis, entity key, roles.

A library any reader can call, not a pipeline stage —
it operates on a frame that has already been parsed, so it is as useful to a CSV reader
as to a parquet reader that got unfamiliar field names.
Everything here returns evidence and a confidence rather than a bare verdict,
since a wrong pick corrupts every metric downstream while looking entirely fine.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.inference.entity import entity_key
from kalanos.analysis.inference.infer import infer_schema
from kalanos.analysis.inference.regularity import regularity
from kalanos.analysis.inference.roles import roles
from kalanos.analysis.inference.timestamp import time_axis


__all__ = [
    "entity_key",
    "infer_schema",
    "regularity",
    "roles",
    "time_axis",
]
