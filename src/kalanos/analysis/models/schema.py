"""What schema inference determines about one parsed frame, before mapping."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import Enum

# External
from pydantic import BaseModel, Field

# Internal
from kalanos.analysis.models.dictionary import GroupHintKind
from kalanos.analysis.models.paths import AnyPath


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class RefusalCode(str, Enum):
    """Why an adapter that bid on a path then could not read it."""

    # fmt: off
    UNSPECIFIED   = "unspecified"
    NO_TIME_INDEX = "no_time_index"
    # fmt: on


class ColumnSpec(BaseModel):
    """One column as inference found it, before mapping assigns it meaning.

    Attributes
    ----------
    name : str
        The column name as it appears in the source.
    dtype : str
        The value type inference detected, e.g. `"float"`, `"int"`, `"str"`.
    """

    name: str
    dtype: str


class SamplingRegularity(BaseModel):
    """How regular a time column's sampling is, as inference determined it.

    Attributes
    ----------
    is_regular : bool
        Whether samples arrive at a consistent interval.
    expected_dt : float or None
        The nominal interval between samples, in the time column's own unit,
        or `None` when no regular interval could be established.
    confidence : float
        The fraction of gaps that landed within tolerance of the nominal interval,
        in `[0, 1]`. Carries the same value whether or not that fraction cleared
        the threshold for `is_regular`.
    evidence : list[str]
        What the gap distribution looked like, independent of the verdict.
    """

    is_regular: bool
    expected_dt: float | None = None
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class TimeSpec(BaseModel):
    """Where time lives in a parsed frame, as inference determined it.

    Attributes
    ----------
    column : str or None
        The column inference identified as the time axis, or `None` when no
        candidate cleared the acceptance threshold. `confidence` and
        `evidence` still describe the best candidate considered and why it
        was refused.
    unit : str
        The time unit as found, e.g. `"ms"`, `"s"`, `"ns"`, or `"unknown"`
        when neither the sampling magnitude nor the column name could resolve one.
        Read from the sampling interval's magnitude, corroborated by a unit suffix on
        `column`'s name when it has one; magnitude wins on disagreement.
    confidence : float
        How confident inference is in this pick, in `[0, 1]` —
        that `column` is the time axis and that `unit` is right.
        Drops when the name and the magnitude disagree on the unit.
    evidence : list[str]
        The signals behind `confidence`: name pattern, monotonicity,
        and the magnitude/name unit check.
    regularity : SamplingRegularity
        How regular `column`'s sampling is. Computed from the same gaps that resolve
        `unit`, so it travels with the time axis rather than being derived again.
    """

    column: str | None
    unit: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    regularity: SamplingRegularity = Field(
        default_factory=lambda: SamplingRegularity(is_regular=False, expected_dt=None)
    )


class CsvDialect(BaseModel):
    """The CSV dialect inference detected, rather than one a caller configured.

    Attributes
    ----------
    delimiter : str
        The field delimiter detected.
    has_header : bool
        Whether the first row holds column names rather than data.
    confidence : float
        How confident inference is in this dialect, in `[0, 1]`.
    evidence : list[str]
        What the sniff observed to reach `confidence`.
    """

    delimiter: str
    has_header: bool
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class EntityKey(BaseModel):
    """The column inference identified as splitting a Source into entities.

    Attributes
    ----------
    column : str
        The column whose values distinguish one entity from another.
    confidence : float
        How confident inference is that `column` is an entity key, in `[0, 1]`.
    evidence : list[str]
        The signals that produced `confidence` — name pattern and cardinality.
    """

    column: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class ColumnRole(BaseModel):
    """What the dictionary made of one column name, as `roles` reports it.

    Attributes
    ----------
    column : str
        The column name as its source spelled it.
    taxonomy_type : str or None
        The taxonomy type the name resolved to,
        or `None` when it matched nothing or matched more than one entry.
    axis : str or None
        The axis or index the name ended in, or `None` when it had neither.
    unit : str or None
        The unit the name ended in, or `None` when it had none —
        what the source claimed, which the dictionary entry's own `unit` may contradict.
    candidates : list[str]
        The taxonomy types that claimed the name, when more than one did.
        Empty otherwise, including when nothing matched at all.
    confidence : float
        `1.0` on a unique match, `0.0` when the name is unmapped or ambiguous —
        dictionary lookup is exact, so there is no partial credit to report.
    evidence : list[str]
        What the lookup found: the normalised stem and how it resolved.
    group_hint : GroupHintKind or None
        How the matched entry expects its members to be named,
        or `None` when nothing resolved.
    """

    column: str
    taxonomy_type: str | None = None
    axis: str | None = None
    unit: str | None = None
    candidates: list[str] = Field(default_factory=list)
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    group_hint: GroupHintKind | None = None


class SourceSchema(BaseModel):
    """What inference determined about one Source.

    Attributes
    ----------
    columns : list[ColumnSpec]
        Every column inference identified, with its inferred type.
    dialect : CsvDialect or None
        The CSV dialect detected, or `None` if not yet assessed.
    entity_key : EntityKey or None
        The column that splits the Source into entities,
        or `None` when no such column was found — the whole file is then one subject.
    time : TimeSpec or None
        Where time lives, or `None` if not yet assessed. A `TimeSpec` with
        `column=None` means assessed and refused, which is not the same thing.
    """

    columns: list[ColumnSpec] = Field(default_factory=list)
    dialect: CsvDialect | None = None
    entity_key: EntityKey | None = None
    time: TimeSpec | None = None


class UnresolvedSource(BaseModel):
    """A Source that could not be carried through to grading, with what was known.

    Most commonly a file `inference` can't resolve: no guessed time column,
    no raised exception, just whatever partial schema was determined before
    resolution stopped. The same model also carries a source `inference`
    resolved in full but a later stage still had to refuse — `schema_so_far`
    is then a complete `SourceSchema`, and `reason` says what that stage
    found wrong with it.

    Attributes
    ----------
    path : UPath
        Location of the file that could not be graded.
    schema_so_far : SourceSchema
        Whatever was determined before resolution stopped,
        or the full schema a later stage refused to carry forward.
    reason : str
        Why resolution stopped, or why the resolved schema was refused,
        for a human reading the report.
    code : RefusalCode
        The machine-readable counterpart to `reason`.
    """

    path: AnyPath
    schema_so_far: SourceSchema
    reason: str
    code: RefusalCode = RefusalCode.UNSPECIFIED
