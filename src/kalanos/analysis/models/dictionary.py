"""What one dictionary.yaml says: the taxonomy of signals, and how field names reach it.

An entry records what a signal *is*: its unit, its expected shape, the range physics
allows. How good a recording of it has to be is policy, and lives in the policy;
see `kalanos.analysis.models.policy`.

Name matching normalises both sides through `normalise_name`, so a column called
`arm_x_mm` and an alias called `arm` meet at the same key.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import re
from enum import Enum
from functools import cached_property

# External
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal
from kalanos.analysis.models.domain import Kind


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# A taxonomy key is `family.name`: the family groups signals for the report and for
# metric gating, the name identifies one signal within it.
_SLUG = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*"
_TAXONOMY_KEY = re.compile(rf"^{_SLUG}\.{_SLUG}$")

# A category is the slug of one reference-taxonomy heading, e.g. `proprioceptive_state`.
_CATEGORY_SLUG = re.compile(rf"^{_SLUG}$")

# Everything that is not a letter or a digit separates one token from the next,
# which folds `observation.state`, `arm-x`, `q[0]` and `Joint Position` into one form.
_SEPARATORS = re.compile(r"[^a-z0-9]+")

# The axis letters a structured field spells its columns with.
# `_is_axis` also accepts a numeric index, which needs no list.
_AXIS_LETTERS = frozenset({"x", "y", "z", "w"})

# Unit suffixes stripped off the end of a field name, so `arm_x_mm` and `arm` agree.
#
# Every entry is two characters or more: a single trailing letter is usually part
# of the name — `gain_k`, `axis_a`, `joint_q` — and stripping one would resolve
# those to whatever `gain`, `axis` or `joint` means. Right after an axis token,
# though, a single letter is still read as a unit, as in `accel_x_g`.
# fmt: off
_UNIT_SUFFIXES = frozenset(
    {
        # Length
        "um", "mm", "cm", "km", "nm",
        # Angle
        "deg", "rad", "mrad",
        # Time
        "ns", "us", "ms", "sec",
        # Electrical
        "mv", "kv", "ma", "ka",
        # Rate
        "hz", "khz", "rpm",
        # Mass
        "mg", "kg",
        # Pressure
        "pa", "kpa", "bar",
        # Temperature
        "degc", "degf",
        # Proportion
        "pct",
    }
)
# fmt: on


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Modality(str, Enum):
    """What a signal is made of, in the reference taxonomy's own vocabulary.

    Distinct from `Kind`, which says how Kalanos carries the payload.
    `Container` and `Structured` have no payload shape at all, and several modalities —
    depth, tactile, radar — arrive as images without being pictures of anything.
    """

    # fmt: off
    AUDIO       = "audio"
    BOOLEAN     = "boolean"
    CATEGORICAL = "categorical"
    CONTAINER   = "container"
    DEPTH       = "depth"
    EVENT       = "event"
    GEOMETRY    = "geometry"
    GRID        = "grid"
    IMAGE       = "image"
    NUMERIC     = "numeric"
    POINT_CLOUD = "point_cloud"
    RADAR       = "radar"
    STRUCTURED  = "structured"
    TACTILE     = "tactile"
    TEXT        = "text"
    VOLUME      = "volume"
    # fmt: on


class Shape(str, Enum):
    """How many numbers one sample of a signal holds, and what they mean.

    A curated vocabulary rather than the reference tables' prose dimensions:
    metrics need to know whether the members of a stream are interchangeable
    joints or the three axes of one vector, and nothing finer.
    """

    # fmt: off
    SCALAR    = "scalar"     # One number per sample
    PER_JOINT = "per_joint"  # One number per degree of freedom, order set by the robot
    VEC3      = "vec3"       # An x/y/z triple
    QUAT      = "quat"       # An x/y/z/w rotation
    FREE      = "free"       # Anything else: images, meshes, documents, message logs
    # fmt: on


class GroupHintKind(str, Enum):
    """How an adapter should expect a signal's columns to be named."""

    # fmt: off
    INDEXED = "indexed"  # A numeric suffix per member, e.g. `q_0`, `q_1`
    AXES    = "axes"     # An axis letter per member, e.g. `pos_x`, `pos_y`
    NONE    = "none"     # One column, or a naming convention nothing can predict
    # fmt: on


class NormalisedName(BaseModel):
    """One field name reduced to the form both sides of a match are compared in.

    Attributes
    ----------
    stem : str
        The name with its namespace, axis and unit removed. The lookup key.
    axis : str or None
        The axis or index the name ended in, or `None` when it had neither.
    unit : str or None
        The unit the name ended in, or `None` when it had none.
        What the source claimed, which the dictionary entry may contradict.
    """

    model_config = ConfigDict(extra="forbid")

    stem: str
    axis: str | None = None
    unit: str | None = None


class NameMatch(BaseModel):
    """What the dictionary made of one field name.

    An ambiguous name resolves to nothing and reports its candidates instead.
    Two entries claiming one name is a defect in the dictionary and is fixed there;
    taking the first would hand a physics-aware metric the wrong signal,
    with nothing downstream able to notice.

    Attributes
    ----------
    name : str
        The field name as the source spelled it.
    normalised : NormalisedName
        What it was reduced to for matching.
    taxonomy_type : str or None
        The entry it resolved to, or `None` when it is unmapped.
    candidates : list[str]
        The taxonomy types that claimed the name, when more than one did.
        Empty otherwise, including when nothing matched at all.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    normalised: NormalisedName
    taxonomy_type: str | None = None
    candidates: list[str] = Field(default_factory=list)


class DictionaryEntry(BaseModel):
    """One signal the taxonomy knows about.

    `extra="forbid"` is the boundary between fact and policy. A threshold copied
    across from the reference tables, or a note about which file format carries
    the signal, fails to parse rather than settling in as a second place to look
    for a number that belongs in the policy.

    Attributes
    ----------
    label : str
        What the report calls this signal.
    category : str
        The slug of the reference-taxonomy heading it sits under.
    modality : Modality
        What the signal is made of.
    kind : Kind or None
        The payload shape a Stream of this signal carries.
        `None` covers two cases: most of the reference taxonomy's administrative
        and provenance types describe a dataset rather than a timestamped channel
        and have no payload to carry, while a handful of genuinely timestamped types
        (a transform-tree snapshot, an object track) carry a `Modality` —
        usually `structured` — that `Kind`'s payload vocabulary has no bucket for yet.
    shape : Shape
        How many numbers one sample holds.
    unit : str or None
        The unit its values are in, e.g. `N.m`. `None` where the signal has none,
        or where the sources disagree too much for one to be named.
    aliases : list[str]
        Field names seen in the wild for this signal, matched after normalisation.
        Kept deliberately short: an alias that matches nothing costs a little memory,
        while one as generic as `value` costs correctness in every file.
    typical_rate_hz : tuple[float, float] or None
        The sampling rate band this signal is usually recorded at, low to high.
        Context for a reader and a prior for inference, never a grading bound.
    plausible_range : tuple[float, float] or None
        The values physics allows, low to high. A sanity bound:
        a reading outside it is not a bad measurement but a wrong one.
    group_hint : GroupHintKind or None
        How its columns are expected to be named, when there is a convention.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    category: str = Field(pattern=_CATEGORY_SLUG.pattern)
    modality: Modality
    kind: Kind | None = None
    shape: Shape
    unit: str | None = None
    aliases: list[str] = Field(default_factory=list)
    typical_rate_hz: tuple[float, float] | None = None
    plausible_range: tuple[float, float] | None = None
    group_hint: GroupHintKind | None = None

    @model_validator(mode="after")
    def _bands_run_low_to_high(self) -> "DictionaryEntry":
        """Refuse a rate band or a plausible range written the wrong way round.

        Returns
        -------
        DictionaryEntry
            `self`, unchanged, once both bands are ordered.

        Raises
        ------
        ValueError
            If either band's low bound is above its high bound,
            or if a rate band is not strictly positive.
        """

        for field_name in ("typical_rate_hz", "plausible_range"):
            band = getattr(self, field_name)
            if band is not None and band[0] > band[1]:
                raise ValueError(f"{field_name} must run low to high, got {band}")

        if self.typical_rate_hz is not None and self.typical_rate_hz[0] <= 0:
            raise ValueError(
                f"typical_rate_hz must be strictly positive, got {self.typical_rate_hz}"
            )
        return self

    @model_validator(mode="after")
    def _aliases_survive_normalisation(self) -> "DictionaryEntry":
        """Refuse an alias that normalises away to nothing.

        An empty or punctuation-only alias indexes under the empty stem, which is
        what any all-punctuation column name reduces to. The entry would then claim
        every such column.

        Returns
        -------
        DictionaryEntry
            `self`, unchanged, once every alias normalises to a real key.

        Raises
        ------
        ValueError
            If any alias reduces to an empty stem.
        """

        empty = [alias for alias in self.aliases if not normalise_name(alias).stem]
        if empty:
            raise ValueError(f"aliases must normalise to a name, got {empty}")
        return self


class Dictionary(BaseModel):
    """One parsed dictionary.yaml: every signal the taxonomy knows, keyed by type.

    Attributes
    ----------
    schema_version : int
        The dictionary format version, so a future breaking change to this shape
        has something to check against.
    entries : dict[str, DictionaryEntry]
        Signals keyed by taxonomy type, e.g. `proprio.joint_torque`.
        A key's family groups signals for the report and for metric gating; matching
        reads the name after it, which therefore has to be unique across families.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    entries: dict[str, DictionaryEntry]

    @model_validator(mode="after")
    def _taxonomy_keys_are_dotted_slugs(self) -> "Dictionary":
        """Refuse a taxonomy key that is not `family.name`.

        Returns
        -------
        Dictionary
            `self`, unchanged, once every key is a dotted slug.

        Raises
        ------
        ValueError
            If any key does not match the taxonomy key pattern.
        """

        malformed = [key for key in self.entries if not _TAXONOMY_KEY.match(key)]
        if malformed:
            raise ValueError(
                f"taxonomy keys must be lowercase `family.name`, got {malformed}"
            )
        return self

    @model_validator(mode="after")
    def _names_are_unique_across_families(self) -> "Dictionary":
        """Refuse two entries whose keys differ only by family.

        Matching drops the family, so `proprio.position` and `derived.position` would
        both answer to `position` and neither would ever resolve. Two entries sharing
        an alias is a real ambiguity that `resolve` reports; two entries sharing a
        name is a naming choice, and it is made here.

        Returns
        -------
        Dictionary
            `self`, unchanged, once every key has a distinct name.

        Raises
        ------
        ValueError
            If two taxonomy keys reduce to the same name.
        """

        by_name: dict[str, list[str]] = {}
        for key in self.entries:
            by_name.setdefault(normalise_name(key).stem, []).append(key)

        collisions = [keys for keys in by_name.values() if len(keys) > 1]
        if collisions:
            raise ValueError(
                f"taxonomy keys must differ by more than their family, got {collisions}"
            )
        return self

    @cached_property
    def alias_index(self) -> dict[str, tuple[str, ...]]:
        """Map every normalised name this dictionary answers to onto its taxonomy types.

        A key with more than one type is what `resolve` reports as ambiguous.
        The taxonomy key itself is indexed alongside the aliases, so an entry
        does not have to repeat its own name to match a column spelled that way.

        Cached on first use, which is safe because the model is frozen.

        Returns
        -------
        dict[str, tuple[str, ...]]
            Normalised stems, each mapped to the taxonomy types claiming it,
            in the order the entries appear.
        """

        index: dict[str, list[str]] = {}
        for taxonomy_type, entry in self.entries.items():
            for name in (taxonomy_type, *entry.aliases):
                claimants = index.setdefault(normalise_name(name).stem, [])
                if taxonomy_type not in claimants:
                    claimants.append(taxonomy_type)
        return {stem: tuple(claimants) for stem, claimants in index.items()}

    def resolve(self, name: str) -> NameMatch:
        """Look one field name up in the taxonomy.

        Parameters
        ----------
        name : str
            The field name as the source spelled it.

        Returns
        -------
        NameMatch
            The taxonomy type it resolved to; or an unmapped match, carrying the
            candidates when several entries claimed the name and nothing when none did.
        """

        normalised = normalise_name(name)
        claimants = self.alias_index.get(normalised.stem, ())

        if len(claimants) == 1:
            return NameMatch(
                name=name, normalised=normalised, taxonomy_type=claimants[0]
            )
        return NameMatch(name=name, normalised=normalised, candidates=list(claimants))


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _is_axis(token: str) -> bool:
    """Check whether a token names an axis of a structured field.

    Parameters
    ----------
    token : str
        One normalised token.

    Returns
    -------
    bool
        Whether `token` is an axis letter or a numeric index.
    """

    return token in _AXIS_LETTERS or token.isdigit()


def _collapse(text: str) -> str:
    """Reduce text to lowercase tokens joined by single underscores.

    Parameters
    ----------
    text : str
        Any field name or fragment of one.

    Returns
    -------
    str
        The lowercased text with every run of separators turned into one
        underscore, and no leading or trailing underscore.
    """

    return _SEPARATORS.sub("_", text.lower()).strip("_")


def _tail_is_plausibly_a_unit(tail: list[str]) -> bool:
    """Decide whether every token in a name's tail reads as a unit.

    A known multi-letter suffix qualifies, and so does any single letter
    other than an axis or a digit — `g`, `a`, `v` — the units too short for
    `_UNIT_SUFFIXES`. Anything longer and unlisted, like `accel` or `cload`,
    reads as part of the name instead.

    Parameters
    ----------
    tail : list[str]
        The tokens found after a mid-name axis letter.

    Returns
    -------
    bool
        Whether the tail should be read as a unit rather than left alone.
    """

    def _is_unit_token(token: str) -> bool:
        return token in _UNIT_SUFFIXES or (
            len(token) == 1 and token not in _AXIS_LETTERS and not token.isdigit()
        )

    return all(_is_unit_token(token) for token in tail)


def _strip_suffixes(local: str) -> NormalisedName:
    """Split a name at a mid-name axis, or peel a trailing axis and unit off it.

    Parameters
    ----------
    local : str
        A collapsed name with its namespace already removed.

    Returns
    -------
    NormalisedName
        The remaining stem, plus whichever axis and unit were found.
    """

    tokens = local.split("_")
    axis: str | None = None
    unit: str | None = None

    # An axis followed by a plausible unit — `accel_x_g`, `gyro_x_rad_s` — splits
    # there instead of at the end, so each axis doesn't land under its own stem.
    # The plausibility check keeps `motor_1_temp_c` and `F_x_Cload` whole.
    for position in range(len(tokens) - 2, 0, -1):
        tail = tokens[position + 1 :]
        if tokens[position] in _AXIS_LETTERS and _tail_is_plausibly_a_unit(tail):
            axis = tokens[position]
            unit = "_".join(tail)
            tokens = tokens[:position]
            break

    stem = "_".join(tokens)

    # The two suffixes appear in either order — `arm_x_mm` and `t_ms` are both real —
    # so peel until neither matches, remembering each kind once.
    peeled = True
    while peeled and "_" in stem:
        peeled = False
        head, _, last = stem.rpartition("_")
        if axis is None and _is_axis(last):
            stem, axis, peeled = head, last, True
        elif unit is None and last in _UNIT_SUFFIXES:
            stem, unit, peeled = head, last, True

    return NormalisedName(stem=stem, axis=axis, unit=unit)


def normalise_name(name: str) -> NormalisedName:
    """Reduce a field name to the form the dictionary is keyed by.

    Applied identically to a source's columns and to an entry's aliases, which is
    what lets a column called `steps.observation.joint_position` reach an entry
    that only lists `joint_position`.

    Parameters
    ----------
    name : str
        The field name as its source spelled it.

    Returns
    -------
    NormalisedName
        The stem to look up, and the axis and unit the name carried.
    """

    # Step 1: a namespace records where a field sits in its file rather than what it is,
    # so only the last dotted segment survives.
    local = name.rsplit(".", 1)[-1]

    # Step 2: a field spelled one column per axis puts the axis in that last segment,
    # so `observation.state.pose.x` needs the segment before it as well.
    # Two is enough; anything further out is namespace again.
    if _is_axis(_collapse(local)):
        local = ".".join(name.rsplit(".", 2)[-2:])

    # Step 3: what is left loses its trailing axis and unit, and becomes the key.
    return _strip_suffixes(_collapse(local))
