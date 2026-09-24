"""Verifies dictionary.yaml: how a field name is normalised and resolved,
what the schema refuses to carry, and the loader's failure modes.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from importlib import resources
from pathlib import Path

# External
import pytest
import yaml

# Internal
from kalanos.analysis.models.dictionary import (
    Dictionary,
    DictionaryEntry,
    GroupHintKind,
    Modality,
    Shape,
    normalise_name,
)
from kalanos.analysis.models.domain import Kind
from kalanos.assets.dictionary import (
    clear_active_dictionary,
    load_default_dictionary,
    load_dictionary,
    use_dictionary,
)

# Local
from helpers import CSV_FIXTURE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The entry from the design sketch, verbatim enough to catch a field the schema
# never grew — every key here is one the reference tables actually supply.
_JOINT_TORQUE = {
    "label": "Joint torque",
    "category": "proprioceptive_state",
    "modality": "numeric",
    "kind": "series",
    "unit": "N.m",
    "shape": "per_joint",
    "aliases": ["effort", "tau_J", "qfrc_actuator", "joint_torque"],
    "typical_rate_hz": [100, 1000],
    "plausible_range": [-200, 200],
    "group_hint": "indexed",
}

# The minimum an entry needs, for tests about something other than its content.
_BARE = {
    "label": "Anything",
    "category": "proprioceptive_state",
    "modality": "numeric",
    "kind": "series",
    "shape": "scalar",
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _dictionary(**entries: dict) -> Dictionary:
    """Build a Dictionary from taxonomy-type/entry pairs.

    Underscores in a keyword name stand in for the dot in a taxonomy type,
    so `proprio_joint_torque=...` becomes `proprio.joint_torque`.

    Parameters
    ----------
    **entries : dict
        Entry payloads, keyed by taxonomy type with its dot written as the
        first underscore.

    Returns
    -------
    Dictionary
        The validated dictionary.
    """

    return Dictionary.model_validate(
        {
            "schema_version": 1,
            "entries": {
                key.replace("_", ".", 1): value for key, value in entries.items()
            },
        }
    )


def _written(tmp_path: Path, payload: dict) -> Path:
    """Write a dictionary payload out as YAML and return its path.

    Parameters
    ----------
    tmp_path : Path
        Pytest's per-test temporary directory.
    payload : dict
        What to serialise.

    Returns
    -------
    Path
        The file written.
    """

    path = tmp_path / "dictionary.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_name_gives_up_its_axis_and_unit_and_keeps_its_stem():
    """Verify `arm_x_mm` reduces to `arm`, with the axis and the unit remembered.

    Both suffixes have to come off for the stem to match an entry aliased `arm`,
    and both are worth keeping: the axis says which column of a vector this is,
    and the unit is what the source claimed the values are in.
    """

    normalised = normalise_name("arm_x_mm")

    assert normalised.stem == "arm"
    assert normalised.axis == "x"
    assert normalised.unit == "mm"


def test_a_unit_suffix_does_not_make_a_short_name_collide_with_a_longer_one():
    """Verify `t_ms` resolves to the entry aliased `t`, never to `target`.

    Stripping `ms` leaves `t`, which is a prefix of several ordinary field names.
    Matching on the whole stem is what keeps them apart.
    """

    dictionary = _dictionary(
        sync_timestamp={**_BARE, "aliases": ["t"]},
        control_target={**_BARE, "aliases": ["target"]},
    )

    match = dictionary.resolve("t_ms")

    assert match.normalised.stem == "t"
    assert match.taxonomy_type == "sync.timestamp"


def test_a_namespace_is_dropped_so_a_nested_field_reaches_a_flat_alias():
    """Verify a dotted path resolves on its last segment."""

    assert normalise_name("steps.observation.joint_position").stem == "joint_position"


def test_an_axis_segment_is_not_mistaken_for_a_namespace():
    """Verify `pose.x` keeps `pose` as its stem.

    A field written one column per axis puts the axis in the last dotted segment,
    so dropping the namespace blindly would leave nothing but `x`.
    """

    normalised = normalise_name("pose.x")

    assert normalised.stem == "pose"
    assert normalised.axis == "x"


def test_an_axis_survives_a_namespace_of_any_depth():
    """Verify a deeply nested field spelled per axis keeps the field it belongs to.

    `observation.state.pose.x` is the ordinary RLDS spelling. Reading the axis as
    the whole name loses `pose`; keeping the namespace makes it match nothing.
    """

    normalised = normalise_name("observation.state.pose.x")

    assert normalised.stem == "pose"
    assert normalised.axis == "x"


def test_a_bracketed_index_reads_as_an_axis():
    """Verify `q[0]` reduces to `q` with the index kept."""

    normalised = normalise_name("q[0]")

    assert normalised.stem == "q"
    assert normalised.axis == "0"


def test_a_single_trailing_letter_is_not_read_as_a_unit():
    """Verify `gain_k`, `axis_a` and `joint_q` all keep their whole name.

    Single letters are units too, but far more often they are part of the name.
    Stripping one would resolve `gain_k` to whatever `gain` means.
    """

    assert normalise_name("gain_k").stem == "gain_k"
    assert normalise_name("gain_k").unit is None
    assert normalise_name("axis_a").stem == "axis_a"
    assert normalise_name("joint_q").stem == "joint_q"


def test_an_axis_makes_the_token_after_it_a_unit():
    """Verify `accel_x_g` reduces to the same stem as its y/z siblings.

    The axis is what tells you the trailing letter is a unit rather than
    part of the name, unlike `gain_k`.
    """

    normalised = normalise_name("accel_x_g")

    assert normalised.stem == "accel"
    assert normalised.axis == "x"
    assert normalised.unit == "g"
    assert normalise_name("accel_y_g").stem == normalised.stem


def test_a_multi_token_unit_after_an_axis_comes_off_whole():
    """Verify `gyro_x_rad_s` reduces to `gyro` with the full unit kept."""

    normalised = normalise_name("gyro_x_rad_s")

    assert normalised.stem == "gyro"
    assert normalised.unit == "rad_s"


def test_a_numeric_index_does_not_turn_what_follows_into_a_unit():
    """Verify `motor_1_temp_c` keeps its whole name.

    A digit followed by more tokens is a device or joint suffix, not an axis
    marking a unit.
    """

    assert normalise_name("motor_1_temp_c").stem == "motor_1_temp_c"


def test_a_name_that_is_only_an_axis_and_a_unit_keeps_its_whole_name():
    """Verify `x_g` is not reduced to an empty stem."""

    assert normalise_name("x_g").stem == "x_g"


def test_a_non_unit_tail_after_an_axis_is_not_split():
    """Verify `F_x_Cload` keeps its whole name rather than reducing to `f`.

    `Cload` is libfranka's name for a load's centre of mass, not a unit, and
    reading it as one would resolve the column to whatever the bare stem `f`
    happens to mean.
    """

    assert normalise_name("F_x_Cload").stem == "f_x_cload"


def test_an_alias_resolves_to_its_taxonomy_type():
    """Verify a name listed under an entry reaches that entry."""

    dictionary = _dictionary(proprio_joint_torque=_JOINT_TORQUE)

    assert dictionary.resolve("tau_J").taxonomy_type == "proprio.joint_torque"


def test_the_taxonomy_key_matches_without_being_repeated_as_an_alias():
    """Verify an entry answers to its own name.

    `proprio.joint_position` has no alias list here, and a column spelled
    `joint_position` still has to reach it.
    """

    dictionary = _dictionary(proprio_joint_position=_BARE)

    assert (
        dictionary.resolve("joint_position").taxonomy_type == "proprio.joint_position"
    )


def test_an_ambiguous_name_is_unmapped_and_carries_every_candidate():
    """Verify two entries claiming one name resolve to neither, and say which two.

    Taking the first would hand a physics-aware metric the wrong signal, and
    nothing downstream could tell.
    """

    dictionary = _dictionary(
        proprio_joint_position={**_BARE, "aliases": ["pos"]},
        derived_tcp_position={**_BARE, "aliases": ["pos"]},
    )

    match = dictionary.resolve("pos")

    assert match.taxonomy_type is None
    assert match.candidates == ["proprio.joint_position", "derived.tcp_position"]


def test_an_unknown_name_is_unmapped_with_no_candidates():
    """Verify a name nothing claims comes back with an empty candidate list."""

    match = _dictionary(proprio_joint_torque=_JOINT_TORQUE).resolve("widget_count")

    assert match.taxonomy_type is None
    assert match.candidates == []


def test_two_entries_whose_keys_differ_only_by_family_are_rejected():
    """Verify a name shared across two families fails to load.

    Matching drops the family, so both entries would answer to `position` and
    neither would ever resolve. The dictionary is where that is fixed.
    """

    with pytest.raises(ValueError, match="differ by more than their family"):
        _dictionary(proprio_position=_BARE, derived_position=_BARE)


def test_an_alias_that_normalises_away_to_nothing_is_rejected():
    """Verify a punctuation-only alias fails to load.

    It would index under the empty stem, which is also what a column named `---`
    reduces to, and the entry would claim every one of them.
    """

    with pytest.raises(ValueError, match="aliases must normalise to a name"):
        DictionaryEntry.model_validate({**_BARE, "aliases": ["-"]})


def test_a_rate_band_that_is_not_strictly_positive_is_rejected():
    """Verify a rate band starting at zero fails to load.

    Zero hertz is not a slow signal but an absent one, and the band is read as a
    prior by inference.
    """

    with pytest.raises(ValueError, match="strictly positive"):
        DictionaryEntry.model_validate({**_BARE, "typical_rate_hz": [0, 100]})


def test_a_threshold_from_the_reference_tables_is_rejected():
    """Verify an entry cannot carry a grading threshold.

    `Good_Threshold` and `Key_Quality_Metrics` sit next to the facts in the source
    tables and are policy. A second home for a threshold is a second answer to
    what counts as good.
    """

    with pytest.raises(ValueError, match="good_threshold"):
        DictionaryEntry.model_validate({**_BARE, "good_threshold": 0.01})


def test_file_format_knowledge_is_rejected():
    """Verify an entry cannot carry which format the signal arrives in.

    That is the adapter's business, and an entry claiming it would go stale the
    first time a new format carried the same signal.
    """

    with pytest.raises(ValueError, match="typical_format"):
        DictionaryEntry.model_validate({**_BARE, "typical_format": "HDF5/RLDS/MCAP"})


def test_a_band_written_backwards_is_rejected():
    """Verify a rate band or plausible range must run low to high."""

    for field, band in (
        ("plausible_range", [200, -200]),
        ("typical_rate_hz", [1000, 100]),
    ):
        with pytest.raises(ValueError, match="must run low to high"):
            DictionaryEntry.model_validate({**_BARE, field: band})


def test_a_taxonomy_key_that_is_not_family_dot_name_is_rejected():
    """Verify a key without a family fails to validate.

    Metric gating reads the family off the key, so a bare name has nothing for it
    to gate on.
    """

    with pytest.raises(ValueError, match=r"family\.name"):
        Dictionary.model_validate(
            {"schema_version": 1, "entries": {"joint_torque": _BARE}}
        )


def test_the_sketched_entry_loads_with_every_field_intact(tmp_path):
    """Verify the design sketch's entry survives a round trip through the loader."""

    path = _written(
        tmp_path,
        {"schema_version": 1, "entries": {"proprio.joint_torque": _JOINT_TORQUE}},
    )

    entry = load_dictionary(path).entries["proprio.joint_torque"]

    assert entry.modality is Modality.NUMERIC
    assert entry.kind is Kind.SERIES
    assert entry.shape is Shape.PER_JOINT
    assert entry.unit == "N.m"
    assert entry.typical_rate_hz == (100, 1000)
    assert entry.plausible_range == (-200, 200)
    assert entry.group_hint is GroupHintKind.INDEXED


def test_a_syntactically_invalid_dictionary_fails_at_load_not_at_first_use(tmp_path):
    """Verify broken YAML is rejected the moment it is loaded."""

    broken = tmp_path / "dictionary.yaml"
    broken.write_text("entries: [this, is, not: a mapping", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid YAML"):
        load_dictionary(broken)


def test_a_schema_invalid_dictionary_fails_at_load_not_at_first_use(tmp_path):
    """Verify YAML that parses but violates the schema is still rejected up front.

    `modality` is missing here — a well-formed file that simply forgot a required
    field, the kind of mistake a hand edit produces.
    """

    path = _written(
        tmp_path,
        {
            "schema_version": 1,
            "entries": {
                "proprio.joint_torque": {
                    "label": "Joint torque",
                    "category": "proprioceptive_state",
                    "kind": "series",
                }
            },
        },
    )

    with pytest.raises(ValueError, match="does not match the dictionary schema"):
        load_dictionary(path)


def test_a_missing_dictionary_file_fails_at_load_with_its_path(tmp_path):
    """Verify a path that does not exist names itself, rather than raising later."""

    with pytest.raises(ValueError, match="no dictionary file at .*absent.yaml"):
        load_dictionary(tmp_path / "absent.yaml")


def test_use_dictionary_installs_an_override_that_load_default_dictionary_returns(
    tmp_path,
):
    """Verify an installed override is what load_default_dictionary then returns."""

    path = _written(
        tmp_path,
        {"schema_version": 1, "entries": {"proprio.joint_torque": _JOINT_TORQUE}},
    )
    override = load_dictionary(path)

    use_dictionary(override)

    assert load_default_dictionary() is override


def test_clear_active_dictionary_restores_the_packaged_default(tmp_path):
    """Verify clearing an installed override falls back to the packaged file."""

    packaged = load_default_dictionary()
    path = _written(
        tmp_path,
        {"schema_version": 1, "entries": {"proprio.joint_torque": _JOINT_TORQUE}},
    )
    use_dictionary(load_dictionary(path))

    clear_active_dictionary()

    assert load_default_dictionary() == packaged


# The packaged dictionary.yaml itself, loaded once for the tests below.
_PACKAGED = load_default_dictionary()

# The taxonomy keys docs/METRICS.md already commits to, matched against the
# packaged file so a rename or a dropped entry breaks a test rather than
# quietly stranding a documented metric requirement.
_DOCUMENTED_TYPES = {
    "proprio.joint_position",
    "proprio.joint_velocity",
    "proprio.joint_torque",
    "proprio.ee_pose",
    "proprio.ee_twist",
    "proprio.motor_current",
}


def test_the_packaged_dictionary_holds_every_reference_type():
    """Verify the shipped dictionary carries all 202 types from the source tables."""

    assert len(_PACKAGED.entries) == 202


def test_the_documented_metric_taxonomy_types_exist():
    """Verify every taxonomy type docs/METRICS.md names is a real packaged entry."""

    missing = _DOCUMENTED_TYPES - _PACKAGED.entries.keys()
    assert not missing, f"documented but missing from dictionary.yaml: {missing}"


def test_a_ros_message_type_resolves_to_its_taxonomy_type():
    """Verify each ROS 2 message-type alias the MCAP adapter relies on resolves.

    Every one of these names exactly one taxonomy entry deliberately: a bare
    `Image` or `Twist` cannot say wrist-from-exterior or command-from-echo, so
    only the unambiguous message types are aliased here.
    """

    expected = {
        "sensor_msgs/msg/JointState": "proprio.joint_position",
        "sensor_msgs/msg/Imu": "extero.imu",
        "nav_msgs/msg/Odometry": "derived.odometry",
        "sensor_msgs/msg/BatteryState": "proprio.battery_power",
        "geometry_msgs/msg/WrenchStamped": "proprio.ee_wrench",
    }

    for message_type, taxonomy_type in expected.items():
        assert _PACKAGED.resolve(message_type).taxonomy_type == taxonomy_type


def test_no_threshold_key_survived_outside_a_comment_line():
    """Verify no line outside a comment introduces a grading key.

    Defence in depth alongside `DictionaryEntry`'s `extra="forbid"`: if a
    future schema change ever widened what a field accepts, a bare `good:`
    or `weight:` copied in from the source tables would otherwise have
    nothing left to catch it.
    """

    raw = resources.files("kalanos.assets").joinpath("dictionary.yaml").read_text()
    body_lines = [
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    ]

    forbidden = ("good:", "bad:", "threshold:", "weight:", "good_threshold:")
    offenders = [line for line in body_lines if any(key in line for key in forbidden)]
    assert not offenders, f"grading keys found in dictionary.yaml: {offenders}"


def test_no_threshold_value_survived_inside_a_free_text_field():
    """Verify no `label` or alias carries a threshold's comparison operator.

    `extra="forbid"` only stops a threshold arriving as its own field; `label`
    and `aliases` are free text and would happily carry one pasted in from the
    source tables by mistake, e.g. `Good_Threshold`'s `"Jerk<0.1 m/s3"` or
    `"range 99.8%"`. Every source threshold uses `<`, `>` or `%` to say how
    good is good, and no legitimate label or alias needs any of them.
    """

    raw = resources.files("kalanos.assets").joinpath("dictionary.yaml").read_text()
    body_lines = [
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    ]

    offenders = [line for line in body_lines if any(char in line for char in "<>%")]
    assert not offenders, f"threshold-shaped text found in dictionary.yaml: {offenders}"


def test_the_alias_index_has_no_residual_collision():
    """Verify pruning actually removed every alias two types both claimed."""

    collisions = {
        stem: types for stem, types in _PACKAGED.alias_index.items() if len(types) > 1
    }
    assert not collisions, f"unresolved alias collisions: {collisions}"


def test_the_csv_fixtures_pose_axes_resolve_through_the_packaged_dictionary():
    """Verify the fixture corpus maps at least one channel to a real taxonomy type.

    `arm_multi_device.csv` carries `tcp_pose_x_mm` and its siblings precisely so
    this resolves — a corpus where every channel stays `unmapped` would prove
    nothing about a metric that requires a taxonomy type to run.
    """

    header = CSV_FIXTURE.read_text().splitlines()[0].split(",")
    pose_columns = [name for name in header if name.startswith("tcp_pose_")]

    assert pose_columns, "expected the fixture to still carry its tcp_pose_* columns"
    for column in pose_columns:
        match = _PACKAGED.resolve(column)
        assert match.taxonomy_type == "proprio.ee_pose", (column, match)
