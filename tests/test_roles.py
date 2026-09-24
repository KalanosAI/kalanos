"""Verifies roles: resolving column names to taxonomy types through the dictionary."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from kalanos.analysis.inference.roles import roles
from kalanos.analysis.models.dictionary import Dictionary, GroupHintKind


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The minimum an entry needs, for tests about something other than its content.
_BARE = {
    "label": "Anything",
    "category": "proprioceptive_state",
    "modality": "numeric",
    "kind": "series",
    "shape": "scalar",
}


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


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_unique_match_resolves_at_full_confidence():
    """Verify a column that matches exactly one entry carries confidence 1.0."""

    dictionary = _dictionary(proprio_joint_torque={**_BARE, "aliases": ["tau_J"]})

    result = roles(["tau_J"], dictionary)

    assert len(result) == 1
    assert result[0].column == "tau_J"
    assert result[0].taxonomy_type == "proprio.joint_torque"
    assert result[0].confidence == 1.0
    assert result[0].evidence


def test_an_alias_and_the_taxonomy_key_itself_both_resolve():
    """Verify a column answers to an entry's own name, not only its aliases."""

    dictionary = _dictionary(proprio_joint_position=_BARE)

    result = roles(["joint_position"], dictionary)

    assert result[0].taxonomy_type == "proprio.joint_position"


def test_axis_and_unit_survive_the_match_off_the_name():
    """Verify a column's axis and unit suffixes carry through onto its role.

    `arm_x_mm` still has to reach the entry aliased `arm`, and what the name
    claimed about its axis and unit stays reported alongside the match.
    """

    dictionary = _dictionary(proprio_ee_pose={**_BARE, "aliases": ["arm"]})

    result = roles(["arm_x_mm"], dictionary)

    assert result[0].taxonomy_type == "proprio.ee_pose"
    assert result[0].axis == "x"
    assert result[0].unit == "mm"


def test_an_ambiguous_name_resolves_to_nothing_but_reports_its_candidates():
    """Verify two entries claiming one name leave the column unresolved.

    Taking either would hand a physics-aware metric the wrong signal, so
    the role carries both candidates instead of guessing.
    """

    dictionary = _dictionary(
        proprio_joint_position={**_BARE, "aliases": ["pos"]},
        derived_tcp_position={**_BARE, "aliases": ["pos"]},
    )

    result = roles(["pos"], dictionary)

    assert result[0].taxonomy_type is None
    assert sorted(result[0].candidates) == [
        "derived.tcp_position",
        "proprio.joint_position",
    ]
    assert result[0].confidence == 0.0
    assert result[0].evidence


def test_an_unknown_name_resolves_to_nothing_with_no_candidates():
    """Verify a column the dictionary has never heard of comes back unmapped."""

    dictionary = _dictionary(proprio_joint_torque={**_BARE, "aliases": ["tau_J"]})

    result = roles(["widget_count"], dictionary)

    assert result[0].taxonomy_type is None
    assert result[0].candidates == []
    assert result[0].confidence == 0.0
    assert result[0].evidence


def test_axis_and_unit_survive_an_ambiguous_or_unknown_match_too():
    """Verify a name's axis and unit are reported even when nothing resolved.

    They come off `normalise_name`, which runs before the lookup — a column
    left as `unmapped.<name>` still needs its axis and unit for grouping,
    whether or not the dictionary could place it.
    """

    dictionary = _dictionary(
        proprio_joint_position={**_BARE, "aliases": ["pos"]},
        derived_tcp_position={**_BARE, "aliases": ["pos"]},
    )

    ambiguous, unknown = roles(["pos_x_mm", "mystery_y_mm"], dictionary)

    assert ambiguous.axis == "x"
    assert ambiguous.unit == "mm"
    assert unknown.axis == "y"
    assert unknown.unit == "mm"


def test_a_resolved_column_carries_its_entrys_group_hint():
    """Verify a match carries its entry's group_hint, and unmapped carries None."""

    dictionary = _dictionary(
        extero_imu={**_BARE, "aliases": ["gyro"], "group_hint": "axes"}
    )

    gyro, unmapped = roles(["gyro", "mystery"], dictionary)

    assert gyro.group_hint is GroupHintKind.AXES
    assert unmapped.group_hint is None


def test_every_column_gets_a_role_in_the_same_order():
    """Verify one role comes back per column, matched or not, in input order."""

    dictionary = _dictionary(proprio_joint_torque={**_BARE, "aliases": ["tau_J"]})

    result = roles(["tau_J", "mystery", "tau_J"], dictionary)

    assert [role.column for role in result] == ["tau_J", "mystery", "tau_J"]
    assert result[0].taxonomy_type == "proprio.joint_torque"
    assert result[1].taxonomy_type is None
