"""Verifies policy.yaml: the default policy's content,
and the loader's failure modes and packaging.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import ast
from pathlib import Path

# External
import pytest
import yaml

# Internal
# The bare package import runs every built-in metric module's @metric
# decorators, so the registry reflects the full built-in set below.
import kalanos.analysis.metrics  # noqa: F401
from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.models.policy import Band, MetricPolicy, Policy, ScoreMode
from kalanos.assets.policy import load_default_policy, load_policy


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


REPO_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = REPO_ROOT / "src" / "kalanos"
LOADER_SOURCE = PACKAGE_ROOT / "assets" / "policy.py"

# Reading a YAML file is a privilege the asset loaders hold and nothing else does:
# one loader per YAML shape, each owning its own schema.
ASSET_LOADERS = {
    LOADER_SOURCE,
    PACKAGE_ROOT / "assets" / "dictionary.py",
    PACKAGE_ROOT / "assets" / "mapping.py",
    PACKAGE_ROOT / "assets" / "bundle.py",
    PACKAGE_ROOT / "assets" / "yaml_strict.py",
}

# render.py writes a Report as YAML; it never reads a configuration file, so it
# is exempt from the import ban without being an asset loader itself.
YAML_WRITERS = {PACKAGE_ROOT / "analysis" / "reporting" / "render.py"}

# A minimal, valid letters table for building a Policy by hand.
_LETTERS = {
    "A": 90.0,
    "B": 80.0,
    "C": 70.0,
    "D": 60.0,
}

# The approved weights, keyed as policy.yaml keys them.
# fmt: off
DEFAULT_METRIC_WEIGHTS: dict[str, float | None] = {
    "timing.effective_hz":                 5,
    "timing.dt_jitter_ms":                 3,
    "timing.drop_rate":                    4,
    "timing.recorded_hz":                  1,
    "timing.recorded_dt_spread_ms":        1,
    "timing.recorded_drop_estimate":       2,
    "timing.monotonic_violations":         4,
    "integrity.missing_pct":               4,
    "integrity.flatline_pct":              4,
    "integrity.spike_pct":                 4,
    "integrity.drift":                     1,
    "integrity.snr_db":                    2,
    "integrity.dead_taxel_pct":            3,
    "integrity.hysteresis":                2,
    "motion.mean_jerk_norm":               1,
    "motion.max_abs_jerk":                 1,
    "motion.velocity_spike_pct":           4,
    "motion.log_dimensionless_jerk":       5,
    "motion.action_chatter":               2,
    "motion.still_drift":                  1,
    "motion.vel_saturation_pct":           2,
    "motion.limit_proximity_pct":          1,
    "motion.hf_vibration_ratio":           2,
    "motion.p99_torque":                   1,
    "motion.max_torque":                   None,
    "motion.mean_torque":                  None,
    "motion.energy_proxy":                 None,
    "vision.sharpness_score":              5,
    "vision.exposure_shift_pct":           3,
    "vision.exposure_level":               None,
    "vision.frame_count_vs_timebase":      4,
    "vision.pts_defect_pct":               2,
    "vision.frozen_frame_pct":             6,
    "vision.max_gap_s":                    None,
    "vision.camera_sync_skew_ms":          None,
    "annotation.task_instruction_missing": 5,
}
# fmt: on

DEFAULT_FAMILY_WEIGHTS = {
    "integrity": 3,
    "timing": 2,
    "motion": 2,
    "vision": 2,
    "annotation": 1,
}


# The metrics a family needs one of graded before it scores in an episode.
DEFAULT_CORE_METRICS = {
    "timing.effective_hz",
    "timing.drop_rate",
    "integrity.missing_pct",
    "integrity.flatline_pct",
    "integrity.spike_pct",
    "integrity.dead_taxel_pct",
    "integrity.hysteresis",
    "motion.hf_vibration_ratio",
    "vision.sharpness_score",
    "vision.exposure_shift_pct",
    "vision.frame_count_vs_timebase",
    "vision.frozen_frame_pct",
    "annotation.task_instruction_missing",
}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _entry(policy: Policy, metric_name: str) -> MetricPolicy:
    """Look up one metric's policy, failing loudly if the policy never named it.

    Parameters
    ----------
    policy : Policy
        The policy to search.
    metric_name : str
        The metric's bare name.

    Returns
    -------
    MetricPolicy
        The metric's policy.
    """

    found = policy.entry_for(metric_name)
    assert found is not None, f"policy carries no entry for {metric_name!r}"
    return found[1]


def _imports_yaml(source: Path) -> bool:
    """Check whether a Python file imports the `yaml` module at all.

    Parameters
    ----------
    source : Path
        Path to the Python file to parse.

    Returns
    -------
    bool
        Whether `source` contains an `import yaml` or `from yaml import ...`.
    """

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "yaml" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == "yaml":
            return True
    return False


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_default_policy_parses_into_a_policy():
    """Verify the packaged default.yaml validates against the Policy schema.

    Its bare metric names must equal every metric registered under the
    built-in `kalanos.analysis.metrics` package: a metric shipped without a
    policy entry, or an entry for a metric that no longer exists, fails
    here instead of the test merely restating default.yaml back at itself.
    """

    policy = load_default_policy()

    assert isinstance(policy, Policy)
    bare_names = {key.partition(".")[2] for key in policy.metrics}
    built_in_names = {
        entry.name
        for entry in registered_metrics()
        if entry.module.startswith("kalanos.analysis.metrics.")
    }
    assert bare_names == built_in_names


def test_the_default_policy_bands_match_the_documented_numbers():
    """Verify the shipped default bands match docs/METRICS.md's numbers.

    Every metric that carries a default band is asserted below. All are
    two-sided except `dt_jitter_ms`, whose bad bound is still to define —
    the one remaining one-sided band. The per-type `spike_pct` and `snr_db`
    bands for joint velocity and joint acceleration are checked too, since a
    type-keyed band is never merged with the default and could drift on its
    own.
    """

    policy = load_default_policy()

    band = _entry(policy, "drop_rate").thresholds["default"]
    assert band.good == pytest.approx(0.01)
    assert band.bad == pytest.approx(0.05)

    band = _entry(policy, "effective_hz").thresholds["default"]
    assert band.good == pytest.approx(0.10)
    assert band.bad == pytest.approx(0.25)

    band = _entry(policy, "missing_pct").thresholds["default"]
    assert band.good == pytest.approx(0.5)
    assert band.bad == pytest.approx(5.0)

    band = _entry(policy, "flatline_pct").thresholds["default"]
    assert band.good == pytest.approx(50.0)
    assert band.bad == pytest.approx(90.0)

    band = _entry(policy, "spike_pct").thresholds["default"]
    assert band.good == pytest.approx(0.1)
    assert band.bad == pytest.approx(2.0)

    band = _entry(policy, "snr_db").thresholds["default"]
    assert band.good == pytest.approx(30.0)
    assert band.bad == pytest.approx(15.0)

    band = _entry(policy, "dt_jitter_ms").thresholds["default"]
    assert band.good == pytest.approx(5.0)
    assert band.bad is None

    band = _entry(policy, "dead_taxel_pct").thresholds["default"]
    assert band.good == pytest.approx(2.0)
    assert band.bad == pytest.approx(10.0)

    band = _entry(policy, "hysteresis").thresholds["default"]
    assert band.good == pytest.approx(0.05)
    assert band.bad == pytest.approx(0.20)

    band = _entry(policy, "mean_jerk_norm").thresholds["default"]
    assert band.good == pytest.approx(0.1)
    assert band.bad == pytest.approx(0.5)

    band = _entry(policy, "max_abs_jerk").thresholds["default"]
    assert band.good == pytest.approx(0.1)
    assert band.bad == pytest.approx(0.5)

    band = _entry(policy, "hf_vibration_ratio").thresholds["default"]
    assert band.good == pytest.approx(0.1)
    assert band.bad == pytest.approx(0.3)

    band = _entry(policy, "spike_pct").thresholds["proprio.joint_velocity"]
    assert band.good == pytest.approx(0.1)
    assert band.bad == pytest.approx(2.0)

    band = _entry(policy, "snr_db").thresholds["proprio.joint_velocity"]
    assert band.good == pytest.approx(20.0)
    assert band.bad == pytest.approx(10.0)

    band = _entry(policy, "snr_db").thresholds["proprio.joint_acceleration"]
    assert band.good == pytest.approx(12.0)
    assert band.bad == pytest.approx(6.0)


def test_the_jerk_metrics_are_report_only_until_they_are_dimensionless():
    """Verify mean_jerk_norm and max_abs_jerk measure but don't grade.

    Normalising by each channel's spread leaves units of 1/s^3, so ordinary smooth
    motion lands far past the candidate band. They stay report-only, with the band
    kept on file, until the metric is made dimensionless (docs/METRICS.md).
    """

    policy = load_default_policy()

    for name in ("mean_jerk_norm", "max_abs_jerk"):
        assert _entry(policy, name).report_only is True, name


def test_velocity_spike_and_dimensionless_jerk_are_report_only_with_no_band():
    """Verify velocity_spike_pct and log_dimensionless_jerk measure but don't grade.

    No band has been set against real recordings for either yet.
    """

    policy = load_default_policy()

    for name in ("velocity_spike_pct", "log_dimensionless_jerk"):
        assert _entry(policy, name).report_only is True, name


def test_effective_hz_grades_relative_deviation_from_a_declared_nominal_rate():
    """Verify effective_hz uses abs_dev against a nominal_hz target, per the docs.

    `target_source` names `DatasetInfo.nominal_rate_hz`: the policy file
    wires an adapter's declared rate to this target, not any Python code.
    """

    policy = load_default_policy()

    effective_hz = _entry(policy, "effective_hz")
    assert effective_hz.mode == ScoreMode.ABS_DEV
    assert effective_hz.target == "nominal_hz"
    assert effective_hz.target_source == "nominal_rate_hz"
    assert effective_hz.limits == {}


def test_the_default_policy_weights_match_the_documented_numbers():
    """Verify every metric and family weight matches the approved values."""

    policy = load_default_policy()

    assert {key: entry.weight for key, entry in policy.metrics.items()} == (
        DEFAULT_METRIC_WEIGHTS
    )
    assert policy.family_weights == DEFAULT_FAMILY_WEIGHTS


def test_the_default_policy_marks_exactly_the_core_metrics():
    """Verify `core: true` sits on the documented core metrics and no others."""

    policy = load_default_policy()

    assert {key for key, entry in policy.metrics.items() if entry.core} == (
        DEFAULT_CORE_METRICS
    )


def test_core_parses_and_defaults_to_false():
    """Verify `core` reads from a metric entry and is off when absent."""

    assert MetricPolicy.model_validate({"core": True}).core is True
    assert MetricPolicy().core is False


def test_the_fail_penalty_stays_undecided():
    """Verify the fail penalty, *to define* in docs/METRICS.md, carries no number."""

    policy = load_default_policy()

    assert policy.fail_penalty is None
    assert policy.fail_penalty_cap is None


def test_the_default_policy_counts_low_episodes_below_70():
    """Verify the default cutoff for a family's low-episode count is 70."""

    assert load_default_policy().low_episode_score == 70


def test_an_extending_policy_can_set_the_low_episode_score(tmp_path):
    """Verify an override's `low_episode_score` replaces the base's."""

    override = tmp_path / "policy.yaml"
    override.write_text(yaml.safe_dump({"extends": "default", "low_episode_score": 80}))

    assert load_policy(override).low_episode_score == 80


def test_a_low_episode_score_above_100_fails_to_load(tmp_path):
    """Verify the cutoff stays on the 0-100 score scale."""

    override = tmp_path / "policy.yaml"
    override.write_text(
        yaml.safe_dump({"extends": "default", "low_episode_score": 101})
    )

    with pytest.raises(ValueError, match="does not match the policy schema"):
        load_policy(override)


def test_an_extending_policy_keeps_the_base_weight_of_a_metric_it_overrides(
    tmp_path,
):
    """Verify an override without `weight` keeps the base weight and its own bands."""

    override = tmp_path / "policy.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "extends": "default",
                "metrics": {
                    "integrity.spike_pct": {
                        "thresholds": {"default": {"good": 0.2, "bad": 3.0}},
                    }
                },
            }
        )
    )

    spike_pct = _entry(load_policy(override), "spike_pct")

    assert spike_pct.weight == 4
    assert spike_pct.thresholds == {"default": Band(good=0.2, bad=3.0)}


def test_an_extending_policy_can_set_a_weight_back_to_undecided(tmp_path):
    """Verify an explicit `weight: null` stays `None` instead of inheriting."""

    override = tmp_path / "policy.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "extends": "default",
                "metrics": {"integrity.spike_pct": {"weight": None}},
            }
        )
    )

    assert _entry(load_policy(override), "spike_pct").weight is None


def test_the_language_conditioned_policy_keeps_the_instruction_weight():
    """Verify language_conditioned keeps task_instruction_missing's weight."""

    policy = load_policy(Path("language_conditioned"))

    assert _entry(policy, "task_instruction_missing").weight == 5


def test_the_language_conditioned_policy_keeps_the_instruction_core_flag():
    """Verify language_conditioned keeps the instruction metric core via `extends`."""

    policy = load_policy(Path("language_conditioned"))

    assert _entry(policy, "task_instruction_missing").core is True


def test_an_extending_policy_can_clear_a_core_flag(tmp_path):
    """Verify an explicit `core: false` overrides the base's flag."""

    override = tmp_path / "policy.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "extends": "default",
                "metrics": {"integrity.spike_pct": {"core": False}},
            }
        )
    )

    assert _entry(load_policy(override), "spike_pct").core is False


def test_an_extending_policy_merges_family_weights_key_by_key(tmp_path):
    """Verify extending `family_weights` changes only the families it names."""

    override = tmp_path / "policy.yaml"
    override.write_text(
        yaml.safe_dump({"extends": "default", "family_weights": {"timing": 5}})
    )

    assert load_policy(override).family_weights == {
        **DEFAULT_FAMILY_WEIGHTS,
        "timing": 5,
    }


def test_a_metric_key_without_exactly_one_dot_fails_to_construct():
    """Verify a metrics key must split into exactly one family and one name."""

    with pytest.raises(ValueError, match="exactly 'family.metric'"):
        Policy(
            schema_version=1, metrics={"drop_rate": MetricPolicy()}, letters=_LETTERS
        )


def test_the_same_bare_metric_name_across_two_families_fails_to_construct():
    """Verify two families cannot both claim the same bare metric name.

    The registry keys a result by its bare function name alone, so a
    duplicate would leave `Policy.entry_for` unable to say which policy applies.
    """

    with pytest.raises(ValueError, match="unique across families"):
        Policy(
            schema_version=1,
            metrics={
                "timing.drop_rate": MetricPolicy(),
                "integrity.drop_rate": MetricPolicy(),
            },
            letters=_LETTERS,
        )


def test_abs_dev_without_a_target_fails_to_construct():
    """Verify mode: abs_dev cannot be declared without naming its target."""

    with pytest.raises(ValueError, match="abs_dev requires a target"):
        MetricPolicy(mode=ScoreMode.ABS_DEV)


def test_a_target_without_abs_dev_fails_to_construct():
    """Verify a target only means something once mode is abs_dev."""

    with pytest.raises(ValueError, match="only meaningful under mode: abs_dev"):
        MetricPolicy(mode=ScoreMode.DIRECT, target="nominal_hz")


def test_target_source_without_a_target_fails_to_construct():
    """Verify target_source cannot be declared without a target for it to fill."""

    with pytest.raises(ValueError, match="only meaningful alongside a target"):
        MetricPolicy(target_source="nominal_rate_hz")


def test_higher_is_better_alongside_abs_dev_fails_to_construct():
    """Verify a deviation-from-target metric cannot also declare a direction.

    A deviation from a target is always lower-is-better by construction.
    """

    with pytest.raises(ValueError, match="no meaning under mode: abs_dev"):
        MetricPolicy(mode=ScoreMode.ABS_DEV, target="nominal_hz", higher_is_better=True)


def test_a_band_with_equal_good_and_bad_bounds_fails_to_construct():
    """Verify a band cannot collapse to a single point.

    Interpolation needs a real span between good and bad.
    """

    with pytest.raises(ValueError, match="interpolation needs a real span"):
        MetricPolicy(thresholds={"default": Band(good=0.01, bad=0.01)})


def test_a_band_ordered_against_its_own_direction_fails_to_construct():
    """Verify a lower-is-better band cannot have good sit past bad."""

    with pytest.raises(ValueError, match="wrong way for higher_is_better=False"):
        MetricPolicy(thresholds={"default": Band(good=0.05, bad=0.01)})


def test_a_band_missing_one_bound_is_left_alone_by_the_ordering_check():
    """Verify a band with only good decided, like dt_jitter_ms, still constructs."""

    entry = MetricPolicy(thresholds={"default": Band(good=2.0)})
    assert entry.thresholds["default"].bad is None


def test_letters_missing_a_grade_fails_to_construct():
    """Verify the letters table must name exactly A, B, C and D."""

    with pytest.raises(ValueError, match="must carry exactly"):
        Policy(schema_version=1, metrics={}, letters={"A": 90.0, "B": 80.0, "C": 70.0})


def test_letters_out_of_order_fails_to_construct():
    """Verify the letters table must strictly descend A > B > C > D."""

    with pytest.raises(ValueError, match="must strictly descend"):
        Policy(
            schema_version=1,
            metrics={},
            letters={"A": 90.0, "B": 95.0, "C": 70.0, "D": 60.0},
        )


def test_entry_for_returns_none_for_a_metric_the_policy_never_mentions():
    """Verify a lookup miss is reported as None rather than raising."""

    policy = load_default_policy()

    assert policy.entry_for("not_a_real_metric") is None


def test_a_syntactically_invalid_policy_fails_at_load_not_at_first_use(tmp_path):
    """Verify broken YAML is rejected the moment it is loaded."""

    broken = tmp_path / "policy.yaml"
    broken.write_text("metrics: [this, is, not: a mapping")

    with pytest.raises(ValueError, match="not valid YAML"):
        load_policy(broken)


def test_a_schema_invalid_policy_fails_at_load_not_at_first_use(tmp_path):
    """Verify YAML that parses but violates the schema is still rejected up front.

    `letters` is missing entirely here —
    a well-formed file that simply forgot a required field,
    the kind of mistake a hand edit produces.
    """

    invalid = tmp_path / "policy.yaml"
    invalid.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "metrics": {"timing.drop_rate": {}},
            }
        )
    )

    with pytest.raises(ValueError, match="does not match the policy schema"):
        load_policy(invalid)


def test_a_policy_override_path_is_read_over_the_packaged_default(tmp_path):
    """Verify a user-supplied policy.yaml, not the packaged one, is what loads."""

    override = tmp_path / "policy.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "schema_version": 7,
                "metrics": {
                    "timing.drop_rate": {
                        "thresholds": {"default": {"good": 0.02, "bad": 0.1}},
                    }
                },
                "letters": _LETTERS,
            }
        )
    )

    policy = load_policy(override)

    assert policy.schema_version == 7
    assert _entry(policy, "drop_rate").thresholds["default"] == Band(good=0.02, bad=0.1)


def test_with_limit_fills_only_metrics_that_target_it():
    """Verify with_limit fills only abs_dev metrics that target the given name."""

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.effective_hz": MetricPolicy(
                mode=ScoreMode.ABS_DEV, target="nominal_hz"
            ),
            "timing.drop_rate": MetricPolicy(mode=ScoreMode.DIRECT),
        },
        letters=_LETTERS,
    )

    filled = policy.with_limit("nominal_hz", 30.0)

    assert _entry(filled, "effective_hz").limits == {"nominal_hz": 30.0}
    assert _entry(filled, "drop_rate").limits == {}


def test_with_limit_leaves_a_declared_limit_alone():
    """Verify a limit the policy already declares is not overwritten."""

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.effective_hz": MetricPolicy(
                mode=ScoreMode.ABS_DEV,
                target="nominal_hz",
                limits={"nominal_hz": 100.0},
            ),
        },
        letters=_LETTERS,
    )

    filled = policy.with_limit("nominal_hz", 30.0)

    assert _entry(filled, "effective_hz").limits == {"nominal_hz": 100.0}


def test_no_other_module_imports_yaml_directly():
    """Verify only the asset loaders and the YAML renderer touch `yaml` directly.

    `docs/ARCHITECTURE.md` states this as a boundary: a module could only break
    it by importing `yaml` directly, so that's what this checks for.
    """

    offenders = [
        path.relative_to(PACKAGE_ROOT)
        for path in PACKAGE_ROOT.rglob("*.py")
        if path not in ASSET_LOADERS | YAML_WRITERS and _imports_yaml(path)
    ]

    assert not offenders, f"modules importing yaml outside the loaders: {offenders}"


def test_no_asset_loader_walks_file_relative_to_itself():
    """Verify every asset loader resolves its file through importlib.resources.

    A `Path(__file__)` walk up the source tree is the packaging bug
    docs/ARCHITECTURE.md warns about: it works from a checkout and silently
    breaks from an installed wheel. Parsing each loader's source directly
    catches a regression back to that pattern.
    """

    offenders = []
    for loader in ASSET_LOADERS:
        tree = ast.parse(loader.read_text(encoding="utf-8"), filename=str(loader))
        if any(
            isinstance(node, ast.Name) and node.id == "__file__"
            for node in ast.walk(tree)
        ):
            offenders.append(loader.relative_to(PACKAGE_ROOT))

    assert not offenders, f"asset loaders walking __file__: {offenders}"


@pytest.mark.parametrize(
    "taxonomy_type",
    [
        "action.action_vector",
        "action.joint_position_command",
        "action.gripper_command",
        "reward.frame_reward",
    ],
)
def test_commands_and_rewards_are_exempt_from_flatline(taxonomy_type):
    """A command holds still legitimately, like a reward; neither is a sensor."""

    band = _entry(load_default_policy(), "flatline_pct").thresholds[taxonomy_type]

    assert band.good is None
    assert band.bad is None


def test_sensors_still_grade_flatline():
    """A stuck sensor is what the metric is for: measured signals keep the band."""

    thresholds = _entry(load_default_policy(), "flatline_pct").thresholds

    assert "proprio.joint_position" not in thresholds
    assert thresholds["default"].bad == pytest.approx(90.0)


def test_torque_noise_is_reported_not_graded():
    """Motor-current channels are noisy by nature; real ALOHA arms read 8-15 dB.

    Joint position noise still grades: the exemption is for torque alone.
    """

    thresholds = _entry(load_default_policy(), "snr_db").thresholds

    assert thresholds["proprio.joint_torque"].good is None
    assert thresholds["proprio.joint_torque"].bad is None
    assert thresholds["default"].bad == pytest.approx(15.0)
