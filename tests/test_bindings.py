"""Verifies channel bindings: semantic identity, validation evidence and precedence."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import shutil
from pathlib import Path

# External
import polars as pl
import pytest
from pydantic import ValidationError
from upath import UPath

# Internal
from kalanos.analysis.adapters import video
from kalanos.analysis.adapters.lerobot.common import (
    resolve_taxonomy,
    taxonomy_and_channels,
)
from kalanos.analysis.bindings import (
    binding_identity,
    resolve_stream,
    typed_views,
)
from kalanos.analysis.metrics.registry import _unmet_node_reason
from kalanos.analysis.models.binding import (
    ActuatorKind,
    BindingOrigin,
    BindingSection,
    Bundle,
    ChannelBinding,
    CommandSemantics,
    FeatureAssertion,
    Quantity,
    Representation,
    RequirementsSection,
    SamePriorityConflict,
    Validation,
    ValidationStatus,
    resolve_feature_types,
)
from kalanos.analysis.models.domain import (
    Channel,
    Episode,
    FramePayload,
    Kind,
    Stream,
    TimestampDtype,
)
from kalanos.analysis.models.errors import MappingOverrideError
from kalanos.analysis.models.mapping import MappingOverride, OverrideOrigin
from kalanos.analysis.models.metrics import (
    ChannelContext,
    MetricStatus,
    Requires,
    StreamContext,
)
from kalanos.analysis.models.provenance import ExecutionTier
from kalanos.analysis.reporting.assemble import grade_episode
from kalanos.analysis.scoring.cameras import camera_key
from kalanos.api import grade
from kalanos.assets.dictionary import load_default_dictionary
from kalanos.assets.policy import load_default_policy
from kalanos.benchmark import benchmark_dataset


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

FIXTURE = Path(__file__).parent / "fixtures/lerobot_v3_tiny"
SOURCE = "acquisition-session"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def signal(names=("p", "effort", "command"), taxonomy="unmapped.mixed"):
    frame = pl.DataFrame(
        {n: [float(i + j) for i in range(100)] for j, n in enumerate(names)}
    )
    return Stream(
        taxonomy_type=taxonomy,
        source_field="mixed",
        source_path=UPath("source.parquet"),
        timestamps=pl.Series("t", [i / 100 for i in range(100)]),
        timestamp_dtype=TimestampDtype.FLOAT64,
        kind=Kind.SERIES,
        is_regular=True,
        payload=FramePayload(frame=frame),
        channels=[Channel(name=n, source_index=i) for i, n in enumerate(names)],
    )


def bind(stream, assertions=(), override=None):
    return resolve_stream(
        stream,
        dictionary=load_default_dictionary(),
        channels=assertions,
        override=override,
        source_identity=SOURCE,
    )


def assertions():
    return [
        ChannelBinding(
            feature="mixed",
            index=0,
            taxonomy_type="proprio.joint_position",
            quantity=Quantity.POSITION,
            unit="rad",
            device="left",
            representation=Representation.CONTINUOUS,
            command=CommandSemantics.NONE,
        ),
        ChannelBinding(
            feature="mixed",
            index=1,
            taxonomy_type="proprio.joint_torque",
            actuator=ActuatorKind.GRIPPER,
            quantity=Quantity.EFFORT,
            unit="Nm",
            device="right",
        ),
        ChannelBinding(
            feature="mixed",
            index=2,
            taxonomy_type="action.gripper_command",
            quantity=Quantity.BINARY,
            representation=Representation.DISCRETE,
            device="right",
        ),
    ]


def _vision_reasons(episode) -> list[str]:
    """The ids of an episode's eligibility reasons that concern its cameras."""

    return [
        r.id
        for r in episode.eligibility.reasons
        if ".vision." in r.id
        or r.id in ("capability:video_quality", "capability:sampled_video_quality")
    ]


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_mixed_views_preserve_source_indices_values_and_original_stream():
    original = signal()
    bound = bind(original, assertions())
    views = typed_views(bound)
    assert isinstance(original.payload, FramePayload)
    assert [s.taxonomy_type for s in views] == [a.taxonomy_type for a in assertions()]
    assert [s.channels[0].source_index for s in views] == [0, 1, 2]
    assert [s.instance for s in views] == ["left", "right", "right"]
    for view in views:
        channel = view.channels[0]
        assert isinstance(view.payload, FramePayload)
        assert view.payload.frame[channel.name].equals(
            original.payload.frame[channel.name]
        )
        assert view.timestamps.equals(original.timestamps)
        assert view.source_field == "mixed"
    assert len(original.channels) == 3
    assert all(c.binding is None for c in original.channels)


def test_whole_feature_override_cannot_erase_explicit_effort_or_discrete_channels():
    override = MappingOverride(
        feature="mixed",
        taxonomy_type="proprio.joint_position",
        origin=OverrideOrigin.ARGUMENT,
    )
    bound = bind(signal(), assertions(), override)
    first = bound.channels[1].binding
    second = bound.channels[2].binding
    assert first is not None
    assert second is not None
    assert first.quantity == Quantity.EFFORT
    assert first.taxonomy_type == "proprio.joint_torque"
    assert second.representation == Representation.DISCRETE
    assert any(c.displaced == "proprio.joint_position" for c in first.conflicts)


def test_gripper_effort_is_not_retyped_by_its_name():
    bound = bind(signal(("left_gripper",), "proprio.joint_torque"))
    b = bound.channels[0].binding
    assert b is not None
    assert b.actuator == ActuatorKind.GRIPPER
    assert b.quantity == Quantity.EFFORT
    assert b.taxonomy_type == "proprio.joint_torque"
    assert b.origin == BindingOrigin.INFERRED


def test_partly_named_vector_preserves_unknowns_under_feature_override():
    spec = {"shape": [2], "names": ["robot0_joint_pos", "mystery_dimension"]}
    dictionary = load_default_dictionary()
    assert resolve_taxonomy("custom_vector", spec, dictionary)[0].startswith(
        "unmapped."
    )
    taxonomy, _, channels = taxonomy_and_channels("custom_vector", spec, dictionary)
    stream = signal(tuple(c.name for c in channels), taxonomy).model_copy(
        update={"channels": channels, "source_field": "custom_vector"}
    )
    bound = bind(stream)
    first = bound.channels[0].binding
    second = bound.channels[1].binding
    assert first is not None
    assert second is not None
    assert first.taxonomy_type == "proprio.joint_position"
    assert second.taxonomy_type.startswith("unmapped.")
    assert bound.channels[1].source_index == 1
    override = MappingOverride(
        feature="custom_vector",
        taxonomy_type="proprio.joint_position",
        origin=OverrideOrigin.ARGUMENT,
    )
    overridden = bind(stream, override=override)
    overridden_second = overridden.channels[1].binding
    assert overridden_second is not None
    assert overridden_second.taxonomy_type.startswith("unmapped.")
    assert overridden_second.conflicts


def test_inferred_name_and_unit_assertion_do_not_mean_validated_physics():
    bound = bind(signal(), assertions())
    for channel in bound.channels:
        b = channel.binding
        assert b is not None
        assert b.capabilities["numeric"].ready
        assert not b.capabilities["derivatives"].ready
        assert not b.capabilities["limits"].ready
        assert not b.capabilities["noise"].ready
        assert b.status != ValidationStatus.VALIDATED


def evidence(property, value, scope=SOURCE, capability=None):
    return Validation(
        property=property,
        value=value,
        validator="acquisition-owner",
        evidence="session-layout-v1",
        scope=scope,
        capability=capability,
    )


def validated_position():
    a = assertions()[0]
    return a.model_copy(
        update={
            "validations": [
                evidence("identity", "mixed[0]"),
                evidence("quantity", "position"),
                evidence("unit", "rad"),
            ]
        }
    )


def test_validation_is_per_property_and_requires_matching_value_and_scope():
    a = validated_position()
    a.validations += [evidence("limits", [-1, 1], "other-session")]
    b = bind(signal(), [a]).channels[0].binding
    assert b is not None
    assert b.capabilities["derivatives"].ready
    assert not b.capabilities["limits"].ready
    assert b.property_status["unit"] == ValidationStatus.VALIDATED
    assert b.status == ValidationStatus.ASSERTED
    assert len(b.invalidated_validations) == 1


def test_capability_scoped_validation_does_not_leak_to_another_capability():
    a = validated_position()
    a.validations = [
        v.model_copy(update={"capability": "derivatives"}) for v in a.validations
    ]
    a.limits = (-1, 1)
    a.validations.append(evidence("limits", [-1, 1], capability="limits"))
    b = bind(signal(), [a]).channels[0].binding
    assert b is not None
    assert b.capabilities["derivatives"].ready
    assert not b.capabilities["limits"].ready


def test_different_source_cannot_reuse_validation():
    a = validated_position()
    b = (
        resolve_stream(
            signal(),
            dictionary=load_default_dictionary(),
            channels=[a],
            source_identity="other-session",
        )
        .channels[0]
        .binding
    )
    assert b is not None
    assert not b.capabilities["derivatives"].ready
    assert len(b.invalidated_validations) == 3


def test_overriding_validated_unit_invalidates_previous_evidence():
    original = bind(signal(), [validated_position()])
    replacement = ChannelBinding(
        feature="mixed", index=0, taxonomy_type="proprio.joint_position", unit="deg"
    )
    b = bind(original, [replacement]).channels[0].binding
    assert b is not None
    assert b.unit == "deg"
    assert b.property_status["unit"] == ValidationStatus.ASSERTED
    assert not b.capabilities["derivatives"].ready
    assert any(
        v.property == "unit" and v.value == "rad" for v in b.invalidated_validations
    )


def test_new_evidence_can_revalidate_an_overridden_unit():
    original = bind(signal(), [validated_position()])
    replacement = validated_position().model_copy(
        update={
            "unit": "deg",
            "validations": [
                evidence("identity", "mixed[0]"),
                evidence("quantity", "position"),
                evidence("unit", "deg"),
            ],
        }
    )
    b = bind(original, [replacement]).channels[0].binding
    assert b is not None
    assert b.capabilities["derivatives"].ready


def test_generic_metrics_need_no_physical_units_but_capability_gate_abstains():
    stream = bind(signal(), assertions())
    assert isinstance(stream.payload, FramePayload)
    ctx = ChannelContext(
        channel=stream.channels[0],
        values=stream.payload.frame["p"],
        stream=StreamContext(stream=stream, is_regular=True),
    )
    assert _unmet_node_reason(Requires(capabilities=["numeric"]), ctx) is None
    derivatives_reason = _unmet_node_reason(Requires(capabilities=["derivatives"]), ctx)
    assert derivatives_reason is not None and "derivatives" in derivatives_reason
    assert ctx.taxonomy_type == "proprio.joint_position"


def test_display_rename_after_binding_preserves_measurements_and_identity():
    original = bind(signal(), assertions())
    assert isinstance(original.payload, FramePayload)
    renamed = original.model_copy(
        update={
            "channels": [
                c.model_copy(update={"name": f"display_{i}"})
                for i, c in enumerate(original.channels)
            ],
            "payload": FramePayload(
                frame=original.payload.frame.rename(
                    {c.name: f"display_{i}" for i, c in enumerate(original.channels)}
                )
            ),
        }
    )

    def episode(s):
        return Episode(id="e", streams=typed_views(s))

    a, b = episode(original), episode(renamed)
    assert binding_identity([a], None) == binding_identity([b], None)
    ga, _ = grade_episode(
        a,
        adapter="test",
        adapter_confidence=1,
        policy=load_default_policy(),
        dictionary=load_default_dictionary(),
    )
    gb, _ = grade_episode(
        b,
        adapter="test",
        adapter_confidence=1,
        policy=load_default_policy(),
        dictionary=load_default_dictionary(),
    )
    assert ga.score == gb.score
    assert [c.metrics for s in ga.streams for c in s.channels] == [
        c.metrics for s in gb.streams for c in s.channels
    ]


def test_declared_permutation_with_updated_bindings_preserves_semantics():
    original = signal()
    assert isinstance(original.payload, FramePayload)
    permutation = [2, 0, 1]
    names = [original.channels[i].name for i in permutation]
    moved = original.model_copy(
        update={
            "channels": [Channel(name=n, source_index=i) for i, n in enumerate(names)],
            "payload": FramePayload(frame=original.payload.frame.select(names)),
        }
    )
    updated = []
    for a in assertions():
        assert a.index is not None
        updated.append(a.model_copy(update={"index": permutation.index(a.index)}))
    a, b = bind(original, assertions()), bind(moved, updated)

    def by_name(s):
        return {
            c.name: (c.binding.quantity, c.binding.unit, c.binding.device)
            for c in s.channels
        }

    assert by_name(a) == by_name(b)
    assert [c.source_index for c in b.channels] == [0, 1, 2]


def test_conflicting_permutation_with_name_evidence_is_rejected():
    a = assertions()[0].model_copy(update={"name": "p"})
    with pytest.raises(MappingOverrideError, match="name mismatch"):
        bind(signal(("effort", "p", "command")), [a])


def test_duplicate_channel_selectors_are_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        BindingSection(id="bad", channels=[assertions()[0], assertions()[0]])


@pytest.mark.parametrize("index", [-1, 999])
def test_bad_indices_never_silently_succeed(index):
    if index < 0:
        with pytest.raises(ValidationError):
            ChannelBinding(
                feature="observation.state",
                index=index,
                taxonomy_type="proprio.joint_position",
            )
    else:
        bundle = Bundle(
            binding=BindingSection(
                id="bad",
                channels=[
                    ChannelBinding(
                        feature="observation.state",
                        index=index,
                        taxonomy_type="proprio.joint_position",
                    )
                ],
            )
        )
        with pytest.raises(MappingOverrideError, match="matched no input"):
            grade(FIXTURE, bundle=bundle)


@pytest.mark.parametrize("bad", [{"validator": " "}, {"evidence": ""}])
def test_validation_evidence_cannot_be_blank(bad):
    data = evidence("unit", "rad").model_dump()
    data.update(bad)
    with pytest.raises(ValidationError):
        Validation(**data)


@pytest.mark.parametrize("tier", [ExecutionTier.STANDARD, ExecutionTier.METADATA])
def test_grade_and_benchmark_share_resolved_identities_and_tiers(tier):
    bundle = Bundle(
        binding=BindingSection(
            id="fixture",
            channels=[
                ChannelBinding(
                    feature="observation.state",
                    index=0,
                    taxonomy_type="proprio.joint_position",
                    quantity=Quantity.POSITION,
                )
            ],
        )
    )
    report = grade(FIXTURE, bundle=bundle, tier=tier)
    bench = benchmark_dataset(str(FIXTURE), bundle=bundle, sample=1, tier=tier)
    assert report.run is not None
    assert bench.configuration["binding"] == report.run.binding
    for key in ("requirements", "policy", "dictionary", "execution", "bundle"):
        assert bench.configuration[key] == getattr(report.run, key)
    assert bench.scope == report.scope
    if tier == ExecutionTier.METADATA:
        assert bench.n_sampled == 0
        assert report.eligibility_counts is not None
        assert report.eligibility_counts.unknown == len(report.episodes)
        declared = [c for s in report.episodes[0].streams for c in s.declared_channels]
        assert declared and all(c.binding is not None for c in declared)


def test_sidecar_is_permanent_read_only_and_explicit_flags_win(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(FIXTURE, root)
    sidecar = root / "kalanos-map.yaml"
    content = (
        "schema_version: 1\nfeatures:\n  observation.velocity: proprio.joint_velocity\n"
    )
    sidecar.write_text(content)
    bundle = Bundle(
        binding=BindingSection(
            id="b", features={"observation.velocity": "proprio.joint_torque"}
        )
    )
    mapping = {"observation.velocity": "proprio.joint_position"}
    report = grade(root, bundle=bundle, mapping=mapping)
    bench = benchmark_dataset(str(root), bundle=bundle, mapping=mapping, sample=0)
    assert report.run is not None
    assert report.run.binding == bench.configuration["binding"]
    assert report.binding_conflicts[0].winner.origin == BindingOrigin.ARGUMENT
    assert sidecar.read_text() == content
    assert len(list(root.glob("*.yaml"))) == 1


def test_report_json_round_trip_retains_resolved_bindings():
    from kalanos.analysis.models.report import Report

    report = grade(FIXTURE)
    reread = Report.model_validate_json(report.model_dump_json())
    assert reread.run is not None and report.run is not None
    assert reread.run.binding == report.run.binding
    assert reread.episodes[0].streams == report.episodes[0].streams


def test_bundle_channel_cannot_spoof_validated_status():
    a = assertions()[0].model_copy(
        update={"origin": BindingOrigin.DECLARED, "status": ValidationStatus.VALIDATED}
    )
    b = bind(signal(), [a]).channels[0].binding
    assert b is not None
    assert b.origin == BindingOrigin.BUNDLE
    assert b.status == ValidationStatus.ASSERTED
    assert not b.capabilities["derivatives"].ready


def test_same_feature_in_two_sources_respects_scoped_selector():
    a = assertions()[0].model_copy(update={"source_identity": SOURCE})
    first = bind(signal(), [a]).channels[0].binding
    second = (
        resolve_stream(
            signal(),
            dictionary=load_default_dictionary(),
            channels=[a],
            source_identity="other-session",
        )
        .channels[0]
        .binding
    )
    assert first is not None and second is not None
    assert first.taxonomy_type == "proprio.joint_position"
    assert second.taxonomy_type.startswith("unmapped.")


def test_global_and_source_scoped_selectors_cannot_overlap_silently():
    a = assertions()[0]
    b = a.model_copy(update={"source_identity": SOURCE})
    with pytest.raises(MappingOverrideError, match="overlapping"):
        bind(signal(), [a, b])


def test_unknown_taxonomy_in_channel_binding_is_configuration_error():
    bundle = Bundle(
        binding=BindingSection(
            id="bad",
            channels=[
                ChannelBinding(
                    feature="observation.state", index=0, taxonomy_type="not.a.type"
                )
            ],
        )
    )
    with pytest.raises(MappingOverrideError):
        grade(FIXTURE, bundle=bundle)


@pytest.mark.parametrize(
    "limits", [(1, 1), (2, 1), (float("-inf"), 1), (0, float("inf"))]
)
def test_declared_physical_limits_must_be_finite_and_ordered(limits):
    with pytest.raises(ValidationError):
        ChannelBinding(
            feature="mixed",
            index=0,
            taxonomy_type="proprio.joint_position",
            limits=limits,
        )


def test_registered_jerk_abstains_without_evidence_and_runs_with_it():
    from kalanos.analysis.metrics.registry import run_stream_metrics

    unvalidated = typed_views(bind(signal(), assertions()))[0]
    validated = typed_views(bind(signal(), [validated_position()]))[0]
    no = run_stream_metrics(StreamContext(stream=unvalidated, is_regular=True))[
        "mean_jerk_norm"
    ]
    yes = run_stream_metrics(StreamContext(stream=validated, is_regular=True))[
        "mean_jerk_norm"
    ]
    assert no.value is None and "derivatives" in no.evidence["reason"]
    assert yes.value is not None


def test_contradictory_quantity_and_taxonomy_are_rejected():
    with pytest.raises(ValidationError, match="quantity contradicts"):
        ChannelBinding(
            feature="mixed",
            index=0,
            taxonomy_type="proprio.gripper_width",
            quantity=Quantity.EFFORT,
        )


def test_taxonomy_override_cannot_leave_a_contradictory_inherited_quantity():
    original = bind(signal(), assertions())
    changed = ChannelBinding(
        feature="mixed", index=1, taxonomy_type="proprio.joint_position"
    )
    with pytest.raises(MappingOverrideError, match="inconsistent binding"):
        bind(original, [changed])


def test_feature_override_preserves_inferred_actuator_kind():
    override = MappingOverride(
        feature="mixed",
        taxonomy_type="proprio.joint_torque",
        origin=OverrideOrigin.ARGUMENT,
    )
    b = (
        bind(signal(("left_gripper",), "action.action_vector"), override=override)
        .channels[0]
        .binding
    )
    assert b is not None
    assert b.actuator == ActuatorKind.GRIPPER
    assert b.quantity == Quantity.EFFORT


def test_binding_precedence_is_argument_file_bundle_sidecar_and_conflicts_are_kept():
    resolved = resolve_feature_types(
        [
            FeatureAssertion(
                feature="f", taxonomy_type="sidecar", origin=BindingOrigin.SIDECAR
            ),
            FeatureAssertion(
                feature="f", taxonomy_type="bundle", origin=BindingOrigin.BUNDLE
            ),
            FeatureAssertion(
                feature="f", taxonomy_type="file", origin=BindingOrigin.FILE
            ),
            FeatureAssertion(
                feature="f", taxonomy_type="arg", origin=BindingOrigin.ARGUMENT
            ),
            FeatureAssertion(
                feature="g", taxonomy_type="same", origin=BindingOrigin.BUNDLE
            ),
            FeatureAssertion(
                feature="g", taxonomy_type="same", origin=BindingOrigin.SIDECAR
            ),
        ]
    )
    assert resolved.types["f"].taxonomy_type == "arg"
    [conflict] = resolved.conflicts
    assert conflict.feature == "f"
    assert [d.origin for d in conflict.displaced] == [
        BindingOrigin.FILE,
        BindingOrigin.BUNDLE,
        BindingOrigin.SIDECAR,
    ]
    # Agreement is not a conflict.
    assert resolved.types["g"].taxonomy_type == "same"


def test_two_assertions_at_one_priority_that_disagree_are_a_configuration_error():
    with pytest.raises(SamePriorityConflict):
        resolve_feature_types(
            [
                FeatureAssertion(
                    feature="f", taxonomy_type="a", origin=BindingOrigin.FILE
                ),
                FeatureAssertion(
                    feature="f", taxonomy_type="b", origin=BindingOrigin.FILE
                ),
            ]
        )


def test_the_default_scope_shows_vision_results_without_grading_them():
    """numeric-core never lets a camera decide or score an episode."""

    report = grade(FIXTURE)

    for episode in report.episodes:
        assert _vision_reasons(episode) == []
        assert "vision" not in episode.score.families
        camera = next(s for s in episode.streams if "images" in s.taxonomy_type)
        assert camera.metrics["sharpness_score"].status.value == "report_only"


def test_each_camera_is_summarised_and_too_few_episodes_are_not_compared():
    """The fixture's 2 episodes are too few to compare its camera across them."""

    report = grade(FIXTURE)

    cameras = {
        camera_key(s) for e in report.episodes for s in e.streams if s.kind == "video"
    }
    assert sorted(c.camera for c in report.cameras) == sorted(cameras)
    assert all(not c.compared and c.n_episodes == 2 for c in report.cameras)


def test_a_sample_cannot_satisfy_video_quality():
    """Under vision-imitation, a sampled run leaves video_quality unknown."""

    bundle = Bundle(requirements=RequirementsSection(id="vision-imitation-v1"))
    # The fixture's episodes hold 8 and 6 frames; the default 10 would read them all.
    report = grade(FIXTURE, bundle=bundle, vision_samples=2)

    assert report.eligibility_counts is not None
    assert report.eligibility_counts.pass_count == 0
    for episode in report.episodes:
        assert "capability:video_quality" in _vision_reasons(episode)


def test_the_vision_preset_cannot_pass_without_a_graded_camera(monkeypatch):
    """With no frame decodable, video quality was not evaluated: unknown, not a pass."""

    monkeypatch.setattr(video, "av", None)
    bundle = Bundle(requirements=RequirementsSection(id="vision-imitation-v1"))
    report = grade(FIXTURE, bundle=bundle)

    assert all(
        "capability:video_quality" in _vision_reasons(e) for e in report.episodes
    )
    assert report.eligibility_counts is not None
    assert report.eligibility_counts.pass_count == 0


def test_a_full_scan_satisfies_video_quality_and_vision_requests_review():
    """Reading every frame satisfies video_quality; critical vision asks for review."""

    bundle = Bundle(requirements=RequirementsSection(id="vision-imitation-v1"))
    report = grade(FIXTURE, bundle=bundle, full_frame_scan=True)

    assert report.eligibility_counts is not None
    assert report.eligibility_counts.blocked == 0
    assert report.eligibility_counts.review == len(report.episodes)
    for episode in report.episodes:
        reasons = _vision_reasons(episode)
        assert "capability:video_quality" not in reasons
        assert any(r.endswith(".vision.sharpness_score") for r in reasons)


def test_a_sample_covering_every_frame_satisfies_video_quality():
    """A sample larger than the episode reads every frame, which is a full read."""

    bundle = Bundle(requirements=RequirementsSection(id="vision-imitation-v1"))
    report = grade(FIXTURE, bundle=bundle, vision_samples=100)

    for episode in report.episodes:
        assert "capability:video_quality" not in _vision_reasons(episode)


def test_a_completed_sample_satisfies_sampled_video_quality_and_grades_vision():
    """sampled_video_quality needs only a sample, and lets the vision family grade."""

    bundle = Bundle(
        requirements=RequirementsSection(
            required_capabilities=["sampled_video_quality"]
        )
    )
    report = grade(FIXTURE, bundle=bundle, vision_samples=2)

    for episode in report.episodes:
        assert "capability:sampled_video_quality" not in _vision_reasons(episode)
        camera = next(s for s in episode.streams if s.kind == "video")
        assert camera.metrics["sharpness_score"].status != MetricStatus.REPORT_ONLY


def test_sampled_video_quality_without_the_decoder_is_unknown(monkeypatch):
    """With no frame decodable, not even a sample was taken."""

    monkeypatch.setattr(video, "av", None)
    bundle = Bundle(
        requirements=RequirementsSection(
            required_capabilities=["sampled_video_quality"]
        )
    )
    report = grade(FIXTURE, bundle=bundle)

    assert all(
        "capability:sampled_video_quality" in _vision_reasons(e)
        for e in report.episodes
    )
