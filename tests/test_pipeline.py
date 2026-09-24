"""Verifies pipeline.run around the adapter: how ids are minted, what a refusal does.

An episode's id comes back qualified by the file's place under the walked root,
so two same-named recordings stay apart.

A deliberate `AdapterRefusal` about the input becomes an `UnresolvedSource` and
the run continues; anything else — a `ValidationError` from our own models
included — is a bug in Kalanos and must propagate rather than being dressed
as a bad input file.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Iterator

# External
import polars as pl
import pytest
from pydantic import ValidationError
from upath import UPath

# Internal
import kalanos
from kalanos.analysis import pipeline
from kalanos.analysis.adapters.discover import (
    AdapterDiscovery,
    LoadedAdapter,
    PluginSource,
)
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.domain import (
    Channel,
    Clock,
    Episode,
    FramePayload,
    Kind,
    Stream,
)
from kalanos.analysis.models.policy import MetricPolicy, Policy, ScoreMode
from kalanos.analysis.pipeline import _with_declared_limits
from kalanos.assets.policy import load_default_policy

# Local
from helpers import CSV_FIXTURE, LEROBOT_FIXTURE


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _StubDirectoryAdapter:
    """A directory-level adapter that bids on a folder and never on a file.

    Parameters
    ----------
    nominal_rate_hz : float or None
        The rate `describe` declares.
    describe_raises : bool
        Whether `describe` raises instead of answering.
    """

    name = "stub_directory"

    def __init__(
        self, *, nominal_rate_hz: float | None = None, describe_raises: bool = False
    ) -> None:
        self._nominal_rate_hz = nominal_rate_hz
        self._describe_raises = describe_raises

    def detect(self, path: UPath) -> float:
        return 0.5 if path.is_dir() else 0.0

    def describe(self, path: UPath) -> DatasetInfo:
        if self._describe_raises:
            raise RuntimeError("describe() is broken")
        return DatasetInfo(
            adapter=self.name, path=path, nominal_rate_hz=self._nominal_rate_hz
        )

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        if sample is not None and sample <= 0:
            return
        yield Episode(
            id="stub_episode",
            streams=[
                Stream(
                    taxonomy_type="unmapped.stub",
                    kind=Kind.SERIES,
                    timestamps=pl.Series("time_s", [0.0, 1.0, 2.0, 3.0]),
                    payload=FramePayload(
                        frame=pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
                    ),
                    source_path=path,
                    clock=Clock.UNKNOWN,
                    is_regular=True,
                    channels=[Channel(name="v")],
                )
            ],
        )


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _discovery_of(adapter) -> AdapterDiscovery:
    """Build a single-adapter AdapterDiscovery for monkeypatching discover_adapters."""

    return AdapterDiscovery(
        adapters=[
            LoadedAdapter(
                name=adapter.name,
                adapter=adapter,
                source=PluginSource.ENTRY_POINT,
                origin="test",
            )
        ],
        failures=[],
    )


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_deliberate_refusal_becomes_an_unresolved_source_and_the_run_completes(
    tmp_path,
):
    """Verify a .csv with no resolvable time axis lands in Report.unresolved."""

    no_time = tmp_path / "no_time.csv"
    no_time.write_text("a,b\n1.0,2.0\n1.1,2.2\n1.2,2.3\n")

    report = pipeline.run(tmp_path, policy=load_default_policy())

    assert report.episodes == []
    [unresolved] = report.unresolved
    assert unresolved.path == no_time
    assert "time axis" in unresolved.reason


def test_a_validation_error_from_a_model_bug_propagates_rather_than_reads_unresolved(
    tmp_path, monkeypatch
):
    """Verify a ValidationError from a bug in Kalanos propagates out of pipeline.run.

    A fake adapter whose `episodes` raises `ValidationError` stands in for a
    model bug reached mid-parse — real fixture data never needs to be broken
    to provoke this, since the fault is in the adapter's own bookkeeping.
    """

    class _BrokenAdapter:
        name = "broken"

        def detect(self, path: UPath) -> float:
            return 1.0

        def describe(self, path: UPath) -> DatasetInfo:
            return DatasetInfo(adapter=self.name, path=path)

        def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
            Channel.model_validate({})
            return iter(())

    (tmp_path / "recording.csv").write_text("a,b\n1,2\n")

    monkeypatch.setattr(
        pipeline,
        "discover_adapters",
        lambda: AdapterDiscovery(
            adapters=[
                LoadedAdapter(
                    name="broken",
                    adapter=_BrokenAdapter(),
                    source=PluginSource.ENTRY_POINT,
                    origin="test",
                )
            ],
            failures=[],
        ),
    )

    with pytest.raises(ValidationError):
        pipeline.run(tmp_path, policy=load_default_policy())


def test_two_same_named_files_in_different_folders_get_distinguishable_ids(tmp_path):
    """Verify an episode's id names the file's place under the walked root."""

    body = CSV_FIXTURE.read_bytes()
    for folder in ("a", "b"):
        subfolder = tmp_path / folder
        subfolder.mkdir()
        (subfolder / "joints.csv").write_bytes(body)

    report = pipeline.run(UPath(tmp_path), policy=load_default_policy())

    assert {episode.id for episode in report.episodes} == {
        "a/joints.csv",
        "b/joints.csv",
    }


def test_a_directory_adapter_claims_its_whole_subtree(tmp_path, monkeypatch):
    """Verify a directory-level adapter's win over the root stops its files
    from ever being offered to selection individually.
    """

    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("x,y\n1,2\n")

    monkeypatch.setattr(
        pipeline, "discover_adapters", lambda: _discovery_of(_StubDirectoryAdapter())
    )

    report = pipeline.run(UPath(tmp_path), policy=load_default_policy())

    assert report.skipped == []
    assert len(report.episodes) == 1


def test_a_directory_no_adapter_claims_is_not_reported_as_skipped(tmp_path):
    """Verify an unclaimed folder is reported through its files alone, never itself."""

    (tmp_path / "unclaimed.bin").write_bytes(b"\x00\x01\x02")

    report = pipeline.run(UPath(tmp_path), policy=load_default_policy())

    assert [item.path.name for item in report.skipped] == ["unclaimed.bin"]


def test_the_declared_rate_reaches_the_graded_metric(tmp_path, monkeypatch):
    """Verify describe()'s nominal_rate_hz reaches the policy as a graded target."""

    monkeypatch.setattr(
        pipeline,
        "discover_adapters",
        lambda: _discovery_of(_StubDirectoryAdapter(nominal_rate_hz=100.0)),
    )

    report = pipeline.run(UPath(tmp_path), policy=load_default_policy())

    [episode] = report.episodes
    [stream] = episode.streams
    reason = stream.metrics["effective_hz"].evidence.get("ungraded_reason", "")
    assert "declared under limits" not in reason


def test_a_describe_that_raises_does_not_lose_the_dataset(tmp_path, monkeypatch):
    """Verify a broken describe() still lets the episodes underneath it grade."""

    monkeypatch.setattr(
        pipeline,
        "discover_adapters",
        lambda: _discovery_of(_StubDirectoryAdapter(describe_raises=True)),
    )

    report = pipeline.run(UPath(tmp_path), policy=load_default_policy())

    assert len(report.episodes) == 1


def test_with_declared_limits_reads_only_what_the_policys_own_target_source_names():
    """Verify the fill is entirely policy-driven: no metric name lives in pipeline.py.

    A metric with no `target_source` is left untouched; one whose
    `target_source` names an unset `DatasetInfo` attribute is skipped
    rather than filled with `None`; a policy file that already declared
    the limit keeps its own value.
    """

    policy = Policy(
        schema_version=1,
        metrics={
            "timing.rate_check": MetricPolicy(
                mode=ScoreMode.ABS_DEV,
                target="nominal_hz",
                target_source="nominal_rate_hz",
            ),
            "timing.unset_source": MetricPolicy(
                mode=ScoreMode.ABS_DEV,
                target="episode_count",
                target_source="episode_count",
            ),
            "timing.no_source": MetricPolicy(),
            "timing.already_declared": MetricPolicy(
                mode=ScoreMode.ABS_DEV,
                target="robot_hz",
                target_source="nominal_rate_hz",
                limits={"robot_hz": 999.0},
            ),
        },
        letters={"A": 90.0, "B": 80.0, "C": 70.0, "D": 60.0},
    )
    info = DatasetInfo(
        adapter="stub", path=UPath("."), nominal_rate_hz=30.0, episode_count=None
    )

    filled = _with_declared_limits(policy, info)

    assert filled.metrics["timing.rate_check"].limits == {"nominal_hz": 30.0}
    assert filled.metrics["timing.unset_source"].limits == {}
    assert filled.metrics["timing.no_source"].limits == {}
    assert filled.metrics["timing.already_declared"].limits == {"robot_hz": 999.0}


def test_the_report_forwards_what_the_adapter_declared():
    """Verify describe()'s declaration reaches the Report, one entry per path read."""

    report = kalanos.grade(LEROBOT_FIXTURE)

    [info] = report.datasets
    assert info.adapter == "lerobot_v3"
    assert info.episode_count == len(report.episodes)
    assert info.robot_type is not None
