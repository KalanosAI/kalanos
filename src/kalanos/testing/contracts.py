"""`check_metric` and `check_adapter` — the properties a plugin must hold.

Each catches mistakes that only show up once the plugin runs for real.

`check_metric`: missing from the registry, non-deterministic between calls,
a `not_applicable` result with a stray value or no reason, a status that
fires on clean data, or a value that reads the same on clean and defective
data and so catches nothing.
`fires_on` is the ordinary way to declare the defect a metric exists to catch —
it names a `Defect` and `apply_defect` builds the defective context;
`defective` is the escape hatch for a fault no injector builds.

`check_adapter`: a confidence bid outside 0.0-1.0 or wrong on a supported or
unsupported path, `describe` naming the wrong adapter or a wrong episode count,
`episodes` returning a materialised list or not being deterministic,
`sample` not honoured, non-monotonic or non-float timestamps, an empty taxonomy type,
or a path that should be declined instead succeeding or raising anything
but `AdapterRefusal`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from collections.abc import Callable, Iterator, Sequence
from os import PathLike
from typing import Any

# External
from upath import UPath

# Internal
from kalanos.analysis.metrics.registry import registered_metrics
from kalanos.analysis.models.adapters import Adapter, AdapterRefusal, DatasetInfo
from kalanos.analysis.models.domain import Episode, Stream
from kalanos.analysis.models.metrics import (
    Level,
    MetricInput,
    MetricResult,
    MetricStatus,
)
from kalanos.analysis.models.policy import Policy
from kalanos.testing.injectors import Defect, apply_defect


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# What a stream is typed when no dictionary entry claimed it.
# Spelled here rather than imported: the contract is what an out-of-tree adapter is
# measured against, so it must not depend on a built-in adapter's private name.
_UNMAPPED_PREFIX = "unmapped."

_PathArg = str | PathLike[str] | UPath


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _check_value_matches_status(name: str, result: MetricResult) -> None:
    """Assert that `result.value` is `None` iff its status is `not_applicable`."""

    if result.status == MetricStatus.NOT_APPLICABLE:
        assert result.value is None, (
            f"{name} returned not_applicable with a value; value must be None"
        )
        reason = result.evidence.get("reason")
        assert isinstance(reason, str) and reason, (
            f"{name} returned not_applicable with no non-empty evidence['reason']"
        )
    else:
        assert result.value is not None, (
            f"{name} returned status {result.status} with no value"
        )


def check_metric(
    func: Callable[[Any], MetricResult],
    *,
    clean: MetricInput,
    fires_on: Defect | None = None,
    defective: MetricInput | None = None,
    policy: Policy | None = None,
) -> None:
    """Assert that a metric function holds every property a registered metric must.

    Checks, in order:

    - `func` is registered exactly once.
    - It returns a `MetricResult` on `clean`.
    - That result is deterministic.
    - A `not_applicable` result carries no value and a reason; any
      other result carries a value.
    - `clean` stays quiet: its status is neither `warning` nor `critical`.
    - When given, `fires_on` or `defective` produces a different result than
      `clean` and `policy` declares an entry for `func`.

    Parameters
    ----------
    func : Callable[[Any], MetricResult]
        The metric function to check, exactly as it was decorated with
        `@metric`.
    clean : MetricInput
        A context built from data with no defect for this metric to catch.
    fires_on : Defect or None
        The fault `func` exists to catch, injected into `clean` via
        `apply_defect` to build the defective context. The ordinary way to
        declare a metric's discrimination; mutually exclusive with `defective`.
    defective : MetricInput or None
        A context already carrying the defect `func` exists to catch,
        for a fault no injector builds. Mutually exclusive with `fires_on`.
    policy : Policy or None
        The policy `func` should be declared in. Skipped when `None`.

    Raises
    ------
    AssertionError
        On the first property that fails, naming `func.__name__`.
    ValueError
        If both `fires_on` and `defective` are given.
    """

    if fires_on is not None and defective is not None:
        raise ValueError(
            "fires_on and defective name the same thing two ways; defective is "
            "for a fault no injector builds, so pass only one"
        )

    name = func.__name__

    matches = [
        entry
        for level in Level
        for entry in registered_metrics(level)
        if entry.name == name
    ]
    assert len(matches) == 1, (
        f"{name} must be registered exactly once via @metric; found {len(matches)}"
    )

    clean_result = func(clean)
    assert isinstance(clean_result, MetricResult), (
        f"{name} must return a MetricResult; got {type(clean_result).__name__}"
    )

    repeat_result = func(clean)
    assert repeat_result == clean_result, (
        f"{name} is not deterministic: two calls on the same context "
        f"gave {clean_result!r} and {repeat_result!r}"
    )

    _check_value_matches_status(name, clean_result)

    assert clean_result.status not in (MetricStatus.WARNING, MetricStatus.CRITICAL), (
        f"{name} fired on data carrying no defect: clean gave {clean_result!r}"
    )

    if fires_on is not None:
        try:
            defective = apply_defect(clean, fires_on)
        except ValueError as exc:
            raise AssertionError(
                f"{name}: could not build a {fires_on.value} defect for the "
                f"context check_metric was given — {exc}"
            ) from exc

    if defective is not None:
        defective_result = func(defective)
        _check_value_matches_status(name, defective_result)

        clean_value, defective_value = clean_result.value, defective_result.value
        same_status = defective_result.status == clean_result.status
        same_value = (clean_value is None and defective_value is None) or (
            clean_value is not None
            and defective_value is not None
            and math.isclose(clean_value, defective_value, rel_tol=1e-9, abs_tol=1e-12)
        )
        assert not (same_status and same_value), (
            f"{name} could not tell clean data from defective data: "
            f"clean gave {clean_result!r}, defective gave {defective_result!r}"
        )

    if policy is not None:
        assert policy.entry_for(name) is not None, (
            f"{name} has no entry in the given policy; it will stay report_only forever"
        )


def _as_paths(
    name: str, side: str, value: _PathArg | Sequence[_PathArg]
) -> list[UPath]:
    """Normalise one path or a sequence of them to a non-empty list of `UPath`.

    Parameters
    ----------
    name : str
        The adapter's name, for the assertion message.
    side : str
        `"supported"` or `"unsupported"`, naming which argument this is.
    value : _PathArg or Sequence[_PathArg]
        A single path, or a sequence of them.

    Returns
    -------
    list[UPath]
        `value`, normalised and non-empty.
    """

    if isinstance(value, str | PathLike | UPath):
        paths = [UPath(value)]
    else:
        paths = [UPath(item) for item in value]

    assert paths, f"{name}: {side} paths must not be empty"
    return paths


def _check_confidence_in_range(name: str, adapter: Adapter, path: UPath) -> float:
    """Assert `adapter.detect(path)` returns a float confidence in `[0.0, 1.0]`.

    Parameters
    ----------
    name : str
        The adapter's name, for the assertion message.
    adapter : Adapter
        The adapter under test.
    path : UPath
        The path to detect on.

    Returns
    -------
    float
        The bid `detect` returned.
    """

    bid = adapter.detect(path)
    assert isinstance(bid, int | float) and not isinstance(bid, bool), (
        f"{name}: detect({path}) must return a float, got {type(bid).__name__}"
    )
    bid = float(bid)
    assert 0.0 <= bid <= 1.0, (
        f"{name}: detect({path}) returned {bid}, outside the 0.0-1.0 range"
    )
    return bid


def _check_describe(name: str, adapter: Adapter, path: UPath) -> DatasetInfo:
    """Assert `adapter.describe(path)` names this adapter and this path.

    Parameters
    ----------
    name : str
        The adapter's name, for the assertion message.
    adapter : Adapter
        The adapter under test.
    path : UPath
        The path to describe.

    Returns
    -------
    DatasetInfo
        The info `describe` returned.
    """

    info = adapter.describe(path)
    assert isinstance(info, DatasetInfo), (
        f"{name}: describe({path}) must return a DatasetInfo, got {type(info).__name__}"
    )
    assert info.adapter == adapter.name, (
        f"{name}: describe({path}) named adapter {info.adapter!r}, "
        f"expected {adapter.name!r}"
    )
    assert info.path == UPath(path), (
        f"{name}: describe({path}) named path {info.path!r}, expected {UPath(path)!r}"
    )
    return info


def _check_timestamps(name: str, stream: Stream, path: UPath) -> None:
    """Assert a stream's timestamps are float, non-null, finite and non-decreasing.

    Parameters
    ----------
    name : str
        The adapter's name, for the assertion message.
    stream : Stream
        The stream whose timestamps are checked.
    path : UPath
        The path the stream came from, for the assertion message.
    """

    timestamps = stream.timestamps
    subject = f"{name}: {path} stream {stream.taxonomy_type!r} timestamps"
    assert timestamps.dtype.is_float(), (
        f"{subject} must be float dtype, got {timestamps.dtype}"
    )
    assert timestamps.null_count() == 0, f"{subject} must not contain nulls"
    assert timestamps.is_finite().all(), f"{subject} must all be finite"
    assert timestamps.diff().drop_nulls().ge(0).all(), (
        f"{subject} must be non-decreasing"
    )


def _check_stream(name: str, stream: Stream, path: UPath) -> None:
    """Assert one stream's timestamps, taxonomy type and payload length are sound.

    Parameters
    ----------
    name : str
        The adapter's name, for the assertion message.
    stream : Stream
        The stream to check.
    path : UPath
        The path the stream came from, for the assertion message.
    """

    _check_timestamps(name, stream, path)

    taxonomy_type = stream.taxonomy_type
    assert taxonomy_type, f"{name}: {path} has a stream with an empty taxonomy_type"
    is_bare_or_stray_unmapped = taxonomy_type.startswith(
        "unmapped"
    ) and not taxonomy_type.startswith(_UNMAPPED_PREFIX)
    assert not is_bare_or_stray_unmapped, (
        f"{name}: {path} stream taxonomy_type {taxonomy_type!r} must name a real "
        f"taxonomy type or start with {_UNMAPPED_PREFIX!r}"
    )

    if stream.payload is not None:
        payload_length, timestamp_length = len(stream.payload), len(stream.timestamps)
        assert payload_length == timestamp_length, (
            f"{name}: {path} stream {taxonomy_type!r} payload has "
            f"{payload_length} samples but timestamps has {timestamp_length}"
        )


def _episode_shape(episodes: list[Episode]) -> list[tuple]:
    """Build a comparable shape for a list of episodes, for the determinism check.

    Compared rather than the models themselves because `Stream` holds a
    `pl.Series` and a `pl.DataFrame`, neither of which compares to a bool
    under `==`.

    Parameters
    ----------
    episodes : list[Episode]
        The episodes to summarise.

    Returns
    -------
    list[tuple]
        One tuple per episode: its `id`, and per stream a tuple of
        `(taxonomy_type, instance, attribution, source_field, timestamp
        values, channel names)`.
    """

    return [
        (
            episode.id,
            tuple(
                (
                    stream.taxonomy_type,
                    stream.instance,
                    stream.attribution,
                    stream.source_field,
                    tuple(stream.timestamps.to_list()),
                    tuple(channel.name for channel in stream.channels),
                )
                for stream in episode.streams
            ),
        )
        for episode in episodes
    ]


def _drain(
    name: str, adapter: Adapter, path: UPath, sample: int | None = None
) -> list[Episode]:
    """Call `adapter.episodes` and assert it returns an iterator, then drain it.

    Parameters
    ----------
    name : str
        The adapter's name, for the assertion message.
    adapter : Adapter
        The adapter under test.
    path : UPath
        The path to read.
    sample : int or None
        Forwarded to `episodes`.

    Returns
    -------
    list[Episode]
        Every episode `episodes` yielded.
    """

    result = adapter.episodes(path, sample=sample)
    assert isinstance(result, Iterator), (
        f"{name}: episodes({path}) must return an Iterator, not a materialised "
        f"list — got {type(result).__name__}"
    )
    return list(result)


def check_adapter(
    adapter: Adapter,
    supported: _PathArg | Sequence[_PathArg],
    unsupported: _PathArg | Sequence[_PathArg],
) -> None:
    """Assert that an adapter holds every property the `Adapter` protocol requires.

    Checks, in order, for each supported path:

    - `detect` bids a float in `[0.0, 1.0]`, and above zero.
    - `describe` names this adapter and this path, and its `episode_count`,
      when declared, matches a full iteration.
    - `episodes` returns an iterator rather than a materialised list, is
      deterministic between two calls, and yields at least one episode.
    - Every stream's timestamps are float, non-null, finite and non-decreasing,
      its taxonomy type is non-empty and either resolved or explicitly `unmapped.*`,
      and its payload length matches its timestamps.
    - `sample=0` yields nothing, and `sample=1` yields at most one episode.

    And for each unsupported path:

    - `detect` bids exactly `0.0`.
    - Reading it anyway either raises `AdapterRefusal` or yields nothing;
      any other exception is a failure.

    Parameters
    ----------
    adapter : Adapter
        The adapter to check.
    supported : _PathArg or Sequence[_PathArg]
        One path, or several, the adapter claims to read.
    unsupported : _PathArg or Sequence[_PathArg]
        One path, or several, the adapter must decline.

    Raises
    ------
    AssertionError
        On the first property that fails, naming `adapter.name` and the path.
    """

    assert isinstance(adapter, Adapter), (
        f"{type(adapter).__name__} does not satisfy the Adapter protocol"
    )
    name = adapter.name
    assert isinstance(name, str) and name, (
        f"{type(adapter).__name__}.name must be a non-empty string"
    )

    supported_paths = _as_paths(name, "supported", supported)
    unsupported_paths = _as_paths(name, "unsupported", unsupported)

    for path in supported_paths:
        bid = _check_confidence_in_range(name, adapter, path)
        assert bid > 0.0, (
            f"{name}: detect({path}) bid zero on a path it is claimed to support"
        )

        info = _check_describe(name, adapter, path)

        first = _drain(name, adapter, path)
        second = _drain(name, adapter, path)
        assert _episode_shape(first) == _episode_shape(second), (
            f"{name}: episodes({path}) is not deterministic between two calls"
        )
        assert first, f"{name}: episodes({path}) yielded no episodes"

        if info.episode_count is not None:
            assert info.episode_count == len(first), (
                f"{name}: describe({path}) declared episode_count="
                f"{info.episode_count}, but a full iteration yielded {len(first)}"
            )

        for episode in first:
            for stream in episode.streams:
                _check_stream(name, stream, path)

        assert _drain(name, adapter, path, sample=0) == [], (
            f"{name}: episodes({path}, sample=0) must yield nothing"
        )
        sampled = _drain(name, adapter, path, sample=1)
        assert len(sampled) <= 1, (
            f"{name}: episodes({path}, sample=1) yielded {len(sampled)} episodes"
        )

    for path in unsupported_paths:
        bid = _check_confidence_in_range(name, adapter, path)
        assert bid == 0.0, (
            f"{name}: detect({path}) bid {bid} on a path it does not support; "
            "must bid 0.0"
        )

        try:
            leftover = list(adapter.episodes(path))
        except AdapterRefusal:
            pass
        except Exception as exc:
            raise AssertionError(
                f"{name}: episodes({path}) must be declined via AdapterRefusal "
                f"or by yielding nothing, not raise {type(exc).__name__}"
            ) from exc
        else:
            assert not leftover, (
                f"{name}: episodes({path}) yielded {len(leftover)} episode(s) "
                "from a path it does not support; must yield nothing or raise "
                "AdapterRefusal"
            )
