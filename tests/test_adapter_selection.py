"""Verifies selection: the highest bid wins, and a tie at the top is an error."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import logging
from collections.abc import Iterator

# External
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters.discover import LoadedAdapter, PluginSource
from kalanos.analysis.adapters.select import select_adapter
from kalanos.analysis.models.adapters import AdapterTie, DatasetInfo
from kalanos.analysis.models.domain import Episode


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _FakeAdapter:
    """A minimal Adapter, bidding a fixed confidence regardless of the path."""

    def __init__(self, name: str, bid: float) -> None:
        self.name = name
        self._bid = bid

    def detect(self, path: UPath) -> float:
        return self._bid

    def describe(self, path: UPath) -> DatasetInfo:
        return DatasetInfo(adapter=self.name, path=path)

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        return iter(())


class _RaisingAdapter:
    """An Adapter whose detect() always raises."""

    name = "raising"

    def detect(self, path: UPath) -> float:
        raise RuntimeError("boom")

    def describe(self, path: UPath) -> DatasetInfo:
        return DatasetInfo(adapter=self.name, path=path)

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        return iter(())


def _loaded(adapter) -> LoadedAdapter:
    """Wrap a fake adapter as `discover_adapters` would.

    Parameters
    ----------
    adapter : Adapter
        The adapter instance to wrap.

    Returns
    -------
    LoadedAdapter
        The adapter, tagged as if it came from an entry point.
    """

    return LoadedAdapter(
        name=adapter.name,
        adapter=adapter,
        source=PluginSource.ENTRY_POINT,
        origin="test",
    )


_PATH = UPath("recording.csv")


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_the_highest_bidder_wins():
    """Verify three adapters bidding 0.9/0.3/0.0 select the 0.9 bidder."""

    adapters = [
        _loaded(_FakeAdapter("high", 0.9)),
        _loaded(_FakeAdapter("mid", 0.3)),
        _loaded(_FakeAdapter("zero", 0.0)),
    ]

    selection = select_adapter(_PATH, adapters)

    assert selection is not None
    assert selection.name == "high"
    assert selection.confidence == 0.9


def test_a_tie_raises_naming_both_adapters_their_bids_and_the_path():
    """Verify a two-way tie raises AdapterTie with both names, both bids, the path."""

    adapters = [
        _loaded(_FakeAdapter("csv", 0.3)),
        _loaded(_FakeAdapter("myformat", 0.3)),
    ]

    with pytest.raises(AdapterTie) as excinfo:
        select_adapter(_PATH, adapters)

    message = str(excinfo.value)
    assert "csv" in message
    assert "myformat" in message
    assert "0.3" in message
    assert str(_PATH) in message
    assert excinfo.value.bids == {"csv": 0.3, "myformat": 0.3}


def test_a_three_way_tie_names_all_three_not_two():
    """Verify a three-way tie's AdapterTie carries every tied name, not just two."""

    adapters = [
        _loaded(_FakeAdapter("a", 0.5)),
        _loaded(_FakeAdapter("b", 0.5)),
        _loaded(_FakeAdapter("c", 0.5)),
    ]

    with pytest.raises(AdapterTie) as excinfo:
        select_adapter(_PATH, adapters)

    assert "a, b and c" in str(excinfo.value)


def test_all_zero_bids_return_none():
    """Verify every adapter bidding zero returns None rather than raising."""

    adapters = [_loaded(_FakeAdapter("a", 0.0)), _loaded(_FakeAdapter("b", 0.0))]

    assert select_adapter(_PATH, adapters) is None


def test_a_raising_detect_is_scored_zero_and_the_run_continues(caplog):
    """Verify a raising detect() is scored zero, and the error and name are logged."""

    adapters = [_loaded(_RaisingAdapter()), _loaded(_FakeAdapter("good", 0.5))]

    with caplog.at_level(logging.WARNING):
        selection = select_adapter(_PATH, adapters)

    assert selection is not None
    assert selection.name == "good"
    assert any(
        "raising" in record.getMessage() and "boom" in record.getMessage()
        for record in caplog.records
    )


def test_a_raising_detect_as_the_only_adapter_returns_none():
    """Verify a lone raising detect() returns None rather than propagating."""

    adapters = [_loaded(_RaisingAdapter())]

    assert select_adapter(_PATH, adapters) is None


def test_two_adapters_sharing_a_name_still_out_bid_correctly():
    """Verify the higher of two same-named adapters wins, not whichever loaded last.

    Discovery deliberately returns duplicate names and leaves resolving them
    to selection — a dict keyed by name would collapse two distinct adapters
    into one bid and could hand the file to the wrong instance.
    """

    low = _FakeAdapter("csv", 0.3)
    high = _FakeAdapter("csv", 0.9)
    adapters = [_loaded(low), _loaded(high)]

    selection = select_adapter(_PATH, adapters)

    assert selection is not None
    assert selection.adapter is high
    assert selection.confidence == 0.9


def test_two_adapters_sharing_a_name_and_bid_still_raise_a_tie():
    """Verify a same-name, same-bid collision still raises rather than picking one."""

    adapters = [_loaded(_FakeAdapter("csv", 0.3)), _loaded(_FakeAdapter("csv", 0.3))]

    with pytest.raises(AdapterTie):
        select_adapter(_PATH, adapters)
