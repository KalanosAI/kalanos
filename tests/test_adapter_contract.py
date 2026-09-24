"""Verifies check_adapter: a table-shaped adapter passes it, broken ones don't."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Callable, Iterator

# External
import polars as pl
import pytest
from upath import UPath

# Internal
from kalanos.analysis.adapters.tabular import ParsedTable, TabularAdapter
from kalanos.analysis.models.adapters import DatasetInfo
from kalanos.analysis.models.domain import Episode, Stream
from kalanos.testing import check_adapter

# Local
from helpers import CSV_FIXTURE, FIXTURES_DIR


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀

SUPPORTED = CSV_FIXTURE
UNSUPPORTED = [FIXTURES_DIR / "video_meta.json", FIXTURES_DIR / "capture_index.json"]


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _CsvContractAdapter(TabularAdapter):
    """The adapter every property must hold for: bids on `.csv`, reads with polars."""

    name = "csv-contract"

    def detect(self, path: UPath) -> float:
        """Bid on suffix alone, so declining a non-CSV path is a real refusal."""

        return 0.9 if path.suffix == ".csv" else 0.0

    def parse(self, path: UPath) -> ParsedTable:
        """Read the CSV with polars."""

        return ParsedTable(frame=pl.read_csv(path))  # pyright: ignore[reportArgumentType]


class _OutOfRangeConfidenceAdapter(_CsvContractAdapter):
    """Bids above the 0.0-1.0 range."""

    name = "out-of-range-confidence"

    def detect(self, path: UPath) -> float:
        """Bid 1.5, outside the range `check_adapter` requires."""

        return 1.5


class _ZeroOnSupportedAdapter(_CsvContractAdapter):
    """Bids zero on every path, including the one it is claimed to support."""

    name = "zero-on-supported"

    def detect(self, path: UPath) -> float:
        """Bid zero unconditionally."""

        return 0.0


class _AlwaysConfidentAdapter(_CsvContractAdapter):
    """Bids confidently on every path, including ones it must decline."""

    name = "always-confident"

    def detect(self, path: UPath) -> float:
        """Bid 0.9 unconditionally."""

        return 0.9


class _WrongAdapterNameAdapter(_CsvContractAdapter):
    """Describes every path as belonging to a different adapter."""

    name = "wrong-name"

    def describe(self, path: UPath) -> DatasetInfo:
        """Name an adapter other than `self.name`."""

        return DatasetInfo(adapter="someone-else", path=path)


class _WrongEpisodeCountAdapter(_CsvContractAdapter):
    """Declares an episode count that a full iteration never matches."""

    name = "wrong-episode-count"

    def describe(self, path: UPath) -> DatasetInfo:
        """Declare 99 episodes regardless of how many the path actually holds."""

        return DatasetInfo(adapter=self.name, path=path, episode_count=99)


class _NonDeterministicAdapter(_CsvContractAdapter):
    """Tags each episode's id with an incrementing call counter."""

    name = "non-deterministic"

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Delegate to the base parse, then rewrite each episode's id per call."""

        self._calls += 1
        for episode in super().episodes(path, sample=sample):
            yield Episode(id=f"{episode.id}-{self._calls}", streams=episode.streams)


class _ListReturningAdapter(_CsvContractAdapter):
    """Materialises episodes into a list instead of yielding them lazily."""

    name = "list-returning"

    def episodes(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, path: UPath, sample: int | None = None
    ) -> list[Episode]:
        """Return a `list`, not an iterator."""

        return list(super().episodes(path, sample=sample))


class _IgnoringSampleAdapter(_CsvContractAdapter):
    """Always yields two episodes, regardless of what `sample` asks for."""

    name = "ignoring-sample"

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Ignore `sample` and yield the base episode twice."""

        for _ in range(2):
            yield from super().episodes(path)


class _IgnoringSampleAboveZeroAdapter(_CsvContractAdapter):
    """Honours `sample=0` but ignores any other `sample` value."""

    name = "ignoring-sample-above-zero"

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Yield nothing for `sample=0`; otherwise always yield two episodes."""

        if sample == 0:
            return
        for _ in range(2):
            yield from super().episodes(path)


class _RaisingOnUnsupportedAdapter(_CsvContractAdapter):
    """Raises a plain exception on a path it does not support, instead of declining."""

    name = "raising-on-unsupported"

    def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
        """Raise on anything that is not a CSV; delegate otherwise."""

        if path.suffix != ".csv":
            raise ValueError("boom")
        yield from super().episodes(path, sample=sample)


class _NotAnAdapter:
    """Has `detect` and `describe` but not `episodes` — not an Adapter."""

    name = "not-an-adapter"

    def detect(self, path: UPath) -> float:
        """Bid confidently; irrelevant, the protocol check fails first."""

        return 1.0

    def describe(self, path: UPath) -> DatasetInfo:
        """Describe minimally; irrelevant, the protocol check fails first."""

        return DatasetInfo(adapter=self.name, path=path)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _rewriting_adapter(
    rewrite: Callable[[Stream], Stream],
) -> type[_CsvContractAdapter]:
    """Build a `_CsvContractAdapter` subclass that rewrites the first stream it yields.

    Parameters
    ----------
    rewrite : Callable[[Stream], Stream]
        Applied to the first stream of the fixture's one episode.

    Returns
    -------
    type[_CsvContractAdapter]
        A subclass whose `episodes` applies `rewrite` before yielding.
    """

    class _RewritingAdapter(_CsvContractAdapter):
        """Delegates to the base parse, then applies `rewrite` to the first stream."""

        name = "rewriting"

        def episodes(self, path: UPath, sample: int | None = None) -> Iterator[Episode]:
            """Rewrite the first stream of each episode the base adapter yields."""

            for episode in super().episodes(path, sample=sample):
                streams = list(episode.streams)
                if streams:
                    streams[0] = rewrite(streams[0])
                yield episode.model_copy(update={"streams": streams})

    return _RewritingAdapter


def _reverse_timestamps(stream: Stream) -> Stream:
    """Reverse a stream's timestamps, turning increasing into decreasing."""

    return stream.model_copy(update={"timestamps": stream.timestamps.reverse()})


def _cast_timestamps_to_int(stream: Stream) -> Stream:
    """Cast a stream's timestamps to an integer dtype."""

    return stream.model_copy(update={"timestamps": stream.timestamps.cast(pl.Int64)})


def _null_first_timestamp(stream: Stream) -> Stream:
    """Replace a stream's first timestamp with a null."""

    values = stream.timestamps.to_list()
    values[0] = None
    new_timestamps = pl.Series(stream.timestamps.name, values, dtype=pl.Float64)
    return stream.model_copy(update={"timestamps": new_timestamps})


def _bare_unmapped_type(stream: Stream) -> Stream:
    """Retype a stream to the bare sentinel `unmapped`, with no name after it."""

    return stream.model_copy(update={"taxonomy_type": "unmapped"})


def _shrink_timestamps_only(stream: Stream) -> Stream:
    """Drop a stream's last timestamp without touching its payload."""

    return stream.model_copy(update={"timestamps": stream.timestamps.head(-1)})


# ░▀█▀░█▀▀░█▀▀░▀█▀░█▀▀
# ░░█░░█▀▀░▀▀█░░█░░▀▀█
# ░░▀░░▀▀▀░▀▀▀░░▀░░▀▀▀


def test_a_tabular_adapter_passes_the_contract():
    """Verify a minimal TabularAdapter subclass passes the contract."""

    check_adapter(_CsvContractAdapter(), SUPPORTED, UNSUPPORTED)


def test_a_single_path_and_a_sequence_are_both_accepted():
    """Verify `supported` accepts a one-element list, not only a bare path."""

    check_adapter(_CsvContractAdapter(), [SUPPORTED], UNSUPPORTED)


def test_an_empty_supported_sequence_is_rejected():
    """Verify an empty `supported` sequence fails, naming which side was empty."""

    with pytest.raises(AssertionError, match="supported"):
        check_adapter(_CsvContractAdapter(), [], UNSUPPORTED)


def test_an_empty_unsupported_sequence_is_rejected():
    """Verify an empty `unsupported` sequence fails, naming which side was empty."""

    with pytest.raises(AssertionError, match="unsupported"):
        check_adapter(_CsvContractAdapter(), SUPPORTED, [])


def test_a_confidence_out_of_range_fails():
    """Verify a bid outside 0.0-1.0 fails with the range named."""

    with pytest.raises(AssertionError, match="0.0-1.0 range"):
        check_adapter(_OutOfRangeConfidenceAdapter(), SUPPORTED, UNSUPPORTED)


def test_a_zero_bid_on_a_supported_path_fails():
    """Verify bidding zero on a claimed-supported path fails."""

    with pytest.raises(AssertionError, match="bid zero"):
        check_adapter(_ZeroOnSupportedAdapter(), SUPPORTED, UNSUPPORTED)


def test_a_positive_bid_on_an_unsupported_path_fails():
    """Verify bidding above zero on an unsupported path fails."""

    with pytest.raises(AssertionError, match="does not support"):
        check_adapter(_AlwaysConfidentAdapter(), SUPPORTED, UNSUPPORTED)


def test_a_describe_that_names_another_adapter_fails():
    """Verify `describe` naming a different adapter fails."""

    with pytest.raises(AssertionError, match="named adapter"):
        check_adapter(_WrongAdapterNameAdapter(), SUPPORTED, UNSUPPORTED)


def test_a_wrong_episode_count_fails():
    """Verify a declared `episode_count` that a full iteration disagrees with fails."""

    with pytest.raises(AssertionError, match="episode_count"):
        check_adapter(_WrongEpisodeCountAdapter(), SUPPORTED, UNSUPPORTED)


def test_a_non_deterministic_adapter_fails():
    """Verify episodes that differ between two calls on the same path fail."""

    with pytest.raises(AssertionError, match="not deterministic"):
        check_adapter(_NonDeterministicAdapter(), SUPPORTED, UNSUPPORTED)


def test_a_list_returning_episodes_fails():
    """Verify `episodes` returning a materialised list rather than an iterator fails."""

    with pytest.raises(AssertionError, match="Iterator"):
        check_adapter(
            _ListReturningAdapter(),  # pyright: ignore[reportArgumentType]
            SUPPORTED,
            UNSUPPORTED,
        )


def test_ignoring_sample_fails():
    """Verify an adapter that ignores `sample` is caught by the `sample=0` check."""

    with pytest.raises(AssertionError, match="sample=0"):
        check_adapter(_IgnoringSampleAdapter(), SUPPORTED, UNSUPPORTED)


def test_ignoring_a_nonzero_sample_fails():
    """Verify an adapter honouring `sample=0` but ignoring `sample=1` fails."""

    with pytest.raises(AssertionError, match="sample=1"):
        check_adapter(_IgnoringSampleAboveZeroAdapter(), SUPPORTED, UNSUPPORTED)


def test_non_monotonic_timestamps_fail():
    """Verify a decreasing timestamp series fails."""

    adapter_cls = _rewriting_adapter(_reverse_timestamps)
    with pytest.raises(AssertionError, match="non-decreasing"):
        check_adapter(adapter_cls(), SUPPORTED, UNSUPPORTED)


def test_integer_timestamps_fail():
    """Verify a non-float timestamp series fails."""

    adapter_cls = _rewriting_adapter(_cast_timestamps_to_int)
    with pytest.raises(AssertionError, match="float dtype"):
        check_adapter(adapter_cls(), SUPPORTED, UNSUPPORTED)


def test_null_timestamps_fail():
    """Verify a null in the timestamp series fails.

    `Stream` construction does not itself reject nulls, so only
    `check_adapter` catches this.
    """

    adapter_cls = _rewriting_adapter(_null_first_timestamp)
    with pytest.raises(AssertionError, match="null"):
        check_adapter(adapter_cls(), SUPPORTED, UNSUPPORTED)


def test_a_bare_unmapped_taxonomy_type_fails():
    """Verify a taxonomy type of the bare word `unmapped`, with no suffix, fails."""

    adapter_cls = _rewriting_adapter(_bare_unmapped_type)
    with pytest.raises(AssertionError, match="taxonomy type"):
        check_adapter(adapter_cls(), SUPPORTED, UNSUPPORTED)


def test_a_mismatched_payload_length_fails():
    """Verify a payload whose length no longer matches its timestamps fails."""

    adapter_cls = _rewriting_adapter(_shrink_timestamps_only)
    with pytest.raises(AssertionError, match="payload has"):
        check_adapter(adapter_cls(), SUPPORTED, UNSUPPORTED)


def test_a_stray_exception_on_an_unsupported_path_fails():
    """Verify a stray exception on an unsupported path fails, chained from the cause."""

    with pytest.raises(AssertionError, match="declined") as excinfo:
        check_adapter(_RaisingOnUnsupportedAdapter(), SUPPORTED, UNSUPPORTED)
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_a_non_adapter_object_fails():
    """Verify an object missing `episodes` fails the protocol check."""

    with pytest.raises(AssertionError, match="Adapter protocol"):
        check_adapter(
            _NotAnAdapter(),  # pyright: ignore[reportArgumentType]
            SUPPORTED,
            UNSUPPORTED,
        )
