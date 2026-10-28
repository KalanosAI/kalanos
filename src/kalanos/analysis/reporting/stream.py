"""Write a report's JSON one episode at a time.

`write_report_stream` writes a report from its header and one episode dict at a time,
and the header's diagnostic results one at a time too,
so the whole report is never held as one dict.
The reason-code table and the compact key table both depend on the whole document,
so it makes two passes over the episodes: one to build the tables, one to write.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, BinaryIO

# External
from pydantic_core import to_json

# Internal
from kalanos.analysis.models.compact_keys import TABLE_KEY as COMPACT_KEY
from kalanos.analysis.models.compact_keys import KeyCounter
from kalanos.analysis.models.diagnostics import DiagnosticsReport
from kalanos.analysis.models.eligibility import EpisodeEligibility
from kalanos.analysis.models.reason_codes import TABLE_KEY as REASONS_KEY
from kalanos.analysis.models.reason_codes import ReasonCoder
from kalanos.analysis.models.report import GradedEpisode, Report
from kalanos.analysis.reporting.card.sections import CardRow
from kalanos.analysis.reporting.omit import omit_empty_defaults


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass
class StreamedReport:
    """A graded run whose episodes are held as compressed JSON, not as models.

    Grading a large dataset into a `Report` holds every graded episode as a model,
    several times the size of its JSON.
    This holds each as the zlib-compressed JSON of its model,
    and the few values the card and the exit code read of it.

    Attributes
    ----------
    header : Report
        Every field of the report but `episodes`, which is empty.
        It was not validated as a `Report`, since it holds no episodes;
        `check_agreement` checked it against the episodes instead.
    blobs : list[bytes]
        Each graded episode's `model_dump_json`, compressed, in report order,
        before its eligibility was decided.
    decisions : dict[str, EpisodeEligibility]
        Each episode's eligibility, by id.
    card_rows : list[CardRow]
        What the card reads of the episodes that did not pass.
    unmapped_types : frozenset[str]
        Every stream type no mapping typed, across the episodes.
    """

    header: Report
    blobs: list[bytes]
    decisions: dict[str, EpisodeEligibility]
    card_rows: list[CardRow]
    unmapped_types: frozenset[str]

    def graded_episodes(self) -> Iterator[GradedEpisode]:
        """Rebuild each graded episode, with its decision, in report order.

        Yields
        ------
        GradedEpisode
            The episode as a `Report` would hold it.
        """

        for blob in self.blobs:
            episode = GradedEpisode.model_validate_json(zlib.decompress(blob))
            decision = self.decisions[episode.id]
            yield episode.model_copy(
                update={
                    "eligibility": decision,
                    "score": episode.score.model_copy(
                        update={"train_ready": decision.compatibility_train_ready}
                    ),
                }
            )

    def episodes(self) -> Iterator[dict[str, Any]]:
        """Dump each episode as `episode_data` does, one at a time.

        Yields
        ------
        dict[str, Any]
            The episode's JSON-mode dict, reason texts uncoded.
        """

        for episode in self.graded_episodes():
            yield episode_data(episode)

    def write(self, out: BinaryIO, *, compact: bool = False) -> None:
        """Write the report's JSON, as `render_json` would write the whole `Report`.

        Parameters
        ----------
        out : BinaryIO
            Where the JSON goes.
        compact : bool
            Replace every key with a short alias, as `render_compact_json` does.
        """

        write_report_stream(self.header, self.episodes, out, compact=compact)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _json(value: Any) -> bytes:
    """Serialize one value as `render_json` does.

    Parameters
    ----------
    value : Any
        A JSON-mode value.

    Returns
    -------
    bytes
        Its compact JSON, with NaN written as null rather than as a bare token,
        which is not valid JSON.
    """

    return to_json(value, inf_nan_mode="null")


def episode_data(episode: GradedEpisode) -> dict[str, Any]:
    """Dump one graded episode as a report file holds it, reason texts uncoded.

    Parameters
    ----------
    episode : GradedEpisode
        The episode to dump.

    Returns
    -------
    dict[str, Any]
        The episode in JSON mode, without the empty values that defaults restore.
    """

    return omit_empty_defaults(episode, episode.model_dump(mode="json"))


def write_report_stream(
    header: Report,
    episodes: Callable[[], Iterable[dict[str, Any]]],
    out: BinaryIO,
    *,
    compact: bool,
) -> None:
    """Write a report as `render_json` or `render_compact_json` would, by episode.

    Parameters
    ----------
    header : Report
        Every field of the report but `episodes`, which it must leave empty.
    episodes : Callable[[], Iterable[dict[str, Any]]]
        Yields each episode as `episode_data` dumps it, in report order.
        Called once per pass, so twice.
    out : BinaryIO
        Where the JSON goes.
    compact : bool
        Replace every key with a short alias, as `render_compact_json` does.

    Raises
    ------
    ValueError
        If `header` holds episodes.
    TypeError
        If a number or a boolean sits under a reason key.
    """

    if header.episodes:
        raise ValueError("a report header must hold no episodes")

    # Step 1: the header in the order the whole dump would hold it.
    # An empty `episodes` is left out of the header by `omit_empty_defaults`,
    # so it is placed back by its position among the fields.
    # The diagnostic results grow with the dataset, so they are dumped one at a time
    # and placed back by their position among the diagnostics' fields.
    # Left out when empty, they are written nowhere, as the whole dump leaves them out.
    head = omit_empty_defaults(
        header,
        header.model_dump(
            mode="json",
            context={"raw_reasons": True},
            exclude={"diagnostics": {"results"}},
        ),
    )
    fields = list(Report.model_fields)
    cut = fields.index("episodes")
    before = [key for key in fields[:cut] if key in head]
    after = [key for key in fields[cut + 1 :] if key in head]
    results = header.diagnostics.results if header.diagnostics is not None else []
    diagnostics: dict[str, Any] = head.get("diagnostics") or {}
    owned = list(DiagnosticsReport.model_fields)
    split = owned.index("results")
    ahead = [key for key in owned[:split] if key in diagnostics]
    behind = [key for key in owned[split + 1 :] if key in diagnostics]

    def diagnostic_results() -> Iterator[dict[str, Any]]:
        """Dump each diagnostic result as the whole header dump would hold it."""

        for r in results:
            yield omit_empty_defaults(
                r, r.model_dump(mode="json", context={"raw_reasons": True})
            )

    # Step 2: build the reason-code table, and count keys over the coded document,
    # walking every part in document order.
    coder = ReasonCoder()
    counter = KeyCounter() if compact else None
    n_episodes = 0

    def feed(part: Any) -> None:
        coded = coder.walk(part)
        if counter is not None:
            counter.count(coded)

    def feed_member(key: str) -> None:
        if key != "diagnostics" or not results:
            feed({key: head[key]})
            return
        feed({key: {k: diagnostics[k] for k in ahead}})
        if counter is not None:
            counter.count({"results": None})
        for result in diagnostic_results():
            feed(result)
        feed({k: diagnostics[k] for k in behind})

    for key in before:
        feed_member(key)
    for episode in episodes():
        feed(episode)
        n_episodes += 1
    for key in after:
        feed_member(key)
    if counter is not None:
        counter.count({REASONS_KEY: None})
        if n_episodes:
            counter.count({"episodes": None})

    # Step 3: write the document, a top-level member at a time.
    def name(key: str) -> bytes:
        aliased = counter.walk({key: None}) if counter is not None else {key: None}
        return _json(next(iter(aliased)))

    def member(key: str, value: Any) -> bytes:
        pair = coder.walk({key: value})
        if counter is not None:
            pair = counter.walk(pair)
        # A one-member object, without its braces, is `"key":value`.
        return _json(pair)[1:-1]

    def write_member(key: str) -> None:
        if key != "diagnostics" or not results:
            out.write(b"," + member(key, head[key]))
            return
        out.write(b"," + name(key) + b":{")
        out.write(b",".join(member(k, diagnostics[k]) for k in ahead))
        out.write((b"," if ahead else b"") + name("results") + b":[")
        for k, result in enumerate(diagnostic_results()):
            coded = coder.walk(result)
            if counter is not None:
                coded = counter.walk(coded)
            out.write((b"," if k else b"") + _json(coded))
        out.write(b"]")
        for k in behind:
            out.write(b"," + member(k, diagnostics[k]))
        out.write(b"}")

    out.write(b"{")
    if counter is not None:
        out.write(_json(COMPACT_KEY) + b":" + _json(counter.table()) + b",")
    out.write(member(REASONS_KEY, coder.table()))
    for key in before:
        write_member(key)
    if n_episodes:
        out.write(b"," + name("episodes") + b":[")
        for k, episode in enumerate(episodes()):
            coded = coder.walk(episode)
            if counter is not None:
                coded = counter.walk(coded)
            out.write((b"," if k else b"") + _json(coded))
        out.write(b"]")
    for key in after:
        write_member(key)
    out.write(b"}")
