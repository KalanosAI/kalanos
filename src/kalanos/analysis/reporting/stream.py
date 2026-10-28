"""Write a report's JSON one episode at a time.

`write_report_stream` writes a report from its header and one episode dict at a time,
so the whole report is never held as one dict.
The reason-code table and the compact key table both depend on the whole document,
so it makes two passes over the episodes: one to build the tables, one to write.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Callable, Iterable
from typing import Any, BinaryIO

# External
from pydantic_core import to_json

# Internal
from kalanos.analysis.models.compact_keys import TABLE_KEY as COMPACT_KEY
from kalanos.analysis.models.compact_keys import KeyCounter
from kalanos.analysis.models.reason_codes import TABLE_KEY as REASONS_KEY
from kalanos.analysis.models.reason_codes import ReasonCoder
from kalanos.analysis.models.report import GradedEpisode, Report
from kalanos.analysis.reporting.omit import omit_empty_defaults


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
    head = omit_empty_defaults(
        header, header.model_dump(mode="json", context={"raw_reasons": True})
    )
    fields = list(Report.model_fields)
    cut = fields.index("episodes")
    before = [key for key in fields[:cut] if key in head]
    after = [key for key in fields[cut + 1 :] if key in head]

    # Step 2: build the reason-code table, and count keys over the coded document,
    # walking every part in document order.
    coder = ReasonCoder()
    counter = KeyCounter() if compact else None
    n_episodes = 0

    def feed(part: Any) -> None:
        coded = coder.walk(part)
        if counter is not None:
            counter.count(coded)

    for key in before:
        feed({key: head[key]})
    for episode in episodes():
        feed(episode)
        n_episodes += 1
    for key in after:
        feed({key: head[key]})
    if counter is not None:
        counter.count({REASONS_KEY: None})
        if n_episodes:
            counter.count({"episodes": None})

    # Step 3: write the document, a top-level member at a time.
    def member(key: str, value: Any) -> bytes:
        pair = {key: value}
        if counter is not None:
            pair = counter.walk(pair)
        # A one-member object, without its braces, is `"key":value`.
        return _json(pair)[1:-1]

    out.write(b"{")
    if counter is not None:
        out.write(_json(COMPACT_KEY) + b":" + _json(counter.table()) + b",")
    out.write(member(REASONS_KEY, coder.table()))
    for key in before:
        out.write(b"," + member(key, coder.walk(head[key])))
    if n_episodes:
        name = counter.walk({"episodes": None}) if counter is not None else None
        out.write(b"," + _json(next(iter(name)) if name else "episodes") + b":[")
        for k, episode in enumerate(episodes()):
            coded = coder.walk(episode)
            if counter is not None:
                coded = counter.walk(coded)
            out.write((b"," if k else b"") + _json(coded))
        out.write(b"]")
    for key in after:
        out.write(b"," + member(key, coder.walk(head[key])))
    out.write(b"}")
