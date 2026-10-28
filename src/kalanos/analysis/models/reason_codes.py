"""Reason codes: each reason text in a report file is stored once, by index.

A dumped report repeats the same reason texts across thousands of episodes.
`encode_reasons` lists every distinct text once, in order of first appearance,
under `reason_codes`, and replaces each occurrence with its index there.
`decode_reasons` turns those indices back into text.

One rule decides what is coded: every value under a key named
`reason`, `reasons` or `detail`, at any depth.
It works on key names in the dumped dict, so it also covers `reason` keys
inside free-form evidence and calibration dicts, and plugin metrics' reasons.
A `reasons` map (a coverage row's counts keyed by text) has its keys coded,
written as decimal strings since JSON keys are strings.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from typing import Any


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TABLE_KEY = "reason_codes"
REASON_KEYS = frozenset({"reason", "reasons", "detail"})


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class ReasonCoder:
    """Codes reason texts across several parts of one document, sharing one table.

    Codes are given in order of first appearance across every `walk` call,
    so walking a document's parts in document order
    gives the codes `encode_reasons` gives the whole document.

    Attributes
    ----------
    codes : dict[str, int]
        Each reason text seen so far, mapped to its code.
    """

    def __init__(self) -> None:
        """Start with an empty table."""

        self.codes: dict[str, int] = {}

    def table(self) -> list[str]:
        """The reason texts seen so far, in code order.

        Returns
        -------
        list[str]
            The text whose code is `i` at index `i`.
        """

        return list(self.codes)

    def walk(self, value: Any) -> Any:
        """Replace every reason text in one part of a dumped report with its code.

        Parameters
        ----------
        value : Any
            A part of a report dumped in JSON mode. Left unchanged.

        Returns
        -------
        Any
            A copy of `value` with every reason text replaced by its code.

        Raises
        ------
        TypeError
            If a number or a boolean sits under a reason key,
            since it would read back as a code.
        """

        match value:
            case dict():
                return {
                    k: self._reason(k, v) if k in REASON_KEYS else self.walk(v)
                    for k, v in value.items()
                }
            case list():
                return [self.walk(v) for v in value]
            case _:
                return value

    def _code(self, text: str) -> int:
        """The code of `text`, given the next free one on its first appearance."""

        return self.codes.setdefault(text, len(self.codes))

    def _reason(self, key: str, value: Any) -> Any:
        """Code the value under one reason key.

        Parameters
        ----------
        key : str
            The reason key.
        value : Any
            Its value.

        Returns
        -------
        Any
            `value` with its texts coded; a `reasons` map has its keys coded.

        Raises
        ------
        TypeError
            If `value` is a number or a boolean.
        """

        match value:
            case None:
                return None
            case str():
                return self._code(value)
            case bool() | int() | float():
                raise TypeError(f"a reason must be text, got {value!r} under {key!r}")
            case list():
                return [
                    self.walk(v) if isinstance(v, dict) else self._reason(key, v)
                    for v in value
                ]
            case dict() if key == "reasons":
                return {str(self._code(k)): self.walk(v) for k, v in value.items()}
            case _:
                return self.walk(value)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def encode_reasons(data: dict[str, Any]) -> dict[str, Any]:
    """Replace every reason text in a dumped report with its code.

    Parameters
    ----------
    data : dict[str, Any]
        A report dumped in JSON mode. Left unchanged.

    Returns
    -------
    dict[str, Any]
        A copy of `data` with `reason_codes` as its first key, listing each distinct
        reason text once in order of first appearance, and every reason text
        replaced by its index in that list.

    Raises
    ------
    TypeError
        If a number or a boolean sits under a reason key,
        since it would read back as a code.
    """

    coder = ReasonCoder()
    coded = coder.walk(data)
    return {TABLE_KEY: coder.table(), **coded}


def decode_reasons(data: dict[str, Any]) -> dict[str, Any]:
    """Replace every reason code in a written report with its text.

    Parameters
    ----------
    data : dict[str, Any]
        A parsed report holding `reason_codes`. Left unchanged.

    Returns
    -------
    dict[str, Any]
        A copy of `data` without `reason_codes`, with every code under a reason key
        replaced by its text. Text already under a reason key is kept.

    Raises
    ------
    ValueError
        If `reason_codes` is not a list of strings,
        or a code is outside it.
    """

    table = data[TABLE_KEY]
    if not isinstance(table, list) or not all(isinstance(t, str) for t in table):
        raise ValueError(f"{TABLE_KEY} must be a list of strings")

    def text(code: int) -> str:
        if not 0 <= code < len(table):
            raise ValueError(
                f"reason code {code} is not in {TABLE_KEY} ({len(table)} entries)"
            )
        return table[code]

    def reason(key: str, value: Any) -> Any:
        match value:
            # bool is a subclass of int, so it must be ruled out before int.
            case bool():
                return value
            case int():
                return text(value)
            case list():
                return [
                    walk(v) if isinstance(v, dict) else reason(key, v) for v in value
                ]
            case dict() if key == "reasons":
                return {
                    text(int(k)) if k.isdecimal() else k: walk(v)
                    for k, v in value.items()
                }
            case _:
                return walk(value)

    def walk(value: Any) -> Any:
        match value:
            case dict():
                return {
                    k: reason(k, v) if k in REASON_KEYS else walk(v)
                    for k, v in value.items()
                }
            case list():
                return [walk(v) for v in value]
            case _:
                return value

    return walk({k: v for k, v in data.items() if k != TABLE_KEY})
