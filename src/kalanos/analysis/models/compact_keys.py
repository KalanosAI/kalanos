"""Compact keys: every object key in a report file replaced by a short alias.

`encode_keys` counts every key at every depth and orders them by count,
most frequent first, ties by key.
The key at index `i` gets the alias `alias(i)`, a bijective base-62 numeral:
62 one-character aliases, then 3,844 two-character ones, then three.
The ordered keys go first in the file, under `_compact`,
as `{"version": 1, "keys": [...]}`,
where `keys[i]` is the key whose alias is `alias(i)`.
Values are never touched, and each object keeps its key order.
`_` is outside the alias alphabet, so `_compact` is never an alias.
`decode_keys` maps every alias back through that table.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections import Counter
from typing import Any


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

TABLE_KEY = "_compact"
COMPACT_VERSION = 1
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class UnsupportedEncoding(ValueError):
    """A compact report declares an encoding version this kalanos can't read."""


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def alias(index: int) -> str:
    """Return the alias of the key at `index` in the key table.

    Parameters
    ----------
    index : int
        A position in the key table, from 0.

    Returns
    -------
    str
        The bijective base-62 numeral of `index`: `"0"` to `"Z"` for 0 to 61,
        then `"00"` to `"ZZ"` for 62 to 3905, then three characters.
    """

    digits: list[str] = []
    n = index + 1
    while n:
        n, rest = divmod(n - 1, len(ALPHABET))
        digits.append(ALPHABET[rest])
    return "".join(reversed(digits))


def is_compact(data: object) -> bool:
    """Tell whether a parsed report is in the compact encoding.

    Parameters
    ----------
    data : object
        A parsed report file.

    Returns
    -------
    bool
        True when `data` is an object whose first key is `_compact`.
    """

    return isinstance(data, dict) and next(iter(data), None) == TABLE_KEY


def encode_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Replace every object key in a dumped report with its alias.

    Parameters
    ----------
    data : dict[str, Any]
        A report dumped in JSON mode. Left unchanged.

    Returns
    -------
    dict[str, Any]
        A copy of `data` with `_compact` as its first key, holding the encoding version
        and the key table, and every key at every depth replaced by its alias.

    Raises
    ------
    ValueError
        If `data` already holds `_compact` at top level.
    """

    if TABLE_KEY in data:
        raise ValueError(f"a report to encode must not hold {TABLE_KEY!r}")

    counts: Counter[str] = Counter()

    def count(value: Any) -> None:
        match value:
            case dict():
                counts.update(value.keys())
                for v in value.values():
                    count(v)
            case list():
                for v in value:
                    count(v)

    count(data)
    keys = sorted(counts, key=lambda k: (-counts[k], k))
    aliases = {key: alias(i) for i, key in enumerate(keys)}

    def walk(value: Any) -> Any:
        match value:
            case dict():
                return {aliases[k]: walk(v) for k, v in value.items()}
            case list():
                return [walk(v) for v in value]
            case _:
                return value

    table = {"version": COMPACT_VERSION, "keys": keys}
    return {TABLE_KEY: table, **walk(data)}


def _table_keys(data: dict[str, Any]) -> list[str]:
    """Check a compact report's key table and return its keys in alias order.

    Parameters
    ----------
    data : dict[str, Any]
        A parsed compact report.

    Returns
    -------
    list[str]
        The table's keys, where the key at index `i` has the alias `alias(i)`.

    Raises
    ------
    UnsupportedEncoding
        If the table declares a version other than `COMPACT_VERSION`.
    ValueError
        If the table is not an object whose `keys` is a list of distinct strings.
    """

    table = data.get(TABLE_KEY)
    if not isinstance(table, dict):
        raise ValueError("malformed compact key table")
    version = table.get("version")
    # `True == 1` and `1.0 == 1`, so the type is checked too.
    if type(version) is not int or version != COMPACT_VERSION:
        raise UnsupportedEncoding(
            f"compact encoding version {version!r} is not supported; "
            f"this kalanos reads version {COMPACT_VERSION}"
        )
    keys = table.get("keys")
    if (
        not isinstance(keys, list)
        or not all(isinstance(k, str) for k in keys)
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("malformed compact key table")
    return keys


def decode_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Replace every alias in a compact report with the key it stands for.

    Parameters
    ----------
    data : dict[str, Any]
        A parsed compact report. Left unchanged.

    Returns
    -------
    dict[str, Any]
        A copy of `data` without `_compact`, with every alias at every depth
        replaced by its key.

    Raises
    ------
    UnsupportedEncoding
        If the key table declares a version other than `COMPACT_VERSION`.
    ValueError
        If the key table is malformed, or an alias is not in it.
    """

    keys = {alias(i): key for i, key in enumerate(_table_keys(data))}

    def key(a: str) -> str:
        if a not in keys:
            raise ValueError(f"compact key {a!r} is not in the key table")
        return keys[a]

    def walk(value: Any) -> Any:
        match value:
            case dict():
                return {key(k): walk(v) for k, v in value.items()}
            case list():
                return [walk(v) for v in value]
            case _:
                return value

    return walk({k: v for k, v in data.items() if k != TABLE_KEY})


def key_aliases(data: dict[str, Any]) -> dict[str, str]:
    """Map each key of a compact report to its alias, without decoding the report.

    Parameters
    ----------
    data : dict[str, Any]
        A parsed compact report.

    Returns
    -------
    dict[str, str]
        Each key in the key table, mapped to its alias.

    Raises
    ------
    UnsupportedEncoding
        If the key table declares a version other than `COMPACT_VERSION`.
    ValueError
        If the key table is malformed.
    """

    return {key: alias(i) for i, key in enumerate(_table_keys(data))}
