"""Leave empty defaults out of a written report.

A written report leaves out a field whose value is null, an empty list or an empty map
when the field's default gives the same value, so loading it back restores that value.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from functools import cache
from typing import Any

# External
from pydantic import BaseModel


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def omit_empty_defaults(model: BaseModel, data: dict[str, Any]) -> dict[str, Any]:
    """Drop the keys of a model's JSON dump whose value is its field's empty default.

    Nested models are walked through lists, tuples and dicts.
    A key that is not a field, and any value inside a free-form dict, is kept as it is.

    Parameters
    ----------
    model : BaseModel
        The model `data` was dumped from.
    data : dict[str, Any]
        `model.model_dump(mode="json")`,
        or the sub-dict of such a dump that `model` produced.

    Returns
    -------
    dict[str, Any]
        A new dict in `data`'s key order, without the empty defaults.
        `data` is left unchanged.
    """

    defaults = _empty_defaults(type(model))
    fields = type(model).model_fields
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in defaults:
            default = defaults[key]
            if type(value) is type(default) and value == default:
                continue
        if key in fields:
            out[key] = _walk(getattr(model, key), value)
        else:
            out[key] = value
    return out


def _walk(obj: Any, value: Any) -> Any:
    """Apply `omit_empty_defaults` to every model found in `obj`, paired with its dump.

    Where the shapes of `obj` and `value` don't line up, `value` is kept whole,
    so the walk can write more than needed but never drops a value.

    Parameters
    ----------
    obj : Any
        A field's value on the model.
    value : Any
        The JSON dump of `obj`.

    Returns
    -------
    Any
        `value`, with empty defaults left out of every nested model's dump.
    """

    match obj, value:
        case BaseModel(), dict():
            return omit_empty_defaults(obj, value)
        case list() | tuple(), list() if len(obj) == len(value):
            return [
                _walk(item, dumped) for item, dumped in zip(obj, value, strict=True)
            ]
        case dict(), dict() if obj.keys() == value.keys():
            return {key: _walk(obj[key], dumped) for key, dumped in value.items()}
        case _:
            return value


@cache
def _empty_defaults(
    cls: type[BaseModel],
) -> dict[str, None | list[Any] | dict[str, Any]]:
    """Map each field of a model whose default is empty to that default's JSON form.

    Parameters
    ----------
    cls : type[BaseModel]
        The model class.

    Returns
    -------
    dict[str, None | list[Any] | dict[str, Any]]
        `None` for a field defaulting to `None`,
        `[]` for one whose factory makes an empty list, tuple or set,
        and `{}` for one whose factory makes an empty dict.
        A field with a non-empty default, or with none, is absent.
    """

    defaults: dict[str, None | list[Any] | dict[str, Any]] = {}
    for name, field in cls.model_fields.items():
        if field.default is None:
            defaults[name] = None
        elif (
            field.default_factory is not None
            and not field.default_factory_takes_validated_data
        ):
            made = field.get_default(call_default_factory=True)
            if isinstance(made, (list, tuple, set, frozenset)) and not made:
                defaults[name] = []
            elif isinstance(made, dict) and not made:
                defaults[name] = {}
    return defaults
