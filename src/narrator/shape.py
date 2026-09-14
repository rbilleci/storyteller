"""Render a schema's own fields as the one illustrative JSON shape its prompt states.

``example_json`` derives the string from the schema instead: it walks
``model.model_fields`` in schema order -- the order guided decoding answers in, so
this order is not incidental -- and renders one example value per field from its type.
A ``Literal`` field with one option (a discriminator, ``kind``) renders that value; a
``Literal`` field with several (``SweepMention.kind``/``.claim``) renders them
pipe-joined, matching the hand-written convention it replaces. ``str`` and
``list[str]`` fields render the placeholder ``\"...\"``. A ``list`` of a nested model
(``mentions: list[SweepMention]``) recurses. An ``int`` field with a concrete default
(``in_game_time_delta_minutes``) renders that default; an ``int | None`` field has no
usable default (``None`` is not an illustrative number), so it must carry
``Field(examples=[N])`` -- a real schema value, not a second hand-typed one -- and
renders ``\"N|null\"``. Any other shape is a field type this module does not yet know
how to render, and raises rather than emit prompt text nobody verified.
"""

from __future__ import annotations

import types
from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel


def example_json(model: type[BaseModel]) -> str:
    """One illustrative JSON object for ``model``'s own fields, in schema order."""
    parts = [f'"{name}": {_example_value(info)}' for name, info in model.model_fields.items()]
    return "{" + ", ".join(parts) + "}"


def _example_value(info) -> str:
    annotation = info.annotation
    origin = get_origin(annotation)
    if origin is Literal:
        values = get_args(annotation)
        return '"' + "|".join(str(value) for value in values) + '"'
    if origin is types.UnionType:
        (inner,) = (arg for arg in get_args(annotation) if arg is not type(None))
        if inner is int:
            example = info.examples[0] if info.examples else info.default
            return f"{example}|null"
    elif annotation is int:
        return str(info.default)
    elif annotation is str:
        return '"..."'
    elif origin is list:
        (item,) = get_args(annotation)
        if get_origin(item) is Annotated:
            item = get_args(item)[0]
        if isinstance(item, type) and issubclass(item, BaseModel):
            return "[" + example_json(item) + "]"
        if item is str:
            return '["..."]'
    raise TypeError(f"narrator.shape.example_json has no rule for {annotation!r}")
