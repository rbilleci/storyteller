"""The inversion-of-control seams a turn classifier is composed from.

``narrator.classify`` used to fuse five things into two functions: which scene facts
the prompt carries, which questions it asks, the schema the answer arrives in, the
policy the engine derives from that answer, and the transport that sends it. This
module separates them so a context that needs a different classifier -- a session-zero
ceremony with no fight and no stock, a table playing a different game with its own
social categories, a test suite with no endpoint -- composes one from parts rather than
forking a 600-line module. ``narrator.classify`` is the default composition, and it
composes to byte-identical prompt and schema bytes to the fused version it replaced;
``tests_narrator/test_classifier_contract.py`` pins that against golden files.

Three measurements from ``narrator.classify`` constrain what may be inverted here, and
each one is a type below rather than a docstring:

What is deliberately *not* inverted: the game-master designator lane
(``interactions.gm_discussion_remainder``) is syntax decided before any classifier
runs, and the resolvers are ordered and closed -- a facet emits evidence about the
player's text, and only the classifier's own resolver list may turn evidence into a
weaker gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, create_model

from narrator.policy_types import CombatSnapshot, InteractionCue, TrustedScope, TurnPolicy

TRUSTED: frozenset[str] = frozenset({"recorded_state", "service_state", "model_sourced"})


@dataclass(frozen=True)
class ClassificationContext:
    """Everything a classifier may know about the turn besides the player's text.

    The first three fields reach the prompt through context providers. The rest
    never reach the prompt at all: they are the channel's own focus and referent
    state, read only by the policy derivation, which is how they stay out of the
    model's view while still deciding where a bare reply lands or which referents a
    referent-less reply inherits.

    ``last_named_person_ids``/``last_names_unrecorded_person``/
    ``awaiting_referent_reply`` are the immediately preceding delivered turn's own
    referents and whether its narration ended in a question -- the same shape
    ``active_focus``/``awaiting_reply`` carry for an addressed interlocutor, but
    tracked independently: a declaration that names a corpse as the object of an
    action, never as someone addressed, sets no interlocutor focus at all, so
    ``ReplyInheritsReferents`` needs its own memory of what was named.
    """

    scope: TrustedScope | None = None
    combat: CombatSnapshot | None = None
    offer_open: bool = False
    active_focus: InteractionCue | None = None
    awaiting_reply: bool = False
    last_named_person_ids: tuple[str, ...] = ()
    last_names_unrecorded_person: bool = False
    awaiting_referent_reply: bool = False


@runtime_checkable
class ContextProvider(Protocol):
    """Renders typed context into prompt lines, under a declared trust class."""

    name: str
    trust: str

    def lines(self, context: ClassificationContext) -> list[str]: ...


@runtime_checkable
class Facet(Protocol):
    """One question the classifier asks, with its schema fields and its derivation.

    ``fields`` maps each field name to a ``(annotation, default)`` pair in the form
    ``pydantic.create_model`` takes; ``...`` marks a required field. ``rules`` returns
    the prompt lines for this facet's block, ``shape`` the fragments it contributes to
    the JSON-shape line, and ``derive`` writes this facet's own policy fields from the
    verdict -- and only its own; cross-field coherence belongs to a ``Resolver``.
    """

    name: str
    fields: Mapping[str, tuple[Any, Any]]

    def rules(self) -> list[str]: ...

    def shape(self) -> list[str]: ...

    def derive(self, verdict: BaseModel, context: ClassificationContext, policy: TurnPolicy) -> TurnPolicy: ...


@runtime_checkable
class Resolver(Protocol):
    """One engine-owned coherence rule over the derived policy."""

    name: str

    def __call__(self, policy: TurnPolicy, verdict: BaseModel, context: ClassificationContext) -> TurnPolicy: ...


class CompositionError(ValueError):
    """A classifier was composed from parts that cannot form one contract."""


def message_preamble(declaration: str) -> list[str]:
    """The lines every classifier prompt opens with: the player's message, quoted."""
    return ["A player at the table sent this message:", "", declaration.strip(), ""]


def scene_lines(providers: Sequence[ContextProvider], context: ClassificationContext) -> list[str]:
    """Render every provider's facts in declaration order."""
    lines: list[str] = []
    for provider in providers:
        lines.extend(provider.lines(context))
    return lines


def compose_schema(
    name: str, doc: str, facets: Sequence[Facet], *, reason_max_length: int = 240
) -> type[BaseModel]:
    """Build the verdict model from the facets' fields, in facet order.

    Field order is part of the contract: vLLM's guided decoding emits properties in
    schema order, so the order here is the order the model answers in, and a reorder
    is a prompt change that needs re-measuring. ``reason`` is always last. Every field
    name must be unique across facets, because two facets claiming one name would
    silently let the later one's derivation read the earlier one's answer.

    The docstring is passed through deliberately: the OpenAI client's strict schema
    carries it as the top-level ``description``, so it is part of the bytes sent.
    """
    fields: dict[str, tuple[Any, Any]] = {}
    for facet in facets:
        for field_name, definition in facet.fields.items():
            if field_name in fields:
                raise CompositionError(f"field {field_name!r} is claimed by two facets")
            if field_name == "reason":
                raise CompositionError("'reason' is the composition's own field")
            fields[field_name] = definition
    fields["reason"] = (str, Field(min_length=1, max_length=reason_max_length))
    return create_model(
        name,
        __doc__=doc,
        __config__=ConfigDict(str_strip_whitespace=True),
        **fields,
    )


def validate_parts(
    providers: Sequence[ContextProvider], facets: Sequence[Facet], resolvers: Sequence[Resolver]
) -> None:
    """Refuse a composition the design forbids, before it can build a prompt."""
    for provider in providers:
        if provider.trust not in TRUSTED:
            raise CompositionError(
                f"context provider {provider.name!r} declares trust {provider.trust!r}; "
                f"only {sorted(TRUSTED)} may reach the classifier"
            )
    names = [facet.name for facet in facets]
    if len(set(names)) != len(names):
        raise CompositionError(f"duplicate facet names: {names}")
    names = [resolver.name for resolver in resolvers]
    if len(set(names)) != len(names):
        raise CompositionError(f"duplicate resolver names: {names}")
