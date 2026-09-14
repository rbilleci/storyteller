"""The hazard-assessment contract: the schema the assessor answers in, and its prompt.

The assessor is one stateless, tool-less, schema-forced model call on the settle
step's exact model path, asked a single typed question: does this message DECLARE an
action the character performs now, or is it a report, a question, or table talk? The
engine keeps final authority in the safe direction only:

- A confirmation is dropped only on a schema-valid answer whose ``kind`` is not
  ``declaration``. Timeout, parse failure, transport failure, an absent assessor —
  every failure keeps the confirmation, so the floor can never end up weaker than the
  lexical floor that shipped before this step.
- The assessment runs only where a confirmation is otherwise about to be authored:
  after the state-based fight sanction, the dead-target rule, and the label-basis
  check have all kept it. It can remove a spurious ask; it can never add or remove a
  mechanical capability, because every durable consequence still resolves through
  audited tools.

The residual this accepts, on the record: a genuine first-strike declaration phrased
in the past tense (\"I attacked the merchant\", meaning now) that the assessor reads as
a report skips the table-safety ask. The tool boundary still gates every consequence
(the fight opens through ``combat_start`` and touches only its roster), and the live
probe corpus (``tests_narrator/test_probe_hazard_assessment.py``) measures exactly
this boundary: every present-tense declaration in the corpus must classify as
``declaration`` for the probe to pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from narrator.classify import FightProvider, PresenceProvider
from narrator.facets import (
    ClassificationContext,
    ContextProvider,
    compose_schema,
    message_preamble,
    scene_lines,
)
from narrator.policy_types import CombatSnapshot, TrustedScope

ASSESSOR_SYSTEM_PROMPT = (
    "You read one player message from a tabletop role-playing game session and judge "
    "what kind of message it is. You never narrate, never resolve anything, and never "
    "invent context beyond what you are shown."
)


@dataclass(frozen=True)
class AssessmentFacet:
    """The one question the assessor asks: what kind of message is this.

    ``declaration`` keeps the risk floor's confirmation; every other kind drops it.
    The set is deliberately about the message's speech act, never about its hazard
    category -- the lexical trigger already decided a hazard word is present, and
    the engine's typed rules (fight sanction, dead target, label basis) already
    decided the scene. The one open question is whether the player is *doing*
    something now. This is the whole composition: one facet, no resolvers, and
    ``derive`` is never called -- ``assessment_downgrades`` below reads the verdict
    directly, since nothing here feeds a ``TurnPolicy``.
    """

    name: str = "assessment"

    @property
    def fields(self) -> dict[str, tuple[Any, Any]]:
        return {"kind": (Literal["declaration", "report", "question", "table_talk"], ...)}

    def rules(self) -> list[str]:
        return [
            "Judge what kind of message this is:",
            "- declaration: the player's character performs an action now, in the fiction.",
            "- report: the player describes or disputes something that already happened.",
            "- question: the player asks something instead of acting.",
            "- table_talk: the player addresses the game master or the table about the "
            "game itself rather than acting in it.",
        ]

    def shape(self) -> list[str]:
        return ['"kind": "declaration|report|question|table_talk"']


ASSESSMENT_FACET = AssessmentFacet()

#: Bounded free text for the diagnostic log and the live probe report only. It is
#: model output over player-influenced input, so it never reaches a player and
#: never enters a prompt. ``compose_schema`` appends this field to every composition,
#: after the facet's own fields, so ``HazardAssessment.reason`` keeps the same
#: position it held as the class's last hand-written field.
HazardAssessment = compose_schema(
    "HazardAssessment",
    "One typed verdict about what a player message is, not what it should cause.",
    (ASSESSMENT_FACET,),
)


def assessment_downgrades(assessment: HazardAssessment | None) -> bool:
    """Whether one assessment removes the pending risk confirmation.

    ``None`` is every failure mode collapsed to one value, and it keeps the
    confirmation: the assessor can only ever make the floor quieter on a valid,
    parsed verdict, never by breaking.
    """
    return assessment is not None and assessment.kind != "declaration"


#: The assessor renders the fight roster alone -- no player side, no turn fact --
#: because its prompt is its own measured artifact, accepted by
#: ``tests_narrator/test_probe_hazard_assessment.py``. The providers are the
#: classifier's; what differs is the composition.
ASSESSOR_PROVIDERS: tuple[ContextProvider, ...] = (
    PresenceProvider(),
    FightProvider(allies=False, defending_turn=False),
)


def assessment_prompt(
    declaration: str,
    scope: TrustedScope | None = None,
    combat: CombatSnapshot | None = None,
) -> str:
    """Render one player message and its typed scene facts into the assessor's input.

    The context is campaign data only — recorded identifiers, statuses, and the open
    fight's roster — never narration or history, so the assessor judges the message
    against facts no other model output can steer. Mirroring ``settle_prompt``'s
    hard-won contract note: the decoder enforces the schema, but the model only sees
    this prompt, so the full JSON shape is stated explicitly.
    """
    lines = message_preamble(declaration)
    lines += scene_lines(ASSESSOR_PROVIDERS, ClassificationContext(scope=scope, combat=combat))
    lines += ["", *ASSESSMENT_FACET.rules(), ""]
    lines += [
        "Answer with one JSON object, every field present: "
        "{" + ", ".join(ASSESSMENT_FACET.shape())
        + ', "reason": "one short sentence"}.',
    ]
    return "\n".join(lines)
