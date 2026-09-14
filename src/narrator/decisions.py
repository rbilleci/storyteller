"""Typed, tool-less player-decision contracts.

The narrator never parses a game-master sentence to discover a decision.  A separate
structured planner emits one of these closed shapes before the narrator agent receives
the turn.  The contracts contain no campaign mutation or channel identity fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from narrator.identifiers import IDENTIFIER
from narrator.interactions import (
    combat_sanctions_violence,
    hazard_scope,
    named_dead_target,
)
from narrator.policy_types import InteractionCue, TrustedScope, TurnPolicy

MAX_DECISION_QUESTION_CHARS = 600
MAX_DECISION_CONTEXT_CHARS = 1_200
MAX_DECISION_OPTION_LABEL_CHARS = 240
MAX_DECISION_OPTIONS = 6
MAX_CUSTOM_APPROACH_CHARS = 600


MAX_ASKED_QUESTIONS = 12

_UNSAFE_PUBLIC = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class DecisionValidationError(ValueError):
    """A decision would expose an invalid or unsafe public contract."""


class _StrictDecisionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, populate_by_name=True
    )


def _public_text(value: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise DecisionValidationError("public decision text must be text")
    from narrator.delivery import scrub_markup

    _, removed = scrub_markup(value)
    if not value or len(value) > limit or _UNSAFE_PUBLIC.search(value) or removed:
        raise DecisionValidationError("public decision text is invalid")
    return value


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise DecisionValidationError("decision identifier is invalid")
    return value


class CurrentAudience(_StrictDecisionModel):
    kind: Literal["current"]


class CharacterAudience(_StrictDecisionModel):
    kind: Literal["characters"]
    character_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_DECISION_OPTIONS)

    @field_validator("character_ids")
    @classmethod
    def _unique_identifiers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_identifier(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise DecisionValidationError("decision audience repeats a character")
        return normalized


class PartyAudience(_StrictDecisionModel):
    kind: Literal["party"]


DecisionAudience = Annotated[
    CurrentAudience | CharacterAudience | PartyAudience, Field(discriminator="kind")
]


class NoTestDirective(_StrictDecisionModel):
    """Forbid decision-relevant tests and defences for one continuation."""

    kind: Literal["no_test"]
    character_id: str

    @field_validator("character_id")
    @classmethod
    def _character_id(cls, value: str) -> str:
        return _identifier(value)


class AttributeTestDirective(_StrictDecisionModel):
    """Bind the next relevant mechanic to one actor and attribute test."""

    kind: Literal["attribute_test"]
    character_id: str
    attribute: Literal["STR", "DEX", "CON", "INT", "WIS", "CHA"]

    @field_validator("character_id")
    @classmethod
    def _character_id(cls, value: str) -> str:
        return _identifier(value)


class CombatDefendDirective(_StrictDecisionModel):
    """Bind the next relevant mechanic to a defender and defence method."""

    kind: Literal["combat_defend"]
    character_id: str
    method: Literal["dodge", "parry"]

    @field_validator("character_id")
    @classmethod
    def _character_id(cls, value: str) -> str:
        return _identifier(value)


class HazardResolutionDirective(_StrictDecisionModel):
    """Bind a confirmed violent action to a resolving mechanic before narration.

    A ``confirm`` on the risk floor's violence confirmation carries this. The guard
    then withholds the turn unless the actor rolled a resolving mechanic, so a confirmed
    attack cannot dissolve into narration that never touched the dice. It binds violence
    only, because violence always resolves through a roll; a confirmed theft or an
    object's destruction may resolve with no test, so binding those would withhold a
    legitimate turn.
    """

    kind: Literal["hazard_resolution"]
    character_id: str
    category: Literal["violence"]

    @field_validator("character_id")
    @classmethod
    def _character_id(cls, value: str) -> str:
        return _identifier(value)


ResolutionDirective = Annotated[
    NoTestDirective | AttributeTestDirective | CombatDefendDirective | HazardResolutionDirective,
    Field(discriminator="kind"),
]


class NoTestProposal(_StrictDecisionModel):
    """A planner mechanic proposal with no actor, account, or transport binding."""

    kind: Literal["no_test"]


class AttributeTestProposal(_StrictDecisionModel):
    """A planner mechanic proposal with no actor, account, or transport binding."""

    kind: Literal["attribute_test"]
    attribute: Literal["STR", "DEX", "CON", "INT", "WIS", "CHA"]


class CombatDefendProposal(_StrictDecisionModel):
    """A planner mechanic proposal with no actor, account, or transport binding."""

    kind: Literal["combat_defend"]
    method: Literal["dodge", "parry"]


ProposedDirective = Annotated[
    NoTestProposal | AttributeTestProposal | CombatDefendProposal, Field(discriminator="kind")
]


class ChoiceOption(_StrictDecisionModel):
    id: str
    label: str

    @field_validator("id")
    @classmethod
    def _option_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        return _public_text(value, limit=MAX_DECISION_OPTION_LABEL_CHARS)


class ApproachOption(ChoiceOption):
    directive: ProposedDirective


class _DecisionDraft(_StrictDecisionModel):
    question: str
    context: str = ""
    audience: DecisionAudience

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        return _public_text(value, limit=MAX_DECISION_QUESTION_CHARS)

    @field_validator("context")
    @classmethod
    def _context(cls, value: str) -> str:
        if not value:
            return ""
        return _public_text(value, limit=MAX_DECISION_CONTEXT_CHARS)


class ConfirmationDraft(_DecisionDraft):
    """A defined high-stakes confirmation with fixed accept/decline meaning."""

    kind: Literal["confirmation"]
    dimension: Literal["confirmation"] = "confirmation"
    #: Set to "violence" only by the engine's risk floor for a violent commitment. A
    #: confirm then binds the action to a resolving mechanic. A planner-requested or a
    #: trade confirmation leaves this empty and binds nothing.
    binds_hazard: Literal["", "violence"] = ""


class ClarificationDraft(_DecisionDraft):
    """Listed intent paths.  Public views always add the custom path."""

    kind: Literal["clarification"]
    dimension: Literal["target", "intent"] = "intent"
    paths: tuple[ChoiceOption, ...] = Field(min_length=1, max_length=MAX_DECISION_OPTIONS - 1)

    @field_validator("paths")
    @classmethod
    def _unique_paths(cls, values: tuple[ChoiceOption, ...]) -> tuple[ChoiceOption, ...]:
        ids = [option.id for option in values]
        if len(ids) != len(set(ids)) or "own_approach" in ids:
            raise DecisionValidationError("clarification paths are not unique")
        return values


class ApproachDraft(_DecisionDraft):
    """A choice whose selected option has an executable mechanic directive."""

    kind: Literal["approach"]
    dimension: Literal["target", "approach"] = "approach"
    approaches: tuple[ApproachOption, ...] = Field(min_length=1, max_length=MAX_DECISION_OPTIONS)

    @field_validator("approaches")
    @classmethod
    def _unique_approaches(
        cls, values: tuple[ApproachOption, ...]
    ) -> tuple[ApproachOption, ...]:
        if len({option.id for option in values}) != len(values):
            raise DecisionValidationError("approach options are not unique")
        return values


DecisionDraft = Annotated[
    ConfirmationDraft | ClarificationDraft | ApproachDraft, Field(discriminator="kind")
]


class ProceedPlan(_StrictDecisionModel):
    kind: Literal["proceed"]


class DeclinedPlan(_StrictDecisionModel):
    """An engine-owned terminal outcome after a bound confirmation declines."""

    kind: Literal["declined"]


class RequestDecisionPlan(_StrictDecisionModel):
    kind: Literal["request"]
    decision: DecisionDraft


Plan = Annotated[
    ProceedPlan | RequestDecisionPlan, Field(discriminator="kind")
]


class PlanOutcome(_StrictDecisionModel):
    """The planner's extra-forbid discriminated result envelope."""

    plan: Plan


class PublicOption(_StrictDecisionModel):
    id: str
    label: str
    custom: bool = False

    @field_validator("id")
    @classmethod
    def _option_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("label")
    @classmethod
    def _label(cls, value: str) -> str:
        return _public_text(value, limit=MAX_DECISION_OPTION_LABEL_CHARS)


class TargetAssignment(_StrictDecisionModel):
    """Internal authorization binding.  It never enters a planner prompt or view."""

    character_id: str
    character_name: str
    adapter_name: str
    subject_id: str

    @field_validator("character_id")
    @classmethod
    def _character_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("character_name")
    @classmethod
    def _character_name(cls, value: str) -> str:
        return _public_text(value, limit=MAX_DECISION_OPTION_LABEL_CHARS)

    @field_validator("adapter_name", "subject_id")
    @classmethod
    def _opaque_identity_part(cls, value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 256 or _UNSAFE_PUBLIC.search(value):
            raise DecisionValidationError("decision identity binding is invalid")
        return value


class PreparedDecision(_StrictDecisionModel):
    """Internal pending state with opaque lifecycle identifiers and a stale guard."""

    request_id: str
    schema_id: Literal["structured-player-decision.v1"] = Field(
        default="structured-player-decision.v1", alias="schema"
    )
    kind: Literal["confirmation", "clarification", "approach"]
    assignments: tuple[TargetAssignment, ...] = Field(min_length=1)
    context_fingerprint: str
    draft: DecisionDraft

    @field_validator("request_id", "context_fingerprint")
    @classmethod
    def _opaque_token(cls, value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 256:
            raise DecisionValidationError("opaque decision field is invalid")
        return value


class PublicDecisionView(_StrictDecisionModel):
    """Player-visible decision content with no routing identifiers or tokens."""

    kind: Literal["confirmation", "clarification", "approach"]
    character_name: str
    question: str
    context: str = ""
    options: tuple[PublicOption, ...]


class DecisionRouting(_StrictDecisionModel):
    """Private adapter routing data that never enters planner prompts or diagnostics."""

    presentation_token: str
    character_id: str


class DecisionView(_StrictDecisionModel):
    """An assignment-specific renderer input.  It contains no account identifier."""

    presentation_token: str
    kind: Literal["confirmation", "clarification", "approach"]
    character_id: str
    character_name: str
    question: str
    context: str = ""
    options: tuple[PublicOption, ...]

    @field_validator("presentation_token")
    @classmethod
    def _token(cls, value: str) -> str:
        if not isinstance(value, str) or len(value) < 16 or len(value) > 256:
            raise DecisionValidationError("presentation token is invalid")
        return value

    @field_validator("character_id")
    @classmethod
    def _character_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("character_name", "question")
    @classmethod
    def _safe_view_text(cls, value: str) -> str:
        return _public_text(value, limit=MAX_DECISION_QUESTION_CHARS)

    @field_validator("context")
    @classmethod
    def _safe_context(cls, value: str) -> str:
        if not value:
            return ""
        return _public_text(value, limit=MAX_DECISION_CONTEXT_CHARS)

    @property
    def public(self) -> PublicDecisionView:
        """Project the view used for player text without its private delivery route."""
        return PublicDecisionView(
            kind=self.kind,
            character_name=self.character_name,
            question=self.question,
            context=self.context,
            options=self.options,
        )

    @property
    def routing(self) -> DecisionRouting:
        """Return only the token and linked character used to route an answer."""
        return DecisionRouting(
            presentation_token=self.presentation_token, character_id=self.character_id
        )


class DecisionSubmission(_StrictDecisionModel):
    """A player answer keyed only by the assignment's presentation token."""

    presentation_token: str
    selection_id: str
    custom_text: str = ""

    @field_validator("presentation_token")
    @classmethod
    def _token(cls, value: str) -> str:
        if not isinstance(value, str) or len(value) < 16 or len(value) > 256:
            raise DecisionValidationError("presentation token is invalid")
        return value

    @field_validator("selection_id")
    @classmethod
    def _selection(cls, value: str) -> str:
        return _identifier(value)


class DecisionResolution(_StrictDecisionModel):
    """Immutable answer supplied to the next planner and narrator continuation."""

    character_id: str
    kind: Literal["confirmation", "clarification", "approach"]
    answer_kind: Literal["confirm", "decline", "option", "custom"]
    selection_id: str
    custom_text: str = ""
    directive: ResolutionDirective | None = None
    action_fingerprint: str = ""

    @field_validator("character_id", "selection_id")
    @classmethod
    def _resolution_id(cls, value: str) -> str:
        return _identifier(value)

    @field_validator("action_fingerprint")
    @classmethod
    def _action_fingerprint(cls, value: str) -> str:
        if value and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
            raise DecisionValidationError("action fingerprint is invalid")
        return value


def public_options(draft: DecisionDraft, catalog=None) -> tuple[PublicOption, ...]:
    """Project one draft into a renderer-safe option list.

    ``catalog`` supplies the confirm/decline/own-approach labels in the table's
    language. Callers that hold a config pass ``config.catalog``; the English default
    keeps this callable from a test that has no config.
    """
    if isinstance(draft, ConfirmationDraft):
        return (
            PublicOption(id="confirm", label=_catalog(catalog).option("confirm")),
            PublicOption(id="decline", label=_catalog(catalog).option("decline")),
        )
    if isinstance(draft, ClarificationDraft):
        return tuple(PublicOption(id=item.id, label=item.label) for item in draft.paths) + (
            PublicOption(id="own_approach", label=_catalog(catalog).option("own_approach"), custom=True),
        )
    return tuple(PublicOption(id=item.id, label=item.label) for item in draft.approaches)


def resolution_from_submission(
    draft: DecisionDraft, character_id: str, submission: DecisionSubmission
) -> DecisionResolution:
    """Validate an answer against its draft and return its immutable typed meaning."""
    options = {option.id: option for option in public_options(draft)}
    option = options.get(submission.selection_id)
    if option is None:
        raise DecisionValidationError("selection is unavailable")
    if isinstance(draft, ConfirmationDraft):
        if submission.custom_text or submission.selection_id not in {"confirm", "decline"}:
            raise DecisionValidationError("confirmation uses fixed choices")
        if submission.selection_id == "decline":
            directive: ResolutionDirective | None = NoTestDirective(
                kind="no_test", character_id=character_id
            )
        elif draft.binds_hazard == "violence":
            directive = HazardResolutionDirective(
                kind="hazard_resolution", character_id=character_id, category="violence"
            )
        else:
            directive = None
        return DecisionResolution(
            character_id=character_id,
            kind=draft.kind,
            answer_kind=submission.selection_id,
            selection_id=submission.selection_id,
            directive=directive,
        )
    if isinstance(draft, ClarificationDraft):
        if submission.selection_id == "own_approach":
            custom = submission.custom_text.strip()
            if not custom or len(custom) > MAX_CUSTOM_APPROACH_CHARS or _UNSAFE_PUBLIC.search(custom):
                raise DecisionValidationError("custom approach is invalid")
            return DecisionResolution(
                character_id=character_id,
                kind=draft.kind,
                answer_kind="custom",
                selection_id="own_approach",
                custom_text=custom,
            )
        if submission.custom_text:
            raise DecisionValidationError("listed clarification paths do not take custom text")
        return DecisionResolution(
            character_id=character_id,
            kind=draft.kind,
            answer_kind="option",
            selection_id=submission.selection_id,
        )
    if submission.custom_text:
        raise DecisionValidationError("approach choices do not take custom text")
    approach = next((item for item in draft.approaches if item.id == submission.selection_id), None)
    if approach is None:
        raise DecisionValidationError("approach selection is unavailable")
    # The coordinator receives ``character_id`` only after principal and token checks.
    # Planner proposals carry no actor identity, so they cannot bind another player.
    proposal = approach.directive
    if proposal.kind == "no_test":
        directive: ResolutionDirective = NoTestDirective(
            kind="no_test", character_id=character_id
        )
    elif proposal.kind == "attribute_test":
        directive = AttributeTestDirective(
            kind="attribute_test", character_id=character_id, attribute=proposal.attribute
        )
    else:
        directive = CombatDefendDirective(
            kind="combat_defend", character_id=character_id, method=proposal.method
        )
    return DecisionResolution(
        character_id=character_id,
        kind=draft.kind,
        answer_kind="option",
        selection_id=approach.id,
        directive=directive,
    )


def planning_prompt(
    declaration: str,
    canon: str,
    eligible_characters: tuple[tuple[str, str], ...],
    prior_resolutions: tuple[DecisionResolution, ...],
    *,
    effective_action: str = "",
    closed_dimensions: tuple[str, ...] = (),
    prior_questions: tuple[AskedQuestion, ...] = (),
    interaction_cue: InteractionCue | None = None,
    recent_narration: str = "",
) -> str:
    """Build the planner input without account IDs or decision lifecycle fields.
    """
    planner_canon = "\n".join(
        line
        for line in canon.splitlines()
        if not re.match(r"^(updated_at|created_at|event_seq|revision):", line.strip())
    )
    resolutions = json.dumps(
        [
            resolution.model_dump(
                exclude={"directive", "character_id", "action_fingerprint"}
            )
            for resolution in prior_resolutions
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    # The block is emitted only when this chain has actually had a question answered.
    # An empty history renders nothing at all -- no heading, no empty list, no blank
    # line -- so a planner call with nothing to report reads a prompt byte-identical to
    # the one that shipped before this milestone. That matters more than prompt-shape
    # uniformity: the first planner call of every turn in every scenario carries an
    # empty history, and an earlier version of this function emitted the heading and
    # the instruction unconditionally, which perturbed the planner's input on every
    # call in the system for a feature that applies only after a question is answered.
    # The instruction's closing sentence is the reason that was not harmless -- "Return
    # proceed unless a genuinely new decision is required" is a live nudge whether or
    # not the list above it is empty.
    answered_block = ""
    if prior_questions:
        answered_questions = json.dumps(
            [
                {"question": item.question, "options": list(item.labels)}
                for item in prior_questions
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        answered_block = (
            f"Questions already asked and answered in this chain:\n{answered_questions}\n\n"
            "Never ask any of those questions again, in any wording and with any reordering "
            "of its options: they are answered. Return proceed unless a genuinely new "
            "decision is required.\n\n"
        )
    # Same discipline ``answered_block`` documents above: rendered only when the field
    # applies, so a turn with no tracked focus (the overwhelming majority of calls)
    # leaves the prompt byte-identical to the one that shipped before this fix.
    interaction_block = ""
    if interaction_cue is not None:
        public_id = interaction_cue.public_npc_id or "none"
        interaction_block = (
            f"Active in-world interlocutor:\nkind: {interaction_cue.kind}\n"
            f"public_npc_id: {public_id}\n"
            "The player is already engaged with this interlocutor. Do not claim no NPC "
            "is present, or that the declaration lacks a target or an owner, when this "
            "interlocutor is set -- direct a clarification at resolving what they do or "
            "say next, not at whether they exist.\n\n"
        )


    recent_block = ""
    if recent_narration:
        recent_block = (
            f"Most recently delivered narration (not necessarily committed to the scene "
            f"record yet):\n{recent_narration.strip()}\n\n"
            "Treat this as established fact when drafting a clarification: do not claim "
            "the scene record omits something this narration already states.\n\n"
            "Before drafting any clarification or approach question, check whether this "
            "narration already reports a roll or check for the same goal the declaration "
            "pursues. If it does, that attempt already happened and does not repeat: do "
            "not draft a question re-asking how to attempt it. Example: the narration "
            "reports \"Rill rolls STR: 13 vs 12, failure\" while trying to free a stuck "
            "object, and the declaration is a fresh attempt at that same object -- the "
            "correct response acknowledges the failure (or proceeds), never a fresh \"how "
            "do you attempt it\" question as if no attempt had occurred.\n\n"
        )
    return (
        "Plan before narration or tools. Return proceed when no defined decision is needed. "
        "Request confirmation only for a high-stakes commitment. Request clarification "
        "when the declared action is ambiguous, with listed paths. Request approach only "
        "when an option must bind the next test or defence, and only when its listed "
        "paths themselves differ in what they bind -- a different attribute, a "
        "different risk, or a different effect. Paths that would resolve identically "
        "are flavour, not a decision: narrate the choice yourself and proceed. Never "
        "infer account identities.\n\n"
        "Audience: use only current or party. Never name an actor, account, or token.\n\n"
        f"Canon:\n{planner_canon}\n\n"
        f"{interaction_block}"
        f"{recent_block}"
        f"Player declaration:\n{declaration}\n\n"
        f"Effective action:\n{effective_action or declaration}\n\n"
        f"Prior typed resolutions:\n{resolutions}\n\n"
        f"Closed dimensions:\n{json.dumps(sorted(closed_dimensions))}\n\n"
        f"{answered_block}"
        "Use the schema exactly. Clarification views add their own-approach path outside "
        "your listed paths. Directives name only a mechanic and never an actor."
    )


def continuation_text(resolutions: tuple[DecisionResolution, ...]) -> str:
    """Render typed answers for the narrator without lifecycle or identity secrets."""
    if not resolutions:
        return ""
    return json.dumps(
        [resolution.model_dump() for resolution in resolutions],
        ensure_ascii=False,
        sort_keys=True,
    )


def hazard_obligation_text(resolutions: tuple[DecisionResolution, ...]) -> str:
    """State the roll a confirmed violent action owes, so withholding stays the backstop.

    The guard withholds a confirmed attack that narrates without rolling. This tells the
    narrator to roll before narrating, so the guard fires only on a genuine miss rather
    than on every confirmed attack.
    """
    actors = sorted(
        {
            resolution.directive.character_id
            for resolution in resolutions
            if isinstance(resolution.directive, HazardResolutionDirective)
        }
    )
    if not actors:
        return ""
    return (
        "\n\nConfirmed violent action:\n"
        f"actor(s): {', '.join(actors)}\n"
        "Resolve the attack through the combat tools before narrating its outcome. Open "
        "combat if none is running, close range if needed, then call combat_attack. Narrate "
        "only what the roll returns; the turn is withheld if no roll resolves the action."
    )


def defence_obligation_text(resolutions: tuple[DecisionResolution, ...]) -> str:
    """State the defence a bound dodge or parry owes, so withholding stays the backstop.

    ``ResolutionGuard`` withholds a turn whose bound ``combat_defend`` never runs. The
    service binds this directive from the player's own one-word answer. No decision view
    therefore exists to carry the obligation into the prompt, the way a presented approach
    does. Stating it here keeps the guard a backstop against a defence narrated without a
    roll, rather than the routine reply to a player who typed "dodge".

    It names the incoming damage requirement for a measured reason.
    ``GameService.combat_defend`` raises ``incoming_damage_required`` when a call omits
    both an NPC attacker and a damage value. A narrator learning that from a refusal
    spends one call discovering it.
    """
    bound = sorted(
        {
            (resolution.directive.character_id, resolution.directive.method)
            for resolution in resolutions
            if isinstance(resolution.directive, CombatDefendDirective)
        }
    )
    if not bound:
        return ""
    declared = "\n".join(f"{character_id}: {method}" for character_id, method in bound)
    return (
        "\n\nDeclared defence:\n"
        f"{declared}\n"
        "Resolve it with combat_defend for that defender and that method, naming the "
        "attacking NPC or the incoming damage. Narrate only what the roll returns; the "
        "turn is withheld if no defence roll resolves it."
    )


def semantic_fingerprint(parts: dict) -> str:
    """Hash a stable, explicit context payload for stale-decision rejection."""
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


class PlannerPolicyError(ValueError):
    """A schema-valid proposal violated an engine-owned planning policy."""


async def _policy_for(classify, text: str, scene: TrustedScope, combat) -> TurnPolicy | None:
    """One routing verdict for text that is not the turn's own, or ``None``.

    Only ``verify_plan``'s label branch needs this: every other decision reuses the
    single verdict ``NarratorService.run`` already computed for the turn.
    """
    from narrator.classify import policy_from

    classification = await classify(text, scope=scene, combat=combat)
    if classification is None:
        return None
    return policy_from(classification, scope=scene, combat=combat)


def planning_bypass(policy: TurnPolicy | None) -> Literal["factual", "observation", ""]:
    """Classify pure reads without relying on a planner or changing committed actions.

    Takes the turn's already-computed verdict rather than deriving its own. Before the
    model classifier this function ran ``classify_turn`` itself, which meant the same
    player text was classified two to five times per turn (once in the service's turn
    loop, then again here, in ``requires_risk_confirmation``, and once or twice more in
    ``verify_plan``). One classification per turn is now threaded from
    ``NarratorService.run``; a ``None`` policy means the classifier could not answer.

    The observation/factual split stays here because it selects between two engine-owned
    bypass behaviors rather than reading the player's intent -- the classifier already
    decided this turn only reads the scene.
    """
    if policy is None or policy.route != "read":
        return ""
    return "observation" if policy.reads_by_observing else "factual"


def requires_risk_confirmation(policy: TurnPolicy | None) -> bool:
    """Apply the engine-owned inclusive risk floor before narration may proceed.

    Factual and passive declarations never reach this function because ``planning_bypass``
    accepts them first. A remaining declaration directed at a known non-hostile person is
    a commitment, even when custom wording omits a known violence or theft verb. This
    conservative floor prevents a planner from downgrading novel phrasing into narration.

    A ``None`` policy -- the classifier could not answer -- requires the confirmation.
    That is the fail-closed direction: the engine does not know what this turn declared,
    so it does not get to assume the turn was harmless. This gate only decides whether
    the floor runs at all; ``verify_plan`` decides what the floor does.
    """
    return policy is None or policy.route == "risk"


def _catalog(catalog=None):
    """The catalog to read player-facing text from, English when none is threaded.

    Every caller that has a config passes its catalog. The English default keeps the
    module importable and keeps a test that builds a draft directly from working, and
    it is the same catalog ``locale/en/narrator.yaml`` ships -- not a second copy of
    the wording, which is the drift this whole change removes.
    """
    if catalog is not None:
        return catalog
    from pathlib import Path

    from narrator.locale import load

    return load("en", Path(__file__).resolve().parents[2] / "locale")


def risk_confirmation(category: str = "", catalog=None) -> ConfirmationDraft:
    """Create the engine-authored confirmation that the planner cannot weaken.

    A violence confirmation binds the action to a resolving mechanic, so a confirmed
    attack rolls or the turn withholds. Theft and destruction do not bind, because their
    mechanic is not universal.
    """
    return ConfirmationDraft(
        kind="confirmation",
        question=_catalog(catalog).confirmation(category),
        audience=CurrentAudience(kind="current"),
        binds_hazard="violence" if category == "violence" else "",
    )


def enemy_turn_defence(enemy_name: str, character_name: str, catalog=None) -> ApproachDraft:
    """Create the engine-authored defence choice for an enemy's own open combat turn.

    No planner call and no custom-text option, for the same reason
    ``_combat_defence_resolution`` needs neither: the rules fix the defence space at
    dodge or parry, and a planner call would add latency and a failure mode while
    contributing no judgment a fixed rule does not already supply.
    """
    return ApproachDraft(
        kind="approach",
        question=_catalog(catalog).text(
            "decisions.enemy_turn_question", enemy=enemy_name, character=character_name
        ),
        context=_catalog(catalog).text(
            "decisions.enemy_turn_context", enemy=enemy_name, character=character_name
        ),
        audience=CurrentAudience(kind="current"),
        approaches=(
            ApproachOption(
                id="parry",
                label=_catalog(catalog).option("parry"),
                directive=CombatDefendProposal(kind="combat_defend", method="parry"),
            ),
            ApproachOption(
                id="dodge",
                label=_catalog(catalog).option("dodge"),
                directive=CombatDefendProposal(kind="combat_defend", method="dodge"),
            ),
        ),
    )


@dataclass(frozen=True)
class AskedQuestion:
    """One narrowing question a player already answered in this decision chain.
    """

    question: str
    labels: tuple[str, ...]


def remember_question(
    asked: tuple[AskedQuestion, ...], draft: DecisionDraft
) -> tuple[AskedQuestion, ...]:
    """Append one answered narrowing draft to a chain's question history.

    A ``ConfirmationDraft`` is never recorded. A confirmation is a yes/no gate rather
    than a narrowing question: the risk floor re-authors the identical wording for every
    unconfirmed hazardous action (``risk_confirmation``), so recording it would make the
    next genuine hazard read as a repeat, and the duplicate guard would then wave that
    hazard through unconfirmed. ``action_is_confirmed`` and ``action_is_declined``
    already bound how often one confirmation can be asked.
    """
    if isinstance(draft, ConfirmationDraft):
        return asked
    labels = tuple(option.label for option in public_options(draft) if not option.custom)
    return (asked + (AskedQuestion(question=draft.question, labels=labels),))[
        -MAX_ASKED_QUESTIONS:
    ]


@dataclass(frozen=True)
class ProgressiveDecisionSession:
    """Retain one pre-action session across planner calls and player views."""

    original_action: str
    effective_action: str
    resolved_dimensions: tuple[tuple[str, str], ...]
    closed_dimensions: frozenset[str]
    action_fingerprint: str
    lifecycle: Literal["planning", "collecting", "ready", "recovery", "segment"]
    completed_views: int
    round_index: int
    confirmation_fingerprint: str = ""
    confirmed_action_fingerprint: str = ""
    declined_action_fingerprint: str = ""


    asked_questions: tuple[AskedQuestion, ...] = ()
    #: Planner-authored labels describe a choice, not a new player declaration.
    #: ``verify_plan`` uses this flag to assess hazards from the player's words.
    #: Custom answers and revised actions reset it.
    effective_from_label: bool = False

    @classmethod
    def start(cls, action: str) -> ProgressiveDecisionSession:
        return cls._build(action, action, (), frozenset(), "planning", 0, 0)

    @classmethod
    def _build(
        cls,
        original_action: str,
        effective_action: str,
        resolved_dimensions: tuple[tuple[str, str], ...],
        closed_dimensions: frozenset[str],
        lifecycle: Literal["planning", "collecting", "ready", "recovery", "segment"],
        completed_views: int,
        round_index: int,
        confirmation_fingerprint: str = "",
        confirmed_action_fingerprint: str = "",
        declined_action_fingerprint: str = "",
        asked_questions: tuple[AskedQuestion, ...] = (),
        effective_from_label: bool = False,
    ) -> ProgressiveDecisionSession:
        fingerprint = semantic_fingerprint(
            {
                "original_action": original_action,
                "effective_action": effective_action,
                "resolved_dimensions": tuple(
                    item for item in resolved_dimensions if item[0] != "confirmation"
                ),
            }
        )
        return cls(
            original_action=original_action,
            effective_action=effective_action,
            resolved_dimensions=resolved_dimensions,
            closed_dimensions=closed_dimensions,
            action_fingerprint=fingerprint,
            lifecycle=lifecycle,
            completed_views=completed_views,
            round_index=round_index,
            confirmation_fingerprint=confirmation_fingerprint,
            confirmed_action_fingerprint=confirmed_action_fingerprint,
            declined_action_fingerprint=declined_action_fingerprint,
            asked_questions=asked_questions,
            effective_from_label=effective_from_label,
        )

    def preparing(self, draft: DecisionDraft) -> ProgressiveDecisionSession:
        """Mark a current draft, freezing the exact action confirmation may approve."""
        confirmation = self.action_fingerprint if isinstance(draft, ConfirmationDraft) else ""
        return replace(
            self,
            lifecycle="collecting",
            round_index=self.round_index + 1,
            confirmation_fingerprint=confirmation,
        )

    def accept(
        self, draft: DecisionDraft, resolutions: tuple[DecisionResolution, ...]
    ) -> ProgressiveDecisionSession:
        """Close answered dimensions and invalidate dependents after an action change."""
        if isinstance(draft, ConfirmationDraft) and self.confirmation_fingerprint != self.action_fingerprint:
            raise PlannerPolicyError("confirmation no longer matches the current action")
        dimension = draft.dimension
        values = tuple((dimension, item.selection_id) for item in resolutions)
        custom_changed = any(item.answer_kind == "custom" for item in resolutions)
        # A selected listed clarification path refines the action, exactly as a custom
        # path does. Without this, a category option ("Examine the environment") left the
        # action unchanged with its only dimension closed, and the next planner proposal
        # repeated the closed dimension and faulted. Refinement turns the follow-up into a
        # question about a new action, which the closed-dimension rule permits.
        clarification_option = isinstance(draft, ClarificationDraft) and any(
            item.answer_kind == "option" for item in resolutions
        )
        changed = custom_changed or clarification_option
        if custom_changed:
            effective = next(
                item.custom_text for item in resolutions if item.answer_kind == "custom"
            )
            # A custom answer is the player's own words, so hazard judgment returns
            # to the ordinary basis.
            effective_from_label = False
        elif clarification_option:
            labels = {option.id: option.label for option in draft.paths}
            selected = next(
                item.selection_id for item in resolutions if item.answer_kind == "option"
            )
            effective = labels.get(selected, self.effective_action)
            # The effective action is now planner-authored meta-language; see the
            # field's own docstring for the live misfire this flag closes.
            effective_from_label = True
        else:
            effective = self.effective_action
            effective_from_label = self.effective_from_label
        if changed:
            retained = tuple(
                item for item in self.resolved_dimensions if item[0] not in {"target", "approach", "confirmation"}
            )
            closed = frozenset(item for item in self.closed_dimensions if item not in {"target", "approach", "confirmation"})
            confirmed = ""
            declined = ""
        else:
            retained = self.resolved_dimensions
            closed = self.closed_dimensions
            confirmed = self.confirmed_action_fingerprint
            declined = self.declined_action_fingerprint
        closed = closed | frozenset({dimension})
        if clarification_option:
            # The refined action may still be ambiguous, so its own dimension reopens.
            # The view budget, not dimension closure, bounds the narrowing chain.
            closed = closed - frozenset({dimension})
        if dimension == "target" and any(value == "random" for _, value in values):
            closed = closed | frozenset({"target"})
        if isinstance(draft, ConfirmationDraft):
            if any(item.answer_kind == "confirm" for item in resolutions):
                confirmed = self.action_fingerprint
            if any(item.answer_kind == "decline" for item in resolutions):
                declined = self.action_fingerprint
        return self._build(
            self.original_action,
            effective,
            retained + values,
            closed,
            "planning",
            self.completed_views + len(resolutions),
            self.round_index,
            confirmed_action_fingerprint=confirmed,
            declined_action_fingerprint=declined,


            asked_questions=remember_question(self.asked_questions, draft),
            effective_from_label=effective_from_label,
        )

    def next_segment(self) -> ProgressiveDecisionSession:
        """Keep valid dimensions while starting an explicitly requested new segment.

        ``completed_views`` resets because a segment boundary is the player explicitly
        asking for another round on the same action. ``asked_questions`` deliberately
        does not: ``replace`` carries it through, so the budget that resets and the
        question history that accumulates are the two halves that together bound a
        clarification chain. Before this, both reset, and the same question could be
        re-asked for as many segments as the player was willing to type into.

        ``declined_action_fingerprint`` resets too. A decline is a verdict on the round
        it was given in, not a permanent block on the action: before this, nothing else
        cleared it short of a changed dimension or a full ``revised``, so one decline
        made every later attempt at the identical action fault forever after, whatever
        the player went on to answer.
        """
        return replace(
            self,
            lifecycle="segment",
            completed_views=0,
            round_index=0,
            confirmation_fingerprint="",
            declined_action_fingerprint="",
        )

    def revised(self, action: str) -> ProgressiveDecisionSession:
        """Start a new action after an explicit recovery revision.

        A revised declaration cannot inherit confirmations, choices, or capacity from
        the action it replaces.  Continuing and retrying use ``next_segment`` instead,
        which deliberately retain valid typed state for the same action.

        It cannot inherit the question history either: ``_build`` starts from an empty
        ``asked_questions``, so a question that was fair to ask about the replaced action
        stays askable about the new one.
        """
        revised_action = action.strip()
        if not revised_action:
            raise PlannerPolicyError("a revised action is required")
        return self._build(revised_action, revised_action, (), frozenset(), "planning", 0, 0)

    @property
    def action_is_confirmed(self) -> bool:
        """Whether a player confirmed this exact effective action."""
        return self.confirmed_action_fingerprint == self.action_fingerprint

    @property
    def action_is_declined(self) -> bool:
        """Whether a player declined this exact effective action."""
        return self.declined_action_fingerprint == self.action_fingerprint


async def verify_plan(
    proposal: PlanOutcome,
    declaration: str,
    session: ProgressiveDecisionSession,
    combat=None,
    scope: TrustedScope | None = None,
    *,
    classify,
    policy: TurnPolicy | None = None,
) -> PlanOutcome | DeclinedPlan:
    """Enforce bypasses, the risk floor, and closed dimensions after every proposal.

    ``policy`` is the turn's already-computed verdict, reused whenever the text being
    judged is the turn's own. It is not always: a planner-authored path label replaces
    ``session.effective_action``, and the label branch below judges the player's original
    words instead. Those two cases are the only ones that spend a second classification,
    which is why ``classify`` is threaded here rather than a policy alone.

    A classification that cannot be obtained resolves to the confirmation, matching
    ``requires_risk_confirmation``: not knowing what a declaration does is not a reason
    to let it through.
    """
    action = session.effective_action or declaration
    scene = hazard_scope(scope, combat)
    if policy is not None and action == declaration:
        pass
    else:
        policy = await _policy_for(classify, action, scene, combat)
    if policy is None:
        return PlanOutcome(
            plan=RequestDecisionPlan(kind="request", decision=risk_confirmation(""))
        )
    if policy.route == "risk" and session.effective_from_label:


        original_policy = await _policy_for(classify, session.original_action, scene, combat)
        if original_policy is not None and original_policy.route != "risk":
            policy = original_policy
    if policy.route in {"social", "read", "out_of_character"}:
        return PlanOutcome(plan=ProceedPlan(kind="proceed"))
    if policy.route == "risk":
        # An open fight already cast the target as an enemy, so attacking a recorded
        # combatant is the mechanic the fight resolves, not an assault on a bystander.
        # The floor proceeds without re-confirming; the guard still requires the roll.
        if combat_sanctions_violence(policy, combat):
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))


        if policy.risk_category in {"violence", "theft"} and named_dead_target(policy):
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))
        if session.action_is_declined:
            return DeclinedPlan(kind="declined")
        if not session.action_is_confirmed:
            return PlanOutcome(
                plan=RequestDecisionPlan(
                    kind="request", decision=risk_confirmation(policy.risk_category)
                )
            )
        if session.action_is_confirmed and isinstance(proposal.plan, RequestDecisionPlan) and isinstance(
            proposal.plan.decision, ConfirmationDraft
        ):
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))
    if isinstance(proposal.plan, RequestDecisionPlan):
        if isinstance(proposal.plan.decision.audience, CharacterAudience):
            raise PlannerPolicyError("planner authored an actor identity")


        if session.action_is_confirmed and isinstance(proposal.plan.decision, ConfirmationDraft):
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))
        if proposal.plan.decision.dimension in session.closed_dimensions:
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))
    return proposal


def planner_repair_prompt(prompt: str) -> str:
    """Request one same-snapshot correction without carrying model diagnostics forward."""
    return (
        f"{prompt}\n\nReturn one schema-valid plan that obeys the decision policy. "
        "Do not include identities, tokens, or actor bindings in directives."
    )


def escalation_prompt(declaration: str, canon_text: str) -> str:
    """Escalation prompt.
    """
    return (
        f"Canon:\n{canon_text}\n\n"
        f"Player declaration:\n{declaration}\n\n"
        f"Effective action:\n{declaration}\n\n"
        "This declaration was already flagged as needing a mechanical check. "
        "Provide the approach now."
    )
