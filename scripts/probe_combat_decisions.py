#!/usr/bin/env python3
"""Write redacted evidence that a bound combat directive reaches its mechanic.

Two obligations in this repository tell the narrator to roll before it narrates, and
``ResolutionGuard`` withholds the whole turn when the roll never happens. Both are prompt
text, so no test using a fake engine can say whether the served model obeys them. This
probe answers that question against the live endpoint and retains the answer.

The attack scenario declares \"I attack the reed thug\" against the fight's own recorded
sole combatant, with the player's turn already open -- the state the engine now
guarantees, since turns open themselves -- so the run measures the sanction and the
resolving ``combat_attack`` alone. Before
milestone M14, ``combat_sanctions_violence``'s residual-word subtraction left the leading
pronoun \"i\" unrecognized, so this exact declaration always withheld the fight's sanction
and presented a confirmation -- which is what first exercised ``hazard_obligation_text``
(shipped in commit ``de8a00c8d17c258dea2ae845454b5c0a6176f024`` with no live measurement
of its own). M14 closes that gap: the identical declaration against the fight's own
recorded, engaged combatant now sanctions immediately, with no confirmation and no round
trip. ``hazard_obligation_text``'s live declare-confirm-resolve coverage therefore moves to
the new-engagement-confirms scenario below, which retargets the identical obligation at an
engagement the fight does not yet record, so M14's own fix does not sanction it away.

The weapon-named-sanction and ooc-question-in-fight scenarios cover milestone M13. The
first declares violence against the fight's own recorded enemy while naming the weapon,
which before M13 withheld the fight's sanction on the residual word \"dagger\" and
re-triggered the confirmation. The second sends an out-of-character question during that
same open fight, which before M13 the hazard branch claimed for the violence verb the
question quotes; it must be answered rather than resolved as a physical event, and the
prompt's turn framing rather than any offline check is what carries that.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsh_mcp.models import NPC
from bsh_mcp.store import CampaignStore
from bsh_mcp.testing import bootstrap_probe_campaign, candidate_tree, resolve_model_and_digest
from narrator import ledger
from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.config import NarratorConfig
from narrator.decisions import DecisionSubmission, risk_confirmation
from narrator.delivery import DECISION_INCOMPLETE_NOTICE
from narrator.engine import (
    announced_rolls,
    normalize_for_similarity,
    roll_facts,
    verdict_mismatches,
)
from narrator.service_assembly import build_narrator_service

SCENARIOS = (
    "defence",
    "attack",
    "refusal-account",
    "unlisted-violence",
    "sanctioned-attack",
    "disengage",
    "ledger-gate-recovery",
    "clarification-converges",
    "new-declaration-heard",
    "withheld-recovery-armed",
    "no-phantom-scene",
    "mechanical-truth",
    "weapon-named-sanction",
    "ooc-question-in-fight",
    "new-engagement-confirms",


    "maneuver-direct-ruling",
    "outlandish-priced-not-granted",
)

#: The disengage and ledger-gate-recovery scenarios both open at a real authored
#: location instead of the other scenarios' placeholder ``docks``, so the narrator's
#: canon digest carries this location's real ``exits`` list
#: (``the-drowned-customs-house``, ``the-road-shrine``) and NPC prose (the fishmonger
#: Rade) rather than an invented one.
_DISENGAGE_START_LOCATION = "the-eel-market"


_M10_SCENARIOS = (
    "clarification-converges",
    "new-declaration-heard",
    "withheld-recovery-armed",
    "no-phantom-scene",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _container_image() -> dict:
    command = ["docker", "inspect", "--format", "{{.Config.Image}} {{.Image}}", "vllm-gemma"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    image, digest = result.stdout.strip().split(maxsplit=1)
    return {"image": image, "image_digest": digest}


def _prepare_campaign(root: Path, scenario: str) -> str:
    """Create one disposable campaign, with an open fight at close range for most scenarios.

    The defence and attack scenarios differ only in who holds the active turn. The
    defence scenario gives it to the enemy, which is when Black Sword Hack asks the
    target to parry or dodge. The attack scenario gives it to the player with the turn
    still closed, so a landed strike requires ``combat_begin_turn`` before
    ``combat_attack``. The unlisted-violence and sanctioned-attack scenarios reuse the
    attack scenario's exact setup; only their declared text and adapter capabilities
    differ. Milestone M13's two scenarios (weapon-named-sanction, ooc-question-in-fight)
    reuse it for the same reason: both need the fight open and the enemy recorded, and
    both differ from sanctioned-attack only in the text the player declares. The refusal-account scenario needs no open fight: the risk floor withholds
    its first turn before any tool call, purely on the adapter's declared capabilities,
    and its second turn is out-of-character narration. Milestone M14's
    new-engagement-confirms scenario reuses that exact same shape -- the reed thug NPC
    present but no fight open -- for the opposite reason: it needs the risk floor to still
    present its confirmation for a declaration M14's own fix would otherwise sanction away
    were a fight already open and recording that NPC. The disengage scenario also
    reuses the attack scenario's fight setup, but at a real authored location instead
    of the placeholder ``docks``, so the narrator sees a real ``exits`` list to flee
    toward. The ledger-gate-recovery scenario needs neither an NPC nor a fight prepared
    at all: unlike every fight-based scenario above, its whole premise is that the
    narrator must create its own target from scratch mid-turn, exactly as the live
    reproduction did, so pre-seeding one here would test a different, easier path.
    Milestone M10's four scenarios take that same shape for the same reason -- see
    ``_M10_SCENARIOS`` for why an open fight would defeat three of them outright.
    """
    game, character_id = bootstrap_probe_campaign(root, title="Probe", weapons=["long knife"])
    at_eel_market = scenario in ("disengage", "ledger-gate-recovery") or scenario in _M10_SCENARIOS
    location_id = _DISENGAGE_START_LOCATION if at_eel_market else "docks"
    with game.store.transaction("probe", character_id, "setup") as transaction:
        if scenario != "ledger-gate-recovery" and scenario not in _M10_SCENARIOS:
            transaction.state.npcs["reed-thug"] = NPC(
                id="reed-thug", name="Reed Thug", level=1, hp=4, hp_max=4, damage=3,
                location_id=location_id,
            )
            transaction.state.scene.present_npcs = ["reed-thug"]
        transaction.state.scene.location_id = location_id
        transaction.state.scene.title = "The Eel Market" if at_eel_market else "Docks"
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_setup"})
    if (
        scenario in ("refusal-account", "ledger-gate-recovery", "new-engagement-confirms")
        or scenario in _M10_SCENARIOS
    ):
        (root / "campaign" / "players.yaml").write_text(
            "players:\n- discord_user_id: terminal-player\n  character_id: "
            f"{character_id}\n  display_name: Rill\n",
            encoding="utf-8",
        )
        return character_id
    opened = game.combat_start(
        pc_ids=[character_id],
        npc_ids=["reed-thug"],
        initial_ranges={"reed-thug": "close"},
        reason="probe",
    )
    if not opened.get("ok"):
        raise ValueError("probe fight did not open")
    # Initiative is rolled, so the scenario's premise is set explicitly rather than
    # assumed. Writing it here keeps the measurement about the obligation text. The
    # holder's turn is left OPEN, matching what the engine now guarantees in
    # production: combat_start opens the first turn and every advance opens the next,
    # so a closed active turn is a state live play can no longer produce.
    holder = "reed-thug" if scenario == "defence" else character_id
    with game.store.transaction("probe", character_id, "arrange") as transaction:
        combat = transaction.state.combat
        combat.order = [holder] + [item for item in combat.order if item != holder]
        combat.active_actor = holder
        for actor_id, actor in combat.actors.items():
            actor.turn_open = actor_id == holder
            actor.actions_used = 0
            actor.actions_taken = []
            actor.actions_max = 1 if actor.side == "npc" else 2
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_arranged"})
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )
    return character_id


class _RuntimeProbeAdapter:
    """Supply one authenticated declaration while retaining only output categories."""

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(
        self, declaration: str, *, capabilities: ChannelCapabilities | None = None
    ) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._declaration = declaration
        if capabilities is not None:


            self.decision_capabilities = capabilities
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.acknowledgements = 0
        self.completed_tools: set[str] = set()
        # ``NarratorService._record_tool_events`` forwards each executed tool's redacted
        # disposition to this recorder. That disposition is the only signal that says a
        # mechanic completed. ``NarratorEngine._tool_log`` cannot answer the question.
        # It appends the name in the before-call hook, ahead of the ceiling check and the
        # guard, so a cancelled or raising call still lands there.
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            self.completed_tools.add(str(fields.get("tool", "")))

    async def turns(self):
        yield InboundTurn("docks", ChannelMessage("Rill", self._declaration, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id, text
        self.output_categories.append("narration")

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result
        self.acknowledgements += 1


class _DisengageAdapter:
    """A fight the player disengages from, then one targeted in-character follow-up.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.completed_tools: set[str] = set()
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            self.completed_tools.add(str(fields.get("tool", "")))

    async def turns(self):
        yield InboundTurn(
            "docks",
            ChannelMessage(
                "Rill",
                "Rill slips clear of Reed Thug entirely and is already running hard down "
                "the road shrine path -- there is no catching her now, this skirmish is "
                "finished.",
                self._principal,
            ),
        )
        yield InboundTurn(
            "docks",
            ChannelMessage(
                "Rill",
                "GM, straight answer: is the skirmish over, and where did we end up?",
                self._principal,
            ),
        )

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id, text
        self.output_categories.append("narration")

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


#: Case-insensitive stems the delivered reply must carry at least one of, to count as
#: naming the true withhold reason rather than an invented one. Drawn from
#: ``NarratorConfig.risk_confirmation_notice`` itself: "No action occurred because this
#: hazard requires an authenticated confirmation." The injected engine note in
#: ``narrator.prompt.turn_prompt`` asks the model to state the reason, not paraphrase
#: it, so a compliant reply carries one of these words.
_REFUSAL_REASON_KEYWORDS: tuple[str, ...] = ("hazard", "confirmation", "authenticat")


class _RefusalAccountAdapter:
    """A withheld hazard turn, then an out-of-character question about why.

    Model text never survives past this class: ``post`` reduces each reply to typed
    booleans and discards the string, matching the redaction discipline every other
    probe adapter in this module keeps.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=False, atomic_decision_delivery=False
    )

    def __init__(self, expected_withhold_notice: str) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._expected_notice = expected_withhold_notice
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.completed_tools: set[str] = set()
        self.withheld_notice_matched = False
        self.reply_names_true_reason = False
        self._posts = 0
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            self.completed_tools.add(str(fields.get("tool", "")))

    async def turns(self):
        yield InboundTurn("docks", ChannelMessage("Rill", "I attack the reed thug", self._principal))
        yield InboundTurn("docks", ChannelMessage("Rill", "GM, why was I refused?", self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self._posts += 1
        if self._posts == 1:
            self.withheld_notice_matched = text == self._expected_notice
            self.output_categories.append("notice")
            return
        normalized = text.casefold()
        self.reply_names_true_reason = any(
            keyword in normalized for keyword in _REFUSAL_REASON_KEYWORDS
        )
        self.output_categories.append("narration")

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


class _LedgerGateRecoveryAdapter:
    """Declare a combat-opening attack against an uncreated NPC, twice, back to back.

    Three things this adapter retains beyond every other probe adapter in this
    module, all typed and none raw model text: ``decision_events``, the redacted
    ``decision_lifecycle`` fields ``NarratorService._record_decision`` already
    reduces a decision-phase outcome to (never narration), which is what proves or
    disproves a ``ledger_gate`` fault; a three-way post classification (``notice``,
    ``incomplete_notice``, ``narration``) against the two known engine-authored
    strings that could name this exact failure; and ``tools_by_turn``, one set of
    successful tool names per turn, attributed to the turn whose own tool calls
    produced them and never pooled across turns. That per-turn attribution matters:
    a legitimate ``notice`` (the plain \"no action occurred\") on a turn whose own
    tool calls genuinely changed nothing is not the false-silence defect this
    milestone fixed, and only a same-turn correlation between the posted notice and
    that turn's own tool successes can tell the two apart.
    ``NarratorService._record_tool_events`` always runs before ``deliver`` for the
    same turn (``NarratorService.run``), so each ``post`` call sees exactly the
    tool successes its own turn accumulated since the previous ``post``.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )


    _DECLARATION = "I draw my knife and attack the fishmonger Rade"

    def __init__(self, decision_fault_notice: str, decision_incomplete_notice: str) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._decision_fault_notice = decision_fault_notice
        self._decision_incomplete_notice = decision_incomplete_notice
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.completed_tools: set[str] = set()
        self.decision_events: list[dict] = []
        #: One entry per ``post`` call, holding exactly the tool names that
        #: succeeded since the previous ``post`` -- that turn's own contribution,
        #: never a cross-turn total.
        self.tools_by_turn: list[set] = []
        self._tools_since_last_post: set = set()
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            tool = str(fields.get("tool", ""))
            self.completed_tools.add(tool)
            self._tools_since_last_post.add(tool)
        elif event == "decision_lifecycle":
            self.decision_events.append(
                {
                    "category": str(fields.get("category", "")),
                    "failure_category": str(fields.get("failure_category", "")),
                }
            )

    async def turns(self):
        yield InboundTurn("docks", ChannelMessage("Rill", self._DECLARATION, self._principal))
        yield InboundTurn("docks", ChannelMessage("Rill", self._DECLARATION, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.tools_by_turn.append(set(self._tools_since_last_post))
        self._tools_since_last_post = set()
        if text == self._decision_fault_notice:
            self.output_categories.append("notice")
        elif text == self._decision_incomplete_notice:
            self.output_categories.append("incomplete_notice")
        else:
            self.output_categories.append("narration")

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


#: Case-insensitive stems that say a reply is about milestone M10's second declaration
#: rather than the attack it replaced. Drawn from the declaration itself
#: (``_NewDeclarationAdapter._SECOND``), the same way ``_REFUSAL_REASON_KEYWORDS`` is
#: drawn from the notice it checks.
_NEW_DECLARATION_KEYWORDS: tuple[str, ...] = ("coin", "crate", "count", "purse")

#: How a notice spells a recovery control. Duplicated from
#: ``narrator.delivery._ADVERTISED_CONTROLS`` rather than imported, on purpose: this
#: probe measures whether production arms what production advertises, and a check that
#: read production's own definition of "advertises" would move whenever that definition
#: moved. Written out here, a notice that stopped naming a control fails the scenario
#: instead of silently redefining it.
_RECOVERY_CONTROL_MARKERS: tuple[str, ...] = ("/continue", "/dismiss", "/retry", "/revise")


def _names_a_recovery_control(text: str) -> bool:
    return any(marker in text for marker in _RECOVERY_CONTROL_MARKERS)


class _ClarificationConvergenceAdapter:
    """Declare something ambiguous, then answer every view the planner presents.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    _DECLARATION = "I carefully cross the mud to the drowned customs house"

    def __init__(self, notices: tuple[str, ...]) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._notices = tuple(notices)
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.completed_tools: set[str] = set()
        self.decision_events: list[dict] = []
        self.views_presented = 0
        self.answers_given = 0
        self.repeated_question_count = 0
        self._question_digests: list[str] = []
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            self.completed_tools.add(str(fields.get("tool", "")))
        elif event == "decision_lifecycle":
            self.decision_events.append(
                {
                    "category": str(fields.get("category", "")),
                    "failure_category": str(fields.get("failure_category", "")),
                }
            )

    async def turns(self):
        yield InboundTurn("docks", ChannelMessage("Rill", self._DECLARATION, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.output_categories.append("notice" if text in self._notices else "narration")

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        for view in rendered:
            self.decision_kinds.append(view.kind)
            self.views_presented += 1
            digest = _sha256(normalize_for_similarity(view.question).encode())
            if digest in self._question_digests:
                self.repeated_question_count += 1
            self._question_digests.append(digest)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        self.answers_given += 1
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


class _NewDeclarationAdapter:
    """Withhold a turn deterministically, then type something else with no control.

    Turn two declares something with no violence in it and types no recovery control at
    all: this class deliberately exposes no ``consume_decision_recovery_control``, which
    is what a player does when they read a notice and simply say something else.

    The discriminator is typed, not textual, and it is deliberately narrow. The retained
    action was an unconfirmed attack on a non-combatant, so a service that adopted it
    runs the risk floor again and presents ``risk_confirmation``'s one fixed question on
    turn two. Counting *any* decision presented on turn two would be wrong, and was
    measured wrong: the planner may legitimately ask its own question about counting
    coins in a market whose canon posts constables, and one run in three did.
    ``risk_confirmation`` authors a single engine-owned string that only a violent
    declaration reaches, so matching it exactly separates \"the abandoned attack drove
    this turn\" from \"the planner had a question about the new action\". That string is
    engine-authored configuration, known before the run starts and never model output,
    so holding it keeps this adapter's no-prose discipline -- the same reasoning
    ``_RefusalAccountAdapter`` uses for ``config.risk_confirmation_notice``.

    ``risk_confirmations_by_turn`` attributes each one to the turn that presented it,
    never pooling them, the way ``_LedgerGateRecoveryAdapter.tools_by_turn`` does for
    tool successes. One reduced boolean corroborates it, and no reply text is retained.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    _FIRST = "I draw my knife and attack the fishmonger Rade"
    _SECOND = "I sit down on the crates and count out my coins"

    def __init__(self) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.completed_tools: set[str] = set()
        #: One entry per ``post``, counting exactly the risk-floor confirmations
        #: presented since the previous ``post`` -- that turn's own, never a total.
        self.risk_confirmations_by_turn: list[int] = []
        self.reply_names_new_action = False


        self._risk_questions = {
            risk_confirmation(category).question
            for category in ("", "violence", "theft", "destructive")
        }
        self._risk_confirmations_since_last_post = 0
        self._views_refused = 0
        self._posts = 0
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            self.completed_tools.add(str(fields.get("tool", "")))

    async def turns(self):
        yield InboundTurn("docks", ChannelMessage("Rill", self._FIRST, self._principal))
        yield InboundTurn("docks", ChannelMessage("Rill", self._SECOND, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.risk_confirmations_by_turn.append(self._risk_confirmations_since_last_post)
        self._risk_confirmations_since_last_post = 0
        self._posts += 1
        if self._posts == 1:
            self.output_categories.append("notice")
            return
        normalized = text.casefold()
        self.reply_names_new_action = any(
            keyword in normalized for keyword in _NEW_DECLARATION_KEYWORDS
        )
        self.output_categories.append("narration")

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        for view in rendered:
            self.decision_kinds.append(view.kind)
            if view.question in self._risk_questions:
                self._risk_confirmations_since_last_post += 1
        if self._views_refused == 0:
            # One refusal, once: it is what makes turn one withhold without asking the
            # model to cooperate. Every later view is delivered normally, so a second
            # confirmation on turn two would be presented rather than swallowed here.
            self._views_refused += 1
            return DecisionDeliveryReceipt(status="unavailable", decision_count=0)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


class _WithheldRecoveryAdapter:
    """Confirm an attack whose resolving roll may not happen, twice, and watch delivery.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    _DECLARATION = "I draw my knife and attack the fishmonger Rade"

    def __init__(self) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.completed_tools: set[str] = set()
        self.model_scene_commits = 0
        self.control_notices = 0
        self.unarmed_control_notices = 0
        self.recovery_arms = 0
        self._notice_awaiting_arm = False
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            tool = str(fields.get("tool", ""))
            self.completed_tools.add(tool)
            if tool == "scene_commit":
                self.model_scene_commits += 1

    async def turns(self):
        yield InboundTurn("docks", ChannelMessage("Rill", self._DECLARATION, self._principal))
        yield InboundTurn("docks", ChannelMessage("Rill", self._DECLARATION, self._principal))

    def _resolve_pending_notice(self) -> None:
        if self._notice_awaiting_arm:
            self.unarmed_control_notices += 1
            self._notice_awaiting_arm = False

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self._resolve_pending_notice()
        if _names_a_recovery_control(text):
            self.control_notices += 1
            self._notice_awaiting_arm = True
            self.output_categories.append("control_notice")
            return
        self.output_categories.append("narration")

    async def decision_recovery(self) -> None:
        self.recovery_arms += 1
        self._notice_awaiting_arm = False

    async def close(self) -> None:
        self._resolve_pending_notice()

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


class _MechanicalTruthAdapter:
    """Two roll-driving declarations, retaining announced numbers and no prose.

    ``post`` reduces each reply to the typed fields ``narrator.engine.announced_rolls``
    parses out of it -- a name, an attribute, and three numbers per announcement line --
    and discards the string, which is the same redaction every other adapter in this
    module keeps. Those fields are the claim under test, so they are the one thing that
    has to survive; nothing else about the reply does.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        self.completed_tools: set[str] = set()
        self.announced: list[dict] = []
        self.recorder = self

    def record(self, event: str, **fields) -> None:
        if event == "tool_call" and fields.get("ok") is True:
            self.completed_tools.add(str(fields.get("tool", "")))

    async def turns(self):
        yield InboundTurn(
            "docks",
            ChannelMessage("Rill", "I attack the reed thug", self._principal),
        )
        yield InboundTurn(
            "docks",
            ChannelMessage(
                "Rill",
                "Before anything else, I stop and listen hard for anyone else coming "
                "down the alley behind us.",
                self._principal,
            ),
        )

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.output_categories.append("narration")
        self.announced.extend(announced_rolls(text))

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


def _recorded_roll_facts(root: Path) -> tuple[dict, ...]:
    """Every roll this run's audit log recorded, read with the engine's own extractor.

    The requirement's claim is about the tool's own ``outcome`` and ``target``, so the
    measurement reads the record rather than the adapter -- the same discipline the
    disengage and no-phantom-scene scenarios already use, and for the same reason: it
    is immune to a narrator reaching a true state through a call shape this probe did
    not anticipate, and it cannot be satisfied by the guard under test reporting
    favourably on itself.

    ``bsh_mcp.store.Transaction.commit`` writes the same ``roll``, ``outcome``,
    ``attribute``, ``target``, and ``individual`` fields into the audit event that
    ``bsh_mcp.results.success`` puts in the envelope, so ``roll_facts`` reads an event
    unchanged apart from the ``ok`` flag, which is implicit in the event existing at
    all: a refused call commits nothing.
    """
    facts: list[dict] = []
    for event in CampaignStore(root).read_events(limit=200):
        if not isinstance(event, dict) or "tool" not in event:
            continue
        facts.extend(roll_facts(str(event["tool"]), {**event, "ok": True}))
    return tuple(facts)


def _announcement_text(announcements: list[dict]) -> str:
    """Rebuild the announcement lines from their parsed fields, losslessly.

    The adapter keeps no prose, so the comparison cannot re-read the reply. Rebuilding
    the canonical line from the five parsed fields and handing it to the production
    comparator is deliberate: it means this probe runs
    ``narrator.engine.verdict_mismatches`` itself rather than a second copy of the
    matching rule that could drift away from the one shipping in the engine. The round
    trip is lossless for every field the comparator reads, which is every field the
    parser produces.
    """
    return "\n".join(
        "{character} rolls {attribute}: {total} vs {target}, {outcome}.".format(
            character=entry["character"],
            attribute=entry["attribute"],
            total=entry["total"],
            target=entry["target"],
            outcome=str(entry["outcome"]).replace("_", " "),
        )
        for entry in announcements
    )


def scenario_checks(
    scenario: str,
    *,
    turns: int,
    errors: bool,
    delivered: int,
    withheld: int,
    output_categories: list,
    decision_kinds: list,
    attempted: set,
    completed: set,
    withheld_notice_matched: bool = True,
    reply_names_true_reason: bool = True,
    combat_closed: bool = True,
    location_changed: bool = True,
    close_in_audit_log: bool = True,
    ledger_gate_faults: int = 0,
    npc_created: bool = True,
    ledger_ratified_after_run: bool = True,
    false_silence_notice_posted: bool = False,
    views_presented: int = 1,
    answers_given: int = 1,
    repeated_question_count: int = 0,
    segment_faults: int = 0,
    stale_confirmations_after_the_new_declaration: int = 0,
    reply_names_new_action: bool = True,
    control_notices: int = 1,
    unarmed_control_notices: int = 0,
    engine_scene_commits: int = 0,
    turns_withheld: int = 1,
    rolls_recorded: int = 1,
    rolls_announced: int = 1,
    verdict_mismatch_count: int = 0,
    target_mismatch_count: int = 0,
    unbacked_announcement_count: int = 0,
    enemy_holds_first_turn: bool = False,
    maneuver_action_spent: bool = True,
    enemy_unharmed: bool = True,
    no_edge_granted: bool = True,
) -> dict:
    """Map one run's observed state onto this scenario's checks, with no endpoint.

    ``_runtime_flow`` needs a served model, so nothing offline could reach the mapping
    that decides a pass. An audit noted the consequence. Repointing a mechanic check from
    the completed set back to the attempted set would leave the offline suite green, which
    is the exact regression the probe exists to catch. The mapping lives here so
    ``tests_narrator/test_probe_combat_decisions.py`` can drive it directly.

    Mechanic checks read ``completed`` and never ``attempted``. ``combat_defend`` raises
    when a call omits both an NPC attacker and a damage value, when a parry finds no
    weapon, and when a parry meets a ranged attack. ``ResolutionGuard`` marks the
    directive used only when the call actually returns ok (``record_tool_result``), so
    a run of refused defences withholds the turn rather than delivering a narrated hit
    no dice rolled.

    ``withheld_notice_matched`` and ``reply_names_true_reason`` apply only to the
    refusal-account scenario; every other scenario ignores them, so their defaults keep
    the mapping's existing callers valid unchanged. ``combat_closed``, ``location_changed``,
    and ``close_in_audit_log`` apply only to the disengage scenario, read from campaign
    files rather than adapter state, for the same reason. ``ledger_gate_faults``,
    ``npc_created``, ``ledger_ratified_after_run``, and ``false_silence_notice_posted``
    apply only to the ledger-gate-recovery scenario, for the same reason again.

    The last block belongs to milestone M10's four scenarios, one group each:
    ``views_presented``, ``answers_given``, ``repeated_question_count`` and
    ``segment_faults`` to clarification-converges;
    ``stale_confirmations_after_the_new_declaration`` and ``reply_names_new_action`` to
    new-declaration-heard; ``control_notices`` and
    ``unarmed_control_notices`` to withheld-recovery-armed; ``engine_scene_commits`` and
    ``turns_withheld`` to no-phantom-scene. Every default keeps this mapping true, so
    every existing caller stays valid unchanged.

    Milestone M11 adds the last five, all to mechanical-truth: ``rolls_recorded`` and
    ``rolls_announced`` are its two premises, and ``verdict_mismatch_count``,
    ``target_mismatch_count``, and ``unbacked_announcement_count`` are the three shapes
    ``narrator.engine.verdict_mismatches`` distinguishes. They are separate checks
    rather than one total so a failing run names which defect it found.

    ``enemy_holds_first_turn`` applies only to new-engagement-confirms, read from
    ``state.json`` after the run: live initiative decides whether the confirmed attack
    may roll in the opening turn at all, so the check accepts either the rolled attack
    or the rules-mandated deferral to the actor's own turn.
    """
    expected_turns = (
        2
        if scenario in (
            "refusal-account", "disengage", "ledger-gate-recovery",
            "new-declaration-heard", "withheld-recovery-armed", "no-phantom-scene",
            "mechanical-truth",
        )
        else 1
    )
    checks = {
        "service_completed": turns == expected_turns and not errors,
        "turn_delivered": delivered == 1 and withheld == 0,
    }
    if scenario == "defence":
        checks["narration_posted"] = output_categories == ["narration"]
        checks["no_confirmation_presented"] = decision_kinds == []
        checks["defence_completed"] = "combat_defend" in completed
        checks["defence_attempted"] = "combat_defend" in attempted
    elif scenario in ("attack", "unlisted-violence"):


        checks["narration_posted"] = output_categories == ["narration"]
        checks["no_confirmation_presented"] = decision_kinds == []
        # No ``turn_opened`` check any more: the engine now opens every turn itself
        # (``combat_start`` the first, every advance the next), so the player's turn
        # is already open here and ``combat_begin_turn`` is an optional no-op the
        # narrator may skip entirely.
        checks["attack_completed"] = "combat_attack" in completed
    elif scenario == "sanctioned-attack":


        checks["sanctioned_without_confirmation"] = delivered == 1 and withheld == 0
    elif scenario == "weapon-named-sanction":
        # The same typed counters sanctioned-attack reads, for the same reason: they are
        # the service's own, not an adapter's guess at content, and they are what flips
        # between the pre- and post-M13 runs.
        checks["sanctioned_with_weapon_named"] = delivered == 1 and withheld == 0
        checks["no_confirmation_presented"] = decision_kinds == []
    elif scenario == "ooc-question-in-fight":
        checks["question_answered_not_withheld"] = delivered == 1 and withheld == 0
        checks["no_confirmation_presented"] = decision_kinds == []
        # "Answered as a question, not resolved as a physical event." A completed combat
        # mechanic is the observable form of resolving one, and the fight is open and the
        # enemy is named, so every one of these was reachable had the model treated the
        # quoted verb as a declaration. This reads ``completed`` rather than ``attempted``
        # for the reason this mapping's own docstring gives.
        checks["no_physical_event_resolved"] = not (
            {"combat_attack", "combat_begin_turn", "combat_defend"} & set(completed)
        )
    elif scenario == "new-engagement-confirms":


        checks["narration_posted"] = output_categories == ["narration"]
        checks["confirmation_presented"] = decision_kinds == ["confirmation"]
        checks["engagement_opened"] = "combat_start" in completed
        # Initiative is rolled live, so the correct resolution branches: the actor
        # who wins the first turn rolls the confirmed attack in this same turn, and
        # an actor who loses it must NOT -- the rules put the enemy first, the engine
        # refuses an out-of-turn attack, and the guard accepts the opened fight
        # (initiative's own audited dice) as the confirmed action's resolution, with
        # the attack rolling on the actor's own turn. ``enemy_holds_first_turn`` is
        # read from ``state.json`` after the run, never guessed from adapter state.
        checks["attack_rolled_or_deferred_to_the_actors_turn"] = (
            "combat_attack" in completed or enemy_holds_first_turn
        )
    elif scenario == "maneuver-direct-ruling":


        checks["narration_posted"] = output_categories == ["narration"]
        checks["no_decisions_presented"] = decision_kinds == []
        checks["maneuver_resolved_by_an_attribute_test"] = "attribute_test" in completed
        checks["maneuver_not_resolved_as_an_attack"] = "combat_attack" not in completed
        checks["maneuver_action_spent"] = maneuver_action_spent
    elif scenario == "outlandish-priced-not-granted":
        # The rejection half of the same bypass: an impossible declaration is
        # answered (refused in fiction, or priced with an ordinary or Disadvantaged
        # test -- either is a pass) but never granted. "Granted" has two observable
        # mechanical forms, both read from the campaign record rather than prose:
        # the enemy's own record changing without the audited rolls that could
        # change it (``enemy_unharmed``), and an Advantage-edged roll rewarding the
        # assertion (``no_edge_granted``, from the audit log's own roll.edge).
        checks["narration_posted"] = output_categories == ["narration"]
        checks["no_decisions_presented"] = decision_kinds == []
        checks["enemy_unharmed"] = enemy_unharmed
        checks["no_edge_granted"] = no_edge_granted
    elif scenario == "disengage":
        # Both of this scenario's turns are ordinary in-character declarations -- unlike
        # refusal-account, neither is an early risk-floor withhold -- so both may reach
        # ``engine.run_turn`` and increment ``delivered``. The generic
        # ``delivered == 1`` shape above (built for a run where exactly one turn reaches
        # the engine) does not fit a two-turn run where both legitimately can, so this
        # scenario names its own delivery bar: every turn that ran must have delivered,
        # none withheld. The requirement's own claim -- combat closes, the scene moves,
        # the close is recorded -- is read from campaign files directly, in ``extra``
        # from ``_runtime_flow``, not guessed from adapter-observed tool names.
        checks["turn_delivered"] = delivered == turns and withheld == 0
        checks["fight_closed"] = combat_closed
        checks["scene_moved"] = location_changed
        checks["close_recorded_in_audit_log"] = close_in_audit_log
    elif scenario == "ledger-gate-recovery":
        # M9. A guard withhold ("the confirmed action rolled no resolving mechanic")
        # is this scenario's own expected, legitimate outcome on either or both
        # turns -- that is exactly the condition it exists to exercise -- and
        # ``NarratorService`` records that withhold reason into ``ServiceReport.errors``
        # the same way it records a genuine framework fault. The generic
        # ``service_completed`` bar above would therefore fail a clean run for the
        # very shape this scenario is built to produce, so it drops the ``not
        # errors`` half here: only the turn count backs this scenario's own
        # completion bar, matching the plain "did the service loop reach both
        # declarations" question it actually needs answered.
        checks["service_completed"] = turns == 2
        # Both turns declare the identical confirmed attack, so either may
        # legitimately withhold (the guard, catching a resolving roll that has not
        # happened yet) or deliver (the roll happened) -- the milestone's fix does
        # not force the model to finish the attack in one call, it only makes a
        # same-turn withhold settle its own debt and tell the truth about it. The
        # generic ``turn_delivered`` bar above (built for a run with exactly one
        # engine-reaching turn) does not fit that tolerance, so this scenario names
        # its own: at least one of the two turns must deliver, proving the run made
        # real progress rather than jamming both.
        checks["turn_delivered"] = delivered >= 1
        # The scenario's whole premise: the campaign held no NPC named Rade before
        # this run, so any progress at all -- withheld or delivered -- proves the
        # narrator actually exercised ``npc_create`` mid-turn rather than routing
        # around the reproduction some other way (a decline, a refusal, an
        # unrelated fault).
        checks["fishmonger_npc_created"] = npc_created


        checks["no_ledger_gate_deadlock"] = ledger_gate_faults == 0
        checks["no_false_silence_notice"] = not false_silence_notice_posted
        # The debt this run's own tool calls opened -- whichever turn opened it --
        # must not still be open once the run ends. A ledger left dirty here is
        # exactly the condition that jams every future decision-gated turn.
        checks["ledger_ratified_after_run"] = ledger_ratified_after_run
    elif scenario == "clarification-converges":


        checks["clarification_presented"] = (
            views_presented >= 1 and "clarification" in decision_kinds
        )
        checks["every_view_answered"] = answers_given == views_presented
        checks["no_segment_notice"] = segment_faults == 0
        checks["no_repeated_question"] = repeated_question_count == 0
        checks["narration_posted"] = output_categories[-1:] == ["narration"]
    elif scenario == "new-declaration-heard":


        checks["turn_delivered"] = delivered == 1 and withheld == 0
        checks["first_turn_withheld_with_notice"] = output_categories[:1] == ["notice"]
        # The decisive claim, and the only one that is typed rather than reduced from
        # prose: the retained action was an unconfirmed attack, so a service still
        # adopting it would run the risk floor again and present a second confirmation.
        checks["no_stale_confirmation"] = stale_confirmations_after_the_new_declaration == 0


        checks["reply_addresses_the_new_declaration"] = reply_names_new_action
    elif scenario == "withheld-recovery-armed":


        checks["service_completed"] = turns == 2
        checks["turn_delivered"] = delivered + withheld == turns
        # The premise: a notice naming a control was actually posted. Without one the
        # run never exercised the advertisement at all and proves nothing.
        checks["control_notice_posted"] = control_notices >= 1
        checks["every_control_notice_armed_recovery"] = unarmed_control_notices == 0
    elif scenario == "no-phantom-scene":

        checks["service_completed"] = turns == 2
        checks["turn_delivered"] = delivered + withheld == turns
        # The premise: at least one turn was actually withheld. A run where both turns
        # delivered never reached the branch that wrote the phantom scene.
        checks["turn_withheld"] = turns_withheld >= 1
        # The claim: the audit log holds no ``scene_commit`` beyond the ones the model
        # itself issued, so the engine's settle step wrote no scene record for a turn
        # nobody saw.
        checks["no_engine_scene_commit"] = engine_scene_commits == 0
        checks["ledger_ratified_after_run"] = ledger_ratified_after_run
    elif scenario == "mechanical-truth":


        checks["service_completed"] = turns == 2
        checks["turn_delivered"] = delivered >= 1
        # Two premises, both of which a run can fail while proving nothing either way.
        # A run whose dice never rolled has no tool outcome to contradict, and a run
        # that rolled but printed no announcement line is the separate, already-shipped
        # dice-visibility defect rather than this milestone's -- either way the three
        # claims below would pass vacuously, which is how a probe silently stops
        # measuring anything.
        checks["rolls_recorded"] = rolls_recorded >= 1
        checks["rolls_announced"] = rolls_announced >= 1
        # The milestone's own three claims, one per shape the comparator distinguishes:
        # the verdict the table read is the verdict ``classify`` computed; the target it
        # read is the target the test actually used; and no line announced dice that
        # never rolled.
        checks["every_announced_verdict_matches"] = verdict_mismatch_count == 0
        checks["every_announced_target_matches"] = target_mismatch_count == 0
        checks["no_unbacked_announcement"] = unbacked_announcement_count == 0
    else:
        # refusal-account: the first turn posts the engine notice, never narration;
        # the second is the only turn the narrator ever sees.
        checks["narration_posted"] = output_categories == ["notice", "narration"]
        checks["no_confirmation_presented"] = decision_kinds == []
        checks["withheld_turn_recorded"] = withheld_notice_matched
        checks["reply_names_true_reason"] = reply_names_true_reason
    return checks


async def _runtime_flow(root: Path, endpoint: str, model_id: str, scenario: str) -> dict:
    """Run the real narrator service against the endpoint and report typed checks.
    """
    _prepare_campaign(root, scenario)
    config = NarratorConfig(campaign_root=root, base_url=endpoint, model_id=model_id)
    if scenario == "refusal-account":
        adapter = _RefusalAccountAdapter(config.risk_confirmation_notice)
    elif scenario == "sanctioned-attack":
        # A bare "attack" against the fight's sole recorded NPC combatant is a shape
        # ``combat_sanctions_violence`` sanctions outright (no named target, no residual
        # word survives the subtraction). Declaring no structured-decision capability at
        # all reproduces the exact condition the pre-repair short-circuit keyed on,
        # independent of ``max_decision_rounds``.
        adapter = _RuntimeProbeAdapter(
            "attack",
            capabilities=ChannelCapabilities(
                structured_decisions=False, atomic_decision_delivery=False
            ),
        )
    elif scenario == "weapon-named-sanction":


        adapter = _RuntimeProbeAdapter(
            "stab the reed thug with my dagger",
            capabilities=ChannelCapabilities(
                structured_decisions=False, atomic_decision_delivery=False
            ),
        )
    elif scenario == "ooc-question-in-fight":


        adapter = _RuntimeProbeAdapter(
            "@GM I attack the reed thug, why was that refused?",
            capabilities=ChannelCapabilities(
                structured_decisions=False, atomic_decision_delivery=False
            ),
        )
    elif scenario == "disengage":
        adapter = _DisengageAdapter()
    elif scenario == "ledger-gate-recovery":
        adapter = _LedgerGateRecoveryAdapter(
            config.decision_fault_notice, DECISION_INCOMPLETE_NOTICE
        )
    elif scenario == "clarification-converges":
        # Both engine-authored strings a decision boundary can post here, so the adapter
        # can tell a notice from narration without keeping either.
        adapter = _ClarificationConvergenceAdapter(
            (config.decision_segment_notice, config.decision_fault_notice)
        )
    elif scenario == "new-declaration-heard":
        adapter = _NewDeclarationAdapter()
    elif scenario in ("withheld-recovery-armed", "no-phantom-scene"):
        # One adapter, two scenarios: they need the identical setup and differ only in
        # what ``extra`` reads afterwards, the same sharing attack and unlisted-violence
        # already use.
        adapter = _WithheldRecoveryAdapter()
    elif scenario == "mechanical-truth":
        adapter = _MechanicalTruthAdapter()
    elif scenario == "new-engagement-confirms":


        adapter = _RuntimeProbeAdapter("I attack the reed thug")
    else:


        declarations = {
            "defence": "I dodge",
            "unlisted-violence": "I am stabbing the reed thug",
            # A committed non-attack maneuver: no violence verb, so it routes to the
            # planner -- the route the mid-combat pace bypass covers.
            "maneuver-direct-ruling": "Rill circles behind the reed thug to get an advantage",
            # Impossible on its face and carrying no hazard verb, so it also routes
            # to the planner: the game master must refuse or price it, never grant it.
            "outlandish-priced-not-granted": (
                "Rill sprouts mighty wings, flies to the top of the bell tower, "
                "and becomes invincible"
            ),
        }
        declaration = declarations.get(scenario, "I attack the reed thug")
        adapter = _RuntimeProbeAdapter(declaration)
    service = build_narrator_service(config, adapter)
    report = await service.run()
    extra: dict = {}
    if isinstance(adapter, _RefusalAccountAdapter):
        extra = {
            "withheld_notice_matched": adapter.withheld_notice_matched,
            "reply_names_true_reason": adapter.reply_names_true_reason,
        }
    elif scenario == "disengage":
        # The requirement's own wording reads ``state.json`` and the audit log, not the
        # adapter's tool-completion set: "within a bounded turn count state.json shows
        # combat.active == false and the scene moved, with the close recorded in the
        # audit log." Reading the campaign files directly after the run is that
        # measurement, taken literally, and it is immune to a narrator that reaches the
        # same true state through a tool-call shape this probe did not anticipate.
        state = CampaignStore(root).read_state()
        close_events = [
            event for event in CampaignStore(root).read_events(limit=50)
            if event.get("tool") == "combat_close"
        ]
        extra = {
            "combat_closed": state.combat.active is False,
            "location_changed": (
                bool(state.scene.location_id)
                and state.scene.location_id != _DISENGAGE_START_LOCATION
            ),
            "close_in_audit_log": bool(close_events)
            and bool(str(close_events[-1].get("reason", "")).strip()),
        }
    elif scenario == "maneuver-direct-ruling":
        # Whether the direct ruling's test spent a real combat action, read from the
        # campaign file. The counter survives the turn advancing (it resets only when
        # a turn opens) and the fight ending (a closed fight keeps its actors), so
        # this is safe to read after the run whatever the ruling led to.
        state = CampaignStore(root).read_state()
        extra = {
            "maneuver_action_spent": any(
                actor.side == "pc" and actor.actions_used >= 1
                for actor in state.combat.actors.values()
            ),
        }
    elif scenario == "outlandish-priced-not-granted":
        # Both "granted" forms, read from the campaign record rather than prose: the
        # enemy's own record (hit points and status can only move through audited
        # rolls, so any change here would itself be tool-backed -- the check is that
        # an impossible declaration produced no such change at all), and any
        # Advantage-edged roll this turn's own events carry. Initiative's Advantage
        # (the scout background) sits inside combat_start's ``initiative`` entries,
        # not under a top-level ``roll``, and combat_start ran during setup anyway,
        # so scanning top-level rolls cannot false-positive on it.
        state = CampaignStore(root).read_state()
        thug = state.npcs.get("reed-thug")
        edge_rolls = [
            event
            for event in CampaignStore(root).read_events(limit=50)
            if isinstance(event.get("roll"), dict)
            and event["roll"].get("edge") == "advantage"
        ]
        extra = {
            "enemy_unharmed": (
                thug is not None and thug.status == "alive" and thug.hp == thug.hp_max
            ),
            "no_edge_granted": not edge_rolls,
        }
    elif scenario == "new-engagement-confirms":
        # Whether initiative gave the opposition the opening turn, read from the
        # campaign file: it decides which branch of the scenario's resolution check
        # is the correct one (see ``scenario_checks``).
        state = CampaignStore(root).read_state()
        combat = state.combat
        active = combat.actors.get(combat.active_actor or "")
        extra = {
            "enemy_holds_first_turn": bool(
                combat.active and active is not None and active.side == "npc"
            ),
        }
    elif scenario == "ledger-gate-recovery":
        # The facts the live reproduction and this milestone's fix are about: whether
        # either turn ever faulted at the ledger gate (the redacted
        # ``decision_lifecycle`` events the adapter already retained); whether any
        # turn posted the "No action occurred" notice despite that same turn's own
        # tool calls actually changing state (a same-turn correlation between
        # ``output_categories`` and ``tools_by_turn`` -- a legitimate notice on a
        # turn with an empty tool set is not this defect); and whether the ledger
        # this run's own tool calls may have opened debt against, read directly from
        # the campaign file rather than guessed from adapter state, is still open
        # once the run ends.
        extra = {
            "ledger_gate_faults": sum(
                1
                for event in adapter.decision_events
                if event["category"] == "fault" and event["failure_category"] == "ledger_gate"
            ),
            "false_silence_notice_posted": any(
                category == "notice" and tools
                for category, tools in zip(adapter.output_categories, adapter.tools_by_turn)
            ),
            "npc_created": "npc_create" in adapter.completed_tools,
            "ledger_ratified_after_run": ledger.is_ratified(root),
        }
    elif scenario == "clarification-converges":
        extra = {
            "views_presented": adapter.views_presented,
            "answers_given": adapter.answers_given,
            "repeated_question_count": adapter.repeated_question_count,
            "segment_faults": sum(
                1 for event in adapter.decision_events if event["category"] == "segment"
            ),
        }
    elif scenario == "new-declaration-heard":
        # Turn one's own risk confirmation sits at index 0 and is expected; every
        # later one could only have come from the action turn one declared.
        extra = {
            "stale_confirmations_after_the_new_declaration": sum(
                adapter.risk_confirmations_by_turn[1:]
            ),
            "reply_names_new_action": adapter.reply_names_new_action,
        }
    elif scenario == "withheld-recovery-armed":
        extra = {
            "control_notices": adapter.control_notices,
            "unarmed_control_notices": adapter.unarmed_control_notices,
        }
    elif scenario == "no-phantom-scene":
        # The requirement's own wording reads the record, not adapter state. The engine's
        # settle-time ``scene_commit`` never passes through the agent's tool hook, so it
        # reaches the audit log without reaching ``tool_events``; the difference between
        # the two counts is exactly the engine's own commits, isolated with no prose
        # comparison. The limit is generous: this run issues well under fifty events.
        audit_scene_commits = sum(
            1
            for event in CampaignStore(root).read_events(limit=200)
            if event.get("tool") == "scene_commit"
        )
        extra = {
            "engine_scene_commits": audit_scene_commits - adapter.model_scene_commits,
            "turns_withheld": report.withheld,
            "ledger_ratified_after_run": ledger.is_ratified(root),
        }
    elif scenario == "mechanical-truth":
        # The truth side comes from the audit log, never the adapter and never the
        # engine's own ``_verdict_log``: the guard this milestone adds is the thing
        # under test, so a measurement that asked it whether it had worked would prove
        # nothing. The claim side is the announcement fields the adapter parsed out of
        # each delivered reply. ``verdict_mismatches`` -- the production comparator,
        # run here rather than reimplemented -- classifies each announcement against
        # the recorded rolls, and the three counts below are its three kinds.
        recorded = _recorded_roll_facts(root)
        mismatches = verdict_mismatches(_announcement_text(adapter.announced), recorded)
        kinds = [entry["kind"] for entry in mismatches]
        extra = {
            "rolls_recorded": len(recorded),
            "rolls_announced": len(adapter.announced),
            "verdict_mismatch_count": kinds.count("verdict"),
            "target_mismatch_count": kinds.count("target"),
            "unbacked_announcement_count": kinds.count("unbacked"),
        }
    return scenario_checks(
        scenario,
        turns=report.turns,
        errors=bool(report.errors),
        delivered=report.delivered,
        withheld=report.withheld,
        output_categories=list(adapter.output_categories),
        decision_kinds=list(adapter.decision_kinds),
        attempted={name for turn in getattr(service.engine, "_tool_log", ()) for name in turn},
        completed=set(adapter.completed_tools),
        **extra,
    )


def _run_runtime(root: Path, endpoint: str, model_id: str, scenario: str) -> int:
    """Emit one machine-readable marker without retaining model prose."""
    checks = asyncio.run(_runtime_flow(root, endpoint, model_id, scenario))
    print(f"COMBAT_RUNTIME={json.dumps(checks, sort_keys=True)}")
    return 0 if all(checks.values()) else 75


def _live_scenario(run_directory: Path, endpoint: str, model_id: str, scenario: str) -> dict:
    """Launch the runtime helper under the narrator interpreter for one scenario.

    Each attempt uses a fresh campaign, so a retry never inherits a partial fight. The
    returned checks come from the last attempt, which keeps a failing measurement visible
    instead of discarding it.
    """
    narrator_python = os.environ.get(
        "NARRATOR_PYTHON",
        str(Path(__file__).resolve().parents[1] / ".narrator-venv" / "bin" / "python"),
    )
    environment = {
        **os.environ,
        "BSH_SERVER_PYTHON": os.environ.get("BSH_SERVER_PYTHON", sys.executable),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    repository = Path(__file__).resolve().parents[1]
    checks: dict = {}
    for attempt in range(3):
        campaign_root = run_directory / f"combat-{scenario}-{attempt}"
        result = subprocess.run(
            [
                narrator_python, "scripts/probe_combat_decisions.py", "--runtime",
                "--scenario", scenario,
                "--campaign-root", str(campaign_root),
                "--endpoint", endpoint,
                "--model-id", model_id,
            ],
            cwd=repository, env=environment, text=True, capture_output=True,
            timeout=300, check=False,
        )
        marker = next(
            (
                line.partition("=")[2]
                for line in result.stdout.splitlines()
                if line.startswith("COMBAT_RUNTIME=")
            ),
            "",
        )
        try:
            checks = json.loads(marker)
        except json.JSONDecodeError:
            checks = {}
        if result.returncode == 0 and checks and all(checks.values()):
            return {"attempts": attempt + 1, "checks": checks}
    if not checks:
        raise ValueError(f"the {scenario} scenario produced no marker")
    return {"attempts": 3, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--campaign-root")
    parser.add_argument("--model-id")
    args = parser.parse_args()
    if args.runtime:
        if not args.campaign_root or not args.model_id or not args.scenario:
            print("runtime requires --campaign-root, --model-id, and --scenario", file=sys.stderr)
            return 2
        return _run_runtime(
            Path(args.campaign_root), args.endpoint.rstrip("/"), args.model_id, args.scenario
        )
    run_directory = os.environ.get("BSH_PROBE_RUN_DIRECTORY", "")
    if not run_directory:
        print("HELD: BSH_PROBE_RUN_DIRECTORY is required.", file=sys.stderr)
        return 75
    destination = Path(run_directory) / "combat-production-probe.json"
    if destination.exists():
        print("HELD: the run-bound report already exists.", file=sys.stderr)
        return 75
    checks: dict = {}
    report = {
        "status": "PASS",
        "candidate_tree": candidate_tree(Path(run_directory)),
        "command_sha256": _sha256(" ".join(sys.argv).encode()),
        "launcher_sha256": (_sha256(Path(os.environ["BSH_MODEL_LAUNCHER"]).read_bytes())
                            if os.environ.get("BSH_MODEL_LAUNCHER") else None),
        "checks": checks,
        "outputs": [],
        "offline": args.offline,
    }
    if not args.offline:
        try:
            endpoint = args.endpoint.rstrip("/")
            model = resolve_model_and_digest(endpoint)
            report.update(model)
            report.update(_container_image())
            checks = {
                scenario: _live_scenario(Path(run_directory), endpoint, model["model_id"], scenario)
                for scenario in SCENARIOS
            }
            report["checks"] = checks
            # The attempt count sits beside the checks rather than among them. A counter
            # can therefore never be mistaken for a check, nor a check for a counter.
            # Every entry under "checks" is a boolean claim.
            failed = [
                f"{scenario}.{name}"
                for scenario, result in checks.items()
                for name, value in result["checks"].items()
                if value is not True
            ]
            report["failed_checks"] = failed
            report["status"] = "PASS" if not failed else "FAIL"
        except (OSError, ValueError, urllib.error.URLError, subprocess.SubprocessError) as error:
            print(
                f"HELD: production endpoint evidence is unavailable: {type(error).__name__}",
                file=sys.stderr,
            )
            return 75
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    digest = _sha256(destination.read_bytes())
    print(f"report_path={destination}")
    print(f"report_sha256={digest}")
    return 0 if report["status"] == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
