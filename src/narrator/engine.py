"""The narrator engine: one Strands agent per channel, one turn at a time.

This is the only module that imports Strands, which is why it lives alone. Everything it
depends on — the tool manifest, the leak filter, the ledger reader, prompt assembly, the
channel contract, the settle schema — is importable without the framework, so the main
test environment pins it all without installing Strands.

Two structural constraints bound any future concurrency work in this module, so a
change starts from what actually holds rather than what \"async\" suggests. First,
``run_turn`` is single-turn-at-a-time per engine, not per channel: it resets and reads
plain instance attributes (``self._calls_this_turn``, ``self._tool_events_this_turn``,
``self._roll_facts_this_turn``, ``self._resolution_guard``, ``self._turn_principal``,
``self._meta_turn_active``, and their neighbors, reset together where ``run_turn``
begins) rather than a value scoped to one call, so two turns racing on the same engine
instance would corrupt each other's bookkeeping; only work that stays inside one
turn's own call tree may run concurrently against this state. Second, the MCP session
itself is not free to overlap: ``_call_tool`` reaches the server through
``self._client.call_tool_sync`` — synchronous and blocking despite running inside an
``async def`` — so two tool calls issued \"concurrently\" from this engine still
serialize on that one client, and only the model-request side of a call (structured
calls with no tool access, or the companion pattern's beside-the-full-call requests)
is actually free to overlap.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from narrator import canon, ledger, payload, policy  # noqa: E402
from narrator.adjudicate import (  # noqa: E402
    ADJUDICATOR_SYSTEM_PROMPT,
    AdjudicateOutcome,
    adjudicate_prompt,
)
from narrator.announcements import (  # noqa: E402, F401
    ANNOUNCED_ROLL_TOOLS,
    DOOM_ANNOUNCED_TOOLS,
    META_PROMISE_PATTERN,
    META_TOOL_NAME_PATTERN,
    ROLL_UNDER_TOOLS,
    _classify_roll_under,
    _drop_meta_leak_sentences,
    _meta_leak_detected,
    _states_helpless_recovery,
    announced_rolls,
    doom_facts,
    format_doom_announcement,
    format_roll_announcement,
    inject_missing_doom_announcements,
    inject_missing_roll_announcements,
    roll_facts,
    runic_stake_line,
    scrub_contradicted_announcements,
    verdict_mismatches,
    withheld_mechanical_lines,
)
from narrator.assess import (  # noqa: E402
    ASSESSOR_SYSTEM_PROMPT,
    HazardAssessment,
    assessment_downgrades,
    assessment_prompt,
)
from narrator.channels.base import InboundTurn  # noqa: E402
from narrator.classify import (  # noqa: E402
    AFTER_COMPANIONS,
    BESIDE_COMPANIONS,
    DEFAULT_TURN_CLASSIFIER,
    DEFENCE_COMPANION,
    HAZARD_COMPANION,
    PERSONS_COMPANION,
    ROUTE_COMPANION,
    TurnClassifier,
)
from narrator.config import THINKING_LEVELS, NarratorConfig  # noqa: E402
from narrator.decisions import (  # noqa: E402
    ApproachDraft,
    ConfirmationDraft,
    DeclinedPlan,
    PlannerPolicyError,
    PlanOutcome,
    ProceedPlan,
    ProgressiveDecisionSession,
    RequestDecisionPlan,
    continuation_text,
    defence_obligation_text,
    escalation_prompt,
    hazard_obligation_text,
    planner_repair_prompt,
    planning_bypass,
    planning_prompt,
    requires_risk_confirmation,
    verify_plan,
)
from narrator.delivery import TurnOutcome, scrub_markup, strip_leaked_notices  # noqa: E402
from narrator.facets import ClassificationContext  # noqa: E402
from narrator.interactions import (  # noqa: E402
    narration_invites_reply,
    read_character_display_names,
    read_combat_snapshot,
    read_trusted_scope,
)
from narrator.prompt import (  # noqa: E402
    THINKING_SECTION,
    assemble_system_prompt,
    language_directive,
    turn_prompt,
)
from narrator.resolution_guard import ResolutionGuard  # noqa: E402
from narrator.server_launch import resolved_server_command, server_env  # noqa: E402

if TYPE_CHECKING:
    # narrator.service imports this module at import time, so a top-level import
    # back would cycle; this one exists for run_turn's type hint only and never
    # runs at import time.
    from narrator.service import PreparedNarrationTurn  # noqa: E402

from narrator.settle import SETTLER_SYSTEM_PROMPT, SettleOutcome, settle_prompt  # noqa: E402
from narrator.sweep import (  # noqa: E402
    SWEEPER_SYSTEM_PROMPT,
    SweepIntroducingOutcome,
    departed_person_ids,
    empty_roster,
    guard_ratification,
    introduced_person_names,
    sweep_prompt,
    validated_mentions,
)

#: Skills loaded through progressive disclosure. ``bsh-gm`` is deliberately absent: it
#: is concatenated into the system prompt instead, because a system prompt never slides
#: out of context while a skill body enters message history and the window evicts it.
#: ``bsh-worldsmith`` is absent because its own front matter forbids the player-facing
#: runtime.
DISCLOSED_SKILLS: tuple[str, ...] = ("skills/bsh-session-zero", "skills/bsh-session-close")

PLANNER_SYSTEM_PROMPT = (
    "You are a pre-action decision planner. You never narrate, call tools, or mutate "
    "campaign state. Return proceed unless a defined confirmation, clarification, or "
    "mechanically binding approach choice is required before play continues."
)


ESCALATION_SYSTEM_PROMPT = (
    "You are a pre-action decision planner. You already indicated this declared "
    "action might require a mechanical check, but did not bind it to a resolving "
    "mechanic. Specify exactly how it resolves: an approach whose options each bind "
    "a directive. Use attribute_test or combat_defend for an option that genuinely "
    "needs a roll; use no_test for an option where, on reflection, no roll is "
    "actually needed. Never infer account identities. Audience: use only current or "
    "party. Never name an actor, account, or token."
)


REPETITION_SIMILARITY_THRESHOLD: float = 0.85


UNCOMMITTED_NARRATION_MAX_CHARS: int = 1_600


PROMISED_CHECK_PATTERN = re.compile(
    r"requires?\s+(?:(?:an?|the)\s+)?(?:[\w/]+\s+){0,2}(?:check|roll|test)\b", re.IGNORECASE
)


def _plan_log_entry(result: PlanOutcome | DeclinedPlan) -> dict:
    """One ``_plan_log`` row for a real, model-authored ``plan_turn`` result.

    Only ``ClarificationDraft`` and ``ConfirmationDraft`` are scanned against
    ``PROMISED_CHECK_PATTERN``: both carry no ``ResolutionDirective`` field at
    all, so a promise in their own text has nothing enforcing it.
    ``ApproachDraft`` is the directive-bound shape (``ApproachOption.directive``,
    ``resolution_guard.ResolutionGuard``) -- a promise there is exactly what production
    already enforces, so it is reported but never flagged.
    """
    if isinstance(result, DeclinedPlan):
        return {"kind": "declined", "promised_unbound_check": False}
    if isinstance(result.plan, ProceedPlan):
        return {"kind": "proceed", "promised_unbound_check": False}
    decision = result.plan.decision
    dkind = decision.kind
    promised = False
    if dkind in ("clarification", "confirmation"):
        text_parts = [decision.question, decision.context]
        if dkind == "clarification":
            text_parts += [path.label for path in decision.paths]
        promised = bool(PROMISED_CHECK_PATTERN.search(" ".join(text_parts)))
    return {"kind": dkind, "promised_unbound_check": promised}

#: The ``_repetition_log`` entry a turn that never reached the detector records --
#: the framework-fault and decision-recovery early returns in ``run_turn``, which
#: produce no candidate narration to compare. Copied with ``dict()`` at each use
#: so no caller can mutate the shared literal.
_NO_REPETITION_CHECK: dict = {
    "triggered": False,
    "retried": False,
    "similarity": 0.0,
    "resolved": None,
    "leaks_scrubbed": 0,
}

#: Sent back to the same channel agent, once, when a candidate narration nearly
#: repeats the prior delivered turn. It never re-requests tools: the turn's
#: mechanics already resolved before this fires, and asking the model to roll or
#: commit again would double the very consequences the ratification barrier
#: exists to keep singular.
REPETITION_NUDGE_PROMPT = (
    "Your last reply repeated the wording of your previous turn on this channel "
    "almost verbatim. A repeated identical declaration must advance, complete, or "
    "complicate the situation -- never simply replay the same moment. Do not call "
    "any tool: this turn's mechanics already resolved. Write only the replacement "
    "narration, and make something genuinely different happen or become clear."
)


TRAVERSAL_CLOCK_PREFIX = "travel-"

#: The ``_traversal_log`` entry a turn that never reaches the traversal-monotone
#: check records -- the framework-fault and decision-recovery early returns, and
#: any withheld turn, none of which read a delivered candidate the fallback could
#: act on. Copied with ``dict()`` at each use so no caller can mutate the shared
#: literal, the same discipline ``_NO_REPETITION_CHECK`` already established.
_NO_TRAVERSAL_CHECK: dict = {"checked": [], "advanced": [], "declaration_repeat": False}


PLANNER_FAILURE_REASONS: tuple[str, ...] = (
    #: The prompt could not be built at all: an unreadable canon digest or campaign root.
    "planner_inputs",
    #: The model call did not answer inside ``config.settle_timeout_seconds``.
    "planner_timeout",
    #: An answer arrived that the ``PlanOutcome`` schema refuses.
    "planner_schema",
    #: A schema-valid proposal an engine-owned planning policy refuses (``verify_plan``,
    #: ``PlannerPolicyError``), including "the planner returned no proposal".
    "planner_policy",
    #: Anything else the planning call raised.
    "planner",
)


class DecisionPlanningError(RuntimeError):
    """The tool-less planner failed, so the normal narrator must not run.

    ``reason`` is one of ``PLANNER_FAILURE_REASONS``, carried so the service can record
    which failure this was without re-inspecting the cause chain.
    """

    def __init__(self, message: str, reason: str = "planner") -> None:
        super().__init__(message)
        self.reason = reason


def _planner_failure_reason(error: BaseException) -> str:
    """Which of ``PLANNER_FAILURE_REASONS`` this raised planning exception is.

    Ordered narrowest first. ``asyncio.wait_for`` raises ``TimeoutError`` (an alias of
    ``asyncio.TimeoutError`` since Python 3.11), ``PlanOutcome.model_validate`` raises
    Pydantic's ``ValidationError``, and both ``PlannerPolicyError`` and ``verify_plan``'s
    own refusals are ``ValueError`` subclasses. Anything else stays ``planner`` rather
    than being forced into a bucket it does not belong in.
    """
    if isinstance(error, TimeoutError):
        return "planner_timeout"
    if isinstance(error, ValidationError):
        return "planner_schema"
    if isinstance(error, PlannerPolicyError):
        return "planner_policy"
    return "planner"


def _plan_failure_entry(reason: str, *, disposition: str) -> dict:
    """One ``_plan_log`` row for a planning attempt that produced no decision.

    ``_plan_log`` only ever recorded successes, so a session whose planner failed left
    the instrument silent about the very turns an operator most needs to see. The row
    carries the same keys ``narrator.soak_harness`` already reads (``kind``,
    ``promised_unbound_check``, ``escalation``, ``final_kind``) plus the typed
    ``reason``; ``kind`` is ``"failed"``, a value outside the decision vocabulary the
    harness counts, so a failure can never be mistaken for a decision that happened.

    ``disposition`` says what the engine did next: ``fault`` (the error was raised and
    the service posted a no-action notice) or ``floor`` (a ``/retry``'s second failure
    fell back to the engine's own risk floor).
    """
    return {
        "kind": "failed",
        "promised_unbound_check": False,
        "escalation": "not_triggered",
        "final_kind": "failed",
        "reason": reason,
        "disposition": disposition,
    }


def normalize_for_similarity(text: str) -> str:
    """Fold case and punctuation, the project's one existing normalization convention.

    The normalizer family, so nobody unifies it by mistake: this function (turn
    similarity), ``narrator.soak_instruments.normalize_for_lenient_match`` (recall
    scoring, kept byte-equal across the interpreter split), and
    ``bsh_mcp.service._normalized_fact`` (fact identity: whitespace, case and
    trailing punctuation only, deliberately *not* token-splitting so a fact's
    punctuation-bearing content survives equality). Three contracts, two shared
    bodies, one distinct -- a change to any one must say which of the three it
    means.
    """
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def narration_similarity(a: str, b: str) -> float:
    """How much one narration repeats another, 0.0 to 1.0, after normalization.
    """
    left, right = normalize_for_similarity(a), normalize_for_similarity(b)
    if not left or not right:
        # An empty side means no comparable content, not a perfect repeat:
        # ``SequenceMatcher("", "").ratio()`` is 1.0, which is the §3.3 misfire.
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _scene_fingerprint(campaign_root: Path | str) -> str:
    """The scene record's content hash, fail-open: unreadable hashes as absent.

    The sweep keys on this fingerprint rather than on tool-event inference: the
    file is the record, so any writer that changed it — the model's own
    ``scene_commit``, the settle commit, a scene transition — is visible here
    without the engine tracking who wrote.
    """
    try:
        return hashlib.sha256(
            (Path(campaign_root) / "campaign" / "scene.md").read_bytes()
        ).hexdigest()
    except OSError:
        return ""


def _character_resource_values(campaign_root: Path | str) -> dict:
    """Every character's current coins, HP, and equipment, keyed by character id.

    Fail-open in the same mold as ``_scene_fingerprint``: an absent directory or one
    malformed character file contributes nothing rather than raising. The sweep guard
    needs real state bound to the specific character a claim names, not a party-wide
    pool — see ``narrator.sweep.guard_ratification`` and ``_match_character`` — so
    this returns one entry per character rather than flat totals. It reads the same
    ``campaign/characters/*.json`` files ``narrator.canon``'s Party resources block
    reads, independently and for a different purpose: the digest renders text for the
    model to read, this returns values for the guard to verify a claim against.
    """
    characters_dir = Path(campaign_root) / "campaign" / "characters"
    characters: dict[str, dict] = {}
    if not characters_dir.is_dir():
        return characters
    for path in sorted(characters_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        character_id = str(payload.get("id") or path.stem)
        raw_coins = payload.get("coins")
        coins = raw_coins if isinstance(raw_coins, int) and not isinstance(raw_coins, bool) else None
        raw_hp = payload.get("hp")
        hp = raw_hp if isinstance(raw_hp, int) and not isinstance(raw_hp, bool) else None
        characters[character_id] = {
            "id": character_id,
            "name": str(payload.get("name") or character_id),
            "coins": coins,
            "hp": hp,
            "equipment": tuple(str(item) for item in payload.get("equipment") or []),
        }
    return characters


def _npc_life_values(campaign_root: Path | str) -> dict:
    """Every recorded NPC's id, name, and current status, keyed by npc id.
    """
    state_path = Path(campaign_root) / "campaign" / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    raw_npcs = payload.get("npcs") if isinstance(payload, dict) else None
    if not isinstance(raw_npcs, dict):
        return {}
    npcs: dict[str, dict] = {}
    for npc_id, entry in raw_npcs.items():
        if not isinstance(entry, dict):
            continue
        npcs[str(npc_id)] = {
            "id": str(entry.get("id") or npc_id),
            "name": str(entry.get("name") or npc_id),
            "status": str(entry.get("status") or ""),
        }
    return npcs


def _config_catalog(engine) -> object | None:
    """The engine's locale catalog, or ``None`` on a stub with no config.

    Fail-open in the union's own terms: no catalog means the English regex arm
    alone, which is the exact pre-localization behavior, and several offline stubs
    exercise the verdict path without ever constructing a config.
    """
    config = getattr(engine, "config", None)
    return getattr(config, "catalog", None) if config is not None else None


def _entity_roster(campaign_root: Path | str) -> dict:
    """Every recorded identifier the sweep may be told about, by kind. Fail-open.
    """
    roster = empty_roster()
    state_path = Path(campaign_root) / "campaign" / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    if isinstance(payload, dict):
        raw_npcs = payload.get("npcs")
        if isinstance(raw_npcs, dict):
            for npc_id, entry in raw_npcs.items():
                if isinstance(entry, dict):
                    roster["npcs"][str(npc_id)] = str(entry.get("status") or "")
                    roster["npc_names"][str(npc_id)] = str(entry.get("name") or npc_id)
        scene = payload.get("scene")
        if isinstance(scene, dict):
            roster["summary"] = str(scene.get("summary") or "")
            raw_persons = scene.get("persons")
            if isinstance(raw_persons, dict):
                for person_id, entry in raw_persons.items():
                    if isinstance(entry, dict):
                        roster["persons"][str(person_id)] = str(entry.get("name") or person_id)
            raw_objects = scene.get("objects")
            if isinstance(raw_objects, dict):
                for object_id, entry in raw_objects.items():
                    if isinstance(entry, dict):
                        roster["objects"][str(object_id)] = str(entry.get("state") or "")
            raw_exits = scene.get("exits")
            if isinstance(raw_exits, list):
                roster["exits"] = [str(exit_id) for exit_id in raw_exits if isinstance(exit_id, str)]
    characters_dir = Path(campaign_root) / "campaign" / "characters"
    if characters_dir.is_dir():
        for path in sorted(characters_dir.glob("*.json")):
            try:
                sheet = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(sheet, dict) and isinstance(sheet.get("id"), str):
                roster["characters"][sheet["id"]] = str(sheet.get("name") or sheet["id"])
    return roster


def _clock_fills(campaign_root: Path | str) -> dict[str, dict]:
    """Every clock's segments and filled count, read directly from ``state.json``.

    Fail-open in ``_character_resource_values``'s mold: an absent file, an
    unparsable one, or one odd-shaped clock entry contributes nothing rather than
    raising. This module stays free of a ``src/bsh_mcp`` dependency (see
    ``TRAVERSAL_CLOCK_PREFIX`` above), so this reads the campaign's own
    ``campaign/state.json`` file directly instead of importing ``CampaignState``.
    ``_advance_stalled_traversal`` is the one caller: it needs a clock's own filled
    count from before and after a turn's own actions, which no existing helper in
    this module reads.
    """
    state_path = Path(campaign_root) / "campaign" / "state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    clocks = payload.get("clocks")
    if not isinstance(clocks, list):
        return {}
    fills: dict[str, dict] = {}
    for entry in clocks:
        if not isinstance(entry, dict):
            continue
        clock_id = entry.get("id")
        segments = entry.get("segments")
        filled = entry.get("filled")
        if not isinstance(clock_id, str):
            continue
        if not isinstance(segments, int) or isinstance(segments, bool):
            continue
        if not isinstance(filled, int) or isinstance(filled, bool):
            continue
        fills[clock_id] = {"segments": segments, "filled": filled}
    return fills


def _assistant_text(result) -> str:
    """Pull the assistant text out of a Strands result across its shapes."""
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        blocks = message.get("content") or []
        return "".join(
            block.get("text", "") for block in blocks if isinstance(block, dict)
        )
    return str(result or "")


def _turn_usage(result) -> dict:
    """This turn's token usage, fail-open: an absent or odd shape reports nothing.

    The field named ``metrics.accumulated_usage`` is the wrong one. Strands never
    resets it, and ``_agent_for`` caches one agent per channel for the whole session,
    so it reports the running session total on every turn. ``agent_invocations`` holds
    one entry per ``invoke_async`` call and its last entry's ``usage`` covers this turn
    alone, summed across the turn's model calls.
    """
    invocations = getattr(getattr(result, "metrics", None), "agent_invocations", None)
    usage = getattr(invocations[-1], "usage", None) if invocations else None
    if not isinstance(usage, dict):
        return {}
    return {
        key: int(value)
        for key, value in usage.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _tool_status(result) -> str:
    """Read the status out of an MCP tool result across its shapes."""
    if isinstance(result, dict):
        return str(result.get("status", "error"))
    return str(getattr(result, "status", "error"))


def redact_tool_event(name: str, *, exception: bool, payload: dict) -> dict:
    """Build one retained tool record from a name and its result envelope.

    The record carries the tool name, its ``ok`` disposition, its error code, and its
    event id, and nothing else. It never carries the call's arguments or the result
    content, so the retained transcript answers which tools ran and whether each rolled
    without exposing player text or canon. A framework-level raise carries no envelope,
    so it records ``ok`` False with an ``exception`` code and no event id rather than
    inventing either.
    """
    if exception:
        return {"tool": name, "ok": False, "error": "exception", "event_id": None}
    return {
        "tool": name,
        "ok": bool(payload.get("ok")),
        "error": str(payload.get("error", "")),
        "event_id": payload.get("event_id"),
    }


#: Sent back to the same channel agent, once, when a meta (game-master-discussion)
#: turn's candidate narrates an action or names/promises a refused tool. Restates
#: the rule rather than the specific violation, matching ``VERDICT_CORRECTION_PROMPT``
#: below: the model already has the turn's own context, and the fix is a rewrite, not
#: new information.
META_CORRECTION_PROMPT = (
    "This turn is out-of-fiction discussion with the game master. Your last reply "
    "narrated a character or an enemy acting, named an engine tool, or promised to "
    "call one -- none of that can happen on this turn: no die rolls, no action "
    "resolves, and nothing is written to the record. Rewrite it as a plain "
    "out-of-fiction answer grounded in the campaign record. Do not call any tool: "
    "this turn's mechanics cannot resolve. If the player wants to act, tell them to "
    "say it without the game-master designator."
)

#: The ``_meta_guard_log`` entry a turn that never reached the guard records --
#: every non-meta turn, plus the framework-fault and decision-recovery early
#: returns. Copied with ``dict()`` at each use, the same discipline
#: ``_NO_REPETITION_CHECK`` and ``_NO_VERDICT_CHECK`` establish.
_NO_META_GUARD_CHECK: dict = {
    "triggered": False,
    "retried": False,
    "resolved": None,
    "sentences_dropped": 0,
    "leaks_scrubbed": 0,
}


#: The ``_verdict_log`` entry a turn that never reached the verdict check records --
#: the framework-fault and decision-recovery early returns in ``run_turn``, which
#: produce no candidate narration to compare. Copied with ``dict()`` at each use so no
#: caller can mutate the shared literal, the discipline ``_NO_REPETITION_CHECK`` and
#: ``_NO_TRAVERSAL_CHECK`` already established.
_NO_VERDICT_CHECK: dict = {
    "triggered": False,
    "retried": False,
    "mismatches": 0,
    "kinds": [],
    "resolved": None,
    "leaks_scrubbed": 0,
    "announcements_scrubbed": 0,
}

#: Sent back to the same channel agent, once, when a candidate narration announces a
#: verdict, a target, or a roll this turn's own tool results do not back, or leaves a
#: survived ``helpless_roll`` unmentioned. It never re-requests tools, for the reason
#: ``REPETITION_NUDGE_PROMPT`` states: the dice already fell, and rolling again would
#: double the consequences the ratification barrier keeps singular. It restates the
#: rule rather than supplying the corrected numbers, because the numbers are already
#: in the model's own context in the tool results it just received, and handing them
#: back would teach the turn loop to write the announcement instead of the model.
VERDICT_CORRECTION_PROMPT = (
    "Your last reply announced a roll that does not match the tool result it came "
    "from: a verdict, a target number, or a roll itself that no tool returned this "
    "turn, or it left a Helpless result you rolled unannounced. Re-read this turn's "
    "tool results. Every announcement line must read "
    "`<Name> rolls <ATTRIBUTE>: rolled <total> vs target <target>, <outcome>.` with the total, the "
    "target, and the outcome copied exactly from the tool result's own roll, total, "
    "target, and outcome fields -- this game rolls under, so a total below the target "
    "is a success and a total at or above it is a failure, and the tool already "
    "decided which. Do not call any tool: this turn's mechanics already resolved. "
    "Write only the replacement narration."
)

#: The ``_recovery_log`` entry a turn without a re-narrated combat recovery records --
#: every ordinary turn, plus the framework-fault and withheld paths. Copied with
#: ``dict()`` at each use so no caller can mutate the shared literal, the discipline
#: ``_NO_REPETITION_CHECK`` and ``_NO_VERDICT_CHECK`` already established.
_NO_RECOVERY_CHECK: dict = {
    "triggered": False,
    "retried": False,
    "resolved": None,
    "fell_back": False,
    "leaks_scrubbed": 0,
}


def recovery_narration_prompt(recovery_lines: list[str]) -> str:
    """The corrective re-invoke sent after the engine rolled an owed combat mechanic.
    """
    facts = "\n".join(f"- {line}" for line in recovery_lines)
    return (
        "Your last reply left this turn's confirmed combat action unresolved, so "
        "the game engine performed the owed mechanic itself through the same "
        "audited tools. These results are now recorded fact:\n\n"
        f"{facts}\n\n"
        "Write this turn's narration again from those results alone. Narrate each "
        "result in the fiction, including a death when the results state one. Do "
        "not repeat or re-ask any question from your earlier reply -- the results "
        "above supersede it. Do not state any roll, damage figure, or outcome the "
        "results above do not contain. Do not call any tool: this turn's mechanics "
        "already resolved. Write only the replacement narration."
    )


def _is_max_tokens_fault(error: BaseException) -> bool:
    """Whether ``error`` is Strands' ``MaxTokensReachedException``, by type or by name.

    By name as well, so a channel built on a Strands release that moves the class --
    or a test double that cannot import it -- still classifies the fault the same way.
    """
    try:
        from strands.types.exceptions import MaxTokensReachedException
    except Exception:  # noqa: BLE001 - the name check below still applies
        MaxTokensReachedException = ()  # type: ignore[assignment]  # noqa: N806
    return isinstance(error, MaxTokensReachedException) or (
        type(error).__name__ == "MaxTokensReachedException"
    )


def tool_call_signature(name: str, arguments) -> str:
    """One tool call's identity for the duplicate instrument: name plus canonical args.

    Sorted-key JSON makes the signature argument-order independent, and an
    unserializable argument falls back to ``repr`` rather than failing -- this is a
    measurement, and a measurement never fails a turn. The signature lives only in
    process memory for the turn it measures; the retained diagnostics keep their
    no-arguments contract (see ``_tool_events_this_turn``).
    """
    try:
        rendered = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=repr)
    except Exception:  # noqa: BLE001 - a measurement never fails a turn
        rendered = repr(arguments)
    return f"{name}:{rendered}"


def duplicate_call_count(signatures) -> int:
    """How many of one turn's tool calls repeat an earlier call byte for byte.

    The instrument behind the reasoning-replay measurement: a re-derived thought
    rolling the same declaration twice and a replayed thought executing its own
    musings both land here, on whichever arm produces them. Counted as repeats past
    the first, so ``[a, a, a]`` scores 2.
    """
    return len(signatures) - len(set(signatures))


def _discard_partial_exchange(agent, prompt_text: str) -> None:
    """Drop the trailing assistant partial(s) and the user message that prompted them.

    Strands appends the user message before the call and the partial assistant
    message when the cap lands, so the exchange sits at the tail of ``agent.messages``.
    Only that tail is touched: trailing assistant messages go, then the one user
    message whose text is ``prompt_text`` if it is now last. Anything else stays.
    """
    messages = getattr(agent, "messages", None)
    if not isinstance(messages, list):
        return
    while messages and messages[-1].get("role") == "assistant":
        messages.pop()
    if messages and messages[-1].get("role") == "user":
        texts = [
            block.get("text") for block in messages[-1].get("content", [])
            if isinstance(block, dict) and "text" in block
        ]
        if texts == [prompt_text]:
            messages.pop()


@dataclass
class _ChannelNarration:
    """One channel's own narration-tracking state.

    Replaces three dicts (``_last_delivered_narration``, ``_uncommitted_narration``,
    ``_last_declaration_text``) that shared this exact key space but not a single
    invalidation rule -- each field below still clears on its own trigger, exactly as
    the three dicts did independently; only the container is unified.
    """

    #: The last narration actually delivered on this channel. Written only when a
    #: turn is not withheld, since a withheld turn's narration never reached a
    #: player and is not the moment a real repeat would replay. ``_avoid_repeat``
    #: reads this; nothing else does.
    last_delivered: str = ""


    uncommitted: list[str] = field(default_factory=list)


    last_declaration: str = ""


class NarratorEngine:
    """Owns the MCP session, the per-channel agents, the settler, and the turn loop."""

    def __init__(
        self,
        config: NarratorConfig,
        turn_classifier: TurnClassifier | None = None,
        defence_companion: TurnClassifier | None = None,
        hazard_companion: TurnClassifier | None = None,
        persons_companion: TurnClassifier | None = None,
        route_companion: TurnClassifier | None = None,
    ) -> None:
        self.config = config
        self._client = None
        #: The classification contract this engine asks the model to answer in. The
        #: default is ``narrator.classify``'s composition; a context that needs a
        #: different set of questions -- session zero, another game's social
        #: vocabulary -- injects its own, and ``classify_intent`` below sends whatever
        #: was injected. The transport stays here because this is the one object a
        #: model is reachable from.
        self.turn_classifier: TurnClassifier = turn_classifier or DEFAULT_TURN_CLASSIFIER
        #: The narrow compositions the typed state selects beside the full one; see
        #: ``narrator.classify``'s companion block for the measurement. Injectable for
        #: the same reason ``turn_classifier`` is.
        self.defence_companion: TurnClassifier = defence_companion or DEFENCE_COMPANION
        self.hazard_companion: TurnClassifier = hazard_companion or HAZARD_COMPANION
        self.persons_companion: TurnClassifier = persons_companion or PERSONS_COMPANION
        self.route_companion: TurnClassifier = route_companion or ROUTE_COMPANION
        #: One entry per ``classify_intent`` call: what each companion did --
        #: ``skipped`` (state did not select it), ``none`` (asked, changed nothing),
        #: ``bound``/``added`` (asked, merged), or ``fault`` (asked, failed; the full
        #: verdict stood). The soak harness reads it like ``_sweep_log``.
        self._companion_log: list[dict] = []
        self._tools: list = []
        self._agents: dict[str, object] = {}
        self._system_prompt = ""
        #: The channel agent's thinking level, one of ``THINKING_LEVELS``. Session
        #: state rather than configuration: ``set_turn_thinking_level`` changes it for
        #: every channel agent this engine holds, and the next request each makes
        #: carries the new level, because the model's request hook and the agent's
        #: system prompt are both read at request time rather than at construction.
        self._turn_thinking_level = config.turn_thinking_level
        #: True only while ``_invoke_channel_agent`` re-runs one faulted attempt with
        #: thinking off; ``_turn_request_overrides`` and ``_turn_system_prompt`` read it.
        self._thinking_suppressed = False
        #: One turn index per thinking attempt that hit its answer bound with no tool
        #: run and was re-run with thinking off. The soak reports it beside the
        #: phantom gates, so a run at ``low`` states how often the bound fired.
        self._thinking_fallbacks: list[int] = []
        self._calls_this_turn = 0
        #: One entry per turn: the canon digest's measurements, read by the soak
        #: harness through the same adapter-close window as the eviction counters.
        self._digest_log: list[dict] = []
        #: One entry per turn: the digest text the turn actually sent. The zone probe
        #: reads it to state whether the bounded block carried a specific fact on the
        #: scored turn, which per-section counts cannot answer for one entry. A
        #: 42-turn session holds roughly 210 kilobytes here, and nothing serialises it:
        #: the harness derives booleans and discards the text.
        self._digest_texts: list[str] = []
        #: One entry per turn: the sweep's disposition — ``record``, ``none``,
        #: ``failed``, ``skipped``, or transiently ``pending`` — read by the soak
        #: harness the same way. Index-aligned with ``_digest_log`` and the other
        #: per-turn logs: a turn that dispatches a background sweep (C2) appends
        #: ``pending`` at its own slot immediately, and ``_resolve_pending_sweep``
        #: patches that same slot in place once the task resolves, so the log never
        #: gains an extra entry and every index still means "this turn's own sweep."
        self._sweep_log: list[str] = []
        #: The one in-flight background sweep (C2), or ``None``. A tuple of the task
        #: and the ``_sweep_log`` index to patch when it resolves — never more than
        #: one at a time, because ``run_turn`` is single-turn-at-a-time per engine
        #: (see this module's own docstring) and ``_resolve_pending_sweep`` always
        #: drains the previous one before a new turn can dispatch another.
        self._pending_sweep: tuple[asyncio.Task, int] | None = None
        #: One entry per turn: the tool names the model requested, in request order.
        #: The hook records the request before the per-turn ceiling decides, so a
        #: cancelled call appears here too — the probe asks what the model reached
        #: for, not what returned. The clock probe reads it to separate two
        #: explanations of a correct answer: the injected digest carried the fact, or
        #: the model called a read tool for it. Without the names, a probe measuring
        #: digest loss cannot exclude the second.
        self._tool_log: list[list[str]] = []
        self._tool_names_this_turn: list[str] = []
        #: This turn's call signatures (``tool_call_signature``) and, one entry per
        #: turn, how many repeated an earlier one byte for byte. The signatures are
        #: process-local and dropped at each turn boundary; only the count is
        #: retained, so the diagnostics' no-arguments contract holds. Captured in the
        #: same ``BeforeToolCallEvent`` hook as ``_tool_names_this_turn``, so a
        #: cancelled attempt counts too: the instrument measures what the model tried
        #: to repeat, not only what the server let through.
        self._tool_signatures_this_turn: list[str] = []
        self._duplicate_call_log: list[int] = []
        #: One redacted record per executed tool this turn: its name, ``ok`` flag,
        #: error code, and event id. It carries no arguments and no result content,
        #: so the disposable-campaign transcript can answer "did combat_attack run,
        #: and did it roll" after exit without exposing player or canon data. The
        #: prior diagnostics contract excluded tool traffic entirely, so no retained
        #: artifact recorded whether a die rolled; this closes that gap.
        self._tool_events_this_turn: list[dict] = []
        #: Every roll this turn's own tool results actually returned, as the typed
        #: fields ``roll_facts`` extracts, reset every ``run_turn``. Unlike
        #: ``_tool_events_this_turn`` beside it, this is not a retained diagnostic: it
        #: is the truth side of ``verdict_mismatches``, held only long enough to check
        #: the candidate narration against it before the turn delivers, and it carries
        #: numbers and an attribute name rather than any narration or argument text.
        self._roll_facts_this_turn: list[dict] = []


        self._doom_facts_this_turn: list[dict] = []
        #: One entry per turn: the verdict check's own record -- whether it fired,
        #: whether the corrective re-invoke ran, and whether the replacement resolved
        #: -- read by the soak harness and the combat probe exactly like
        #: ``_repetition_log`` and ``_traversal_log`` beside it.
        self._verdict_log: list[dict] = []
        #: One entry per turn: the meta-turn narration guard's own record, the same
        #: shape ``_verdict_log`` and ``_recovery_log`` keep -- whether a
        #: game-master-discussion candidate narrated an action or named/promised a
        #: refused tool, whether the corrective re-invoke ran, and whether it or the
        #: sentence-drop fallback resolved it.
        self._meta_guard_log: list[dict] = []
        #: One entry per turn: how many roll-under announcements
        #: ``inject_missing_roll_announcements`` had to add because the narration did
        #: not already state them. Zero is the expected common case once
        #: ``skills/bsh-gm/SKILL.md`` stops asking the model to write this line itself;
        #: a nonzero count here, not a withheld turn or a wrong number, is what a
        #: model that still tries and still omits it now looks like, and it is the
        #: number a soak run should watch in this metric's place.
        self._announcement_injection_log: list[dict] = []


        self._request_log: list[dict] = []
        #: One entry per turn: that turn's own token usage, read from the invocation
        #: Strands opens per ``invoke_async`` rather than from the agent-lifetime
        #: accumulator. Characters state what the request holds; these state what the
        #: endpoint charged for it, including the prefix cache hits vLLM reports as
        #: ``cacheReadInputTokens``. ``run_turn`` appends the primary call's own entry
        #: before any of its four corrective re-invokes can run, so each folds its own
        #: usage into that same entry through ``_fold_corrective_usage`` instead of
        #: going uncounted -- the log keeps one entry per turn, and that entry now
        #: prices what the turn actually cost.
        self._usage_log: list[dict] = []
        self._resolution_guard: ResolutionGuard | None = None
        #: Non-empty exactly while a recovery re-narration re-invoke is running: the
        #: refusal ``_corrective_freeze_refusal`` returns for any ``combat_*`` call.
        #: The re-invoked model is asked to narrate an already-rolled mechanic; a
        #: model that answered by attacking again would double the consequences the
        #: ratification barrier keeps singular, so the ``BeforeToolCallEvent`` hook
        #: rejects the whole combat surface for the duration and nothing else.
        self._corrective_tool_freeze: str = ""
        #: True while the main agent invoke of a game-master-discussion turn runs
        #: (``PreparedNarrationTurn.meta``): the ``BeforeToolCallEvent`` hook then
        #: refuses every tool outside ``policy.GM_DISCUSSION_TOOLS`` through
        #: ``_meta_tool_refusal``, which is what makes "nothing mechanical can happen
        #: on an addressed turn" structural rather than a prompt instruction. Set and
        #: cleared in ``run_turn`` beside ``_turn_principal``, with the same
        #: no-leak-across-turns guarantee; the corrective re-invokes after that clear
        #: are not covered, the same accepted bound ``_avoid_repeat`` records.
        self._meta_turn_active: bool = False
        #: The authenticated principal behind the turn currently inside
        #: ``run_turn``'s agent call, or None. ``_bind_character_create_account``
        #: reads it so a ``character_create`` call always links the new character to
        #: the player who actually sent the turn, whatever account identifier the
        #: model typed.
        self._turn_principal = None
        #: Every character's coins, HP, and equipment as of the start of the current
        #: turn, keyed by character id, reset every ``run_turn`` and read by
        #: ``_sweep``'s guard to tell whether a claimed item change actually happened
        #: this turn for the specific character it names. Coin and HP claims verify
        #: against the current value directly and need no snapshot; equipment has no
        #: single figure to check the same way, so the guard checks whether that one
        #: character's own equipment changed.
        self._resources_before_turn: dict = {}
        #: One entry per real ``plan_turn`` call that reached the model: the
        #: resolved ``recent_narration`` block's character length (0 when the
        #: tail was empty). See the append site in ``plan_turn`` for what reads it.
        self._planner_recent_narration_log: list[int] = []


        self._plan_log: list[dict] = []
        #: One entry per ``plan_turn`` call that entered the risk-confirmation branch
        #: (``requires_risk_confirmation(policy)``): whether the classifier had faulted
        #: for this declaration (``policy_none``), whether the effective action was a
        #: planner-authored label (``effective_from_label``), whether either of those
        #: means ``verify_plan`` itself awaits a model call rather than returning
        #: synchronously (``would_yield``), and whether the branch actually reached the
        #: point where ``assess_hazard`` is called (``reached_confirmation``). Measures
        #: how often a speculative ``assess_hazard`` task started beside ``verify_plan``
        #: would have anything to overlap with, and how often it would be consumed
        #: versus dispatched-and-discarded. Gates nothing; a soak run's ``close()`` can
        #: read this the same way it already reads ``_plan_log``.
        self._risk_branch_log: list[dict] = []
        #: One entry per turn: the repetition detector's disposition, in the
        #: ``sweep_log``/``tool_log`` mold. ``triggered`` is False when there was no
        #: prior delivered turn to compare against or the similarity stayed under
        #: threshold. ``resolved`` is None until a retry actually ran, then True
        #: when the retried narration dropped back under threshold and False when
        #: it did not (delivered anyway -- the no-loop bound, never a second retry).
        self._repetition_log: list[dict] = []


        self._recovery_log: list[dict] = []
        #: This channel's own narration-tracking state -- see ``_ChannelNarration``.
        self._narration: dict[str, _ChannelNarration] = {}
        #: One entry per turn: the traversal-monotone fallback's disposition, in
        #: the ``_repetition_log``/``_sweep_log`` mold -- which open traversal
        #: clocks were inspected and which the engine's own fallback commit
        #: advanced this turn.
        self._traversal_log: list[dict] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Open the MCP session, read the tool surface, and assert the policy.

        The surface assertion runs before any channel adapter starts, so a violation
        stops the process rather than reaching a player.
        """
        from mcp import StdioServerParameters, stdio_client
        from strands.tools.mcp import MCPClient

        command = resolved_server_command(self.config)
        self._client = MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command=command[0],
                    args=list(command[1:]),
                    env=server_env(self.config),
                )
            )
        )
        self._client.__enter__()
        try:
            self._tools = self._client.list_tools_sync()
            self._system_prompt = assemble_system_prompt(self.config.repo_root)
        except Exception:
            # The session and its server child are already live. Leaking them on a
            # startup failure would make the security stop the one path that orphans a
            # process, which is the opposite of what it is for.
            self.stop()
            raise
        # The server must publish exactly the served surface: the model-facing set plus
        # the two engine-only tools. A server missing ``ledger_settle`` or
        # ``ability_apply_ruling`` would leave the settle or adjudicate step silently
        # broken, so it fails here instead.
        served = {
            getattr(tool, "tool_name", None) or getattr(tool, "name", "")
            for tool in self._tools
        }
        if served != policy.MCP_SERVED_TOOLS:
            self.stop()
            raise policy.ToolSurfaceError(
                f"server tools differ from the served manifest: "
                f"unexpected {sorted(served - policy.MCP_SERVED_TOOLS)}, "
                f"missing {sorted(policy.MCP_SERVED_TOOLS - served)}"
            )

        # The served names are not the agent's names. Only ``Agent.tool_names``
        # observes the skills plugin's contribution and the engine-only filter, so
        # build one agent here and discard it. Asserting inside ``_agent_for`` deferred
        # the check to a channel's first turn: an audit found that a channel yielding
        # zero turns never ran it. Construction opens no socket, so the discarded agent
        # costs one object.
        try:
            self._build_agent()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Close the MCP session so the server observes end of file and exits.

        Best-effort only for a pending background sweep (C2): this method is
        synchronous and callable from a context with no running event loop (see
        ``__init__``'s own error-handling paths below), so it can only cancel a
        pending sweep task here, not await it -- cancelling loses whatever fact
        that sweep would have committed. A caller that can await should call
        ``flush_pending_sweep()`` first; ``NarratorService.run()``'s own shutdown
        does, which is the path that matters for a live table.
        """
        self._agents.clear()
        if self._pending_sweep is not None:
            self._pending_sweep[0].cancel()
            self._pending_sweep = None
        if self._client is not None:
            self._client.__exit__(None, None, None)
            self._client = None

    # -- agent construction --------------------------------------------------

    def _agent_for(self, channel_id: str):
        """One agent per channel. Sessions are shared per channel, never per user."""
        if channel_id in self._agents:
            return self._agents[channel_id]
        self._agents[channel_id] = self._build_agent()
        return self._agents[channel_id]

    def _model(
        self,
        max_tokens: int | None = None,
        origin: str = "turn",
        temperature: float | None = None,
        request_overrides=None,
    ):
        from narrator.model import NarratorOpenAIModel

        return NarratorOpenAIModel(
            client_args={"base_url": self.config.base_url, "api_key": "not-required"},
            model_id=self.config.model_id,
            params={
                "temperature": self.config.temperature if temperature is None else temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
            },
            recorder=lambda row: self._record_request(origin, row),
            usage_recorder=lambda usage: self._record_request_usage(origin, usage),
            request_overrides=request_overrides,
        )

    @property
    def turn_thinking_level(self) -> str:
        """The channel agent's thinking level in force, one of ``THINKING_LEVELS``."""
        return self._turn_thinking_level

    def set_turn_thinking_level(self, level: str) -> str:
        """Apply ``level`` to every channel agent from its next request on.

        Returns the level as applied. Raises ``ValueError`` for a name outside
        ``THINKING_LEVELS`` and changes nothing in that case. Agents already built keep
        their conversation: only their system prompt is re-pointed here, and their
        model reads ``_turn_request_overrides`` afresh on each request.
        """
        normalized = str(level or "").strip().lower()
        if normalized not in THINKING_LEVELS:
            raise ValueError(
                f"thinking level must be one of {', '.join(THINKING_LEVELS)}; got {level!r}"
            )
        self._turn_thinking_level = normalized
        prompt = self._turn_system_prompt()
        for agent in self._agents.values():
            agent.system_prompt = prompt
        return normalized

    def _turn_system_prompt(self) -> str:
        """The channel agent's system prompt: the assembled one, plus the thinking
        section only when the *session* asked for thinking and the language section
        only when the table's catalog is not English, so an English, thinking-off
        session sends the exact bytes every recorded measurement used."""
        prompt = self._system_prompt
        if self._thinking_section_applies():
            prompt += THINKING_SECTION
        tag = self._active_language_tag()
        if tag:
            prompt += language_directive(tag)
        return prompt

    def _thinking_section_applies(self) -> bool:
        """Whether the system prompt carries ``THINKING_SECTION``.

        The *session's* level decides this, never ``_effective_thinking_level``, and the
        difference is the whole point of this method. Two unrelated facts used to share
        one value: \"this table configured thinking off\", and \"this engine is re-running
        one attempt with thinking suppressed because a thought overran its budget\"
        (``_invoke_channel_agent``). Only the first is a reason to drop the section.

        A session configured at ``off`` is untouched: it never reaches the fallback at
        all (``_invoke_channel_agent`` re-raises at ``off``), and its prompt is the
        same bytes it has always been.
        """
        return self._turn_thinking_level != "off"

    def _active_language_tag(self) -> str:
        """The table's catalog tag, or ``""`` for English.

        Shared by ``_turn_system_prompt`` and the per-turn prompt: a live probe
        (three turns, ``fr-FR``) found the system-prompt instruction alone, though
        confirmed present in the request, did not move this model off English
        narration -- so the per-turn reminder is not a belt-and-braces duplicate,
        it is the part carrying the effect. See ``narrator.prompt.turn_prompt``'s
        ``language_tag`` docstring note.
        """
        tag = self.config.catalog.language
        return "" if tag.startswith("en") else tag

    def _effective_thinking_level(self) -> str:
        """The level this *request* runs at: the session's, or off during a fallback.

        Read by ``_turn_request_overrides`` and by the request log. The system prompt
        deliberately does not read it; see ``_thinking_section_applies``.
        """
        return "off" if self._thinking_suppressed else self._turn_thinking_level

    def _turn_request_overrides(self) -> dict:
        """The request keys the channel agent's model writes for the level in force.

        Empty at ``off``, which leaves the request byte-identical to the one the
        narrator has always sent. Otherwise the call's ``max_tokens`` tightens to the
        budget plus an answer allowance, and two vLLM request fields ride in
        ``extra_body``, which the OpenAI client merges into the JSON body:
        ``chat_template_kwargs.enable_thinking`` switches Gemma 4's chat template from
        prefilling an empty thought to inviting one, and ``thinking_token_budget`` is
        the ceiling the endpoint enforces by forcing the thought channel closed -- the
        model then still answers, where a ``max_tokens`` cut would have returned
        nothing. Only this origin ever carries them; see
        ``NarratorConfig.turn_thinking_level`` for why the other lanes never think.
        """
        level = self._effective_thinking_level()
        if level == "off":
            return {}
        budget = self.config.turn_thinking_budget(level)
        return {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_token_budget": budget,
            },
            # Bounds a post-thought content loop at seconds rather than the minute the
            # turn's own ceiling allows; see ``DEFAULT_THINKING_ANSWER_TOKENS``.
            "max_tokens": min(
                self.config.max_tokens, budget + self.config.turn_thinking_answer_tokens
            ),
        }

    def _record_request(self, origin: str, row: dict) -> None:
        """Log one assembled request's payload composition against its turn.

        The turn number is the digest log's length, because ``run_turn`` appends this
        turn's digest before any request leaves. The channel agent's model is built once
        per channel, so the origin has to be bound at construction while the turn number
        is read at request time.
        """
        thinking = self._effective_thinking_level() if origin == "turn" else "off"
        self._request_log.append(
            {"turn": len(self._digest_log), "origin": origin, "thinking": thinking, **row}
        )

    def _record_request_usage(self, origin: str, usage: dict) -> None:
        """Attach one request's own usage to the row that request created.

        The model layer reports usage after the request it belongs to, and requests on
        one model instance never overlap, so the newest row of that origin still
        awaiting usage is that request's row. A usage record arriving with no such row
        is dropped rather than guessed at, which keeps a stray from renaming another
        request's measurement.

        This is the per-request reading the agent-level accumulator cannot give.
        ``_turn_usage`` sums the channel agent's calls for a turn, so its cache figure
        covers the turn request and any tool-loop continuation together. It never covers
        the settle or sweep requests, which ride their own model instances, and this
        recorder logs those under their own origin.
        """
        for row in reversed(self._request_log):
            if row.get("origin") == origin and "usage" not in row:
                row["usage"] = {
                    key: int(value)
                    for key, value in usage.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
                return

    def _append_tool_log(self) -> None:
        """Close one turn's tool record: the names, and how many calls repeated.

        Both delivery paths call this exactly once per turn -- the fault branch and
        the delivered branch -- so ``_tool_log`` and ``_duplicate_call_log`` stay
        aligned to the same turn index the other per-turn logs use.
        """
        self._tool_log.append(list(self._tool_names_this_turn))
        self._duplicate_call_log.append(duplicate_call_count(self._tool_signatures_this_turn))

    def _fold_corrective_usage(self, result) -> None:
        """Add one corrective re-invoke's usage into this turn's already-logged entry.

        ``run_turn`` appends this turn's usage (``_turn_usage(result)`` on the primary
        call) before any of its four corrective re-invokes can fire -- the meta-leak
        guard, the repetition nudge, the verdict correction, the combat-recovery
        renarration -- so each of those calls ``agent.invoke_async`` again on the same
        agent and, unfolded, that cost never reached ``_usage_log`` at all. This adds
        it into the entry already there instead of appending a second one, so
        ``_usage_log`` keeps exactly one entry per turn and that entry now prices the
        turn's true cost. Reads ``_usage_log`` through ``getattr`` rather than a plain
        attribute access and is guarded on it being non-empty: both hold by
        construction for every ``run_turn`` caller, which all run after ``run_turn``'s
        own primary-call append, but the four corrective methods this feeds are also
        driven in ``tests/narrator_units_shared.py``'s own tests against a bare
        ``NarratorEngine.__new__`` instance that never runs ``__init__`` at all, and a
        raise here must never turn a successful correction into a reported failure.
        """
        usage_log = getattr(self, "_usage_log", None)
        if not usage_log:
            return
        addition = _turn_usage(result)
        if not addition:
            return
        folded = dict(usage_log[-1])
        for key, value in addition.items():
            folded[key] = folded.get(key, 0) + value
        usage_log[-1] = folded

    def _build_agent(self):
        """Construct one channel agent and assert its tool surface.

        ``start`` calls this once for the ordering guarantee and discards the agent.
        ``_agent_for`` calls it again per channel, so every agent a player reaches has
        had its own surface asserted rather than inheriting a startup probe's verdict.
        """
        from strands import Agent, AgentSkills
        from strands.agent.conversation_manager import SlidingWindowConversationManager
        from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent

        # The engine-only tools must never register on the model's agent. A
        # prompt-injected waive would erase pending outcomes, and a prompt-injected
        # ruling would resolve a demon's theft, both from the scene record.
        model_facing_tools = [
            tool
            for tool in self._tools
            if (getattr(tool, "tool_name", None) or getattr(tool, "name", ""))
            not in policy.ENGINE_ONLY_TOOLS
        ]

        # Skills ship with the checkout, not with campaign state. Pointing these at
        # ``campaign_root`` silently loaded nothing and the agent ran skill-less.
        skill_paths = [
            str(self.config.repo_root / relative) for relative in DISCLOSED_SKILLS
        ]
        agent = Agent(
            model=self._model(request_overrides=self._turn_request_overrides),
            tools=model_facing_tools,
            plugins=[AgentSkills(skills=skill_paths)],
            system_prompt=self._turn_system_prompt(),
            conversation_manager=SlidingWindowConversationManager(
                window_size=self.config.window_size
            ),
            callback_handler=None,
        )

        def _enforce_ceiling(event: BeforeToolCallEvent) -> None:
            self._calls_this_turn += 1
            try:
                self._tool_names_this_turn.append(str(event.tool_use.get("name", "")))
                self._tool_signatures_this_turn.append(
                    tool_call_signature(
                        str(event.tool_use.get("name", "")),
                        event.tool_use.get("input") or {},
                    )
                )
            except Exception:  # noqa: BLE001 - a measurement never fails a turn
                self._tool_names_this_turn.append("")
            if self._calls_this_turn > self.config.max_tool_calls_per_turn:
                event.cancel_tool = (
                    f"per-turn tool-call ceiling of "
                    f"{self.config.max_tool_calls_per_turn} reached"
                )
                return
            freeze_refusal = self._corrective_freeze_refusal(
                str((getattr(event, "tool_use", {}) or {}).get("name", ""))
            )
            if freeze_refusal:
                event.cancel_tool = freeze_refusal
                return
            meta_refusal = self._meta_tool_refusal(
                str((getattr(event, "tool_use", {}) or {}).get("name", ""))
            )
            if meta_refusal:
                event.cancel_tool = meta_refusal
                return
            called_name = str((getattr(event, "tool_use", {}) or {}).get("name", ""))
            if called_name and called_name not in policy.PLAYER_FACING_TOOLS:


                event.cancel_tool = (
                    f"unknown tool {called_name!r}; no server has ever served it. "
                    f"The served tools are: {', '.join(sorted(policy.PLAYER_FACING_TOOLS))}."
                )
                return
            self._bind_character_create_account(event)
            guard = self._resolution_guard
            if guard is not None:
                tool_use = getattr(event, "tool_use", {}) or {}
                arguments = tool_use.get("input") or tool_use.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}
                refusal = guard.validate(str(tool_use.get("name", "")), arguments)
                if refusal:
                    event.cancel_tool = refusal

        agent.hooks.add_callback(BeforeToolCallEvent, _enforce_ceiling)

        def _record_result(event: AfterToolCallEvent) -> None:
            tool_use = getattr(event, "tool_use", {}) or {}
            name = str(tool_use.get("name", ""))
            failed = event.exception is not None
            payload = {} if failed else self._tool_result_payload(getattr(event, "result", {}) or {})
            redacted = redact_tool_event(name, exception=failed, payload=payload)
            if name and name not in policy.PLAYER_FACING_TOOLS:
                # Distinguishes a hallucinated name from a genuine server-side raise
                # in the retained transcript. Reached whether the cancellation above
                # caught it (``failed`` False, empty payload) or, as a backstop, a
                # name reaches here some other way and Strands' own registry lookup
                # raised its bare "Unknown tool" exception instead (``failed`` True).
                redacted = {**redacted, "ok": False, "error": "unknown_tool"}
            self._tool_events_this_turn.append(redacted)


            self._roll_facts_this_turn.extend(roll_facts(name, payload))
            self._doom_facts_this_turn.extend(doom_facts(name, payload))
            guard = self._resolution_guard
            if guard is None or failed:
                return
            arguments = tool_use.get("input") or tool_use.get("arguments") or {}
            if not isinstance(arguments, dict):
                return
            guard.record_tool_result(name, arguments, payload)

        agent.hooks.add_callback(AfterToolCallEvent, _record_result)

        # The guarantee, asserted against what the agent actually holds. This is the
        # only check that observes both the plugin's contribution and the engine-only
        # filter rather than assuming them.
        policy.assert_tool_surface(set(agent.tool_names))
        return agent

    def _bind_character_create_account(self, event) -> None:
        """Overwrite a ``character_create`` call's account id with the turn's own.

        The account identifier decides which player a new character answers to for
        the rest of the campaign, and the model's copy of it is hearsay: it can only
        repeat what a prompt showed it, and a prompt-injected turn could type someone
        else's. The engine knows the authenticated principal of the turn it is
        running, so the binding is taken from there, unconditionally, whenever one
        exists. A turn with no principal (an adapter or test that stamps none)
        leaves the model's argument alone rather than blanking it. Never raises: a
        binding failure must not cost the turn, and the server still validates
        whatever arrives.
        """
        try:
            tool_use = getattr(event, "tool_use", {}) or {}
            if str(tool_use.get("name", "")) != "character_create":
                return
            subject = str(getattr(self._turn_principal, "subject_id", "") or "")
            if not subject:
                return
            for key in ("input", "arguments"):
                arguments = tool_use.get(key)
                if isinstance(arguments, dict):
                    arguments["discord_user_id"] = subject
        except Exception:  # noqa: BLE001 - the server-side validation still stands
            return

    @staticmethod
    def _tool_result_payload(result) -> dict:
        """Read an MCP JSON envelope without treating text as a successful test."""
        if not isinstance(result, dict):
            dump = getattr(result, "model_dump", None)
            if callable(dump):
                try:
                    result = dump(by_alias=True)
                except TypeError:
                    result = dump()
            if not isinstance(result, dict):
                structured = getattr(result, "structuredContent", None)
                if not isinstance(structured, dict):
                    structured = getattr(result, "structured_content", None)
                if isinstance(structured, dict):
                    return structured
                content = getattr(result, "content", None)
                result = {"content": content} if isinstance(content, list) else {}
        structured = result.get("structuredContent")
        if not isinstance(structured, dict):
            structured = result.get("structured_content")
        if isinstance(structured, dict):
            return structured
        content = result.get("content", []) if isinstance(result, dict) else []
        if not isinstance(content, list):
            return {}
        for item in content:
            if not isinstance(item, dict):
                continue
            candidate = item.get("json")
            if isinstance(candidate, dict):
                return candidate
            text = item.get("text")
            if isinstance(text, str):
                try:
                    candidate = json.loads(text)
                except ValueError:
                    continue
                if isinstance(candidate, dict):
                    return candidate
        return {}

    async def _settle_once(self, prompt: str):
        """One structured settle request through the model layer's parse endpoint.
        """
        return await self._structured_once(
            prompt, SettleOutcome, system_prompt=SETTLER_SYSTEM_PROMPT,
            max_tokens=self.config.settle_max_tokens, origin="settle",
        )

    async def _structured_once(
        self,
        prompt: str,
        schema: type,
        *,
        system_prompt: str,
        max_tokens: int,
        origin: str,
        temperature: float | None = None,
    ):
        """One tool-less, schema-forced request on the model layer's parse endpoint.

        Every structured step this engine runs -- settle, assess, classify, sweep, plan,
        escalate, adjudicate -- is this one request shape with a different schema,
        system prompt, token ceiling and diagnostic origin. Before this method each
        step carried its own copy of the same six-line event loop; the named
        ``_x_once`` methods that remain are the seams the offline suites stub, and each
        now delegates here. ``origin`` tags the request in the payload log so the
        per-step measurements stay separable.
        """
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        model = self._model(max_tokens=max_tokens, origin=origin, temperature=temperature)
        async for event in model.structured_output(schema, messages, system_prompt=system_prompt):
            output = event.get("output")
            if output is not None:
                return output
        return None

    async def assess_hazard(
        self, declaration: str, *, scope=None, combat=None
    ) -> HazardAssessment | None:
        """One typed speech-act verdict for a lexically flagged declaration.

        Called by this engine's own risk floor and by ``NarratorService``'s, in both
        cases only after the typed rules have kept a confirmation. Returns ``None``
        on timeout, parse failure, or any transport fault -- the collapsed value
        ``narrator.assess.assessment_downgrades`` reads as "keep the confirmation" --
        so the floor can never end up weaker than the lexical floor alone.

        Temperature is pinned to 0.0: this is a classification, and the live probe
        that accepts it (``tests_narrator/test_probe_hazard_assessment.py``) measures
        the greedy decode, so production must run the decode the probe measured.

        An unstarted engine answers ``None`` without building a model: the offline
        suites drive ``plan_turn`` on engines that never called ``start()``, and the
        suite contract says they reach no live endpoint. A started engine is the one
        configuration that serves players, and it is the one that assesses.
        """
        if self._client is None:
            return None
        try:
            prompt = assessment_prompt(declaration, scope, combat)
            return await asyncio.wait_for(
                self._assess_once(prompt), timeout=self.config.assess_timeout_seconds
            )
        except Exception:  # noqa: BLE001 - every assessor fault keeps the confirmation
            return None

    async def _assess_once(self, prompt: str):
        """One structured assessment request, on the settle step's exact model path."""
        return await self._structured_once(
            prompt, HazardAssessment, system_prompt=ASSESSOR_SYSTEM_PROMPT,
            max_tokens=self.config.assess_max_tokens, origin="assess", temperature=0.0,
        )

    async def _policy_for_planning(self, declaration: str, scope, combat):
        """The routing verdict every hazard decision in ``plan_turn`` shares.

        ``None`` means the classifier could not answer, and every consumer below reads
        that as "confirm": ``planning_bypass`` refuses to bypass, and
        ``requires_risk_confirmation`` returns True. The service already withheld the
        turn once on an unreadable classification, so reaching here with ``None`` means
        the classifier answered for the turn and then failed for this declaration --
        rare, and still not a reason to let an unread declaration through.
        """
        from narrator.classify import policy_from

        classification = await self.classify_intent(declaration, scope=scope, combat=combat)
        if classification is None:
            return None
        return policy_from(classification, scope=scope, combat=combat)

    async def classify_intent(
        self, declaration: str, *, scope=None, combat=None, offer_open: bool = False
    ):
        """One typed routing verdict for a player message, in any language.

        This is the replacement for ``narrator.interactions.classify_turn``'s English
        lexical ladder, and it fires on every addressed turn rather than only on a
        flagged one. ``None`` collapses every failure -- timeout, parse failure,
        transport fault, unstarted engine -- and a ``None`` verdict routes no turn:
        ``NarratorService`` posts ``config.classifier_fault_notice`` and resolves
        nothing. Failing closed is the whole contract, because the alternative the
        lexicon offered was measured worse than nothing on the adversarial corpus.

        Temperature is pinned to 0.0 for the same reason ``assess_hazard`` pins it:
        this is a classification, and ``tests_narrator/test_probe_classifier.py``
        measures the greedy decode, so production must run the decode the probe
        measured.

        An unstarted engine answers ``None`` without building a model, matching
        ``assess_hazard``. Offline suites therefore supply their own classifier rather
        than reaching an endpoint; see ``NarratorService._classify``'s duck-typed seam.

        The typed state also selects up to four narrow companions, composed
        declaratively in ``narrator.classify.BESIDE_COMPANIONS``/``AFTER_COMPANIONS``.
        A beside companion (defence, persons) is started concurrently with the full
        call, so it costs no latency; an after companion (hazard, route) runs once the
        full verdict is known, only when that verdict claims something the engine
        cannot see or routes the turn somewhere a second question is owed. Each merges
        monotonically and each fails open to the full verdict: a companion fault is
        today's behavior, never a withheld turn. Only the full call fails closed.
        """
        if self._client is None:
            return None
        context = ClassificationContext(scope=scope, combat=combat, offer_open=offer_open)
        record = {companion.key: "skipped" for companion in (*BESIDE_COMPANIONS, *AFTER_COMPANIONS)}
        tasks: dict[str, asyncio.Task] = {}
        try:
            for companion in BESIDE_COMPANIONS:
                if companion.applies(context):
                    tasks[companion.key] = asyncio.ensure_future(
                        self._companion_once(
                            getattr(self, companion.classifier_attr), declaration, context,
                            origin=companion.origin,
                        )
                    )
            prompt = self.turn_classifier.prompt(declaration, context)
            verdict = await asyncio.wait_for(
                self._classify_once(prompt), timeout=self.config.classify_timeout_seconds
            )
        except Exception:  # noqa: BLE001 - every classifier fault withholds the turn
            for task in tasks.values():
                task.cancel()
            record["full"] = "fault"
            self._companion_log.append(record)
            return None
        if verdict is None:
            for task in tasks.values():
                task.cancel()
            record["full"] = "empty"
            self._companion_log.append(record)
            return None
        for companion in BESIDE_COMPANIONS:
            task = tasks.get(companion.key)
            if task is None:
                continue
            companion_result = await task
            merged = companion.merge(verdict, companion_result, context)
            record[companion.key] = (
                "fault" if companion_result is None
                else companion.changed_label if merged is not verdict
                else "none"
            )
            verdict = merged
        # Both after companions can apply on one verdict (a question-shaped route that
        # also claims an acceptance the engine cannot see): today's C4 fix dispatches
        # both together instead of one after the other. Safe because ``applies`` and
        # ``merge`` are field-disjoint in the way that matters -- ``hazard_companion_applies``
        # reads hazard/accepts_offer/named_person_ids/names_unrecorded_person and
        # ``route_companion_applies`` reads only ``route``, and neither merge touches a
        # field the other's ``applies`` reads (``merge_hazard`` writes only ``hazard``,
        # ``merge_route`` writes only ``also_declares_act``/``declared_act_kind`` and
        # never ``route``) -- so evaluating both ``applies`` calls against the
        # pre-after-merge verdict, before either merge runs, reproduces the sequential
        # reading exactly. Both firing at once is rare; the two-pass shape mirrors
        # ``BESIDE_COMPANIONS`` above rather than introducing a second pattern.
        after_tasks: dict[str, asyncio.Task] = {}
        for companion in AFTER_COMPANIONS:
            if companion.applies(verdict, context):
                after_tasks[companion.key] = asyncio.ensure_future(
                    self._companion_once(
                        getattr(self, companion.classifier_attr), declaration, context,
                        origin=companion.origin,
                    )
                )
        for companion in AFTER_COMPANIONS:
            task = after_tasks.get(companion.key)
            if task is None:
                continue
            companion_result = await task
            merged = companion.merge(verdict, companion_result, context)
            record[companion.key] = (
                "fault" if companion_result is None
                else companion.changed_label if merged is not verdict
                else "none"
            )
            verdict = merged
        self._companion_log.append(record)
        return verdict

    async def _companion_once(
        self, classifier: TurnClassifier, declaration: str, context: ClassificationContext, *, origin: str
    ):
        """One narrow companion request. Never raises: every fault answers ``None``.

        Same model path, same greedy decode and the same timeout as the full call; the
        difference is the schema, which is the whole point.
        """
        try:
            prompt = classifier.prompt(declaration, context)
            return await asyncio.wait_for(
                self._structured_once(
                    prompt, classifier.schema, system_prompt=classifier.system_prompt,
                    max_tokens=self.config.classify_max_tokens, origin=origin, temperature=0.0,
                ),
                timeout=self.config.classify_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - a companion fault is the full verdict, unchanged
            return None

    async def _classify_once(self, prompt: str):
        """One structured classification request, on the assessor's exact model path."""
        return await self._structured_once(
            prompt, self.turn_classifier.schema,
            system_prompt=self.turn_classifier.system_prompt,
            max_tokens=self.config.classify_max_tokens, origin="classify", temperature=0.0,
        )

    async def _sweep_once(self, prompt: str):
        """One structured sweep request, on the settle step's exact model path."""
        return await self._structured_once(
            prompt, SweepIntroducingOutcome, system_prompt=SWEEPER_SYSTEM_PROMPT,
            max_tokens=self.config.sweep_max_tokens, origin="sweep",
        )

    async def _plan_once(self, prompt: str, *, repair: bool = False):
        """Run one tool-less decision plan on the existing guided-output model path."""
        return await self._structured_once(
            prompt, PlanOutcome, system_prompt=PLANNER_SYSTEM_PROMPT,
            max_tokens=self.config.decision_max_tokens,
            origin="decision_repair" if repair else "decision_plan",
            temperature=0.0 if repair else None,
        )

    async def _escalate_once(self, declaration: str, canon_text: str) -> ApproachDraft | None:
        """One schema-narrowed replan, forced to ``ApproachDraft`` alone.

        No repair attempt and no fallback to the full ``PlanOutcome`` union: the
        caller (``_escalate_promised_check``) already treats any failure here as
        fail-open, so a second try would only spend another call for the same
        expected outcome. Returns ``None`` on anything but a clean parse --
        schema validation failure, timeout, or an empty stream alike -- and never
        raises, matching ``_sweep``'s fail-open contract for the same reason: this
        repairs fiction-level text, never durable state.
        """
        prompt = escalation_prompt(declaration, canon_text)
        try:
            return await self._structured_once(
                prompt, ApproachDraft, system_prompt=ESCALATION_SYSTEM_PROMPT,
                max_tokens=self.config.decision_max_tokens, origin="decision_plan_escalation",
            )
        except Exception:  # noqa: BLE001 - fail-open, mirrors ``_sweep``
            return None

    async def _escalate_promised_check(self, declaration: str, canon_text: str) -> tuple[PlanOutcome | None, str]:
        """Resolve one promised-unbound-check escalation. Returns ``(replacement,
        disposition)``: ``replacement`` is ``None`` when the caller should keep its
        original result (a failed escalation, fail-open); otherwise it is what
        ``plan_turn`` should return instead. ``disposition`` is one of ``bound``,
        ``collapsed``, or ``failed``, for ``_plan_log``.
        """
        try:
            draft = await asyncio.wait_for(
                self._escalate_once(declaration, canon_text), timeout=self.config.settle_timeout_seconds,
            )
        except Exception:  # noqa: BLE001 - fail-open on timeout too
            draft = None
        if draft is None:
            return None, "failed"
        if any(option.directive.kind != "no_test" for option in draft.approaches):
            return PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=draft)), "bound"
        return PlanOutcome(plan=ProceedPlan(kind="proceed")), "collapsed"

    def _recent_narration_for(self, channel_id: str) -> str:
        """This channel's accumulated uncommitted-narration tail, oldest first."""
        channel = self._narration.get(channel_id)
        return "\n\n".join(channel.uncommitted) if channel is not None else ""

    def _extend_uncommitted_narration(
        self, channel_id: str, narration: str, *, scene_changed: bool
    ) -> None:
        """Append a delivered turn's narration, or clear the tail a commit made stale.

        Called only for a turn that actually delivered narration (``run_turn``
        gates on ``deliverable and narration.strip()`` before calling this);
        every other turn -- withheld, faulted, or blank -- leaves the tail exactly
        as it was, the same posture ``_ChannelNarration.last_delivered`` already takes.
        Trimming drops the oldest entries first, keeping at least one no matter
        how far over ``UNCOMMITTED_NARRATION_MAX_CHARS`` a single turn's own
        narration runs alone, so one long turn is truncated by staying whole and
        eventually aging out rather than being silently dropped to nothing.
        """
        if scene_changed:
            # Something committed this turn -- the model's own scene_commit, a
            # settle commit, or the sweep. The digest already carries everything
            # up to this point, so the tail's job is done. Only the tail clears:
            # this channel's other narration fields (last delivered, last
            # declaration) are unrelated facts and stay exactly as they were.
            channel = self._narration.get(channel_id)
            if channel is not None:
                channel.uncommitted = []
            return
        buffer = self._narration.setdefault(channel_id, _ChannelNarration()).uncommitted
        buffer.append(narration)
        total = sum(len(entry) for entry in buffer)
        while len(buffer) > 1 and total > UNCOMMITTED_NARRATION_MAX_CHARS:
            total -= len(buffer.pop(0))

    async def plan_turn(
        self, turn: InboundTurn, eligible_characters, prior_resolutions=(), session=None,
        interaction_cue=None, recent_narration: str | None = None, policy=None,
        retry: bool = False,
    ) -> PlanOutcome | DeclinedPlan:
        """Return the pre-action plan without constructing or invoking an agent.

        ``recent_narration`` defaults to this channel's own tracked
        ``_ChannelNarration.uncommitted`` tail, the same way ``scope`` below resolves
        itself from the campaign rather than requiring every caller to supply it.
        Passing an explicit string (including ``\"\"``) overrides that resolution,
        which is what lets a test exercise a specific narration tail without
        first driving ``run_turn`` through the exact turns that would produce it.

        ``policy`` is the routing verdict the service already computed for this turn's
        own text. It is reused below whenever the action being planned *is* that text,
        which is every first round; before this parameter existed the engine
        re-classified the identical declaration the service had classified a moment
        earlier -- a second ~0.6s model call per planner-routed turn that could only
        agree or, on a stochastic case, disagree with the verdict the turn was already
        routed on. A later round whose effective action is a planner-authored label
        still classifies that label, exactly as before.
        """
        if recent_narration is None:
            recent_narration = self._recent_narration_for(turn.channel_id)
        if self.config.max_decision_rounds <= 0:
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))
        session = session or ProgressiveDecisionSession.start(turn.mention.text)
        declaration = session.effective_action


        scope = read_trusted_scope(self.config.campaign_root)
        # The risk floor reads the open fight: attacking a recorded combatant is the
        # fight's mechanic, not an assault on a bystander, so it needs no confirmation.
        combat = read_combat_snapshot(self.config.campaign_root)
        # One classification serves every decision below. Before the model classifier,
        # this branch ran the lexical ``classify_turn`` three to five times over the same
        # declaration -- once inside ``planning_bypass``, once inside
        # ``requires_risk_confirmation``, and once or twice more inside ``verify_plan`` --
        # on top of the one the service had already run in its turn loop. Classifying
        # once and threading the verdict is what keeps the model call affordable here,
        # and the service's own verdict is reused when it judged this same text.
        if policy is None or declaration != turn.mention.text:
            policy = await self._policy_for_planning(declaration, scope, combat)
        if planning_bypass(policy):
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))


        if (
            policy is not None
            and policy.bare_answer
            and narration_invites_reply(recent_narration or "")
        ):
            return PlanOutcome(plan=ProceedPlan(kind="proceed"))
        if requires_risk_confirmation(policy):
            # Measurement only, ahead of a proposed change: ``policy is None`` is the
            # one condition under which ``verify_plan`` itself awaits a model call at
            # this call site (the effective action always equals the declaration here,
            # so its own re-classification only fires on that branch or on
            # ``effective_from_label``'s second, conditional one) -- see
            # ``_risk_branch_log``'s own docstring.
            policy_was_none = policy is None
            would_yield = policy_was_none or session.effective_from_label
            floor = await verify_plan(
                PlanOutcome(plan=ProceedPlan(kind="proceed")), declaration, session,
                combat=combat, scope=scope, classify=self.classify_intent, policy=policy,
            )


            reached_confirmation = (
                isinstance(floor, PlanOutcome)
                and isinstance(floor.plan, RequestDecisionPlan)
                and isinstance(floor.plan.decision, ConfirmationDraft)
            )
            self._risk_branch_log.append({
                "policy_none": policy_was_none,
                "effective_from_label": session.effective_from_label,
                "would_yield": would_yield,
                "reached_confirmation": reached_confirmation,
            })
            if reached_confirmation and assessment_downgrades(
                await self.assess_hazard(declaration, scope=scope, combat=combat)
            ):
                return PlanOutcome(plan=ProceedPlan(kind="proceed"))
            return floor
        try:
            # The planner reads the public digest, never the narrator's. Every planner
            # output except ``proceed`` becomes player-visible decision text, so it must
            # not carry hidden location canon or an NPC's unrevealed role and faction.
            digest = canon.render_digest(
                self.config.campaign_root,
                self.config.repo_root,
                self.config.canon_scene_max_chars,
                self.config.canon_location_max_chars,
                self.config.canon_npcs_max_chars,
                public=True,
            )
            prompt = planning_prompt(
                turn.channel_text(),
                digest.text,
                tuple(eligible_characters),
                tuple(prior_resolutions),
                effective_action=session.effective_action,
                closed_dimensions=tuple(session.closed_dimensions),


                prior_questions=session.asked_questions,


                interaction_cue=interaction_cue,
                # This channel's uncommitted-narration tail (``_recent_narration_for``
                # above), so the planner stops drafting a clarification that denies a
                # fact the model itself narrated a turn or two ago. See
                # ``narrator.decisions.planning_prompt``'s own docstring.
                recent_narration=recent_narration,
            )
            # Measures the actual per-call cost of the ``recent_narration`` block:
            # 0 on every call where the tail was empty (the common case whenever
            # the record is current), the block's own character length on every
            # call where it fired. Mirrors ``_digest_log``'s one-entry-per-call
            # mold; a soak run's ``close()`` can read this the same way it already
            # reads ``_digest_log``/``_sweep_log``.
            self._planner_recent_narration_log.append(len(recent_narration))
        except Exception as error:  # noqa: BLE001 - unreadable planning inputs fail closed
            self._plan_log.append(_plan_failure_entry("planner_inputs", disposition="fault"))
            raise DecisionPlanningError(
                "planner inputs are unavailable", "planner_inputs"
            ) from error
        for repair in (False, True):
            try:
                outcome = await asyncio.wait_for(
                    self._plan_once(planner_repair_prompt(prompt) if repair else prompt, repair=repair),
                    timeout=self.config.settle_timeout_seconds,
                )
                if outcome is None:
                    raise PlannerPolicyError("planner returned no proposal")
                proposal = PlanOutcome.model_validate(outcome)
                result = await verify_plan(
                    proposal, declaration, session, combat=combat, scope=scope,
                    classify=self.classify_intent,
                )
                entry = _plan_log_entry(result)


                if entry["promised_unbound_check"]:
                    replacement, disposition = await self._escalate_promised_check(
                        declaration, digest.text
                    )
                    entry["escalation"] = disposition
                    if replacement is not None:
                        result = replacement
                        entry["final_kind"] = _plan_log_entry(result)["kind"]
                    else:
                        entry["final_kind"] = entry["kind"]
                else:
                    entry["escalation"] = "not_triggered"
                    entry["final_kind"] = entry["kind"]
                self._plan_log.append(entry)
                return result
            except Exception as error:  # noqa: BLE001 - one deterministic repair only
                if not repair:
                    continue
                reason = _planner_failure_reason(error)
                if retry:
                    # One bounded fallback, the same mold as ``_escalate_promised_check``
                    # above: the narrowed re-ask has already been spent (that is what
                    # ``repair`` was), so instead of raising a second identical fault at
                    # a player who typed exactly what the notice recommended, fall back
                    # to the engine's own floor. It is the identical composition the
                    # risk branch runs, so a hazard still draws its confirmation and
                    # nothing bypasses ``verify_plan``; only the planner's narrowing
                    # questions are given up. A floor that itself raises falls through
                    # to the service's generic planner fault, which bounds this to one
                    # extra call.
                    self._plan_log.append(_plan_failure_entry(reason, disposition="floor"))
                    return await verify_plan(
                        PlanOutcome(plan=ProceedPlan(kind="proceed")), declaration, session,
                        combat=combat, scope=scope, classify=self.classify_intent,
                        policy=policy,
                    )
                self._plan_log.append(_plan_failure_entry(reason, disposition="fault"))
                raise DecisionPlanningError(
                    "planner could not produce a safe decision", reason
                ) from error
        raise DecisionPlanningError("planner could not produce a safe decision")

    async def _sweep(
        self, narration: str, resources_before: dict | None = None
    ) -> Literal["record", "none", "failed"]:
        """Recover durable facts from delivered narration, or decline. Fail-open.

        ``resources_before`` defaults to ``self._resources_before_turn`` -- the
        snapshot taken at the top of the *current* ``run_turn`` -- for every ordinary,
        synchronous caller. A backgrounded sweep (C2, ``_dispatch_sweep``) must pass
        its own turn's snapshot explicitly instead: by the time a deferred sweep task
        actually runs, ``self._resources_before_turn`` may already belong to a later
        turn, and reading it lazily here would compare this sweep's claim against the
        wrong turn's starting state.

        Before any record reaches ``scene_commit``, ``guard_ratification`` checks it
        against real state, not tool-call inference: ``_character_resource_values``
        reads every character's actual current coins, HP, and equipment fresh, right
        here, keyed by character id, and ``resources_before`` -- the same
        shape -- supplies the equipment comparison. An earlier version treated \"some
        tool call succeeded this turn\" as proof a claimed figure was real, which an
        independent audit found ratified
        a fabricated total whenever an unrelated, non-mutating call (a
        ``character_sheet`` lookup, say) happened to also run; a call succeeding has
        no necessary connection to the specific figure a record claims. A second
        version checked the claim against every character's values pooled together,
        which the same audit found ratified a real figure attributed to the wrong
        character (`Ossa now has 3 coins` ratifying because *Rill* really had 3).
        ``guard_ratification`` now resolves which character a claim names and checks
        only that character's own state.
        """
        if resources_before is None:
            resources_before = self._resources_before_turn
        try:
            roster = _entity_roster(self.config.campaign_root)
            swept = await asyncio.wait_for(
                self._sweep_once(sweep_prompt(narration, roster)),
                timeout=self.config.sweep_timeout_seconds,
            )
            if swept is None:
                return "failed"
            record = swept.outcome
            if record.kind == "record":
                # Cross-check, not trust: an identifier outside the roster the engine
                # wrote is a hallucination and is dropped before the guard reads it.
                record = record.model_copy(
                    update={"mentions": validated_mentions(record.mentions, roster)}
                )
                swept = swept.model_copy(update={"outcome": record})
            resources_now = _character_resource_values(self.config.campaign_root)
            swept = guard_ratification(
                swept,
                characters_now=resources_now,
                characters_before=resources_before,
                npcs_now=_npc_life_values(self.config.campaign_root),
                objects_now=dict(roster["objects"]),
                scene_summary=str(roster.get("summary", "")),
            )
            outcome = swept.outcome
            if outcome.kind == "none":
                return "none"
            arguments = {
                "public_summary": outcome.public_summary,
                "visible_changes": list(outcome.visible_changes),
                "in_game_time_delta_minutes": 0,
            }


            mention_refs = sorted({mention.entity_id for mention in outcome.mentions})
            if mention_refs:
                arguments["refs"] = mention_refs
            # Slice B: a person the narration introduced rides the same ratified
            # commit, through the same ``persons`` argument the model has. The key is
            # added only when there is someone to add, so a record introducing nobody
            # commits the exact arguments it always did.
            introduced = introduced_person_names(outcome.introduced_persons, roster)
            if introduced:
                arguments["persons"] = [{"name": name} for name in introduced]
            departed = departed_person_ids(outcome.mentions, roster)
            if departed:
                arguments["departed_persons"] = departed
            status = self._call_tool("scene_commit", arguments, origin="sweep")
            return "record" if status == "success" else "failed"
        except Exception:  # noqa: BLE001 - recovery must not become a failure mode
            return "failed"

    def _dispatch_sweep(self, narration: str) -> None:
        """Appends ``pending`` to ``_sweep_log`` at this turn's own slot immediately
        (index alignment is fixed at dispatch time, not at resolution time) and
        captures ``self._resources_before_turn`` now, by value -- this turn's own
        snapshot, not whatever ``self._resources_before_turn`` happens to hold when
        the task actually runs, which may already belong to a later turn. Never
        raises: a task that fails to even start would be indistinguishable from one
        that fails inside ``_sweep``, and both already resolve to ``\"failed\"``.
        """
        index = len(self._sweep_log)
        self._sweep_log.append("pending")
        task = asyncio.ensure_future(self._sweep(narration, dict(self._resources_before_turn)))
        self._pending_sweep = (task, index)

    async def _resolve_pending_sweep(self) -> None:
        """Await and apply the previous turn's backgrounded sweep, if one is in
        flight. Patches its ``_sweep_log`` slot from ``pending`` to the resolved
        disposition in place, preserving index alignment with every other per-turn
        log. Called at the top of ``run_turn``, before the digest render
        (``engine.py``'s own module docstring: "the record-before-next-digest
        ordering the current sequential flow guarantees") -- so a fact the sweep
        just committed is always in the very next digest, never one turn late --
        and from ``flush_pending_sweep`` for a caller that is shutting down instead
        of starting another turn.

        A pending task's own ``_sweep`` never raises (fail-open, matches every other
        caller), so this needs no try/except around the await; a cancelled task
        (``flush_pending_sweep`` was never reached before shutdown tore down the
        event loop) raises ``CancelledError``, which is left to propagate rather
        than papered over, since that path means the fact truly was lost and
        pretending otherwise would misreport ``_sweep_log``.
        """
        if self._pending_sweep is None:
            return
        task, index = self._pending_sweep
        self._pending_sweep = None
        kind = await task
        self._sweep_log[index] = kind

    async def flush_pending_sweep(self) -> None:
        """Resolve any in-flight background sweep (C2) before shutdown.

        Public because it is meant to be called from outside this class, by a
        caller that is about to call ``stop()`` -- ``NarratorService.run()``'s own
        shutdown ``finally`` block, and ``soak_harness``'s own report-building
        ``close()`` -- so the last turn's sweep, if one is pending, still runs to
        completion and still writes canon instead of being silently abandoned.
        ``stop()`` itself cannot do this: it is synchronous, called from contexts
        with no guarantee of a running event loop (``NarratorEngine.__init__``'s own
        error-handling paths, for one), so it only cancels a pending task rather than
        awaiting it -- a caller that can await must call this first.
        """
        await self._resolve_pending_sweep()

    async def _adjudicate_once(self, prompt: str):
        """One structured adjudication request, on the settle step's model path."""
        return await self._structured_once(
            prompt, AdjudicateOutcome, system_prompt=ADJUDICATOR_SYSTEM_PROMPT,
            max_tokens=self.config.settle_max_tokens, origin="adjudicate",
        )

    async def _adjudicate(self, narration: str) -> str:
        """Resolve every open pending ruling from the narration. Fail-open.

        Returns ``resolved`` (all rulings applied), ``partial`` (a tool call failed),
        ``none`` (no rulings), or ``failed`` (state unreadable). Each ruling clears in
        one ``ability_apply_ruling`` call: the model's transcribe-first choice when it
        answers, or the ruling's declared default when it does not. A model failure
        therefore costs a default resolution, never a wedged turn.
        """
        try:
            rulings = ledger.read_pending_rulings(self.config.campaign_root)
        except ledger.LedgerUnreadable:
            return "failed"
        if not rulings:
            return "none"
        resolved = 0
        for ruling in rulings:
            choice = ""
            try:
                outcome = await asyncio.wait_for(
                    self._adjudicate_once(adjudicate_prompt(ruling, narration)),
                    timeout=self.config.settle_timeout_seconds,
                )
                if outcome is not None:
                    choice = outcome.choice
            except Exception:  # noqa: BLE001 - a model fault falls to the default
                choice = ""
            status = self._call_tool(
                "ability_apply_ruling",
                {"ruling_id": ruling["id"], "choice": choice,
                 "source": "model" if choice else "default"},
                origin="adjudicate",
            )
            if status == "success":
                resolved += 1
        return "resolved" if resolved == len(rulings) else "partial"

    # -- the settle step -----------------------------------------------------


    WITHHELD_SETTLE_REASON = (
        "The turn was withheld before delivery, so no narration reached the table and "
        "no scene entry is owed. The tool calls that opened this debt are already on "
        "the record with their own audit events."
    )

    def _call_tool(self, name: str, arguments: dict, origin: str = "settle") -> str:
        """Run one engine-authored tool call through the MCP session, return status.

        ``origin`` labels the tool_use_id so the audit trail distinguishes a settle
        commit from a sweep record; the default keeps every settle identifier
        byte-identical to the pre-sweep form.

        Records the same redacted event a model-issued call gets, the way
        ``_engine_combat_call`` already does for a combat recovery -- without this,
        every settle commit, sweep record, and adjudicated ruling was invisible to
        the diagnostic transcript's per-tool line, which README.md promises for
        every executed tool. A raising call is recorded and re-raised unchanged:
        this method decides nothing about what a fault means, its callers already
        do (``_settle`` retries, ``_settle_withheld`` fails closed, ``_sweep`` and
        ``_adjudicate`` fail open).
        """
        try:
            result = self._client.call_tool_sync(
                tool_use_id=f"engine-{origin}-{name}", name=name, arguments=arguments
            )
        except Exception:  # noqa: BLE001 - record the fault, then let it propagate
            self._tool_events_this_turn.append(
                redact_tool_event(name, exception=True, payload={})
            )
            raise
        self._tool_events_this_turn.append(
            redact_tool_event(name, exception=False, payload=self._tool_result_payload(result))
        )
        return _tool_status(result)

    #: The four range bands, nearest first, exactly as ``bsh_mcp.rules`` orders them.
    #: The combat recovery walks a band index down to close with real combat_move calls.
    _RANGE_BANDS: tuple[str, ...] = ("close", "nearby", "far_away", "distant")

    def _combat_recovery_view(self) -> dict | None:
        """The combat block and NPC statuses one recovery decision needs, or None.

        Read directly from ``campaign/state.json``, the way ``narrator.ledger`` reads
        the fiction debt: the recovery decides from the campaign's own record, never
        from anything the framework or the model said. Absent, unreadable, or
        fight-less state returns None, and the recovery then changes nothing.
        """
        try:
            raw = json.loads(
                (Path(self.config.campaign_root) / "campaign" / "state.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError):
            return None
        combat = raw.get("combat") if isinstance(raw, dict) else None
        if not isinstance(combat, dict) or not combat.get("active"):
            return None
        npcs = raw.get("npcs") if isinstance(raw, dict) else None
        return {"combat": combat, "npcs": npcs if isinstance(npcs, dict) else {}}

    def _engine_combat_call(self, guard, name: str, arguments: dict) -> dict:
        """One engine-authored combat call: executed, redacted, recorded, guarded.

        Mirrors exactly what the ``AfterToolCallEvent`` hook does for a model-issued
        call -- the redacted diagnostic event, the roll facts the announcement
        injection reads, and the guard's own satisfaction accounting -- so an
        engine-rolled mechanic is indistinguishable downstream from one the model
        remembered to roll. A raising call records an exception event and returns an
        empty payload, so the recovery degrades to the withhold it was replacing
        rather than crashing the turn loop.
        """
        try:
            payload = self._tool_result_payload(
                self._client.call_tool_sync(
                    tool_use_id=f"engine-combat-recovery-{name}",
                    name=name,
                    arguments=arguments,
                )
            )
        except Exception:  # noqa: BLE001 - recovery must never crash the table
            self._tool_events_this_turn.append(
                redact_tool_event(name, exception=True, payload={})
            )
            return {}
        self._tool_events_this_turn.append(
            redact_tool_event(name, exception=False, payload=payload)
        )
        self._roll_facts_this_turn.extend(roll_facts(name, payload))
        self._doom_facts_this_turn.extend(doom_facts(name, payload))
        guard.record_tool_result(name, arguments, payload)
        return payload

    def _recover_unrolled_combat(self, guard: ResolutionGuard) -> list[str]:
        """Roll the combat mechanic a confirmed or bound decision owes when the model
        did not, and return the engine-authored mechanical lines to append.

        Two recoveries, each strictly conditioned and each falling back to the
        existing withhold when its conditions miss:

        * A bound dodge/parry rolls against the enemy whose turn is open. The
          directive carries the defender and the method; the fight names the attacker.
        * A confirmed attack rolls against the fight's sole living enemy on the
          actor's own open turn, closing range with real ``combat_move`` calls first
          when needed. When every remaining action goes to closing range, the attack
          is impossible until the actor's next turn, and the hazard is satisfied by
          that engaged deferral -- the same rules-made-it-impossible reasoning
          ``ResolutionGuard.record_tool_result`` grants a ``combat_start`` whose
          initiative gave the opposition the first move. A fight with several living
          enemies leaves the target the model's to choose, so the withhold stands.

        Every call here is a real, audited engine tool call: the dice still live in
        the MCP server, the events still land in the audit log, and the injected
        announcement line downstream reads from the same recorded roll facts a
        model-issued call would have produced.
        """
        lines: list[str] = []
        if self._client is None:
            return lines

        for directive in guard.unused_defence_directives():
            view = self._combat_recovery_view()
            if view is None:
                break
            combat, npcs = view["combat"], view["npcs"]
            active_id = str(combat.get("active_actor") or "")
            actors = combat.get("actors")
            active = actors.get(active_id) if isinstance(actors, dict) else None
            npc = npcs.get(active_id)
            if (
                not isinstance(active, dict)
                or active.get("side") != "npc"
                or not active.get("turn_open")
                or int(active.get("actions_used", 0)) >= int(active.get("actions_max", 0))
                or not isinstance(npc, dict)
                or npc.get("status") != "alive"
            ):
                continue
            arguments = {
                "defender_id": directive.character_id,
                "attacker_id": active_id,
                "method": directive.method,
            }
            if guard.validate("combat_defend", arguments) is not None:
                continue
            payload = self._engine_combat_call(guard, "combat_defend", arguments)
            summary = str(payload.get("summary", "")).strip()
            if payload.get("ok") and summary:
                lines.append(summary)

        for actor_id in guard.unsatisfied_hazard_actors():
            view = self._combat_recovery_view()
            if view is None:
                break
            combat, npcs = view["combat"], view["npcs"]
            active_id = str(combat.get("active_actor") or "")
            if active_id.casefold() != str(actor_id).casefold():
                continue
            actors = combat.get("actors")
            actor = actors.get(active_id) if isinstance(actors, dict) else None
            if (
                not isinstance(actor, dict)
                or actor.get("side") != "pc"
                or not actor.get("turn_open")
            ):
                continue
            remaining = int(actor.get("actions_max", 0)) - int(actor.get("actions_used", 0))
            if remaining <= 0:
                continue
            order = combat.get("order") or []
            living = [
                npc_id
                for npc_id in order
                if isinstance(npcs.get(npc_id), dict) and npcs[npc_id].get("status") == "alive"
            ]
            if len(living) != 1:
                continue
            target_id = str(living[0])
            ranges = combat.get("ranges") if isinstance(combat.get("ranges"), dict) else {}
            band = str(ranges.get(target_id, "nearby"))
            steps = (
                self._RANGE_BANDS.index(band) if band in self._RANGE_BANDS else 1
            )
            move_failed = False
            while steps > 0 and remaining > 0:
                payload = self._engine_combat_call(
                    guard,
                    "combat_move",
                    {"character_id": active_id, "target_id": target_id},
                )
                if not payload.get("ok"):
                    move_failed = True
                    break
                summary = str(payload.get("summary", "")).strip()
                if summary:
                    lines.append(summary)
                steps -= 1
                remaining -= 1
            if move_failed:
                continue
            if steps == 0 and remaining > 0:
                payload = self._engine_combat_call(
                    guard,
                    "combat_attack",
                    {
                        "attacker_id": active_id,
                        "target_id": target_id,
                        "attack_type": "melee",
                    },
                )
                summary = str(payload.get("summary", "")).strip()
                if payload.get("ok") and summary:
                    lines.append(summary)
            elif remaining == 0:
                guard.satisfy_hazard(actor_id)
        return lines

    def _corrective_freeze_refusal(self, tool_name: str) -> str:
        """The freeze's refusal for this tool during a re-narration, or ``""``.

        Scoped to the combat surface: every ``combat_*`` tool mutates fight state,
        and a re-invoked model that attacked again would double consequences whose
        dice already fell. Everything else stays subject only to the ordinary
        ceiling and guard checks -- the freeze must not stop, say, a voluntary
        ``scene_commit`` recording the outcome it was just asked to narrate.
        """
        if self._corrective_tool_freeze and tool_name.startswith("combat_"):
            return self._corrective_tool_freeze
        return ""

    def _meta_tool_refusal(self, tool_name: str) -> str:
        """The refusal a game-master-discussion turn owes this tool, or ``""``.

        Allowlist, not blocklist: only ``policy.GM_DISCUSSION_TOOLS`` -- the pure
        reads and the skills loader -- pass while ``_meta_turn_active`` holds, so a
        tool added to the surface later is refused here by default rather than
        silently becoming a mechanic an out-of-fiction turn can trigger.
        """
        if not self._meta_turn_active:
            return ""
        if tool_name in policy.GM_DISCUSSION_TOOLS:
            return ""
        return (
            "this turn is out-of-fiction discussion with the game master; no die "
            "rolls, no action resolves, and nothing is written to the record -- "
            "answer from the campaign record instead"
        )

    async def _resolve_recovery_narration(
        self, agent, narration: str, recovery_lines: list[str]
    ) -> tuple[str, dict]:
        """The delivered text for a turn whose combat mechanic the engine rolled.

        Exactly one attempt, and a faulted or blank reply falls back to the
        pre-existing delivery shape (draft plus engine-authored lines), for the
        reason ``_avoid_repeat`` states at length: the dice already fell, and
        withholding a turn whose mechanics resolved strands real consequences
        behind silence. ``_corrective_tool_freeze`` holds for the duration so the
        re-invoked model cannot roll a second combat mechanic; the announcement
        injection and verdict correction downstream still run against the reply,
        reading the same recorded roll facts the recovery's own calls produced.
        """
        record = {
            "triggered": True,
            "retried": False,
            "resolved": None,
            "fell_back": False,
            "leaks_scrubbed": 0,
        }
        block = "\n".join(recovery_lines)
        fallback = f"{narration}\n\n{block}" if narration.strip() else block
        self._corrective_tool_freeze = (
            "the engine already performed this turn's owed combat mechanic; "
            "narrate from the stated results instead of calling tools"
        )
        try:
            result = await agent.invoke_async(recovery_narration_prompt(recovery_lines))
            self._fold_corrective_usage(result)
        except Exception:  # noqa: BLE001 - the mechanics must still reach the table
            record["retried"] = True
            record["resolved"] = False
            record["fell_back"] = True
            return fallback, record
        finally:
            self._corrective_tool_freeze = ""
        record["retried"] = True
        renarrated, removed = scrub_markup(_assistant_text(result))
        record["leaks_scrubbed"] = len(removed)
        if not renarrated.strip():
            record["resolved"] = False
            record["fell_back"] = True
            return fallback, record
        record["resolved"] = True
        return renarrated, record

    async def _guard_meta_narration(self, narration: str, agent) -> tuple[str, dict]:
        """Never deliver a game-master-discussion candidate that narrates an action
        or names/promises a tool this turn refuses.

        Exactly one corrective re-invoke, the same bounded shape
        ``_correct_verdicts`` and ``_avoid_repeat`` use and for the same reason: the
        turn must still answer something, and a second retry against a model that
        keeps narrating would only delay that. ``self._meta_turn_active`` is
        reactivated for the retry's own duration and always cleared after, because
        by the time this runs the outer ``finally`` in ``run_turn`` has already
        cleared it for the *original* invocation -- without re-arming it here, a
        retry that decided to actually call a mechanical tool, rather than merely
        talk about one, would reach the server for real on a turn whose whole
        contract is that nothing mechanical happens.

        A retry that still leaks, or faults, falls back to
        ``_drop_meta_leak_sentences`` rather than withholding the turn or delivering
        the leak: dropping the one bad sentence costs less than either.
        """
        record = dict(_NO_META_GUARD_CHECK)
        if not narration.strip() or not _meta_leak_detected(narration):
            return narration, record
        record["triggered"] = True
        self._meta_turn_active = True
        try:
            result = await agent.invoke_async(META_CORRECTION_PROMPT)
            self._fold_corrective_usage(result)
        except Exception:  # noqa: BLE001 - a correction fault falls back to sentence removal
            record["retried"] = True
            record["resolved"] = False
            cleaned, dropped = _drop_meta_leak_sentences(narration)
            record["sentences_dropped"] = dropped
            return cleaned, record
        finally:
            self._meta_turn_active = False
        record["retried"] = True
        corrected, removed = scrub_markup(_assistant_text(result))
        record["leaks_scrubbed"] = len(removed)
        if corrected.strip() and not _meta_leak_detected(corrected):
            record["resolved"] = True
            return corrected, record
        record["resolved"] = False
        cleaned, dropped = _drop_meta_leak_sentences(corrected if corrected.strip() else narration)
        record["sentences_dropped"] = dropped
        return cleaned, record

    async def _settle(
        self, narration: str
    ) -> tuple[Literal["commit", "waive", "none", "failed"], int, str]:
        """Close the ledger by commit or audited waive. Returns (kind, attempts, error).

        ``kind`` is ``commit``, ``waive``, ``none`` (ledger already clear), or
        ``failed``. A failure leaves the debt standing and the turn withheld, which is
        the fail-closed default this engine keeps from the nudge era; unlike then, a
        failure requires the settler to break twice, not the narrator to change its
        mind twice.
        """
        try:
            debts = ledger.read_fiction_debt(self.config.campaign_root)
        except ledger.LedgerUnreadable as error:
            # An unreadable ledger already withholds the turn through ``is_ratified``;
            # raising here instead would crash the service loop. Fail closed, loudly.
            return ("failed", 0, f"ledger unreadable: {error}")
        if not debts:
            return ("none", 0, "")

        prompt = settle_prompt(debts, narration)
        last_error = ""
        for attempt in range(1, self.config.max_settle_attempts + 1):
            try:
                settled = await asyncio.wait_for(
                    self._settle_once(prompt),
                    timeout=self.config.settle_timeout_seconds,
                )
                if settled is None:
                    last_error = "structured output returned no object"
                    continue
                outcome = settled.outcome
                if outcome.kind == "commit":
                    status = self._call_tool(
                        "scene_commit",
                        {
                            "public_summary": outcome.public_summary,
                            "visible_changes": list(outcome.visible_changes),
                            "in_game_time_delta_minutes": int(
                                outcome.in_game_time_delta_minutes
                            ),
                        },
                    )
                    if status == "success":
                        return ("commit", attempt, "")
                    last_error = f"scene_commit returned {status}"
                    continue
                status = self._call_tool("ledger_settle", {"reason": outcome.reason})
                if status == "success":
                    return ("waive", attempt, "")
                last_error = f"ledger_settle returned {status}"
            except Exception as error:  # noqa: BLE001 - fail closed, never crash the table
                last_error = f"{type(error).__name__}: {error}"
        return ("failed", self.config.max_settle_attempts, last_error)

    async def _settle_withheld(self) -> tuple[Literal["waive", "none", "failed"], int, str]:
        """Close a withheld turn's own debt by waive, never by a scene record.

        Passing an empty narration, which the framework-fault branch already did, only
        made a commit unlikely. This removes the branch instead: a withheld turn waives,
        deterministically, with an engine-authored reason and no model round trip at all.
        Nothing mechanical is lost -- ``ledger_settle`` preserves every waived entry with
        its realized stake text in the audit event (``GameService.ledger_settle``).
        """
        try:
            debts = ledger.read_fiction_debt(self.config.campaign_root)
        except ledger.LedgerUnreadable as error:
            # Same fail-closed shape as ``_settle``: an unreadable ledger already
            # withholds the turn, and raising here would crash the service loop.
            return ("failed", 0, f"ledger unreadable: {error}")
        if not debts:
            return ("none", 0, "")
        try:
            status = self._call_tool("ledger_settle", {"reason": self.WITHHELD_SETTLE_REASON})
        except Exception as error:  # noqa: BLE001 - fail closed, never crash the table
            return ("failed", 1, f"{type(error).__name__}: {error}")
        if status == "success":
            return ("waive", 1, "")
        return ("failed", 1, f"ledger_settle returned {status}")

    # -- the progression detector ---------------------------------------------

    async def _avoid_repeat(self, channel_id: str, narration: str, agent) -> tuple[str, dict]:
        """Never deliver a candidate that near-repeats the prior delivered turn.

        Returns the narration to deliver and a compact log record. When there is
        no prior delivered turn on this channel, or this candidate stays under
        ``REPETITION_SIMILARITY_THRESHOLD``, the record reports untriggered and the
        candidate returns unchanged: this is the common case and costs nothing.

        Exactly one retry. ``REPETITION_NUDGE_PROMPT`` asks the same channel agent
        to rewrite without repeating; whatever comes back is delivered, whether or
        not it actually dropped under threshold, because a second retry could loop
        against a model that keeps repeating and withholding a turn whose dice
        already resolved would strand real consequences behind silence -- the same
        failure class the ratification barrier exists to avoid elsewhere in this
        module. A blank or faulted retry falls back to the original candidate
        rather than delivering nothing.
        """
        channel = self._narration.get(channel_id)
        prior = channel.last_delivered if channel is not None else ""
        record = {
            "triggered": False,
            "retried": False,
            "similarity": 0.0,
            "resolved": None,
            "leaks_scrubbed": 0,
        }
        if not prior or not narration.strip():
            return narration, record
        record["similarity"] = narration_similarity(narration, prior)
        if record["similarity"] < REPETITION_SIMILARITY_THRESHOLD:
            return narration, record
        record["triggered"] = True
        try:
            result = await agent.invoke_async(REPETITION_NUDGE_PROMPT)
            self._fold_corrective_usage(result)
        except Exception:  # noqa: BLE001 - a nudge fault delivers the original candidate
            record["retried"] = True
            record["resolved"] = False
            return narration, record
        record["retried"] = True
        retried_narration, retried_removed = scrub_markup(_assistant_text(result))
        record["leaks_scrubbed"] = len(retried_removed)
        if not retried_narration.strip():
            record["resolved"] = False
            return narration, record
        record["resolved"] = narration_similarity(retried_narration, prior) < (
            REPETITION_SIMILARITY_THRESHOLD
        )
        return retried_narration, record

    async def _correct_verdicts(self, narration: str, agent) -> tuple[str, dict]:
        """Never deliver a roll announcement this turn's own tool results contradict.

        Returns the narration to deliver and a compact log record. A turn that rolled
        nothing, or whose announcements all match, records untriggered and returns the
        candidate unchanged: this is the common case and costs nothing.

        Exactly one retry, and the replacement is delivered whether or not it actually
        resolved, for the reason ``_avoid_repeat`` states at length: a second retry
        could loop against a model that keeps printing the same wrong number, and
        withholding a turn whose dice already resolved strands real consequences behind
        silence. A blank or faulted retry falls back to the original candidate rather
        than delivering nothing. The consequence worth naming: this guard makes a
        contradicted verdict rare and observable, not impossible -- ``resolved`` False
        in ``_verdict_log`` is a turn that reached the table still mismatched, and it is
        the number a soak run should watch.
        """
        record = dict(_NO_VERDICT_CHECK)
        if not narration.strip():
            return narration, record
        if not self._roll_facts_this_turn and not announced_rolls(narration, _config_catalog(self)):
            return narration, record
        mismatches = verdict_mismatches(
            narration, tuple(self._roll_facts_this_turn), _config_catalog(self)
        )
        record["mismatches"] = len(mismatches)
        record["kinds"] = sorted({str(entry["kind"]) for entry in mismatches})
        if not mismatches:
            return narration, record
        record["triggered"] = True
        facts = tuple(self._roll_facts_this_turn)

        def _unresolved(text: str) -> tuple[str, dict]:
            # The retry did not settle it: cut the contradicted announcement(s)
            # rather than deliver them beside the engine's own true line.
            scrubbed, cut = scrub_contradicted_announcements(text, facts, _config_catalog(self))
            record["resolved"] = False
            record["announcements_scrubbed"] = cut
            return scrubbed, record

        try:
            result = await agent.invoke_async(VERDICT_CORRECTION_PROMPT)
            self._fold_corrective_usage(result)
        except Exception:  # noqa: BLE001 - a correction fault delivers the original
            record["retried"] = True
            return _unresolved(narration)
        record["retried"] = True
        corrected, removed = scrub_markup(_assistant_text(result))
        record["leaks_scrubbed"] = len(removed)
        if not corrected.strip():
            return _unresolved(narration)
        remaining = verdict_mismatches(corrected, facts, _config_catalog(self))
        if remaining:
            return _unresolved(corrected)
        record["resolved"] = True
        return corrected, record

    # -- the traversal-monotone fallback --------------------------------------

    def _advance_stalled_traversal(
        self, before: dict[str, dict], declaration_is_repeat: bool
    ) -> dict:
        """Guarantee monotonic progress on every open traversal clock this turn.

        ``TRAVERSAL_CLOCK_PREFIX`` identifies a traversal clock among every clock
        the campaign holds, so this never touches a clock some other system uses
        for something else (an alarm, a tide, a pursuit).

        ``was_full_before`` closes both: a clock already at or above its own
        segment count in ``before`` is, for this method's purposes, the same
        shape as a clock absent from ``before`` -- either way, whatever
        ``current`` now holds is entirely this turn's own doing, never a stale
        carryover, because the only way a full clock's fill can read below its
        segment count in ``after`` is a same-turn reopen. Three situations
        trigger an engine-authored nudge:

        * a clock that just opened this turn (absent from ``before``), or a
          clock that was already full in ``before`` and got reopened this turn,
          sitting at zero fill -- opening (or reopening) only ever happens on a
          genuine incomplete movement declaration (skills/bsh-gm/SKILL.md's own
          instruction), so the turn that opens or reopens it already earns its
          own segment of progress without a separate repeat check; and
        * a clock that was *not* full in ``before``, already existed, and made
          zero progress this turn while this turn's own declaration repeats the
          previous one on this channel almost verbatim -- the exact repeat shape
          ``_avoid_repeat`` already detects for narration, reused here for the
          player's own declaration instead of reinvented.

        A clock the model itself advanced this turn -- including a genuine
        same-turn restart-then-progress -- is left alone: a clock that was full
        in ``before`` and now reads some positive value below its segment count
        matches neither of the two triggers above (its own restart-relative
        progress is exactly the model's to keep), and a ``clock_updates`` aimed
        at a different, unrelated clock never touches this one -- the check
        reads each clock's own filled count, never \"did anything change\".
        """
        after = _clock_fills(self.config.campaign_root)
        checked: list[str] = []
        advanced: list[str] = []
        for clock_id, current in after.items():
            if not clock_id.startswith(TRAVERSAL_CLOCK_PREFIX):
                continue
            if current["filled"] >= current["segments"]:
                continue  # already complete: nothing left to nudge
            checked.append(clock_id)
            previous = before.get(clock_id)
            was_full_before = previous is not None and previous["filled"] >= previous["segments"]
            just_opened_at_zero = (
                (previous is None or was_full_before) and current["filled"] == 0
            )
            stalled_repeat = (
                previous is not None
                and not was_full_before
                and current["filled"] <= previous["filled"]
                and declaration_is_repeat
            )
            if not (just_opened_at_zero or stalled_repeat):
                continue
            status = self._call_tool(
                "scene_commit",
                {
                    "public_summary": (
                        "The party keeps at it; progress on the crossing continues."
                    ),
                    "clock_updates": {clock_id: 1},
                },
                origin="traversal_fallback",
            )
            if status == "success":
                advanced.append(clock_id)
        return {
            "checked": checked,
            "advanced": advanced,
            "declaration_repeat": declaration_is_repeat,
        }

    # -- the turn loop -------------------------------------------------------

    async def _invoke_channel_agent(self, agent, prompt_text: str):
        """Run one channel-agent turn, falling back to thinking off on a bounded loop.

        The retry is allowed only in that exact shape. Thinking must be on, because
        at ``off`` the cap is the turn's own 8,192 and nothing cheaper exists to fall
        back to; and no tool may have run in the attempt, because a tool call is a
        committed transaction -- a re-run would roll the same declaration twice, which
        the fault notice (\"any dice already rolled still count\") exists to prevent.
        Inside that shape the attempt left nothing durable: the partial exchange is
        dropped from history, the same prompt is sent once more with thinking
        suppressed, and the session's level is restored whatever happens. Any other
        fault, and a second failure, propagate to ``run_turn``'s fault path unchanged.

        Suppressed means the *request* carries no thought budget and no template
        switch (``_turn_request_overrides``). It does not mean the system prompt
        reverts to a thinking-off session's bytes: this attempt has already lost one
        turn's worth of the model's attention and the tool call it owed, and
        ``THINKING_SECTION`` is the instruction aimed squarely at recovering that. See
        ``_thinking_section_applies``, which the re-pointing below reads.
        """
        try:
            return await agent.invoke_async(prompt_text)
        except Exception as error:  # noqa: BLE001 - only one shape is retried below
            if not _is_max_tokens_fault(error):
                raise
            if self._effective_thinking_level() == "off" or self._calls_this_turn > 0:
                raise
        self._thinking_fallbacks.append(len(self._digest_log))
        _discard_partial_exchange(agent, prompt_text)
        self._thinking_suppressed = True
        try:
            agent.system_prompt = self._turn_system_prompt()
            return await agent.invoke_async(prompt_text)
        finally:
            self._thinking_suppressed = False
            agent.system_prompt = self._turn_system_prompt()

    async def run_turn(
        self,
        turn: InboundTurn | PreparedNarrationTurn,
        decision_resolutions=(),
        decision_action_fingerprint: str = "",
    ) -> TurnOutcome:
        """Run one turn, settle the ledger, and return what the delivery path needs.

        ``turn`` is a bare ``InboundTurn`` from a caller with nothing more to say (a
        probe, most engine-level tests) or the service's own richer
        ``PreparedNarrationTurn``; normalized to the latter immediately below, so the
        rest of this method reads its extension fields directly instead of through a
        ``getattr(turn, \"field\", default)`` scattered at each site. Imported here, not
        at module level: ``narrator.service`` imports this module at import time, so a
        top-level import back would cycle.
        """
        from narrator.service import PreparedNarrationTurn

        if not isinstance(turn, PreparedNarrationTurn):
            turn = PreparedNarrationTurn(source=turn, interaction_cue=None)

        # C2: resolve the previous turn's backgrounded sweep, if any, before this
        # turn reads or renders anything -- in particular before the digest render
        # below, so a fact that sweep just committed is in *this* turn's canon, never
        # one turn late. Single-turn-at-a-time per engine (this module's own
        # docstring) guarantees at most one pending sweep ever exists here.
        await self._resolve_pending_sweep()

        agent = self._agent_for(turn.channel_id)
        # A cached agent keeps its conversation across turns, but its system prompt was
        # only ever set at construction. Re-pointing it here (cheap: string
        # concatenation, no request) is what makes a mid-session ``/language`` switch
        # -- which lands on ``config.language_state``, not on this agent -- take
        # effect on the very next turn rather than the channel's next fresh agent.
        agent.system_prompt = self._turn_system_prompt()

        scene_before = _scene_fingerprint(self.config.campaign_root)
        # The sweep guard's equipment check needs a "before" snapshot; coins and HP
        # verify against the current value directly and read fresh inside ``_sweep``.
        self._resources_before_turn = _character_resource_values(self.config.campaign_root)


        clocks_before_turn = _clock_fills(self.config.campaign_root)
        declaration_text = turn.channel_text()
        declaration_channel = self._narration.setdefault(turn.channel_id, _ChannelNarration())
        prior_declaration = declaration_channel.last_declaration
        declaration_channel.last_declaration = declaration_text
        declaration_is_repeat = bool(prior_declaration) and (
            narration_similarity(declaration_text, prior_declaration)
            >= REPETITION_SIMILARITY_THRESHOLD
        )

        digest = canon.render_digest(
            self.config.campaign_root,
            self.config.repo_root,
            self.config.canon_scene_max_chars,
            self.config.canon_location_max_chars,
            self.config.canon_npcs_max_chars,
        )
        self._digest_log.append(digest.stats)
        self._digest_texts.append(digest.text)

        # Every digest in the history is superseded the moment this turn renders its
        # own, so the strip runs here and needs no comparison. It removes characters
        # from resident messages and never a message, so the conversation window's
        # eviction schedule is unchanged: the window counts messages.
        payload.strip_superseded_canon(agent.messages)

        self._calls_this_turn = 0
        self._tool_names_this_turn = []
        self._tool_signatures_this_turn = []
        self._tool_events_this_turn = []
        self._roll_facts_this_turn = []
        self._doom_facts_this_turn = []
        # The authenticated identity behind this turn, for the character_create
        # account binding in ``_bind_character_create_account``. Set before the agent
        # call and cleared with the guard, so a hook can never bind a tool call to a
        # previous turn's player.
        self._turn_principal = getattr(turn.mention, "principal", None)
        # Out-of-fiction game-master discussion (the ``@GM`` designator lane). The
        # local survives past the finally below, where the hook-facing flag clears,
        # because the sweep, tail, traversal, and comparison-point skips further down
        # all still need the verdict after the invoke.
        meta_turn = bool(turn.meta)
        self._meta_turn_active = meta_turn
        guard = ResolutionGuard(
            tuple(decision_resolutions), action_fingerprint=decision_action_fingerprint,
            social_request=turn.social_test,
        )
        self._resolution_guard = guard
        try:
            continuation = continuation_text(tuple(decision_resolutions))
            player_turn = turn.channel_text()
            if continuation:
                player_turn += f"\n\nTyped player decisions:\n{continuation}"
            player_turn += hazard_obligation_text(tuple(decision_resolutions))
            player_turn += defence_obligation_text(tuple(decision_resolutions))
            result = await self._invoke_channel_agent(
                agent,
                turn_prompt(
                    player_turn,
                    canon=digest.text,
                    interaction_cue=turn.interaction_cue,
                    social_test=turn.social_test,
                    trade_offer=turn.trade_offer,
                    romance_escalation_blocked=turn.romance_escalation_blocked,
                    withheld_note=turn.withheld_note,


                    turn_framing=turn.turn_framing,
                    # Only the ``question_with_act`` framing reads this, and only to
                    # name the tool that owns the action the same turn declared. The
                    # empty string leaves that framing on its generic wording and every
                    # other framing byte-identical.
                    declared_act_tool=turn.declared_act_tool,
                    language_tag=self._active_language_tag(),
                ),
            )
        except Exception as error:  # noqa: BLE001 - a framework fault must not crash the table
            # The sweep log's one-entry-per-turn contract holds on this path too:
            # a faulted turn delivered nothing, so its disposition is skipped. The
            # tool log keeps the same contract: a fault after two calls still
            # records those two, aligned to this turn's index.
            self._append_tool_log()
            self._usage_log.append({})
            self._sweep_log.append("skipped")
            self._repetition_log.append(dict(_NO_REPETITION_CHECK))
            self._traversal_log.append(dict(_NO_TRAVERSAL_CHECK))
            self._verdict_log.append(dict(_NO_VERDICT_CHECK))
            self._recovery_log.append(dict(_NO_RECOVERY_CHECK))
            self._meta_guard_log.append(dict(_NO_META_GUARD_CHECK))


            settle_kind, settle_attempts, settle_error = "none", 0, ""
            if not ledger.is_ratified(self.config.campaign_root):
                settle_kind, settle_attempts, settle_error = await self._settle_withheld()
            return TurnOutcome(
                narration="",
                ratified=ledger.is_ratified(self.config.campaign_root),
                withheld=True,
                settle=settle_kind,
                settle_attempts=settle_attempts,
                error=(
                    f"{type(error).__name__}: {error} (settle also failed: {settle_error})"
                    if settle_kind == "failed"
                    else f"{type(error).__name__}: {error}"
                ),
                tool_events=tuple(self._tool_events_this_turn),
                mechanical_lines=withheld_mechanical_lines(
                    self._roll_facts_this_turn,
                    read_character_display_names(self.config.campaign_root),
                    _config_catalog(self),
                ),
            )
        finally:
            self._resolution_guard = None
            self._turn_principal = None
            self._meta_turn_active = False

        self._append_tool_log()
        self._usage_log.append(_turn_usage(result))

        narration, removed = scrub_markup(_assistant_text(result))
        if not turn.turn_framing:
            # A declared action never legitimately restates an engine notice --
            # only a question-shaped turn's withheld-note block invites that (see
            # ``prompt._withheld_note_block``). Reproducing one here is the sliding
            # window carrying an earlier turn's own withheld-note block into an
            # unrelated later reply, the live shape ``strip_leaked_notices``
            # documents. Every other notice-posting path (``_post_notice`` and
            # siblings) never reaches the model at all, so this only ever catches a
            # leak, never a legitimate engine post.
            narration, leaked_notices = strip_leaked_notices(
                narration, tuple(self.config.catalog.notices.values())
            )
            removed.extend(leaked_notices)

        meta_guard_record = dict(_NO_META_GUARD_CHECK)
        if meta_turn:


            narration, meta_guard_record = await self._guard_meta_narration(narration, agent)
        self._meta_guard_log.append(meta_guard_record)

        recovery_record = dict(_NO_RECOVERY_CHECK)
        guard_error = guard.unused_error()
        if guard_error:


            recovery_lines = self._recover_unrolled_combat(guard)
            guard_error = guard.unused_error()
            if not guard_error and recovery_lines:
                narration, recovery_record = await self._resolve_recovery_narration(
                    agent, narration, recovery_lines
                )
        self._recovery_log.append(recovery_record)
        if guard_error:


            settle_kind, attempts, settle_error = "none", 0, ""
            if not ledger.is_ratified(self.config.campaign_root):
                settle_kind, attempts, settle_error = await self._settle_withheld()
            self._sweep_log.append("skipped")
            self._repetition_log.append(dict(_NO_REPETITION_CHECK))
            self._traversal_log.append(dict(_NO_TRAVERSAL_CHECK))
            self._verdict_log.append(dict(_NO_VERDICT_CHECK))
            reported_error = (
                f"{guard_error} (settle also failed: {settle_error})"
                if settle_kind == "failed"
                else guard_error
            )
            return TurnOutcome(
                narration="",
                ratified=ledger.is_ratified(self.config.campaign_root),
                withheld=True,
                settle=settle_kind,
                settle_attempts=attempts,
                error=reported_error,
                decision_recovery=True,
                tool_events=tuple(self._tool_events_this_turn),
                mechanical_lines=withheld_mechanical_lines(
                    self._roll_facts_this_turn,
                    read_character_display_names(self.config.campaign_root),
                    _config_catalog(self),
                ),
            )


        narration, repetition = await self._avoid_repeat(turn.channel_id, narration, agent)
        self._repetition_log.append(repetition)


        narration, verdicts = await self._correct_verdicts(narration, agent)
        self._verdict_log.append(verdicts)

        # This runs after the verdict correction above, not before: a model that
        # disobeys the "do not write this yourself" instruction and states its own
        # (possibly wrong) announcement gets that stated line corrected first, so what
        # this checks for "already announced" is the corrected text, not a contradicted
        # draft. See ``inject_missing_roll_announcements`` for why the announcement
        # itself no longer depends on the model at all.
        narration, injected = inject_missing_roll_announcements(
            narration,
            tuple(self._roll_facts_this_turn),
            read_character_display_names(self.config.campaign_root),
            self.config.catalog,
        )
        # Doom lines follow the roll-under lines they may share a turn with (a
        # repeated attack rolls both): what happened to the target reads first, the
        # Doom consequence after, as its own line -- see
        # ``inject_missing_doom_announcements`` and ``skills/bsh-gm/SKILL.md``'s "its
        # own short line distinct from the hit or miss it rode in on."
        narration, doom_injected = inject_missing_doom_announcements(
            narration,
            tuple(self._doom_facts_this_turn),
            read_character_display_names(self.config.campaign_root),
            self.config.catalog,
        )
        self._announcement_injection_log.append(
            {"injected": injected, "doom_injected": doom_injected}
        )

        # Adjudicate any pending mechanical ruling from the narration before the gate.
        # Fail-open: a model fault applies the ruling's declared default, so the turn
        # is never wedged, only defaulted.
        if ledger.has_open_rulings(self.config.campaign_root):
            await self._adjudicate(narration)

        settle_kind, attempts, settle_error = "none", 0, ""
        if not ledger.is_ratified(self.config.campaign_root):
            settle_kind, attempts, settle_error = await self._settle(narration)

        ratified = ledger.is_ratified(self.config.campaign_root)
        # The turn delivers only when the ledger is clear AND no mechanical ruling is
        # still open; either withholds. A resolved ruling clears here.
        deliverable = ratified and not ledger.has_open_rulings(self.config.campaign_root)

        # The sweep runs only on turns the table will actually read: a withheld
        # turn's narration never reaches players, and committing facts from text
        # nobody saw would write canon for events that did not happen at the table.
        # The fingerprint comparison covers every writer at once — the model's own
        # scene_commit, a settle commit, a scene transition — so the sweep fires
        # exactly when the delivered turn left no mark on the record. An absent or
        # unreadable record (empty fingerprint) skips: the sweep maintains an
        # existing scene record, and a campaign without one has nothing to maintain.
        # A game-master-discussion turn additionally skips: its narration is
        # out-of-fiction talk about the game, and sweeping it would write table
        # conversation into the scene record as canon.
        sweep_kind = "skipped"
        if (
            deliverable
            and not meta_turn
            and narration.strip()
            and scene_before
            and _scene_fingerprint(self.config.campaign_root) == scene_before
        ):
            # C2: dispatched, not awaited. ``_dispatch_sweep`` appends this turn's
            # own ``pending`` slot to ``_sweep_log`` itself (index alignment is
            # fixed here, at dispatch time), so this branch must not also append.
            self._dispatch_sweep(narration)
            sweep_kind = "pending"
        else:
            self._sweep_log.append(sweep_kind)

        # ``scene_before`` already proved nothing had committed by the time the
        # sweep *dispatched*; whether it will turns on a background task this turn
        # cannot see the result of yet (C2), so this fingerprint read can only ever
        # observe "unchanged" for a turn that just dispatched one. That understates
        # ``scene_changed`` by exactly one turn whenever the pending sweep goes on
        # to commit: this turn's own narration lands in the uncommitted-narration
        # tail below even though the record catches up next turn. Costs nothing but
        # a redundant, already-bounded/trimmed tail entry that ages out within a few
        # turns (``UNCOMMITTED_NARRATION_MAX_CHARS``) -- not silent data loss, and
        # not worth threading the pending task's own eventual result back into a
        # fingerprint comparison that already happened. A missing or unreadable
        # record reads as "unchanged" (``scene_after == ""`` only matches
        # ``scene_before == ""``, and an absent record starts every buffer empty
        # regardless), so a campaign with no scene record accumulates nothing
        # rather than raising.
        scene_after = _scene_fingerprint(self.config.campaign_root)
        if deliverable and narration.strip() and not meta_turn:
            self._extend_uncommitted_narration(
                turn.channel_id, narration, scene_changed=not (scene_before and scene_after == scene_before)
            )


        if deliverable and not meta_turn:
            traversal = self._advance_stalled_traversal(
                clocks_before_turn, declaration_is_repeat
            )
        else:
            traversal = dict(_NO_TRAVERSAL_CHECK)
        self._traversal_log.append(traversal)

        # Only a turn a player will actually see becomes the next comparison
        # point. A withheld turn's narration never reached the table, so it must
        # not stand in as "the prior delivered turn" for the next one. A
        # game-master-discussion answer is likewise excluded: it is not fiction, and
        # a player asking the same out-of-fiction question twice deserves the same
        # answer twice, not a repetition nudge against it.
        if deliverable and narration.strip() and not meta_turn:
            self._narration.setdefault(turn.channel_id, _ChannelNarration()).last_delivered = narration

        return TurnOutcome(
            narration=narration,
            ratified=ratified,
            withheld=not deliverable,
            settle=settle_kind,
            settle_attempts=attempts,
            sweep=sweep_kind,
            leaks_scrubbed=(
                len(removed)
                + repetition.get("leaks_scrubbed", 0)
                + recovery_record.get("leaks_scrubbed", 0)
            ),
            error=settle_error if settle_kind == "failed" else "",
            social_test_outcome=guard.social_outcome,
            tool_events=tuple(self._tool_events_this_turn),
        )
