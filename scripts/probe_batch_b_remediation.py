#!/usr/bin/env python3
"""Four independent measurements against the served endpoint, one per Batch B item. Each
compares a baseline (the real Batch A structural guard active, reproducing production)
against the guard disabled (reproducing the pre-Batch-A defect, to prove the harness
actually exercises it) against the guard disabled plus the candidate prompt/skill text
alone (measuring whether the wording, by itself, closes the gap the guard also closes) --
the plan's own protocol: \"with the guard from Batch A disabled ... measured alone\".

``--mode meta-framing``
    Item 1. An open NPC turn (\"rade\" acting), then \"@GM why are you asking me what rade
    does?\" as a meta turn. Baseline: item 1's ``_guard_meta_narration`` active. Disabled:
    monkey-patched to a no-op. Candidate: disabled guard + the new sentence appended to
    ``_TURN_FRAMINGS[\"gm_discussion\"]``. Metric: ``narrator.engine._meta_leak_detected``
    on the meta turn's own delivered text.

``--mode withheld-wording``
    Item 2. \"kiss rade\" (withheld, romance) then \"kick rade\" (declaration). Item 3's
    ``_enemy_turn_defence_phase`` and item 2's own route-gated note surfacing are both
    disabled for this probe so the note reaches \"kick rade\"'s own prompt exactly as it
    did before either fix -- otherwise there is no leak opportunity left to measure the
    wording against. Baseline: the current ``_withheld_note_block`` wording. Candidate:
    the reworded block (\"Engine note, not for the player...\"). Metric: the notice string
    absent from \"kick rade\"'s own delivered text.

``--mode search-skill``
    Item 6. A dead NPC's body, then \"i loot NAME's body and take his possessions\".
    Baseline: the shipped ``SKILL.md``. Candidate: the shipped text plus the search/loot
    paragraph. Metric: no clarifying question in the delivered reply (the turn either
    states findings directly or, since it may still ask for a confirmation the theft
    floor owns independently, does not ask \"what are you looking for\").

Every run uses a fresh disposable campaign and a fresh channel id, so no repeat carries
state from a previous one. Prints one summary block per configuration; no model prose is
retained beyond the single boolean the metric needs.

Usage (narrator interpreter, live endpoint required):
    .narrator-venv/bin/python scripts/probe_batch_b_remediation.py --mode meta-framing --repeats 10
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bsh_mcp.models import NPC  # noqa: E402
from bsh_mcp.testing import bootstrap_probe_campaign  # noqa: E402
from narrator import engine as engine_module  # noqa: E402
from narrator import prompt as prompt_module  # noqa: E402
from narrator.channels.base import (  # noqa: E402
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.config import NarratorConfig  # noqa: E402
from narrator.decisions import DecisionSubmission  # noqa: E402
from narrator.engine import (  # noqa: E402, F401
    META_CORRECTION_PROMPT,
    NarratorEngine,
    _meta_leak_detected,
)
from narrator.service import NarratorService  # noqa: E402


class _ScriptedAdapter:
    """Yields a fixed turn sequence; answers any presented decision with its first option."""

    name = "terminal"
    decision_capabilities = ChannelCapabilities(structured_decisions=True, atomic_decision_delivery=True)

    def __init__(self, declarations: list[str]) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._declarations = declarations
        self.delivered: list[str] = []
        self.posted: list[str] = []

    async def turns(self):
        for text in self._declarations:
            yield InboundTurn("probe-channel", ChannelMessage("Rill", text, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.posted.append(text)
        self.delivered.append(text)

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(tuple(views)))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


def _prepare_campaign(root: Path, *, npc_id: str, npc_name: str, npc_hp: int, npc_alive: bool, open_turn: str) -> str:
    """One disposable campaign: an authenticated PC, one NPC, an open fight.
    """
    game, character_id = bootstrap_probe_campaign(
        root, title="Batch B probe", name="Vessa", origin="decadent",
        backgrounds=("forbidden-knowledge", "snake-blood", "vicious"),
    )
    with game.store.transaction("probe", character_id, "setup") as transaction:
        transaction.state.scene.location_id = "the-eel-market"
        transaction.state.scene.title = "The eel market at low water"
        if npc_alive:
            transaction.state.npcs[npc_id] = NPC(
                id=npc_id, name=npc_name, level=1, hp=npc_hp, hp_max=npc_hp, damage=4,
                location_id="the-eel-market",
            )
        else:
            transaction.state.npcs[npc_id] = NPC(
                id=npc_id, name=npc_name, level=1, hp=0, hp_max=npc_hp, damage=4,
                location_id="the-eel-market", status="dead",
            )
        transaction.state.scene.present_npcs = [npc_id]
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_setup"})
    (root / "campaign" / "players.yaml").write_text(
        f"players:\n- discord_user_id: probe-player\n  character_id: {character_id}\n  display_name: Vessa\n",
        encoding="utf-8",
    )
    if npc_alive:
        opened = game.combat_start(
            pc_ids=[character_id], npc_ids=[npc_id],
            initial_ranges={npc_id: "close"}, reason="probe",
        )
        if not opened.get("ok"):
            raise ValueError("probe fight did not open")
        holder = npc_id if open_turn == "npc" else character_id
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
    return character_id


def _disable_enemy_turn_defence(service: NarratorService) -> None:
    """Neutralize item 3 so an NPC's open turn falls through to ordinary routing,
    exactly as it did before item 3 existed -- needed to reproduce the historical
    "kick rade" -> "what does he do" state items 1 and 2 measure against."""

    async def _noop(self, turn, policy, open_turn):
        del self, open_turn
        return "continue", (), turn

    service._enemy_turn_defence_phase = _noop.__get__(service)  # noqa: SLF001


def _disable_meta_guard(engine: NarratorEngine) -> None:
    """Neutralize item 1's corrective re-invoke, so only the prompt text (if any) can
    stop a meta turn from narrating an action or naming/promising a tool."""

    async def _noop(self, narration, agent):
        del self, agent
        return narration, dict(engine_module._NO_META_GUARD_CHECK)  # noqa: SLF001

    engine._guard_meta_narration = _noop.__get__(engine)  # noqa: SLF001


async def _run_sequence(campaign_root: Path, declarations: list[str], *, disable_meta_guard: bool, disable_enemy_turn_defence: bool, extra_system_prompt: str = "") -> list[str]:
    """Drive one scripted sequence through the real ``NarratorService``.

    ``NarratorService.run()`` owns ``engine.start()``/``engine.stop()`` itself (it
    asserts the frozen tool surface before the first turn and tears the MCP session
    down in its own ``finally``), so this never calls either directly. The extra
    system-prompt text is injected by wrapping ``engine.start`` rather than setting
    ``engine._system_prompt`` beforehand, because that attribute does not exist until
    ``start()`` has actually run.
    """
    config = NarratorConfig(campaign_root=campaign_root)
    engine = NarratorEngine(config)
    if extra_system_prompt:
        real_start = engine.start

        def _start_with_extra() -> None:
            real_start()
            engine._system_prompt = engine._system_prompt + "\n\n" + extra_system_prompt  # noqa: SLF001

        engine.start = _start_with_extra
    if disable_meta_guard:
        _disable_meta_guard(engine)
    adapter = _ScriptedAdapter(declarations)
    service = NarratorService(config, adapter, engine)
    if disable_enemy_turn_defence:
        _disable_enemy_turn_defence(service)
    await service.run()
    return adapter.delivered


async def _mode_meta_framing(repeats: int, *, candidate: bool) -> dict:
    """Item 1. Returns {"leaked": n, "total": repeats}."""
    original = prompt_module._TURN_FRAMINGS["gm_discussion"]  # noqa: SLF001
    if candidate:
        prompt_module._TURN_FRAMINGS["gm_discussion"] = original + (  # noqa: SLF001
            " Do not narrate what any character or enemy does next, and do not name "
            "engine tools."
        )
    leaked = 0
    try:
        for i in range(repeats):
            with tempfile.TemporaryDirectory(prefix=f"probe-meta-{i}-") as tmp:
                root = Path(tmp)
                _prepare_campaign(root, npc_id="rade", npc_name="Rade", npc_hp=5, npc_alive=True, open_turn="npc")
                delivered = await _run_sequence(
                    root,
                    ["kick rade", "@GM why are you asking me what rade does?"],
                    disable_meta_guard=True,
                    disable_enemy_turn_defence=True,
                )
                meta_reply = delivered[-1] if delivered else ""
                hit = engine_module._meta_leak_detected(meta_reply)  # noqa: SLF001
                if hit:
                    leaked += 1
                print(f"  [{i+1}/{repeats}] leak={hit} reply={meta_reply[:160]!r}")
    finally:
        prompt_module._TURN_FRAMINGS["gm_discussion"] = original  # noqa: SLF001
    return {"leaked": leaked, "total": repeats}


async def _mode_withheld_wording(repeats: int, *, candidate: bool) -> dict:
    """Item 2. Returns {"leaked": n, "total": repeats}.

    Calls ``engine.run_turn`` directly rather than driving the real romance-withhold
    exchange through ``NarratorService``: item 2's own service-side root-cause fix
    (``service.py``, gated on ``turn_framing``) means a real "kiss rade" -> "kick
    rade" pair no longer surfaces the note to "kick rade"'s own prompt at all -- that
    fix is exactly why the leak this probe measures is no longer reachable in
    production. This forces the note into a declaration-shaped turn's prompt
    directly, which is what "measured alone" has to mean once the structural fix
    already removes the only path that used to deliver it there.
    """
    from narrator.service import PreparedNarrationTurn

    original_block = prompt_module._withheld_note_block  # noqa: SLF001
    if candidate:
        def _reworded_block(note: str) -> str:
            return (
                "\n\nEngine note, not for the player and not to be repeated: the "
                f"previous turn was withheld. Reason: {note}\n"
                "Only if the player asks why nothing happened, explain that reason in "
                "your own words."
            )

        prompt_module._withheld_note_block = _reworded_block  # noqa: SLF001
    leaked = 0
    try:
        for i in range(repeats):
            with tempfile.TemporaryDirectory(prefix=f"probe-withheld-{i}-") as tmp:
                root = Path(tmp)


                _prepare_campaign(root, npc_id="rade", npc_name="Rade", npc_hp=5, npc_alive=True, open_turn="npc")
                config = NarratorConfig(campaign_root=root)
                romance_notice = config.romance_boundary_notice
                engine = NarratorEngine(config)
                engine.start()
                try:
                    turn = PreparedNarrationTurn(
                        source=InboundTurn(
                            "probe-channel",
                            ChannelMessage(
                                "Rill", "kick rade",
                                ChannelPrincipal("terminal", "probe-player", "Vessa"),
                            ),
                        ),
                        interaction_cue=None,
                        turn_framing="",
                        withheld_note=romance_notice,
                    )
                    outcome = await engine.run_turn(turn)
                finally:
                    engine.stop()
                kick_reply = outcome.narration
                hit = romance_notice in kick_reply
                if hit:
                    leaked += 1
                print(f"  [{i+1}/{repeats}] leak={hit} reply={kick_reply[:160]!r}")
    finally:
        prompt_module._withheld_note_block = original_block  # noqa: SLF001
    return {"leaked": leaked, "total": repeats}


def _open_pc_second_action(root: Path, character_id: str) -> bool:
    """Whether ``character_id``'s own combat turn is still open with an action left
    and the recorded enemy is still alive -- the real precondition for a second
    attack to mean anything, read the same fail-closed way
    ``narrator.interactions.open_npc_turn`` reads the campaign's own record."""
    import json as _json

    try:
        raw = _json.loads((root / "campaign" / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    combat = raw.get("combat") if isinstance(raw, dict) else None
    if not isinstance(combat, dict) or not combat.get("active"):
        return False
    actors = combat.get("actors")
    actor = actors.get(character_id) if isinstance(actors, dict) else None
    if not isinstance(actor, dict) or not actor.get("turn_open"):
        return False
    if int(actor.get("actions_max", 0)) - int(actor.get("actions_used", 0)) <= 0:
        return False
    npcs = raw.get("npcs") if isinstance(raw, dict) else None
    npc = npcs.get("rade") if isinstance(npcs, dict) else None
    return isinstance(npc, dict) and npc.get("status") == "alive"


#: The mechanic's own proper name, not a bare "doom" -- the setting's own prose
#: ("the dread of the drowning coast") already uses "dread"/"doom" atmospherically
#: on turns that roll nothing, so matching that word alone would flag ordinary,
#: correct flavor text. "Doom die"/"Doom's die" is specific to the game object
#: the mechanic manipulates, which is a much stronger signal that the sentence is
#: making a mechanical claim rather than setting a mood.
_DOOM_DIE_MENTION_PATTERN = re.compile(r"\bdoom(?:'s)?\s+die\b", re.IGNORECASE)

#: One clause around the word "Doom", not the whole reply -- scopes a die-grade
#: search to sentences actually about the Doom die, so an unrelated d-number
#: elsewhere in the same reply (a claymore's own d8, a whip's d4) is never
#: mistaken for a Doom claim. Deliberately no sentence-final anchor: a reply that
#: never terminates the clause (cut short, or trailing into a question) still
#: needs its Doom mention scanned.
_DOOM_SENTENCE_PATTERN = re.compile(r"[^.!?\n]*\bdoom\b[^.!?\n]*", re.IGNORECASE)


def _phantom_doom_claim(text: str, doom_facts: list) -> bool:
    """Whether the narration names the Doom die mechanically with no real roll
    behind it this turn -- a fabricated mechanical fact, not merely atmosphere."""
    return not doom_facts and bool(_DOOM_DIE_MENTION_PATTERN.search(text))


def _doom_grade_check(text: str, doom_facts: list) -> tuple[bool, bool, set]:
    """Whether the narration's own Doom-mentioning sentences (beyond the engine's
    mandatory injected line, already stripped by the caller) restate a die grade,
    and whether any restated grade disagrees with this turn's real outcome.

    Returns ``(restated, contradicted, mentioned)``. Only meaningful when
    ``doom_facts`` is non-empty -- a Doom sentence with no real fact behind it at
    all is ``_phantom_doom_claim``'s own question, not this one, and this function
    returns ``(False, False, set())`` rather than double-counting it. Compared
    against each fact's own ``current_die``, the state a player needs to know now,
    not ``previous_die``: a sentence describing the roll itself ("rolling her own
    d6") names the die honestly without stating the outcome, and must not count as
    a contradiction of a downgrade it never claimed to report.
    """
    if not doom_facts:
        return False, False, set()
    current_grades = {str(fact.get("current_die", "")) for fact in doom_facts} - {""}
    mentioned: set[str] = set()
    for sentence in _DOOM_SENTENCE_PATTERN.findall(text):
        mentioned.update(m.lower() for m in re.findall(r"\bd(?:4|6|8)\b", sentence, re.IGNORECASE))
    restated = bool(mentioned)
    contradicted = restated and not (mentioned & current_grades)
    return restated, contradicted, mentioned


async def _mode_doom_skill(repeats: int, *, candidate: bool) -> dict:
    """Item 4. Returns {\"contradicted\", \"total\", \"no_second_attack\", \"phantom_doom\"}.

    Also counts ``phantom_doom``: either turn's narration naming the Doom die
    against a turn this function's own read of ``engine._doom_facts_this_turn``
    shows rolled none, across every sample regardless of whether a valid second
    attack was ever reached. A pilot surfaced this on every one of several
    candidate- and baseline-condition replies stating a real Doom consequence
    (\"her Doom die steps down\", \"the Doom die... holds\") that ``doom_facts_this_
    turn`` could not confirm -- traced (not guessed) to ``narrator.engine.doom_
    facts`` reading only a top-level ``doom`` list, which no ``combat_attack``
    reply has ever actually carried: the real roll rides under ``repeat_action_
    doom``, a single dict, so a real, correctly-triggered repeat-action roll was
    invisible to this whole measurement's own ground truth, in every prior run of
    this probe, not only the samples printed above. Fixed at the source
    (``doom_facts`` now reads ``repeat_action_doom``/``critical_failure_doom``/
    ``called_on_doom`` too, and ``combat_defend``'s envelope now carries its own
    ``critical_failure_doom``; see ``tests_narrator/test_engine_doom_
    announcements.py``), so this metric should now read at or near zero on a
    correct engine -- a nonzero count post-fix is the more concerning, narrower
    signal of a narration claiming Doom with no engine record of the roll at all,
    which is what \"fabricated\" should be reserved for once the tracking gap is
    closed.
    """
    from narrator.engine import format_doom_announcement
    from narrator.service import PreparedNarrationTurn

    extra = (
        "\n\nThe engine announces the Doom result and any drop of the die; write the "
        "dread, not the number."
        if candidate else ""
    )
    contradicted = 0
    restated = 0
    no_second_attack = 0
    no_roll = 0
    phantom_doom = 0
    phantom_turns_seen = 0
    for i in range(repeats):
        with tempfile.TemporaryDirectory(prefix=f"probe-doom-{i}-") as tmp:
            root = Path(tmp)
            character_id = _prepare_campaign(
                root, npc_id="rade", npc_name="Rade", npc_hp=5, npc_alive=True, open_turn="pc"
            )
            config = NarratorConfig(campaign_root=root)
            engine = NarratorEngine(config)
            engine.start()
            if extra:
                engine._system_prompt = engine._system_prompt + extra  # noqa: SLF001
            principal = ChannelPrincipal("terminal", "probe-player", "Vessa")

            def _attack_turn() -> PreparedNarrationTurn:
                return PreparedNarrationTurn(
                    source=InboundTurn("probe-channel", ChannelMessage("Rill", "i attack rade", principal)),
                    interaction_cue=None, turn_framing="",
                )

            try:
                first = await engine.run_turn(_attack_turn())
                phantom_turns_seen += 1
                first_doom_facts = engine._doom_facts_this_turn  # noqa: SLF001
                if _phantom_doom_claim(first.narration, first_doom_facts):
                    phantom_doom += 1
                    tool_names = list(engine._tool_names_this_turn)  # noqa: SLF001
                    print(
                        f"  [{i+1}/{repeats}] PHANTOM DOOM on turn 1: tools_called={tool_names} "
                        f"reply={first.narration!r}"
                    )
                if not _open_pc_second_action(root, character_id):
                    no_second_attack += 1
                    print(
                        f"  [{i+1}/{repeats}] no second attack possible after turn 1 "
                        f"(fight ended, turn closed, or Rade already dead) "
                        f"turn1_reply={first.narration[-160:]!r}"
                    )
                    continue
                outcome = await engine.run_turn(_attack_turn())
                phantom_turns_seen += 1
            finally:
                engine.stop()
            reply = outcome.narration
            doom_facts = engine._doom_facts_this_turn  # noqa: SLF001
            if not doom_facts:
                no_roll += 1
            if _phantom_doom_claim(reply, doom_facts):
                phantom_doom += 1
                tool_names = list(engine._tool_names_this_turn)  # noqa: SLF001
                print(
                    f"  [{i+1}/{repeats}] PHANTOM DOOM on turn 2: tools_called={tool_names} "
                    f"reply={reply!r}"
                )
            stripped = reply
            for fact in doom_facts:
                injected_line = format_doom_announcement(fact, "Vessa")
                stripped = stripped.replace(injected_line, "")
            was_restated, was_contradicted, mentioned = _doom_grade_check(stripped, doom_facts)
            if was_contradicted:
                contradicted += 1
            elif was_restated:
                restated += 1
            print(
                f"  [{i+1}/{repeats}] doom_facts={len(doom_facts)} "
                f"restated={was_restated} contradicted={was_contradicted} "
                f"grades_mentioned={mentioned} reply={reply[-220:]!r}"
            )
    attempted = repeats - no_second_attack
    print(
        f"  {no_second_attack}/{repeats} samples never reached a valid second attack "
        f"(fight ended on the first hit or the turn otherwise closed); "
        f"{no_roll}/{max(attempted, 1)} of the rest rolled no Doom fact at all "
        f"(model chose not to attack again on its own second message); "
        f"{phantom_doom}/{phantom_turns_seen} turns named the Doom die with no roll behind it; "
        f"{restated}/{max(attempted, 1)} of the rest restated a real grade correctly but "
        f"redundantly with the engine's own announcement"
    )
    return {
        "contradicted": contradicted, "restated": restated, "total": attempted,
        "no_second_attack": no_second_attack, "phantom_doom": phantom_doom,
        "phantom_turns_seen": phantom_turns_seen,
    }


async def _mode_search_skill(repeats: int, *, candidate: bool) -> dict:
    """Item 6. Returns {"asked": n, "total": repeats}."""
    extra = (
        "\n\nWhen a player searches or loots a body, container, or place, state what "
        "is there and record it with inventory_update/scene_commit; do not ask what "
        "they are looking for. Do not restate table rules to the player."
        if candidate else ""
    )
    asked = 0
    for i in range(repeats):
        with tempfile.TemporaryDirectory(prefix=f"probe-search-{i}-") as tmp:
            root = Path(tmp)
            _prepare_campaign(root, npc_id="rade", npc_name="Rade", npc_hp=5, npc_alive=False, open_turn="pc")
            delivered = await _run_sequence(
                root, ["i loot rades body and take his possessions"],
                disable_meta_guard=False, disable_enemy_turn_defence=False,
                extra_system_prompt=extra,
            )
            reply = delivered[-1] if delivered else ""
            # The defect is specifically asking the player to narrow an already-general
            # declaration ("what are you looking for specifically?", seq 58) instead of
            # stating findings -- not any question mark. An ordinary "What do you do
            # next?" turn handback is the expected shape of nearly every reply and matched
            # here in the pilot, a false-positive that would have measured this item's own
            # defect rate as noise. "looking for" is the fix instruction's own phrase
            # ("do not ask what they are looking for") verbatim, kept narrow on purpose.
            hit = "looking for" in reply.lower()
            if hit:
                asked += 1
            print(f"  [{i+1}/{repeats}] clarifying_question={hit} reply={reply!r}")
    return {"asked": asked, "total": repeats}


MODES = {
    "meta-framing": _mode_meta_framing,
    "withheld-wording": _mode_withheld_wording,
    "doom-skill": _mode_doom_skill,
    "search-skill": _mode_search_skill,
}


async def _main(args) -> int:
    fn = MODES[args.mode]
    for candidate in (False, True):
        label = "candidate" if candidate else "baseline (guard disabled)"
        print(f"=== {args.mode}: {label} ===")
        started = time.perf_counter()
        result = await fn(args.repeats, candidate=candidate)
        print(f"  RESULT {result}  ({time.perf_counter() - started:.0f}s)\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
