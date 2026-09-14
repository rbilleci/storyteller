"""Cross-cutting helpers shared by every service mixin.

Holds the ``guard`` decorator, which every guarded tool method across the
package's mixins applies; ``success_after_commit``, which every committed tool
reports its result through; and ``CommonMixin``, whose methods (doom-step
application, helplessness, runic on-kill rolls, NPC/combat-actor id
resolution) are called from every other mixin via ``self``.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

from pydantic import ValidationError

from .. import effects, results, rules
from ..data import RulesDataError
from ..dice import DEPLETED
from ..models import Character, CombatState, Condition, slugify
from ..resolution import match_id
from ..store import CampaignError, Transaction, resolve_or_raise

DOOMED_CONDITION_ID = "doomed"

logger = logging.getLogger("bsh_mcp.service")


def guard(method: Callable) -> Callable:
    """Translate every expected failure into the structured error envelope."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except CampaignError as error:
            return error.as_dict()
        except RulesDataError as error:
            return results.failure(
                "rules_data_error",
                str(error),
                ["Run scripts/validate_campaign.py to identify the broken rules table."],
            )
        except ValidationError as error:
            return results.failure("invalid_input", str(error))
        except (ValueError, KeyError) as error:


            logger.exception("%s: unmapped %s", method.__name__, type(error).__name__)
            return results.failure("invalid_input", str(error))

    return wrapper


def success_after_commit(
    transaction: Transaction,
    summary: str,
    *,
    sequence: int | None = None,
    state_changes: list[str] | None = None,
    **extra,
) -> dict:
    """The envelope every committed tool reports through: ``transaction.changes``
    and ``.warnings`` as ``state_changes``/``warnings``, plus whatever else the
    tool wants to add. ``sequence`` defaults to ``None`` for a branch that reports
    success without committing (an idempotent no-op, e.g. ``combat_begin_turn``
    finding the turn already open) -- ``results.success`` already treats an absent
    ``sequence`` as "no new audit event". ``state_changes`` defaults to
    ``transaction.changes``, but a caller with a richer description than
    ``transaction.record``'s own (``character_advance`` lists exactly which
    benefits applied, not just "advanced to level N") may pass its own.

    Call this only after the transaction's ``with`` block has exited (or from a
    branch that returns before it does, as a handful of ``use_ability`` branches
    do): ``transaction.changes``/``.warnings`` are read live, so a call before the
    tool's own mutations finished would report an incomplete list.
    """
    return results.success(
        summary,
        sequence=sequence,
        state_changes=state_changes if state_changes is not None else transaction.changes,
        warnings=transaction.warnings,
        **extra,
    )


class CommonMixin:
    """Cross-cutting helpers every other mixin may call via ``self``."""

    def _effect_sources(self, character: Character) -> set[str]:
        """Return the effect-source keys a character's backgrounds and Gifts grant."""
        return effects.source_ids(character)

    def _bodyguard_redirect(self, transaction, defender: Character, applied: int) -> int:
        """Split incoming damage with a standing bodyguard warding the defender.

        Scans player characters for a ``bodyguard_ward`` condition targeting the
        defender. The first standing warder absorbs the larger half (rounded up)
        against their own hit points, and the defender takes the remainder. Returns
        the defender's reduced share; mutates the warder in the same transaction.
        """
        if applied <= 0:
            return applied
        for warder_id in self.store.character_ids():
            if warder_id == defender.id:
                continue
            warder = transaction.character(warder_id)
            if warder.status != "ok":
                continue
            wards = any(
                c.effect_id == "bodyguard_ward" and c.target_id == defender.id
                for c in warder.conditions
            )
            if not wards:
                continue
            absorbed = (applied + 1) // 2  # the guard takes the larger half
            previous = warder.hp
            warder.hp = max(0, warder.hp - absorbed)
            transaction.touch_character(warder.id)
            transaction.record(
                f"{warder.id}: bodyguard absorbed {absorbed} for {defender.id} "
                f"({previous} -> {warder.hp})"
            )
            if warder.hp == 0 and warder.status == "ok":
                if self._become_helpless(transaction, warder, "guarding an ally") == "helpless":
                    transaction.record(f"{warder.id}: reduced to 0 hit points and Helpless")
                    transaction.warn(
                        f"{warder.name} is Helpless after guarding {defender.name}."
                    )
            return applied - absorbed  # only the first ward redirects
        return applied

    def _apply_doom_step(self, transaction: Transaction, character: Character, new_die: str) -> None:
        character.doom_die = new_die
        transaction.touch_character(character.id)
        if new_die == DEPLETED and not any(
            condition.id == DOOMED_CONDITION_ID for condition in character.conditions
        ):
            character.conditions = character.conditions + [
                Condition(
                    id=DOOMED_CONDITION_ID,
                    label="Doomed",
                    effect="disadvantage_all",
                    scope="until_long_rest",
                    source="Doom die depleted",
                )
            ]
            transaction.record(f"{character.id}: Doom die depleted; the character is Doomed")

    def _become_helpless(self, transaction: Transaction, character: Character, reason: str) -> str:
        """Set ``character`` Helpless, or dead when a runic weapon claims a helpless wielder.

        A runic weapon whose start-of-session INT test succeeded kills its wielder the
        moment the wielder becomes helpless. This central helper applies that fate at every
        site that drops a character to zero hit points, and returns the resulting status so
        the caller adds its own Helpless narration only when the wielder survives.
        """
        weapon = character.runic_weapon
        if weapon is not None and weapon.kills_helpless:
            character.status = "dead"
            transaction.record(
                f"{character.id}: runic weapon {weapon.name} killed the helpless wielder ({reason})"
            )
            transaction.warn(
                f"{character.name}'s runic weapon {weapon.name} claims its helpless wielder; "
                f"{character.name} dies."
            )
            return "dead"
        character.status = "helpless"
        return "helpless"

    def _runic_on_kill(self, transaction: Transaction, character: Character) -> dict:
        """Roll the runic weapon's on-kill d6 after it fells an enemy.

        The heal face restores the wielder d6 hit points, which the engine applies. The
        other five faces name a fiction the narrator realizes, so this returns their text.
        """
        face = self.roller.die(6)
        outcome = {
            "face": face,
            "roll": {
                "notation": "d6",
                "dice": [face],
                "selected": face,
                "modifier": 0,
                "total": face,
                "edge": "single",
            },
            "text": self.data.runic_on_kill(face),
        }
        if face == self.data.runic_on_kill_heal_face():
            healed = self.roller.notation(self.data.runic_on_kill_heal_die())
            before = character.hp
            character.hp = min(character.hp_max, character.hp + healed)
            outcome["healed"] = character.hp - before
            outcome["healing_roll"] = {
                "notation": self.data.runic_on_kill_heal_die(),
                "dice": [healed],
                "selected": healed,
                "modifier": 0,
                "total": healed,
                "edge": "single",
            }
            transaction.touch_character(character.id)
            transaction.record(
                f"{character.id}: runic weapon restored {outcome['healed']} on the kill"
            )
        return outcome

    def _mandatory_doom(
        self, transaction: Transaction, character: Character, trigger: str
    ) -> dict | None:
        """Roll Doom for a critical failure or a repeated combat action."""
        if character.doom_die == DEPLETED:
            transaction.warn(
                f"{character.name} is already Doomed: the Doom die is spent and cannot be rolled "
                f"({trigger})."
            )
            return None
        outcome = rules.roll_doom(self.roller, character.doom_die)
        if outcome.downgraded:
            self._apply_doom_step(transaction, character, outcome.current_die)
            transaction.record(
                f"{character.id}: Doom {outcome.previous_die} -> {outcome.current_die} ({trigger})"
            )
        payload = outcome.as_dict()
        payload["trigger"] = trigger
        return payload

    @staticmethod
    def _doom_log(*entries: object) -> list[dict]:
        """Serialize the Doom die rolls a tool triggered, in call order.

        Each entry is a ``DoomOutcome`` (its ``as_dict`` carries the rolled face
        under ``roll``) or the dict ``_mandatory_doom`` returns; ``None`` entries
        drop out. A tool commits this list so ``events.jsonl`` records every rolled
        Doom face. Without it the mandatory, repeat-action, and called-on-Doom
        faces reach only the return envelope, and the journal cannot replay them.
        """
        log: list[dict] = []
        for entry in entries:
            if entry is None:
                continue
            log.append(entry.as_dict() if hasattr(entry, "as_dict") else entry)
        return log

    def _resolve_npc_id(
        self,
        transaction: Transaction,
        requested: str,
        extra_steps: list[str] | None = None,
    ) -> str:
        """Resolve a narrator-supplied NPC identifier to a canonical NPC id.

        Precedence: exact id, unique case-folded id, unique case-folded NPC
        name. An unknown or ambiguous identifier raises, so a wrong guess
        never lands on another NPC.
        """
        npcs = transaction.state.npcs
        resolved = resolve_or_raise(
            requested,
            npcs,
            lambda nid: npcs[nid].name,
            entity="NPC",
            entity_plural="NPCs",
            code_prefix="npc",
            extra_steps=extra_steps,
        )
        if resolved != requested:
            transaction.warn(
                f"resolved NPC id {requested!r} to {resolved!r}; use {resolved!r} in later calls."
            )
        return resolved

    def _resolve_combat_actor(
        self, transaction: Transaction, combat: CombatState, requested: str
    ) -> str:
        """Resolve a narrator-supplied identifier to a canonical combat-actor id.

        Precedence: exact actor id, unique case-folded actor id, unique
        case-folded character or NPC name among the combat's actors. Kept apart
        from ``resolve_or_raise``'s shared "valid ids" error shape: an unmatched
        combat actor is not part of this combat, which is a different concept
        from not existing at all, with a different next step (the turn order).
        """

        def _display_name(actor_id: str) -> str:
            actor = combat.actors[actor_id]
            if actor.side == "pc":
                return transaction.character(actor_id).name
            npc = transaction.state.npcs.get(actor_id)
            return npc.name if npc else actor_id

        match = match_id(requested, combat.actors, _display_name)
        if match.resolved is not None:
            if match.resolved != requested:
                transaction.warn(
                    f"resolved combat actor {requested!r} to {match.resolved!r}; "
                    f"use {match.resolved!r} in later calls."
                )
            return match.resolved
        if match.matches:
            raise CampaignError(
                "actor_ambiguous",
                f"{requested!r} matches {len(match.matches)} combat actors: "
                f"{', '.join(sorted(match.matches))}.",
                [f"Use one exact id. Combat order: {', '.join(combat.order)}."],
            )
        raise CampaignError(
            "actor_not_in_combat",
            f"{requested!r} is not part of this combat.",
            [f"Combat order: {', '.join(combat.order)}."],
        )

    def _unique_id(self, base: str, taken: set[str]) -> str:
        candidate = slugify(base)
        if candidate not in taken:
            return candidate
        for suffix in range(2, 100):
            probe = f"{candidate}-{suffix}"
            if probe not in taken:
                return probe
        raise CampaignError("id_exhausted", f"cannot allocate a unique id for {base!r}")

    def _time_string(self, minutes: int) -> str:
        day = minutes // 1440
        remainder = minutes % 1440
        return f"day {day + 1}, {remainder // 60:02d}:{remainder % 60:02d}"
