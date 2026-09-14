"""Combat: initiative, turn lifecycle, movement, attacks, and defence."""

from __future__ import annotations

from .. import effects, rules
from ..dice import Roll, step_up
from ..models import RANGE_BANDS, CombatActor, CombatState, StakesDeclaration
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from ..store import CampaignError, Transaction
from .common import guard, success_after_commit


class CombatStartResult(ToolEnvelopeSuccess):
    initiative: list[dict]
    order: list[str]
    active_actor: str | None
    ranges: dict[str, str]
    next_step: str


class CombatBeginTurnResult(ToolEnvelopeSuccess):
    actor_id: str
    round: int
    actions_max: int
    actions_used: int
    next_step: str


class CombatMoveResult(ToolEnvelopeSuccess):
    actor_id: str
    target_id: str
    from_band: str
    to_band: str
    actions_remaining: int
    turn_advanced: bool
    combat_over: bool
    next_actor: str | None
    round: int
    next_step: str


class CombatAttackResult(ToolEnvelopeSuccess):
    attacker_id: str
    target_id: str
    attribute: str
    threat_modifier: int
    damage: dict | None
    applied_damage: int
    runic_on_kill: dict | None
    secondary_hits: list[dict]
    auto_hit: bool
    unaware_strike: bool
    poisoned: bool
    target_npc: dict
    actions_used: int
    actions_max: int
    repeat_action_doom: dict | None
    critical_failure_doom: dict | None
    turn_advanced: bool
    combat_over: bool
    next_actor: str | None
    round: int
    marvel_usage: list[dict]
    next_step: str


class CombatDefendResult(ToolEnvelopeSuccess):
    defender_id: str
    attacker_id: str
    method: str
    attribute: str
    critical_failure_doom: dict | None
    incoming_damage: int
    applied_damage: int
    absorbed: int
    shield_broken: bool
    defender: dict
    turn_advanced: bool
    combat_over: bool
    next_actor: str | None
    round: int
    next_step: str


class CombatEndTurnResult(ToolEnvelopeSuccess):
    combat_active: bool
    round: int
    next_actor: str | None
    marvel_usage: list[dict]
    next_step: str


class CombatCloseResult(ToolEnvelopeSuccess):
    combat_active: bool
    round: int
    reason: str
    marvel_usage: list[dict]
    next_step: str


class CombatMixin:
    """Initiative, turn lifecycle, movement, attacks, and defence."""

    # -- combat --------------------------------------------------------------

    @guard
    def combat_start(
        self,
        pc_ids: list[str],
        npc_ids: list[str],
        initial_ranges: dict[str, str] | None = None,
        reason: str = "",
        stakes_success: str = "",
        stakes_failure: str = "",
        stakes_hidden: str = "",
    ) -> CombatStartResult | ToolEnvelopeFailure:
        """Roll initiative and open combat."""
        if not pc_ids:
            raise CampaignError("no_participants", "combat needs at least one player character.")
        reason = reason.strip()
        ranges = {
            key.strip(): value.strip().lower() for key, value in (initial_ranges or {}).items()
        }
        for band in ranges.values():
            if band not in RANGE_BANDS:
                raise CampaignError(
                    "invalid_range_band",
                    f"{band!r} is not a range band.",
                    [f"Use one of {', '.join(RANGE_BANDS)}."],
                )

        with self.store.transaction("combat_start", actor_id="", reason=reason) as transaction:
            if transaction.state.combat.active:
                raise CampaignError(
                    "combat_already_active",
                    "combat is already running.",
                    ["Call combat_end_turn to advance, or resolve the current fight first."],
                )
            npc_ids = [
                self._resolve_npc_id(
                    transaction,
                    npc_id,
                    ["Call npc_create for each opponent before combat_start."],
                )
                for npc_id in npc_ids
            ]

            initiative: list[dict] = []
            actors: dict[str, CombatActor] = {}
            before: list[str] = []
            after: list[str] = []

            for requested_id in pc_ids:
                character = transaction.character(requested_id)
                character_id = character.id
                self._require_actable(character)
                init_ctx = effects.apply(
                    "initiative",
                    effects.InitiativeContext(),
                    self.data.effects,
                    self._effect_sources(character),
                )
                payload = self._resolve_character_test(
                    transaction,
                    character,
                    "WIS",
                    advantage=init_ctx.advantage,
                    disadvantage=False,
                    opponent_level=None,
                    call_on_doom=False,
                    trigger="critical failure on initiative",
                )
                test = payload["test"]
                if test.outcome == "critical_success":
                    bucket, first_actions = "before", 3
                elif test.outcome == "success":
                    bucket, first_actions = "before", 2
                elif test.outcome == "failure":
                    bucket, first_actions = "after", 2
                else:
                    bucket, first_actions = "after", 1
                actors[character_id] = CombatActor(
                    id=character_id,
                    side="pc",
                    bucket=bucket,
                    actions_max=first_actions,
                    first_turn_actions=first_actions,
                )
                (before if bucket == "before" else after).append(character_id)
                entry = {
                    "character_id": character_id,
                    "name": character.name,
                    "roll": test.roll.as_dict(),
                    "outcome": test.outcome,
                    "bucket": bucket,
                    "first_turn_actions": first_actions,
                }
                if payload.get("critical_doom"):
                    entry["doom"] = payload["critical_doom"]
                initiative.append(entry)

            for npc_id in npc_ids:
                actors[npc_id] = CombatActor(
                    id=npc_id, side="npc", bucket="before", actions_max=1, first_turn_actions=1
                )
                ranges.setdefault(npc_id, "nearby")

            order = before + list(npc_ids) + after
            combat = CombatState(
                active=True,
                round=1,
                order=order,
                active_actor=order[0] if order else None,
                actors=actors,
                ranges=ranges,
                reason=reason,
                started_event=transaction.state.event_seq + 1,
                stakes_success=str(stakes_success).strip(),
                stakes_failure=str(stakes_failure).strip(),
                stakes_hidden=str(stakes_hidden).strip(),
            )
            transaction.record(f"combat opened with {len(order)} actors")
            # The first turn opens here, not on a later combat_begin_turn call the
            # narrator must remember: see _open_combat_turn for the wedged live
            # session this closes.
            if order:
                self._open_combat_turn(transaction, combat, order[0])
            transaction.state.combat = combat
            sequence = transaction.commit(
                {
                    "outcome": "combat_started",
                    "initiative": initiative,
                    "order": order,
                    "active_actor": order[0] if order else None,
                }
            )

        return success_after_commit(
            transaction,
            f"Combat begins. Round 1. {len(before)} act before the opposition, {len(after)} after.",
            sequence=sequence,
            outcome="combat_started",
            narration_facts=[
                f"{entry['name']} acts {'before' if entry['bucket'] == 'before' else 'after'} "
                f"the opposition with {entry['first_turn_actions']} actions this round."
                for entry in initiative
            ],
            initiative=initiative,
            order=order,
            active_actor=order[0] if order else None,
            ranges=ranges,
            next_step=(
                self._turn_next_step(combat, order[0], combat_over=False)
                + (
                    " A player character's declared attack resolves on their own turn, "
                    "after the opposition acts."
                    if combat.actors[order[0]].side == "npc"
                    else ""
                )
                if order
                else "No actor entered the combat order."
            ),
        )

    def _require_combat(self, transaction: Transaction) -> CombatState:
        combat = transaction.state.combat
        if not combat.active:
            raise CampaignError(
                "combat_not_active",
                "no combat is running.",
                ["Call combat_start with the participating characters and NPCs."],
            )
        return combat

    @guard
    def combat_begin_turn(self, actor_id: str) -> CombatBeginTurnResult | ToolEnvelopeFailure:
        """Confirm the active actor's turn is open, opening it if it is not.
        """
        with self.store.transaction(
            "combat_begin_turn", actor_id=actor_id, reason="begin turn"
        ) as transaction:
            combat = self._require_combat(transaction)
            actor_id = self._resolve_combat_actor(transaction, combat, actor_id)
            transaction.actor_id = actor_id
            actor = combat.actors[actor_id]
            if combat.active_actor != actor_id:
                raise CampaignError(
                    "not_active_actor",
                    f"the active actor is {combat.active_actor!r}, not {actor_id!r}.",
                    [
                        f"{combat.active_actor!r}'s turn is already open: resolve it. "
                        f"{actor_id!r} acts when the turn order reaches them.",
                    ],
                )
            already_open = actor.turn_open
            if already_open:
                remaining = actor.actions_max - actor.actions_used
                sequence = None
            else:
                if actor.side == "pc":
                    character = transaction.character(actor_id)
                    self._require_actable(character)
                actor = self._open_combat_turn(transaction, combat, actor_id)
                remaining = actor.actions_max
                transaction.state.combat = combat
                sequence = transaction.commit(
                    {
                        "outcome": "turn_opened",
                        "round": combat.round,
                        "actions_max": actor.actions_max,
                    }
                )

        return success_after_commit(
            transaction,
            f"{actor_id} has {remaining} action(s) left this turn."
            if already_open
            else f"{actor_id} has {actor.actions_max} action(s) this turn.",
            sequence=sequence,
            outcome="turn_already_open" if already_open else "turn_opened",
            narration_facts=[],
            actor_id=actor_id,
            round=combat.round,
            actions_max=actor.actions_max,
            actions_used=actor.actions_used,
            next_step=(
                f"Resolve {actor_id!r}'s action(s) with combat_attack, combat_move, or "
                "attribute_test. The turn advances itself when the last action is spent; "
                "combat_end_turn is only needed to end the turn early."
                if actor.side == "pc"
                else (
                    "State the threat, ask the target to parry or dodge, then call "
                    "combat_defend. The defence spends this enemy's action and the turn "
                    "advances itself."
                )
            ),
        )

    @guard
    def combat_move(self, character_id: str, target_id: str) -> CombatMoveResult | ToolEnvelopeFailure:
        """Spend one action to close one range band toward a target.

        The Black Sword Hack Systems Reference Document (SRD) at
        https://blackswordhack.github.io/2combat.html grants two actions per turn, and
        one may be a move that closes a band. Melee needs close range, so a fight opened
        at nearby had no way to reach the enemy before this tool; ``combat_attack``
        refused every strike and the turn could not resolve.

        Range is one band per opponent relative to the party, so closing toward one NPC
        writes that NPC's band. ``rules.move_band`` steps it one step toward close.
        """
        with self.store.transaction(
            "combat_move", actor_id=character_id, reason=f"move toward {target_id}"
        ) as transaction:
            combat = self._require_combat(transaction)
            character_id = self._resolve_combat_actor(transaction, combat, character_id)
            transaction.actor_id = character_id
            actor = combat.actors.get(character_id)
            if actor is None or actor.side != "pc":
                raise CampaignError(
                    "invalid_mover",
                    f"{character_id!r} is not a player character in this combat.",
                    ["Only a player character spends an action to move."],
                )
            if not actor.turn_open:
                raise CampaignError(
                    "turn_not_open",
                    f"it is not {character_id!r}'s turn: the active actor is "
                    f"{combat.active_actor!r} in round {combat.round}.",
                    [
                        f"Resolve {combat.active_actor!r}'s turn first; "
                        f"{character_id!r} moves on their own turn.",
                    ],
                )
            if actor.actions_used >= actor.actions_max:
                raise CampaignError(
                    "no_actions_remaining",
                    f"{character_id!r} has used {actor.actions_used} of {actor.actions_max} actions.",
                    ["Call combat_end_turn and continue with the next actor."],
                )

            target_id = self._resolve_npc_id(transaction, target_id)
            npc = transaction.state.npcs[target_id]
            if npc.status != "alive":
                raise CampaignError(
                    "target_not_available",
                    f"{npc.name} is {npc.status}; moving toward them changes nothing.",
                    ["Choose another target, or call combat_end_turn."],
                )
            # Same roster boundary combat_attack enforces: range bands exist for the
            # fight's combatants, and writing one for an outsider is the first half
            # of attacking someone combat_start never admitted.
            target = combat.actors.get(target_id)
            if target is None or target.side != "npc":
                raise CampaignError(
                    "target_not_in_combat",
                    f"{npc.name} is not a combatant in this fight. No action was spent.",
                    [
                        "Move toward one of the fight's own combatants.",
                        "A person joins a fight only through combat_start; if the "
                        "fiction genuinely turns on someone new, close this fight "
                        "with combat_close and open the new one.",
                    ],
                )

            band = combat.ranges.get(target_id, "nearby")
            new_band = rules.move_band(band, "closer")
            if new_band == band:
                raise CampaignError(
                    "already_closest",
                    f"{npc.name} is already at {band} range; a move cannot close further.",
                    ["Attack now, or call combat_end_turn."],
                )
            combat.ranges = {**combat.ranges, target_id: new_band}

            actor.actions_used += 1
            actor.actions_taken = actor.actions_taken + ["move"]
            combat.actors = {**combat.actors, character_id: actor}
            transaction.state.combat = combat
            transaction.record(f"{character_id}: moved toward {target_id}, range {band} -> {new_band}")
            remaining = actor.actions_max - actor.actions_used
            turn_advanced = False
            combat_over = False
            next_actor: str | None = None
            if actor.actions_used >= actor.actions_max:
                turn_advanced = True
                combat_over, next_actor, _, _, _ = self._advance_combat_turn(
                    transaction, combat, character_id, tool="combat_move"
                )
            sequence = transaction.commit(
                {
                    "outcome": "moved",
                    "target_id": target_id,
                    "from_band": band,
                    "to_band": new_band,
                    "actions_remaining": remaining,
                    "turn_advanced": turn_advanced,
                    "next_actor": next_actor,
                    "round": combat.round,
                }
            )

        return success_after_commit(
            transaction,
            f"{character_id} closes to {new_band} range with {npc.name}.",
            sequence=sequence,
            outcome="moved",
            narration_facts=[],
            actor_id=character_id,
            target_id=target_id,
            from_band=band,
            to_band=new_band,
            actions_remaining=remaining,
            turn_advanced=turn_advanced,
            combat_over=combat_over,
            next_actor=next_actor,
            round=combat.round,
            next_step=(
                self._turn_next_step(combat, next_actor, combat_over)
                if turn_advanced
                else (
                    f"{character_id!r} has {remaining} action(s) left. Attack {target_id!r} "
                    f"if now at close range, or call combat_end_turn to end the turn early."
                )
            ),
        )

    @guard
    def combat_attack(
        self,
        attacker_id: str,
        target_id: str,
        attack_type: str = "melee",
        advantage: bool = False,
        disadvantage: bool = False,
        two_handed: bool = False,
        weapon_effect: str = "none",
        unarmed: bool = False,
        runic: bool = False,
        one_handed_blade: bool = False,
        target_unaware: bool = False,
        poisoned: bool = False,
    ) -> CombatAttackResult | ToolEnvelopeFailure:
        """Resolve one player attack against one NPC.

        Set ``runic`` true to strike with the attacker's sentient runic weapon: damage
        equals the wielder's attribute the weapon's personality fixes, and a kill triggers
        the on-kill d6. A runic strike carries no weapon_effect flourish and is not unarmed.
        Set ``one_handed_blade`` true when the weapon in hand is a one-handed blade; a
        Sword master then tests DEX instead of STR on a melee attack. A Hunter's first
        ranged attack of a fight hits automatically without a test and adds their level
        to damage. Set ``target_unaware`` true when the fiction holds the target unaware
        of the attacker; an Assassin's strike then deals damage equal to their DEX score
        in place of the rolled weapon damage. The engine honours the declaration only
        while the target has not itself reacted this fight (its own turn opened, or it
        attacked someone) and this character has not already spent their own unaware
        strike this fight; outside that window the declaration is ignored and normal
        damage applies. Being struck does not itself count as the target reacting, so a
        second Assassin can still find the same target unaware. Set ``poisoned`` true to
        coat the strike with one prepared Herbalist poison dose: refused without a dose in
        stock, spent on the swing whether it hits or misses, and adding d6 damage on a
        damaging hit.
        """
        if attack_type not in ("melee", "ranged"):
            raise CampaignError(
                "invalid_attack_type",
                f"{attack_type!r} is not an attack type.",
                ["Use melee or ranged."],
            )
        if runic and (weapon_effect != "none" or unarmed or poisoned):
            raise CampaignError(
                "runic_attack_conflict",
                "a runic strike takes no weapon_effect, no poison, and is not unarmed.",
                [
                    "Call combat_attack with runic true, weapon_effect 'none', "
                    "poisoned false, and unarmed false.",
                ],
            )
        if one_handed_blade and two_handed:
            raise CampaignError(
                "inconsistent_weapon_declaration",
                "a weapon cannot be declared both one-handed and two-handed.",
                ["Call combat_attack with only one_handed_blade or two_handed set true."],
            )
        if one_handed_blade and unarmed:
            raise CampaignError(
                "inconsistent_weapon_declaration",
                "an unarmed strike carries no blade.",
                ["Call combat_attack with unarmed false, or drop one_handed_blade."],
            )
        if poisoned and unarmed:
            raise CampaignError(
                "inconsistent_weapon_declaration",
                "an unarmed strike carries no envenomed blade.",
                ["Call combat_attack with unarmed false, or drop poisoned."],
            )
        effect_rule = self.data.effect_rule(weapon_effect)
        if effect_rule is None:
            raise CampaignError(
                "weapon_effect_not_implemented",
                f"{weapon_effect!r} is not an implemented weapon effect. No dice were rolled and "
                "no state changed.",
                [
                    f"Implemented effects: {', '.join(self.data.implemented_effects())}.",
                    "Call combat_attack again with weapon_effect 'none', then narrate the "
                    "flourish and record any durable consequence with scene_commit.",
                ],
            )
        if poisoned and effect_rule.get("damage", "normal") != "normal":
            raise CampaignError(
                "poison_needs_a_damaging_attack",
                "disarm, pin_down, and shove deal no damage for poison to ride.",
                ["Call combat_attack with weapon_effect 'none' or 'brutal', or drop poisoned."],
            )

        with self.store.transaction(
            "combat_attack", actor_id=attacker_id, reason=f"attack {target_id}"
        ) as transaction:
            combat = self._require_combat(transaction)
            attacker_id = self._resolve_combat_actor(transaction, combat, attacker_id)
            transaction.actor_id = attacker_id
            actor = combat.actors.get(attacker_id)
            if actor is None or actor.side != "pc":
                raise CampaignError(
                    "invalid_attacker",
                    f"{attacker_id!r} is not a player character in this combat.",
                    ["NPC attacks are resolved through combat_defend on the player's side."],
                )
            if not actor.turn_open:
                raise CampaignError(
                    "turn_not_open",
                    f"it is not {attacker_id!r}'s turn: the active actor is "
                    f"{combat.active_actor!r} in round {combat.round}.",
                    [
                        f"Resolve {combat.active_actor!r}'s turn first"
                        + (
                            " (narrate its action, ask the player to parry or dodge, "
                            "then call combat_defend)"
                            if combat.actors.get(combat.active_actor)
                            and combat.actors[combat.active_actor].side == "npc"
                            else ""
                        )
                        + f"; {attacker_id!r} attacks on their own turn.",
                    ],
                )
            if actor.actions_used >= actor.actions_max:
                raise CampaignError(
                    "no_actions_remaining",
                    f"{attacker_id!r} has used {actor.actions_used} of {actor.actions_max} actions.",
                    ["Call combat_end_turn and continue with the next actor."],
                )

            target_id = self._resolve_npc_id(transaction, target_id)
            npc = transaction.state.npcs[target_id]
            if npc.status != "alive":
                raise CampaignError(
                    "target_not_available",
                    f"{npc.name} is {npc.status} and cannot be attacked.",
                    ["Choose another target, or call combat_end_turn."],
                )
            # The fight's roster is the consent record: combat_start is the one gate
            # that puts a person into a fight, so an attack may reach only who that
            # gate admitted. Before this check, any recorded living NPC resolved as a
            # target and ``combat.ranges`` defaulted them a band, so a ranged attack
            # could roll dice and apply damage to a bystander the fight never
            # included, with no confirmation anywhere.
            target = combat.actors.get(target_id)
            if target is None or target.side != "npc":
                raise CampaignError(
                    "target_not_in_combat",
                    f"{npc.name} is not a combatant in this fight. No dice were "
                    "rolled and no action was spent.",
                    [
                        "Attack one of the fight's own combatants.",
                        "A person joins a fight only through combat_start; if the "
                        "fiction genuinely turns on someone new, close this fight "
                        "with combat_close and open the new one.",
                    ],
                )

            band = combat.ranges.get(target_id, "nearby")
            remaining = actor.actions_max - actor.actions_used
            if attack_type == "melee" and band != "close":
                raise CampaignError(
                    "target_out_of_reach",
                    f"{npc.name} is at {band} range; melee needs close range. No dice "
                    f"were rolled and no action was spent: {attacker_id!r} still has "
                    f"{remaining} of {actor.actions_max} action(s). Do not narrate this "
                    "as a swing or a miss -- the attack never happened.",
                    [
                        f"Call combat_move with character_id {attacker_id!r} and target_id "
                        f"{target_id!r} to spend one action closing a band, then attack.",
                    ],
                )
            if attack_type == "ranged" and band == "distant":
                raise CampaignError(
                    "target_out_of_range",
                    f"{npc.name} is distant; a ranged attack needs far away range or "
                    "closer. No dice were rolled and no action was spent: "
                    f"{attacker_id!r} still has {remaining} of {actor.actions_max} "
                    "action(s). Do not narrate this as a shot or a miss -- the attack "
                    "never happened.",
                    [
                        f"Call combat_move with character_id {attacker_id!r} and target_id "
                        f"{target_id!r} to close a band, then attack.",
                    ],
                )

            character = transaction.character(attacker_id)
            self._require_actable(character)
            if runic and character.runic_weapon is None:
                raise CampaignError(
                    "no_runic_weapon",
                    f"{character.name} wields no runic weapon.",
                    ["Grant one with grant_runic_weapon, or attack without runic."],
                )
            if poisoned and character.doses.get("poison", 0) < 1:
                raise CampaignError(
                    "no_poison_dose",
                    f"{character.name} holds no poison doses.",
                    [
                        "Declare a poison preparation with use_ability, then replenish on a "
                        "qualifying long rest, or attack without poisoned.",
                    ],
                )

            repeat_doom = None
            if "attack" in actor.actions_taken:
                repeat_doom = self._mandatory_doom(
                    transaction, character, "repeated attack action in one turn"
                )

            live = effects.active_ids(character)
            first_ranged = attack_type == "ranged" and actor.ranged_attacks_made == 0
            target_actor = combat.actors.get(target_id)
            attacker_charge_available = not actor.unaware_strike_used
            unaware_verified = (
                target_actor is not None
                and not target_actor.reacted
                and attacker_charge_available
            )
            attack_ctx = effects.apply(
                "attack",
                effects.AttackContext(
                    attack_type=attack_type,
                    attribute="STR" if attack_type == "melee" else "DEX",
                    first_ranged_in_combat=first_ranged,
                    one_handed_blade=one_handed_blade,
                    level=character.level,
                    target_unaware_verified=unaware_verified,
                    target_unaware_declared=target_unaware,
                    unarmed=unarmed,
                    weapons_held=list(character.weapons),
                    declared_choices=dict(character.declared_choices),
                ),
                self.data.effects,
                self._effect_sources(character),
                active_ids=live,
            )
            for note in attack_ctx.notes:
                transaction.warn(note)
            if target_unaware and not unaware_verified:
                if target_actor is None:
                    transaction.warn(
                        f"{npc.name} is not enrolled in this combat; target_unaware does "
                        "not bind and normal damage applies."
                    )
                elif target_actor.reacted:
                    transaction.warn(
                        f"{npc.name} has already reacted this fight; target_unaware does "
                        "not bind and normal damage applies."
                    )
                elif not attacker_charge_available:
                    transaction.warn(
                        f"{character.name} has already spent their unaware strike this "
                        "fight; target_unaware does not bind and normal damage applies."
                    )
            attribute = attack_ctx.attribute
            if attack_ctx.auto_hit:
                target = character.attributes.get(attribute)
                test = rules.TestResult(
                    roll=Roll(
                        notation="auto",
                        dice=[],
                        selected=0,
                        total=0,
                        target=target,
                        note="first arrow of the fight hits automatically",
                    ),
                    outcome="success",
                    target=target,
                )
                payload = {
                    "test": test,
                    "critical_doom": None,
                    "threat": 0,
                    "crit_success_max": 1,
                }
                transaction.record(
                    f"{character.id}: first arrow of the fight hits without a test"
                )
            else:
                payload = self._resolve_character_test(
                    transaction,
                    character,
                    attribute,
                    advantage=advantage,
                    disadvantage=disadvantage,
                    opponent_level=npc.level,
                    call_on_doom=False,
                    trigger="critical failure on an attack",
                    in_combat=True,
                )
                test = payload["test"]

            damage_payload = None
            applied = 0
            effect_notes: list[str] = []
            runic_kill = None
            deals_damage = effect_rule.get("damage", "normal") == "normal"
            #: Whether the unaware-strike override actually determined the damage dealt,
            #: as opposed to merely qualifying (a qualifying attempt still spends the
            #: attacker's charge even when a runic strike takes precedence over it).
            unaware_strike_applied = False
            poison_rolled = None
            #: Set only inside the normal weapon/unarmed die branch below, when
            #: the kept base-component die matches a Riddle-of-Steel-style break
            #: face. Applied after the target's hit points are written.
            weapon_break_name = None

            if test.succeeded:
                if deals_damage:
                    if runic:
                        weapon = character.runic_weapon
                        runic_attr = self.data.runic_damage_attribute(weapon.personality)
                        applied = getattr(character.attributes, runic_attr)
                        damage_payload = {
                            "total": applied,
                            "runic": True,
                            "attribute": runic_attr,
                            "personality": weapon.personality,
                        }
                    elif attack_ctx.damage_override_attribute:
                        unaware_strike_applied = True
                        applied = character.attributes.get(attack_ctx.damage_override_attribute)
                        damage_payload = {
                            "total": applied,
                            "override": attack_ctx.damage_override_effect,
                            "attribute": attack_ctx.damage_override_attribute,
                        }
                        transaction.record(
                            f"{character.id}: strike against unaware {npc.id} deals the "
                            f"{attack_ctx.damage_override_attribute} score, no damage die rolled"
                        )
                    else:
                        base_die = attack_ctx.damage_die_override or (
                            character.unarmed_damage if unarmed else character.weapon_damage
                        )
                        die = (
                            step_up(base_die, attack_ctx.damage_die_steps)
                            if attack_ctx.damage_die_steps
                            else base_die
                        )
                        _, damage_disadvantage, damage_notes = rules.damage_edges(
                            character, two_handed
                        )
                        for note in damage_notes:
                            transaction.warn(note)
                        outcome = rules.roll_damage(
                            self.roller,
                            die,
                            two_handed=two_handed,
                            disadvantage=damage_disadvantage,
                            critical=test.outcome == "critical_success",
                            brutal=weapon_effect == "brutal",
                        )
                        damage_payload = outcome.as_dict()
                        if die != base_die:
                            damage_payload["die_stepped"] = {"from": base_die, "to": die}
                        if (
                            attack_ctx.weapon_break_face is not None
                            and attack_ctx.weapon_break_name
                            and outcome.rolls[-1].selected == attack_ctx.weapon_break_face
                        ):
                            weapon_break_name = attack_ctx.weapon_break_name
                        applied = outcome.total
                        dealt = effects.apply(
                            "damage_dealt",
                            effects.DamageDealtContext(total=applied, roller=self.roller),
                            self.data.effects,
                            self._effect_sources(character),
                            active_ids=live,
                        )
                        if dealt.added:
                            applied = dealt.total
                            damage_payload["added"] = dealt.added
                        for ended in dealt.end_effects:
                            character.conditions = [
                                c for c in character.conditions if c.effect_id != ended
                            ]
                            transaction.touch_character(character.id)
                            effect_notes.append(f"the {ended.replace('_', ' ')} ends")
                            transaction.record(f"{character.id}: stance {ended} ended")
                        if test.outcome == "critical_success":
                            crit_ctx = effects.apply(
                                "critical_damage",
                                effects.DamageContext(
                                    total=applied, attributes=character.attributes
                                ),
                                self.data.effects,
                                self._effect_sources(character),
                                active_ids=live,
                            )
                            if crit_ctx.total != applied:
                                applied = crit_ctx.total
                                damage_payload["total_after_effects"] = applied
                    if attack_ctx.damage_bonus:
                        applied += attack_ctx.damage_bonus
                        damage_payload["level_bonus"] = attack_ctx.damage_bonus
                    if poisoned:
                        poison_rolled = self.roller.notation("d6")
                        applied += poison_rolled
                        damage_payload["poison"] = poison_rolled
                    previous_hp = npc.hp
                    npc.hp = max(0, npc.hp - applied)
                    transaction.record(
                        f"{npc.id}: hit points {previous_hp} -> {npc.hp} "
                        f"({applied} damage from {character.id})"
                    )
                    if weapon_break_name and weapon_break_name in character.weapons:
                        weapons = list(character.weapons)
                        weapons.remove(weapon_break_name)
                        character.weapons = weapons
                        if attack_ctx.weapon_break_effect in character.declared_choices:
                            character.declared_choices = {
                                k: v
                                for k, v in character.declared_choices.items()
                                if k != attack_ctx.weapon_break_effect
                            }
                        transaction.touch_character(character.id)
                        transaction.record(
                            f"{character.id}: {weapon_break_name} breaks and is lost"
                        )
                        transaction.warn(
                            f"{character.name}'s {weapon_break_name} breaks on the damage roll."
                        )
                        effect_notes.append(
                            f"{character.name}'s {weapon_break_name} shatters from the blow."
                        )
                else:
                    effect_notes.append(effect_rule["rule"])
                    flag = effect_rule.get("target_flag")
                    if flag and flag not in npc.flags:
                        npc.flags = npc.flags + [flag]
                        transaction.record(f"{npc.id}: gained flag {flag}")
                    if effect_rule.get("range_change") == "away":
                        new_band = rules.move_band(band, "away")
                        combat.ranges = {**combat.ranges, target_id: new_band}
                        transaction.record(f"{npc.id}: range {band} -> {new_band}")

                if npc.hp == 0:
                    npc.status = "dead"
                    transaction.record(f"{npc.id}: reduced to 0 hit points and dead")
                    if runic:
                        runic_kill = self._runic_on_kill(transaction, character)

            transaction.state.npcs = {**transaction.state.npcs, target_id: npc}

            # Cleave and impale both read a declarative {selection, condition} pair
            # off the weapon effect's own row rather than branching on its identifier
            # -- the same "target_flag"/"range_change" generic-field style disarm,
            # pin_down, and shove already use. Cleave selects every other enemy in
            # the target's band unconditionally; impale selects one at random, only
            # once the primary target is defeated. Both apply the same rolled total
            # a second time rather than rolling fresh damage.
            secondary_hits: list[dict] = []
            secondary_rule = effect_rule.get("secondary_targets")
            if secondary_rule and test.succeeded and deals_damage and applied > 0:
                condition_met = secondary_rule.get("condition") != "target_defeated" or (
                    npc.status == "dead"
                )
                if condition_met:
                    candidates = [
                        other_id
                        for other_id, other_band in combat.ranges.items()
                        if other_id != target_id
                        and other_band == band
                        and other_id in combat.order
                        and transaction.state.npcs.get(other_id) is not None
                        and transaction.state.npcs[other_id].status == "alive"
                    ]
                    chosen = (
                        candidates
                        if secondary_rule.get("selection") == "all"
                        else ([self.roller.choice(candidates)] if candidates else [])
                    )
                    for other_id in chosen:
                        other = transaction.state.npcs[other_id]
                        other_previous = other.hp
                        other.hp = max(0, other.hp - applied)
                        if other.hp == 0:
                            other.status = "dead"
                        transaction.state.npcs = {**transaction.state.npcs, other_id: other}
                        transaction.record(
                            f"{other_id}: hit points {other_previous} -> {other.hp} "
                            f"({applied} {weapon_effect} damage from {character.id})"
                        )
                        secondary_hits.append(
                            {
                                "target_id": other_id,
                                "applied": applied,
                                "dead": other.status == "dead",
                            }
                        )
                    if secondary_hits:
                        names = ", ".join(
                            transaction.state.npcs[hit["target_id"]].name for hit in secondary_hits
                        )
                        effect_notes.append(
                            f"the {weapon_effect} also strikes {names} for {applied} damage."
                        )

            actor.actions_used += 1
            actor.actions_taken = actor.actions_taken + ["attack"]
            if attack_type == "ranged":
                actor.ranged_attacks_made += 1
            if attack_ctx.damage_override_attribute:
                actor.unaware_strike_used = True
            combat.actors = {**combat.actors, attacker_id: actor}

            if poisoned:
                remaining_poison = character.doses.get("poison", 0) - 1
                if remaining_poison:
                    character.doses = {**character.doses, "poison": remaining_poison}
                else:
                    character.doses = {k: v for k, v in character.doses.items() if k != "poison"}
                transaction.touch_character(character.id)
                transaction.record(
                    f"{character.id}: poison dose spent on the strike ({remaining_poison} left)"
                )

            alive = [
                candidate
                for candidate in transaction.state.npcs.values()
                if candidate.id in combat.order and candidate.status == "alive"
            ]
            enrolled = [
                candidate
                for candidate in transaction.state.npcs.values()
                if candidate.id in combat.order
            ]
            fraction = 0.0
            if enrolled:
                fraction = 1 - (len(alive) / len(enrolled))
            morale = rules.npc_morale_warning(
                npc.hp if npc.status == "alive" else 99, fraction, self.data
            )
            if morale:
                transaction.warn(morale)

            transaction.state.combat = combat
            # The turn advances itself when this attack spent the last action, and
            # the fight closes itself when it dropped the last enemy -- both
            # deterministic consequences the engine owns rather than tool calls the
            # narrator must remember (see _open_combat_turn).
            turn_advanced = False
            combat_over = False
            next_actor: str | None = None
            marvel_usage: list[dict] = []
            if actor.actions_used >= actor.actions_max or not alive:
                turn_advanced = True
                combat_over, next_actor, _, marvel_usage, _ = self._advance_combat_turn(
                    transaction, combat, attacker_id, tool="combat_attack"
                )
            commit_payload = {
                "roll": test.roll.as_dict(),
                "outcome": test.outcome,
                "attribute": attribute,
                "target_id": target_id,
                "damage": damage_payload,
                "applied_damage": applied,
                "weapon_effect": weapon_effect,
                "doom": self._doom_log(
                    repeat_doom, test.doom, payload.get("critical_doom")
                ),
                "runic_on_kill": runic_kill,
                "secondary_hits": secondary_hits,
                "auto_hit": attack_ctx.auto_hit,
                "unaware_strike": unaware_strike_applied,
                "poisoned": poisoned,
                "turn_advanced": turn_advanced,
                "combat_over": combat_over,
                "next_actor": next_actor,
                "round": combat.round,
            }
            if marvel_usage:
                commit_payload["marvel_usage"] = marvel_usage
            if (
                test.outcome == "critical_success"
                and payload["crit_success_max"] > 1
                and test.roll.selected > 1
            ):
                commit_payload["crit_success_max"] = payload["crit_success_max"]
            sequence = transaction.commit(commit_payload)

        if test.succeeded and deals_damage:
            summary = (
                f"{character.name} hits {npc.name} for {applied} damage. "
                f"{npc.name} is {'dead' if npc.status == 'dead' else f'at {npc.hp} hit points'}."
            )
        elif test.succeeded:
            summary = f"{character.name} lands a {weapon_effect} against {npc.name}."
        else:
            summary = f"{character.name} misses {npc.name}."

        facts = [summary]
        if attack_ctx.auto_hit:
            facts.append(f"{character.name}'s first arrow of the fight strikes without a test.")
        if unaware_strike_applied:
            facts.append(
                f"{character.name} strikes while {npc.name} is unaware: damage equals "
                f"{character.name}'s {attack_ctx.damage_override_attribute} score."
            )
        if poison_rolled is not None:
            facts.append(f"The poison bites for {poison_rolled} more.")
        elif poisoned:
            facts.append("The poisoned stroke goes wide; the dose is spent.")
        facts.extend(effect_notes)
        if runic_kill:
            facts.append(runic_kill["text"])
            if runic_kill.get("healed"):
                facts.append(
                    f"{character.name}'s runic weapon restores {runic_kill['healed']} hit points."
                )
        if payload.get("critical_doom"):
            facts.append(f"{character.name}'s Doom answers the critical failure.")
        if repeat_doom:
            facts.append(f"{character.name} pays Doom for repeating the same action.")

        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            roll=test.roll.as_dict(),
            outcome=test.outcome,
            narration_facts=facts,
            attacker_id=attacker_id,
            target_id=target_id,
            attribute=attribute,
            threat_modifier=test.threat_modifier,
            damage=damage_payload,
            applied_damage=applied,
            runic_on_kill=runic_kill,
            secondary_hits=secondary_hits,
            auto_hit=attack_ctx.auto_hit,
            unaware_strike=unaware_strike_applied,
            poisoned=poisoned,


            target_npc=self._npc_summary(npc),
            actions_used=actor.actions_used,
            actions_max=actor.actions_max,
            repeat_action_doom=repeat_doom,
            critical_failure_doom=payload.get("critical_doom"),
            turn_advanced=turn_advanced,
            combat_over=combat_over,
            next_actor=next_actor,
            round=combat.round,
            marvel_usage=marvel_usage,
            next_step=(
                self._turn_next_step(combat, next_actor, combat_over)
                if turn_advanced
                else (
                    f"{attacker_id!r} has {actor.actions_max - actor.actions_used} action(s) left. "
                    "Take another action, or call combat_end_turn to end the turn early. "
                    "A second attack this turn costs a Doom roll."
                )
            ),
        )

    @guard
    def combat_defend(
        self,
        defender_id: str,
        attacker_id: str = "",
        method: str = "dodge",
        incoming_damage: int | None = None,
        ranged: bool = False,
        shield: bool = False,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> CombatDefendResult | ToolEnvelopeFailure:
        """Resolve one player defence against an incoming attack."""
        if method not in ("parry", "dodge"):
            raise CampaignError(
                "invalid_defence_method",
                f"{method!r} is not a defence method.",
                ["Use parry or dodge."],
            )
        if method == "parry" and ranged:
            raise CampaignError(
                "parry_against_ranged",
                "a ranged attack cannot be parried.",
                ["Call combat_defend again with method 'dodge'."],
            )

        with self.store.transaction(
            "combat_defend", actor_id=defender_id, reason=f"defend against {attacker_id or 'threat'}"
        ) as transaction:
            character = transaction.character(defender_id)
            defender_id = character.id
            transaction.actor_id = defender_id
            self._require_actable(character)

            if method == "parry" and not character.weapons:
                raise CampaignError(
                    "nothing_to_parry_with",
                    f"{character.name} holds no weapon or object and cannot parry.",
                    ["Call combat_defend again with method 'dodge'."],
                )

            if attacker_id:
                attacker_id = self._resolve_npc_id(
                    transaction,
                    attacker_id,
                    ["Omit attacker_id for an environmental threat."],
                )
            elif (
                transaction.state.combat.active
                and transaction.state.combat.active_actor in transaction.state.npcs
            ):


                attacker_id = str(transaction.state.combat.active_actor)
                transaction.warn(
                    f"no attacker named; defaulted to {attacker_id!r}, whose turn is open."
                )
            npc = transaction.state.npcs.get(attacker_id) if attacker_id else None
            turn_advanced = False
            combat_over = False
            next_actor: str | None = None
            attacker_turn_exhausted = False
            if npc is not None:
                combat = transaction.state.combat
                npc_actor = combat.actors.get(npc.id)
                if combat.active and npc_actor is not None and npc_actor.turn_open:
                    if npc_actor.actions_used >= npc_actor.actions_max:
                        raise CampaignError(
                            "npc_no_actions_remaining",
                            f"{npc.id!r} has used {npc_actor.actions_used} of "
                            f"{npc_actor.actions_max} action(s) this turn.",
                            [
                                f"Call combat_end_turn for {npc.id!r} before resolving "
                                "another attack from it.",
                            ],
                        )
                    npc_actor.actions_used += 1
                    npc_actor.actions_taken = [*npc_actor.actions_taken, "attack"]
                    npc_actor.reacted = True
                    combat.actors = {**combat.actors, npc.id: npc_actor}
                    transaction.state.combat = combat
                    if npc_actor.actions_used >= npc_actor.actions_max:
                        # Defending already spends the attacker's one action (SKILL.md's
                        # own instruction says so); closing its turn here, rather than
                        # waiting on a second combat_end_turn call the narrator must
                        # remember to make, is what stops that reliance from deadlocking
                        # combat_begin_turn/combat_attack for whoever goes next. Deferred
                        # until after this defence's own damage resolves below: whether
                        # combat itself is now over depends on whether this same roll
                        # just dropped the defender, and checking that here would still
                        # see them standing.
                        attacker_turn_exhausted = True
                elif (
                    combat.active
                    and npc_actor is not None
                    and npc_actor.reacted
                    and npc_actor.actions_used >= npc_actor.actions_max
                ):


                    raise CampaignError(
                        "npc_no_actions_remaining",
                        f"{npc.id!r} already acted this round; its turn is closed.",
                        [
                            f"The active actor is {combat.active_actor!r} in round "
                            f"{combat.round}. Resolve that turn; {npc.id!r} attacks again "
                            "on its own turn, and not before -- do not narrate its attack "
                            "until the order reaches it.",
                        ],
                    )
                elif combat.active and npc_actor is not None and not npc_actor.reacted:
                    npc_actor.reacted = True
                    combat.actors = {**combat.actors, npc.id: npc_actor}
                    transaction.state.combat = combat
            # Range coherence for the enemy's own strike. A live fight ran two full
            # melee "paddle strikes" from NEARBY range because this tool never read
            # the attacker's band -- while the player's own melee was correctly
            # refused for reach at the same distance. An enemy gets a move and an
            # action each turn, so a melee attacker at nearby closes to close as part
            # of the same strike (recorded, so the fight's geometry stays true), and
            # a melee attacker at far away or distant cannot reach at all this turn.
            if npc is not None and transaction.state.combat.active and not ranged:
                live_combat = transaction.state.combat
                band = live_combat.ranges.get(npc.id)
                if band in ("far_away", "distant"):
                    raise CampaignError(
                        "attacker_out_of_reach",
                        f"{npc.name} is at {band.replace('_', ' ')} range: a melee "
                        "strike cannot reach from there, and an enemy closes one band "
                        "per turn. No dice were rolled and nothing was resolved.",
                        [
                            "For a thrown or missile attack, call combat_defend with "
                            "ranged true (dodge only).",
                            f"Otherwise {npc.id!r} spends this turn closing in: "
                            "narrate the advance and call combat_end_turn for it.",
                        ],
                    )
                if band == "nearby":
                    live_combat.ranges = {**live_combat.ranges, npc.id: "close"}
                    transaction.state.combat = live_combat
                    transaction.record(
                        f"{npc.id}: range nearby -> close (closed with its move to "
                        "strike in melee)"
                    )
                    transaction.warn(
                        f"{npc.name} closes from nearby to close range with its move "
                        "to deliver this strike; narrate the advance."
                    )
            damage_value = incoming_damage if incoming_damage is not None else (
                self._npc_damage(npc) if npc else None
            )
            if damage_value is None:
                combat_now = transaction.state.combat
                steps = ["Call campaign_status to read the NPC's damage value."]
                if combat_now.active and combat_now.active_actor:
                    active_now = combat_now.actors.get(combat_now.active_actor)
                    if active_now is not None and active_now.side == "pc":
                        # The usual real cause of an attacker-less defence mid-fight:
                        # the narrator invented an enemy attack during the player's
                        # own turn. Name the truth, or the model retries this exact
                        # call forever -- a live turn recorded seven identical
                        # refusals before the narrator gave up and narrated a hit
                        # that never rolled.
                        steps = [
                            f"It is {combat_now.active_actor!r}'s turn and no enemy "
                            "attack is pending: an enemy acts only on its own turn. "
                            f"Resolve {combat_now.active_actor!r}'s remaining "
                            "action(s) instead of defending.",
                        ]
                raise CampaignError(
                    "incoming_damage_required",
                    "state the incoming damage, or name an NPC attacker whose damage "
                    "is recorded. No dice were rolled and nothing was resolved.",
                    steps,
                )

            use_shield = shield and character.shield and method == "parry"
            attribute = "STR" if method == "parry" else "DEX"
            payload = self._resolve_character_test(
                transaction,
                character,
                attribute,
                advantage=advantage or use_shield,
                disadvantage=disadvantage,
                opponent_level=npc.level if npc else None,
                call_on_doom=False,
                trigger="critical failure on a defence",
                in_combat=True,
            )
            test = payload["test"]

            applied = 0
            absorbed = 0
            shield_broken = False
            stances_spent: list[dict] = []
            if not test.succeeded:
                ignore_armour = test.outcome == "critical_failure"
                applied, absorbed = rules.apply_armour(
                    int(damage_value),
                    character.armour_protection(),
                    ignore_armour=ignore_armour,
                )
                incoming = effects.apply(
                    "damage_incoming",
                    effects.DamageIncomingContext(applied=applied, armour=character.armour),
                    self.data.effects,
                    self._effect_sources(character),
                    active_ids=effects.active_ids(character),
                )
                applied = incoming.applied
                applied = self._bodyguard_redirect(transaction, character, applied)
                previous = character.hp
                character.hp = max(0, character.hp - applied)
                transaction.touch_character(character.id)
                transaction.record(
                    f"{character.id}: hit points {previous} -> {character.hp} ({applied} damage)"
                )
                if ignore_armour and use_shield:
                    character.shield = False
                    shield_broken = True
                    transaction.record(f"{character.id}: shield destroyed on a critical failure")
                if character.hp == 0 and character.status == "ok":
                    if self._become_helpless(
                        transaction, character, "a defence at 0 hit points"
                    ) == "helpless":
                        transaction.record(f"{character.id}: reduced to 0 hit points and Helpless")
                        transaction.warn(
                            f"{character.name} is Helpless. Call helpless_roll once the fight ends "
                            "or the character reaches safety."
                        )

                for ended in incoming.end_effects:
                    character.conditions = [
                        c for c in character.conditions if c.effect_id != ended
                    ]
                    staked = character.declared_choices.get(ended, "")
                    if ended in character.declared_choices:
                        character.declared_choices = {
                            k: v for k, v in character.declared_choices.items() if k != ended
                        }
                    lost = ""
                    if staked and staked in character.weapons:
                        weapons = list(character.weapons)
                        weapons.remove(staked)  # exactly one instance, as the break path does
                        character.weapons = weapons
                        lost = staked
                    transaction.touch_character(character.id)
                    transaction.record(f"{character.id}: stance {ended} ended")
                    if lost:
                        transaction.record(f"{character.id}: {lost} is lost ignoring the attack")
                        transaction.warn(
                            f"{character.name} ignores the attack entirely and loses {lost}."
                        )
                    else:
                        transaction.warn(
                            f"{character.name} ignores the attack entirely; the staked "
                            f"{staked or 'weapon'} was no longer held."
                        )
                    stances_spent.append({"effect_id": ended, "weapon_lost": lost})

            if attacker_turn_exhausted:
                turn_advanced = True
                combat_over, next_actor, _, _, _ = self._advance_combat_turn(
                    transaction, combat, npc.id, tool="combat_defend"
                )
            active_actor = (
                transaction.state.combat.active_actor
                if transaction.state.combat.active
                else None
            )
            commit_payload = {
                "roll": test.roll.as_dict(),
                "outcome": test.outcome,
                "attribute": attribute,
                "method": method,
                "incoming_damage": int(damage_value),
                "applied_damage": applied,
                "absorbed": absorbed,
                "doom": self._doom_log(test.doom, payload.get("critical_doom")),
            }
            if (
                test.outcome == "critical_success"
                and payload["crit_success_max"] > 1
                and test.roll.selected > 1
            ):
                commit_payload["crit_success_max"] = payload["crit_success_max"]
            if stances_spent:
                commit_payload["stances_spent"] = stances_spent
            sequence = transaction.commit(commit_payload)

        if test.succeeded:
            summary = f"{character.name} {'parries' if method == 'parry' else 'dodges'} the attack."
        else:
            summary = (
                f"{character.name} fails to {method} and takes {applied} damage: "
                f"{character.hp}/{character.hp_max} hit points remain."
            )

        facts = [summary]
        if absorbed:
            facts.append(f"Armour absorbs {absorbed} damage.")
        if shield_broken:
            facts.append(f"{character.name}'s shield breaks.")
        if character.status == "helpless":
            facts.append(f"{character.name} falls Helpless.")
        for spent in stances_spent:
            facts.append(
                f"{character.name} ignores the attack entirely"
                + (f" and loses {spent['weapon_lost']}." if spent["weapon_lost"] else ".")
            )

        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            roll=test.roll.as_dict(),
            outcome=test.outcome,
            narration_facts=facts,
            defender_id=defender_id,
            attacker_id=attacker_id,
            method=method,
            # The defence's own test attribute (STR for a parry, DEX for a dodge).
            # Every other rolling tool already names its attribute in the envelope;
            # this one did not, so the defender's roll was invisible to the
            # roll-announcement machinery and no defence was ever announced.
            attribute=attribute,
            # A critical defence failure rolls mandatory Doom the same way a
            # critical attack failure does, but until now this envelope named it
            # nowhere at all -- only the audit log's own committed ``doom`` list
            # (test_event_log.py's own docstring: "combat_defend omits Doom from
            # its return envelope, so the event is the only surface that can
            # record it, which is exactly the gap") ever carried it. Named to match
            # combat_attack's own ``critical_failure_doom`` key exactly, so
            # narrator.engine.doom_facts reads both with the same lookup.
            critical_failure_doom=payload.get("critical_doom"),
            incoming_damage=int(damage_value),
            applied_damage=applied,
            absorbed=absorbed,
            shield_broken=shield_broken,
            defender=self._character_summary(character),
            turn_advanced=turn_advanced,
            combat_over=combat_over,
            next_actor=next_actor,
            round=transaction.state.combat.round,
            next_step=(
                self._turn_next_step(transaction.state.combat, next_actor, combat_over)
                if turn_advanced
                else (
                    "Defending costs the defender no action. "
                    f"The active actor is {active_actor!r}; continue resolving that turn."
                    if active_actor
                    else "No combat is running; defending consumed no action."
                )
            ),
        )

    def _open_combat_turn(self, transaction, combat, actor_id: str) -> CombatActor:
        """Open one actor's turn: reset its counters and set the round's budget.
        """
        actor = combat.actors[actor_id]
        if actor.side == "pc":
            actor.actions_max = actor.first_turn_actions if combat.round == 1 else 2
        else:
            actor.actions_max = 1
        actor.actions_used = 0
        actor.actions_taken = []
        actor.turn_open = True
        actor.reacted = True
        combat.actors = {**combat.actors, actor_id: actor}
        transaction.record(f"{actor_id}: turn opened in round {combat.round}")
        return actor

    def _turn_next_step(self, combat, next_actor: str | None, combat_over: bool) -> str:
        """The one instruction a combat result gives once the turn advanced itself."""
        if combat_over:
            return (
                "Combat is over. Call helpless_roll for any player character at 0 hit "
                "points, then call scene_commit for what the fight changed."
            )
        if next_actor is None:
            return "No actor is able to act."
        side = combat.actors[next_actor].side
        if side == "npc":
            return (
                f"{next_actor!r} acts now in round {combat.round} and its turn is already "
                "open: narrate its action, ask the player to parry or dodge, then call "
                "combat_defend."
            )
        return (
            f"{next_actor!r} acts now in round {combat.round} and their turn is already "
            f"open with {combat.actors[next_actor].actions_max} action(s): resolve them "
            "with combat_attack, combat_move, or attribute_test. The turn advances "
            "itself when the last action is spent."
        )

    def _is_combatant_available(self, transaction, combat, candidate_id: str) -> bool:
        """Whether a combatant can still act: an alive NPC or an ``ok``-status PC."""
        candidate = combat.actors[candidate_id]
        if candidate.side == "npc":
            npc = transaction.state.npcs.get(candidate_id)
            return npc is not None and npc.status == "alive"
        character = transaction.character(candidate_id)
        return character.status == "ok"

    def _close_combat_bookkeeping(
        self,
        transaction,
        combat,
        standing_pcs: list[str],
        *,
        tool: str,
        outcome: str,
        fallback_reason: str,
        record_suffix: str = "",
    ) -> list[dict]:
        """Roll reusable-marvel Usage Dice for standing PCs and stamp the fight's
        fiction-debt entry.

        Shared by ``_advance_combat_turn`` (a decisive close, one side empty) and
        ``combat_close`` (any other close), which differ only in the audit record's
        own message, the ``tool``/``outcome`` tags on the debt entry, and the
        reason it falls back to when the fight itself declared none.
        """
        combat.active = False
        combat.active_actor = None
        transaction.record(f"combat closed after round {combat.round}{record_suffix}")
        marvel_usage: list[dict] = []
        for character_id in combat.order:
            participant = combat.actors[character_id]
            if participant.side != "pc":
                continue
            character = transaction.character(character_id)
            for power_id in list(character.powers):
                resolved = self.data.subsystem_power(power_id)
                if not resolved or resolved["entry"].get("single_use"):
                    continue
                if resolved["subsystem"] != "twisted_science":
                    continue
                usage = self._roll_marvel_usage(character, power_id)
                marvel_usage.append(
                    {
                        "character_id": character.id,
                        "power_id": power_id,
                        "marvel": resolved["entry"]["name"],
                        **usage,
                    }
                )
                transaction.touch_character(character.id)
                transaction.record(
                    f"{character.id}: rolled {power_id} Usage Die after combat"
                )
        # A fight's durable fiction crystallizes at its end. The debt
        # entry carries any stakes declared at combat_start, realized
        # by whether a player character is still standing.
        transaction.add_fiction_debt(
            tool=tool,
            actor_id=",".join(standing_pcs),
            reason=combat.reason or fallback_reason,
            outcome=outcome,
            stakes=StakesDeclaration(
                success=combat.stakes_success,
                failure=combat.stakes_failure,
                hidden=combat.stakes_hidden,
            ),
            realized="success" if standing_pcs else "failure",
        )
        return marvel_usage

    def _advance_combat_turn(
        self, transaction, combat, actor_id: str, *, tool: str
    ) -> tuple[bool, str | None, int, list[dict], list[str]]:
        """Close ``actor_id``'s turn and advance to the next available combatant.

        Shared by ``combat_end_turn``, by ``combat_defend`` (when a resolved
        defence exhausts the attacking NPC's own action budget), and by
        ``combat_attack``/``combat_move``/``attribute_test`` (when a player
        character spends their last action), rather than leaving that transition
        to a second tool call the narrator must remember to make. The next
        actor's turn opens here too, for the reason ``_open_combat_turn``
        documents. Returns
        ``(combat_over, next_actor, round_number, marvel_usage, standing_pcs)``.
        """
        actor = combat.actors[actor_id]
        actor.turn_open = False
        combat.actors = {**combat.actors, actor_id: actor}

        alive_npcs = [
            actor_key
            for actor_key in combat.order
            if combat.actors[actor_key].side == "npc"
            and self._is_combatant_available(transaction, combat, actor_key)
        ]
        standing_pcs = [
            actor_key
            for actor_key in combat.order
            if combat.actors[actor_key].side == "pc"
            and self._is_combatant_available(transaction, combat, actor_key)
        ]

        combat_over = not alive_npcs or not standing_pcs
        round_number = combat.round
        next_actor = None
        marvel_usage: list[dict] = []

        if combat_over:
            marvel_usage = self._close_combat_bookkeeping(
                transaction, combat, standing_pcs,
                tool=tool, outcome="combat_ended", fallback_reason="combat resolved",
            )
        else:
            index = combat.order.index(actor_id)
            for offset in range(1, len(combat.order) + 1):
                candidate_id = combat.order[(index + offset) % len(combat.order)]
                if not self._is_combatant_available(transaction, combat, candidate_id):
                    continue
                if (index + offset) >= len(combat.order):
                    combat.round += 1
                next_actor = candidate_id
                break
            combat.active_actor = next_actor
            round_number = combat.round
            if next_actor is not None:
                self._open_combat_turn(transaction, combat, next_actor)

        transaction.state.combat = combat
        return combat_over, next_actor, round_number, marvel_usage, standing_pcs

    @guard
    def combat_end_turn(self, actor_id: str = "") -> CombatEndTurnResult | ToolEnvelopeFailure:
        """Close the active turn and advance to the next actor."""
        with self.store.transaction(
            "combat_end_turn", actor_id=actor_id, reason="end turn"
        ) as transaction:
            combat = self._require_combat(transaction)
            current = (
                self._resolve_combat_actor(transaction, combat, actor_id)
                if actor_id
                else combat.active_actor
            )
            transaction.actor_id = current or ""
            if current not in combat.actors:
                raise CampaignError(
                    "actor_not_in_combat",
                    f"{current!r} is not part of this combat.",
                    [f"Combat order: {', '.join(combat.order)}."],
                )
            if combat.active_actor != current:
                raise CampaignError(
                    "not_active_actor",
                    f"the active actor is {combat.active_actor!r}, not {current!r}.",
                    [f"Call combat_end_turn for {combat.active_actor!r}."],
                )

            combat_over, next_actor, round_number, marvel_usage, standing_pcs = (
                self._advance_combat_turn(transaction, combat, current, tool="combat_end_turn")
            )
            sequence = transaction.commit(
                {
                    "outcome": "combat_ended" if combat_over else "turn_ended",
                    "round": round_number,
                    "next_actor": next_actor,
                    "marvel_usage": marvel_usage,
                }
            )

        if combat_over:
            summary = "Combat ends."
            facts = ["The fight is over."]
            if not standing_pcs:
                facts.append("No player character is still standing.")
        else:
            summary = f"{current} ends the turn. {next_actor} acts next in round {round_number}."
            facts = []

        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            outcome="combat_ended" if combat_over else "turn_ended",
            narration_facts=facts,
            combat_active=not combat_over,
            round=round_number,
            next_actor=next_actor,
            marvel_usage=marvel_usage,
            next_step=(
                "Combat is over. Call helpless_roll for any player character at 0 hit points, "
                "then call scene_commit for what the fight changed."
                if combat_over
                else self._turn_next_step(combat, next_actor, combat_over)
            ),
        )

    @guard
    def combat_close(self, reason: str) -> CombatCloseResult | ToolEnvelopeFailure:
        """Close an active combat for any reason other than elimination.

        ``combat_end_turn`` ends a fight only when one side is empty: no NPC is
        alive, or no player character is standing. Every other way a fight ends
        -- the party fleeing, an NPC surrendering, a truce, or any other mutual
        disengagement -- had no tool to close it, so an abandoned fight could
        stay ``combat.active`` indefinitely, blocking every later
        ``combat_start`` and contradicting the fiction. This tool is that
        missing path: it closes the fight, clears the active actor, and rolls
        the same post-combat bookkeeping (reusable marvel Usage Dice, the
        fiction-debt entry) that a decisive ``combat_end_turn`` would.

        ``reason`` is free text, not a closed vocabulary: the whole point of
        this tool is to close a fight for any reason an elimination check does
        not already cover, so a fixed word list would just reproduce the
        stuck-fight bug for any reason outside it. A non-empty reason is
        required because the audit event is the only durable record of why
        the fight ended without a decisive blow.
        """
        reason = reason.strip()
        if not reason:
            raise CampaignError(
                "empty_close_reason",
                "a combat close needs a reason. The audit event is the only record of "
                "why the fight ended without a decisive combat_end_turn.",
                [
                    "State in a few words why the fight ends (fled, surrendered, a truce, "
                    "etc.), then call again."
                ],
            )

        with self.store.transaction("combat_close", actor_id="", reason=reason) as transaction:
            combat = self._require_combat(transaction)
            round_number = combat.round

            standing_pcs = [
                actor_key
                for actor_key in combat.order
                if combat.actors[actor_key].side == "pc"
                and self._is_combatant_available(transaction, combat, actor_key)
            ]

            # Same crystallization ``combat_end_turn`` performs at a decisive close: the
            # debt entry carries any stakes declared at combat_start, realized by whether
            # a player character is still standing when the fight closes.
            marvel_usage = self._close_combat_bookkeeping(
                transaction, combat, standing_pcs,
                tool="combat_close", outcome="combat_closed", fallback_reason=reason,
                record_suffix=f": {reason}",
            )

            transaction.state.combat = combat
            sequence = transaction.commit(
                {
                    "outcome": "combat_closed",
                    "reason": reason,
                    "round": round_number,
                    "marvel_usage": marvel_usage,
                }
            )

        return success_after_commit(
            transaction,
            f"Combat ends: {reason}.",
            sequence=sequence,
            outcome="combat_closed",
            narration_facts=[f"The fight ends: {reason}."],
            combat_active=False,
            round=round_number,
            reason=reason,
            marvel_usage=marvel_usage,
            next_step=(
                "Call helpless_roll for any player character at 0 hit points, then call "
                "scene_commit for what the fight changed and where the party goes."
            ),
        )
