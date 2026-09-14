"""Roll-under attribute tests (solo and group), Usage Die rolls, and Doom rolls."""

from __future__ import annotations

from typing import NotRequired

from .. import effects, rules
from ..dice import DEPLETED
from ..models import ATTRIBUTE_NAMES, Character, StakesDeclaration
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from ..store import CampaignError, Transaction
from .common import DOOMED_CONDITION_ID, guard, success_after_commit
from .scene import _realized_branch


class AttributeTestResult(ToolEnvelopeSuccess):
    character_id: str
    attribute: str
    target: int
    threat_modifier: int
    failure_mode: str
    stakes: dict
    realized: str
    combat_action: NotRequired[dict]
    next_step: NotRequired[str]
    category: NotRequired[str]
    called_on_doom: NotRequired[dict]
    doom: NotRequired[dict]
    torn_veil: NotRequired[dict]


class GroupTestResult(ToolEnvelopeSuccess):
    attribute: str
    individual: list[dict]
    successes: int
    participants: int
    stakes: dict
    realized: str
    category: str


class UsageRollResult(ToolEnvelopeSuccess):
    resource: dict


class DoomRollResult(ToolEnvelopeSuccess):
    doom: dict
    subtract_from_test: int


class AttributeTestsMixin:
    """Roll-under attribute tests, Usage Die rolls, and Doom rolls."""

    def _require_stakes(
        self,
        transaction: Transaction,
        stakes_success: str,
        stakes_failure: str,
        stakes_hidden: str,
        reason: str,
        *,
        required: bool,
    ) -> StakesDeclaration:
        """Validate narrator-declared stakes.

        Required stakes reject blank branches: a narrator who cannot state
        what a roll risks has no business calling for it. Quality is never
        adjudicated — a duplicate or reason-echo only warns, because the
        server must not judge fiction.
        """
        stakes = StakesDeclaration(
            success=str(stakes_success).strip(),
            failure=str(stakes_failure).strip(),
            hidden=str(stakes_hidden).strip(),
        )
        if required and (not stakes.success or not stakes.failure):
            raise CampaignError(
                "empty_stakes",
                "declare what durably changes on success and on failure before rolling. "
                "A roll with no stakes should not be rolled.",
                [
                    "State the consequence of each branch in one sentence each.",
                    "If nothing durable is at stake, do not call for a test: the action "
                    "simply happens.",
                ],
            )
        if stakes.success and stakes.success == stakes.failure:
            transaction.warn(
                "stakes_success and stakes_failure are identical; the roll decides nothing "
                "unless the branches differ."
            )
        cleaned_reason = str(reason).strip()
        if cleaned_reason and cleaned_reason in (stakes.success, stakes.failure):
            transaction.warn(
                "a stakes branch repeats the reason verbatim; it declares no consequence."
            )
        return stakes

    def _require_attribute(self, attribute: str) -> str:
        key = attribute.strip().upper()
        if key not in ATTRIBUTE_NAMES:
            raise CampaignError(
                "invalid_attribute",
                f"{attribute!r} is not an attribute.",
                [f"Use one of {', '.join(ATTRIBUTE_NAMES)}."],
            )
        return key

    def _require_test_category(self, category: str) -> str:
        normalized = category.strip().lower()
        if not normalized:
            return ""
        legal = self.data.effects.get("test_categories", [])
        if normalized not in legal:
            raise CampaignError(
                "invalid_test_category",
                f"{category!r} is not a test category.",
                [f"Use one of: {', '.join(sorted(legal))}. Or omit category."],
            )
        return normalized

    def _require_actable(self, character: Character) -> None:
        if character.status == "dead":
            raise CampaignError(
                "character_dead",
                f"{character.name} is dead and takes no actions.",
                ["Create a replacement character with character_create."],
            )
        if character.status == "helpless":
            raise CampaignError(
                "character_helpless",
                f"{character.name} is Helpless at 0 hit points and takes no actions.",
                [
                    "Resolve the fight, then call helpless_roll for this character.",
                    "Another character may drag them to safety; record that with scene_commit.",
                ],
            )

    # -- attribute tests -----------------------------------------------------

    @guard
    def attribute_test(
        self,
        character_id: str,
        attribute: str,
        reason: str,
        stakes_success: str,
        stakes_failure: str,
        stakes_hidden: str = "",
        advantage: bool = False,
        disadvantage: bool = False,
        opponent_level: int | None = None,
        call_on_doom: bool = False,
        failure_mode: str = "gm_choice",
        category: str = "",
    ) -> AttributeTestResult | ToolEnvelopeFailure:
        """Resolve one roll-under attribute test with declared stakes."""
        key = self._require_attribute(attribute)
        category = self._require_test_category(category)
        if failure_mode not in ("fail", "success_at_cost", "gm_choice"):
            raise CampaignError(
                "invalid_failure_mode",
                f"{failure_mode!r} is not a failure mode.",
                ["Use fail, success_at_cost, or gm_choice."],
            )

        with self.store.transaction(
            "attribute_test", actor_id=character_id, reason=reason
        ) as transaction:
            stakes = self._require_stakes(
                transaction, stakes_success, stakes_failure, stakes_hidden, reason,
                required=True,
            )
            character = transaction.character(character_id)
            transaction.actor_id = character.id
            self._require_actable(character)
            payload = self._resolve_character_test(
                transaction,
                character,
                key,
                advantage=advantage,
                disadvantage=disadvantage,
                opponent_level=opponent_level,
                call_on_doom=call_on_doom,
                trigger="critical failure on an attribute test",
                category=category,
            )
            test = payload["test"]
            if category == "sorcery" and test.outcome == "critical_failure":
                payload["torn_veil"] = self._resolve_torn_veil(transaction, character)
            realized = _realized_branch(test.outcome)
            transaction.add_fiction_debt(
                tool="attribute_test",
                actor_id=character.id,
                reason=transaction.reason,
                outcome=test.outcome,
                stakes=stakes,
                realized=realized,
            )


            combat = transaction.state.combat
            combat_action = None
            if (
                combat.active
                and combat.active_actor == character.id
                and character.id in combat.actors
                and combat.actors[character.id].side == "pc"
                and combat.actors[character.id].turn_open
            ):
                actor = combat.actors[character.id]
                actor.actions_used += 1
                actor.actions_taken = actor.actions_taken + ["test"]
                combat.actors = {**combat.actors, character.id: actor}
                transaction.state.combat = combat
                transaction.record(
                    f"{character.id}: the test spent a combat action "
                    f"({actor.actions_used} of {actor.actions_max})"
                )
                turn_advanced = False
                combat_over = False
                next_actor = None
                if actor.actions_used >= actor.actions_max:
                    turn_advanced = True
                    combat_over, next_actor, _, _, _ = self._advance_combat_turn(
                        transaction, combat, character.id, tool="attribute_test"
                    )
                combat_action = {
                    "actions_used": actor.actions_used,
                    "actions_max": actor.actions_max,
                    "turn_advanced": turn_advanced,
                    "combat_over": combat_over,
                    "next_actor": next_actor,
                    "round": combat.round,
                }
            commit_payload = {
                "roll": test.roll.as_dict(),
                "outcome": test.outcome,
                "attribute": key,
                "failure_mode": failure_mode,
                "stakes": stakes.model_dump(),
                "realized": realized,
                "doom": self._doom_log(test.doom, payload.get("critical_doom")),
            }
            if category:
                commit_payload["category"] = category
            if payload.get("torn_veil"):
                commit_payload["torn_veil"] = payload["torn_veil"]
            if combat_action is not None:
                commit_payload["combat_action"] = combat_action
            sequence = transaction.commit(commit_payload)

        next_step = None
        if combat_action is not None:
            if combat_action["turn_advanced"]:
                next_step = self._turn_next_step(
                    combat, combat_action["next_actor"], combat_action["combat_over"]
                )
            else:
                remaining = combat_action["actions_max"] - combat_action["actions_used"]
                next_step = (
                    f"The test spent one of {character.id!r}'s combat actions; "
                    f"{remaining} action(s) left this turn."
                )
        return self._test_envelope(
            transaction,
            character,
            key,
            payload,
            sequence,
            reason,
            failure_mode,
            stakes=stakes,
            realized=realized,
            category=category,
            combat_action=combat_action,
            next_step=next_step,
        )

    def _resolve_character_test(
        self,
        transaction: Transaction,
        character: Character,
        attribute: str,
        *,
        advantage: bool,
        disadvantage: bool,
        opponent_level: int | None,
        call_on_doom: bool,
        trigger: str,
        category: str = "",
        in_combat: bool = False,
    ) -> dict:
        """Shared attribute-test pipeline: conditions, Doom, Threat Level, dice.

        ``in_combat`` is an engine-set fact — true only for combat_attack's
        to-hit roll and combat_defend's defense roll, set by their own internal
        logic. It is never a served-tool parameter and never derived from
        caller-supplied state; it widens the critical-success band only for a
        Gift (e.g. Battle Hardened) that hooks attribute_test with
        when_in_combat.
        """
        crit_success_max = 1
        if category or in_combat:
            test_ctx = effects.apply(
                "attribute_test",
                effects.TestContext(attribute=attribute, category=category, in_combat=in_combat),
                self.data.effects,
                self._effect_sources(character),
                active_ids=effects.active_ids(character),
            )
            if test_ctx.advantage and not advantage:
                advantage = True
                transaction.warn(
                    f"{character.name}'s background grants Advantage on a {category} test."
                )
            crit_success_max = test_ctx.crit_success_max

        edge_advantage, edge_disadvantage, notes = rules.character_edges(
            character, attribute, advantage, disadvantage
        )
        for note in notes:
            transaction.warn(note)

        threat = rules.threat_modifier(character.level, opponent_level)
        doom_called = None
        doom_penalty = 0
        if call_on_doom:
            if character.doom_die == DEPLETED:
                raise CampaignError(
                    "doom_depleted",
                    f"{character.name} has a spent Doom die and cannot call on Doom.",
                    ["Take a long rest in a safe place to restore the Doom die."],
                )
            doom_called = rules.call_on_doom(self.roller, character.doom_die)
            doom_penalty = doom_called.roll.selected if doom_called.roll else 0
            self._apply_doom_step(transaction, character, doom_called.current_die)
            transaction.record(
                f"{character.id}: called on Doom, {doom_called.previous_die} -> "
                f"{doom_called.current_die}, subtracting {doom_penalty}"
            )

        target = character.attributes.get(attribute)
        test = rules.resolve_test(
            self.roller,
            target=target,
            advantage=edge_advantage,
            disadvantage=edge_disadvantage,
            modifier=threat - doom_penalty,
            crit_success_max=crit_success_max,
        )
        test.threat_modifier = threat
        test.doom_penalty = doom_penalty
        test.doom = doom_called

        critical_doom = None
        if test.outcome == "critical_failure":
            critical_doom = self._mandatory_doom(transaction, character, trigger)

        return {
            "test": test,
            "critical_doom": critical_doom,
            "threat": threat,
            "crit_success_max": crit_success_max,
        }

    def _test_envelope(
        self,
        transaction: Transaction,
        character: Character,
        attribute: str,
        payload: dict,
        sequence: int,
        reason: str,
        failure_mode: str,
        stakes: StakesDeclaration | None = None,
        realized: str = "",
        category: str = "",
        combat_action: dict | None = None,
        next_step: str | None = None,
    ) -> dict:
        test = payload["test"]
        verb = {
            "critical_success": "critically succeeds",
            "success": "succeeds",
            "failure": "fails",
            "critical_failure": "critically fails",
        }[test.outcome]
        summary = f"{character.name} {verb} on a {attribute} test."
        facts = [f"{character.name} {verb} at: {reason}" if reason else summary]
        if test.outcome == "failure" and failure_mode == "success_at_cost":
            facts.append(
                "The narrator owes a concrete cost: the action happens, but something is lost, "
                "delayed, or noticed."
            )
        extra = {


            "character_id": character.id,
            "attribute": attribute,
            "target": test.target,
            "threat_modifier": test.threat_modifier,
            "failure_mode": failure_mode,
        }
        if combat_action is not None:
            extra["combat_action"] = combat_action
        if next_step is not None:
            extra["next_step"] = next_step
        if category:
            extra["category"] = category
        if stakes is not None:
            extra["stakes"] = stakes.model_dump()
            extra["realized"] = realized
            realized_text = stakes.success if realized == "success" else stakes.failure
            if realized_text:
                # The fiction the narrator is licensed to speak and the fiction
                # the files recorded are the same string. Hidden stakes never
                # enter narration_facts.
                facts.append(realized_text)
        if test.doom is not None:
            extra["called_on_doom"] = test.doom.as_dict()
        if payload.get("critical_doom"):
            extra["doom"] = payload["critical_doom"]
            facts.append(f"{character.name}'s Doom answers the critical failure.")
        if payload.get("torn_veil"):
            extra["torn_veil"] = payload["torn_veil"]
            facts.append(f"{character.name}'s casting tears the veil.")
        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            roll=test.roll.as_dict(),
            outcome=test.outcome,
            narration_facts=facts,
            **extra,
        )

    @guard
    def group_test(
        self,
        character_ids: list[str],
        attribute: str,
        reason: str,
        stakes_success: str,
        stakes_failure: str,
        stakes_hidden: str = "",
        advantage_character_ids: list[str] | None = None,
        disadvantage_character_ids: list[str] | None = None,
        opponent_level: int | None = None,
        category: str = "",
    ) -> GroupTestResult | ToolEnvelopeFailure:
        """Resolve one group test. The group succeeds when at least half succeed."""
        key = self._require_attribute(attribute)
        category = self._require_test_category(category)
        if not character_ids:
            raise CampaignError(
                "no_participants",
                "a group test needs at least one character id.",
                ["Call campaign_status to list valid character ids."],
            )
        individual: list[dict] = []
        with self.store.transaction(
            "group_test", actor_id=",".join(character_ids), reason=reason
        ) as transaction:
            stakes = self._require_stakes(
                transaction, stakes_success, stakes_failure, stakes_hidden, reason,
                required=True,
            )
            # Resolve every id first so an advantage list naming 'Ossa' still
            # matches the participant loaded as 'ossa'.
            advantaged = {
                self.store.resolve_character_id(c)
                for c in (advantage_character_ids or [])
            }
            disadvantaged = {
                self.store.resolve_character_id(c)
                for c in (disadvantage_character_ids or [])
            }
            resolved_participants: list[str] = []
            for character_id in character_ids:
                character = transaction.character(character_id)
                resolved_participants.append(character.id)
                self._require_actable(character)
                payload = self._resolve_character_test(
                    transaction,
                    character,
                    key,
                    advantage=character.id in advantaged,
                    disadvantage=character.id in disadvantaged,
                    opponent_level=opponent_level,
                    call_on_doom=False,
                    trigger="critical failure on a group test",
                    category=category,
                )
                test = payload["test"]
                entry = {
                    "character_id": character.id,
                    "name": character.name,
                    "roll": test.roll.as_dict(),
                    "outcome": test.outcome,
                    "target": test.target,
                    "succeeded": test.succeeded,
                }
                if payload.get("critical_doom"):
                    entry["doom"] = payload["critical_doom"]
                individual.append(entry)

            transaction.actor_id = ",".join(resolved_participants)
            successes = sum(1 for entry in individual if entry["succeeded"])
            group_success = rules.group_succeeds(successes, len(individual))
            realized = "success" if group_success else "failure"
            transaction.add_fiction_debt(
                tool="group_test",
                actor_id=transaction.actor_id,
                reason=transaction.reason,
                outcome=realized,
                stakes=stakes,
                realized=realized,
            )
            group_commit_payload = {
                "outcome": realized,
                "attribute": key,
                "individual": individual,
                "successes": successes,
                "participants": len(individual),
                "stakes": stakes.model_dump(),
                "realized": realized,
            }
            if category:
                group_commit_payload["category"] = category
            sequence = transaction.commit(group_commit_payload)

        outcome = "success" if group_success else "failure"
        return success_after_commit(
            transaction,
            f"The group {'succeeds' if group_success else 'fails'}: "
            f"{successes} of {len(individual)} pass the {key} test.",
            sequence=sequence,
            outcome=outcome,
            narration_facts=[
                f"{entry['name']} {'passes' if entry['succeeded'] else 'falls short'} "
                f"on {key}."
                for entry in individual
            ]
            + ([stakes.success if group_success else stakes.failure]
               if (stakes.success if group_success else stakes.failure) else []),
            attribute=key,
            individual=individual,
            successes=successes,
            participants=len(individual),
            stakes=stakes.model_dump(),
            realized=outcome,
            category=category,
        )

    # -- usage dice and Doom -------------------------------------------------

    @guard
    def usage_roll(
        self,
        owner_id: str,
        resource_id: str,
        reason: str,
        advantage: bool = False,
        disadvantage: bool = False,
        stakes_success: str = "",
        stakes_failure: str = "",
        stakes_hidden: str = "",
    ) -> UsageRollResult | ToolEnvelopeFailure:
        """Roll one Usage Die. A result of 1 or 2 steps the die down."""
        with self.store.transaction(
            "usage_roll", actor_id=owner_id, reason=reason
        ) as transaction:
            stakes = self._require_stakes(
                transaction, stakes_success, stakes_failure, stakes_hidden, reason,
                required=False,
            )
            character = transaction.character(owner_id)
            transaction.actor_id = character.id
            resource = next(
                (item for item in character.resources if item.id == resource_id), None
            )
            if resource is None:
                raise CampaignError(
                    "resource_not_found",
                    f"{character.name} carries no resource with id {resource_id!r}.",
                    [
                        "Call character_sheet to list the character's tracked resources.",
                        "Record a new resource through scene_commit before rolling it.",
                    ],
                )
            if resource.die == DEPLETED:
                raise CampaignError(
                    "resource_depleted",
                    f"{resource.name} is depleted and cannot be rolled.",
                    ["Resupply in fiction, then record the new Usage Die."],
                )

            outcome = rules.roll_usage(
                self.roller, resource.die, advantage=advantage, disadvantage=disadvantage
            )
            if outcome.downgraded:
                resource.die = outcome.current_die
                character.resources = list(character.resources)
                transaction.touch_character(character.id)
                transaction.record(
                    f"{character.id}: {resource.id} {outcome.previous_die} -> {outcome.current_die}"
                )
            usage_outcome = (
                "depleted" if outcome.depleted
                else ("downgraded" if outcome.downgraded else "held")
            )
            # A downgrade or depletion is durable fiction even with no stakes
            # declared: the resource's decline must survive a restart's read.
            if not stakes.is_empty() or outcome.downgraded:
                transaction.add_fiction_debt(
                    tool="usage_roll",
                    actor_id=character.id,
                    reason=transaction.reason,
                    outcome=usage_outcome,
                    stakes=stakes,
                    realized="failure" if outcome.downgraded else "success",
                )
            sequence = transaction.commit(
                {
                    "roll": outcome.roll.as_dict(),
                    "outcome": usage_outcome,
                    "resource_id": resource_id,
                    "stakes": stakes.model_dump(),
                }
            )

        if outcome.depleted:
            summary = f"{resource.name} runs out."
        elif outcome.downgraded:
            summary = f"{resource.name} drops to {outcome.current_die}."
        else:
            summary = f"{resource.name} holds at {outcome.current_die}."
        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            roll=outcome.roll.as_dict(),
            outcome="depleted" if outcome.depleted else ("downgraded" if outcome.downgraded else "held"),
            narration_facts=[summary],
            resource={
                "id": resource.id,
                "name": resource.name,
                "previous_die": outcome.previous_die,
                "current_die": outcome.current_die,
            },
        )

    @guard
    def doom_roll(
        self,
        character_id: str,
        reason: str,
        mode: str = "roll",
        disadvantage: bool = False,
        stakes_success: str = "",
        stakes_failure: str = "",
        stakes_hidden: str = "",
    ) -> DoomRollResult | ToolEnvelopeFailure:
        """Roll, call on, or restore a character's Doom die."""
        if mode not in ("roll", "call_on_doom", "restore"):
            raise CampaignError(
                "invalid_mode",
                f"{mode!r} is not a Doom mode.",
                ["Use roll, call_on_doom, or restore."],
            )

        with self.store.transaction(
            "doom_roll", actor_id=character_id, reason=reason
        ) as transaction:
            stakes = self._require_stakes(
                transaction, stakes_success, stakes_failure, stakes_hidden, reason,
                required=False,
            )
            if mode == "restore" and not stakes.is_empty():
                transaction.warn("stakes are ignored on a Doom restore; nothing is rolled.")
            character = transaction.character(character_id)
            transaction.actor_id = character.id

            if mode == "restore":
                outcome = rules.restore_doom(character.doom_max)
                character.doom_die = character.doom_max
                character.conditions = [
                    condition
                    for condition in character.conditions
                    if condition.id != DOOMED_CONDITION_ID
                ]
                transaction.touch_character(character.id)
                transaction.record(f"{character.id}: Doom restored to {character.doom_max}")
            else:
                if character.doom_die == DEPLETED:
                    raise CampaignError(
                        "doom_depleted",
                        f"{character.name} has a spent Doom die.",
                        ["Take a long rest in a safe place to restore the Doom die."],
                    )
                if mode == "roll":
                    outcome = rules.roll_doom(
                        self.roller, character.doom_die, disadvantage=disadvantage
                    )
                else:
                    outcome = rules.call_on_doom(
                        self.roller, character.doom_die, disadvantage=disadvantage
                    )
                if outcome.downgraded:
                    self._apply_doom_step(transaction, character, outcome.current_die)
                    transaction.record(
                        f"{character.id}: Doom {outcome.previous_die} -> {outcome.current_die}"
                    )

            if mode != "restore" and (not stakes.is_empty() or outcome.depleted):
                # A spent Doom die is durable fiction: the character is Doomed
                # until a long rest, and the files must say so on their own.
                transaction.add_fiction_debt(
                    tool="doom_roll",
                    actor_id=character.id,
                    reason=transaction.reason,
                    outcome="doomed" if outcome.depleted else outcome.mode,
                    stakes=stakes,
                    realized="failure" if outcome.downgraded else "success",
                )
            sequence = transaction.commit(
                {
                    "roll": outcome.roll.as_dict() if outcome.roll else None,
                    "outcome": outcome.mode,
                    "doom": outcome.as_dict(),
                    "stakes": stakes.model_dump(),
                }
            )

        if mode == "restore":
            summary = f"{character.name}'s Doom die returns to {character.doom_max}."
        elif outcome.depleted:
            summary = f"{character.name}'s Doom die is spent. {character.name} is Doomed."
        elif outcome.downgraded:
            summary = f"{character.name}'s Doom falls to {outcome.current_die}."
        else:
            summary = f"{character.name}'s Doom holds at {outcome.current_die}."

        return success_after_commit(
            transaction,
            summary,
            sequence=sequence,
            roll=outcome.roll.as_dict() if outcome.roll else None,
            outcome=outcome.mode,
            narration_facts=[summary],
            doom=outcome.as_dict(),
            subtract_from_test=outcome.roll.selected
            if mode == "call_on_doom" and outcome.roll
            else 0,
        )
