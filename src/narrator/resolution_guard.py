"""Runtime enforcement of one continuation's selected mechanical directives.

Moved out of ``decisions.py``, which stayed the home of this class even though it is a
different concern from everything else that module holds: ``decisions.py`` defines the
typed, tool-less contracts a planner emits (what a resolution's own directive *is*);
this module enforces one already-selected set of them against the live MCP tool calls
a turn actually makes. The five directive/resolution types it validates against
(``AttributeTestDirective``, ``CombatDefendDirective``, ``DecisionResolution``,
``HazardResolutionDirective``, ``NoTestDirective``) stay in ``decisions.py`` and are
imported back here -- they are genuinely part of that module's own contract vocabulary,
used throughout it, not private to this class.
"""

from __future__ import annotations

from narrator.decisions import (
    AttributeTestDirective,
    CombatDefendDirective,
    DecisionResolution,
    HazardResolutionDirective,
    NoTestDirective,
)
from narrator.social import SocialTestGuard


class ResolutionGuard:
    """Enforce one continuation's selected mechanical directives before MCP calls."""

    _relevant_tools = frozenset({"attribute_test", "group_test", "combat_defend"})

    def __init__(
        self, resolutions: tuple[DecisionResolution, ...] = (), action_fingerprint: str = "",
        social_request=None,
    ) -> None:
        self._resolutions = tuple(resolutions)
        all_directives = tuple(
            resolution.directive
            for resolution in self._resolutions
            if resolution.directive is not None
        )
        # Hazard directives are tracked apart from the test-and-defence directives, so
        # they never enter the attribute_test/combat_defend matching in ``validate``.
        # A confirmed violent action is satisfied by a roll, not by matching an actor to
        # a pre-selected mechanic, so it needs its own after-the-fact accounting.
        self._directives = tuple(
            item for item in all_directives if not isinstance(item, HazardResolutionDirective)
        )
        self._hazards = tuple(
            item for item in all_directives if isinstance(item, HazardResolutionDirective)
        )
        self._hazards_satisfied: set[str] = set()
        self._action_fingerprint = action_fingerprint
        self._used: set[int] = set()
        #: Directive indexes whose bound attempt the engine itself refused (the call
        #: matched the directive and executed, but returned ok False). A refusal
        #: unlocks the actor's other mechanics -- a bound parry the engine refuses
        #: against a ranged attack must not lock out the dodge that can resolve --
        #: without counting as the resolution itself.
        self._refused: set[int] = set()
        self._social_request = social_request
        self._social_used = False
        self._social_substitution = ""
        self._social_test_guard = SocialTestGuard()
        self._social_outcome = None

    def validate(self, tool_name: str, arguments: dict) -> str | None:
        """Return a refusal reason before a mismatched model call reaches MCP."""
        request = self._social_request
        if request is not None and tool_name == "use_ability":
            return self._validate_social_ability_arguments(arguments)
        if request is not None and tool_name == "attribute_test":
            refusal = self.validate_social_test(
                actor_id=str(arguments.get("character_id", "")),
                attribute=str(arguments.get("attribute", "")),
                action_fingerprint=request.action_fingerprint,
            )
            # No pre-execution marking here: ``record_tool_result`` marks the
            # matching directive used only once the test actually returns ok, the
            # same deferral ``validate`` itself now keeps below.
            return refusal
        if tool_name not in self._relevant_tools:
            return None
        if any(
            resolution.action_fingerprint
            and resolution.action_fingerprint != self._action_fingerprint
            for resolution in self._resolutions
        ):
            return "the selected decision belongs to a different action"
        if any(
            isinstance(item, NoTestDirective)
            and self._targets_character(tool_name, arguments, item.character_id)
            for item in self._directives
        ):
            return "the selected decision forbids a test or defence for this action"
        expected = [
            (index, item)
            for index, item in enumerate(self._directives)
            if index not in self._used and not isinstance(item, NoTestDirective)
        ]
        if not expected:
            return None


        for index, item in expected:
            if self._directive_matches(index, item, tool_name, arguments):
                return None
        return "the selected decision requires a different actor or mechanic"

    def _directive_matches(
        self, index: int, item, tool_name: str, arguments: dict
    ) -> bool:
        """Whether one call is the bound mechanic (or its post-refusal fallback).

        A bound attribute test also accepts the same actor's own ``combat_attack``.
        A live turn bound a STR test for \"I punch the trader in the face\", the model
        resolved it through the real attack tool instead -- audited dice, damage
        applied, an action spent -- and the guard withheld the turn as unresolved, so
        the player never saw a roll that genuinely happened. An attack is the
        stricter mechanic, not an evasion of the bound one; the mirror-image
        equivalence already holds in ``_hazard_tool_matches``, where a confirmed
        attack accepts an ``attribute_test`` as its resolution.
        """
        if isinstance(item, AttributeTestDirective):
            if tool_name == "combat_attack" and self._same_actor(
                arguments.get("attacker_id"), item.character_id
            ):
                return True
            if tool_name == "grant_runic_weapon" and self._same_actor(
                arguments.get("character_id"), item.character_id
            ):
                return True
            return (
                tool_name == "attribute_test"
                and self._same_actor(arguments.get("character_id"), item.character_id)
                and arguments.get("attribute") == item.attribute
            )
        if isinstance(item, CombatDefendDirective):
            return (
                tool_name == "combat_defend"
                and self._same_actor(arguments.get("defender_id"), item.character_id)
                and (arguments.get("method") == item.method or index in self._refused)
            )
        return False

    @staticmethod
    def _same_actor(supplied: object, bound: str) -> bool:
        """Whether a tool argument names the bound character, tolerating spelling.

        The engine itself resolves a loose identifier -- ``Transaction.character``
        accepts a display name or a case variant and warns -- so a call the engine
        would resolve to the bound character must not be refused here on the raw
        string alone: "Rill" and "rill" are the same person. Hyphens fold to spaces
        for the same reason the engine's resolver folds them. An empty argument
        never matches.
        """
        left = str(supplied or "").strip().casefold().replace("-", " ")
        right = str(bound or "").strip().casefold().replace("-", " ")
        return bool(left) and left == right

    @classmethod
    def _targets_character(cls, tool_name: str, arguments: dict, character_id: str) -> bool:
        if tool_name == "attribute_test":
            return cls._same_actor(arguments.get("character_id"), character_id)
        if tool_name == "combat_defend":
            return cls._same_actor(arguments.get("defender_id"), character_id)
        if tool_name == "group_test":
            participants = arguments.get("character_ids")
            return isinstance(participants, list) and any(
                cls._same_actor(entry, character_id) for entry in participants
            )
        return False

    def unsatisfied_hazard_actors(self) -> tuple[str, ...]:
        """Actors whose confirmed violent action has not yet rolled a resolving mechanic.
        """
        return tuple(
            hazard.character_id
            for hazard in self._hazards
            if hazard.character_id not in self._hazards_satisfied
        )

    def satisfy_hazard(self, actor_id: str) -> None:
        """Mark one confirmed action resolved by engine-performed combat steps.

        Only ``NarratorEngine``'s combat recovery calls this, and only after it has
        itself spent the actor's whole combat turn engaging -- every remaining action
        went to real, audited ``combat_move`` calls and the attack roll is therefore
        impossible until the actor's next turn. That is the same rules-made-it-
        impossible deferral ``record_tool_result`` already grants a ``combat_start``
        whose initiative gave the opposition the first turn.
        """
        self._hazards_satisfied.add(actor_id)

    def unused_defence_directives(self) -> tuple[CombatDefendDirective, ...]:
        """Bound dodge/parry directives no ``combat_defend`` call has matched yet.
        """
        return tuple(
            item
            for index, item in enumerate(self._directives)
            if index not in self._used and isinstance(item, CombatDefendDirective)
        )

    def unused_error(self) -> str:
        required = [
            index
            for index, item in enumerate(self._directives)
            if not isinstance(item, NoTestDirective)
        ]
        if any(index not in self._used for index in required):
            return "the selected decision's required mechanic was not used"
        if self._social_request is not None and self._social_outcome is None:
            return "the bound social test was not used"
        if any(
            hazard.character_id not in self._hazards_satisfied for hazard in self._hazards
        ):
            return "the confirmed action rolled no resolving mechanic"
        return ""

    def _hazard_tool_matches(self, tool_name: str, arguments: dict, actor_id: str) -> bool:
        """Whether one executed tool resolves a confirmed violent action for the actor.

        ``grant_runic_weapon`` counts for its own actor: a confirmed "takes up the
        black blade" resolved through the granting tool rolled real, audited dice
        (the weapon's 2d6 INT and its d20 session test) -- see the same finding on
        ``_directive_matches``.
        """
        if tool_name == "combat_attack":
            return self._same_actor(arguments.get("attacker_id"), actor_id)
        if tool_name == "grant_runic_weapon":
            return self._same_actor(arguments.get("character_id"), actor_id)
        if tool_name == "attribute_test":
            return self._same_actor(arguments.get("character_id"), actor_id)
        if tool_name == "group_test":
            participants = arguments.get("character_ids")
            return isinstance(participants, list) and any(
                self._same_actor(entry, actor_id) for entry in participants
            )
        return False

    @property
    def social_outcome(self):
        """The actual attribute-test result, bound to this exact social request."""
        return self._social_outcome

    def record_tool_result(self, tool_name: str, arguments: dict, payload: dict) -> None:
        """Capture only a successful, bound attribute-test result after execution."""
        # A bound directive resolves only when its call actually returned ok. An
        # engine-refused bound attempt (the call executed and the rules refused it)
        # is recorded in ``_refused`` instead, unlocking the actor's other mechanics
        # in ``validate`` without ever counting as the resolution -- the deferral
        # that stops a run of refused defences from delivering a narrated hit no
        # dice rolled. A call ``validate`` cancelled never reaches this hook at all.
        for index, item in enumerate(self._directives):
            if index in self._used or isinstance(item, NoTestDirective):
                continue
            if self._directive_matches(index, item, tool_name, arguments):
                if payload.get("ok") is True:
                    self._used.add(index)
                else:
                    self._refused.add(index)
                break


        if self._hazards and payload.get("ok") is True:
            for hazard in self._hazards:
                if self._hazard_tool_matches(tool_name, arguments, hazard.character_id):
                    self._hazards_satisfied.add(hazard.character_id)
                elif (
                    tool_name == "combat_start"
                    and isinstance(arguments.get("pc_ids"), list)
                    and any(
                        self._same_actor(entry, hazard.character_id)
                        for entry in arguments["pc_ids"]
                    )
                    and not self._same_actor(
                        payload.get("active_actor"), hazard.character_id
                    )
                    and payload.get("active_actor") is not None
                ):
                    self._hazards_satisfied.add(hazard.character_id)
        request = self._social_request
        if (
            request is None
            or tool_name != "attribute_test"
            or not self._social_used
            or self._social_outcome is not None
        ):
            return
        if payload.get("ok") is not True:
            return
        outcome = str(payload.get("outcome", ""))
        if outcome not in {"success", "critical_success", "failure", "critical_failure"}:
            return
        outcome = self._social_test_guard.resolve(
            request,
            actor_id=str(arguments.get("character_id", "")),
            attribute=str(arguments.get("attribute", "")),
            action_fingerprint=request.action_fingerprint,
            success=outcome in {"success", "critical_success"},
        )
        if outcome is not None:
            self._social_outcome = outcome

    def validate_social_test(self, *, actor_id: str, attribute: str, action_fingerprint: str) -> str | None:
        """Gate a social test by actor, exact attribute, and continuation fingerprint."""
        request = self._social_request
        if request is None:
            return "no bound social test exists"
        if self._social_used:
            return "the bound social test already resolved"
        if action_fingerprint != request.action_fingerprint:
            return "the bound social test belongs to a different action"
        selected = self._social_substitution
        allowed = (selected,) if selected else request.allowed_attributes
        if actor_id != request.actor_id or attribute not in allowed:
            return "the bound social test requires a different actor or attribute"
        self._social_used = True
        return None

    def _validate_social_ability_arguments(self, arguments: dict) -> str | None:
        """Permit only one approved, actor-bound substitution before the bound test."""
        request = self._social_request
        if request is None:
            return None
        if self._social_used or self._social_substitution:
            return "the bound social test already resolved"
        actor_id = str(arguments.get("character_id", ""))
        ability_id = str(arguments.get("ability_id", ""))
        mode = str(arguments.get("mode", "use"))
        substitutions = dict(getattr(request, "ability_substitutions", ()))
        attribute = substitutions.get(ability_id)
        if actor_id != request.actor_id or mode != "use" or attribute is None:
            return "the selected ability substitution is unavailable"
        authorizer = getattr(request, "ability_authorizer", None)
        if not callable(authorizer) or not authorizer(actor_id, ability_id):
            return "the selected ability substitution is unavailable"
        self._social_substitution = attribute
        return None
