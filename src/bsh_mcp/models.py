"""Typed campaign file models.

Every authoritative campaign file validates through one of these models before
it is written and after it is read. A file that fails validation never replaces
a valid file on disk.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

ATTRIBUTE_NAMES: tuple[str, ...] = ("STR", "DEX", "CON", "INT", "WIS", "CHA")
ORIGINS: tuple[str, ...] = ("barbarian", "civilised", "decadent")

#: Every field below typed ``ArmourCategory`` shares this one vocabulary; before this
#: alias existed, each field re-spelled the same four-value ``Literal`` independently.
ArmourCategory = Literal["none", "light", "medium", "heavy"]
ARMOUR_CATEGORIES: tuple[ArmourCategory, ...] = ("none", "light", "medium", "heavy")
#: Armour protection subtracted from incoming damage, minimum zero. Lives here rather
#: than in ``rules.py`` (which imports it) because ``models`` sits below ``rules`` in
#: this package's layering (dice/rules pure -> models -> store -> service -> server),
#: so this is the only direction a shared constant can flow without a cycle.
ARMOUR_PROTECTION: dict[ArmourCategory, int] = {"none": 0, "light": 1, "medium": 2, "heavy": 3}

RangeBand = Literal["close", "nearby", "far_away", "distant"]
RANGE_BANDS: tuple[RangeBand, ...] = ("close", "nearby", "far_away", "distant")

#: Closed state vocabulary for a typed barrier or scene object (``SceneObject``).
#: Free-string facts drifted and contradicted themselves in play (a locked door
#: narrated open, a bolt thrown from one side swinging free from the other), so this
#: field follows the same closed-enum convention every other state-like field in this
#: module already uses (``Character.armour``, ``Character.status``, ``NPC.status``,
#: ``Condition.scope``) rather than staying a free string that can drift the same way.
ObjectState = Literal["locked", "unlocked", "barred", "open", "closed"]
OBJECT_STATES: tuple[ObjectState, ...] = ("locked", "unlocked", "barred", "open", "closed")

#: Identifier grammar for every character, NPC, location, clock, and resource.
#: The grammar rejects path separators, ``..``, absolute paths, and null bytes,
#: so an identifier can never escape the campaign directory.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

#: Canonical ``dice.py`` die notation (lowercase ``dN``) or the ``"depleted"``
#: sentinel a Usage Die reaches when stepped past ``d4``. Deliberately not a
#: ``Literal`` of the standard set: ``rules/*.json`` uses dice outside it (``d3``,
#: ``d100``), so this only catches malformed notation (stray case, a missing digit),
#: not an unfamiliar-but-valid die. Not imported from ``dice.py`` -- see
#: ``UsageResource._backfill_maximum`` on why this module stays a leaf with respect
#: to the dice vocabulary.
DieNotation = Annotated[str, Field(pattern=r"^(d[0-9]+|depleted)$")]
#: As :data:`DieNotation`, but also accepting the empty string
#: ``UsageResource.maximum`` uses for "no known maximum" (a legacy resource already
#: depleted at first load, before this field existed to record one).
OptionalDieNotation = Annotated[str, Field(pattern=r"^(d[0-9]+|depleted)?$")]

Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
#: As :data:`Identifier`, but also accepting the empty string a field uses for
#: "not yet set" or "not applicable" (an NPC or scene with no location placed yet, a
#: condition with no bound effect or ward target).
OptionalIdentifier = Annotated[str, Field(pattern=r"^([a-z0-9][a-z0-9_-]{0,63})?$")]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    """Convert a display name into a safe identifier."""
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:64] or "unnamed"


def is_valid_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Attributes(StrictModel):
    STR: int = Field(ge=1, le=20)
    DEX: int = Field(ge=1, le=20)
    CON: int = Field(ge=1, le=20)
    INT: int = Field(ge=1, le=20)
    WIS: int = Field(ge=1, le=20)
    CHA: int = Field(ge=1, le=20)

    def get(self, name: str) -> int:
        key = name.strip().upper()
        if key not in ATTRIBUTE_NAMES:
            raise KeyError(key)
        return int(getattr(self, key))

    def adjust(self, name: str, delta: int) -> None:
        key = name.strip().upper()
        if key not in ATTRIBUTE_NAMES:
            raise KeyError(key)
        setattr(self, key, max(1, min(20, getattr(self, key) + delta)))


class UsageResource(StrictModel):
    """A named consumable tracked by a Usage Die (rations, arrows, oil, torches)."""

    id: Identifier
    name: str
    die: DieNotation = "d6"
    note: str = ""
    #: The die this resource counts as "full" -- what a Resourceful spend restores
    #: it to. Backfilled below rather than left for a caller to set: a resource
    #: constructed today backfills to its own starting die, and a campaign file
    #: written before this field existed adopts its current die as the recorded
    #: maximum, the best evidence available since the file predates any mechanic
    #: that could have raised the die. Left empty (deliberately unguessed) for a
    #: legacy resource already depleted at first load; nothing records what it
    #: once was, so a replenish against it is refused rather than fabricated.
    maximum: OptionalDieNotation = ""

    @model_validator(mode="after")
    def _backfill_maximum(self) -> UsageResource:
        # Import locally: `dice` is a leaf module with no project imports, so this
        # does not create a cycle, but importing it at module scope would make
        # `models` a second entry point into the dice vocabulary for no benefit.
        from .dice import DEPLETED

        if not self.maximum and self.die != DEPLETED:
            self.maximum = self.die
        return self


class Condition(StrictModel):
    """A mechanical condition that the rules engine reads, not decorative text."""

    id: Identifier
    label: str
    effect: Literal[
        "disadvantage_all",
        "disadvantage_dex",
        "disadvantage_damage",
        "none",
    ] = "none"
    scope: Literal["session", "until_long_rest", "permanent", "scene"] = "session"
    source: str = ""
    #: The effect-registry id an activated stance carries, so combat hooks find it.
    effect_id: OptionalIdentifier = ""
    #: The ally a targeted stance (e.g. Bodyguard) wards, empty otherwise.
    target_id: OptionalIdentifier = ""


class RunicWeapon(StrictModel):
    """A sentient runic weapon a character wields (rules/subsystems.json runic_weapons).

    The weapon fixes its damage to one of the wielder's attributes by its personality,
    carries its own INT score, and, when its start-of-session INT test succeeds, kills a
    wielder who becomes helpless during that session. ``kills_helpless`` records that
    per-session verdict, which ``session_close`` re-rolls each session.
    """

    name: str
    personality: Literal[
        "brutal", "vicious", "patient", "cunning", "judgemental", "prideful"
    ]
    weapon_int: int = Field(ge=1, le=20)
    kills_helpless: bool = False


class Character(StrictModel):
    id: Identifier
    name: str
    discord_user_id: str = ""
    origin: Literal["barbarian", "civilised", "decadent"]
    backgrounds: list[str] = Field(default_factory=list)
    level: int = Field(default=1, ge=1, le=10)
    stories: int = Field(default=0, ge=0)
    attributes: Attributes
    hp: int = Field(ge=0)
    hp_max: int = Field(ge=1)
    doom_die: DieNotation = "d6"
    doom_max: DieNotation = "d6"
    weapon_damage: DieNotation = "d6"
    unarmed_damage: DieNotation = "d4"
    armour: ArmourCategory = "none"
    shield: bool = False
    weapons: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    coins: int = Field(default=0, ge=0)
    resources: list[UsageResource] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    scars: list[str] = Field(default_factory=list)
    gifts: list[str] = Field(default_factory=list)
    spells: list[str] = Field(default_factory=list)
    #: Remaining uses of each bounded-resource ability, keyed by effect id.
    pools: dict[str, int] = Field(default_factory=dict)
    #: The in-game day (``CampaignState.day``) this character's ``reset: "day"``
    #: pools were last refreshed for. ``None`` means never refreshed. Compared
    #: lazily against the current day at the top of ``use_ability``'s ``resource``
    #: branch, mirroring how ``faerie_uses_day`` gates the faerie daily counters
    #: below -- no tool sweeps this eagerly on a day rollover.
    pools_day: int | None = None
    #: Bound subsystem powers, e.g. "demon:abyss" or "spirit:fire_spirit".
    powers: list[str] = Field(default_factory=list)
    #: Faerie ties gated to once per day track their day-count here; the map resets
    #: when ``faerie_uses_day`` no longer matches the current in-game day.
    faerie_uses_day: int | None = None
    faerie_uses_today: dict[str, int] = Field(default_factory=dict)
    #: Long rests still required before each recovering faerie tie may be invoked again.
    faerie_recovery_rests: dict[str, int] = Field(default_factory=dict)
    #: Current Usage Die of each built non-single-use twisted-science marvel, keyed by
    #: power id (e.g. "marvel:firelance" -> "d6"); a depleted marvel drops from the map.
    marvel_usage: dict[str, str] = Field(default_factory=dict)
    #: The sentient runic weapon the character wields, or None. A rare in-play discovery.
    runic_weapon: RunicWeapon | None = None
    #: Standing per-character declarations set ahead of the moment they matter (an
    #: "intent"-activation use_ability call), keyed by the granting effect id --
    #: e.g. herbalist_stock -> "poison". A generic mailbox: any future background
    #: can reuse it for its own declared choice. Consumed and cleared by whatever
    #: hook resolves it (Herbalist's is rest's own hook), not read directly by name
    #: anywhere outside the effect registry.
    declared_choices: dict[str, str] = Field(default_factory=dict)
    #: Discrete, single-use consumable counts by type id (e.g. "healing_balm" ->
    #: 3) -- a plain count, not a Usage Die: using one just decrements it, there is
    #: no roll-to-survive step. Generic storage any future background could reuse;
    #: Herbalist is the first.
    doses: dict[str, int] = Field(default_factory=dict)
    status: Literal["ok", "helpless", "dead"] = "ok"
    last_short_rest_day: int | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    notes: str = ""

    @field_validator("hp")
    @classmethod
    def _hp_non_negative(cls, value: int) -> int:
        return max(0, value)

    @field_validator("doses")
    @classmethod
    def _doses_non_negative(cls, values: dict[str, int]) -> dict[str, int]:
        if any(isinstance(value, bool) or value < 0 for value in values.values()):
            raise ValueError("dose counts cannot be negative")
        return values

    def has_effect(self, effect: str) -> bool:
        return any(condition.effect == effect for condition in self.conditions)

    def armour_protection(self) -> int:
        return ARMOUR_PROTECTION[self.armour]


class NPC(StrictModel):
    id: Identifier
    name: str
    level: int = Field(ge=1, le=10)
    hp: int = Field(ge=0)
    hp_max: int = Field(ge=1)
    damage: int = Field(ge=0)
    armour: ArmourCategory = "none"
    motive: str = ""
    actions: list[str] = Field(default_factory=list)
    location_id: OptionalIdentifier = ""
    status: Literal["alive", "dead", "fled", "captured"] = "alive"
    flags: list[str] = Field(default_factory=list)
    notes: str = ""


class CombatActor(StrictModel):
    id: Identifier
    side: Literal["pc", "npc"]
    bucket: Literal["before", "after"] = "before"
    actions_max: int = Field(default=2, ge=0, le=3)
    actions_used: int = Field(default=0, ge=0)
    actions_taken: list[str] = Field(default_factory=list)
    turn_open: bool = False
    first_turn_actions: int = Field(default=2, ge=1, le=3)
    #: Ranged attacks this actor has made in the current fight, for Hunter's
    #: first-arrow auto-hit. Reset only when combat_start rebuilds the actor,
    #: not per turn.
    ranged_attacks_made: int = Field(default=0, ge=0)
    #: Whether this actor has itself acted in the current fight: its own turn
    #: opened, or (as an NPC) it named itself the attacker in a combat_defend.
    #: Being on the receiving end of an attack does NOT set this -- an NPC
    #: struck once by an Assassin has not thereby "reacted", so a second
    #: Assassin can still find it unaware. Reset only when combat_start
    #: rebuilds the actor, not per turn.
    reacted: bool = False
    #: Whether this actor (as an attacker) has already spent their once-per-fight
    #: Assassin unaware-strike this fight. Scoped to the attacker, not the target,
    #: so two Assassins can each land their own strike on the same still-unreacted
    #: target. Reset only when combat_start rebuilds the actor, not per turn.
    unaware_strike_used: bool = False


class CombatState(StrictModel):
    active: bool = False
    round: int = Field(default=0, ge=0)
    order: list[Identifier] = Field(default_factory=list)
    active_actor: Identifier | None = None
    actors: dict[str, CombatActor] = Field(default_factory=dict)
    ranges: dict[str, RangeBand] = Field(
        default_factory=dict
    )
    reason: str = ""
    started_event: int | None = None
    # Optional narrator-declared stakes for the whole fight, recorded at
    # combat_start and realized when combat ends. Flat strings by design:
    # the local narrator model mangles nested argument shapes.
    stakes_success: str = ""
    stakes_failure: str = ""
    stakes_hidden: str = ""


class StakesDeclaration(StrictModel):
    """Narrator-declared stakes recorded before the dice roll.

    The narrator already must know what a roll risks before calling for it;
    these fields make that declaration a recorded input. The server stores
    the text verbatim and tags which branch the dice realized. It never
    invents fiction.
    """

    success: str = ""
    failure: str = ""
    hidden: str = ""

    def is_empty(self) -> bool:
        return not (self.success or self.failure or self.hidden)


class FictionDebt(StrictModel):
    """One unratified durable-fiction entry awaiting scene_commit.

    Entries are created inside the same transaction as the mechanics they
    describe, so even a narrator that never calls scene_commit leaves the
    realized stakes in canon. scene_commit ratifies and clears them.
    """

    seq: int = Field(ge=1)
    tool: str
    actor_id: str = ""
    reason: str = ""
    outcome: str = ""
    stakes: StakesDeclaration = Field(default_factory=StakesDeclaration)
    realized: Literal["success", "failure", "none"] = "none"
    at: str = Field(default_factory=utc_now)

    def realized_public_text(self) -> str:
        """The realized public-facing stake, or an empty string."""
        if self.realized == "success":
            return self.stakes.success
        if self.realized == "failure":
            return self.stakes.failure
        return ""


class PendingRuling(StrictModel):
    """A mechanical obligation awaiting a fictional choice.

    Created inside the invocation's own transaction, kept separate from
    ``fiction_debt`` because ``scene_commit`` and ``ledger_settle`` clear that list
    wholesale and would erase a mechanical obligation. The narrator engine's
    adjudicate step resolves it through ``ability_apply_ruling``; the delivery gate
    withholds the turn while any remain. Defaults empty, so older campaigns validate.
    """

    id: str
    actor_id: str = ""
    kind: Literal["", "demon_revenge_steal", "demon_revenge_destroy_ally_weapon"] = ""
    question: str = ""
    options: list[str] = Field(default_factory=list)
    default_strategy: Literal["first", "random"] = "random"
    at: str = Field(default_factory=utc_now)


class Clock(StrictModel):
    id: Identifier
    name: str
    segments: int = Field(ge=1, le=12)
    filled: int = Field(default=0, ge=0)
    note: str = ""

    @field_validator("filled")
    @classmethod
    def _cap(cls, value: int) -> int:
        return max(0, value)


TRAVERSAL_CLOCK_PREFIX = "travel-"


def traversal_clock_id(location_a: str, location_b: str) -> str:
    """Direction-normalized clock id for a traversal between two locations.
    """
    first, second = sorted((location_a.strip(), location_b.strip()))
    return f"{TRAVERSAL_CLOCK_PREFIX}{first}-{second}"


class SceneObject(StrictModel):
    """A barrier (a door, gate, bolt) or a stateful prop (a chest, a lever).
    """

    id: Identifier
    state: ObjectState
    note: str = ""
    traversal_segments: int | None = Field(default=None, ge=1, le=12)


#: The knowledge scopes a durable statement may carry. This is a closed set:
#: ``public`` is party knowledge; ``world_hidden`` is game-master-only;
#: ``party_secret`` (B2) is knowledge the party holds that the world's people do
#: not -- the arm-3 bucket semantics, typed.
StatementScope = Literal["public", "world_hidden", "party_secret"]


class Statement(StrictModel):
    """One durable prose fact with the metadata the prose cannot carry.
    """

    seq: int = Field(default=0, ge=0)
    text: str = Field(min_length=1)
    scope: StatementScope = "public"
    refs: tuple[Identifier, ...] = ()
    source: Literal["model", "engine", "authored"] = "model"
    superseded_at: int = Field(default=0, ge=0)


def live_statements(statements: list[Statement], scopes: tuple[str, ...] | None = None) -> list[Statement]:
    """The statements a view renders: not superseded, filtered by scope, in order.

    The one derivation every fact-reading view shares: the scene render,
    the compatibility properties below, and any
    future scoped view are this function plus formatting.
    """
    return [
        statement
        for statement in statements
        if not statement.superseded_at and (scopes is None or statement.scope in scopes)
    ]


class ScenePerson(StrictModel):
    """A person present in the fiction who holds no mechanical record yet.
    """

    id: Identifier
    name: str
    role: str = ""
    source: Literal["authored", "model"] = "model"
    first_event: int = Field(default=0, ge=0)
    last_event: int = Field(default=0, ge=0)


class Scene(StrictModel):
    location_id: OptionalIdentifier = ""
    title: str = ""
    summary: str = ""


    statements: list[Statement] = Field(default_factory=list)
    exits: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)
    present_npcs: list[str] = Field(default_factory=list)
    # Typed state for a barrier or object, keyed by id -- an exit's id when the
    # object gates passage, or any other scene-local id otherwise. scene_commit's
    # object_updates upserts entries here instead of the visible_facts free-text
    # list, so a barrier's state cannot drift into self-contradiction the way prose
    # could. Defaults empty, so campaigns written before this field validate
    # unchanged.
    objects: dict[Identifier, SceneObject] = Field(default_factory=dict)
    #: Open-vocabulary descriptors of the current scene ("natural", "urban",
    #: whatever a future mechanic needs), set through scene_commit like any other
    #: durable scene fact. Ambient state a hook may consult, never a parameter on
    #: the tool that consults it -- e.g. Herbalist's rest-preparation hook reads
    #: this generically rather than rest() taking a background-specific flag.
    environment_tags: list[str] = Field(default_factory=list)
    #: Persons present who hold no mechanical record (``ScenePerson``), keyed by id.
    #: Additive and defaulting empty, the ``objects`` precedent, so a campaign written
    #: before this field validates unchanged.
    persons: dict[Identifier, ScenePerson] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_fact_lists(cls, data):
        """Lift a pre-migration record's fact lists into statements.

        Accepts every historical shape forever: ``visible_facts`` becomes ``public``
        and ``hidden_facts`` becomes ``world_hidden``, order preserved, ``seq`` 0
        (the committing event is unrecorded in the old shape). Constructor keywords
        pass through the same lift, so ``Scene(visible_facts=[...])`` keeps working.
        """
        if not isinstance(data, dict):
            return data
        lifted: list = []
        for key, scope in (("visible_facts", "public"), ("hidden_facts", "world_hidden")):
            for text in data.pop(key, None) or []:
                if isinstance(text, str) and text.strip():
                    lifted.append({"seq": 0, "text": text, "scope": scope, "source": "model"})
        if lifted:
            data["statements"] = list(data.get("statements") or []) + lifted
        return data

    @property
    def visible_facts(self) -> list[str]:
        """What the party knows, in record order: the historical reader's shape."""
        return [s.text for s in live_statements(self.statements, ("public", "party_secret"))]

    @property
    def hidden_facts(self) -> list[str]:
        """Game-master-only facts, in record order: the historical reader's shape."""
        return [s.text for s in live_statements(self.statements, ("world_hidden",))]


class MerchantStock(StrictModel):
    """Finite seller state read only by the engine-only purchase path."""

    version: int = Field(default=0, ge=0)
    items: dict[Identifier, int] = Field(default_factory=dict)
    prices: dict[Identifier, int] = Field(default_factory=dict)

    @field_validator("items")
    @classmethod
    def _nonnegative_items(cls, values: dict[str, int]) -> dict[str, int]:
        if any(isinstance(value, bool) or value < 0 for value in values.values()):
            raise ValueError("merchant stock cannot be negative")
        return values

    @field_validator("prices")
    @classmethod
    def _nonnegative_prices(cls, values: dict[str, int]) -> dict[str, int]:
        if any(isinstance(value, bool) or value < 0 for value in values.values()):
            raise ValueError("merchant prices cannot be negative")
        return values


class PurchaseReceipt(StrictModel):
    """A durable idempotency marker. It holds no dialogue or account subject."""

    trade_fingerprint: str = Field(min_length=64, max_length=64)
    event_seq: int = Field(ge=1)


class EnginePurchaseRequest(StrictModel):
    """Authenticated terms supplied only after the service opens confirmation."""

    decision_key: str = Field(min_length=16, max_length=256)
    action_fingerprint: str = Field(min_length=64, max_length=64)
    trade_fingerprint: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=64, max_length=64)
    buyer_id: Identifier
    seller_id: Identifier
    item_id: Identifier
    quantity: int = Field(ge=1, le=99)
    price_copper: int = Field(ge=0, le=100000)
    stock_version: int | None = Field(default=None, ge=0)


class CampaignState(StrictModel):
    schema_version: int = SCHEMA_VERSION
    session: int = Field(default=1, ge=1)
    in_game_minutes: int = Field(default=0, ge=0)
    event_seq: int = Field(default=0, ge=0)
    scene: Scene = Field(default_factory=Scene)
    clocks: list[Clock] = Field(default_factory=list)
    npcs: dict[str, NPC] = Field(default_factory=dict)
    combat: CombatState = Field(default_factory=CombatState)
    # Unratified durable fiction. Mechanics tools append entries inside their
    # own transactions; scene_commit ratifies and clears. Defaults empty, so
    # campaigns written before this field validate unchanged.
    fiction_debt: list[FictionDebt] = Field(default_factory=list)
    # Mechanical obligations awaiting a fictional choice (e.g. which possession a
    # demon steals). Kept separate from fiction_debt so a scene_commit cannot erase
    # them. The engine's adjudicate step drains them; the delivery gate withholds
    # while any remain. Defaults empty, so older campaigns validate unchanged.
    pending_rulings: list[PendingRuling] = Field(default_factory=list)
    merchant_stocks: dict[Identifier, MerchantStock] = Field(default_factory=dict)
    purchase_receipts: dict[str, PurchaseReceipt] = Field(default_factory=dict)
    # A social substitution is spent by the trusted narrator before its bound roll.
    # The following MCP ``use_ability`` call consumes this acknowledgement instead of
    # charging a second use.  The key contains only canonical identifiers.
    social_ability_receipts: dict[str, str] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now)

    @property
    def day(self) -> int:
        return self.in_game_minutes // 1440


class PlayerLink(StrictModel):
    discord_user_id: str
    character_id: Identifier
    display_name: str = ""


class PlayersFile(StrictModel):
    players: list[PlayerLink] = Field(default_factory=list)

    def by_discord_id(self, discord_user_id: str) -> PlayerLink | None:
        for link in self.players:
            if link.discord_user_id == discord_user_id:
                return link
        return None

    def by_character_id(self, character_id: str) -> PlayerLink | None:
        for link in self.players:
            if link.character_id == character_id:
                return link
        return None


class Manifest(StrictModel):
    title: str = "Untitled Campaign"
    rules_version: str = "Black Sword Hack UCE SRD v1.0.2"
    world_root: str = "world"
    session: int = Field(default=1, ge=1)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    notes: str = ""
