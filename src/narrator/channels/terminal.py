"""Terminal transport for one local player.

The adapter only translates terminal traffic into channel turns and delivered text.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator
from time import monotonic

from pydantic import ValidationError

from narrator.channels.base import (  # noqa: F401
    LOCAL_COMMANDS,
    RECOVERY_CONTROLS,
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionCollectionCancelled,
    DecisionDeliveryReceipt,
    InboundTurn,
    LanguageControl,
    MentionCandidate,
    MentionDirectoryControl,
    ThinkingLevelControl,
)
from narrator.channels.diagnostics import DiagnosticRecorder
from narrator.channels.terminal_ui import PromptToolkitTerminalUI, SigilCompleter, _TerminalUI
from narrator.decisions import DecisionSubmission, DecisionView

#: The shapes a numbered-prompt answer takes, recognized here so a stale one never
#: narrates. ``collect_decision`` accepts a bare number and the ``N. text`` and ``N) text``
#: forms. This pattern is a strict superset of that grammar: it also matches a bare ``3.``
#: and ``3)``, which the collector rejects. Every divergence is still a numbered-answer
#: shape rather than an ordinary declaration, so the extra reach runs in the safe
#: direction. An audit confirmed the superset relation rather than an equality.
#:
#: Trailing text requires the punctuation. A bare number followed by words is ordinary
#: narration: "12 gold to the merchant" names a price, not an option, so it passes through.
_STALE_DECISION_ANSWER = re.compile(r"\d+[.)]?|\d+[.)]\s+\S.*", re.DOTALL)


#: The description the worked example carries. It names an approach rather than a
#: mechanic, so a player reads it as illustration and not as a listed choice.
# The worked example a custom-option prompt shows. Lives in the catalog so a table
# reading in another language gets an example in that language.


def _catalog(catalog=None):
    """This terminal's player-facing text, English when no catalog is threaded.

    The adapter is constructed by ``scripts/narrator_serve.py``, which passes the
    configured language's catalog. The English default keeps the module importable and
    keeps a renderer callable from a test that builds no config -- and it is the same
    ``locale/en/terminal.yaml`` production loads, not a second copy of the wording.
    """
    if catalog is not None:
        return catalog
    from pathlib import Path

    from narrator.locale import load

    return load("en", Path(__file__).resolve().parents[3] / "locale", domain="terminal")


def _custom_option_example(options, catalog=None) -> str:
    """The exact line a player may type to answer with a custom approach, or "".

    This is the one definition of the shape this channel teaches. The guidance quotes it,
    and ``test_the_stated_grammar_is_the_one_collect_decision_accepts`` types it through
    ``collect_decision``. A single definition therefore feeds both the instruction and its
    proof, so changing the taught shape fails that test unless the collector accepts it.

    An audit forced that arrangement. The earlier test typed a hard-coded literal, so it
    observed the guidance not at all. Replacing the guidance with a colon form the
    collector rejects left the test passing while the prompt taught a refused shape.
    """
    for index, option in enumerate(options, start=1):
        if option.custom:
            example = _catalog(catalog).text("decision.custom_example")
            return f"{index}. {example}"
    return ""


def _custom_option_guidance(options, catalog=None) -> list[str]:
    """State the inline grammar a custom option requires, or state nothing.

    ``collect_decision`` accepts a custom approach only as the option number, then ``.``
    or ``)``, then the description. Until this function existed the player learned that
    grammar by failing. Selecting the option by its bare number drew a correction, and
    neither the prompt nor the decision help had mentioned the shape.

    The example carries the option's own number, so a player copies a line that works
    rather than translating a rule. A view holding no custom option produces no lines,
    which leaves every other prompt byte-identical.
    """
    example = _custom_option_example(options)
    if not example:
        return []
    # The sentence quotes the example's own opening token rather than rebuilding it, so
    # the instruction and the worked example cannot state different separators. An audit
    # noted that a sentence restating "." while the example moved to ")" would leave the
    # binding test green and still misinstruct a player who read the sentence.
    marker = example.split(maxsplit=1)[0]
    for option in options:
        if option.custom:
            return [
                f"For {option.label!r}, type {marker} followed by your description. "
                f"Example: {example}",
                "Plain text on its own works too: describe what you do and it is taken "
                "as that approach.",
            ]
    return []


_INDEX_ANSWER = re.compile(r"(\d+)[.)]?")

#: An index, a separator, then a description -- the shape a custom approach needs, and
#: the shape ``_custom_option_example`` teaches.
_INDEX_WITH_TEXT = re.compile(r"(\d+)[.)]\s+(.+)", re.DOTALL)


def _select_option(options, choice: str) -> tuple[str, str]:
    """Map one typed answer onto ``(option id, inline description)``, or ``(\"\", \"\")``.

    The rules, in order, and why the order matters:

    1. An index, bare or with the separator the prompt itself prints. An index that
       names no listed option refuses here and never falls through to rule 4: a player
       who typed \"9\" at a four-option list meant an option, not a description, and
       silently recording \"9\" as their approach would be worse than refusing it.
    2. An index, separator, and description: the custom-approach shape.
    3. An exact option label, case-folded. Labels are what the prompt prints, so this is
       the player answering with what they were shown. The listed identifier still
       matches too, exactly as before.
    4. Free text, but only when the view carries a custom option -- which is to say only
       when \"describe your own approach\" is genuinely on offer. A leading slash is
       excluded so a mistyped command draws a refusal rather than becoming an approach,
       and a leading colon stays excluded with it: the command marker moved from ``:``
       to ``/``, and a player reaching for the old spelling still meant a command.

    A view with no custom option -- a confirmation, an approach list -- reaches rule 4
    and refuses, which is deliberate. \"I confirm nothing\" must not confirm, and the only
    honest reading of free text against a fixed yes/no gate is that it is neither.
    """
    stripped = choice.strip()
    if not stripped:
        return ("", "")
    listed = tuple(options)

    def by_index(raw: str) -> str:
        index = int(raw) - 1
        return listed[index].id if 0 <= index < len(listed) else ""

    index_only = _INDEX_ANSWER.fullmatch(stripped)
    if index_only:
        return (by_index(index_only.group(1)), "")
    inline = _INDEX_WITH_TEXT.fullmatch(stripped)
    if inline:
        selected = by_index(inline.group(1))
        return (selected, inline.group(2).strip() if selected else "")
    folded = stripped.casefold()
    for option in listed:
        if stripped == option.id or folded == option.label.strip().casefold():
            return (option.id, "")
    if stripped[0] in "/:":
        return ("", "")
    custom = next((option for option in listed if option.custom), None)
    if custom is not None:
        return (custom.id, stripped)
    return ("", "")


#: The "Thinking" indicator's animation, one frame per tick. A palindrome rather than
#: a one-directional cycle so the glyph pulses -- swelling from a plain asterisk to a
#: full burst and back -- reading as "still working" rather than "spinning," which
#: fits a narrator turn's actual shape better: it is prose generation and a handful of
#: tool calls, not a mechanical wait with a fixed rotation.
_THINKING_FRAMES: tuple[str, ...] = ("*", "✢", "✳", "✻", "✳", "✢")
_THINKING_FRAME_SECONDS = 0.15


def _format_status_line(
    snapshot, character_id: str, fallback_name: str = "", catalog=None
) -> tuple[str, str]:
    """Compose this terminal's own status line, split into a left and right half.

    Reads only ``snapshot["characters"][character_id]`` -- a status line is one
    player's own view, so a party-wide snapshot never leaks another character's HP or
    conditions onto this terminal. The left half is the character's own state (name,
    HP, Doom die, conditions); the right half is the scene -- location and day, or,
    once a fight is active, the round and whose turn it is. Combat context replaces
    scene/day there, because whose turn it is matters more than where the scene is
    until it ends. ``PromptToolkitTerminalUI`` right-justifies the second half against
    the first at render time, against the terminal's own current width, so this
    itself stays two plain, unpadded strings.

    An empty ``character_id`` is the session-zero state: no character exists yet, so
    the left half carries ``fallback_name`` (the player's own label) rather than a
    blank, and the scene half still renders.
    """
    cat = _catalog(catalog)
    character = snapshot.get("characters", {}).get(character_id) or {}
    left_parts = [str(character.get("name") or character_id or fallback_name or cat.text("status.no_character"))]

    hp, hp_max = character.get("hp"), character.get("hp_max")
    if hp is not None and hp_max is not None:
        life_status = character.get("status") or "ok"
        suffix = f" ({life_status})" if life_status != "ok" else ""
        left_parts.append(cat.text("status.hp", hp=hp, hp_max=hp_max) + suffix)

    doom_die = character.get("doom_die")
    if doom_die:
        left_parts.append(cat.text("status.doom", die=doom_die))

    conditions = character.get("conditions") or []
    if conditions:
        left_parts.append("⚠ " + ", ".join(conditions))

    right_parts = []
    if snapshot.get("combat_active"):
        combat_text = cat.text("status.combat")
        round_number = snapshot.get("combat_round")
        if round_number is not None:
            combat_text += cat.text("status.round", round=round_number)
        active_actor = snapshot.get("combat_active_actor")
        if active_actor:
            combat_text += ", " + cat.text("status.turn", name=active_actor)
        right_parts.append(combat_text)
    else:
        location = snapshot.get("scene_title") or snapshot.get("location_id")
        if location:
            right_parts.append(location)
        day = snapshot.get("day")
        if day is not None:
            right_parts.append(cat.text("status.day", day=day))

    return "  ".join(left_parts), "  ".join(right_parts)


#: The printed Black Sword Hack sheet's own attribute order, which is also
#: ``bsh_mcp.models.Attributes``'s field order -- restated rather than imported
#: because the narrator environment deliberately never imports the rules engine.
_ATTRIBUTE_ORDER = ("STR", "DEX", "CON", "INT", "WIS", "CHA")

#: Armour protection values, mirroring ``Character.armour_protection`` in the rules
#: engine (same no-import rule as ``_ATTRIBUTE_ORDER``). "none" is absent because a
#: sheet prints it bare rather than as "none (0)".
_ARMOUR_PROTECTION = {"light": 1, "medium": 2, "heavy": 3}


def _titleize(identifier: str) -> str:
    """Render a stored identifier ("street-urchin") as sheet text ("Street Urchin")."""
    return " ".join(part.capitalize() for part in re.split(r"[-_\s]+", identifier) if part)


def _format_character_sheet(snapshot, character_id: str, catalog=None) -> str:
    """Compose this terminal's own character sheet from the cached status snapshot.

    Reads only ``snapshot["characters"][character_id]`` -- the same one-character
    discipline ``_format_status_line`` keeps, so a party-wide snapshot never prints
    another character's sheet on this terminal. Returns "" when the snapshot holds no
    such character (session zero, or no snapshot yet), which the ``/character``
    command turns into its own notice.

    The layout follows the printed Black Sword Hack sheet: identity and advancement
    first, then the attribute row and vitals, then arms and belongings, then powers
    and afflictions, then notes. Sections a sheet may legitimately lack (gifts,
    spells, conditions, scars...) print only when present; the belongings a player
    always owns a claim to (weapons, equipment) print "none" explicitly, because for
    an inventory the absence is the information. Field labels use the ``**bold**``
    markup the terminal renderer already styles, and no line opens with a list
    marker, so nothing here re-renders as a bullet.
    """
    cat = _catalog(catalog)
    character = (snapshot.get("characters") or {}).get(character_id)
    if not isinstance(character, dict):
        return ""
    sheet = character.get("sheet")
    sheet = sheet if isinstance(sheet, dict) else {}

    def joined(key: str) -> str:
        return ", ".join(str(entry) for entry in sheet.get(key) or [])

    name = str(character.get("name") or character_id)
    origin = _titleize(str(sheet.get("origin") or ""))
    level = sheet.get("level")
    descriptor = "  ".join(
        part for part in (origin, cat.text("sheet.level", level=level) if level else "") if part
    )
    identity = [f"# {name} — {descriptor}" if descriptor else f"# {name}"]
    backgrounds = ", ".join(_titleize(entry) for entry in sheet.get("backgrounds") or [])
    if backgrounds:
        identity.append(f"{cat.text('sheet.backgrounds')} {backgrounds}")
    stories = sheet.get("stories")
    if stories is not None and level:
        # A character advances after collecting Stories equal to their current level
        # (rules/advancement.json); level 10 is the cap, so no target prints there.
        target = f"/{level}" if isinstance(level, int) and level < 10 else ""
        identity.append(f"{cat.text('sheet.stories')} {stories}{target}")

    attributes = sheet.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    vitals = []
    attribute_row = "  ".join(
        f"{name_} {attributes[name_]}" for name_ in _ATTRIBUTE_ORDER if name_ in attributes
    )
    if attribute_row:
        vitals.append(attribute_row)
    vitals_parts = []
    hp, hp_max = character.get("hp"), character.get("hp_max")
    if hp is not None and hp_max is not None:
        life_status = character.get("status") or "ok"
        suffix = f" ({life_status})" if life_status != "ok" else ""
        vitals_parts.append(f"{cat.text('sheet.hp')} {hp}/{hp_max}{suffix}")
    doom_die = character.get("doom_die")
    if doom_die:
        doom_max = str(sheet.get("doom_max") or "")
        diminished = f" (max {doom_max})" if doom_max and doom_max != doom_die else ""
        vitals_parts.append(f"{cat.text('sheet.doom')} {doom_die}{diminished}")
    armour = str(sheet.get("armour") or "")
    if armour:
        protection = _ARMOUR_PROTECTION.get(armour)
        armour_text = f"{armour} ({protection})" if protection else armour
        if sheet.get("shield"):
            armour_text += cat.text("sheet.shield")
        vitals_parts.append(f"{cat.text('sheet.armour')} {armour_text}")
    if vitals_parts:
        vitals.append("  ".join(vitals_parts))

    belongings = []
    weapon_damage = str(sheet.get("weapon_damage") or "")
    unarmed = str(sheet.get("unarmed_damage") or "")
    damage_parts = [part for part in (
        f"damage {weapon_damage}" if weapon_damage else "",
        f"unarmed {unarmed}" if unarmed else "",
    ) if part]
    damage = f" — {', '.join(damage_parts)}" if damage_parts else ""
    belongings.append(f"{cat.text('sheet.weapons')} {joined('weapons') or cat.text('sheet.none')}{damage}")
    belongings.append(f"{cat.text('sheet.equipment')} {joined('equipment') or cat.text('sheet.none')}")
    coins = sheet.get("coins")
    if coins is not None:
        belongings.append(f"{cat.text('sheet.coins')} {coins}")
    if sheet.get("languages"):
        belongings.append(f"{cat.text('sheet.languages')} {joined('languages')}")
    resources = ", ".join(
        f"{entry.get('name')} {entry.get('die')}".strip()
        for entry in sheet.get("resources") or []
        if isinstance(entry, dict) and entry.get("name")
    )
    if resources:
        belongings.append(f"{cat.text('sheet.resources')} {resources}")
    doses = sheet.get("doses")
    if isinstance(doses, dict) and doses:
        belongings.append(
            f"{cat.text('sheet.doses')} "
            + ", ".join(f"{_titleize(str(kind))} ×{count}" for kind, count in doses.items())
        )

    abilities = []
    gifts = ", ".join(_titleize(entry) for entry in sheet.get("gifts") or [])
    if gifts:
        abilities.append(f"{cat.text('sheet.gifts')} {gifts}")
    if sheet.get("spells"):
        abilities.append(f"{cat.text('sheet.spells')} {joined('spells')}")
    if sheet.get("powers"):
        abilities.append(f"{cat.text('sheet.powers')} {joined('powers')}")
    runic_weapon = str(sheet.get("runic_weapon") or "")
    if runic_weapon:
        abilities.append(f"{cat.text('sheet.runic_weapon')} {runic_weapon}")
    conditions = character.get("conditions") or []
    if conditions:
        abilities.append(f"{cat.text('sheet.conditions')} ⚠ " + ", ".join(conditions))
    if sheet.get("scars"):
        abilities.append(f"{cat.text('sheet.scars')} {joined('scars')}")

    notes = str(sheet.get("notes") or "")
    blocks = [identity, vitals, belongings, abilities]
    if notes:
        blocks.append([f"{cat.text('sheet.notes')}", notes])
    return "\n\n".join("\n".join(block) for block in blocks if block) + "\n"


#: The recovery commands that resolve on the line they are typed on, spelled with the
#: leading slash this channel uses. ``/revise`` is excluded because it may either carry
#: its new action inline or wait for the next line, so it needs its own two branches.
_IMMEDIATE_RECOVERY_COMMANDS = frozenset(
    f"/{name}" for name in RECOVERY_CONTROLS - {"revise"}
)

#: Every command word this channel has ever answered to, spelled with the colon marker
#: the channel used before slash commands replaced it. A line opening with one of
#: these is a player reaching for a command by its old spelling, and silently handing
#: ":quit" to the narrator as story text would be the worst answer available; a
#: one-line redirect is the kindest.
_LEGACY_COLON_COMMANDS = frozenset(
    f":{name}" for name in RECOVERY_CONTROLS | LOCAL_COMMANDS | {"character"}
)

#: What ``/character`` answers when the snapshot holds no sheet for this terminal's
#: character: before the startup refresh, or at session zero before
#: ``character_create`` writes the player link.
# ``sheet.unlinked`` in the terminal catalog carries this wording now.


class TerminalAdapter:
    """Read one local player's terminal lines and print delivered game-master text.

    ``author`` and ``character_id`` default to empty, which means "resolve from the
    campaign's own player links": each ``update_status`` snapshot carries
    ``players[player_id]``, so a session that begins at session zero — no characters
    at all — binds to the character the moment ``character_create`` writes the link,
    and a premade campaign binds at the startup refresh. Passing either explicitly
    pins it for the whole session, which is what tests use.
    """

    name = "terminal"

    def __init__(
        self,
        *,
        channel_id: str = "terminal",
        author: str = "",
        player_id: str = "terminal-player",
        character_id: str = "",
        recorder: DiagnosticRecorder | None = None,
        ui: _TerminalUI | None = None,
        catalog=None,
    ) -> None:
        self.channel_id = channel_id
        self._author = author
        self.player_id = player_id
        self._character_id = character_id
        self._linked_character_id = ""
        self._linked_display_name = ""
        self.recorder = recorder
        self.ui = ui or PromptToolkitTerminalUI(
            completer=SigilCompleter(commands=self._command_specs, mentions=self._mention_candidates)
        )
        #: This terminal's player-facing text. ``scripts/narrator_serve.py`` passes the
        #: configured language's catalog; the English default keeps a test that builds
        #: an adapter with no config working.
        self._catalog = _catalog(catalog)
        self._termination_state = ""
        self._presented_decisions: set[str] = set()
        self._decision_recovery_active = False
        self._decision_recovery_control = ""
        self._decision_revise_pending = False
        self._last_status: tuple[str, str] = ("", "")
        self._last_snapshot: dict = {}
        self._thinking_task: asyncio.Task[None] | None = None
        #: The narrator's thinking-level control, bound by ``NarratorService`` at
        #: startup; ``None`` until then, when ``/thinking`` answers that the level
        #: cannot be changed here rather than pretending to.
        self._thinking_control: ThinkingLevelControl | None = None
        #: The table's language control, bound the same way; ``None`` until then,
        #: when ``/language`` answers that the language cannot be changed here.
        self._language_control: LanguageControl | None = None
        #: The @-mention completion roster, bound the same way; ``None`` until then,
        #: when the completer simply offers no mention candidates.
        self._mention_control: MentionDirectoryControl | None = None

    @property
    def author(self) -> str:
        """This player's display label: pinned, else linked character, else Player."""
        return self._author or self._linked_display_name or "Player"

    @property
    def character_id(self) -> str:
        """This player's character id: pinned, else the campaign's current link."""
        return self._character_id or self._linked_character_id

    @property
    def decision_capabilities(self) -> ChannelCapabilities:
        """The local terminal implements the current atomic decision presentation."""
        return ChannelCapabilities(structured_decisions=True, atomic_decision_delivery=True)

    async def _render(self, prefix: str, text: str, *, kind: str) -> None:
        if self.recorder is not None:
            ending = "" if text.endswith("\n") else "\n"
            self.recorder.record(
                "terminal_output", kind=kind, text=f"{prefix}{text}{ending}"
            )
        await self.ui.render(prefix, text)

    def _terminate(self, state: str) -> None:
        if self._termination_state:
            return
        self._termination_state = state
        if self.recorder is not None:
            self.recorder.record("terminal_termination", state=state)

    def bind_thinking_level(self, control: ThinkingLevelControl) -> None:
        """Accept the control ``/thinking`` acts through (``ThinkingLevelAdapter``)."""
        self._thinking_control = control

    def _thinking_level(self) -> str:
        """The level in force, or empty when this terminal holds no control."""
        if self._thinking_control is None:
            return ""
        return str(self._thinking_control.read() or "")

    async def _answer_thinking_command(self, command: str) -> str:
        """Answer ``/thinking`` (show) or ``/thinking <level>`` (set) in catalog text.

        Returns the response text; the caller renders and records it. The level is
        deliberately absent from the status line: ``/thinking`` alone shows it.
        """
        control = self._thinking_control
        if control is None:
            return self._catalog.text("commands.thinking_unavailable")
        levels = ", ".join(control.levels)
        words = command.split(maxsplit=1)
        if len(words) == 1:
            return self._catalog.text(
                "commands.thinking_show", level=self._thinking_level(), levels=levels
            )
        requested = words[1].strip()
        try:
            applied = control.write(requested)
        except ValueError:
            return self._catalog.text(
                "commands.thinking_unknown", level=requested, levels=levels
            )
        return self._catalog.text("commands.thinking_set", level=applied)

    def bind_language(self, control: LanguageControl) -> None:
        """Accept the control ``/language`` acts through (``LanguageAdapter``)."""
        self._language_control = control

    async def _answer_language_command(self, command: str) -> str:
        """Answer ``/language`` (show) or ``/language <tag>`` (set) in catalog text.

        A successful set is two catalogs: the control's ``write`` flips the engine's
        narrator-domain strings, and this adapter reloads its own terminal-domain
        catalog for the same tag, so the status line, sheet and command replies switch
        with the notices rather than trailing them in the old language. The terminal
        catalog is loaded *before* the write, so either both domains switch or -- on a
        catalog that refuses validation -- neither does; the refusal takes the unknown
        wording because to the player the outcome is the same, the language cannot be
        selected, and the loader's detail is a translator's error, not a player's.
        The confirmation renders from the *new* catalog: the first words the player
        reads after switching are in the language they switched to.
        """
        from narrator.locale import LocaleError, load

        control = self._language_control
        if control is None:
            return self._catalog.text("commands.language_unavailable")
        languages = ", ".join(control.languages)
        words = command.split(maxsplit=1)
        if len(words) == 1:
            return self._catalog.text(
                "commands.language_show", language=control.read(), languages=languages
            )
        requested = words[1].strip()
        try:
            catalog = load(requested, control.locale_root, domain="terminal")
            applied = control.write(requested)
        except (LocaleError, ValueError):
            return self._catalog.text(
                "commands.language_unknown", language=requested, languages=languages
            )
        self._catalog = catalog
        return self._catalog.text("commands.language_set", language=applied)

    def bind_mentions(self, control: MentionDirectoryControl) -> None:
        """Accept the control this channel's @-mention completion reads from."""
        self._mention_control = control

    def _mention_candidates(self) -> tuple[MentionCandidate, ...]:
        """The current @-mention roster, or empty before ``NarratorService`` binds one.

        Omits this terminal's own player: a player does not address themselves by
        name, and offering their own character back to them read as a defect rather
        than a candidate worth typing.
        """
        if self._mention_control is None:
            return ()
        own_name = self.author
        return tuple(
            candidate
            for candidate in self._mention_control.candidates()
            if not (candidate.kind == "pc" and candidate.sigil == own_name)
        )

    def _command_specs(self) -> tuple[tuple[str, str], ...]:
        """Every command this terminal currently answers to, with its catalog summary.

        Recovery commands (``RECOVERY_CONTROLS``) only appear while a recovery window
        is open -- offering ``/retry`` when it would be refused is worse than not
        offering it. ``/character`` is terminal-only vocabulary, not part of the
        cross-module ``LOCAL_COMMANDS``/``RECOVERY_CONTROLS`` contract, so it is added
        here rather than in ``channels.base``.
        """
        names = set(LOCAL_COMMANDS) | {"character"}
        if self._decision_recovery_active:
            names |= set(RECOVERY_CONTROLS)
        return tuple(
            (name, self._catalog.text(f"autocomplete.{name}")) for name in sorted(names)
        )

    async def turns(self) -> AsyncIterator[InboundTurn]:
        """Yield normal input lines; handle terminal commands without a turn."""
        while True:
            # No name on the live prompt itself: the status line (``update_status``
            # below) already carries it every turn, and repeating it there was pure
            # duplication. The heavy-weight chevron (not the thinner U+203A) reads
            # larger at the same font size -- the same glyph Claude Code's own
            # terminal prompt uses. The echoed line once it is submitted is a
            # different question: ``self.author`` names it there, ready for a
            # multiplayer table where more than one player's lines can appear one
            # after another with nothing else to tell them apart.
            prompt = "❯ "
            if self.recorder is not None:
                self.recorder.record("terminal_output", kind="prompt", text=prompt)
            try:
                text = await self.ui.read_line(prompt, speaker=self.author)
            except EOFError:
                self._terminate("end_of_input")
                return
            except KeyboardInterrupt:
                self._terminate("keyboard_interrupt")
                return

            if self.recorder is not None:
                self.recorder.record("terminal_input", text=text)
            command = text.strip()
            if not command:
                continue
            if command == "/quit":
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response="")
                self._terminate("local_quit")
                return
            if command == "/help":
                response = self._catalog.text("commands.help")
                if self._decision_recovery_active:
                    response = (
                        self._catalog.text("commands.recovery_help")
                    )
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response=response)
                await self._render("", response, kind="local_response")
                continue
            if command == "/character":
                response = (
                    _format_character_sheet(self._last_snapshot, self.character_id, self._catalog)
                    or self._catalog.text("sheet.unlinked")
                )
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response=response)
                await self._render("", response, kind="local_response")
                continue
            if command == "/thinking" or command.startswith("/thinking "):
                response = await self._answer_thinking_command(command)
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response=response)
                await self._render("", response, kind="local_response")
                continue
            if command == "/language" or command.startswith("/language "):
                response = await self._answer_language_command(command)
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response=response)
                await self._render("", response, kind="local_response")
                continue
            if self._decision_recovery_active and command in _IMMEDIATE_RECOVERY_COMMANDS:
                self._decision_recovery_active = False
                self._decision_revise_pending = False
                self._decision_recovery_control = command[1:]
                yield InboundTurn(
                    channel_id=self.channel_id,
                    mention=ChannelMessage(
                        self.author,
                        text,
                        principal=ChannelPrincipal(self.name, self.player_id, self.author),
                    ),
                )
                continue
            if self._decision_recovery_active and command.startswith("/revise "):
                revised = text.strip()[len("/revise "):].strip()
                if not revised:
                    await self._render("", self._catalog.text("commands.revise_after"), kind="local_response")
                    continue
                self._decision_recovery_active = False
                self._decision_recovery_control = "revise"
                yield InboundTurn(
                    channel_id=self.channel_id,
                    mention=ChannelMessage(
                        self.author,
                        revised,
                        principal=ChannelPrincipal(self.name, self.player_id, self.author),
                    ),
                )
                continue
            if command == "/revise" and self._decision_recovery_active:
                self._decision_revise_pending = True
                await self._render("", self._catalog.text("commands.revise_now"), kind="local_response")
                continue
            if self._decision_revise_pending:
                self._decision_revise_pending = False
                self._decision_recovery_active = False
                self._decision_recovery_control = "revise"
                yield InboundTurn(
                    channel_id=self.channel_id,
                    mention=ChannelMessage(
                        self.author,
                        text,
                        principal=ChannelPrincipal(self.name, self.player_id, self.author),
                    ),
                )
                continue
            if self._decision_recovery_active and _STALE_DECISION_ANSWER.fullmatch(command):


                response = (
                    self._catalog.text("commands.no_decision_open")
                )
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response=response)
                await self._render("", response, kind="local_response")
                continue
            if command.startswith("/"):
                response = self._catalog.text("commands.unknown", command=command)
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response=response)
                await self._render("", response, kind="local_response")
                continue
            if command.split(maxsplit=1)[0] in _LEGACY_COLON_COMMANDS:
                response = self._catalog.text("commands.slash_required")
                if self.recorder is not None:
                    self.recorder.record("local_command", command=command, response=response)
                await self._render("", response, kind="local_response")
                continue

            yield InboundTurn(
                channel_id=self.channel_id,
                mention=ChannelMessage(
                    self.author,
                    text,
                    principal=ChannelPrincipal(
                        adapter_name=self.name,
                        subject_id=self.player_id,
                        display_name=self.author,
                    ),
                ),
            )

    async def post(self, channel_id: str, text: str) -> None:
        # Every reply this channel ever posts -- narration, a fault, a withheld
        # notice, a decision boundary -- reaches this one method (``narrator.delivery``
        # funnels all of them through ``adapter.post``), so stopping the thinking
        # indicator here, unconditionally, covers every path that answers a turn
        # without ``NarratorService.run``'s own loop needing to know which branch it
        # took. The one turn outcome with no reply at all (``phase == "pending"``, a
        # decision now awaiting the player's own answer) stops it explicitly instead,
        # where that phase is decided.
        await self.set_thinking(channel_id, False)
        await self._render("GM> ", text, kind="delivered_message")

    async def update_status(self, snapshot) -> None:
        """Push this terminal's own character status to the UI's status line.

        Also the one place this adapter learns which character its player is linked
        to: the snapshot's ``players`` map is read on every refresh, so a character
        created mid-session (session zero) binds on the delivery that follows its
        creation, with no restart and no adapter-side campaign read.
        """
        link = (snapshot.get("players") or {}).get(self.player_id) or {}
        if isinstance(link, dict):
            self._linked_character_id = str(link.get("character_id") or "")
            self._linked_display_name = str(link.get("display_name") or "")
        # ``/character`` reads this cache rather than campaign files: the adapter owns
        # inbound mechanics only, so durable state reaches it exclusively through the
        # snapshots the delivery path pushes -- the sheet is as fresh as the status line.
        self._last_snapshot = snapshot if isinstance(snapshot, dict) else {}
        self._last_status = _format_status_line(
            snapshot, self.character_id, fallback_name=self.author, catalog=self._catalog
        )
        # A thinking indicator owns the status line while it runs (below); a snapshot
        # arriving mid-turn -- the startup refresh in NarratorService.run, ahead of any
        # turn -- must still cache it for later, but must not overwrite the animation
        # a player is currently watching.
        if self._thinking_task is None:
            await self.ui.set_status(*self._last_status)

    async def set_thinking(self, channel_id: str, thinking: bool) -> None:
        """Animate the status line for the stretch between a turn and its reply.

        ``channel_id`` is accepted, not used: this terminal serves exactly one
        channel, always ``self.channel_id``, and the parameter exists only so the
        ``ThinkingIndicatorAdapter`` protocol stays meaningful for an adapter that
        someday serves several. Starting twice is a no-op (the running ticker keeps
        its own elapsed clock rather than restarting it); stopping restores whatever
        ``update_status`` last computed, cached in ``self._last_status`` rather
        than re-read, so the restored line is exactly what a legitimate snapshot
        produced and never a stale placeholder.
        """
        if thinking:
            if self._thinking_task is not None:
                return
            self._thinking_task = asyncio.ensure_future(self._animate_thinking())
            return
        task, self._thinking_task = self._thinking_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.ui.set_status(*self._last_status)

    async def _animate_thinking(self) -> None:
        started = monotonic()
        frame = 0
        while True:
            elapsed = int(monotonic() - started)
            glyph = _THINKING_FRAMES[frame % len(_THINKING_FRAMES)]
            # No right half: the indicator is the only thing worth showing while it
            # runs, and the scene/location it would otherwise share the row with
            # cannot have changed yet -- the turn it describes has not resolved.
            await self.ui.set_status(f"{glyph} Thinking ({elapsed}s)")
            frame += 1
            await asyncio.sleep(_THINKING_FRAME_SECONDS)

    async def present_decision(self, view) -> bool:
        """Render only this terminal's linked assignment, without transcript capture."""
        # Views normally arrive from the coordinator as validated Pydantic models.
        # Re-validating their dumped data preserves that boundary even if a future
        # adapter or test bypasses construction with ``model_construct``.
        try:
            view = DecisionView.model_validate(view.model_dump())
        except (AttributeError, ValidationError):
            return False
        if view.character_id != self.character_id:
            return False
        lines = [
            f"Decision for {view.character_name} ({view.kind})",
            view.question,
        ]
        if view.context:
            lines.extend(("", view.context))
        lines.append("")
        lines.extend(
            f"{index}. {option.label}"
            for index, option in enumerate(view.options, start=1)
        )
        lines.append(self._catalog.text("decision.enter_number"))
        lines.extend(_custom_option_guidance(view.options))
        await self.ui.render("", "\n".join(lines) + "\n")
        self._presented_decisions.add(view.presentation_token)
        return True

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        """Acknowledge the whole local assignment or reject it before rendering."""
        candidates = tuple(views)
        if len(candidates) != 1:
            return DecisionDeliveryReceipt(status="unavailable", decision_count=0)
        view = candidates[0]
        try:
            valid = DecisionView.model_validate(view.model_dump())
        except (AttributeError, ValidationError):
            return DecisionDeliveryReceipt(status="unavailable", decision_count=0)
        if valid.character_id != self.character_id:
            return DecisionDeliveryReceipt(status="unavailable", decision_count=0)
        if not await self.present_decision(valid):
            return DecisionDeliveryReceipt(status="unavailable", decision_count=0)
        return DecisionDeliveryReceipt(status="delivered", decision_count=1)

    @property
    def decision_session_ended(self) -> bool:
        """Whether a terminal control ended the session while a decision was active."""
        return self._termination_state in {
            "end_of_input", "keyboard_interrupt", "local_quit", "adapter_closed"
        }

    async def collect_decision(self, views):
        """Collect one typed selection through the UI's disposable decision history."""
        available = tuple(view for view in views if view.character_id == self.character_id)
        if len(available) != 1:
            return DecisionCollectionCancelled(reason="dismissed")
        view = available[0]
        if view.presentation_token not in self._presented_decisions:
            return DecisionCollectionCancelled(reason="dismissed")
        while True:
            try:
                text = await self.ui.read_decision_line("Decision> ")
            except EOFError:
                self._terminate("end_of_input")
                return DecisionCollectionCancelled(reason="session_ended")
            except KeyboardInterrupt:
                self._terminate("keyboard_interrupt")
                return DecisionCollectionCancelled(reason="session_ended")
            choice = text.strip()
            if choice in {"/quit", "quit"}:
                self._terminate("local_quit")
                return DecisionCollectionCancelled(reason="session_ended")
            if choice == "/dismiss":
                self._presented_decisions.discard(view.presentation_token)
                return DecisionCollectionCancelled(reason="dismissed")
            if choice == "/help":
                help_lines = [
                    self._catalog.text("decision.help"),
                    *_custom_option_guidance(view.options, self._catalog),
                ]
                await self.ui.render("", "\n".join(help_lines) + "\n")
                continue
            if choice == "/character":
                # A decision is exactly when the sheet matters -- whether to fight is
                # a question about HP and Doom -- so the command answers here too,
                # without consuming the prompt.
                await self.ui.render(
                    "",
                    _format_character_sheet(self._last_snapshot, self.character_id, self._catalog)
                    or self._catalog.text("sheet.unlinked"),
                )
                continue
            selected, inline_custom = _select_option(view.options, choice)
            if not selected:
                await self.ui.render("", self._catalog.text("decision.choose_listed"))
                continue
            custom = ""
            if selected == "own_approach":
                if not inline_custom:
                    # Reached only by a bare index naming the custom option: the player
                    # selected "describe your own approach" and described nothing.
                    await self.ui.render(
                        "", self._catalog.text("decision.describe_approach")
                    )
                    continue
                custom = inline_custom
            return (
                ChannelPrincipal(
                    adapter_name=self.name,
                    subject_id=self.player_id,
                    display_name=self.author,
                ),
                DecisionSubmission(
                    presentation_token=view.presentation_token,
                    selection_id=selected,
                    custom_text=custom,
                ),
            )

    async def acknowledge_decision(self, result) -> None:
        """Keep semantic errors local and omit decision prompts from diagnostics."""
        messages = {
            "invalid": self._catalog.text("errors.invalid"),
            "conflict": self._catalog.text("errors.conflict"),
            "stale": self._catalog.text("errors.stale"),
            "cancelled": self._catalog.text("errors.cancelled"),
        }
        text = messages.get(getattr(result, "status", ""), self._catalog.text("errors.rejected"))
        await self.ui.render("", text)

    async def decision_recovery(self) -> None:
        """Enable recovery controls after the engine posts a no-action boundary."""
        self._decision_recovery_active = True
        self._decision_revise_pending = False

    def consume_decision_recovery_control(self) -> str:
        """Return one terminal-only recovery command without exposing it as player text."""
        control = self._decision_recovery_control
        self._decision_recovery_control = ""
        return control

    async def close(self) -> None:
        self._terminate("adapter_closed")
        # A turn interrupted mid-flight (Ctrl-C) can leave the ticker still running;
        # left uncancelled it would keep calling ``self.ui.set_status`` against a UI
        # that is closing under it.
        if self._thinking_task is not None:
            await self.set_thinking(self.channel_id, False)
        await self.ui.close()
