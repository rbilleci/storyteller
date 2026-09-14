"""The roll and Doom announcement subsystem ``NarratorEngine`` calls.

Moved out of ``narrator.engine`` verbatim: every function and constant here is
pure -- built from tool-call payloads, narration text, and the typed facts those
payloads yield -- with no dependency on ``self`` or any class, so it stands alone.
"""

from __future__ import annotations

import re

from narrator import policy

#: The roll announcement ``skills/bsh-gm/SKILL.md`` mandates as the literal first line
#: of any reply carrying a ``roll`` field: ``<Name> rolls <ATTRIBUTE>: rolled <total>
#: vs target <target>, <outcome>.`` The ``rolled``/``target`` labels are optional in
#: the pattern (a model that still writes the older bare form, or "vs a target of 9",
#: parses identically). Deliberately unanchored. The obvious reading -- the skill
#: says "first line", so anchor to a line start -- fails the very file the rule lives
#: in: every exemplar there sits mid-sentence inside backticks, and the test that holds
#: those exemplars to ``classify`` has to find them. Unanchored costs nothing, because
#: the shape ``<word> rolls <LETTERS>: <digits> vs <digits>, <verdict>`` is not a
#: sentence ordinary fiction produces by accident, and it gains the ability to catch a
#: model that buries an announcement mid-paragraph instead of leading with it.
#:
#: Tolerant of the emphasis a live model adds, and of ``vs.`` for ``vs``, since neither
#: changes the four numbers this compares. The name and the attribute must both be
#: alphabetic, which is what keeps the skill file's own
#: ``<Name> rolls <ATTRIBUTE>: <total> vs <target>, <outcome>`` format specification
#: from parsing as an announcement about a character called "<Name>".
_ROLL_ANNOUNCEMENT_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z][\w'\-]*)\s+rolls\s+(?P<attribute>[A-Za-z]{2,4})\s*:\s*"
    r"(?:rolled\s+)?(?P<total>\d+)\s+vs\.?\s+(?:a\s+)?(?:target\s+(?:of\s+)?)?(?P<target>\d+)\s*,\s*"
    r"(?P<outcome>critical\s+success|critical\s+failure|success|failure)\b",
    re.IGNORECASE,
)

#: How an announced verdict word maps onto the ``outcome`` string a tool returns
#: (``bsh_mcp.rules.classify``'s own four values). The skill instructs the model to
#: write "critical success" and "critical failure" with a space, so the underscored
#: wire values never appear in narration and this is the only translation needed.
_ANNOUNCED_OUTCOMES: dict[str, str] = {
    "success": "success",
    "failure": "failure",
    "critical success": "critical_success",
    "critical failure": "critical_failure",
}

#: The ``helpless_roll`` result that is not a recovery. Every other face of
#: ``rules/helpless-table.json`` restores ``survivor_hp_die`` hit points and leaves a
#: living character, so every other face is a recovery the table must hear about.
_HELPLESS_DEATH = "killed"


def announced_rolls(text: str, catalog=None) -> tuple[dict, ...]:
    """Every roll announcement the text's own words state, as typed fields.

    Reads text alone and asks nothing about truth, exactly like
    ``narrator.sweep.stated_coin_totals`` and its siblings: this reports what the
    narration claims, and ``verdict_mismatches`` below decides whether the claim
    holds. Returning typed fields rather than the matched line keeps every caller --
    including ``scripts/probe_combat_decisions.py``, which must retain no model prose
    -- able to compare numbers without holding narration.
    """
    found = []
    for match in _ROLL_ANNOUNCEMENT_PATTERN.finditer(text):
        outcome = _ANNOUNCED_OUTCOMES.get(
            re.sub(r"\s+", " ", match.group("outcome")).strip().casefold()
        )
        if outcome is None:  # pragma: no cover - the pattern admits no other word
            continue
        found.append(
            {
                "character": match.group("name"),
                "attribute": match.group("attribute").upper(),
                "total": int(match.group("total")),
                "target": int(match.group("target")),
                "outcome": outcome,
            }
        )
    # The language-independent arm of the union (docs/spec-language-independent-
    # guards.md §3.6): ``Catalog.parse_rolls`` is derived from the same
    # ``roll.announcement`` string ``render_roll`` renders, so a language's detector
    # ships with its translation and cannot drift from the renderer. The English
    # regex above stays as the looser arm -- it catches the model's English
    # variations ("vs.", "a target of") that no template derivation would.
    if catalog is not None:
        seen = {(f["attribute"], f["total"], f["target"], f["outcome"]) for f in found}
        for parsed in catalog.parse_rolls(text):
            key = (parsed["attribute"], parsed["total"], parsed["target"], parsed["outcome"])
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "character": parsed["name"],
                    "attribute": parsed["attribute"],
                    "total": parsed["total"],
                    "target": parsed["target"],
                    "outcome": parsed["outcome"],
                }
            )
    return tuple(found)


def _classify_roll_under(selected: int, total: int, target: int) -> str:
    """Classify a roll-under result, duplicated byte for byte from
    ``bsh_mcp.rules.classify`` rather than imported -- the identical ``mcp``
    version-conflict reason ``TRAVERSAL_CLOCK_PREFIX`` above states for
    ``normalize_for_similarity``'s own duplicate of
    ``narrator.soak_instruments.normalize_for_lenient_match``.

    Needed only for ``grant_runic_weapon``'s own envelope (``src/bsh_mcp/
    service.py``), whose ``details.session_test`` field carries a roll's raw
    numbers (``Roll.as_dict()``) but no already-classified ``outcome`` string the
    way every other roll-carrying tool's envelope does -- that tool sets
    ``details["session_test"] = session_test.roll.as_dict()``, the bare roll, never
    the ``TestResult.as_dict()`` that would carry ``outcome`` alongside it. Every
    other branch in ``roll_facts`` carries an ``outcome`` out of its own envelope
    verbatim, by design (``verdict_mismatches``'s own docstring: "this never
    recomputes a verdict and never needs to"); this one function is the sole,
    narrow exception, confined to the one tool whose envelope leaves no verbatim
    outcome to carry. ``tests/test_narrator_units.py`` cross-checks this against
    the real ``bsh_mcp.rules.classify`` directly (that test file runs in the
    project environment, not this module's own ``mcp``-1.29 narrator environment,
    so the import restriction does not bind there), so a future change to the real
    function's own logic fails that test rather than silently drifting unnoticed.
    """
    if selected == 1:
        return "critical_success"
    if selected == 20:
        return "critical_failure"
    return "success" if total < target else "failure"


def roll_facts(tool: str, payload: dict) -> tuple[dict, ...]:
    """Every roll one tool result actually returned, as the same typed fields.
    """

    def _fact(
        character: str, attribute: str, roll: object, outcome: object, target: object,
        name: str = "", kills_helpless: bool | None = None,
    ) -> dict | None:
        if not isinstance(roll, dict):
            return None
        total = roll.get("total")
        if not isinstance(total, int) or isinstance(total, bool):
            return None
        for candidate in (roll.get("target"), target):
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                resolved_target = candidate
                break
        else:
            resolved_target = None
        return {
            "tool": tool,
            "character": str(character or ""),
            "attribute": str(attribute or "").upper(),
            "total": total,
            "target": resolved_target,
            "outcome": str(outcome or ""),
            "name": str(name or ""),
            "kills_helpless": kills_helpless if isinstance(kills_helpless, bool) else None,
        }

    if not isinstance(payload, dict) or not payload.get("ok"):
        return ()
    facts: list[dict] = []
    individual = payload.get("individual")
    if isinstance(individual, list):
        for entry in individual:
            if not isinstance(entry, dict):
                continue
            fact = _fact(
                entry.get("name") or entry.get("character_id") or "",
                payload.get("attribute", ""),
                entry.get("roll"),
                entry.get("outcome"),
                entry.get("target"),
            )
            if fact is not None:
                facts.append(fact)
    initiative = payload.get("initiative")
    if isinstance(initiative, list):
        for entry in initiative:
            if not isinstance(entry, dict):
                continue
            fact = _fact(
                entry.get("name") or entry.get("character_id") or "",
                "WIS",  # combat_start always rolls initiative as a WIS test.
                entry.get("roll"),
                entry.get("outcome"),
                entry.get("target"),
            )
            if fact is not None:
                facts.append(fact)
    details = payload.get("details")
    if isinstance(details, dict):
        session_test = details.get("session_test")
        if isinstance(session_test, dict):
            selected = session_test.get("selected")
            total = session_test.get("total")
            target = session_test.get("target")
            if (
                isinstance(selected, int) and not isinstance(selected, bool)
                and isinstance(total, int) and not isinstance(total, bool)
                and isinstance(target, int) and not isinstance(target, bool)
            ):
                fact = _fact(
                    payload.get("character_id") or "",  # the wielder
                    "INT",  # the weapon's session test is always an INT test.
                    session_test,
                    _classify_roll_under(selected, total, target),
                    target,
                    name=details.get("name") or "",  # the roller is the weapon
                    kills_helpless=details.get("kills_helpless"),
                )
                if fact is not None:
                    facts.append(fact)
    runic_session_tests = payload.get("runic_session_tests")
    if isinstance(runic_session_tests, list):
        # ``session_close`` re-rolls every runic weapon's session test for the next
        # session, one entry per weapon, each already classified by the engine.
        for entry in runic_session_tests:
            if not isinstance(entry, dict):
                continue
            fact = _fact(
                entry.get("character_id") or "",
                "INT",
                entry.get("roll"),
                entry.get("outcome"),
                entry.get("weapon_int"),
                name=entry.get("weapon_name") or "",
                kills_helpless=entry.get("kills_helpless"),
            )
            if fact is not None:
                facts.append(fact)
    # The roller's own identifier, whichever key the tool spells it under.
    # ``defender_id`` outranks ``attacker_id`` deliberately: ``combat_defend``'s
    # envelope carries both, and the character who rolled is the defender -- the
    # attacker is the enemy whose strike the roll answers.
    fact = _fact(
        payload.get("character_id")
        or payload.get("defender_id")
        or payload.get("attacker_id")
        or "",
        payload.get("attribute", ""),
        payload.get("roll"),
        payload.get("outcome"),
        payload.get("target"),
    )
    if fact is not None:
        facts.append(fact)
    return tuple(facts)


DOOM_ANNOUNCED_TOOLS: frozenset[str] = frozenset(
    {"attribute_test", "group_test", "combat_attack", "combat_defend", "combat_start"}
)


def doom_facts(tool: str, payload: dict) -> tuple[dict, ...]:
    """Every Doom-die roll one tool result actually returned, as typed fields.

    Mirrors ``roll_facts`` for the Doom die rather than an attribute test: the truth
    side of a comparison read from the tool's own envelope, never from narration.

    An earlier version of this function read only a top-level ``doom`` list, on the
    strength of ``events.jsonl``'s own committed record (seq 10 and 13 of a live
    session, one entry each, one held and one downgraded from d6 to d4) -- but that
    record is ``bsh_mcp.service``'s internal ``commit_payload``, written to the audit
    log through ``_doom_log``, and not the same dict as the *return envelope* a tool
    call actually hands back to this narrator. The two share many keys and were easy
    to conflate; they are not the same object in code, and only the return envelope
    is this function's own input. Checked directly against a live ``combat_attack``
    (two genuine, separately-committed calls, the second landing while the first
    still held ``"attack"`` in the actor's persisted ``actions_taken``): the rules
    engine rolled Doom correctly both times, but the *return* envelope carried no
    ``doom`` key at all -- the same real roll rode under ``repeat_action_doom``
    instead, a single dict, never a list. This function silently saw nothing for
    the whole time Batch A shipped, which is exactly the gap Batch B's live
    measurement surfaced: a model narrating a real, correctly-rolled Doom
    consequence that this function could not confirm, because it was never looking
    in the place the roll actually rode.

    Each source below is a ``DoomOutcome.as_dict()`` (``mode``, ``previous_die``,
    ``current_die``, ``downgraded``, ``depleted``, and ``roll`` when a roll actually
    happened) or ``_mandatory_doom``'s own dict, the same shape plus a ``trigger``
    string this function does not need:

    * ``doom``: a list on no tool's actual return envelope today, kept for a tool
      that later adopts the shape ``_doom_log`` already gives the audit log, or a
      bare dict, which is what ``attribute_test`` returns for its own critical
      failure today.
    * ``repeat_action_doom``, ``critical_failure_doom``, ``called_on_doom``: single
      dicts (or absent), the actual per-concept keys ``attribute_test``,
      ``combat_attack``, and ``combat_defend`` use, never more than one real roll
      behind any one key on a single call. A repeated attack that also crits can
      populate both ``repeat_action_doom`` and ``critical_failure_doom`` on the same
      ``combat_attack`` reply -- two independent Doom rolls, both real, both owed an
      announcement.
    * ``initiative``: ``combat_start``'s own per-character list, each entry nesting
      its own single ``_mandatory_doom`` dict under ``entry["doom"]`` -- because
      initiative is per-character where an attack or a defence names only the one
      character who rolled.
    * ``individual``: ``group_test``'s own per-participant list, the same nested
      shape as ``initiative`` for the same reason -- a group test can roll Doom for
      more than one participant in a single call.

    All of these read one shared ``character_id`` (or ``defender_id``, or
    ``attacker_id``) for the single-actor tools; ``initiative`` and ``individual``
    resolve their own actor per entry, since those two are the only shapes naming
    more than one person.

    A roll-less entry (``mode`` other than a roll -- ``restore``, or a Doom die
    already spent before this call) carries no ``roll`` key and contributes no fact:
    there is no total to announce, and ``bsh_mcp.service`` itself already refuses a
    roll against an already-depleted die rather than returning one of these.
    """

    def _fact(character: str, entry: object) -> dict | None:
        if not isinstance(entry, dict):
            return None
        roll = entry.get("roll")
        total = roll.get("total") if isinstance(roll, dict) else None
        if not isinstance(total, int) or isinstance(total, bool):
            return None
        return {
            "tool": tool,
            "character": str(character or ""),
            "previous_die": str(entry.get("previous_die", "")),
            "current_die": str(entry.get("current_die", "")),
            "total": total,
            "downgraded": bool(entry.get("downgraded", False)),
            "depleted": bool(entry.get("depleted", False)),
        }

    if not isinstance(payload, dict) or not payload.get("ok") or tool not in DOOM_ANNOUNCED_TOOLS:
        return ()
    facts: list[dict] = []
    character_id = (
        payload.get("character_id")
        or payload.get("defender_id")
        or payload.get("attacker_id")
        or ""
    )
    doom_field = payload.get("doom")
    if isinstance(doom_field, list):
        for entry in doom_field:
            fact = _fact(character_id, entry)
            if fact is not None:
                facts.append(fact)
    elif isinstance(doom_field, dict):
        fact = _fact(character_id, doom_field)
        if fact is not None:
            facts.append(fact)
    for key in ("repeat_action_doom", "critical_failure_doom", "called_on_doom"):
        fact = _fact(character_id, payload.get(key))
        if fact is not None:
            facts.append(fact)
    for list_key in ("initiative", "individual"):
        entries = payload.get(list_key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            fact = _fact(entry.get("character_id") or entry.get("name") or "", entry.get("doom"))
            if fact is not None:
                facts.append(fact)
    return tuple(facts)


def format_doom_announcement(fact: dict, name: str, catalog=None) -> str:
    """Render one real Doom fact in this language, from canonical inputs.

    Three mutually exclusive result phrases -- held, downgraded, spent -- checked in
    that priority order because a depleting roll can also carry ``downgraded: True``
    (stepping into the spent state is itself a step down), and "the die is spent" is
    the fact a table needs, not "the die drops to spent".
    """
    from narrator.decisions import _catalog

    cat = _catalog(catalog)
    if fact["depleted"]:
        result = cat.text("doom_results.depleted")
    elif fact["downgraded"]:
        result = cat.text("doom_results.downgraded", next=fact["current_die"])
    else:
        result = cat.text("doom_results.held")
    return cat.text(
        "doom.announcement",
        name=name, die=fact["previous_die"], total=fact["total"], result=result,
    )


def inject_missing_doom_announcements(
    narration: str, facts: tuple[dict, ...], names: dict[str, str], catalog=None
) -> tuple[str, int]:
    """Prepend the mandatory Doom announcement for every real Doom fact this turn's
    own tool results carry, the same structural treatment
    ``inject_missing_roll_announcements`` gives the roll-under announcement and for
    the same reason: the line carries no creative content the model could contribute
    beyond what the tool result already returned with certainty.

    Always prepended after any roll-under lines: a roll-under fact and a Doom fact
    can share one tool call (a repeated attack rolls both), and the roll-under
    announcement -- what happened to the target -- reads first, with the Doom
    consequence following as its own line, matching ``skills/bsh-gm/SKILL.md``'s \"its
    own short line distinct from the hit or miss it rode in on.\"
    """
    if not facts:
        return narration, 0
    lines = [
        format_doom_announcement(
            fact,
            names.get(fact.get("character") or "")
            or str(fact.get("character") or "").replace("_", " ").replace("-", " ").title()
            or "Someone",
            catalog,
        )
        for fact in facts
    ]
    prefix = "\n".join(lines)
    amended = f"{prefix}\n\n{narration}" if narration.strip() else prefix
    return amended, len(facts)


META_TOOL_NAME_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(name)
        for name in sorted(policy.PLAYER_FACING_TOOLS - policy.GM_DISCUSSION_TOOLS)
    )
    + r")\b"
)


META_PROMISE_PATTERN = re.compile(
    r"\bI(?:'ll|\s+will)\s+(?:now\s+)?(?:call|roll|resolve|invoke)\b", re.IGNORECASE
)


def _meta_leak_detected(text: str) -> bool:
    """Whether ``text`` names a refused tool or promises to invoke one right now."""
    return bool(META_TOOL_NAME_PATTERN.search(text) or META_PROMISE_PATTERN.search(text))


def _drop_meta_leak_sentences(text: str) -> tuple[str, int]:
    """Paragraph breaks (``\\n\\n``) survive; sentences kept within one paragraph
    rejoin with a single space, which is the one formatting cost of a path meant
    to be rare -- the corrective re-invoke above resolves most cases cleanly.
    """
    dropped = 0
    kept_paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        kept = [sentence for sentence in sentences if not _meta_leak_detected(sentence)]
        dropped += len(sentences) - len(kept)
        if kept:
            kept_paragraphs.append(" ".join(kept))
    return "\n\n".join(kept_paragraphs).strip(), dropped


def _states_helpless_recovery(narration: str, fact: dict) -> bool:
    """Whether the narration states the recovery one ``helpless_roll`` returned.

    Two signals count, either alone. The table result's own name -- ``Scratched``,
    ``Missed``, ``Impaired``, ``Injured``, ``Butchered`` in
    ``rules/helpless-table.json`` -- appearing as a whole word, which is the roll's
    literal outcome word; or a stated hit-point figure, since every non-fatal face
    restores ``survivor_hp_die`` hit points and saying the new total is saying the
    character came back. Matching the outcome word alone would miss a narration that
    reports the restored hit points in plain fiction, and matching a figure alone
    would miss the several faces whose consequence is a condition rather than a number.

    ``Missed`` is a word ordinary combat fiction also uses, so a turn that happens to
    say "the blade missed" satisfies this rule without meaning to. That is a false
    negative -- the guard stays quiet on a turn it might have flagged -- and it is the
    safe direction here for the same reason ``narrator.sweep``'s detection stays
    narrow: this guard's error budget belongs to not disturbing correct turns.
    """
    outcome = fact.get("outcome", "")
    if outcome and re.search(rf"\b{re.escape(outcome)}\b", narration, re.IGNORECASE):
        return True
    # The figure signal is any decimal digit (Unicode ``\d``), not the English
    # "hit points"/"hp" unit: a French or Japanese narration states the restored
    # total without either word, and widening toward "stated" is this guard's own
    # documented safe direction -- it stays quiet rather than demanding a rewrite
    # of a turn that did announce the recovery in its own language (spec §1.4).
    return bool(re.search(r"\d", narration))


def verdict_mismatches(
    narration: str, facts: tuple[dict, ...] | list[dict], catalog=None
) -> tuple[dict, ...]:
    """Every announced roll this turn's own tool results do not back, and every
    unannounced recovery.

    An announcement resolves against the rolls sharing its attribute and its stated
    total, which is enough to identify the roll even when two characters test the same
    attribute in one turn, and the character name is deliberately not part of the
    match: a model that prints the right numbers under a misspelled name has made a
    different, smaller mistake than the one this guard exists to catch, and failing it
    here would spend the guard's credibility on spelling. Three outcomes:

    A false positive costs one extra model call and never a turn -- ``_correct_verdicts``
    asks for a rewrite rather than withholding -- which is why ``unbacked`` is included
    despite a narration that recaps an earlier turn's roll in this exact format being
    able to trip it.
    """
    mismatches: list[dict] = []
    tests = [
        fact for fact in facts if fact.get("attribute") and fact.get("total") is not None
    ]
    for line in announced_rolls(narration, catalog):
        candidates = [
            fact
            for fact in tests
            if fact["attribute"] == line["attribute"] and fact["total"] == line["total"]
        ]
        if any(
            fact["target"] == line["target"] and fact["outcome"] == line["outcome"]
            for fact in candidates
        ):
            continue
        if any(fact["target"] == line["target"] for fact in candidates):
            kind = "verdict"
        elif candidates:
            kind = "target"
        else:
            kind = "unbacked"
        mismatches.append(
            {
                "kind": kind,
                "attribute": line["attribute"],
                "total": line["total"],
                "announced_target": line["target"],
                "announced_outcome": line["outcome"],
            }
        )
    for fact in facts:
        if fact.get("tool") != "helpless_roll":
            continue
        if str(fact.get("outcome", "")).casefold() == _HELPLESS_DEATH:
            continue
        if _states_helpless_recovery(narration, fact):
            continue
        mismatches.append(
            {
                "kind": "helpless",
                "attribute": "",
                "total": fact.get("total"),
                "announced_target": None,
                "announced_outcome": "",
            }
        )
    return tuple(mismatches)


def scrub_contradicted_announcements(
    narration: str, facts: tuple[dict, ...] | list[dict], catalog=None
) -> tuple[str, int]:
    """Cut every announcement the facts contradict out of ``narration``; keep the rest.
    """
    contradicted = {
        (m["attribute"], m["total"], m["announced_target"], m["announced_outcome"])
        for m in verdict_mismatches(narration, facts, catalog)
        if m["kind"] in ("verdict", "target", "unbacked")
    }
    if not contradicted:
        return narration, 0
    removed = 0

    def _cut(match: re.Match) -> str:
        nonlocal removed
        outcome = _ANNOUNCED_OUTCOMES.get(
            re.sub(r"\s+", " ", match.group("outcome")).strip().casefold()
        )
        key = (
            match.group("attribute").upper(),
            int(match.group("total")),
            int(match.group("target")),
            outcome,
        )
        if key not in contradicted:
            return match.group(0)
        removed += 1
        return ""

    scrubbed = re.sub(
        _ROLL_ANNOUNCEMENT_PATTERN.pattern + r"[.。]?[ \t]*",
        _cut,
        narration,
        flags=re.IGNORECASE,
    )
    if catalog is not None:
        # The localized arm of the same cut: the catalog's own pattern locates the
        # announcement in the table's language, and its display vocabularies map the
        # matched words back to the canonical key the contradiction set carries.
        by_attribute = {v: k for k, v in catalog.attributes.items()}
        by_outcome = {v: k for k, v in catalog.outcomes.items()}

        def _cut_localized(match: re.Match) -> str:
            nonlocal removed
            key = (
                by_attribute.get(match.group("attribute")),
                int(match.group("total")),
                int(match.group("target")),
                by_outcome.get(match.group("outcome")),
            )
            if key not in contradicted:
                return match.group(0)
            removed += 1
            return ""

        scrubbed = re.sub(
            catalog.roll_pattern().pattern + r"[.。]?[ \t]*", _cut_localized, scrubbed
        )
    if not removed:
        return narration, 0
    # Collapse what the cut left behind: a line that was only the announcement,
    # or a leading space before the prose that followed it on the same line.
    lines = [line.rstrip() for line in scrubbed.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, removed


#: The roll-under tools ``skills/bsh-gm/SKILL.md``'s "Announcing rolls" policy covers
#: -- the same four ``narrator/soak_instruments.py``'s own ``ROLL_UNDER_TOOLS`` measures,
#: kept as its own copy here rather than imported, because a production module must
#: not depend on an evidence script. Each resolves to a player character rolling
#: (``combat_attack``'s own docstring: "Resolve one player attack against one NPC";
#: an enemy's action is resolved through the defending player's own ``combat_defend``
#: roll, never a roll of the enemy's own), so every fact these tools contribute names
#: a party character, never an NPC.
ROLL_UNDER_TOOLS: frozenset[str] = frozenset(
    {"attribute_test", "group_test", "combat_attack", "combat_defend"}
)


ANNOUNCED_ROLL_TOOLS: frozenset[str] = ROLL_UNDER_TOOLS | {
    "combat_start", "grant_runic_weapon", "session_close",
}


def format_roll_announcement(fact: dict, name: str, catalog=None) -> str:
    """Render one real roll fact in the exact shape SKILL.md's "Announcing rolls"
    policy specifies: ``<Name> rolls <ATTRIBUTE>: rolled <total> vs target <target>,
    <outcome>.``

    Both numbers are labelled because the bare ``4 vs 9`` form left a live table
    unable to tell which number the dice produced and which the character sheet
    supplied -- in a roll-UNDER game the low number winning reads backwards to
    anyone arriving from roll-high systems.

    ``fact["outcome"]`` carries the rules engine's own literal, ``critical_failure``
    with an underscore (``bsh_mcp.rules``'s ``Outcome`` literal); the policy's example
    spells the critical forms with a space, so that substitution happens here, once,
    rather than asking every caller to remember it.
    """
    from narrator.decisions import _catalog

    return _catalog(catalog).render_roll(
        name=name,
        attribute=str(fact["attribute"]),
        total=int(fact["total"]),
        target=int(fact["target"]),
        outcome=str(fact.get("outcome", "")),
    )


def inject_missing_roll_announcements(
    narration: str, facts: tuple[dict, ...], names: dict[str, str], catalog=None
) -> tuple[str, int]:
    """Prepend the mandatory announcement line for every real roll-under fact this
    turn's own tool results carry that the narration does not already state.

    Checking what the narration already states, rather than always prepending
    unconditionally, is what keeps this safe to run alongside a model that has not
    perfectly learned the new \"do not write this yourself\" instruction
    (``skills/bsh-gm/SKILL.md``): a model that disobeys and states a line anyway is
    recognised via the same ``announced_rolls`` match ``verdict_mismatches`` already
    uses, and gets no second, redundant line. Matching is on attribute and total
    only, the same deliberate choice ``verdict_mismatches`` documents: identifying the
    roll, not the character name's spelling, is this function's job.

    A fact carrying its own ``name`` (a runic weapon's session test, where the roller
    is the weapon) is announced under that name verbatim. Otherwise a fact whose
    character id is not in ``names`` (never expected in practice, since every one of
    the other tools resolves to a party character already recorded in
    ``campaign/players.yaml`` -- see ``ROLL_UNDER_TOOLS``) falls back to a readable
    transform of the bare id rather than silently dropping the announcement or
    raising, because a slightly-off name is a far smaller defect than the omission
    this function exists to eliminate.

    A runic weapon's session test carries one more line beside its announcement --
    ``runic_stake_line`` below -- stating what the verdict means for the wielder,
    added whether or not the announcement itself was missing and never twice.

    Returns the narration with any missing lines prepended, oldest fact first, and
    how many announcements were added (stake lines are not counted).
    """
    candidates = [
        fact
        for fact in facts
        if fact.get("tool") in ANNOUNCED_ROLL_TOOLS
        and fact.get("attribute")
        and isinstance(fact.get("total"), int)
        and isinstance(fact.get("target"), int)
    ]
    if not candidates:
        return narration, 0
    announced = announced_rolls(narration, catalog)

    def _display(fact: dict) -> str:
        # A fact's own ``name`` is the roller's display name when the roller is not
        # a character at all -- a runic weapon testing its own INT -- and is used
        # verbatim; every other shape resolves the character id through ``names``.
        return (
            str(fact.get("name") or "")
            or _wielder(fact)
        )

    def _wielder(fact: dict) -> str:
        return (
            names.get(fact.get("character") or "")
            or str(fact.get("character") or "")
            .replace("_", " ")
            .replace("-", " ")
            .title()
            or "Someone"
        )

    lines: list[str] = []
    missing = 0
    for fact in candidates:
        if not any(
            line["attribute"] == fact["attribute"] and line["total"] == fact["total"]
            for line in announced
        ):
            lines.append(format_roll_announcement(fact, _display(fact), catalog))
            missing += 1
        stake = runic_stake_line(fact, _wielder(fact), catalog)
        # The stake line is attached to the fact, not to the announcement: a model
        # that wrote the announcement itself still owes the table the verdict's
        # meaning, and one that already carries this exact sentence gets no second.
        if stake and stake not in narration:
            lines.append(stake)
    if not lines:
        return narration, 0
    prefix = "\n".join(lines)
    amended = f"{prefix}\n\n{narration}" if narration.strip() else prefix
    return amended, missing


def withheld_mechanical_lines(
    facts: tuple[dict, ...] | list[dict], names: dict[str, str], catalog=None
) -> tuple[str, ...]:
    """The engine-authored announcement and stake lines a turn owes even when it
    will not deliver.
    """
    amended, _ = inject_missing_roll_announcements("", tuple(facts), names, catalog)
    return tuple(line for line in amended.split("\n") if line.strip())


def runic_stake_line(fact: dict, wielder: str, catalog=None) -> str:
    """The one sentence that says what a runic weapon's session test decided, or ``\"\"``
    for any fact that is not one.
    """
    verdict = fact.get("kills_helpless")
    if not isinstance(verdict, bool) or not fact.get("name"):
        return ""
    from narrator.decisions import _catalog

    key = "armed" if verdict else "disarmed"
    if fact.get("tool") == "session_close":
        key += "_next"
    return _catalog(catalog).text(
        f"runic.{key}", name=str(fact["name"]), wielder=wielder
    )
