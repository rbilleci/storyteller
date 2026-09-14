"""The single outbound choke point. Every message a player sees crosses this module.

Two rules hold at this boundary. Requirement ``NR-DELIVERY-GATE``: no model text reaches
a player while the fiction-debt ledger holds unratified outcomes. Requirement
``NR-LEAK-FILTER``: raw tool-call markup never reaches a player.

The filter scans rather than pattern-matches. An earlier version used regular
expressions and failed on the shape this project's own tools produce: three model-facing tools
declare object or array-of-object parameters, so ``new_clocks``, ``initial_ranges`` and
``clock_updates`` all nest, and a delimiter inside a nested group never matched.
Balanced delimiters are not a regular language. The attempted workaround also cost 1.69
seconds of processor time on a 32,000-character reply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from narrator import ledger, status
from narrator.channels.base import RECOVERY_CONTROLS as _RECOVERY_CONTROLS

#: Wrapper openers, longest first so ``<|tool_call>`` never matches as a bare ``<|``.
_OPENERS: tuple[str, ...] = ("<|tool_call>", "<tool_call>", "<|")

#: Wrapper closers. Any of them terminates a wrapped call.
_CLOSERS: tuple[str, ...] = ("<|/tool_call|>", "<tool_call|>", "</tool_call>")

#: The delimiter that identifies a bare tool-call body, which carries no wrapper at all.
_ARG_DELIMITER = '<|"|>'

#: Every control-token name this project has recorded or that the model's chat template
#: emits. A truncated bare ``<`` strips only when it prefixes one of these, so ordinary
#: prose survives. This tuple is the single source of truth: ``tests`` derives its
#: coverage from it, so adding a name here adds its wrapped and truncated cases at once.
#:
#: An audit found an earlier version of this tuple holding two names. Gemma delimits
#: every turn with ``<start_of_turn>`` and ``<end_of_turn>``, so a reply cut at
#: ``max_tokens`` leaked both to a player. Two names were fewer than the family the
#: filter already had to match.
_CONTROL_TOKEN_NAMES: tuple[str, ...] = (
    "tool_call",
    "turn",
    "start_of_turn",
    "end_of_turn",
    "end",
    "bos",
    "eos",
    "pad",
)


@dataclass(frozen=True)
class TurnOutcome:
    """What one narrator turn produced, and whether a player may see it."""

    narration: str
    ratified: bool
    withheld: bool
    #: How the ledger closed: "commit", "waive", "none" (already clear), or "failed".
    settle: Literal["commit", "waive", "none", "failed"] = "none"
    settle_attempts: int = 0
    #: The sweep's disposition: ``record``, ``none``, ``failed``, ``skipped`` (never
    #: attempted), or ``pending`` (dispatched in the background, C2; resolves into
    #: one of the other three at the start of the *next* turn, and patches this
    #: turn's own ``_sweep_log`` slot in place when it does — this turn's own
    #: ``TurnOutcome.sweep`` stays ``pending`` forever, since a ``TurnOutcome`` is
    #: immutable and already returned by the time the real disposition is known).
    #: Delivery ignores it either way — the sweep never gates a turn.
    sweep: Literal["record", "none", "failed", "skipped", "pending"] = "skipped"
    leaks_scrubbed: int = 0
    error: str = ""
    decision_recovery: bool = False
    # The engine attaches this only after an authenticated attribute-test result.
    # It stays typed process-local data and never becomes player dialogue.
    social_test_outcome: object | None = None
    # One redacted record per executed tool this turn: name, ``ok``, error code,
    # and event id, never arguments or result content. The service records these to
    # the diagnostic transcript so a retained session can show which tools ran and
    # whether each rolled. Delivery never reads this; it withholds nothing.
    tool_events: tuple[dict, ...] = ()


    mechanical_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionPhaseOutcome:
    """A decision phase result. It is not narration and never carries model prose.

    ``status`` is only ever set by ``present_decisions`` below, to ``"fault"`` or
    ``"presented"``. Distinct from ``narrator.service.RoutingPhase``, a larger,
    unrelated vocabulary answering a different question (what should the turn router
    do next) despite the similar name.
    """

    status: Literal["fault", "presented"]
    kind: str = ""
    target_count: int = 0
    failure_category: str = ""


@dataclass(frozen=True)
class TurnPost:
    """One outbound player-visible message, typed: meaning stored, bytes derived.

    ``kind`` is what the player is reading: model ``narration`` or an engine-authored
    ``notice``. ``origin`` is whether a real ``NarratorEngine.run_turn`` outcome
    stands behind the post (``turn``) or a pre-engine service branch posted it
    (``service``); it is static per delivery function, never a caller judgment.
    ``notice_text`` is captured as rendered at post time -- never re-rendered from
    ``notice_key``, because the catalog is live state (the ``/language`` switch) and
    the posted bytes are what ``NarratorService._record_withhold`` feeds the next
    turn's model-facing note. ``notice_key`` is provenance metadata for instruments
    and retrospectives, ``\"\"`` where the selection site does not know it.

    The channel adapter contract is unchanged: adapters still receive ``text``, the
    flat string. Typed at rest, flat at the edges.
    """

    kind: Literal["narration", "notice"]
    origin: Literal["turn", "service"]
    model_text: str = ""
    engine_lines: tuple[str, ...] = ()
    notice_key: str = ""
    notice_text: str = ""

    @property
    def text(self) -> str:
        """The exact channel bytes, byte-identical to the pre-``TurnPost`` composition.

        Dice a withheld turn's own tools rolled are real and audited whatever became
        of the narration; ``engine_lines`` above the notice keep the rolls visible so
        ``/retry`` -- which never re-rolls -- loses nothing.
        """
        if self.kind == "narration":
            return self.model_text
        if self.engine_lines:
            return "\n".join(self.engine_lines) + "\n\n" + self.notice_text
        return self.notice_text


DECISION_INCOMPLETE_NOTICE = (
    "No action was lost: the game master's setup for this action already reached "
    "the record and does not need repeating. The action itself did not finish "
    "resolving this turn. Use /retry, /revise, or /dismiss."
)


#: The tools whose refusal explains a withheld decision turn: each is a bound or
#: expected mechanic a confirmed action resolves through. Kept apart from
#: ``narrator.engine.ROLL_UNDER_TOOLS`` for the same reason that set is its own copy:
#: this module names what a *notice* may cite, not what an announcement must match.
_REFUSABLE_MECHANIC_TOOLS: frozenset[str] = frozenset(
    {
        "attribute_test",
        "combat_attack",
        "combat_defend",
        "combat_move",
        "combat_start",
        "group_test",
    }
)


def _refused_mechanic(outcome: TurnOutcome) -> str:
    """The most recent refused mechanic call this turn, as ``tool (error_code)``.

    Both halves are engine-authored -- the tool name from this project's own served
    surface and the error code from ``bsh_mcp``'s ``CampaignError`` vocabulary,
    carried in the already-redacted ``tool_events`` -- so citing them leaks no model
    text and no arguments. The empty string means no mechanic was refused.
    """
    for event in reversed(outcome.tool_events):
        if (
            isinstance(event, dict)
            and not event.get("ok")
            and event.get("tool") in _REFUSABLE_MECHANIC_TOOLS
        ):
            tool = str(event.get("tool", ""))
            error = str(event.get("error", "") or "").strip()
            return f"{tool} ({error})" if error else tool
    return ""


def _turn_changed_state(outcome: TurnOutcome) -> bool:
    """Whether this turn's own tool calls recorded at least one state-mutating success.

    ``outcome.tool_events`` is the engine's redacted per-tool record -- name, ``ok``,
    error code, event id, never arguments or results (see ``redact_tool_event`` in
    ``narrator.engine``). A bare ``ok: true`` is not enough: an independent audit
    (result-1.json, blocker AUD-1) found this function's first version treating any
    successful call -- including a registered read-only lookup such as
    ``campaign_status`` or ``character_sheet`` (``src/bsh_mcp/server.py``'s
    ``# -- read-only tools --`` section, neither ever opening a
    ``store.transaction()``) -- as proof state changed, and reproduced it directly:
    a decision-recovery turn whose only tool event was a successful
    ``campaign_status`` call (a call the server's own instructions tell the model to
    make "after any tool error," exactly the circumstance around a stalled decision)
    posted ``DECISION_INCOMPLETE_NOTICE``'s claim that setup "already reached the
    record" when nothing was recorded. This is the same "any tool call succeeded"
    versus "the state-changing tool call succeeded" defect shape an earlier audit
    already found in ``_sweep``'s coin/HP/item guard and in ``_advance_stalled_traversal``
    (``narrator.engine``, both citing their own AUD-1 findings) -- one audit per
    milestone, so far, catching the identical mistake in a new location.

    ``event_id`` is the correct signal, already redacted into every entry:
    ``results.success()`` (``src/bsh_mcp/results.py``) sets it only when the caller
    passes ``sequence=``, and every mutating tool passes its transaction's own
    ``sequence`` there, while every read-only tool passes neither, so ``event_id`` is
    non-``None`` only for a call that actually reached ``transaction.commit()``.
    """
    return any(
        isinstance(event, dict) and event.get("ok") and event.get("event_id") is not None
        for event in outcome.tool_events
    )


def _identifier_start(text: str, brace: int) -> int | None:
    """Walk back from a ``{`` over an identifier, returning where the call begins."""
    index = brace
    while index > 0 and (text[index - 1].isalnum() or text[index - 1] == "_"):
        index -= 1
    if index == brace:
        return None
    if not (text[index].isalpha() or text[index] == "_"):
        return None
    return index


def _balanced_end(text: str, brace: int) -> int:
    """Index just past the brace group starting at ``brace``.

    Falls back to the end of the string when the group never closes, because a reply cut
    at ``max_tokens`` leaves markup open, and an unterminated call is still markup.
    """
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def _find_bare_call(text: str, start: int) -> tuple[int, int] | None:
    """Find the next ``name{...}`` body whose braces contain the argument delimiter.

    Depth matters. The delimiter may sit at any nesting level, which is exactly what the
    regular-expression version could not express.
    """
    index = text.find("{", start)
    while index != -1:
        begin = _identifier_start(text, index)
        if begin is not None:
            end = _balanced_end(text, index)
            if _ARG_DELIMITER in text[index:end]:
                return begin, end
        index = text.find("{", index + 1)
    return None


def _find_wrapped_call(text: str, start: int) -> tuple[int, int] | None:
    """Find the next wrapped call or control token, terminating at a closer or the end."""
    best: tuple[int, str] | None = None
    for opener in _OPENERS:
        found = text.find(opener, start)
        if found != -1 and (best is None or found < best[0]):
            best = (found, opener)
    if best is None:
        return None

    begin, opener = best
    after = begin + len(opener)

    if opener == "<|":
        # A bare control token such as ``<|turn>``. Run to its closing bracket, or to
        # the end of the string when a truncated reply cut it before it closed.
        for closer in ("|>", ">"):
            found = text.find(closer, after)
            if found != -1:
                return begin, found + len(closer)
        return begin, len(text)

    ends = []
    for closer in _CLOSERS:
        found = text.find(closer, after)
        if found != -1:
            ends.append(found + len(closer))
    return begin, min(ends) if ends else len(text)


def _find_closing_token(text: str, start: int) -> tuple[int, int] | None:
    """Find a closing-style control token such as ``<turn|>``.
    """
    index = text.find("<", start)
    while index != -1:
        end = text.find("|>", index)
        if end != -1:
            body = text[index + 1 : end]
            if body and all(character.isalnum() or character == "_" for character in body):
                return index, end + 2
        index = text.find("<", index + 1)
    return None


def _find_truncated_opener(text: str, start: int) -> tuple[int, int] | None:
    """Find a control-token opener cut before its closing bracket, at end of string.

    ``Narration.<|tool_call`` reaches a player verbatim otherwise, because no opener
    matched and no closer exists.

    A bare ``<`` is the dangerous case, because prose supplies it constantly. This is a
    game about rolling under a target number, so ``Roll under <13`` and ``a<b`` and
    ``<3`` are ordinary narration. An audit found an earlier rule here, which accepted
    any run of alphanumeric characters, deleting every one of those to end of string.
    A bare ``<`` therefore counts only when what follows is a prefix of a control-token
    name we have actually recorded. ``<|`` needs no such guard, because prose does not
    produce it.
    """
    for prefix in ("<|tool_call", "<tool_call", "<|", "<"):
        index = text.rfind(prefix, start)
        if index == -1:
            continue
        tail = text[index:]
        if ">" in tail or "\n" in tail:
            continue
        body = tail[len(prefix) :]
        if prefix in ("<|tool_call", "<tool_call"):
            return index, len(text)
        if prefix == "<|" and body and all(
            character.isalnum() or character == "_" for character in body
        ):
            return index, len(text)
        if prefix == "<" and len(body) >= 2 and any(
            name.startswith(body) for name in _CONTROL_TOKEN_NAMES
        ):
            return index, len(text)
    return None


def scrub_markup(text: str) -> tuple[str, list[str]]:
    """Strip every tool-call and control-token spelling from player-facing text.

    Returns the cleaned text and the fragments removed. Whitespace is repaired only at
    each removal seam, never across the whole message: an earlier version collapsed
    every run of two or more spaces once anything was removed, which flattened nested
    list indentation and Markdown tables on exactly the turns that leaked.
    """
    removed: list[str] = []
    out: list[str] = []
    cursor = 0

    while cursor < len(text):
        candidates = [
            span
            for span in (
                _find_wrapped_call(text, cursor),
                _find_bare_call(text, cursor),
                _find_closing_token(text, cursor),
                _find_truncated_opener(text, cursor),
            )
            if span is not None
        ]
        if not candidates:
            out.append(text[cursor:])
            break

        begin, end = min(candidates, key=lambda span: span[0])
        head = text[cursor:begin]
        removed.append(text[begin:end])
        tail = text[end:]

        # Repair only the seam the removal created, never the whole message.
        if head.endswith((" ", "\t")) and tail[:1] in (" ", "\t"):
            tail = tail.lstrip(" \t")
        elif head.endswith((" ", "\t")) and tail[:1] == "\n":
            head = head.rstrip(" \t")

        out.append(head)
        text = tail
        cursor = 0

    cleaned = "".join(out)
    return cleaned.strip(), removed


def strip_leaked_notices(text: str, notices: tuple[str, ...]) -> tuple[str, list[str]]:
    """Strip any engine-authored notice string a model reproduced verbatim.

    Only notices carrying no ``$placeholder`` are checked -- a templated notice such
    as ``decision_refused``'s ``$refusal`` never appears in model text verbatim by
    construction, so it could never be found this way. Whitespace is repaired only at
    each removal's own seam, the same discipline ``scrub_markup`` above keeps and for
    the same reason: collapsing whitespace across the whole message flattens list
    indentation and Markdown tables on exactly the turns that leaked.
    """
    removed: list[str] = []
    for notice in notices:
        if not notice or "$" in notice:
            continue
        while True:
            begin = text.find(notice)
            if begin < 0:
                break
            end = begin + len(notice)
            head, tail = text[:begin], text[end:]
            if head.endswith((" ", "\t")) and tail[:1] in (" ", "\t"):
                tail = tail.lstrip(" \t")
            elif head.endswith((" ", "\t")) and tail[:1] == "\n":
                head = head.rstrip(" \t")
            text = head + tail
            removed.append(notice)
    return (text.strip() if removed else text), removed


#: How a notice spells a recovery control a player may then type. The control names
#: themselves live in ``narrator.channels.base.RECOVERY_CONTROLS``, which is the single
#: authority for the set; this only renders each one the way a notice writes it, with the
#: leading slash the channels parse. Deriving it means a control added there is covered
#: here without a second edit.
_ADVERTISED_CONTROLS: frozenset[str] = frozenset(
    f"/{name}" for name in _RECOVERY_CONTROLS
)


def advertises_recovery_controls(text: str) -> bool:
    """Whether one posted text tells a player to type a recovery control.
    """
    return any(control in text for control in _ADVERTISED_CONTROLS)


async def refresh_status(adapter, campaign_root) -> None:
    """Push a fresh status-line snapshot to a channel that renders one.

    Optional and best-effort, the same way ``present_decisions`` treats
    ``deliver_decision_views``: most adapters lack ``update_status`` entirely, and this
    carries mechanical state (``narrator.status``), never narration or the fiction-debt
    ledger, so it never gates or blocks a turn.
    """
    update_status = getattr(adapter, "update_status", None)
    if callable(update_status):
        await update_status(status.snapshot(campaign_root))


async def _post(adapter, channel_id: str, post: TurnPost, campaign_root) -> TurnPost:
    """Post one typed outbound message, refresh status, and arm advertised controls.

    The arming rule is typed rather than positional: only a ``notice`` post may arm,
    so model narration that happens to contain \"/retry\" can never arm a control. The
    pre-``TurnPost`` module enforced the same rule positionally, giving narration
    its own ``adapter.post`` call site that bypassed the notice path;
    ``TurnPost.kind`` now states it.
    """
    await adapter.post(channel_id, post.text)
    await refresh_status(adapter, campaign_root)
    if post.kind == "notice" and advertises_recovery_controls(post.text):
        recovery = getattr(adapter, "decision_recovery", None)
        if callable(recovery):
            await recovery()
    return post


async def deliver(adapter, turn, outcome: TurnOutcome, config) -> TurnPost:
    """Post exactly one message for this turn, or the stall notice, and return it.

    A withheld turn posts an engine-authored notice from configuration rather than
    silence, so a stalled table sees a diagnosable pause. No notice here can leak
    unratified fiction, because the model never wrote any of them. Three notices,
    selected here and nowhere else. ``fault_notice`` posts when this turn's story
    text is gone: the turn errored, or the reply scrubbed away to nothing — an audit
    proved the first draft routed that second state to the unsettled notice, which
    then claimed \"outcomes are not yet settled\" on a turn whose ledger had closed
    cleanly. ``withheld_notice`` posts only when the turn ran, produced text, and its
    outcomes genuinely never settled, so its claim is true everywhere it appears. The
    distinction lets a table and an operator tell a lost story from a stuck ledger
    without opening a log.

    Returns the ``TurnPost`` it posted, always with ``origin=\"turn\"``: this function
    is called exactly once per ``run_turn`` outcome and nowhere else, so the caller
    -- and every instrument reading a retained post log -- gets the provenance the
    retired ``engine_turn_posts`` reconstruction had to infer from text.
    """
    if outcome.decision_recovery:
        if _turn_changed_state(outcome):
            key, text = "decision_incomplete", config.catalog.notice("decision_incomplete")
        else:
            refusal = _refused_mechanic(outcome)
            if refusal:
                key = "decision_refused"
                text = config.catalog.render(
                    config.catalog.notice("decision_refused"), refusal=refusal
                )
            else:
                key, text = "decision_fault", config.decision_fault_notice
    elif outcome.error or (not outcome.withheld and not outcome.narration.strip()):
        key, text = "fault", config.fault_notice
    elif outcome.withheld:
        key, text = "withheld", config.withheld_notice
    else:
        return await _post(
            adapter,
            turn.channel_id,
            TurnPost(kind="narration", origin="turn", model_text=outcome.narration),
            config.campaign_root,
        )
    return await _post(
        adapter,
        turn.channel_id,
        TurnPost(
            kind="notice",
            origin="turn",
            engine_lines=outcome.mechanical_lines,
            notice_key=key,
            notice_text=text,
        ),
        config.campaign_root,
    )


async def present_decisions(adapter, views, campaign_root) -> DecisionPhaseOutcome:
    """Gate and render assignment views without reading or writing campaign state.

    A request is never exposed while the existing delivery gate cannot establish a
    settled ledger and no open ruling.  The adapter receives only renderer-safe views.
    """
    if not ledger.is_ratified(campaign_root):
        return DecisionPhaseOutcome(status="fault", failure_category="ledger_gate")
    if ledger.has_open_rulings(campaign_root):
        return DecisionPhaseOutcome(status="fault", failure_category="ruling_gate")
    from narrator.channels.base import DecisionDeliveryReceipt, structured_decision_capabilities

    if not structured_decision_capabilities(adapter).supports_structured_decisions():
        return DecisionPhaseOutcome(status="fault", failure_category="unsupported_channel")
    deliver_views = getattr(adapter, "deliver_decision_views", None)
    if not callable(deliver_views):
        return DecisionPhaseOutcome(status="fault", failure_category="unsupported_channel")
    try:
        receipt = await deliver_views(views)
    except Exception:  # noqa: BLE001 - presentation must fail closed
        return DecisionPhaseOutcome(status="fault", failure_category="presentation")
    if not isinstance(receipt, DecisionDeliveryReceipt):
        return DecisionPhaseOutcome(status="fault", failure_category="receipt")
    if receipt.status != "delivered" or receipt.decision_count != len(views):
        return DecisionPhaseOutcome(status="fault", failure_category="audience")
    return DecisionPhaseOutcome(
        status="presented", kind=views[0].kind if views else "", target_count=len(views)
    )


async def deliver_decision_fault(
    adapter, turn, config, notice: str | None = None, *, notice_key: str = ""
) -> TurnPost:
    """Post the engine-authored decision fault without routing it through narration.
    """
    text = notice if notice is not None else config.decision_fault_notice
    key = notice_key or ("decision_fault" if notice is None else "")
    return await _post(
        adapter,
        turn.channel_id,
        TurnPost(kind="notice", origin="service", notice_key=key, notice_text=text),
        config.campaign_root,
    )


async def deliver_decision_recovery(adapter, turn, config) -> TurnPost:
    """Post a no-action segment boundary without exposing planner diagnostics."""
    return await _post(
        adapter,
        turn.channel_id,
        TurnPost(
            kind="notice",
            origin="service",
            notice_key="decision_segment",
            notice_text=config.decision_segment_notice,
        ),
        config.campaign_root,
    )


async def deliver_decision_decline(adapter, turn, config) -> TurnPost:
    """Post the terminal no-action result of a declined risk confirmation.

    A decline is terminal, and ``decision_declined_notice`` names no control, so
    ``_post`` arms nothing here. The routing is still through it, so the rule stays
    "the wording decides" rather than "each caller decides".
    """
    return await _post(
        adapter,
        turn.channel_id,
        TurnPost(
            kind="notice",
            origin="service",
            notice_key="decision_declined",
            notice_text=config.decision_declined_notice,
        ),
        config.campaign_root,
    )


async def deliver_no_action(
    adapter, turn, notice: str, campaign_root: str = ".", *, notice_key: str = ""
) -> TurnPost:
    """Post one configuration-authored boundary notice without invoking the narrator."""
    return await _post(
        adapter,
        turn.channel_id,
        TurnPost(kind="notice", origin="service", notice_key=notice_key, notice_text=notice),
        campaign_root,
    )
