"""Drives one soak session: the replay adapter, the runner, and sandbox setup.

``_SoakAdapter`` and ``run_soak`` run the real ``NarratorService`` loop
against a generated transcript and assemble the ``SoakReport`` from
``narrator.soak_instruments``; ``bootstrap_sandbox`` creates the disposable
campaign and characters a run needs. ``scripts/soak_session.py`` is the CLI
entry point that calls this module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from narrator.channels.replay import TranscriptReplayAdapter
from narrator.config import DEFAULT_THINKING_BUDGETS, NarratorConfig
from narrator.delivery import scrub_markup
from narrator.engine import PLANNER_FAILURE_REASONS
from narrator.service import NarratorService
from narrator.soak_instruments import (
    AUTHORED_TOKENS,
    CONTINUITY_TOKENS,
    PLAYERS,
    REPO_ROOT,
    ZONE_A_TOKENS,
    ZONE_B_TOKENS,
    ZONE_C_TOKENS,
    SoakReport,
    _authored_canon_text,
    _canon_text,
    _clock_fill_state,
    authored_fact_stated,
    authored_quotation_spans,
    canon_entries,
    commit_discipline_gate,
    engine_turn_posts,
    find_raw_leaks,
    fixture_payloads,
    normalize_for_lenient_match,
    phantom_coin_claim_gate,
    phantom_hp_claim_gate,
    phantom_roll_claim_gate,
    read_engine_version,
    rest_declaration_turns,
    sample_cache_metrics,
    score_clock_recall,
    score_exchange,
    score_hazard_probe,
    score_instrument_blindness,
    score_order_recall,
    score_phantom_coin_claims,
    score_phantom_hp_claims,
    score_phantom_roll_claims,
    score_rest_handling,
    score_roll_announcements,
    score_thinking_fallbacks,
    score_zone_recall,
    server_root,
    summarize_cache,
    summarize_commit_discipline,
    summarize_payload,
)


class _SoakAdapter(TranscriptReplayAdapter):
    """The replay adapter plus the three measurements the report needs.

    ``close`` snapshots the conversation manager because the service loop awaits it
    before ``engine.stop()`` empties the agent cache. Reading the manager any later
    reads nothing.
    """

    def __init__(
        self,
        *args,
        report: SoakReport,
        engine_ref,
        service_ref=None,
        metrics_endpoint: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.report = report
        self.engine_ref = engine_ref
        #: The running service, for logs the service rather than the engine keeps
        #: (``_routing_log``). Optional so older call sites keep working.
        self.service_ref = service_ref
        #: Filled in ``close`` from the engine's per-turn digest texts, which the zone
        #: probe reads by turn index. It never enters the serialised report.
        self.digest_texts: list[str] = []
        #: The endpoint's Prometheus surface, empty when the caller suppresses the
        #: cache instrument. ``sample_cache`` is the only reader.
        self.metrics_endpoint = metrics_endpoint
        #: One entry per sampling point, each holding the turn it follows and that
        #: reading's counters. Turn 0 is the baseline the caller takes before the first
        #: turn, so a run of N turns produces N+1 entries and N per-turn deltas.
        self.cache_samples: list[dict] = []
        self._turn_started = 0.0
        self._turn_backfill = 0

    def sample_cache(self) -> None:
        """Record one counter reading against the turns completed so far.

        A refused reading enters the list with ``counters`` of None rather than dropping
        out of it, so the summary can state that a specific turn went unmeasured instead
        of silently attributing that turn's tokens to its neighbour.
        """
        if not self.metrics_endpoint:
            return
        sample = sample_cache_metrics(self.metrics_endpoint)
        self.cache_samples.append(
            {
                "turn": len(self.posted),
                "counters": sample["counters"] if sample else None,
                "missing": sample["missing"] if sample else [],
                "config": sample["config"] if sample else {},
            }
        )

    async def turns(self):
        async for turn in super().turns():
            self.report.fed_messages += len(turn.backfill) + 1
            # The replay grammar authors every mention "player", so authorship evidence
            # comes from the backfill lines, which carry the real speaker names.
            for message in turn.backfill:
                self.report.authors[message.author] = (
                    self.report.authors.get(message.author, 0) + 1
                )
            self._turn_backfill = len(turn.backfill)
            self._turn_started = time.monotonic()
            yield turn

    async def post(self, channel_id: str, text: str) -> None:
        await super().post(channel_id, text)
        self.report.per_turn.append(
            {
                "turn": len(self.posted),
                "backfill": self._turn_backfill,
                "latency_s": round(time.monotonic() - self._turn_started, 2),
                "posted_chars": len(text),


                "posted_excerpt": text[:800],
            }
        )
        # After the post, not before: the turn's model calls have all completed by the
        # time the adapter posts, so this reading closes that turn's window and opens
        # the next one. The latency figure above excludes it for the same reason.
        self.sample_cache()

    async def close(self) -> None:
        engine = self.engine_ref()
        digest_log = list(getattr(engine, "_digest_log", []))
        if digest_log:
            sections_per_turn = [entry["scene_sections"] for entry in digest_log]
            survival: dict[str, dict[str, int]] = {}
            for turn_sections in sections_per_turn:
                for section in turn_sections:
                    row = survival.setdefault(
                        section["name"], {"full": 0, "partial": 0, "absent": 0}
                    )
                    row[section["state"]] += 1
            self.report.canon = {
                "per_turn_total_chars": [entry["total_chars"] for entry in digest_log],
                "max_total_chars": max(entry["total_chars"] for entry in digest_log),
                "truncation_counts": {
                    block: sum(1 for entry in digest_log if entry["truncated"][block])
                    for block in ("scene", "location", "npcs")
                },


                "scene_section_survival": survival,
                # Retained visible facts against total, per turn. The list accrues
                # across the session and across locations, so this pair is the
                # measurement that says whether a fixed bound still fits it.
                "visible_facts_per_turn": [
                    [section["items_retained"], section["items"]]
                    for turn_sections in sections_per_turn
                    for section in turn_sections
                    if section["name"] == "Visible facts"
                ],
                # The location and present-NPC keys the digest resolved each turn. A
                # superseded digest carries the location file and NPC entries the
                # party had then, so removing those copies only removes canon a
                # session actually left behind when these keys change. A run whose
                # keys never change cannot exercise that path, and this states which.
                "location_ids": [entry["location_id"] for entry in digest_log],
                "distinct_location_ids": sorted(
                    {entry["location_id"] for entry in digest_log}
                ),
                "distinct_npc_sets": sorted(
                    {tuple(entry["present_npcs"]) for entry in digest_log}
                ),
                "scene_sections_per_turn": sections_per_turn,
            }
        self.digest_texts = list(getattr(engine, "_digest_texts", []))
        request_log = list(getattr(engine, "_request_log", []))
        if request_log:
            self.report.payload = summarize_payload(
                request_log, list(getattr(engine, "_usage_log", []))
            )
        tool_log = list(getattr(engine, "_tool_log", []))
        if tool_log:
            counts: dict[str, int] = {}
            for names in tool_log:
                for name in names:
                    counts[name] = counts.get(name, 0) + 1
            self.report.tools = {"per_turn": tool_log, "counts": counts}
        # The reasoning-replay instrument: same-name-same-arguments call attempts
        # within one turn, one count per turn (``NarratorEngine._duplicate_call_log``).
        # Recorded on both arms so the replay A/B and the baseline state the same
        # measurement; it gates nothing.
        duplicate_log = list(getattr(engine, "_duplicate_call_log", []))
        if duplicate_log:
            self.report.duplicate_tool_calls = {
                "per_turn": duplicate_log,
                "duplicate_calls": sum(duplicate_log),
                "turns_with_duplicates": sum(1 for count in duplicate_log if count),
            }
        # Thinking attempts re-run with thinking suppressed after hitting their answer
        # bound with no tool run (``NarratorEngine._invoke_channel_agent``): the turn
        # index, and whether the retry reached a tool at all. ``score_thinking_fallbacks``
        # reads the same per-turn tool log ``report.tools`` above publishes rather than
        # counting tools a second way; see its docstring for why the join is sound and
        # what a bare turn list hid.
        self.report.thinking_fallback_turns = score_thinking_fallbacks(
            list(getattr(engine, "_thinking_fallbacks", [])), tool_log
        )
        service = self.service_ref() if self.service_ref is not None else None
        routing_log = list(getattr(service, "_routing_log", []))
        if routing_log:
            from collections import Counter

            self.report.routing = {
                "per_turn": routing_log,
                "turns": len(routing_log),
                "risk_asks": sum(1 for row in routing_log if row["route"] == "risk"),
                "risk_asks_naming_unrecorded_person": sum(
                    1 for row in routing_log
                    if row["route"] == "risk" and row["names_unrecorded_person"]
                ),
                "turns_naming_unrecorded_person": sum(
                    1 for row in routing_log if row["names_unrecorded_person"]
                ),
                "interlocutor_kinds": dict(Counter(row["interlocutor_kind"] or "none" for row in routing_log)),
                # The trailing-question route defect's own reading: how many turns the
                # classifier said declared an act beside their question, and which tool
                # the framing then named for each. Without it a session where a rest
                # reached no tool cannot say whether the classifier missed the
                # declaration or the model ignored a correct one.
                "declared_act_turns": sum(1 for row in routing_log if row.get("also_declares_act")),
                "declared_act_kinds": dict(
                    Counter(
                        row.get("declared_act_kind") or "none"
                        for row in routing_log
                        if row.get("also_declares_act")
                    )
                ),
                "declared_act_tools": dict(
                    Counter(
                        row.get("declared_act_tool") or "none"
                        for row in routing_log
                        if row.get("also_declares_act")
                    )
                ),
            }
        sweep_log = list(getattr(engine, "_sweep_log", []))
        if sweep_log:
            self.report.sweep = {
                "per_turn": sweep_log,
                "counts": {
                    kind: sweep_log.count(kind)
                    for kind in ("record", "none", "failed", "skipped")
                },
            }
        recovery_log = list(getattr(engine, "_recovery_log", []))
        if any(entry.get("triggered") for entry in recovery_log):
            self.report.recovery = {
                "per_turn": recovery_log,
                "triggered": sum(1 for entry in recovery_log if entry.get("triggered")),
                "resolved": sum(
                    1 for entry in recovery_log if entry.get("resolved") is True
                ),
                "fell_back": sum(1 for entry in recovery_log if entry.get("fell_back")),
            }
        recent_narration_log = list(getattr(engine, "_planner_recent_narration_log", []))
        if recent_narration_log:
            nonzero = [n for n in recent_narration_log if n > 0]
            self.report.planner_recent_narration = {
                "plan_turn_calls": len(recent_narration_log),
                "calls_with_tail": len(nonzero),
                "max_chars": max(recent_narration_log),
                "mean_chars_when_present": round(sum(nonzero) / len(nonzero), 1) if nonzero else 0.0,
            }
        plan_log = list(getattr(engine, "_plan_log", []))
        if plan_log:
            promised = [entry for entry in plan_log if entry["promised_unbound_check"]]
            self.report.promised_unbound_checks = {
                "plan_turn_decisions": len(plan_log),
                "kind_counts": {
                    kind: sum(1 for entry in plan_log if entry["kind"] == kind)
                    for kind in (
                        # ``failed`` is not a decision: it is a planning attempt that
                        # produced none (``engine._plan_failure_entry``). Counted here
                        # so a session's planner faults are visible in the same row as
                        # its decisions rather than only in the total.
                        "proceed", "clarification", "confirmation", "approach", "declined",
                        "failed",
                    )
                },
                "failure_reasons": {
                    reason: sum(1 for entry in plan_log if entry.get("reason") == reason)
                    for reason in PLANNER_FAILURE_REASONS
                },
                "failure_dispositions": {
                    disposition: sum(
                        1 for entry in plan_log if entry.get("disposition") == disposition
                    )
                    for disposition in ("fault", "floor")
                },
                "promised_unbound_check_count": len(promised),
                "promised_unbound_check_kinds": {
                    kind: sum(1 for entry in promised if entry["kind"] == kind)
                    for kind in ("clarification", "confirmation")
                },
                "escalation_counts": {
                    disposition: sum(
                        1 for entry in promised if entry.get("escalation") == disposition
                    )
                    for disposition in ("bound", "collapsed", "failed")
                },
            }
        risk_branch_log = list(getattr(engine, "_risk_branch_log", []))
        if risk_branch_log:
            would_yield = [entry for entry in risk_branch_log if entry["would_yield"]]
            reached = [entry for entry in risk_branch_log if entry["reached_confirmation"]]
            self.report.risk_branch = {
                "risk_branch_entries": len(risk_branch_log),
                "policy_none_count": sum(1 for entry in risk_branch_log if entry["policy_none"]),
                "effective_from_label_count": sum(
                    1 for entry in risk_branch_log if entry["effective_from_label"]
                ),
                "would_yield_count": len(would_yield),
                "reached_confirmation_count": len(reached),
                # Would a speculative ``assess_hazard`` task started beside
                # ``verify_plan`` actually have overlapped with something AND been
                # consumed, versus dispatched then thrown away, versus never
                # dispatched at all (``verify_plan`` never yielded).
                "would_overlap_and_consumed_count": sum(
                    1 for entry in risk_branch_log
                    if entry["would_yield"] and entry["reached_confirmation"]
                ),
                "would_overlap_but_discarded_count": sum(
                    1 for entry in risk_branch_log
                    if entry["would_yield"] and not entry["reached_confirmation"]
                ),
            }
        for agent in getattr(engine, "_agents", {}).values():
            manager = agent.conversation_manager
            self.report.removed_message_count = int(
                getattr(manager, "removed_message_count", 0)
            )
            self.report.final_message_count = len(agent.messages)
        await super().close()


async def run_soak(
    campaign_root: Path,
    transcript: Path,
    window_size: int,
    report: SoakReport,
    plant_turn: int = 0,
    recall_turn: int = 0,
    authored_turn: int = 0,
    clock_turn: int = 0,
    zone_turns: dict | None = None,
    order_turns: dict | None = None,
    exchange_turns: dict | None = None,
    hazard_turns: dict | None = None,
    require_continuity: bool = False,
    measure_cache: bool = True,
    thinking_level: str = NarratorConfig.turn_thinking_level,
    thinking_budget_low: int | None = None,
) -> SoakReport:
    """The continuity probe scores two stages independently. ``committed``: every fact
    token appears in canon (``campaign/scene.md`` or ``campaign/state.json``). An
    audit established how it gets there: in all four evidence runs the model called
    ``scene_commit`` itself during the plant turn — every plant event carries
    ``ratified_seqs: []``, which a settle-path commit never does, and three carry
    ``hidden_changes``, a field the settler cannot pass. Voluntary facts reach canon
    by model discretion when asked; the settle path guarantees roll-debt only. ``recalled``: every fact token appears
    in the narrator's posted reply at the recall turn, whose mention deliberately
    names neither token. Recall without canon retrieval is impossible past the
    eviction boundary, because the plant text left the conversation window and the
    replay grammar keeps it out of the recall turn's backfill.

    Model recall is a judgment, not a mechanism, so it enters the exit code only when
    ``require_continuity`` is set; the measurement itself always lands in the report.

    ``measure_cache`` samples the endpoint's prefix cache counters at every turn
    boundary and fills ``report.cache``. It defaults on and gates nothing, because the
    counters are the endpoint's property rather than this session's: a second client
    inside the window moves them, and ``exclusive_window`` reports that instead of
    failing the run. Passing False suppresses every metrics request.
    """
    config = NarratorConfig(
        campaign_root=campaign_root,
        repo_root=REPO_ROOT,
        window_size=window_size,
        turn_thinking_level=thinking_level,
        **(
            {"turn_thinking_budgets": {**DEFAULT_THINKING_BUDGETS, "low": thinking_budget_low}}
            if thinking_budget_low
            else {}
        ),
    )
    # Recorded on the report so two runs compared on the phantom gates state which
    # level each ran at; an attribute rather than a dataclass field so every retained
    # report written before the level existed still loads unchanged.
    report.thinking_level = thinking_level
    report.thinking_budgets = dict(config.turn_thinking_budgets)
    service_box: list[NarratorService] = []
    metrics_endpoint = f"{server_root(config.base_url)}/metrics" if measure_cache else ""
    adapter = _SoakAdapter(
        transcript,
        backfill_limit=config.backfill_limit,
        echo=False,
        report=report,
        engine_ref=lambda: service_box[0].engine,
        service_ref=lambda: service_box[0],
        metrics_endpoint=metrics_endpoint,
    )
    service = NarratorService(config, adapter)
    service_box.append(service)

    # The baseline reading, taken before the first turn sends anything. Every counter
    # this instrument reads is process-global and monotonic, so the run's consumption is
    # this reading subtracted from the last one.
    adapter.sample_cache()
    engine_version = read_engine_version(config.base_url) if measure_cache else ""

    outcome = await service.run()

    report.turns = outcome.turns
    report.delivered = outcome.delivered
    report.withheld = outcome.withheld
    report.settle_attempts = outcome.settle_attempts
    report.commits = outcome.commits
    report.waives = outcome.waives
    report.leaks_scrubbed = outcome.leaks_scrubbed
    report.errors = list(outcome.errors)
    if adapter.cache_samples:
        # The request count comes from the payload summary rather than from the turn
        # count, because one turn assembles a turn request plus any settle and sweep
        # requests. Comparing that total against the endpoint's own completion count is
        # what states whether this session held the endpoint alone.
        report.cache = summarize_cache(
            adapter.cache_samples,
            int((report.payload or {}).get("requests", 0)),
            (report.payload or {}).get("usage_totals", {}),
            metrics_endpoint,
            next(
                (
                    sample["config"]
                    for sample in adapter.cache_samples
                    if sample.get("config")
                ),
                {},
            ),
            engine_version,
        )
    report.compaction_triggered = (
        report.removed_message_count > 0
        and report.final_message_count <= window_size
    )

    report.check(
        "two-players",
        set(report.authors) == set(PLAYERS),
        f"authors seen: {sorted(report.authors)}",
    )
    report.check(
        "volume",
        report.fed_messages >= 1000,
        f"{report.fed_messages} player messages reached the narrator",
    )
    report.check(
        "compaction",
        report.compaction_triggered,
        f"removed {report.removed_message_count}, "
        f"final window {report.final_message_count} of {window_size}",
    )
    report.check("zero-errors", not report.errors, f"{len(report.errors)} turn errors")
    report.check(
        "one-post-per-turn",
        len(adapter.posted) == report.turns,
        f"{len(adapter.posted)} posts for {report.turns} turns",
    )
    residue = [text for text in adapter.posted if scrub_markup(text)[1]]
    report.check(
        "no-markup-delivered",
        not residue,
        f"{len(residue)} posted messages still carried markup",
    )


    leak_detail = [
        {"excerpt": text[:200], "matches": leaks}
        for text in adapter.posted
        for leaks in (find_raw_leaks(text),)
        if leaks
    ]
    report.raw_leaks = {
        "posts_scanned": len(adapter.posted),
        "posts_with_leaks": len(leak_detail),
        "detail": leak_detail,
    }
    report.check(
        "no-raw-leaks-delivered",
        not leak_detail,
        f"{len(leak_detail)} of {len(adapter.posted)} posted messages carried raw "
        "error text or unstripped markup",
    )


    if report.tools:
        turn_tool_log = report.tools.get("per_turn", [])
        aligned_posts = engine_turn_posts(service_box[0].post_log if service_box else [])
        if len(aligned_posts) != len(turn_tool_log):


            error = (
                f"{len(aligned_posts)} engine-backed posts for {len(turn_tool_log)} "
                "run_turn tool-log entries; the post accounting is defective"
            )
            report.dice_visibility = {"scorable": False, "reason": error}
            report.check("roll-turns-announce-outcome", False, f"unscorable: {error}")

            report.check("phantom-roll-claims", False, f"unscorable: {error}")

            report.check("phantom-hp-claims", False, f"unscorable: {error}")


            report.check("phantom-coin-claims", False, f"unscorable: {error}")


            report.rest_handling = {"scorable": False, "reason": error}
        else:
            report.dice_visibility = score_roll_announcements(
                turn_tool_log, aligned_posts, config.catalog
            )
            report.dice_visibility["scorable"] = True
            roll_turn_count = report.dice_visibility["roll_turn_count"]
            unannounced = report.dice_visibility["unannounced_turns"]
            announced_count = roll_turn_count - len(unannounced)
            # The gate gates on "at least one roll-under turn is confirmed announcing
            # its outcome in the policy's shape" -- a run with zero roll turns, or
            # where every roll turn stayed silent, fails here, which is what keeps the
            # check from passing vacuously. It does not gate on every roll turn
            # announcing: model-authored prose is a judgment call, not a mechanical
            # guarantee this project's skill-level policy alone can enforce, and a
            # live 42-turn run at seed 1234 measured partial compliance (some roll
            # turns announce, some do not) across several rounds of policy wording --
            # the exact, honest rate rides in this field's
            # roll_turn_count/unannounced_turns for anyone auditing the policy's
            # real-world reliability rather than only its existence.
            report.check(
                "roll-turns-announce-outcome",
                announced_count > 0,
                f"{roll_turn_count} roll-under turns, {announced_count} announced, "
                f"{len(unannounced)} unannounced (turns {unannounced})" if roll_turn_count
                else "zero roll-under turns occurred; the run never exercised a scorable roll",
            )


            report.dice_visibility.update(
                score_phantom_roll_claims(turn_tool_log, aligned_posts, config.catalog)
            )
            report.instrument_blindness = score_instrument_blindness(aligned_posts, config.catalog)
            phantom_passed, phantom_detail = phantom_roll_claim_gate(report.dice_visibility)
            report.check("phantom-roll-claims", phantom_passed, phantom_detail)


            report.dice_visibility.update(score_phantom_hp_claims(turn_tool_log, aligned_posts))
            hp_phantom_passed, hp_phantom_detail = phantom_hp_claim_gate(report.dice_visibility)
            report.check("phantom-hp-claims", hp_phantom_passed, hp_phantom_detail)


            report.dice_visibility.update(score_phantom_coin_claims(turn_tool_log, aligned_posts))
            coin_phantom_passed, coin_phantom_detail = phantom_coin_claim_gate(report.dice_visibility)
            report.check("phantom-coin-claims", coin_phantom_passed, coin_phantom_detail)


            claimed_turns = {t for t in (plant_turn, recall_turn, authored_turn, clock_turn) if t}
            claimed_turns |= {t for t in (zone_turns or {}).values() if t}
            claimed_turns |= {t for t in (order_turns or {}).values() if t}
            claimed_turns |= {t for t in (exchange_turns or {}).values() if t}
            claimed_turns |= {t for t in (hazard_turns or {}).values() if t}
            rest_turns = rest_declaration_turns(len(adapter.posted), claimed_turns)
            report.rest_handling = score_rest_handling(
                adapter.posted, rest_turns, turn_tool_log, aligned_posts
            )
    # A withheld turn must post an engine-authored notice, never model text.
    # ``TurnPost`` records the fact at the boundary: an engine-backed post whose
    # ``kind`` is "notice" is exactly "``deliver`` posted a notice for a ``run_turn``
    # outcome" -- including the decision-recovery notices the retired two-string text
    # match (``{config.withheld_notice, config.fault_notice}``) could not see, and
    # unmoved by a mid-session ``/language`` switch that re-renders the catalog.
    notice_posts = sum(
        1
        for post in (service_box[0].post_log if service_box else [])
        if post.origin == "turn" and post.kind == "notice"
    )
    report.check(
        "withheld-posts-are-a-notice",
        notice_posts >= report.withheld,
        f"{notice_posts} engine-notice posts for {report.withheld} withheld turns",
    )
    if report.payload:
        # The dedup mechanism's own guarantee, read off the assembled requests rather
        # than off the history: one request carries the current digest and no other.
        # The miniature soak in tests_narrator/test_soak_live.py runs this check on
        # every validation pass, so a regression fails there instead of in evidence.
        blocks = report.payload["chars"]["canon_blocks_max"]
        report.check(
            "one-canon-block-per-request",
            blocks <= 1,
            f"at most {blocks} canon digests in one request, "
            f"{report.payload['chars']['canon_superseded_chars_max']} superseded chars",
        )

    if authored_turn:
        posted_reply = (
            adapter.posted[authored_turn - 1] if len(adapter.posted) >= authored_turn else ""
        )
        reply = posted_reply.lower()
        found = [token for token in AUTHORED_TOKENS if token in reply]
        lenient_reply = normalize_for_lenient_match(reply)
        lenient_found = [
            token
            for token in AUTHORED_TOKENS
            if normalize_for_lenient_match(token) in lenient_reply
        ]
        states_fact = authored_fact_stated(reply)
        report.continuity_authored = {
            "authored_turn": authored_turn,
            "fact_tokens": list(AUTHORED_TOKENS),
            "recalled": len(found) == len(AUTHORED_TOKENS),
            "tokens_in_reply": found,
            # Whether the reply states the fact, in any of the renderings recorded
            # beside AUTHORED_SUNSET_TERMS. This is the channel the gate reads, because
            # a narrator that answers correctly in its own words has not lost canon.
            # ``recalled`` above keeps measuring quotation fidelity, which is a
            # separate property worth watching: both replies that state the fact
            # without the token presented invented wording as a verbatim quotation of
            # an authored notice.
            "states_fact": states_fact,
            # The lenient channel separates formatting drift from loss in the
            # retained report. It gates nothing.
            "lenient": {
                "recalled": len(lenient_found) == len(AUTHORED_TOKENS),
                "tokens_in_reply": lenient_found,
            },


            "quotation": authored_quotation_spans(
                posted_reply, _authored_canon_text(REPO_ROOT)
            ),
            "reply_excerpt": posted_reply[:600],
        }
        if require_continuity:
            report.check(
                "continuity-authored",
                report.continuity_authored["states_fact"],
                f"states_fact={states_fact} verbatim={report.continuity_authored['recalled']} "
                f"tokens_in_reply={found} lenient={lenient_found}",
            )

    if zone_turns:
        canon_text = _canon_text(campaign_root)
        tool_log = (report.tools or {}).get("per_turn", [])
        report.zones = {}
        for zone, tokens, recall_key in (
            ("B_outside_window_inside_digest", ZONE_B_TOKENS, "zone_b_recall"),
            ("C_outside_both", ZONE_C_TOKENS, "zone_c_recall"),
            ("A_inside_window", ZONE_A_TOKENS, "zone_a_recall"),
        ):
            turn = zone_turns.get(recall_key, 0)
            if not turn:
                continue
            reply = adapter.posted[turn - 1] if len(adapter.posted) >= turn else ""
            # The scored fact's own token settles digest presence: per-section counts
            # state how many entries survived, never which ones.
            texts = adapter.digest_texts
            in_digest = (
                all(token in texts[turn - 1].lower() for token in tokens)
                if len(texts) >= turn
                else False
            )
            digest_turns = sum(
                1 for text in texts if all(token in text.lower() for token in tokens)
            )
            report.zones[zone] = score_zone_recall(
                zone,
                tokens,
                reply,
                canon_text,
                in_digest,
                digest_turns,
                tool_log[turn - 1] if len(tool_log) >= turn else [],
                canon_entries(campaign_root),
            )
            report.zones[zone]["recall_turn"] = turn
            report.zones[zone]["plant_turn"] = zone_turns.get(
                recall_key.replace("_recall", "_plant"), 0
            )

    if order_turns and order_turns.get("order_recall"):
        recall = order_turns["order_recall"]
        tool_log = (report.tools or {}).get("per_turn", [])
        report.order = score_order_recall(
            adapter.posted[recall - 1] if len(adapter.posted) >= recall else "",
            _canon_text(campaign_root),
            adapter.digest_texts,
            recall,
            tool_log[recall - 1] if len(tool_log) >= recall else [],
            canon_entries(campaign_root),
        )
        report.order["turns"] = dict(order_turns)

    # Every in-session plant enters the commit-discipline record, whichever probe put it
    # there. Guarding this on the ordering probe left a zone-only run reporting nothing
    # for its two plants, which the option list at ../README.md#soaking-a-long-session
    # documents as an independent choice.
    if exchange_turns:
        report.exchange = score_exchange(canon_entries(campaign_root), exchange_turns)
    if hazard_turns:
        report.hazard = score_hazard_probe(
            report.routing, service_box[0].post_log if service_box else [], hazard_turns
        )

    if order_turns or zone_turns or exchange_turns:
        report.commit_discipline = summarize_commit_discipline(
            campaign_root, report, adapter.posted, order_turns, zone_turns, exchange_turns
        )
        if require_continuity:
            passed, detail = commit_discipline_gate(report.commit_discipline)
            report.check("commit-discipline", passed, detail)

    if clock_turn:
        reply = (
            adapter.posted[clock_turn - 1]
            if len(adapter.posted) >= clock_turn
            else ""
        )
        canon_text = _canon_text(campaign_root)
        sections = (report.canon or {}).get("scene_sections_per_turn", [])
        section_state = "unmeasured"
        if len(sections) >= clock_turn:
            section_state = next(
                (
                    section["state"]
                    for section in sections[clock_turn - 1]
                    if section["name"] == "Clocks"
                ),
                "unmeasured",
            )
        tool_log = (report.tools or {}).get("per_turn", [])
        tool_names = tool_log[clock_turn - 1] if len(tool_log) >= clock_turn else []
        report.clock = score_clock_recall(
            reply, canon_text, _clock_fill_state(campaign_root), section_state, tool_names
        )
        report.clock["clock_turn"] = clock_turn
        if require_continuity:
            # Slice 1 measured this and gated nothing, because a miss was the finding
            # that decided whether a mechanism shipped. The mechanism shipped, so the
            # measurement becomes a requirement: the Clocks section must survive every
            # cut, and a miss now fails the run.
            report.check(
                "continuity-clock",
                report.clock["recalled"],
                f"recalled={report.clock['recalled']} "
                f"lenient={report.clock['lenient']['recalled']} "
                f"section={report.clock['clocks_section_state']} "
                f"in_canon={report.clock['present_in_canon']} "
                f"read_tool={report.clock['read_tool_called']}",
            )

    if plant_turn and recall_turn:
        canon = _canon_text(campaign_root).lower()
        committed = all(token in canon for token in CONTINUITY_TOKENS)
        reply = (
            adapter.posted[recall_turn - 1].lower()
            if len(adapter.posted) >= recall_turn
            else ""
        )
        found = [token for token in CONTINUITY_TOKENS if token in reply]
        recalled = len(found) == len(CONTINUITY_TOKENS)
        lenient_canon = normalize_for_lenient_match(canon)
        lenient_committed = all(
            normalize_for_lenient_match(token) in lenient_canon
            for token in CONTINUITY_TOKENS
        )
        lenient_reply = normalize_for_lenient_match(reply)
        lenient_found = [
            token
            for token in CONTINUITY_TOKENS
            if normalize_for_lenient_match(token) in lenient_reply
        ]
        report.continuity = {
            "plant_turn": plant_turn,
            "recall_turn": recall_turn,
            "fact_tokens": list(CONTINUITY_TOKENS),
            "committed": committed,
            "recalled": recalled,
            "tokens_in_reply": found,
            # The lenient channel covers the write side too: a model that files
            # the fact unhyphenated would fail strict `committed` while the fact
            # sits in canon. It gates nothing.
            "lenient": {
                "committed": lenient_committed,
                "recalled": len(lenient_found) == len(CONTINUITY_TOKENS),
                "tokens_in_reply": lenient_found,
            },
            "recall_reply_chars": len(reply),
            # The first measurement produced a partial recall whose reply was not
            # retained, so the miss could not be classified as place-loss or
            # paraphrase. The excerpt makes every future miss classifiable.
            "recall_reply_excerpt": adapter.posted[recall_turn - 1][:600]
            if len(adapter.posted) >= recall_turn
            else "",
        }
        if require_continuity:
            report.check(
                "continuity",
                committed and recalled,
                f"committed={committed} recalled={recalled} tokens_in_reply={found} "
                f"lenient_committed={lenient_committed} lenient_tokens={lenient_found}",
            )
    return report


def bootstrap_sandbox(
    server_python: str,
    sandbox: Path,
    prefill_facts: int = 0,
    clock: bool = False,
    zone: bool = False,
    order: bool = False,
) -> None:
    """Create a campaign with both players, using the interpreter that holds mcp 2.x.

    With ``prefill_facts`` at 0 and ``clock`` false, this runs exactly the two
    subprocesses every recorded soak ran, so the default sandbox stays identical to
    the evidence path. Each fixture adds one further ``scene_commit`` call.
    """
    for directory in ("rules", "world"):
        shutil.copytree(REPO_ROOT / directory, sandbox / directory)
    subprocess.run(
        [server_python, str(REPO_ROOT / "scripts" / "new_campaign.py"),
         "--root", str(sandbox), "--title", "Soak session", "--seed-scene"],
        check=True, capture_output=True, cwd=str(REPO_ROOT),
    )
    creation = (
        "import sys; sys.path.insert(0, 'src')\n"
        "from bsh_mcp.service import build_service\n"
        f"service = build_service({str(sandbox)!r})\n"
        "party = [\n"
        "    ('player-rill', 'Rill', 'barbarian', ['scout', 'hunter', 'survivor'],"
        " ['hunting bow', 'long knife'], 'light'),\n"
        "    ('player-ossa', 'Ossa', 'civilised', ['street-urchin', 'diplomat', 'bookworm'],"
        " ['duelling dagger'], 'none'),\n"
        "]\n"
        "for account, name, origin, backgrounds, weapons, armour in party:\n"
        "    result = service.character_create(account, name, origin, backgrounds,"
        " weapons=weapons, armour=armour)\n"
        "    assert result['ok'], result['message']\n"
    )
    subprocess.run(
        [server_python, "-c", creation], check=True, capture_output=True, cwd=str(REPO_ROOT)
    )

    payloads = fixture_payloads(prefill_facts, clock, zone, order)
    if not payloads:
        return
    fixture = (
        "import json, sys; sys.path.insert(0, 'src')\n"
        "from bsh_mcp.service import build_service\n"
        f"service = build_service({str(sandbox)!r})\n"
        f"for payload in json.loads({json.dumps(json.dumps(payloads))}):\n"
        "    result = service.scene_commit(**payload)\n"
        "    assert result['ok'], result['message']\n"
    )
    subprocess.run(
        [server_python, "-c", fixture], check=True, capture_output=True, cwd=str(REPO_ROOT)
    )
