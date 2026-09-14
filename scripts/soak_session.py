#!/usr/bin/env python3
"""Soak the narrator with an extended two-player session and prove compaction fires.

The session shape is deterministic. ``generate_transcript`` emits blocks of chatter
between two players, Rill and Ossa, each block closed by one ``@GM`` mention, seeded so
the same arguments always produce the same 1,050 lines. Block size stays under the
50-message backfill cap in ``NarratorConfig.backfill_limit``, so every generated player
message reaches the narrator rather than falling off the buffer.

The runner drives the real ``NarratorService`` loop, unmodified. Instrumentation lives
in a subclass of the replay adapter, because of an ordering fact in
``NarratorService.run``: the ``finally`` block awaits ``adapter.close()`` before
``engine.stop()`` clears the agent cache. ``close`` is therefore the one moment a
harness can read ``conversation_manager.removed_message_count`` from the live agent
without touching ``src/narrator``.

Compaction here means eviction by ``SlidingWindowConversationManager``: Strands
increments ``removed_message_count`` each time ``reduce_context`` or window
enforcement drops messages. The check demands ``removed_message_count > 0`` and a
final message list at or under ``window_size``. Both numbers enter the report, so the
evidence records how far past the boundary the session ran, not only that it crossed.

Run it against a disposable campaign root:

    export BSH_SERVER_PYTHON=\"$(uv run python -c 'import sys; print(sys.executable)')\"
    .narrator-venv/bin/python scripts/soak_session.py         --bootstrap --messages 1050 --report /tmp/soak-report.json

``--bootstrap`` shells out to ``BSH_SERVER_PYTHON`` to create the campaign and both
characters, because ``bsh_mcp`` needs ``mcp`` 2.x and this interpreter holds 1.29.0 for
``strands-agents``. The endpoint must already serve the model; do not start it without
authorization.

``--exchange-probe`` serves the same item from the other side. It passes one object
between the two players three times, so the first and third passes file near-identical
record entries while the second reverses them. A mechanism that drops a restatement as a
duplicate drops the third pass, and the record's newest surviving statement then names
the wrong holder. The probe scores no recall, and its own ordered-holder reading
(``score_exchange``) gates nothing: it retains the record's statements about that
object in record order, and a reader classifies them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.config import THINKING_LEVELS, NarratorConfig  # noqa: E402
from narrator.soak_harness import bootstrap_sandbox, run_soak  # noqa: E402
from narrator.soak_instruments import SoakReport, generate_transcript  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="", help="prepared campaign root; empty with --bootstrap creates one")
    parser.add_argument("--bootstrap", action="store_true", help="create a sandbox campaign with Rill and Ossa")
    parser.add_argument("--messages", type=int, default=1050, help="player messages to generate")
    parser.add_argument("--block", type=int, default=25, help="player messages per @GM mention")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--window-size", type=int, default=NarratorConfig.window_size)
    parser.add_argument(
        "--thinking", choices=THINKING_LEVELS, default=NarratorConfig.turn_thinking_level,
        help="the channel agent's thinking level for this run (NarratorConfig.turn_thinking_level)",
    )
    parser.add_argument("--report", default="", help="write the JSON report to this path")
    parser.add_argument(
        "--continuity", action="store_true",
        help="plant a fact early and demand its recall late; see CONTINUITY_TOKENS",
    )
    parser.add_argument("--plant-turn", type=int, default=3)
    parser.add_argument(
        "--recall-turn", type=int, default=0,
        help="0 places the recall two mentions before the end",
    )
    parser.add_argument(
        "--continuity-authored", action="store_true",
        help="score recall of authored location details; plants nothing",
    )
    parser.add_argument("--authored-turn", type=int, default=2)
    parser.add_argument(
        "--clock-probe", action="store_true",
        help="plant one clock at bootstrap and score a token-free recall late; see CLOCK_TOKENS",
    )
    parser.add_argument(
        "--clock-turn", type=int, default=0,
        help="0 places the clock question one mention before the end",
    )
    parser.add_argument(
        "--prefill-facts", type=int, default=0,
        help="visible facts written into the record at bootstrap; 105 saturates the shipped bound",
    )
    parser.add_argument(
        "--zone-probe", action="store_true",
        help="score recall at three fact ages: inside the window, digest-only, and neither",
    )
    parser.add_argument(
        "--order-probe", action="store_true",
        help="move one object through three places and score ordered recall; see ORDER_PLACES",
    )
    parser.add_argument(
        "--order-move-first", type=int, default=12,
        help="turn moving the object the first time; it must fall outside the final window",
    )
    parser.add_argument(
        "--order-move-second", type=int, default=22,
        help="turn moving the object the second time; it must fall outside the final window",
    )
    parser.add_argument(
        "--order-recall", type=int, default=0,
        help="0 places the ordering question five mentions before the end",
    )
    parser.add_argument(
        "--exchange-probe", action="store_true",
        help="pass one object between the two players three times; see EXCHANGE_OBJECT",
    )
    parser.add_argument(
        "--exchange-give-first", type=int, default=15,
        help="turn passing the object from the giver to the taker",
    )
    parser.add_argument(
        "--exchange-return", type=int, default=25,
        help="turn passing the object back to the giver",
    )
    parser.add_argument(
        "--exchange-give-second", type=int, default=33,
        help="turn passing the object to the taker a second time; it repeats leg one",
    )
    parser.add_argument(
        "--hazard-probe", action="store_true",
        help=(
            "plant three hazard declarations naming people the record holds no NPC for, "
            "so the routing instrument's 'risk asks naming an unrecorded person' reads "
            "from a session that declares hazards; see HAZARD_PLANTS; gates nothing"
        ),
    )
    parser.add_argument("--hazard-strike-person", type=int, default=9, help="turn striking the seeded authored person")
    parser.add_argument("--hazard-strike-unrecorded", type=int, default=21, help="turn striking an unrecorded role")
    parser.add_argument("--hazard-loot-corpse", type=int, default=36, help="turn looting a corpse nobody rolled dead")
    parser.add_argument(
        "--zone-c-plant", type=int, default=20,
        help="turn planting the digest-only fact; it must fall outside the final window",
    )
    parser.add_argument(
        "--require-continuity", action="store_true",
        help=(
            "fail the exit code when an enabled recall mode misses, or, if "
            "--order-probe/--zone-probe/--exchange-probe is also passed, when a "
            "plant one of them placed never reached canon; off, probes only report"
        ),
    )
    parser.add_argument(
        "--thinking-budget-low", type=int, default=None,
        help=(
            "override the low thinking level's token budget for this run "
            "(NarratorConfig.turn_thinking_budgets; default from DEFAULT_THINKING_BUDGETS)"
        ),
    )
    arguments = parser.parse_args()

    import os

    server_python = os.environ.get("BSH_SERVER_PYTHON", "")
    if not server_python or shutil.which(server_python) is None:
        print(
            "error: BSH_SERVER_PYTHON must name an executable interpreter holding mcp 2.x.\n"
            "  export BSH_SERVER_PYTHON=\"$(uv run python -c 'import sys; print(sys.executable)')\"",
            file=sys.stderr,
        )
        return 2
    if arguments.block > NarratorConfig.backfill_limit + 1:
        print(
            f"error: --block {arguments.block} exceeds the backfill cap of "
            f"{NarratorConfig.backfill_limit}; generated chatter would be dropped unfed",
            file=sys.stderr,
        )
        return 2

    if arguments.root:
        sandbox = Path(arguments.root).resolve()
    elif arguments.bootstrap:
        sandbox = Path(tempfile.mkdtemp(prefix="bsh-soak-"))
    else:
        print("error: pass --root or --bootstrap", file=sys.stderr)
        return 2
    if arguments.prefill_facts < 0:
        print("error: --prefill-facts cannot be negative", file=sys.stderr)
        return 2
    if arguments.zone_probe and arguments.continuity:
        print(
            "error: --zone-probe supersedes --continuity; the zone probe scores the "
            "continuity probe's one zone plus two the continuity probe cannot reach",
            file=sys.stderr,
        )
        return 2
    if arguments.bootstrap:
        bootstrap_sandbox(
            server_python,
            sandbox,
            prefill_facts=arguments.prefill_facts,
            clock=arguments.clock_probe,
            zone=arguments.zone_probe,
            order=arguments.order_probe,
        )
    elif (
        arguments.prefill_facts
        or arguments.clock_probe
        or arguments.zone_probe
        or arguments.order_probe
    ):
        print(
            "error: --prefill-facts, --clock-probe, --zone-probe, and --order-probe "
            "write their fixtures during --bootstrap; pass --bootstrap or prepare the "
            "record yourself",
            file=sys.stderr,
        )
        return 2

    total_mentions = arguments.messages // arguments.block
    plant_turn = recall_turn = authored_turn = clock_turn = 0
    # --require-continuity implies the episodic probe on its own, which is how every
    # run before the zone probe enabled it. With --zone-probe set it stays a pure
    # gating flag, because the zone probe already scores that probe's one zone.
    if arguments.continuity or (arguments.require_continuity and not arguments.zone_probe):
        plant_turn = arguments.plant_turn
        recall_turn = arguments.recall_turn or max(plant_turn + 2, total_mentions - 2)
    if arguments.continuity_authored:
        authored_turn = arguments.authored_turn
    zone_turns: dict = {}
    if arguments.zone_probe:
        # Zone B plants at bootstrap, so it claims no mention. The three recalls sit
        # last, and zone A plants immediately before them, which is what places its
        # fact inside the 40-message window at its own recall turn.
        zone_turns = {
            "zone_c_plant": arguments.zone_c_plant,
            "zone_a_plant": total_mentions - 4,
            "zone_b_recall": total_mentions - 3,
            "zone_c_recall": total_mentions - 2,
            "zone_a_recall": total_mentions - 1,
        }
        if min(zone_turns.values()) < 1:
            print(
                f"error: {total_mentions} mentions cannot hold the zone probe's five "
                "turns; raise --messages or lower --block",
                file=sys.stderr,
            )
            return 2
    order_turns: dict = {}
    if arguments.order_probe:
        # The first place plants at bootstrap, so it claims no mention. Both moves sit
        # far enough before the recall that the conversation window no longer holds
        # either at scoring time, which is what makes the record the only carrier.
        order_turns = {
            "order_move_first": arguments.order_move_first,
            "order_move_second": arguments.order_move_second,
            "order_recall": arguments.order_recall or total_mentions - 5,
        }
        if min(order_turns.values()) < 1:
            print(
                f"error: {total_mentions} mentions cannot hold the ordering probe's "
                "three turns; raise --messages or lower --block",
                file=sys.stderr,
            )
            return 2
        if not (
            order_turns["order_move_first"]
            < order_turns["order_move_second"]
            < order_turns["order_recall"]
        ):
            print(
                f"error: the ordering probe needs move {order_turns['order_move_first']} "
                f"before move {order_turns['order_move_second']} before recall "
                f"{order_turns['order_recall']}",
                file=sys.stderr,
            )
            return 2
    exchange_turns: dict = {}
    if arguments.exchange_probe:
        # No bootstrap fixture: all three legs plant by mention, because the first leg
        # is the pass that the third repeats, and a bootstrap plant would sit at the
        # head of the visible-facts list where no eviction reaches it.
        exchange_turns = {
            "exchange_give_first": arguments.exchange_give_first,
            "exchange_return": arguments.exchange_return,
            "exchange_give_second": arguments.exchange_give_second,
        }
        if min(exchange_turns.values()) < 1:
            print(
                f"error: {total_mentions} mentions cannot hold the exchange probe's "
                "three turns; raise --messages or lower --block",
                file=sys.stderr,
            )
            return 2
        if not (
            exchange_turns["exchange_give_first"]
            < exchange_turns["exchange_return"]
            < exchange_turns["exchange_give_second"]
        ):
            print(
                f"error: the exchange probe needs pass "
                f"{exchange_turns['exchange_give_first']} before return "
                f"{exchange_turns['exchange_return']} before pass "
                f"{exchange_turns['exchange_give_second']}",
                file=sys.stderr,
            )
            return 2
    hazard_turns: dict = {}
    if arguments.hazard_probe:
        hazard_turns = {
            "hazard_strike_person": arguments.hazard_strike_person,
            "hazard_strike_unrecorded": arguments.hazard_strike_unrecorded,
            "hazard_loot_corpse": arguments.hazard_loot_corpse,
        }
        if min(hazard_turns.values()) < 1 or max(hazard_turns.values()) > total_mentions:
            print(
                f"error: {total_mentions} mentions cannot hold the hazard probe's three "
                "turns; raise --messages, lower --block, or move the turns",
                file=sys.stderr,
            )
            return 2
    if arguments.clock_probe:
        default_clock = total_mentions if zone_turns else max(2, total_mentions - 1)
        clock_turn = arguments.clock_turn or default_clock
    assigned = [
        turn
        for turn in (
            plant_turn,
            recall_turn,
            authored_turn,
            clock_turn,
            *zone_turns.values(),
            *order_turns.values(),
            *exchange_turns.values(),
            *hazard_turns.values(),
        )
        if turn
    ]
    if len(set(assigned)) != len(assigned):
        print(
            f"error: two probes claim one mention: plant={plant_turn} "
            f"recall={recall_turn} authored={authored_turn} clock={clock_turn} "
            f"zones={zone_turns} order={order_turns} exchange={exchange_turns} hazard={hazard_turns}",
            file=sys.stderr,
        )
        return 2

    transcript = sandbox / "soak-transcript.txt"
    text = generate_transcript(
        arguments.messages, arguments.block, arguments.seed,
        plant_turn=plant_turn, recall_turn=recall_turn, authored_turn=authored_turn,
        clock_turn=clock_turn, zone_turns=zone_turns, order_turns=order_turns,
        exchange_turns=exchange_turns, hazard_turns=hazard_turns,
    )
    transcript.write_text(text, encoding="utf-8")

    report = SoakReport(
        seed=arguments.seed,
        window_size=arguments.window_size,
        generated_messages=arguments.messages,
    )
    asyncio.run(run_soak(
        sandbox, transcript, arguments.window_size, report,
        plant_turn=plant_turn, recall_turn=recall_turn, authored_turn=authored_turn,
        clock_turn=clock_turn, zone_turns=zone_turns, order_turns=order_turns,
        exchange_turns=exchange_turns,
        hazard_turns=hazard_turns or None,
        require_continuity=arguments.require_continuity,
        thinking_level=arguments.thinking,
        thinking_budget_low=arguments.thinking_budget_low,
    ))

    payload = json.dumps(report.__dict__, indent=2)
    if arguments.report:
        Path(arguments.report).write_text(payload + "\n", encoding="utf-8")
    print(payload)

    failed = report.failed()
    print(
        f"\nsoak: {report.fed_messages} player messages over {report.turns} turns, "
        f"{report.delivered} delivered, {report.withheld} withheld, "
        f"{report.commits} commits, {report.waives} waives, "
        f"{report.removed_message_count} messages evicted, "
        f"final window {report.final_message_count} of {report.window_size}; "
        + ("all checks passed" if not failed else f"FAILED: {', '.join(failed)}")
    )
    if report.canon:
        cut = {
            name: states["absent"] + states["partial"]
            for name, states in report.canon["scene_section_survival"].items()
            if states["absent"] or states["partial"]
        }
        print(f"scene sections cut on any turn (turns affected): {cut or 'none'}")
    if report.clock:
        print(
            f"clock probe: turn {report.clock['clock_turn']}, "
            f"recalled={report.clock['recalled']} "
            f"lenient={report.clock['lenient']['recalled']} "
            f"clocks_section={report.clock['clocks_section_state']} "
            f"read_tool_called={report.clock['read_tool_called']}"
        )
    for zone, scored in (report.zones or {}).items():
        print(
            f"zone {zone}: turn {scored['recall_turn']}, "
            f"recalled={scored['recalled']} lenient={scored['lenient']['recalled']} "
            f"committed={scored['committed']} in_digest={scored['in_digest_at_recall']} "
            f"digest_turns={scored['digest_turns']}"
        )
    if report.exchange:
        print(
            f"exchange probe: legs {[leg['turn'] for leg in report.exchange['legs']]}, "
            f"expected final holder {report.exchange['expected_final_holder']}, "
            f"record entries naming the {report.exchange['object']}: "
            f"{report.exchange['entry_count']}, "
            f"names in the newest entry: {report.exchange['final_entry_names'] or 'none'}"
        )
    if report.order:
        print(
            f"order probe: turn {report.order['turns']['order_recall']}, "
            f"ordered={report.order['ordered']} "
            f"lenient={report.order['lenient']['ordered']} "
            f"recalled={report.order['recalled']} "
            f"committed={report.order['committed']} "
            f"in_digest={report.order['in_digest_at_recall']} "
            f"digest_turns={report.order['digest_turns']}"
        )
    if report.payload:
        chars = report.payload["chars"]
        print(
            f"payload: {report.payload['turn_requests']} turn requests, "
            f"max {chars['total_chars_max']} chars, "
            f"superseded canon mean {chars['canon_superseded_chars_mean']} chars "
            f"({report.payload['superseded_share_mean']:.1%} of the request), "
            f"canon blocks max {chars['canon_blocks_max']}, "
            f"usage {report.payload['usage_totals'] or 'unreported'}"
        )
        per_request = report.payload["per_request_usage"]
        rate = per_request["hit_rate"]
        print(
            f"per-request reuse: {per_request['cached_tokens']} of "
            f"{per_request['input_tokens']} input tokens "
            f"({'unmeasured' if rate is None else f'{rate:.1%}'}), "
            f"{per_request['requests_with_cache_key']} of "
            f"{per_request['requests_with_usage']} requests reported the cache key"
        )
    if report.cache:
        rate = report.cache["hit_rate"]
        print(
            f"prefix cache: {report.cache['window']['prefix_cache_hits']} of "
            f"{report.cache['window']['prefix_cache_queries']} queried prompt tokens "
            f"served from cache ({'unmeasured' if rate is None else f'{rate:.1%}'}), "
            f"{report.cache['requests_observed']} requests observed against "
            f"{report.cache['requests_recorded']} recorded "
            f"(exclusive={report.cache['exclusive_window']}), "
            f"usage cache key present={report.cache['usage_cache_key_present']}, "
            f"vLLM {report.cache['engine_version'] or 'unreported'}"
        )
    print(f"sandbox retained at {sandbox}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
