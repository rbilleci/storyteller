#!/usr/bin/env python3
"""Write evidence from one disposable production typed-barrier run.

The probe deliberately does not route the unlock through a narrator tool call.
``scene_commit``'s ``object_updates`` parameter lives at the ``GameService`` layer
(``src/bsh_mcp/service.py``), inside this slice's declared paths; exposing it as a
narrator-facing MCP tool argument is a ``src/bsh_mcp/server.py`` change, and that file
sits outside this slice's declared paths (``record.json``'s own ``paths`` list omits
it). This probe therefore locks and unlocks the gate the same way every existing
probe's scenario setup mutates campaign state -- directly through ``GameService``,
matching ``scripts/probe_combat_decisions.py``'s and
``scripts/probe_social_interactions.py``'s own convention. What this probe measures
live is the model's own reads of the typed record across repeated turns: whether its
own narration or its own ``scene_commit`` location changes ever move the party past
a barrier the record still calls locked, and whether they resume once the record
says otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsh_mcp.service import GameService
from bsh_mcp.store import CampaignStore
from bsh_mcp.testing import bootstrap_probe_campaign
from narrator.channels.base import (
    ChannelCapabilities,
    ChannelMessage,
    ChannelPrincipal,
    DecisionDeliveryReceipt,
    InboundTurn,
)
from narrator.config import NarratorConfig
from narrator.decisions import DecisionSubmission
from narrator.service_assembly import build_narrator_service


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_tree(run_directory: Path) -> str:
    """Name the tree this run measures, or say the run is unbound.

    The same helper ``scripts/probe_social_interactions.py`` and
    ``scripts/probe_combat_decisions.py`` already use for the identical field: deriving
    the identifier from the run directory's own parent path is correct by construction,
    independent of the git index's staging state at the moment this script runs.
    """
    parent = run_directory.resolve().parent.name
    if len(parent) == 40 and all(character in "0123456789abcdef" for character in parent):
        return parent
    return "unbound"


def _models(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as response:
        payload = json.loads(response.read())
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise ValueError("endpoint returned no model")
    model_id = models[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("endpoint model identifier is invalid")
    return {
        "model_id": model_id,
        "models_digest": _sha256(json.dumps(payload, sort_keys=True).encode()),
    }


def _container_image() -> dict:
    command = ["docker", "inspect", "--format", "{{.Config.Image}} {{.Image}}", "vllm-gemma"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    image, digest = result.stdout.strip().split(maxsplit=1)
    return {"image": image, "image_digest": digest}


#: Two real authored, mutually-connected locations (world/locations/index.yaml), so
#: the narrator's canon digest carries a real exits list and the party's declared
#: passage names a resolvable destination -- the same real-location convention
#: ``scripts/probe_combat_decisions.py``'s disengage scenario established.
_START_LOCATION = "the-eel-market"
_FAR_LOCATION = "the-road-shrine"

#: The barrier's id. Not itself a location id: an object id names the physical thing
#: gating passage, matching how ``Scene.objects`` is keyed generally, not only for
#: exits.
_BARRIER_ID = "market-gate"


_LOCKED_DECLARATION = (
    "I check the market gate again, without trying to force it, to see if it's open."
)


_UNLOCKED_RECORD_REQUEST = (
    "The gate is open now. Move us through it to the road shrine and record that "
    "move in the scene right now, in this reply."
)
#: Declared as an already-accomplished fact, kept as the second attempt: this mirrors
#: ``scripts/probe_combat_decisions.py``'s disengage scenario, which declares the
#: flight as already finished ("Rill slips clear ... this skirmish is finished")
#: rather than asking whether it may happen.
_UNLOCKED_DECLARATION = (
    "The gate stands open now. I'm already through it and standing at the road "
    "shrine -- there is no more distance left to cover."
)


_UNLOCKED_NUDGE = (
    "GM, straight answer: where am I now -- still at the market, or through to the "
    "road shrine?"
)

#: How many identical, repeated passage attempts run while the barrier is locked.
#: "Repeated turns" is the criterion's own wording: a single attempt could pass by
#: coincidence, so this scenario asks more than once, verbatim, the same way
#: repeated-declaration scenarios elsewhere in this repository do.
_LOCKED_ATTEMPTS = 3


_BARRIER_OPENED_CLAIM = re.compile(
    r"\b(?:gate|door|bar|beam|latch|bolt)\b[^.]{0,60}\b(?:swings? (?:open|wide)|"
    r"gives? way|breaks? open|snaps?|bursts? open|(?:is|stands?|hangs?) open|"
    r"(?:is |are )?no longer (?:fully )?(?:barred|locked|blocked))\b"
    r"|\bclear path\b"
    r"|\bsteps? through the (?:open|broken|bent|shattered)\b"
    r"|\byou can now pass\b",
    re.IGNORECASE,
)


_ARRIVAL_CLAIM = re.compile(
    r"\bat the road shrine\b|\breached the road shrine\b|\bthrough to the road shrine\b",
    re.IGNORECASE,
)


_NEGATION_WINDOW = 30

_NEGATION_WORDS = re.compile(
    r"\b(?:cannot|can't|can not|unable to|won't|will not|never|not yet|no way)\b",
    re.IGNORECASE,
)


def _claim_matches(pattern: re.Pattern, text: str) -> bool:
    """True when ``pattern`` matches somewhere a nearby negation does not cancel.

    Scans every match, not just the first: a text carrying one negated mention and
    one genuine claim (unlikely here, but not excluded by construction) must still
    flag on the genuine one rather than stopping at the negated one.
    """
    for match in pattern.finditer(text):
        window = text[max(0, match.start() - _NEGATION_WINDOW):match.start()]
        if not _NEGATION_WORDS.search(window):
            return True
    return False


def _turn_contradicts_state(entry: dict) -> list[str]:
    """Every reason this one delivered turn contradicts state.json, or an empty list.
    """
    reasons: list[str] = []
    if entry["barrier_state"] in ("locked", "barred") and entry["location_id"] != _START_LOCATION:
        reasons.append(
            "the party's recorded location moved while the barrier record still "
            f"reads {entry['barrier_state']!r}"
        )
    if entry["barrier_state"] in ("locked", "barred") and _claim_matches(
        _BARRIER_OPENED_CLAIM, entry["narration"]
    ):
        reasons.append(
            "the narration claims the barrier itself opened while the record still "
            f"reads {entry['barrier_state']!r}"
        )
    if entry["location_id"] != _FAR_LOCATION and _claim_matches(
        _ARRIVAL_CLAIM, entry["narration"]
    ):
        reasons.append(
            "the narration claims arrival at, or newly-open access to, the far "
            f"location while the record still shows {entry['location_id']!r}"
        )
    return reasons


def _prepare_campaign(root: Path) -> str:
    """Create one disposable campaign with the barrier locked through the typed record."""
    game, character_id = bootstrap_probe_campaign(root, title="Probe", weapons=["long knife"])
    with game.store.transaction("probe", character_id, "setup") as transaction:
        transaction.state.scene.location_id = _START_LOCATION
        transaction.state.scene.title = "The Eel Market"
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_setup"})
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )
    locked = game.scene_commit(
        "The market gate toward the road shrine stands barred.",
        object_updates={
            _BARRIER_ID: {
                "state": "locked",
                "note": "an iron gate, barred from the customs side",
            }
        },
    )
    if not locked.get("ok"):
        raise ValueError("probe barrier setup was unavailable")
    return character_id


class _RuntimeProbeAdapter:
    """Supply the declared turns, unlock mid-run, and read state.json after each post.

    The unlock runs as a side effect between two ``yield`` points inside ``turns``.
    ``NarratorService.run`` pulls one turn at a time from this async generator and
    fully processes each pulled turn (including the ``post`` call below) before asking
    for the next one, so the unlock is guaranteed to land only after every locked-phase
    turn has already been delivered and scored.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self, locked_declarations, unlocked_declarations, unlock, state_reader) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._locked_declarations = locked_declarations
        self._unlocked_declarations = unlocked_declarations
        self._unlock = unlock
        self._state_reader = state_reader
        self.decision_kinds: list[str] = []
        self.output_categories: list[str] = []
        #: One entry per delivered turn: the state read immediately after it (which
        #: decides pass or fail) and the narration posted for it (retained beside the
        #: state read, per this criterion's own wording, for human review).
        self.turn_log: list[dict] = []

    async def turns(self):
        for text in self._locked_declarations:
            yield InboundTurn(_START_LOCATION, ChannelMessage("Rill", text, self._principal))
        self._unlock()
        for text in self._unlocked_declarations:
            yield InboundTurn(_START_LOCATION, ChannelMessage("Rill", text, self._principal))

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.output_categories.append("narration")
        state = self._state_reader()
        barrier = state.scene.objects.get(_BARRIER_ID)
        self.turn_log.append(
            {
                "turn": len(self.turn_log) + 1,
                "location_id": state.scene.location_id,
                "barrier_state": barrier.state if barrier else None,
                "narration": text,
            }
        )

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        self.decision_kinds.extend(view.kind for view in rendered)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


#: Checks that should be deterministic every attempt: the run completed and every
#: declared turn reached the engine. Kept apart from the barrier-consistency checks
#: below because a mechanical failure here means something unrelated is broken, while
#: a barrier-consistency miss is the live, model-dependent behavior this milestone
#: measures.
def _mechanically_sound(checks: dict) -> bool:
    return bool(checks.get("service_completed"))


#: How many independent runs to score. Fewer than M5's five: this scenario's per-turn
#: signal (locked-state-plus-far-location is a hard contradiction, not a partial
#: figure) needs less statistical weight to be meaningful, and each attempt here
#: costs five model turns against the shared endpoint rather than one.
_LIVE_ATTEMPTS = 3

#: How many of ``_LIVE_ATTEMPTS`` may fail the mechanical run itself (an unrelated
#: engine hiccup, not a barrier-consistency defect) before the whole run is
#: considered broken rather than merely sampling a live model.
_MAX_MECHANICAL_FAILURES = 1


_MIN_PASSAGE_AFTER_UNLOCK = 1


async def _runtime_flow(root: Path, endpoint: str, model_id: str) -> dict:
    """Run the real narrator service, engine, and MCP tool calls against the endpoint."""
    _prepare_campaign(root)
    game = GameService(root)
    store: CampaignStore = game.store

    def unlock() -> None:
        # Two details scenario development measured live. First, the note is passed
        # explicitly, not left at its prior value: leaving the locked-phase note ("an
        # iron gate, barred from the customs side") in place after flipping state left
        # the record self-contradictory in exactly the way this milestone exists to
        # prevent, and the live model read the stale note over the updated state
        # field. Second, the target state is "open", not "unlocked": the live model
        # correctly distinguished the two on its own ("The gate is currently unlocked
        # but remains closed") when this probe used "unlocked" here, which is a
        # legitimate reading of a five-state enum -- unlocked describes the latch, not
        # the leaf -- but leaves passage still blocked on a fact this probe did not
        # intend to test. "open" is unambiguous, and is the state this scenario means
        # to test recovery into.
        released = game.scene_commit(
            "Someone throws the bar; the gate swings open.",
            object_updates={
                _BARRIER_ID: {
                    "state": "open",
                    "note": "an iron gate, standing open on its hinges",
                }
            },
        )
        if not released.get("ok"):
            raise ValueError("probe barrier release was unavailable")

    adapter = _RuntimeProbeAdapter(
        locked_declarations=[_LOCKED_DECLARATION] * _LOCKED_ATTEMPTS,
        unlocked_declarations=[_UNLOCKED_RECORD_REQUEST, _UNLOCKED_DECLARATION, _UNLOCKED_NUDGE],
        unlock=unlock,
        state_reader=store.read_state,
    )
    config = NarratorConfig(
        campaign_root=root,
        base_url=endpoint,
        model_id=model_id,
        max_decision_rounds=2,
    )
    service = build_narrator_service(config, adapter)
    report = await service.run()

    turn_log = adapter.turn_log
    locked_entries = turn_log[:_LOCKED_ATTEMPTS]
    unlocked_entries = turn_log[_LOCKED_ATTEMPTS:]
    # The one check this whole scenario exists for: every delivered turn scored
    # against state.json and its own retained narration by ``_turn_contradicts_state``,
    # per AUD-1's correction -- not merely "barrier still locked and location moved",
    # which an independent audit found blind to a narrated barrier-state claim that
    # never moved the location field, and to a narrated arrival claim once the
    # barrier had already flipped away from "locked".
    contradictory_turns = [
        {**entry, "reasons": reasons}
        for entry in turn_log
        for reasons in [_turn_contradicts_state(entry)]
        if reasons
    ]
    return {
        "service_completed": report.turns == len(turn_log) and not report.errors,
        "turn_log": turn_log,
        "zero_contradictions": len(contradictory_turns) == 0,
        "contradiction_count": len(contradictory_turns),
        "contradictory_turns": contradictory_turns,
        "blocked_while_locked": all(
            entry["location_id"] == _START_LOCATION for entry in locked_entries
        ),
        "passage_after_unlock": any(
            entry["location_id"] != _START_LOCATION for entry in unlocked_entries
        ),
    }


def _run_runtime(root: Path, endpoint: str, model_id: str) -> int:
    """Emit one machine-readable marker. Exit reflects mechanical soundness only.

    Whether the barrier stayed consistent is aggregated across every attempt by
    ``_live_production_flow``, not decided by this single attempt.
    """
    checks = asyncio.run(_runtime_flow(root, endpoint, model_id))
    print(f"WORLD_STATE_RUNTIME={json.dumps(checks, sort_keys=True)}")
    return 0 if _mechanically_sound(checks) else 75


def _live_production_flow(run_directory: Path, endpoint: str, model_id: str) -> dict:
    """Launch the runtime helper under the narrator interpreter, ``_LIVE_ATTEMPTS`` times.
    """
    narrator_python = os.environ.get(
        "NARRATOR_PYTHON",
        str(Path(__file__).resolve().parents[1] / ".narrator-venv" / "bin" / "python"),
    )
    environment = {
        **os.environ,
        "BSH_SERVER_PYTHON": os.environ.get("BSH_SERVER_PYTHON", sys.executable),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    repository = Path(__file__).resolve().parents[1]
    live_attempts: list[dict] = []
    for attempt in range(_LIVE_ATTEMPTS):
        campaign_root = run_directory / f"endpoint-campaign-{attempt}"
        result = subprocess.run(
            [
                narrator_python,
                "scripts/probe_world_state.py",
                "--runtime",
                "--campaign-root", str(campaign_root),
                "--endpoint", endpoint,
                "--model-id", model_id,
            ],
            cwd=repository,
            env=environment,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        marker = next(
            (
                line.partition("=")[2]
                for line in result.stdout.splitlines()
                if line.startswith("WORLD_STATE_RUNTIME=")
            ),
            "",
        )
        try:
            checks = json.loads(marker)
        except json.JSONDecodeError:
            checks = {}
        live_attempts.append({"attempt": attempt, "returncode": result.returncode, "checks": checks})

    mechanical_failures = [
        entry for entry in live_attempts
        if entry["returncode"] != 0 or not entry["checks"] or not _mechanically_sound(entry["checks"])
    ]
    contradictions_total = sum(
        entry["checks"].get("contradiction_count", 0) for entry in live_attempts
    )
    passage_after_unlock_count = sum(
        1 for entry in live_attempts if entry["checks"].get("passage_after_unlock")
    )
    aggregate = {
        "live_attempts": live_attempts,
        "attempts_total": _LIVE_ATTEMPTS,
        "mechanical_failures": len(mechanical_failures),
        "mechanical_failures_allowed": _MAX_MECHANICAL_FAILURES,
        "contradictions_total": contradictions_total,
        "passage_after_unlock_count": passage_after_unlock_count,
        "passage_after_unlock_min_required": _MIN_PASSAGE_AFTER_UNLOCK,
        "mechanical_failures_within_budget": len(mechanical_failures) <= _MAX_MECHANICAL_FAILURES,
        "zero_contradictions": contradictions_total == 0,
        "minimum_passage_met": passage_after_unlock_count >= _MIN_PASSAGE_AFTER_UNLOCK,
    }
    if (
        len(mechanical_failures) > _MAX_MECHANICAL_FAILURES
        or contradictions_total
        or passage_after_unlock_count < _MIN_PASSAGE_AFTER_UNLOCK
    ):
        raise ValueError(
            "endpoint-backed world-state production flow did not meet its aggregate "
            f"bar: {len(mechanical_failures)} mechanical failure(s) (budget "
            f"{_MAX_MECHANICAL_FAILURES}), {contradictions_total} barrier-state "
            f"contradiction(s), {passage_after_unlock_count}/"
            f"{_MIN_PASSAGE_AFTER_UNLOCK} required post-unlock passages -- "
            f"{json.dumps(aggregate, sort_keys=True)}"
        )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--campaign-root")
    parser.add_argument("--model-id")
    args = parser.parse_args()
    if args.runtime:
        if not args.campaign_root or not args.model_id:
            print("runtime requires --campaign-root and --model-id", file=sys.stderr)
            return 2
        return _run_runtime(Path(args.campaign_root), args.endpoint.rstrip("/"), args.model_id)
    run_directory = os.environ.get("BSH_PROBE_RUN_DIRECTORY", "")
    if not run_directory:
        print("HELD: BSH_PROBE_RUN_DIRECTORY is required.", file=sys.stderr)
        return 75
    destination = Path(run_directory) / "world-state-production-probe.json"
    if destination.exists():
        print("HELD: the run-bound report already exists.", file=sys.stderr)
        return 75
    checks: dict = {}
    report = {
        "status": "PASS",
        "candidate_tree": _candidate_tree(Path(run_directory)),
        "command_sha256": _sha256(" ".join(sys.argv).encode()),
        "launcher_sha256": (_sha256(Path(os.environ["BSH_MODEL_LAUNCHER"]).read_bytes())
                            if os.environ.get("BSH_MODEL_LAUNCHER") else None),
        "checks": checks,
        "outputs": [],
        "offline": args.offline,
    }
    if not args.offline:
        try:
            endpoint = args.endpoint.rstrip("/")
            model = _models(endpoint)
            report.update(model)
            report.update(_container_image())
            checks = _live_production_flow(Path(run_directory), endpoint, model["model_id"])
            report["checks"] = checks
        except (OSError, ValueError, urllib.error.URLError, subprocess.SubprocessError) as error:
            # The message is safe to print: every value in it is a structured check
            # result (booleans, counts, ids) or the retained narration this module
            # already writes into the report, never a secret.
            print(
                f"HELD: production endpoint evidence is unavailable: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return 75
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    digest = _sha256(destination.read_bytes())
    print(f"report_path={destination}")
    print(f"report_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
