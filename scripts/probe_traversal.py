#!/usr/bin/env python3
"""Write evidence from disposable production traversal-clock runs.

``causeway`` and ``switchback`` are M8's own two scenarios, added to the exact
harness M7's own gate-check line called \"shared with M8\". Both exercise the
traversal clock ``bsh_mcp.models.traversal_clock_id``/``SceneObject.traversal_segments``
and ``NarratorEngine._advance_stalled_traversal`` build: a movement declaration
that does not complete in one turn opens one clock sized to the crossing's own
authored distance, and every repeated identical declaration afterward advances it
by at least one segment, mechanically guaranteed even when the model's own turn
does not call ``clock_updates`` itself.

Two real, mutually-connected authored locations (world/locations/index.yaml)
back each scenario, distinct from the pairs ``scripts/probe_world_state.py`` (the
eel market and the road shrine) and the pre-existing ``cliff-path`` scenario (the
road shrine and the river cave) already measure, so none of this milestone's own
evidence overfits the fix to a setting an earlier milestone already exercised:
``causeway`` links the drowned customs house and the river cave, ``switchback``
links the black bell crypt and the river cave.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsh_mcp.models import traversal_clock_id
from bsh_mcp.service import GameService
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
from narrator.engine import REPETITION_SIMILARITY_THRESHOLD, narration_similarity
from narrator.service_assembly import build_narrator_service


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_tree(run_directory: Path) -> str:
    """Name the tree this run measures, or say the run is unbound.

    The same helper ``scripts/probe_world_state.py``, ``scripts/probe_social_interactions.py``,
    and ``scripts/probe_combat_decisions.py`` already use for the identical field:
    deriving the identifier from the run directory's own parent path is correct by
    construction, independent of the git index's staging state at the moment this
    script runs.
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


#: Every runtime scenario this probe knows, in the ``scripts/probe_combat_decisions.py``
#: ``SCENARIOS``/``--scenario`` mold. ``cliff-path`` is M7's own pre-existing
#: scenario, kept working unchanged; ``causeway`` and ``switchback`` are M8's.
SCENARIOS = ("cliff-path", "causeway", "switchback")


_REPORT_SCENARIOS = ("causeway", "switchback")


_START_LOCATION = "the-road-shrine"
_FAR_LOCATION = "the-river-cave"


_DECLARATION = (
    "I start down the cliff path toward the river cave, watching my footing on "
    "the wet rock."
)


_DECLARED_TURNS = 12

#: M8's own two scenarios: authored path, segment count, and scene title, keyed by
#: scenario name. ``segments`` sizes the traversal clock's typed exit
#: (``SceneObject.traversal_segments``) and this scenario's own turn budget per leg
#: (``segments + 1``, each criterion's own bound).
_SCENARIOS_META = {
    "causeway": {
        "start": "the-drowned-customs-house",
        "far": "the-river-cave",
        "title": "The Drowned Customs House",
        "segments": 3,
    },
    "switchback": {
        "start": "the-black-bell-crypt",
        "far": "the-river-cave",
        "title": "The Black Bell Crypt",
        "segments": 6,
    },
}


_DECLARATIONS = {
    "causeway": {
        "out": (
            "I start across the causeway toward the river cave, keeping pace for "
            "the crossing ahead."
        ),
        "back": (
            "I start back across the causeway toward the drowned customs house, "
            "keeping pace for the crossing ahead."
        ),
    },
    "switchback": {
        "out": (
            "I start down the switchback trail toward the river cave, watching my "
            "footing on the turns."
        ),
        "back": "",
    },
}


def _prepare_campaign(root: Path, scenario: str) -> None:
    """Create one disposable campaign for the requested scenario.
    """
    game, character_id = bootstrap_probe_campaign(root, title="Probe", weapons=["long knife"])

    if scenario == "cliff-path":
        start_location, title = _START_LOCATION, "The Road Shrine"
    else:
        meta = _SCENARIOS_META[scenario]
        start_location, title = meta["start"], meta["title"]

    with game.store.transaction("probe", character_id, "setup") as transaction:
        transaction.state.scene.location_id = start_location
        transaction.state.scene.title = title
        transaction.scene_dirty = True
        transaction.commit({"outcome": "probe_setup"})
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: terminal-player\n  character_id: "
        f"{character_id}\n  display_name: Rill\n",
        encoding="utf-8",
    )

    if scenario in _SCENARIOS_META:
        meta = _SCENARIOS_META[scenario]
        clock_id = traversal_clock_id(meta["start"], meta["far"])
        typed = game.scene_commit(
            "An authored path links the two places.",
            object_updates={
                clock_id: {"state": "open", "traversal_segments": meta["segments"]}
            },
        )
        if not typed.get("ok"):
            raise ValueError("probe typed-exit setup was unavailable")


class _RuntimeProbeAdapter:
    """Send the same declaration every turn; retain every posted reply.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(self, declaration: str, turn_count: int, config: NarratorConfig) -> None:
        del config  # retained signature; notice classification is typed post-run now
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._declaration = declaration
        self._turn_count = turn_count
        #: One entry per posted reply, in order: the raw text and whether it is an
        #: engine-authored notice rather than model narration. Notices (a withheld
        #: turn, a decision-segment boundary) are excluded from the repetition
        #: score below, the same way ``NarratorEngine._ChannelNarration.last_delivered``
        #: is only ever written from real narration. ``is_notice`` is filled by
        #: ``_classify_notices`` from the service's typed post log after the run.
        self.turn_log: list[dict] = []

    async def turns(self):
        for _ in range(self._turn_count):
            yield InboundTurn(
                _START_LOCATION, ChannelMessage("Rill", self._declaration, self._principal)
            )

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        self.turn_log.append(
            {
                "turn": len(self.turn_log) + 1,
                "text": text,
                "is_notice": None,
            }
        )

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


class _ArrivalProbeAdapter:
    """Send a repeated declaration per leg, stopping each leg early on arrival.
    """

    name = "terminal"
    decision_capabilities = ChannelCapabilities(
        structured_decisions=True, atomic_decision_delivery=True
    )

    def __init__(
        self,
        *,
        start_location: str,
        far_location: str,
        out_declaration: str,
        back_declaration: str,
        max_turns_per_leg: int,
        clock_id: str,
        segments: int,
        state_reader,
    ) -> None:
        self._principal = ChannelPrincipal("terminal", "terminal-player", "Rill")
        self._start = start_location
        self._far = far_location
        self._out_declaration = out_declaration
        self._back_declaration = back_declaration
        self._max_turns = max_turns_per_leg
        self._clock_id = clock_id
        self._segments = segments
        self._state_reader = state_reader
        self.turn_log: list[dict] = []
        #: How many of ``turn_log``'s entries belong to the outbound leg; the
        #: remainder, if any, belong to the return leg. Set once the outbound
        #: loop below ends, before any return-leg turn is ever yielded --
        #: ``_runtime_flow`` slices ``turn_log`` on this rather than re-inferring
        #: the boundary from location transitions after the fact.
        self.outbound_turn_count = 0

    def _clock_filled(self, state) -> int | None:
        clock = next((entry for entry in state.clocks if entry.id == self._clock_id), None)
        return clock.filled if clock else None

    async def turns(self):
        for _ in range(self._max_turns):
            state = self._state_reader()
            filled = self._clock_filled(state)
            if state.scene.location_id == self._far or (
                filled is not None and filled >= self._segments
            ):
                break
            yield InboundTurn(
                self._start, ChannelMessage("Rill", self._out_declaration, self._principal)
            )
        self.outbound_turn_count = len(self.turn_log)
        if not self._back_declaration:
            return
        # The return leg never stops on ``location_id == self._start`` alone:
        # unlike the outbound leg's ``self._far`` check, ``self._start`` is
        # where the party is recorded as standing *before* the outbound leg
        # even begins, so that check would read true on this loop's very first
        # look -- before a single return-leg turn has run -- whenever the
        # outbound leg's own ``location_id`` follow-through never actually
        # landed (the compliance gap this scenario's own checks already treat
        # as out of scope; see ``scenario_checks``). Only the clock's own
        # genuine reset-then-refill, the same signal ``scenario_checks`` scores
        # this leg by, ends it.
        seen_reset = False
        for _ in range(self._max_turns):
            state = self._state_reader()
            filled = self._clock_filled(state)
            if filled is not None and filled < self._segments:
                seen_reset = True
            if seen_reset and filled is not None and filled >= self._segments:
                break
            yield InboundTurn(
                self._start, ChannelMessage("Rill", self._back_declaration, self._principal)
            )

    async def post(self, channel_id: str, text: str) -> None:
        del channel_id
        state = self._state_reader()
        clock = next((entry for entry in state.clocks if entry.id == self._clock_id), None)
        self.turn_log.append(
            {
                "turn": len(self.turn_log) + 1,
                "text": text,
                "is_notice": None,
                "location_id": state.scene.location_id,
                "clock_filled": clock.filled if clock else None,
                "clock_segments": clock.segments if clock else None,
            }
        )

    async def close(self) -> None:
        return None

    async def deliver_decision_views(self, views) -> DecisionDeliveryReceipt:
        rendered = tuple(views)
        return DecisionDeliveryReceipt(status="delivered", decision_count=len(rendered))

    async def collect_decision(self, views):
        view = tuple(views)[0]
        selection = "confirm" if view.kind == "confirmation" else view.options[0].id
        return self._principal, DecisionSubmission(
            presentation_token=view.presentation_token, selection_id=selection
        )

    async def acknowledge_decision(self, result) -> None:
        del result


def _classify_notices(turn_log: list[dict], service) -> None:
    """Fill each recorded turn's ``is_notice`` from the service's typed post log.
    """
    if len(service.post_log) != len(turn_log):
        raise RuntimeError(
            f"post log carries {len(service.post_log)} posts for {len(turn_log)} "
            "recorded turns; notice attribution is broken"
        )
    for entry, post in zip(turn_log, service.post_log):
        entry["is_notice"] = post.kind == "notice"


def _repeated_delivered_pairs(turn_log: list[dict]) -> list[dict]:
    """Consecutive real-narration turns whose similarity meets or exceeds threshold.

    Only narration-to-narration pairs are scored: an engine-authored notice sits
    between two narrations without becoming a comparison point on either side,
    matching how ``NarratorEngine._ChannelNarration.last_delivered`` is written --
    never from a notice, only from delivered narration.
    """
    narrations = [entry for entry in turn_log if not entry["is_notice"]]
    exceeding: list[dict] = []
    for previous, current in zip(narrations, narrations[1:]):
        score = narration_similarity(previous["text"], current["text"])
        if score >= REPETITION_SIMILARITY_THRESHOLD:
            exceeding.append(
                {
                    "turn": current["turn"],
                    "predecessor_turn": previous["turn"],
                    "similarity": score,
                }
            )
    return exceeding


def _leg_progress(
    leg_log: list[dict], segments: int, *, allow_stale_start: bool = False
) -> tuple[int | None, list[int | None]]:
    """This leg's own completion turn and its own clean fill sequence.
    """
    values = [entry["clock_filled"] for entry in leg_log]
    dropped = 0
    while values and (
        values[0] is None or (allow_stale_start and values[0] >= segments)
    ):
        values.pop(0)
        dropped += 1
    completion: int | None = None
    for offset, value in enumerate(values):
        if value is not None and value >= segments:
            completion = dropped + offset + 1
            break
    return completion, values


def _strictly_increasing(values: list[int | None]) -> bool:
    """True when every value is present and strictly greater than the one before.

    Empty is False: "strictly increasing" asks for at least one measured point
    that actually moved, not vacuous truth over zero turns.
    """
    if not values:
        return False
    previous: int | None = None
    for value in values:
        if value is None:
            return False
        if previous is not None and value <= previous:
            return False
        previous = value
    return True


def scenario_checks(
    scenario: str,
    *,
    turns: int,
    errors: bool,
    narration_count: int,
    exceeding_threshold_count: int = 0,
    outbound_log: list[dict] | None = None,
    return_log: list[dict] | None = None,
    segments: int = 0,
) -> dict:
    """Map one run's observed state onto this scenario's checks, with no endpoint.
    """
    checks = {
        "service_completed": turns > 0 and not errors,
        "narration_delivered": narration_count > 0,
    }
    if scenario == "cliff-path":
        checks["zero_replay"] = exceeding_threshold_count == 0
        return checks

    outbound_log = outbound_log or []
    return_log = return_log or []
    bound = segments + 1
    turns_out, outbound_fill = _leg_progress(outbound_log, segments)
    checks["outbound_completed"] = turns_out is not None
    checks["outbound_within_bound"] = turns_out is not None and turns_out <= bound
    checks["outbound_fill_strictly_increasing"] = _strictly_increasing(outbound_fill)

    if scenario == "causeway":
        turns_back, return_fill = _leg_progress(return_log, segments, allow_stale_start=True)
        checks["return_completed"] = turns_back is not None
        checks["return_within_bound"] = turns_back is not None and turns_back <= bound
        checks["return_fill_strictly_increasing"] = _strictly_increasing(return_fill)
        checks["turn_counts_within_one"] = (
            turns_out is not None
            and turns_back is not None
            and abs(turns_out - turns_back) <= 1
        )
    elif scenario == "switchback":
        checks["fill_at_least_turn_index"] = bool(outbound_log) and all(
            entry["clock_filled"] is not None and entry["clock_filled"] >= index
            for index, entry in enumerate(outbound_log, start=1)
        )
        checks["no_replay_above_threshold"] = exceeding_threshold_count == 0
    return checks


async def _runtime_flow(root: Path, endpoint: str, model_id: str, scenario: str) -> dict:
    """Run the real narrator service, engine, and MCP tool calls against the endpoint."""
    _prepare_campaign(root, scenario)
    config = NarratorConfig(
        campaign_root=root,
        base_url=endpoint,
        model_id=model_id,
        max_decision_rounds=2,
    )

    if scenario == "cliff-path":
        adapter = _RuntimeProbeAdapter(_DECLARATION, _DECLARED_TURNS, config)
        service = build_narrator_service(config, adapter)
        report = await service.run()
        _classify_notices(adapter.turn_log, service)
        turn_log = adapter.turn_log
        narration_count = sum(1 for entry in turn_log if not entry["is_notice"])
        exceeding = _repeated_delivered_pairs(turn_log)
        return {
            "turn_log": turn_log,
            "narration_count": narration_count,
            "notice_count": len(turn_log) - narration_count,
            "repetition_log": list(service.engine._repetition_log),  # noqa: SLF001 - retained diagnostic, read-only
            "exceeding_threshold": exceeding,
            "checks": scenario_checks(
                scenario,
                turns=report.turns,
                errors=bool(report.errors),
                narration_count=narration_count,
                exceeding_threshold_count=len(exceeding),
            ),
        }

    meta = _SCENARIOS_META[scenario]
    segments = meta["segments"]
    clock_id = traversal_clock_id(meta["start"], meta["far"])
    declarations = _DECLARATIONS[scenario]
    store = GameService(root).store
    adapter = _ArrivalProbeAdapter(
        start_location=meta["start"],
        far_location=meta["far"],
        out_declaration=declarations["out"],
        back_declaration=declarations["back"],
        max_turns_per_leg=segments + 1,
        clock_id=clock_id,
        segments=segments,
        state_reader=store.read_state,
    )
    service = build_narrator_service(config, adapter)
    report = await service.run()
    _classify_notices(adapter.turn_log, service)

    full_log = adapter.turn_log
    outbound_log = full_log[: adapter.outbound_turn_count]
    return_log = full_log[adapter.outbound_turn_count :]
    narration_count = sum(1 for entry in full_log if not entry["is_notice"])
    exceeding = _repeated_delivered_pairs(full_log)
    return {
        "turn_log": full_log,
        "outbound_turn_count": adapter.outbound_turn_count,
        "narration_count": narration_count,
        "notice_count": len(full_log) - narration_count,
        "repetition_log": list(service.engine._repetition_log),  # noqa: SLF001
        "traversal_log": list(service.engine._traversal_log),  # noqa: SLF001
        "exceeding_threshold": exceeding,
        "checks": scenario_checks(
            scenario,
            turns=report.turns,
            errors=bool(report.errors),
            narration_count=narration_count,
            exceeding_threshold_count=len(exceeding),
            outbound_log=outbound_log,
            return_log=return_log,
            segments=segments,
        ),
    }


def _mechanically_sound(measurement: dict) -> bool:
    return (
        bool(measurement.get("checks", {}).get("service_completed"))
        and measurement.get("narration_count", 0) > 0
    )


def _run_runtime(root: Path, endpoint: str, model_id: str, scenario: str) -> int:
    """Emit one machine-readable marker. Exit reflects mechanical soundness only."""
    measurement = asyncio.run(_runtime_flow(root, endpoint, model_id, scenario))
    print(f"TRAVERSAL_RUNTIME={json.dumps(measurement, sort_keys=True)}")
    return 0 if _mechanically_sound(measurement) else 75


def _live_scenario(run_directory: Path, endpoint: str, model_id: str, scenario: str) -> dict:
    """Launch the runtime helper under the narrator interpreter for one scenario.

    Up to two attempts, each with a fresh campaign, matching
    ``scripts/probe_combat_decisions.py``'s own retry mold but capped lower: this
    probe's own scenarios cost several turns per attempt (a bidirectional
    causeway crossing, or a six-segment switchback), against the shared
    endpoint's own live cost, so a retry here absorbs one transient sampling
    miss rather than statistically searching for a pass. The returned
    measurement comes from the last attempt, which keeps a failing run visible
    instead of discarding it.
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
    measurement: dict = {}
    for attempt in range(2):
        campaign_root = run_directory / f"traversal-{scenario}-{attempt}"
        result = subprocess.run(
            [
                narrator_python, "scripts/probe_traversal.py", "--runtime",
                "--scenario", scenario,
                "--campaign-root", str(campaign_root),
                "--endpoint", endpoint,
                "--model-id", model_id,
            ],
            cwd=repository, env=environment, text=True, capture_output=True,
            timeout=900, check=False,
        )
        marker = next(
            (
                line.partition("=")[2]
                for line in result.stdout.splitlines()
                if line.startswith("TRAVERSAL_RUNTIME=")
            ),
            "",
        )
        try:
            measurement = json.loads(marker)
        except json.JSONDecodeError:
            measurement = {}
        if result.returncode == 0 and measurement and all(measurement.get("checks", {}).values()):
            return {"attempts": attempt + 1, "measurement": measurement}
    if not measurement:
        raise ValueError(f"the {scenario} scenario produced no marker")
    return {"attempts": 2, "measurement": measurement}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--campaign-root")
    parser.add_argument("--model-id")
    args = parser.parse_args()
    if args.runtime:
        if not args.campaign_root or not args.model_id or not args.scenario:
            print("runtime requires --campaign-root, --model-id, and --scenario", file=sys.stderr)
            return 2
        return _run_runtime(
            Path(args.campaign_root), args.endpoint.rstrip("/"), args.model_id, args.scenario
        )
    run_directory = os.environ.get("BSH_PROBE_RUN_DIRECTORY", "")
    if not run_directory:
        print("HELD: BSH_PROBE_RUN_DIRECTORY is required.", file=sys.stderr)
        return 75
    destination = Path(run_directory) / "traversal-production-probe.json"
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
            checks = {
                scenario: _live_scenario(Path(run_directory), endpoint, model["model_id"], scenario)
                for scenario in _REPORT_SCENARIOS
            }
            report["checks"] = checks
            # The attempt count sits beside the checks rather than among them. A
            # counter can therefore never be mistaken for a check, nor a check
            # for a counter. Every entry under "checks"."measurement"."checks"
            # is a boolean claim.
            failed = [
                f"{scenario}.{name}"
                for scenario, entry in checks.items()
                for name, value in entry["measurement"].get("checks", {}).items()
                if value is not True
            ]
            report["failed_checks"] = failed
            report["status"] = "PASS" if not failed else "FAIL"
        except (OSError, ValueError, urllib.error.URLError, subprocess.SubprocessError) as error:
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
