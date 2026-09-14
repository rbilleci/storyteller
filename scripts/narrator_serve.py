#!/usr/bin/env python3
"""Run the narrator against one channel. Replaces the Hermes gateway.

Usage::

    python scripts/narrator_serve.py --channel replay --script scripts/demo-session.txt
    python scripts/narrator_serve.py --channel replay --root /tmp/sandbox
    python scripts/narrator_serve.py --dry-run
    python scripts/narrator_serve.py --turns 3 --transcript /tmp/run.log

The endpoint must already serve the model. Do not start or restart it without
authorization.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.channels.diagnostics import (  # noqa: E402
    DiagnosticRecorder,
    DiagnosticTranscriptRecorder,
)
from narrator.channels.replay import TranscriptReplayAdapter  # noqa: E402
from narrator.channels.terminal import TerminalAdapter  # noqa: E402
from narrator.config import (  # noqa: E402
    DEFAULT_THINKING_BUDGETS,
    THINKING_LEVELS,
    NarratorConfig,
    turn_thinking_from_env,
)
from narrator.engine import DISCLOSED_SKILLS, NarratorEngine  # noqa: E402
from narrator.server_launch import resolved_server_command  # noqa: E402
from narrator.service import install_signal_handlers  # noqa: E402
from narrator.service_assembly import build_narrator_service  # noqa: E402


class _LimitedReplayAdapter(TranscriptReplayAdapter):
    """A replay adapter that stops after ``limit`` turns and can tee to a file.

    The turn cap belongs here rather than in the service, because stopping early is a
    property of this harness, not of the narrator. A Discord adapter has no turn count
    to cap.
    """

    def __init__(self, *args, limit: int = 0, transcript_log=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.transcript_log = transcript_log

    async def turns(self):
        produced = 0
        async for turn in super().turns():
            if self.limit and produced >= self.limit:
                return
            produced += 1
            if self.transcript_log:
                self.transcript_log.write(f"\n--- turn {produced} ---\n")
                self.transcript_log.write(turn.channel_text() + "\n")
            yield turn

    async def post(self, channel_id: str, text: str) -> None:
        await super().post(channel_id, text)
        if self.transcript_log:
            self.transcript_log.write(f"\nGM: {text}\n")
            self.transcript_log.flush()


def _dry_run(config: NarratorConfig) -> int:
    """List the tools, the resources, the disclosed skills and the configuration.

    This reaches the model server not at all and the Model Context Protocol server only
    for listing, so an operator can check the wiring while the endpoint is down.

    An audit found an earlier version printing tools only, while its docstring claimed
    skills and the harness it replaced also printed the 6 resource identifiers. Both are
    here now, because a diagnostic that omits half the server surface hides exactly the
    wiring failure it exists to catch.
    """
    engine = NarratorEngine(config)
    try:
        engine.start()
        names = sorted(
            getattr(tool, "tool_name", None) or getattr(tool, "name", "")
            for tool in engine._tools
        )
        print(f"{len(names)} tools served by the Model Context Protocol server:")
        for name in names:
            print(f"  {name}")

        # Six registrations at src/bsh_mcp/server.py, but the protocol splits them:
        # 4 concrete resources and 2 parameterised templates. Listing only the first
        # call would print 4 against the 6 the documentation states.
        concrete = engine._client.list_resources_sync()
        concrete = getattr(concrete, "resources", concrete)
        templates = engine._client.list_resource_templates_sync()
        templates = getattr(templates, "resourceTemplates", templates)
        print(f"\n{len(concrete) + len(templates)} resources:")
        for resource in concrete:
            print(f"  {getattr(resource, 'uri', resource)}")
        for template in templates:
            print(f"  {getattr(template, 'uriTemplate', template)}  (template)")

        print(f"\n{len(DISCLOSED_SKILLS)} skills disclosed through the plugin:")
        for relative in DISCLOSED_SKILLS:
            print(f"  {relative}")
        print("  (skills/bsh-gm is concatenated into the system prompt, not disclosed)")

        print(f"\ncampaign root: {config.campaign_root}")
        print(f"repository root: {config.repo_root}")
        print(f"endpoint: {config.base_url}")
        print(f"model: {config.model_id}")
        print(f"settle attempts per turn: {config.max_settle_attempts}")
        print(f"decision rounds per turn: {config.max_decision_rounds}")
    finally:
        engine.stop()
    return 0


def _preflight(config: NarratorConfig) -> str:
    """Return an operator-facing reason the server child cannot launch, or an empty string.

    ``server_launch.resolved_server_command`` falls back to ``uv run`` when
    ``BSH_SERVER_PYTHON`` is unset, and ``uv`` is not on the default PATH on this host.
    Without this check the failure surfaced as a 60-line Strands traceback ending in
    ``FileNotFoundError: 'uv'``, which names the symptom and not the fix.

    ``shutil.which`` decides every case, rather than a separate absolute-path branch. An
    audit found that branch testing ``Path.exists``, which is true for a directory and
    for a non-executable file, so pointing ``BSH_SERVER_PYTHON`` at a virtual environment
    instead of its ``bin/python`` passed the check and then produced the traceback this
    function exists to replace. ``which`` returns ``None`` for both, and it honours
    ``execvp`` semantics for a bare name, so a relative path keeps working.
    """
    command = resolved_server_command(config)
    executable = command[0]
    if shutil.which(executable) is not None:
        return ""

    override = os.environ.get("BSH_SERVER_PYTHON")
    hint = (
        "It must name an executable interpreter holding mcp 2.x, for example:\n"
        "  export BSH_SERVER_PYTHON=/absolute/path/to/project/.venv/bin/python"
    )
    if override:
        return f"BSH_SERVER_PYTHON names {override!r}, which is not an executable file.\n{hint}"
    return (
        f"cannot find {executable!r} on PATH, and BSH_SERVER_PYTHON is unset.\n{hint}"
    )


def _record_error(recorder: DiagnosticRecorder | None, error: str) -> None:
    """Record an error category without retaining untrusted model error text."""
    if recorder is not None:
        recorder.record("error", category=error.partition(":")[0].split(maxsplit=1)[0])


def _record_output(recorder: DiagnosticRecorder | None, text: str) -> None:
    """Record terminal output the entry point writes outside a channel adapter."""
    if recorder is not None:
        recorder.record("terminal_output", kind="service_output", text=text)


def _decision_rounds(value: str) -> int:
    """Parse the public decision-rounds setting at the executable boundary."""
    try:
        rounds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if rounds < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return rounds


#: Matches ``NarratorConfig.max_decision_rounds``'s production dataclass default
#: (``src/narrator/config.py``). Named here rather than imported, because this value
#: must resolve before a ``NarratorConfig`` exists, and one literal both modules'
#: docstrings name is less fragile across the module boundary than reaching into the
#: dataclass field default from this file.
_DEFAULT_DECISION_ROUNDS = 2


def _resolve_decision_rounds(cli_value: int | None, environ: dict | None = None) -> int:
    """Resolve this launch's decision-rounds setting: CLI flag, then environment, then default.

    ``environ`` defaults to the real process environment; a test supplies its own
    mapping so this stays a pure function with no monkeypatching required.
    """
    if cli_value is not None:
        return cli_value
    source = os.environ if environ is None else environ
    return _decision_rounds(source.get("BSH_DECISION_ROUNDS", str(_DEFAULT_DECISION_ROUNDS)))


def _resolve_language(cli_value: str | None, environ: dict | None = None) -> str:
    """Resolve this launch's table language: ``--language``, then ``BSH_LANGUAGE``, then en-US.

    Mirrors ``_resolve_decision_rounds`` immediately above, for the same reason: this
    file built ``NarratorConfig`` directly, with no ``language=`` argument at all, so
    neither ``--language`` (which did not exist) nor ``BSH_LANGUAGE`` ever reached the
    engine -- every launch narrated in English regardless of either one, silently,
    because ``narrator.config.load_config`` is the only reader of ``BSH_LANGUAGE`` and
    has no production caller (see that function's own docstring and
    ``_resolve_decision_rounds`` above, which independently notes the same gap for
    decision rounds). ``environ`` defaults to the real process environment; a test
    supplies its own mapping so this stays a pure function with no monkeypatching
    required.
    """
    if cli_value is not None:
        return cli_value
    source = os.environ if environ is None else environ
    return (source.get("BSH_LANGUAGE") or "en-US").strip() or "en-US"


def _resolve_external_diagnostic_target(target: str, campaign_root: Path) -> Path:
    """Resolve a diagnostic target only when it and its partial file avoid campaign data."""
    resolved_target = Path(target).expanduser().resolve()
    resolved_root = campaign_root.resolve()
    partial_target = resolved_target.with_name(f"{resolved_target.name}.partial")
    for candidate in (resolved_target, partial_target):
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            continue
        raise ValueError("--diagnostic-transcript must be outside the campaign root")
    return resolved_target


async def _main(
    arguments: argparse.Namespace,
    transcript_log,
    recorder: DiagnosticRecorder | None = None,
) -> int:
    if recorder is not None:
        recorder.record("session_start", channel=arguments.channel)
    config = NarratorConfig(
        campaign_root=Path(arguments.root).resolve(),
        repo_root=REPO_ROOT,
        base_url=arguments.base_url,
        model_id=arguments.model,
        max_settle_attempts=0
        if arguments.no_commit_nudge
        else NarratorConfig.max_settle_attempts,
        max_decision_rounds=getattr(arguments, "decision_rounds", _DEFAULT_DECISION_ROUNDS),
        turn_thinking_level=getattr(arguments, "turn_thinking_level", NarratorConfig.turn_thinking_level),
        turn_thinking_budgets=getattr(arguments, "turn_thinking_budgets", dict(DEFAULT_THINKING_BUDGETS)),
        language=getattr(arguments, "language", None) or NarratorConfig.language,
    )
    stdin_is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    stdout_is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    if arguments.channel == "terminal" and not (stdin_is_tty and stdout_is_tty):
        problem = "the terminal channel requires TTY standard input and output"
        print(
            f"error: {problem}; use --channel replay --script PATH for automation",
            file=sys.stderr,
        )
        _record_error(recorder, "terminal_not_tty")
        if recorder is not None:
            recorder.record("session_termination", state="preflight_failed")
        return 2
    problem = _preflight(config)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        _record_error(recorder, "preflight")
        if recorder is not None:
            recorder.record("session_termination", state="preflight_failed")
        return 2

    if arguments.dry_run:
        return _dry_run(config)

    if arguments.channel == "terminal":
        # The terminal renders in the table's own language. ``narrator.locale`` falls
        # back to English for a language with no catalog, so this never fails to start.
        from narrator.locale import load as load_catalog

        adapter = TerminalAdapter(
            recorder=recorder,
            catalog=load_catalog(config.language, config.locale_root, domain="terminal"),
        )
    else:
        adapter = _LimitedReplayAdapter(
            arguments.script,
            backfill_limit=config.backfill_limit,
            limit=arguments.turns,
            transcript_log=transcript_log,
        )
    service = build_narrator_service(config, adapter)
    install_signal_handlers(service)

    report = await service.run()
    summary = (
        f"\n{report.turns} turns, {report.delivered} delivered, "
        f"{report.withheld} withheld, {report.commits} commits, {report.waives} waives, "
        f"{report.leaks_scrubbed} leaks scrubbed"
    )
    print(summary)
    _record_output(recorder, summary + "\n")
    if transcript_log:
        transcript_log.write(summary + "\n")
    for error in report.errors:
        print(f"  error: {error}", file=sys.stderr)
        _record_error(recorder, error)
    status = 1 if report.errors else 0
    if recorder is not None:
        recorder.record(
            "session_summary",
            turns=report.turns,
            delivered=report.delivered,
            withheld=report.withheld,
            commits=report.commits,
            waives=report.waives,
            leaks_scrubbed=report.leaks_scrubbed,
        )
        recorder.record(
            "session_termination",
            state="completed" if status == 0 else "service_error",
        )
    return status


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. Extracted so a test can drive its real defaults.

    ``main`` calls this and nothing else to build the parser, so a test that calls
    ``_build_parser().parse_args([])`` sees exactly the same ``decision_rounds`` default
    (``None``) a real zero-argument launch parses, rather than a reimplementation that
    could drift from it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("BSH_CAMPAIGN_ROOT", str(REPO_ROOT)), help="campaign root")
    parser.add_argument("--channel", default="replay", choices=["replay", "terminal"])
    parser.add_argument("--script", default=str(REPO_ROOT / "scripts" / "demo-session.txt"))
    parser.add_argument("--base-url", default=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("BSH_LLM_MODEL", "google/gemma-4-26B-A4B-it"))
    parser.add_argument(
        "--turns", type=int, default=0, help="stop after this many turns; 0 runs all"
    )
    parser.add_argument(
        "--decision-rounds",
        type=_decision_rounds,
        default=None,
        help=(
            "structured decision rounds per turn; defaults to BSH_DECISION_ROUNDS or "
            f"{_DEFAULT_DECISION_ROUNDS}"
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=THINKING_LEVELS,
        default=None,
        help="the narrator's thinking level for this run; defaults to BSH_TURN_THINKING or low",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="the table's language tag for this run; defaults to BSH_LANGUAGE or en-US",
    )
    parser.add_argument("--transcript", default="", help="write the run log to this path")
    parser.add_argument(
        "--diagnostic-transcript",
        default="",
        help="write an opt-in terminal diagnostic JSON Lines file to this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list tools, resources, skills and configuration, then exit without a turn",
    )
    parser.add_argument(
        "--no-commit-nudge",
        action="store_true",
        help="disable the settle step; reproduces the unprompted "
        "arm of DEFER-SMALL-MODEL-COMMIT-DISCIPLINE",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    arguments = parser.parse_args()

    try:
        arguments.decision_rounds = _resolve_decision_rounds(arguments.decision_rounds)
    except argparse.ArgumentTypeError as error:
        parser.error(f"BSH_DECISION_ROUNDS {error}")

    # The thinking level and budgets resolve here, at the shell, for the same reason
    # the decision rounds do: a malformed BSH_TURN_THINKING* must stop the launch, not
    # the first turn. --thinking wins over the environment for the level alone.
    try:
        level, budgets = turn_thinking_from_env()
    except ValueError as error:
        parser.error(str(error))
    arguments.turn_thinking_level = arguments.thinking or level
    arguments.turn_thinking_budgets = budgets
    arguments.language = _resolve_language(arguments.language)

    if arguments.diagnostic_transcript and (
        arguments.channel != "terminal" or arguments.dry_run
    ):
        parser.error("--diagnostic-transcript requires an interactive terminal channel")

    diagnostic_target = None
    if arguments.diagnostic_transcript:
        try:
            diagnostic_target = _resolve_external_diagnostic_target(
                arguments.diagnostic_transcript,
                Path(arguments.root),
            )
        except ValueError as error:
            parser.error(str(error))
        except OSError as error:
            print(
                f"warning: diagnostic recorder unavailable: {type(error).__name__}",
                file=sys.stderr,
            )

    recorder = None
    if diagnostic_target is not None:
        try:
            recorder = DiagnosticTranscriptRecorder(diagnostic_target)
        except Exception as error:  # noqa: BLE001 - diagnostic capture cannot block gameplay
            print(
                f"warning: diagnostic recorder unavailable: {type(error).__name__}",
                file=sys.stderr,
            )

    completed = False
    try:
        if not arguments.transcript:
            status = asyncio.run(_main(arguments, None, recorder))
        else:
            with open(arguments.transcript, "w", encoding="utf-8") as handle:
                status = asyncio.run(_main(arguments, handle, recorder))
        completed = status == 0
        return status
    except BaseException as error:
        _record_error(recorder, type(error).__name__)
        if recorder is not None:
            recorder.record("session_termination", state="abnormal_exit")
        raise
    finally:
        if recorder is not None:
            try:
                result = recorder.close(complete=completed)
            except Exception as error:  # noqa: BLE001 - preserve the narrator exit status
                print(
                    f"warning: diagnostic recorder finalization failed: {type(error).__name__}",
                    file=sys.stderr,
                )
            else:
                if not result.completed:
                    reason = result.failure or "incomplete session"
                    print(
                        f"warning: diagnostic transcript remains at {result.partial_target}: {reason}",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    raise SystemExit(main())
