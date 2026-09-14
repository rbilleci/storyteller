#!/usr/bin/env python3
"""Launch one disposable terminal campaign.

Three entry points, one per way a table starts:

- default / ``--premade barbarian``: the recorded Rill campaign, exactly as before.
- ``--premade civilised`` / ``--premade decadent``: the same seeded scene with that
  origin's premade character, so every supported origin has a jump-in path.
- ``--session-zero``: a fresh campaign with no characters and no seeded scene; the
  narrator runs the ``bsh-session-zero`` procedure and creates the character through
  the tools.

Each premade is created through ``GameService.character_create`` with a fixed seed,
so its rolled sheet is reproducible run to run and still fully audited.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bsh_mcp.service import GameService, build_service  # noqa: E402

TERMINAL_PLAYER_ID = "terminal-player"
RILL_BACKGROUNDS = ["scout", "hunter", "survivor"]
SESSION_ZERO = "session-zero"


PREMADES: dict[str, dict] = {
    "barbarian": {
        "name": "Rill",
        "backgrounds": RILL_BACKGROUNDS,
        "weapons": ["long knife"],
        "armour": "none",
        "shield": False,
        "seed": 1101,
    },
    "civilised": {
        "name": "Maren",
        "backgrounds": ["sword-master", "bodyguard", "diplomat"],
        "weapons": None,
        "armour": "light",
        "shield": True,
        "seed": 1102,
    },
    "decadent": {
        "name": "Vessa",
        "backgrounds": ["forbidden-knowledge", "snake-blood", "vicious"],
        "weapons": None,
        "armour": "none",
        "shield": False,
        "seed": 1103,
    },
}

_DEFAULT_DIAGNOSTIC = object()


def resolve_narrator_python(repo_root: Path, environment: Mapping[str, str]) -> str:
    """Resolve the narrator interpreter without changing its environment."""
    return environment.get("NARRATOR_PYTHON") or str(
        repo_root / ".narrator-venv" / "bin" / "python"
    )


def default_diagnostic_target(
    repo_root: Path,
    environment: Mapping[str, str],
    *,
    now: datetime | None = None,
    process_id: int | None = None,
    token: str | None = None,
) -> Path:
    """Create the private state directory and return one unique transcript target."""
    xdg_value = environment.get("XDG_STATE_HOME", "")
    xdg_root = Path(xdg_value) if xdg_value else None
    if xdg_root is not None and xdg_root.is_absolute():
        state_root = xdg_root
    else:
        home_value = environment.get("HOME", "")
        home = Path(home_value) if home_value else Path.home()
        if not home.is_absolute():
            raise ValueError("HOME must be absolute when XDG_STATE_HOME is not absolute")
        state_root = home / ".local" / "state"

    diagnostic_root = (state_root / "storyteller" / "diagnostics").resolve()
    try:
        diagnostic_root.relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("the default diagnostic directory must be outside the repository")

    application_root = diagnostic_root.parent
    application_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    application_root.chmod(0o700)
    diagnostic_root.mkdir(mode=0o700, exist_ok=True)
    diagnostic_root.chmod(0o700)

    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    pid = process_id if process_id is not None else os.getpid()
    random_token = token or secrets.token_hex(8)
    return diagnostic_root / f"terminal-{timestamp}-{pid}-{random_token}.jsonl"


def default_status_log_target(
    repo_root: Path,
    environment: Mapping[str, str],
    *,
    now: datetime | None = None,
    process_id: int | None = None,
    token: str | None = None,
) -> Path:
    """Return one unique target for the narrator child's routed-away status stream.
    """
    return default_diagnostic_target(
        repo_root, environment, now=now, process_id=process_id, token=token
    ).with_suffix(".stderr.log")


def _open_status_log(target: Path) -> TextIO:
    """Create the target's parent directory and open it for writing, mode 0600."""
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = target.open("w", encoding="utf-8")
    os.chmod(target, 0o600)
    return handle


def bootstrap_campaign(
    repo_root: Path,
    campaign_root: Path,
    project_python: str = sys.executable,
    mode: str = "barbarian",
) -> None:
    """Copy immutable sources and prepare the campaign for the selected mode.

    ``mode`` is an origin key from ``PREMADES`` — seed the scene and create that
    origin's premade through the service — or ``SESSION_ZERO``, which leaves the
    campaign with no characters and no seeded scene so the narrator runs the
    ``bsh-session-zero`` procedure and every sheet enters through the tools.
    """
    if mode != SESSION_ZERO and mode not in PREMADES:
        raise ValueError(f"unknown terminal campaign mode: {mode!r}")
    shutil.copytree(repo_root / "rules", campaign_root / "rules")
    shutil.copytree(repo_root / "world", campaign_root / "world")
    bootstrap_environment = os.environ.copy()
    bootstrap_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        project_python,
        str(repo_root / "scripts" / "new_campaign.py"),
        "--root",
        str(campaign_root),
    ]
    if mode != SESSION_ZERO:
        command.append("--seed-scene")
    subprocess.run(command, check=True, env=bootstrap_environment)

    if mode == SESSION_ZERO:
        game = GameService(campaign_root)
        if game.store.character_ids():
            raise RuntimeError("a session-zero campaign must start with no characters")
        if game.store.read_players().players:
            raise RuntimeError("a session-zero campaign must start with no player links")
        if game.store.read_state().scene.title:
            raise RuntimeError("a session-zero campaign must start with no seeded scene")
        return

    premade = PREMADES[mode]
    game = build_service(campaign_root, seed=premade["seed"])
    created = game.character_create(
        discord_user_id=TERMINAL_PLAYER_ID,
        name=premade["name"],
        origin=mode,
        backgrounds=list(premade["backgrounds"]),
        weapons=premade["weapons"],
        armour=premade["armour"],
        shield=premade["shield"],
    )
    if not created.get("ok"):
        raise RuntimeError(f"{premade['name']} creation failed: {created}")

    character_id = created["character_id"]
    character_ids = game.store.character_ids()
    character = game.store.read_character(character_id)
    players = game.store.read_players().players
    if character_ids != [character_id] or character.name != premade["name"]:
        raise RuntimeError(
            f"terminal campaign does not contain one {premade['name']} character"
        )
    if character.origin != mode:
        raise RuntimeError(f"{premade['name']} does not carry the {mode} origin")
    if character.discord_user_id != TERMINAL_PLAYER_ID:
        raise RuntimeError(f"{premade['name']} does not link to terminal-player")
    if len(players) != 1 or players[0].discord_user_id != TERMINAL_PLAYER_ID:
        raise RuntimeError("terminal campaign does not contain one player link")
    if players[0].character_id != character_id:
        raise RuntimeError("terminal-player links to the wrong character")
    if not game.store.read_state().scene.title:
        raise RuntimeError("terminal campaign lacks the seeded scene")


def child_environment(
    environment: Mapping[str, str], project_python: str = sys.executable
) -> dict[str, str]:
    """Build the narrator child environment without overwriting an MCP override."""
    child = dict(environment)
    child.setdefault("BSH_SERVER_PYTHON", project_python)
    child.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return child


def serve_command(
    repo_root: Path,
    campaign_root: Path,
    environment: Mapping[str, str],
    diagnostic_transcript: Path | None = None,
) -> list[str]:
    """Build the terminal narrator command with no gameplay options."""
    command = [
        resolve_narrator_python(repo_root, environment),
        str(repo_root / "scripts" / "narrator_serve.py"),
        "--root",
        str(campaign_root),
        "--channel",
        "terminal",
    ]
    if diagnostic_transcript is not None:
        command.extend(["--diagnostic-transcript", str(diagnostic_transcript)])
    return command


def preserve_campaign_evidence(
    campaign_root: Path, transcript_target: Path | None, messages: TextIO
) -> None:
    """Copy the final state and event log out of the disposable campaign.

    The campaign lives under a ``TemporaryDirectory`` that vanishes at exit, so its
    ``state.json`` and ``campaign/logs/events.jsonl`` — the only record of which tools
    ran and what state they left — die with it. This copies both beside the diagnostic
    transcript before that deletion. It fails open: a copy error prints a warning and
    changes neither the session nor its exit code, because retention must never cost a
    turn. Retention rides with the diagnostic transcript, so a disabled transcript
    disables it too.
    """
    if transcript_target is None:
        return
    sources = {
        transcript_target.with_suffix(".state.json"): campaign_root / "campaign" / "state.json",
        transcript_target.with_suffix(".events.jsonl"): campaign_root
        / "campaign"
        / "logs"
        / "events.jsonl",
    }
    for destination, source in sources.items():
        try:
            if not source.is_file():
                continue
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o600)
        except OSError as error:
            print(
                f"warning: could not preserve {source.name}: {type(error).__name__}",
                file=messages,
                flush=True,
            )


class _SignalExit(Exception):  # noqa: N818 -- internal control flow, not a public error
    """Carry a handled launcher signal until its child exits."""

    def __init__(self, number: int) -> None:
        self.number = number


def _raise_signal_exit(number, frame) -> None:
    del frame
    raise _SignalExit(number)


def launch(
    repo_root: Path = REPO_ROOT,
    environment: Mapping[str, str] | None = None,
    project_python: str = sys.executable,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    diagnostic_transcript: Path | None | object = _DEFAULT_DIAGNOSTIC,
    status_stream: TextIO | None = None,
    mode: str = "barbarian",
) -> int:
    """Run the terminal child and remove its campaign root when it exits."""
    parent_environment = dict(environment or os.environ)
    messages = status_stream if status_stream is not None else sys.stderr
    selected_transcript = diagnostic_transcript
    if selected_transcript is _DEFAULT_DIAGNOSTIC:
        try:
            selected_transcript = default_diagnostic_target(repo_root, parent_environment)
        except OSError as error:
            selected_transcript = None
            print(
                f"warning: diagnostic recorder unavailable: {type(error).__name__}",
                file=messages,
                flush=True,
            )
    if selected_transcript is not None:
        selected_transcript = Path(selected_transcript).expanduser().resolve()
        print(
            f"Diagnostic transcript: {selected_transcript}",
            file=messages,
            flush=True,
        )


    status_log_target: Path | None = None
    status_log_handle: TextIO | None = None
    try:
        status_log_target = (
            selected_transcript.with_suffix(".stderr.log")
            if selected_transcript is not None
            else default_status_log_target(repo_root, parent_environment)
        )
        status_log_handle = _open_status_log(status_log_target)
    except OSError as error:
        status_log_target = None
        status_log_handle = None
        print(
            f"warning: narrator status log unavailable: {type(error).__name__}",
            file=messages,
            flush=True,
        )
    if status_log_target is not None:
        print(f"Narrator status log: {status_log_target}", file=messages, flush=True)
    # A status log that could not be opened must still never fall back to inheriting
    # this process's stderr -- that inheritance is the defect this exists to remove.
    child_stderr = status_log_handle if status_log_handle is not None else subprocess.DEVNULL

    previous_handlers = {
        number: signal.signal(number, _raise_signal_exit)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    child = None
    try:
        with tempfile.TemporaryDirectory(prefix="storyteller-terminal-") as temporary_root:
            campaign_root = Path(temporary_root)
            bootstrap_campaign(repo_root, campaign_root, project_python, mode)
            environment_for_child = child_environment(parent_environment, project_python)
            child = popen_factory(
                serve_command(
                    repo_root,
                    campaign_root,
                    parent_environment,
                    selected_transcript,
                ),
                env=environment_for_child,
                stderr=child_stderr,
            )
            try:
                exit_code = child.wait()
            except _SignalExit as interruption:
                child.send_signal(interruption.number)
                exit_code = child.wait()
            # Copy the evidence out before the TemporaryDirectory block deletes it.
            preserve_campaign_evidence(campaign_root, selected_transcript, messages)
            return exit_code
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        if status_log_handle is not None:
            status_log_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    diagnostic_group = parser.add_mutually_exclusive_group()
    diagnostic_group.add_argument(
        "--diagnostic-transcript",
        type=Path,
        default=_DEFAULT_DIAGNOSTIC,
        help="write the terminal diagnostic JSON Lines file to this path",
    )
    diagnostic_group.add_argument(
        "--no-diagnostic-transcript",
        action="store_const",
        const=None,
        dest="diagnostic_transcript",
        help="disable terminal diagnostic recording",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--premade",
        choices=sorted(PREMADES),
        default="barbarian",
        help="jump into the seeded scene with this origin's premade character",
    )
    mode_group.add_argument(
        "--session-zero",
        action="store_const",
        const=SESSION_ZERO,
        dest="premade",
        help="start with no characters; the narrator runs the bsh-session-zero procedure",
    )
    arguments = parser.parse_args()
    try:
        return launch(
            diagnostic_transcript=arguments.diagnostic_transcript,
            mode=arguments.premade,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
