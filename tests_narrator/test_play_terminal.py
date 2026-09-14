"""Test play terminal.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import play_terminal  # noqa: E402


class _FakeChild:
    """Stands in for the ``subprocess.Popen`` handle ``launch`` waits on."""

    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def wait(self) -> int:
        return self.exit_code


def _fake_project_python_bootstrap(monkeypatch) -> None:
    """Skip the real campaign bootstrap; these tests pin stream routing, not seeding."""
    monkeypatch.setattr(play_terminal, "bootstrap_campaign", lambda *args, **kwargs: None)


def test_launch_never_lets_the_narrator_child_inherit_this_processs_stderr(
    monkeypatch, tmp_path
):
    _fake_project_python_bootstrap(monkeypatch)
    captured_kwargs: dict = {}

    def fake_popen_factory(command, **kwargs):
        captured_kwargs.update(kwargs)
        # A real child writes its library warnings to the handle it was given. Doing
        # that here proves the handle this test receives is a real, writable file
        # rather than a placeholder the assertions below could pass by accident.
        stderr_handle = kwargs["stderr"]
        stderr_handle.write("reasoningContent is not supported in multi-turn ...\n")
        stderr_handle.flush()
        return _FakeChild()

    status_stream = io.StringIO()
    exit_code = play_terminal.launch(
        repo_root=REPO_ROOT,
        environment={},
        popen_factory=fake_popen_factory,
        diagnostic_transcript=tmp_path / "session.jsonl",
        status_stream=status_stream,
    )

    assert exit_code == 0
    stderr_argument = captured_kwargs["stderr"]
    # Never the two shapes that mean "inherit this process's own stderr": an
    # unspecified keyword defaults to None inside subprocess.Popen, and passing None
    # explicitly means the same thing.
    assert stderr_argument is not None
    assert stderr_argument is not subprocess.STDOUT

    status_log_path = tmp_path / "session.stderr.log"
    assert status_log_path.is_file()
    assert "reasoningContent" in status_log_path.read_text(encoding="utf-8")

    # The launcher's own status stream -- what a player would see printed alongside
    # the game -- carries the log's path, never the raw text the child wrote into it.
    launcher_output = status_stream.getvalue()
    assert str(status_log_path) in launcher_output
    assert "reasoningContent" not in launcher_output


def test_launch_falls_back_to_devnull_rather_than_inherit_when_the_log_cannot_open(
    monkeypatch, tmp_path
):
    _fake_project_python_bootstrap(monkeypatch)

    def broken_open_status_log(target):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(play_terminal, "_open_status_log", broken_open_status_log)

    captured_kwargs: dict = {}

    def fake_popen_factory(command, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeChild()

    status_stream = io.StringIO()
    exit_code = play_terminal.launch(
        repo_root=REPO_ROOT,
        environment={},
        popen_factory=fake_popen_factory,
        diagnostic_transcript=tmp_path / "session.jsonl",
        status_stream=status_stream,
    )

    assert exit_code == 0
    # The one acceptable fallback: discard, never inherit.
    assert captured_kwargs["stderr"] == subprocess.DEVNULL
    assert "warning: narrator status log unavailable" in status_stream.getvalue()


def test_launch_still_routes_stderr_away_with_diagnostics_disabled(monkeypatch, tmp_path):
    """``--no-diagnostic-transcript`` passes ``diagnostic_transcript=None``; routing must not depend on it."""
    _fake_project_python_bootstrap(monkeypatch)
    captured_kwargs: dict = {}

    def fake_popen_factory(command, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeChild()

    status_stream = io.StringIO()
    exit_code = play_terminal.launch(
        repo_root=REPO_ROOT,
        environment={"HOME": str(tmp_path), "XDG_STATE_HOME": str(tmp_path / "state")},
        popen_factory=fake_popen_factory,
        diagnostic_transcript=None,
        status_stream=status_stream,
    )

    assert exit_code == 0
    assert captured_kwargs["stderr"] not in (None, subprocess.STDOUT)
    assert "Narrator status log:" in status_stream.getvalue()
    # No diagnostic transcript line, because recording is off.
    assert "Diagnostic transcript:" not in status_stream.getvalue()


def test_default_status_log_target_is_a_sibling_of_the_diagnostic_naming_scheme(tmp_path):
    environment = {"HOME": str(tmp_path), "XDG_STATE_HOME": str(tmp_path / "state")}
    target = play_terminal.default_status_log_target(REPO_ROOT, environment)

    assert target.suffix == ".log"
    assert target.name.endswith(".stderr.log")
    assert target.parent == (tmp_path / "state" / "storyteller" / "diagnostics")
