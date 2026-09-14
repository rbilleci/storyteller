"""Release setup works from source archives and honors endpoint configuration."""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import narrator_serve  # noqa: E402
import play_terminal  # noqa: E402


def test_narrator_interpreter_resolution_requires_no_git_metadata(tmp_path):
    assert play_terminal.resolve_narrator_python(tmp_path, {}) == str(
        tmp_path / ".narrator-venv" / "bin" / "python"
    )
    assert play_terminal.resolve_narrator_python(
        tmp_path, {"NARRATOR_PYTHON": "/custom/python"}
    ) == "/custom/python"


def test_cli_reads_deployment_environment_and_explicit_flags_win(monkeypatch, tmp_path):
    monkeypatch.setenv("BSH_CAMPAIGN_ROOT", str(tmp_path))
    monkeypatch.setenv("BSH_LLM_BASE_URL", "http://localhost:8765/v1")
    monkeypatch.setenv("BSH_LLM_MODEL", "test-model")
    args = narrator_serve._build_parser().parse_args([])
    assert (args.root, args.base_url, args.model) == (
        str(tmp_path), "http://localhost:8765/v1", "test-model"
    )
    args = narrator_serve._build_parser().parse_args([
        "--root", "/explicit", "--base-url", "http://localhost:9999/v1",
        "--model", "explicit-model",
    ])
    assert (args.root, args.base_url, args.model) == (
        "/explicit", "http://localhost:9999/v1", "explicit-model"
    )


def test_server_interpreter_default_requires_no_git_metadata(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "release_test_config", REPO_ROOT / "tests_narrator" / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("BSH_SERVER_PYTHON", raising=False)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    assert module.resolve_bsh_mcp_interpreter() == str(tmp_path / ".venv" / "bin" / "python")


def test_logprob_probe_builds_unique_case_ids_without_external_metadata():
    from probe_classifier_logprobs import _prompts

    prompts = _prompts()
    assert prompts
    assert len({item["id"] for item in prompts}) == len(prompts)
