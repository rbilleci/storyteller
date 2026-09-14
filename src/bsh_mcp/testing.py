"""Shared setup for the probe scripts under ``scripts/probe_*.py``.

Nine probe scripts each build a disposable campaign to measure one narrator
behavior against a live model endpoint. Comparing all nine found their setup
genuinely diverges past the first several lines -- each authors its own NPCs,
combat arrangement, and starting location for its own scenario, and forcing
that into one shared function would be a false uniformity, not a
simplification. What is byte-identical across all nine is the campaign
bootstrap before any of that: copy ``rules``/``world``, initialize the store,
create the one player character, and confirm it landed. That is what this
module holds, plus two small measurement utilities every "live" probe also
duplicated (``candidate_tree``, ``resolve_model_id``/``resolve_model_and_digest``).

A caller continues past :func:`bootstrap_probe_campaign` with its own
scenario-specific setup, exactly as it does today.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from .service import GameService


def bootstrap_probe_campaign(
    root: Path,
    *,
    title: str,
    name: str = "Rill",
    origin: str = "barbarian",
    backgrounds: tuple[str, ...] = ("scout", "hunter", "survivor"),
    weapons: list[str] | None = None,
    discord_user_id: str = "probe-player",
) -> tuple[GameService, str]:
    """Copy ``rules``/``world``, initialize the store, and create one character.

    Returns ``(game, character_id)``. ``character_id`` is the id
    ``character_create`` actually assigned -- read this rather than assuming a
    slug of ``name``, the mistake ``scripts/probe_social_interactions.py`` made
    hardcoding ``"rill"`` into its own ``players.yaml`` write.

    ``weapons=None`` rolls starting weapons from the origin table instead of
    fixing one, which is what a non-barbarian origin (e.g. Vessa the decadent)
    needs -- ``character_create`` treats ``None`` and "omitted" identically.
    """
    repository = Path(__file__).resolve().parents[2]
    shutil.copytree(repository / "rules", root / "rules")
    shutil.copytree(repository / "world", root / "world")
    game = GameService(root)
    game.store.initialize(title=title)
    created = game.character_create(
        discord_user_id=discord_user_id,
        name=name,
        origin=origin,
        backgrounds=list(backgrounds),
        weapons=weapons,
    )
    if not created.get("ok"):
        raise ValueError(f"probe character was unavailable: {created}")
    return game, str(created["character_id"])


def candidate_tree(run_directory: Path) -> str:
    """Name the tree this run measures, or say the run is unbound.
    """
    parent = run_directory.resolve().parent.name
    if len(parent) == 40 and all(character in "0123456789abcdef" for character in parent):
        return parent
    return "unbound"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolve_model_id(endpoint: str) -> str:
    """The served model's id, with no validation beyond the endpoint answering."""
    with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as response:
        payload = json.loads(response.read())
    return str(payload["data"][0]["id"])


def resolve_model_and_digest(endpoint: str) -> dict:
    """The served model's id plus a digest of the full ``/models`` response.

    The digest lets a report show whether two runs measured the identical
    served configuration (quantization, LoRA, sampling defaults) without
    reprinting the whole payload -- ``resolve_model_id`` above is enough when a
    probe only needs the id itself.
    """
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
