"""Construct the production ``NarratorService`` object graph.

Was a function inside ``scripts/narrator_serve.py`` (the CLI entry point) until 8
probe scripts under ``scripts/`` also needed it and imported it from there,
turning an entry point into a de facto library module. This is the shared home.
"""

from __future__ import annotations

from bsh_mcp.service import GameService
from narrator.config import NarratorConfig
from narrator.interactions import read_trusted_scope
from narrator.service import NarratorService


def build_narrator_service(config: NarratorConfig, adapter) -> NarratorService:
    """Construct the production narrator with its authoritative trade boundary."""
    game = GameService(config.campaign_root)
    game.provision_scene_merchants(read_trusted_scope(config.campaign_root).present_npc_ids)
    return NarratorService(config, adapter, purchase_executor=game.narrator_purchase_executor())
