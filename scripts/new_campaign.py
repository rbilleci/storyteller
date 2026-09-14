#!/usr/bin/env python3
"""Create or reset the campaign directory.

Usage::

    uv run python scripts/new_campaign.py --title "The Ashen Bell"
    uv run python scripts/new_campaign.py --title "The Ashen Bell" --force
    uv run python scripts/new_campaign.py --seed-scene

``--force`` overwrites existing campaign files. It never touches ``world/`` or
``rules/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bsh_mcp.models import CampaignState, Scene, Statement  # noqa: E402
from bsh_mcp.store import CampaignError, CampaignStore, _atomic_write_text  # noqa: E402

STARTING_SCENE = Scene(
    location_id="the-eel-market",
    title="The eel market at low water",
    summary=(
        "Low tide has emptied the channel below Vey. The eel market runs on planks above "
        "black mud. Sera Vane has not rung the evening bell for two days, and the "
        "fishmongers have started to notice."
    ),
    statements=[
        Statement(text="The bell tower above the market stands silent.", source="authored"),
        Statement(text="Sera Vane, the bell-keeper, has not been seen for two days.", source="authored"),
        Statement(
            text="Salt Magistrate clerks are counting barrels and asking few questions.",
            source="authored",
        ),
        Statement(
            text="Sera Vane walked to the road shrine at night and did not return.",
            scope="world_hidden", source="authored",
        ),
        Statement(
            text="A Choir Below listener watches the market from the drowned customs house.",
            scope="world_hidden", source="authored",
        ),
    ],
    exits=["the-drowned-customs-house", "the-road-shrine"],
    hooks=[
        "Find out why the bell-keeper stopped ringing the evening bell.",
        "Learn what the Choir Below wants from the estuary bells.",
    ],
    present_npcs=[],
)


def seed_authored_persons(store: CampaignStore, state: CampaignState) -> list[str]:
    """Seed ``scene.persons`` from the scene location's ``visible_entities``.
    """
    persons = dict(state.scene.persons)
    seeded: list[str] = []
    for entity_id, person in store.authored_persons_for(state.scene.location_id).items():
        if entity_id in persons or entity_id in state.npcs:
            continue
        persons[entity_id] = person
        seeded.append(entity_id)
    state.scene.persons = persons
    return seeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT), help="project root directory")
    parser.add_argument("--title", default="The Ashen Bell", help="campaign title")
    parser.add_argument("--force", action="store_true", help="overwrite an existing campaign")
    parser.add_argument(
        "--seed-scene",
        action="store_true",
        help="write the starter Black Estuary scene into campaign/state.json",
    )
    arguments = parser.parse_args()

    store = CampaignStore(Path(arguments.root))
    try:
        store.initialize(title=arguments.title, force=arguments.force)
    except CampaignError as error:
        print(f"error: {error.message}", file=sys.stderr)
        for step in error.allowed_next_steps:
            print(f"  next: {step}", file=sys.stderr)
        return 1

    if arguments.seed_scene:
        state: CampaignState = store.read_state()
        state.scene = STARTING_SCENE.model_copy(deep=True)
        seed_authored_persons(store, state)
        _atomic_write_text(
            store.state_path, json.dumps(state.model_dump(), indent=2, ensure_ascii=False) + "\n"
        )
        _atomic_write_text(store.scene_path, store.render_scene_markdown(state))

    print(f"campaign ready at {store.campaign_dir}")
    print(f"  manifest: {store.manifest_path}")
    print(f"  state:    {store.state_path}")
    print(f"  scene:    {store.scene_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
