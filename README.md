# Storyteller

A Black Sword Hack game master with a deterministic rules engine and an AI narrator.
The narrator creates fiction; the engine rolls dice and records campaign state through
the Model Context Protocol (MCP). Includes the original Black Estuary setting,
terminal play, scripted replay, and localized interfaces.

This is an experimental source release for Linux with Python 3.11 or newer.
Run from a checkout or extracted source archive; standalone wheel installs are not supported.
Narration requires a separately running model endpoint. Dice and storage tests run
without one. See [known limitations](DEFERRED.md) before using it for a campaign.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these
commands from the repository root:

```bash
uv sync --locked --group dev
uv venv --python 3.11 .narrator-venv
uv pip install --python .narrator-venv/bin/python -r requirements-narrator.txt
uv pip install --python .narrator-venv/bin/python -e . --no-deps
export BSH_SERVER_PYTHON="$PWD/.venv/bin/python"
export NARRATOR_PYTHON="$PWD/.narrator-venv/bin/python"
```

The two environments are required: the MCP server needs MCP 2.x, while
Strands 1.50.2 requires MCP 1.x. Keep the narrator's editable install at
`--no-deps` to avoid installing the server's conflicting MCP requirement there.
The narrator requirements include the shared rules-engine dependencies.

The current narrator is configured for `google/gemma-4-26B-A4B-it` at
`http://localhost:8000/v1`. See [model setup](docs/model-setup.md) for the
serving requirements. The launchers do not manage the model server.

```bash
scripts/play-session-zero.sh
# Or start with a pregenerated character:
scripts/play-premade-barbarian.sh
scripts/play-premade-civilised.sh
scripts/play-premade-decadent.sh
```

Each launch creates a fresh campaign through the engine and removes that temporary
campaign at exit. No existing campaign or recorded play session ships in this repository.

## At the terminal

Use `/help` for commands and `/character` to inspect your audited sheet.
Use `/language fr` to select a catalog and request narration in that language.
`/thinking off|low|medium|high` changes the narrator's thinking budget.
Enter a numbered decision, or a custom approach as `N. text`.
`/dismiss` cancels a decision. Recovery notices may offer `/retry`,
`/continue`, or `/revise text`. `/quit`, Control-C, or Control-D exits.

Catalogs ship for English, French, German, Spanish, Portuguese, Korean, Chinese,
Japanese, Finnish, Swedish, Norwegian Bokmål, Lithuanian, Dutch, and Russian,
including regional variants under [locale/](locale/). Set `BSH_LANGUAGE` for
the launch default. The narrator receives a language instruction as well as
localized engine messages; natural-language quality still depends on the model.

The launcher records diagnostics by default in
`$XDG_STATE_HOME/storyteller/diagnostics` or
`$HOME/.local/state/storyteller/diagnostics`. These include player input,
delivered narration, and final campaign/audit snapshots. Disable transcript and
snapshot recording with:

```bash
uv run python scripts/play_terminal.py --no-diagnostic-transcript
```

A private stderr log is still written for troubleshooting. Treat diagnostics and
campaign backups as private; redact them before sharing. See [security](SECURITY.md).

## Persistent campaigns and MCP

Create the campaign in this checkout (it is ignored by Git), then validate it:

```bash
uv run python scripts/new_campaign.py --title "The Ashen Bell" --seed-scene
uv run python scripts/validate_campaign.py
.narrator-venv/bin/python scripts/narrator_serve.py --dry-run
```

The dry run lists the served tools without contacting the model. To run a persistent
terminal session or expose the rules engine to an MCP client:

```bash
.narrator-venv/bin/python scripts/narrator_serve.py --channel terminal
uv run bsh-mcp
```

For a separate campaign root, copy `rules/` and `world/` there, create with
`scripts/new_campaign.py --root /absolute/path/to/campaign-root`, and set
`BSH_CAMPAIGN_ROOT` or pass `--root` to the narrator. The root contains a
`campaign/` subdirectory. Never hand-edit an active campaign.
Creation refuses to overwrite existing state unless explicitly given `--force`.

At session end, ask the narrator to close the session; `session_close` writes
a summary and backup. Unresolved outcomes must be settled or explicitly accepted.
You can also run `bash scripts/backup_campaign.sh` for an additional backup.

For scripted play against a generated two-player campaign:

```bash
bash scripts/demo_sandbox.sh scripts/demo-combat.txt
```

This prints the temporary campaign path and retains it for inspection.
[config/env.example](config/env.example) documents environment variables.
The application does not automatically load `.env`; export its values in your shell.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the offline checks,
[architecture](docs/architecture.md) for the trust boundaries,
and [testing](docs/testing.md) for the regression and live probe suites.
Prompts, tool descriptions, game skills, and golden fixtures are behavior-sensitive
assets; changing them requires live remeasurement.

## License

Project contributions are licensed under **GNU GPL version 3 only**.
See [LICENSE](LICENSE). Black Sword Hack SRD material retains **CC BY 4.0**;
see [NOTICE](NOTICE) and [rules attribution](rules/attribution.md) for scope,
credits, source links, and adaptations.

Black Sword Hack is by Alexandre "Kobayashi" Jeannette, published by The Merry Mushmen.
Storyteller is an independent project. Model weights and third-party packages are
obtained separately under their own licenses.
