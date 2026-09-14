# Contributing

Install the two environments described in [README.md](README.md).
The supported development workflow is Linux, Python 3.11+, and uv.
Run the offline acceptance gates from the repository root:

```bash
bash scripts/check.sh
```

The command runs Ruff, Python compilation, the rules-engine tests, and the
narrator tests excluding those marked `live`. It defaults to `.venv` and
`.narrator-venv`, with `BSH_SERVER_PYTHON` and `NARRATOR_PYTHON` overrides.
It works from a source archive without Git metadata. Node.js is optional;
`npm run check` delegates to the same command.

Optional Git hooks:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook runs fast checks and pre-push runs the offline suites.
The hooks do not invoke a model endpoint. See [testing](docs/testing.md) for
live checks, prompt-change acceptance, and recording reproducible results.

Campaign mutations must pass through the engine's store and MCP tools.
A narrated die result, HP total, Doom change, or death without an audit event
is a defect. Keep diagnostics off the MCP stdout protocol stream.
Preserve the equality checks on the served and model-facing tool surfaces.

Write American English in code and documentation. Preserve SRD vocabulary
such as `defence`. Do not mechanically rewrite model-facing prompts, game
skills, tool descriptions, locale content, or golden fixtures: their wording
affects behavior. Add tests for changed mechanics and trust boundaries.

Submit only content you have the right to contribute. Project contributions
are accepted under GPL-3.0-only; preserve existing third-party notices and
identify sources and licenses for any imported material. SRD adaptations must
retain the attribution and modification notices in [NOTICE](NOTICE).
Do not include private campaign state, credentials, model weights, diagnostics,
local environments, or generated development-agent configuration in a contribution.
Use newly authored synthetic fixtures instead of copying play-session text into
tests, probes, comments, or documentation. Keep regression checks but omit private
session identifiers and source-log references.
