# Architecture

Two interpreters communicate over stdio. `bsh_mcp` owns mechanics and durable
campaign state; the Strands narrator orchestrates fiction and structured decisions.
The MCP server requires MCP 2.x; Strands requires MCP 1.x.

## Rules and storage

`dice.py` provides injectable randomness. `rules.py` and `effects.py` resolve
mechanics; `service/` exposes operations and `server.py` wires the MCP surface.
`models.py` validates campaign files and `data.py` loads the rules tables.

`CampaignStore.transaction` takes the campaign lock, loads and validates state,
applies an operation, revalidates it, replaces files atomically, and appends an
audit event. Validation failure before commit writes nothing. File replacement
is atomic individually; this is not a crash-atomic multi-file database or an
event-sourced store. Root confinement protects campaign paths.

The campaign root contains `campaign/`, `rules/`, and `world/`.
The application checkout separately supplies prompts, skills, and catalogs.
Campaign files are runtime data and are ignored in the source repository.

## Narration and delivery

The engine prepends a bounded canon digest, classifies the player's intent,
applies the relevant decision and consent policy, and runs the narrator agent.
Classification facets share a schema-forced call; narrow companion calls add
bindings for specific contexts. Recorded state and model-sourced persons have
different trust levels. The assessor, classifier, planner, and settler are
separate from the channel agent's optional thinking.

Mechanical outcomes create fiction debt. Settlement executes a validated
`scene_commit` or an audited `ledger_settle` waiver. The sweep can record durable
facts already stated in otherwise uncommitted narration. Pending rulings are
resolved by engine orchestration. Session closing refuses unresolved debt unless
the caller explicitly accepts it, preserving the outstanding outcomes.

`policy.py` equality-checks both tool surfaces before play. The model cannot call
`ledger_settle` or `ability_apply_ruling`. `delivery.py` is the outbound
boundary: it withholds unsettled/faulted turns and strips tool markup. Guards
do not establish the truth of all unconstrained prose.

A channel adapter yields inbound turns, posts engine-authorized output, and closes.
It does not own the ledger or contact the model. The terminal adds structured
decisions, sheet rendering, and private diagnostic recording; replay supplies
scripted turns. Social purchases require authenticated confirmation before mutation.

## Content and localization

`SOUL.md`, the game skills, `config/narrator-context.md`, and prompt builders
supply model instructions. They are behavior-sensitive source assets.
`locale/` contains narrator and terminal catalogs. The English catalog defines
the schema; translations must preserve placeholders, recovery controls, and keys.
The active language also informs narration. Rules and authored world files are
not automatically translated.

See [limitations](../DEFERRED.md) for unresolved behavior and
[testing](testing.md) for the prompt-change contract.
