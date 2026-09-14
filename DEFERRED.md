# Known limitations

This experimental release needs a human game master to review surprising narration.
The rules engine owns dice and durable state, but model compliance and recall remain
imperfect. Passing offline tests is not a claim that every live narration is correct.

## Narration and memory

The model can omit a required tool, invent a fact, or expose reasoning in ordinary
prose. Settlement and delivery guards catch specific failure classes, not arbitrary
semantic claims. Tool-call markup is filtered; reasoning prose is harder to recognize.
A bounded canon digest and sliding conversation window can lose older context during
long or multi-location campaigns. The full state is retained, but recall is not
guaranteed. Repeated facts, stale statements, and incorrect role references can affect
later narration. Review campaign records and retain backups.

Classification can vary with model version, serving configuration, language, and
concurrent load. Known weak cases include role-based interlocutor references and
declarations ending in a question. Catalog structure is validated automatically;
translations and multilingual test labels still need native-speaker review.

## Channels and rules

Terminal and replay are implemented. Discord requires a new channel adapter;
credentials alone do not enable it. The historical `discord_user_id` field is
currently also used for channel account identifiers.

Many supernatural power effects require narrator judgment; the engine automates
their dice, resource costs, and selected consequences. Fictional preconditions
cannot all be verified mechanically. See [rules attribution](rules/attribution.md)
for house rules and [SRD reference](docs/srd/dark-pacts.md) for power descriptions.

The event journal supports auditing, not complete event replay or undo.
Atomic file replacement is per file; process failure during a multi-file commit
can leave partial state. Preserve backups and validate before resuming.

## Development notes

Regression tests use synthetic examples and generic rule declarations. Recorded
session excerpts, source-session references, internal planning documents,
development-agent files, and pre-release Git history are not part of this release.
Application messages and original setting content remain as testable source assets.
Current contracts are documented in
[architecture](docs/architecture.md) and [testing](docs/testing.md).
