# Testing

Run `bash scripts/check.sh` for all offline gates. The rules-engine suite uses
`.venv`; the narrator suite uses `.narrator-venv` and excludes `live` tests.

```bash
uv run --locked pytest
BSH_SERVER_PYTHON="$PWD/.venv/bin/python" \
  .narrator-venv/bin/python -m pytest -c pytest-narrator.ini -m "not live" -q
```

Tests build isolated synthetic campaigns. No stored player campaign or historical
transcript is required. The release retains regression corpora, live probes,
and golden prompts as maintenance tools.

Use synthetic examples, not captured player dialogue, when adding a regression.
Keep source-session identifiers and private evidence paths out of comments and
case metadata. `tests/test_release_content.py` checks release-content hygiene.
Runtime messages, tool syntax, and original setting references are source assets,
not session records. Revised test inputs require fresh live measurements before
claiming classifier acceptance; offline success does not transfer an older result.

## Live checks

Live tests contact your configured model server and consume its capacity.
Run them deliberately against an already running endpoint:

```bash
BSH_SERVER_PYTHON="$PWD/.venv/bin/python" \
  .narrator-venv/bin/python -m pytest -c pytest-narrator.ini -m live -q
```

Use `BSH_LLM_BASE_URL` and `BSH_LLM_MODEL` where supported; individual probe
scripts also expose endpoint/model options in `--help`.
Do not infer a live guarantee from an offline pass.

To measure a long generated session:

```bash
BSH_SERVER_PYTHON="$PWD/.venv/bin/python" \
  .narrator-venv/bin/python scripts/soak_session.py \
  --bootstrap --messages 1050 --report /tmp/storyteller-soak.json
```

The harness has continuity, fact-depth, clock, zone, order, exchange, and hazard
instruments. Read `--help` for each probe's options and distinguish measured
checks from diagnostics that do not gate success. Prefix-cache metrics belong
to the shared server and are affected by other clients.

Some standalone regression probes require an output directory supplied by
`BSH_PROBE_RUN_DIRECTORY`. Create a fresh empty directory for each run.
`BSH_MODEL_LAUNCHER` optionally names your serving script for a recorded hash;
without it, that field is null. Missing external evidence must stay unmeasured.

## Prompt changes

Model prompts, tool descriptions, skills, schemas, and their property order can
change behavior even when the edit looks editorial. Preserve the shipped bytes
during unrelated maintenance.

For intentional changes, run the relevant live classifier, regression-corpus,
assessor, and persons-routing probes at ten repeats per case. Measure unaffected
fields too. Record the model, serving configuration, source revision, commands,
and results with synthetic data, then regenerate the goldens using
`scripts/capture_classifier_goldens.py`. Keep the resulting fixture diff in the
same change. See [golden acceptance](../tests_narrator/golden/ACCEPTANCE.md).

The test suites do not require internal development-agent tooling.
