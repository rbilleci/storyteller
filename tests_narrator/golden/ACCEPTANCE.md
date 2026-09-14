# Classifier and assessor prompt contract

The files in this directory pin the rendered classifier, assessor, and companion
prompts and strict schemas, including property order. They are source fixtures
consumed by `test_classifier_contract.py`; they are not temporary output.

The initial release preserves the existing prompt and schema bytes. Historical
experiment journals are not included. No new live acceptance is implied by
packaging this release. See [known limitations](../../DEFERRED.md) for current
classification and language weaknesses.

For a prompt or schema change:

1. Run the relevant classifier, regression-corpus, assessor, and persons-routing
   live probes at ten repeats per case, including fields the edit did not target.
2. Record model and serving versions, source revision, commands, results, and any
   regressions using synthetic data.
3. Regenerate with `scripts/capture_classifier_goldens.py`.
4. Review and commit the fixture diff with the change.

The offline contract test must pass, but cannot replace live remeasurement.
