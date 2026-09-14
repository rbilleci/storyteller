"""Combat probe artifact tests without a live model endpoint.

The probe's value is its live measurement, which cannot run on every validation pass
against a shared graphics processing unit (GPU). What runs here is the probe's own
contract: it writes one run-bound report, and it refuses to overwrite one. A probe that
silently rewrote its report could publish a passing run over a failing one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import resolve_bsh_mcp_interpreter  # noqa: E402

PROBE = "scripts/probe_combat_decisions.py"
INTERPRETER = resolve_bsh_mcp_interpreter()


def test_combat_probe_writes_one_redacted_run_bound_report(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "BSH_PROBE_RUN_DIRECTORY": str(tmp_path)}
    result = subprocess.run(
        [INTERPRETER, PROBE, "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = tmp_path / "combat-production-probe.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["offline"] is True
    assert payload["outputs"] == []
    assert payload["checks"] == {}
    assert "model_id" not in payload  # an offline run records no endpoint identity

    repeat = subprocess.run(
        [INTERPRETER, PROBE, "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert repeat.returncode == 75


def test_the_combat_probe_requires_a_run_directory(tmp_path: Path):
    """Without a run directory the probe holds rather than writing evidence anywhere."""
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "BSH_PROBE_RUN_DIRECTORY"}
    result = subprocess.run(
        [INTERPRETER, PROBE, "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 75
    assert "BSH_PROBE_RUN_DIRECTORY" in result.stderr


def test_the_probe_counts_only_completed_mechanics():
    """A refused call must never read as a roll, which is the probe's whole discriminator.

    ``NarratorEngine._tool_log`` records an attempted call. A probe reading it would
    pass while ``combat_defend`` raised and no defence resolved. This pins the signal
    that separates the two, so a later edit cannot quietly widen it back to attempts.

    The snippet runs under the storyteller interpreter because the probe imports
    ``bsh_mcp``, which needs ``mcp`` 2.x, while this suite runs on ``mcp`` 1.29.0.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts');"
        "from probe_combat_decisions import _RuntimeProbeAdapter as Adapter;"
        "a = Adapter('dodge');"
        "a.record('tool_call', tool='combat_defend', ok=False, error='incoming_damage_required',"
        " event_id=None);"
        "assert a.completed_tools == set(), a.completed_tools;"
        "a.record('tool_call', tool='combat_defend', ok=True, error='', event_id=7);"
        "assert a.completed_tools == {'combat_defend'}, a.completed_tools;"
        "a.record('decision_lifecycle', category='answered');"
        "assert a.completed_tools == {'combat_defend'}, a.completed_tools;"
        "print('DISCRIMINATOR_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "DISCRIMINATOR_OK" in result.stdout


def test_the_runtime_mode_refuses_an_incomplete_invocation(tmp_path: Path):
    """The runtime helper never guesses a scenario, campaign root, or model."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [INTERPRETER, PROBE, "--runtime", "--scenario", "defence"],
        cwd=root, env={**os.environ}, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2


def test_a_mechanic_check_reads_completion_and_never_attempt():
    """Drive the mapping the live flow uses, which no offline test could reach before.

    An audit noted that repointing a mechanic check from the completed set back to the
    attempted set would leave this suite green, because `_runtime_flow` needs a served
    model. `scenario_checks` is that mapping, extracted so this test drives it directly.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=1, errors=False, delivered=1, withheld=0,"
        " output_categories=['narration'], decision_kinds=[]);"
        # A call the engine attempted and the mechanic refused must not read as a pass.
        "refused = s('defence', **base, attempted={'combat_defend'}, completed=set());"
        "assert refused['defence_completed'] is False, refused;"
        "assert refused['defence_attempted'] is True, refused;"
        "rolled = s('defence', **base, attempted={'combat_defend'}, completed={'combat_defend'});"
        "assert rolled['defence_completed'] is True, rolled;"
        "attack = dict(base, decision_kinds=['confirmation']);"
        "half = s('attack', **attack, attempted={'combat_attack'},"
        " completed=set());"
        "assert half['attack_completed'] is False, half;"


        "full = dict(attack, attempted={'combat_attack'},"
        " completed={'combat_attack'});"
        "unlisted = s('unlisted-violence', **full);"
        "matching_attack = s('attack', **full);"
        "assert unlisted == matching_attack, (unlisted, matching_attack);"


        "assert unlisted['no_confirmation_presented'] is False, unlisted;"
        "assert unlisted['attack_completed'] is True, unlisted;"


        "withheld_base = dict(turns=1, errors=False, delivered=0, withheld=1,"
        " output_categories=['narration'], decision_kinds=[]);"
        "withheld = s('sanctioned-attack', **withheld_base, attempted=set(), completed=set());"
        "assert withheld['turn_delivered'] is False, withheld;"
        "assert withheld['sanctioned_without_confirmation'] is False, withheld;"
        "sanctioned = s('sanctioned-attack', **base, attempted=set(), completed=set());"
        "assert sanctioned['turn_delivered'] is True, sanctioned;"
        "assert sanctioned['sanctioned_without_confirmation'] is True, sanctioned;"
        "print('MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "MAPPING_OK" in result.stdout


def test_the_refusal_account_mapping_reads_the_two_typed_signals_not_prose():
    """Neither of the refusal-account scenario's two extra checks can come from parsing
    model text in this mapping: both are booleans the adapter already reduced text to
    before this function ever runs. A call that flips either boolean must flip exactly
    the check it names, and nothing else in the two-turn shape.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=2, errors=False, delivered=1, withheld=0,"
        " output_categories=['notice', 'narration'], decision_kinds=[],"
        " attempted=set(), completed=set());"
        "good = s('refusal-account', **base, withheld_notice_matched=True,"
        " reply_names_true_reason=True);"
        "assert good['service_completed'] is True, good;"
        "assert good['withheld_turn_recorded'] is True, good;"
        "assert good['reply_names_true_reason'] is True, good;"
        "wrong_notice = s('refusal-account', **base, withheld_notice_matched=False,"
        " reply_names_true_reason=True);"
        "assert wrong_notice['withheld_turn_recorded'] is False, wrong_notice;"
        "assert wrong_notice['reply_names_true_reason'] is True, wrong_notice;"
        "invented = s('refusal-account', **base, withheld_notice_matched=True,"
        " reply_names_true_reason=False);"
        "assert invented['withheld_turn_recorded'] is True, invented;"
        "assert invented['reply_names_true_reason'] is False, invented;"
        # A one-turn run, or narration leaking before the notice, must not read as a pass.
        "short = s('refusal-account', **dict(base, turns=1));"
        "assert short['service_completed'] is False, short;"
        "swapped = s('refusal-account', **dict(base, output_categories=['narration', 'notice']));"
        "assert swapped['narration_posted'] is False, swapped;"
        "print('REFUSAL_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "REFUSAL_MAPPING_OK" in result.stdout


def test_the_refusal_adapter_reduces_replies_to_booleans_and_retains_no_text():
    """The adapter must never carry raw model text past ``post``, matching every other
    probe adapter's redaction discipline in this module.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import asyncio, sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import _RefusalAccountAdapter as Adapter;"
        "a = Adapter('No action occurred because this hazard requires an authenticated"
        " confirmation.');"
        "asyncio.run(a.post('docks', 'No action occurred because this hazard requires an"
        " authenticated confirmation.'));"
        "assert a.withheld_notice_matched is True, a.withheld_notice_matched;"
        "assert a.output_categories == ['notice'], a.output_categories;"
        "asyncio.run(a.post('docks', 'You were refused because the hazard needed an"
        " authenticated confirmation you never gave.'));"
        "assert a.reply_names_true_reason is True, a.reply_names_true_reason;"
        "assert a.output_categories == ['notice', 'narration'], a.output_categories;"
        # Redaction: no attribute may still hold the reply text itself. The engine's own
        # static notice is expected to live in ``_expected_notice`` -- that string is
        # config-authored, known before the run starts, and never model output -- so the
        # check targets the reply's distinguishing phrase instead of a shared word.
        "stored = [v for v in vars(a).values() if isinstance(v, str) and 'needed an' in v.casefold()];"
        "assert stored == [], stored;"
        "b = Adapter('No action occurred because this hazard requires an authenticated"
        " confirmation.');"
        "asyncio.run(b.post('docks', 'x'));"
        "asyncio.run(b.post('docks', 'The stars wheeled overhead and nothing was explained.'));"
        "assert b.reply_names_true_reason is False, b.reply_names_true_reason;"
        "print('REDACTION_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "REDACTION_OK" in result.stdout


def test_the_ledger_gate_recovery_mapping_reads_the_ledger_gate_and_notice_signals():
    """M9's live mapping, driven offline like the others.

    A ``ledger_gate`` fault, or a false "no action occurred" post, must each flip
    exactly the check that names it -- the two direct pins this scenario exists
    for -- and nothing else in the shape. ``service_completed`` deliberately ignores
    ``errors`` here: a guard withhold is this scenario's own expected outcome, and
    ``NarratorService`` records that withhold reason into ``ServiceReport.errors``
    the same way it records a genuine framework fault, so a clean two-turn run with
    a legitimate withhold must still read as a pass.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=2, errors=False, output_categories=['narration', 'narration'],"
        " decision_kinds=['confirmation'], attempted=set(),"
        " completed={'npc_create', 'combat_start', 'combat_begin_turn', 'combat_attack'});"
        # The clean case: both turns delivered, no ledger-gate fault, the ledger closed.
        "clean = s('ledger-gate-recovery', **base, delivered=2, withheld=0,"
        " ledger_gate_faults=0, npc_created=True, ledger_ratified_after_run=True,"
        " false_silence_notice_posted=False);"
        "assert clean['service_completed'] is True, clean;"
        "assert clean['turn_delivered'] is True, clean;"
        "assert clean['no_ledger_gate_deadlock'] is True, clean;"
        "assert clean['no_false_silence_notice'] is True, clean;"
        "assert clean['ledger_ratified_after_run'] is True, clean;"
        # The guard-withhold-then-recover case this milestone's fix targets: turn 1
        # withholds (its own resolving roll never ran) with a truthful notice, turn 2
        # delivers cleanly, with no unrelated turn in between -- still a pass, and
        # ``errors`` being True from the legitimate withhold must not sink it.
        "recovered = s('ledger-gate-recovery', **dict(base, errors=True,"
        " output_categories=['incomplete_notice', 'narration']), delivered=1, withheld=1,"
        " ledger_gate_faults=0, npc_created=True, ledger_ratified_after_run=True,"
        " false_silence_notice_posted=False);"
        "assert recovered['service_completed'] is True, recovered;"
        "assert recovered['turn_delivered'] is True, recovered;"
        "assert recovered['no_ledger_gate_deadlock'] is True, recovered;"
        "assert recovered['no_false_silence_notice'] is True, recovered;"
        # A run where both turns legitimately withhold with no progress of their own
        # -- the model made two cautious calls, neither mutated state -- still passes
        # every M9 pin: no jam, no false claim, ledger clean. Only turn_delivered
        # fails, which is a model-capability signal, not this milestone's claim.
        "cautious = s('ledger-gate-recovery', **dict(base, errors=True,"
        " output_categories=['incomplete_notice', 'notice']), delivered=0, withheld=2,"
        " ledger_gate_faults=0, npc_created=True, ledger_ratified_after_run=True,"
        " false_silence_notice_posted=False);"
        "assert cautious['service_completed'] is True, cautious;"
        "assert cautious['turn_delivered'] is False, cautious;"
        "assert cautious['no_ledger_gate_deadlock'] is True, cautious;"
        "assert cautious['no_false_silence_notice'] is True, cautious;"
        # The exact live-transcript failure, reproduced in the mapping: the retry
        # faults at the ledger gate. This must fail this specific check and no other.
        "jammed = s('ledger-gate-recovery', **dict(base, errors=True, output_categories=["
        "'incomplete_notice', 'notice']), delivered=0, withheld=2,"
        " ledger_gate_faults=1, npc_created=True, ledger_ratified_after_run=False,"
        " false_silence_notice_posted=False);"
        "assert jammed['no_ledger_gate_deadlock'] is False, jammed;"
        "assert jammed['ledger_ratified_after_run'] is False, jammed;"
        # The false-silence failure, isolated: a notice post whose own turn's tool
        # calls really did change state, with no ledger-gate fault at all -- the live
        # transcript's FIRST failure shape, distinct from its second.
        "silenced = s('ledger-gate-recovery', **dict(base, errors=True, output_categories=["
        "'notice', 'narration']), delivered=1, withheld=1,"
        " ledger_gate_faults=0, npc_created=True, ledger_ratified_after_run=True,"
        " false_silence_notice_posted=True);"
        "assert silenced['no_false_silence_notice'] is False, silenced;"
        "assert silenced['no_ledger_gate_deadlock'] is True, silenced;"
        # A run that never even created the fishmonger did not exercise the
        # reproduction at all and must not read as a pass.
        "skipped = s('ledger-gate-recovery', **dict(base, delivered=2, withheld=0,"
        " ledger_gate_faults=0, npc_created=False, ledger_ratified_after_run=True,"
        " false_silence_notice_posted=False));"
        "assert skipped['fishmonger_npc_created'] is False, skipped;"
        "print('LEDGER_GATE_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "LEDGER_GATE_MAPPING_OK" in result.stdout


def test_the_ledger_gate_recovery_adapter_classifies_notices_and_retains_no_text():
    """The adapter must never carry raw model text past ``post``, matching every other
    probe adapter's redaction discipline in this module, while still telling the two
    known engine-authored notices apart from narration and from each other, and
    attributing each turn's own successful tool calls to that turn alone.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import asyncio, sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import _LedgerGateRecoveryAdapter as Adapter;"
        "a = Adapter('No action occurred because the game master could not prepare a"
        " safe decision. Use /retry, /revise, or /dismiss.', 'INCOMPLETE-NOTICE-SENTINEL');"
        # Turn 1: npc_create succeeds, then the guard withholds -- a truthful
        # 'incomplete_notice' should follow, carrying turn 1's own tool success.
        "a.record('tool_call', tool='npc_create', ok=True, error='', event_id=1);"
        "asyncio.run(a.post('docks', 'INCOMPLETE-NOTICE-SENTINEL'));"
        # Turn 2: nothing succeeds this time -- a truthful 'notice' follows, carrying
        # an empty tool set.
        "a.record('tool_call', tool='combat_attack', ok=False, error='not_active_actor',"
        " event_id=2);"
        "asyncio.run(a.post('docks', 'No action occurred because the game master could"
        " not prepare a safe decision. Use /retry, /revise, or /dismiss.'));"
        "asyncio.run(a.post('docks', 'Rill draws steel and the fishmonger stumbles"
        " back, blood at his shoulder.'));"
        "assert a.output_categories == ['incomplete_notice', 'notice', 'narration'], a.output_categories;"
        "assert a.tools_by_turn == [{'npc_create'}, set(), set()], a.tools_by_turn;"
        "assert a.completed_tools == {'npc_create'}, a.completed_tools;"
        "a.record('decision_lifecycle', category='answered');"
        "a.record('decision_lifecycle', category='fault', failure_category='ledger_gate');"
        "assert a.decision_events == ["
        "{'category': 'answered', 'failure_category': ''},"
        "{'category': 'fault', 'failure_category': 'ledger_gate'}"
        "], a.decision_events;"
        # Redaction: no stored attribute may hold the real narration text.
        "leaked = [v for v in vars(a).values() if isinstance(v, str) and 'blood at his' in v];"
        "assert leaked == [], leaked;"
        "print('LEDGER_GATE_REDACTION_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "LEDGER_GATE_REDACTION_OK" in result.stdout


def test_the_clarification_convergence_mapping_reads_the_segment_and_repeat_signals():
    """Three failing shapes, each isolated to the one check that names it: a segment
    notice reached while holding answers, a question re-asked, and a run where the
    planner never asked anything at all -- which exercises nothing this milestone is
    about and must not read as a pass.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=1, errors=False, delivered=1, withheld=0,"
        " output_categories=['narration'], decision_kinds=['clarification', 'clarification'],"
        " attempted=set(), completed=set());"
        "clean = s('clarification-converges', **base, views_presented=2, answers_given=2,"
        " repeated_question_count=0, segment_faults=0);"
        "assert all(clean.values()), clean;"

        "discarded = s('clarification-converges', **dict(base, delivered=0,"
        " output_categories=['notice']), views_presented=2, answers_given=2,"
        " repeated_question_count=0, segment_faults=1);"
        "assert discarded['no_segment_notice'] is False, discarded;"
        "assert discarded['no_repeated_question'] is True, discarded;"
        "assert discarded['clarification_presented'] is True, discarded;"
        "assert discarded['every_view_answered'] is True, discarded;"
        # A re-asked question flips that check and nothing else.
        "repeated = s('clarification-converges', **base, views_presented=2, answers_given=2,"
        " repeated_question_count=1, segment_faults=0);"
        "assert repeated['no_repeated_question'] is False, repeated;"
        "assert repeated['no_segment_notice'] is True, repeated;"
        "assert repeated['turn_delivered'] is True, repeated;"
        # A run that asked nothing proves nothing about a narrowing chain.
        "vacuous = s('clarification-converges', **dict(base, decision_kinds=[]),"
        " views_presented=0, answers_given=0, repeated_question_count=0, segment_faults=0);"
        "assert vacuous['clarification_presented'] is False, vacuous;"
        "assert vacuous['no_segment_notice'] is True, vacuous;"
        "assert vacuous['turn_delivered'] is True, vacuous;"
        # A view presented and never answered is a different defect, and it is caught.
        "unanswered = s('clarification-converges', **base, views_presented=2, answers_given=1,"
        " repeated_question_count=0, segment_faults=0);"
        "assert unanswered['every_view_answered'] is False, unanswered;"
        "print('CONVERGENCE_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "CONVERGENCE_MAPPING_OK" in result.stdout


def test_the_new_declaration_mapping_reads_the_stale_confirmation_and_two_booleans():
    """The decisive check is typed: a confirmation presented on the second turn can only
    come from the retained attack, because the second declaration carries no violence.
    The two reduced booleans are separate checks, so a model that answered the right
    action and a service that planned the wrong one cannot cancel out.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=2, errors=False, delivered=1, withheld=0,"
        " output_categories=['notice', 'narration'], decision_kinds=['confirmation'],"
        " attempted=set(), completed=set());"
        "clean = s('new-declaration-heard', **base,"
        " stale_confirmations_after_the_new_declaration=0, reply_names_new_action=True);"
        "assert all(clean.values()), clean;"
        # The exact pre-repair shape: the retained attack drove turn two, so the risk
        # floor asked for a second confirmation about an action the player abandoned.
        "stale = s('new-declaration-heard', **dict(base,"
        " decision_kinds=['confirmation', 'confirmation']),"
        " stale_confirmations_after_the_new_declaration=1, reply_names_new_action=True);"
        "assert stale['no_stale_confirmation'] is False, stale;"
        "assert stale['reply_addresses_the_new_declaration'] is True, stale;"
        "assert stale['first_turn_withheld_with_notice'] is True, stale;"
        # A reply that never mentions the new action flips only its own check. There is
        # deliberately no converse check forbidding a mention of the abandoned one; see
        # the comment above ``_NEW_DECLARATION_KEYWORDS``' neighbour in the probe.
        # A planner question about the new action is not a stale confirmation, so a run
        # carrying an extra decision kind on turn two still passes: only the risk
        # floor's own re-fire counts (see ``_NewDeclarationAdapter``).
        "planner_asked = s('new-declaration-heard', **dict(base,"
        " decision_kinds=['confirmation', 'clarification']),"
        " stale_confirmations_after_the_new_declaration=0, reply_names_new_action=True);"
        "assert all(planner_asked.values()), planner_asked;"
        "silent = s('new-declaration-heard', **base,"
        " stale_confirmations_after_the_new_declaration=0, reply_names_new_action=False);"
        "assert silent['reply_addresses_the_new_declaration'] is False, silent;"
        "assert silent['no_stale_confirmation'] is True, silent;"
        "assert 'reply_omits_the_abandoned_action' not in silent, silent;"
        # Turn one must actually have withheld, or the retention never happened.
        "no_notice = s('new-declaration-heard', **dict(base,"
        " output_categories=['narration', 'narration']),"
        " stale_confirmations_after_the_new_declaration=0, reply_names_new_action=True);"
        "assert no_notice['first_turn_withheld_with_notice'] is False, no_notice;"
        "print('NEW_DECLARATION_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "NEW_DECLARATION_MAPPING_OK" in result.stdout


def test_the_withheld_recovery_and_phantom_scene_mappings_read_their_own_signals():
    """Both scenarios tolerate either turn delivering or withholding, and both drop the
    ``not errors`` half of ``service_completed``, because a guard withhold is their
    expected outcome and ``NarratorService`` records its reason into
    ``ServiceReport.errors`` exactly as it records a framework fault. Each also carries
    its own premise check, so a run that never reached the branch under test fails
    rather than passing empty.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=2, errors=True, delivered=1, withheld=1,"
        " output_categories=['control_notice', 'narration'],"
        " decision_kinds=['confirmation', 'confirmation'], attempted=set(),"
        " completed={'npc_create', 'combat_start'});"
        "armed = s('withheld-recovery-armed', **base, control_notices=1,"
        " unarmed_control_notices=0);"
        "assert all(armed.values()), armed;"
        # The pre-repair shape: ``deliver`` posted a notice naming /retry and armed
        # nothing, so the terminal answered "Unknown command" to its own advice.
        "unarmed = s('withheld-recovery-armed', **base, control_notices=1,"
        " unarmed_control_notices=1);"
        "assert unarmed['every_control_notice_armed_recovery'] is False, unarmed;"
        "assert unarmed['control_notice_posted'] is True, unarmed;"
        "assert unarmed['service_completed'] is True, unarmed;"
        # A run where no notice named a control never exercised the advertisement.
        "quiet = s('withheld-recovery-armed', **dict(base, delivered=2, withheld=0,"
        " output_categories=['narration', 'narration']), control_notices=0,"
        " unarmed_control_notices=0);"
        "assert quiet['control_notice_posted'] is False, quiet;"
        "assert quiet['every_control_notice_armed_recovery'] is True, quiet;"
        "assert quiet['turn_delivered'] is True, quiet;"
        # no-phantom-scene, same setup, different reading.
        "clean = s('no-phantom-scene', **base, engine_scene_commits=0, turns_withheld=1,"
        " ledger_ratified_after_run=True);"
        "assert all(clean.values()), clean;"
        # The defect: the engine's own settle step committed a scene for a turn nobody
        # saw. Only its check flips.
        "phantom = s('no-phantom-scene', **base, engine_scene_commits=1, turns_withheld=1,"
        " ledger_ratified_after_run=True);"
        "assert phantom['no_engine_scene_commit'] is False, phantom;"
        "assert phantom['turn_withheld'] is True, phantom;"
        "assert phantom['ledger_ratified_after_run'] is True, phantom;"
        # A model that issued its own scene_commit is not this defect: the count the
        # check reads is already the difference, so a model-only commit reads zero.
        "model_only = s('no-phantom-scene', **base, engine_scene_commits=0,"
        " turns_withheld=1, ledger_ratified_after_run=True);"
        "assert model_only['no_engine_scene_commit'] is True, model_only;"
        # A run where nothing withheld never reached the branch that wrote the scene.
        "delivered_both = s('no-phantom-scene', **dict(base, delivered=2, withheld=0),"
        " engine_scene_commits=0, turns_withheld=0, ledger_ratified_after_run=True);"
        "assert delivered_both['turn_withheld'] is False, delivered_both;"
        "assert delivered_both['no_engine_scene_commit'] is True, delivered_both;"
        # M9's guarantee still has to hold on this scenario's own run.
        "dirty = s('no-phantom-scene', **base, engine_scene_commits=0, turns_withheld=1,"
        " ledger_ratified_after_run=False);"
        "assert dirty['ledger_ratified_after_run'] is False, dirty;"
        "print('M10_WITHHELD_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "M10_WITHHELD_MAPPING_OK" in result.stdout


def test_every_scenario_name_reaches_both_the_gate_and_the_runtime_dispatch():
    """A scenario absent from ``SCENARIOS`` is a scenario the gate never runs.

    ``SCENARIOS`` is simultaneously the argparse ``choices`` and the report loop, so
    adding a name adds it to the gate -- and forgetting to add one leaves a scenario
    that passes by never running. This also pins that every name dispatches to a real
    adapter rather than falling through to the default declaration branch, which would
    silently measure the attack scenario four extra times.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import inspect, sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "import probe_combat_decisions as probe;"
        "expected = {'defence', 'attack', 'refusal-account', 'unlisted-violence',"
        " 'sanctioned-attack', 'disengage', 'ledger-gate-recovery',"
        " 'clarification-converges', 'new-declaration-heard', 'withheld-recovery-armed',"
        " 'no-phantom-scene', 'mechanical-truth', 'weapon-named-sanction',"
        " 'ooc-question-in-fight', 'new-engagement-confirms',"
        " 'maneuver-direct-ruling', 'outlandish-priced-not-granted'};"
        "assert set(probe.SCENARIOS) == expected, probe.SCENARIOS;"
        "assert len(probe.SCENARIOS) == len(expected), probe.SCENARIOS;"
        "flow = inspect.getsource(probe._runtime_flow);"
        "missing = [name for name in probe._M10_SCENARIOS if name not in flow];"
        "assert missing == [], missing;"
        "checks = inspect.getsource(probe.scenario_checks);"
        "absent = [name for name in probe._M10_SCENARIOS if name not in checks];"
        "assert absent == [], absent;"
        # M11's scenario dispatches to its own adapter and its own check block for the
        # same reason M10's four do: falling through to the default branch would
        # silently measure the attack scenario one more time under a new name.
        "assert 'mechanical-truth' in flow, flow[:200];"
        "assert '_MechanicalTruthAdapter' in flow, flow[:200];"
        "assert 'mechanical-truth' in checks, checks[:200];"
        # M14's new scenario dispatches to its own adapter and its own check block too,
        # for the same reason: a fallthrough would silently measure the attack scenario
        # under a fifteenth name instead of proving anything about a fresh engagement.
        "assert 'new-engagement-confirms' in flow, flow[:200];"
        "assert 'new-engagement-confirms' in checks, checks[:200];"
        # The combat-pace pair declares through the declarations mapping (the same
        # shape defence and unlisted-violence use) and each owns its own check block
        # and its own state-read extra in the flow.
        "assert 'maneuver-direct-ruling' in flow, flow[:200];"
        "assert 'maneuver-direct-ruling' in checks, checks[:200];"
        "assert 'outlandish-priced-not-granted' in flow, flow[:200];"
        "assert 'outlandish-priced-not-granted' in checks, checks[:200];"
        "setup = inspect.getsource(probe._prepare_campaign);"
        "assert '_M10_SCENARIOS' in setup, setup[:200];"
        "assert 'new-engagement-confirms' in setup, setup[:200];"
        "print('SCENARIO_REGISTRY_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "SCENARIO_REGISTRY_OK" in result.stdout


def test_the_m10_adapters_reduce_every_signal_and_retain_no_prose():
    """Family C for milestone M10's three adapters, one property each.

    The convergence adapter counts a re-asked question without keeping the question:
    it stores a digest of the normalized text, so a repeat is countable and no prose
    survives. The new-declaration adapter attributes each presented decision kind to the
    turn that presented it and reduces the reply to two booleans. The withheld-recovery
    adapter resolves an unarmed notice lazily -- at the next post, or at ``close`` --
    because ``_post_notice`` posts first and arms second, and it separates the model's
    own ``scene_commit`` calls from everything else.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import asyncio, sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import _ClarificationConvergenceAdapter as Conv,"
        " _NewDeclarationAdapter as NewDecl, _WithheldRecoveryAdapter as Withheld;"
        "from narrator.channels.base import DecisionDeliveryReceipt;"
        "from narrator.decisions import DecisionView, PublicOption;"
        "options = (PublicOption(id='near', label='The near way'),"
        " PublicOption(id='own_approach', label='Describe your own approach', custom=True));"
        "view = lambda question, token: DecisionView(presentation_token=token,"
        " kind='clarification', character_id='rill', character_name='Rill',"
        " question=question, options=options);"
        "c = Conv(('SEGMENT-NOTICE-SENTINEL',));"
        "asyncio.run(c.deliver_decision_views((view('Which way to the tower?', 't' * 32),)));"


        "asyncio.run(c.deliver_decision_views((view('which way to the tower', 'u' * 32),)));"
        "asyncio.run(c.deliver_decision_views((view('How much noise will you make?', 'v' * 32),)));"
        "assert c.views_presented == 3, c.views_presented;"
        "assert c.repeated_question_count == 1, c.repeated_question_count;"
        "asyncio.run(c.post('docks', 'SEGMENT-NOTICE-SENTINEL'));"
        "asyncio.run(c.post('docks', 'The tower door hangs open on a dark stair.'));"
        "assert c.output_categories == ['notice', 'narration'], c.output_categories;"
        "c.record('decision_lifecycle', category='segment', failure_category='view_limit');"
        "assert c.decision_events == [{'category': 'segment',"
        " 'failure_category': 'view_limit'}], c.decision_events;"
        "leaked = [v for v in vars(c).values() if isinstance(v, str) and 'tower' in v.casefold()];"
        "assert leaked == [], leaked;"
        "nested = [x for v in vars(c).values() if isinstance(v, list) for x in v"
        " if isinstance(x, str) and 'tower' in x.casefold()];"
        "assert nested == [], nested;"
        "n = NewDecl();"
        "asyncio.run(n.deliver_decision_views((view('Confirm?', 'w' * 32),)));"
        "asyncio.run(n.post('docks', 'notice text'));"
        "asyncio.run(n.post('docks', 'Rill sits on the crates and counts out her coins.'));"
        "assert n.risk_confirmations_by_turn == [0, 0], n.risk_confirmations_by_turn;"
        "assert n.reply_names_new_action is True, n.reply_names_new_action;"
        # A view whose question is not the risk floor's own is a planner question about
        # the new action, which is legitimate and must not be counted.
        "asyncio.run(n.deliver_decision_views((view('Do you keep a hand on the purse?',"
        " 'y' * 32),)));"
        "asyncio.run(n.post('docks', 'She keeps a hand on it.'));"
        "assert n.risk_confirmations_by_turn == [0, 0, 0], n.risk_confirmations_by_turn;"
        # The defect's own shape: the risk floor's one fixed question, re-presented on a
        # turn that declared no violence at all.
        "from narrator.decisions import risk_confirmation;"
        "stale = NewDecl();"
        "asyncio.run(stale.post('docks', 'notice text'));"
        "asyncio.run(stale.deliver_decision_views("
        "(view(risk_confirmation().question, 'x' * 32),)));"
        "asyncio.run(stale.post('docks', 'Rill lunges at the fishmonger Rade once more.'));"
        "assert stale.risk_confirmations_by_turn == [0, 1], stale.risk_confirmations_by_turn;"
        "assert stale.reply_names_new_action is False, stale.reply_names_new_action;"
        "held = [v for v in vars(n).values() if isinstance(v, str) and 'crates' in v.casefold()];"
        "assert held == [], held;"
        "w = Withheld();"
        # A notice naming a control, followed by the arming call: not a violation.
        "asyncio.run(w.post('docks', 'No action occurred. Use /retry, /revise, or /dismiss.'));"
        "asyncio.run(w.decision_recovery());"
        "asyncio.run(w.post('docks', 'Rill hesitates, blade half drawn.'));"
        "assert w.control_notices == 1 and w.unarmed_control_notices == 0,"
        " (w.control_notices, w.unarmed_control_notices);"
        # A notice naming a control with no arming call: resolved at the next post.
        "asyncio.run(w.post('docks', 'Use /continue, /revise, or /dismiss.'));"
        "asyncio.run(w.post('docks', 'The market goes on around her.'));"
        "assert w.unarmed_control_notices == 1, w.unarmed_control_notices;"
        # And one still pending at shutdown is resolved by close, which the service
        # always calls in its finally block.
        "asyncio.run(w.post('docks', 'Use /retry, /revise, or /dismiss.'));"
        "asyncio.run(w.close());"
        "assert w.unarmed_control_notices == 2, w.unarmed_control_notices;"
        "assert w.recovery_arms == 1, w.recovery_arms;"
        "w.record('tool_call', tool='scene_commit', ok=True, error='', event_id=5);"
        "w.record('tool_call', tool='scene_commit', ok=False, error='empty', event_id=None);"
        "w.record('tool_call', tool='npc_create', ok=True, error='', event_id=6);"
        "assert w.model_scene_commits == 1, w.model_scene_commits;"
        "assert w.completed_tools == {'scene_commit', 'npc_create'}, w.completed_tools;"
        "spilled = [v for v in vars(w).values() if isinstance(v, str) and 'blade' in v.casefold()];"
        "assert spilled == [], spilled;"
        "print('M10_ADAPTER_REDACTION_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "M10_ADAPTER_REDACTION_OK" in result.stdout


def test_the_mechanical_truth_mapping_reads_the_three_mismatch_kinds_and_two_premises():
    """The two premise checks are pinned in both directions on purpose. A run that rolled
    nothing, or that rolled and announced nothing, must not read as a pass -- with no
    announcement to check, all three mismatch counts are zero and the milestone's own
    claims would otherwise report satisfied by a run that measured nothing at all.
    That is the exact way this probe could stop testing anything without going red.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=2, errors=True, delivered=1, withheld=1,"
        " output_categories=['narration', 'narration'],"
        " decision_kinds=['confirmation'], attempted=set(),"
        " completed={'combat_begin_turn', 'combat_attack'});"
        "clean = s('mechanical-truth', **base, rolls_recorded=2, rolls_announced=2,"
        " verdict_mismatch_count=0, target_mismatch_count=0,"
        " unbacked_announcement_count=0);"
        "assert all(clean.values()), clean;"

        "inverted = s('mechanical-truth', **base, rolls_recorded=2, rolls_announced=2,"
        " verdict_mismatch_count=1, target_mismatch_count=0,"
        " unbacked_announcement_count=0);"
        "assert inverted['every_announced_verdict_matches'] is False, inverted;"
        "assert inverted['every_announced_target_matches'] is True, inverted;"
        "assert inverted['no_unbacked_announcement'] is True, inverted;"
        "assert inverted['rolls_announced'] is True, inverted;"

        "invented = s('mechanical-truth', **base, rolls_recorded=2, rolls_announced=2,"
        " verdict_mismatch_count=0, target_mismatch_count=3,"
        " unbacked_announcement_count=0);"
        "assert invented['every_announced_target_matches'] is False, invented;"
        "assert invented['every_announced_verdict_matches'] is True, invented;"
        # A line announcing dice that never rolled at all.
        "phantom = s('mechanical-truth', **base, rolls_recorded=2, rolls_announced=3,"
        " verdict_mismatch_count=0, target_mismatch_count=0,"
        " unbacked_announcement_count=1);"
        "assert phantom['no_unbacked_announcement'] is False, phantom;"
        "assert phantom['every_announced_verdict_matches'] is True, phantom;"
        # Premise one: a run whose dice never rolled proves nothing either way.
        "unrolled = s('mechanical-truth', **base, rolls_recorded=0, rolls_announced=0,"
        " verdict_mismatch_count=0, target_mismatch_count=0,"
        " unbacked_announcement_count=0);"
        "assert unrolled['rolls_recorded'] is False, unrolled;"
        "assert unrolled['rolls_announced'] is False, unrolled;"
        "assert unrolled['every_announced_verdict_matches'] is True, unrolled;"
        # Premise two: dice rolled, nothing announced. That is the dice-visibility
        # defect rather than this one, and it must still not read as a pass here.
        "silent = s('mechanical-truth', **base, rolls_recorded=2, rolls_announced=0,"
        " verdict_mismatch_count=0, target_mismatch_count=0,"
        " unbacked_announcement_count=0);"
        "assert silent['rolls_recorded'] is True, silent;"
        "assert silent['rolls_announced'] is False, silent;"
        # A run where both turns jammed is not a pass either.
        "jammed = s('mechanical-truth', **dict(base, delivered=0, withheld=2),"
        " rolls_recorded=2, rolls_announced=2, verdict_mismatch_count=0,"
        " target_mismatch_count=0, unbacked_announcement_count=0);"
        "assert jammed['turn_delivered'] is False, jammed;"
        "print('M11_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "M11_MAPPING_OK" in result.stdout


def test_the_mechanical_truth_adapter_keeps_only_numbers_and_compares_against_the_log():
    """Family C for M11's adapter, plus the two pure helpers the live flow depends on.

    The adapter must reduce a reply to announcement fields and retain no prose, and the
    audit-log reader must recover the same roll facts the tool envelope carried, so the
    comparison the live flow runs is between the model's printed claim and the record's
    own numbers -- never between the guard under test and itself.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import asyncio, sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import _MechanicalTruthAdapter as Truth,"
        " _announcement_text;"
        "from narrator.engine import verdict_mismatches;"
        "a = Truth();"

        "asyncio.run(a.post('docks', 'Ossa rolls WIS: 3 vs 9, failure.\\n'"
        " 'Rill rolls WIS: 11 vs 14, success.\\n'"
        " 'A cart rattles through the courtyard.'));"
        "assert a.output_categories == ['narration'], a.output_categories;"
        "assert len(a.announced) == 2, a.announced;"
        "assert a.announced[0] == {'character': 'Ossa', 'attribute': 'WIS', 'total': 3,"
        " 'target': 9, 'outcome': 'failure'}, a.announced[0];"
        # No prose survives: the distinctive words of the reply are gone.
        "held = [v for v in vars(a).values() if isinstance(v, str) and 'cart' in v.casefold()];"
        "assert held == [], held;"
        "nested = [str(x) for v in vars(a).values() if isinstance(v, list) for x in v"
        " if 'cart' in str(x).casefold() or 'courtyard' in str(x).casefold()];"
        "assert nested == [], nested;"
        # The rebuild is lossless for every field the comparator reads.
        "rebuilt = _announcement_text(a.announced);"
        "assert 'Ossa rolls WIS: 3 vs 9, failure.' in rebuilt, rebuilt;"
        "assert 'Rill rolls WIS: 11 vs 14, success.' in rebuilt, rebuilt;"
        # And a critical round-trips through the underscore the wire value carries.
        "crit = Truth();"
        "asyncio.run(crit.post('docks', 'Rill rolls DEX: 1 vs 12, critical success.'));"
        "assert crit.announced[0]['outcome'] == 'critical_success', crit.announced;"
        "assert 'critical success' in _announcement_text(crit.announced);"
        # What the live flow computes: the record's numbers against the printed claim.
        "recorded = ({'tool': 'group_test', 'character': 'Ossa', 'attribute': 'WIS',"
        " 'total': 3, 'target': 9, 'outcome': 'success'},"
        " {'tool': 'group_test', 'character': 'Rill', 'attribute': 'WIS', 'total': 11,"
        " 'target': 10, 'outcome': 'failure'});"
        "kinds = [e['kind'] for e in verdict_mismatches(rebuilt, recorded)];"
        "assert kinds == ['verdict', 'target'], kinds;"
        "print('M11_ADAPTER_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "M11_ADAPTER_OK" in result.stdout


def test_the_m13_mappings_read_the_typed_counters_and_the_completed_mechanic_set():
    """Milestone M13's two live mappings, driven offline like the others.

    Both scenarios hold the sanctioned-attack scenario's campaign setup and capability
    shape and change only the declared text, so their checks must key on the service's own
    typed counters rather than on anything parsed out of model prose. The
    ooc-question-in-fight scenario additionally has to distinguish "answered" from
    "resolved", and a completed combat mechanic is the observable form of resolving: the
    fight is open and the enemy is named, so each of those tools was reachable had the
    model treated the quoted verb as a declaration.

    Reading ``completed`` rather than ``attempted`` is the load-bearing half. Repointing
    that check at ``attempted`` would leave this suite green while the live scenario
    stopped detecting a resolved attack, which is the regression class the probe exists
    to catch.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=1, errors=False, delivered=1, withheld=0,"
        " output_categories=['narration'], decision_kinds=[],"
        " attempted=set(), completed=set());"
        # weapon-named-sanction: the fight's sanction must survive the weapon name.
        "good = s('weapon-named-sanction', **base);"
        "assert good['sanctioned_with_weapon_named'] is True, good;"
        "assert good['no_confirmation_presented'] is True, good;"
        # A withheld turn is exactly the pre-M13 shape and must not read as a pass.
        "withheld = s('weapon-named-sanction', **dict(base, delivered=0, withheld=1));"
        "assert withheld['sanctioned_with_weapon_named'] is False, withheld;"
        # A presented confirmation is the other pre-M13 shape.
        "confirmed = s('weapon-named-sanction', **dict(base, decision_kinds=['confirmation']));"
        "assert confirmed['no_confirmation_presented'] is False, confirmed;"
        # ooc-question-in-fight: answered, and nothing physically resolved.
        "answered = s('ooc-question-in-fight', **base);"
        "assert answered['question_answered_not_withheld'] is True, answered;"
        "assert answered['no_physical_event_resolved'] is True, answered;"
        # Each combat mechanic, completed, is a resolved physical event. Written out
        # rather than looped because this snippet runs as one physical line.
        "attacked = s('ooc-question-in-fight', **dict(base, completed={'combat_attack'}));"
        "assert attacked['no_physical_event_resolved'] is False, attacked;"
        "opened = s('ooc-question-in-fight', **dict(base, completed={'combat_begin_turn'}));"
        "assert opened['no_physical_event_resolved'] is False, opened;"
        "defended = s('ooc-question-in-fight', **dict(base, completed={'combat_defend'}));"
        "assert defended['no_physical_event_resolved'] is False, defended;"
        # An attempted-but-not-completed mechanic must not itself fail the check, which is
        # what pins this to the completed set rather than the attempted one.
        "attempted_only = s('ooc-question-in-fight', **dict(base, attempted={'combat_attack'}));"
        "assert attempted_only['no_physical_event_resolved'] is True, attempted_only;"
        # A withheld question is the pre-M13 shape: the hazard branch claimed it.
        "blocked = s('ooc-question-in-fight', **dict(base, delivered=0, withheld=1));"
        "assert blocked['question_answered_not_withheld'] is False, blocked;"
        # Both scenarios are one-turn runs; a two-turn run must not read as complete.
        "long_weapon = s('weapon-named-sanction', **dict(base, turns=2));"
        "assert long_weapon['service_completed'] is False, long_weapon;"
        "long_ooc = s('ooc-question-in-fight', **dict(base, turns=2));"
        "assert long_ooc['service_completed'] is False, long_ooc;"
        "print('M13_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "M13_MAPPING_OK" in result.stdout


def test_the_m14_mappings_read_no_confirmation_and_the_new_engagement_checks():
    """Milestone M14's live mappings, driven offline like every other scenario's.
    """
    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=1, errors=False, delivered=1, withheld=0,"
        " output_categories=['narration'], decision_kinds=[],"
        " attempted=set(), completed={'combat_attack'});"
        # attack: no confirmation, mechanic completes -- the post-M14 pass shape,
        # with no combat_begin_turn call needed at all.
        "clean_attack = s('attack', **base);"
        "assert clean_attack['no_confirmation_presented'] is True, clean_attack;"
        "assert 'turn_opened' not in clean_attack, clean_attack;"
        "assert clean_attack['attack_completed'] is True, clean_attack;"
        # A presented confirmation is exactly the pre-M14 shape and must not read as a
        # pass anymore.
        "stale_attack = s('attack', **dict(base, decision_kinds=['confirmation']));"
        "assert stale_attack['no_confirmation_presented'] is False, stale_attack;"
        # unlisted-violence reuses the identical mapping; only its declared text differs
        # in ``_runtime_flow``.
        "clean_unlisted = s('unlisted-violence', **base);"
        "assert clean_unlisted == clean_attack, (clean_unlisted, clean_attack);"
        # new-engagement-confirms: no open fight, so the confirmation must still be
        # presented, and the resolving continuation must still open the engagement.
        "engagement_base = dict(turns=1, errors=False, delivered=1, withheld=0,"
        " output_categories=['narration'], decision_kinds=['confirmation'],"
        " attempted=set(),"
        " completed={'combat_start', 'combat_attack'});"
        "confirmed = s('new-engagement-confirms', **engagement_base);"
        "assert confirmed['confirmation_presented'] is True, confirmed;"
        "assert confirmed['engagement_opened'] is True, confirmed;"
        "assert confirmed['attack_rolled_or_deferred_to_the_actors_turn'] is True, confirmed;"
        # The rules-mandated deferral: initiative gave the opposition the first turn,
        # so no combat_attack rolled this turn -- and that reads as a pass, because
        # rolling it would have been the out-of-turn defect the engine refuses.
        "deferred = s('new-engagement-confirms', **dict(engagement_base,"
        " completed={'combat_start'}), enemy_holds_first_turn=True);"
        "assert deferred['attack_rolled_or_deferred_to_the_actors_turn'] is True, deferred;"
        # An unrolled, undeferred attack must not read as a pass.
        "dropped = s('new-engagement-confirms', **dict(engagement_base,"
        " completed={'combat_start'}));"
        "assert dropped['attack_rolled_or_deferred_to_the_actors_turn'] is False, dropped;"
        # A withheld confirmation (never answered) must not read as a pass: nothing in
        # the engagement opened yet.
        "unconfirmed = s('new-engagement-confirms', **dict(engagement_base,"
        " decision_kinds=[], completed=set()));"
        "assert unconfirmed['confirmation_presented'] is False, unconfirmed;"
        "assert unconfirmed['engagement_opened'] is False, unconfirmed;"
        "print('M14_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "M14_MAPPING_OK" in result.stdout


def test_the_combat_pace_mappings_read_decisions_state_and_edges():
    """The maneuver-direct-ruling and outlandish-priced-not-granted mappings.

    outlandish-priced-not-granted passes only when nothing was granted: the enemy's
    record untouched and no Advantage-edged roll in the audit log -- a refusal in
    fiction and a priced ordinary test are both passes, which is why no mechanic
    completion is required at all.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    snippet = (
        "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src');"
        "from probe_combat_decisions import scenario_checks as s;"
        "base = dict(turns=1, errors=False, delivered=1, withheld=0,"
        " output_categories=['narration'], attempted=set());"
        # The pass shape: no decisions, the maneuver's own attribute_test, a spent
        # action.
        "clean = s('maneuver-direct-ruling', **base, decision_kinds=[],"
        " completed={'attribute_test'}, maneuver_action_spent=True);"
        "assert all(clean.values()), clean;"
        # The measured defect shape -- the declaration resolved as a strike -- is a
        # fail on both fidelity checks, where it used to be a pass.
        "as_attack = s('maneuver-direct-ruling', **base, decision_kinds=[],"
        " completed={'combat_attack'}, maneuver_action_spent=True);"
        "assert as_attack['maneuver_resolved_by_an_attribute_test'] is False, as_attack;"
        "assert as_attack['maneuver_not_resolved_as_an_attack'] is False, as_attack;"
        # Rolling both resolves one declaration twice and is not a pass either.
        "both = s('maneuver-direct-ruling', **base, decision_kinds=[],"
        " completed={'attribute_test', 'combat_attack'}, maneuver_action_spent=True);"
        "assert both['maneuver_resolved_by_an_attribute_test'] is True, both;"
        "assert both['maneuver_not_resolved_as_an_attack'] is False, both;"
        # A presented questionnaire is exactly the pre-bypass shape and must fail.
        "asked = s('maneuver-direct-ruling', **base, decision_kinds=['clarification'],"
        " completed={'attribute_test'}, maneuver_action_spent=True);"
        "assert asked['no_decisions_presented'] is False, asked;"
        # A ruling that never rolled, or rolled without spending the action, fails.
        "unrolled = s('maneuver-direct-ruling', **base, decision_kinds=[],"
        " completed=set(), maneuver_action_spent=False);"
        "assert unrolled['maneuver_resolved_by_an_attribute_test'] is False, unrolled;"
        "assert unrolled['maneuver_action_spent'] is False, unrolled;"
        # Outlandish: a refusal with no mechanic at all is a clean pass.
        "refused = s('outlandish-priced-not-granted', **base, decision_kinds=[],"
        " completed=set(), enemy_unharmed=True, no_edge_granted=True);"
        "assert all(refused.values()), refused;"
        # A priced test is also a pass; a granted edge or a touched enemy is not.
        "priced = s('outlandish-priced-not-granted', **base, decision_kinds=[],"
        " completed={'attribute_test'}, enemy_unharmed=True, no_edge_granted=True);"
        "assert all(priced.values()), priced;"
        "granted = s('outlandish-priced-not-granted', **base, decision_kinds=[],"
        " completed=set(), enemy_unharmed=False, no_edge_granted=False);"
        "assert granted['enemy_unharmed'] is False, granted;"
        "assert granted['no_edge_granted'] is False, granted;"
        "print('PACE_MAPPING_OK')"
    )
    result = subprocess.run(
        [INTERPRETER, "-c", snippet], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "PACE_MAPPING_OK" in result.stdout
