"""The locale catalog: the one mechanism for player-facing text in the narrator.

These run in the main suite because the strings they cover are the ones a player reads,
and a defect here reaches a table directly rather than through the model.

The shape of the guarantee: the English catalog is the schema. A translation that
loses a placeholder, renames a recovery control, drops a key, or collapses two display
forms into one does not load. That is deliberately stricter than falling back, because
a missing translation degrades to readable English while a malformed one has no safe
reading -- a die result with no target, or a notice advertising a control the channel
will then refuse.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from narrator.config import NarratorConfig
from narrator.locale import SOURCE_LANGUAGE, Catalog, LocaleError, load

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCALE_ROOT = REPO_ROOT / "locale"

#: Every language shipped. A new directory joins this by existing.
SHIPPED = sorted(p.name for p in LOCALE_ROOT.iterdir() if (p / "narrator.yaml").is_file())


def _mutated(tmp_path: Path, change) -> Path:
    """A locale tree whose French catalog has been altered by ``change``."""
    root = tmp_path / "locale"
    root.mkdir()
    shutil.copytree(LOCALE_ROOT / SOURCE_LANGUAGE, root / SOURCE_LANGUAGE)
    shutil.copytree(LOCALE_ROOT / "fr", root / "fr")
    path = root / "fr" / "narrator.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return root


def test_every_shipped_language_loads():
    assert SOURCE_LANGUAGE in SHIPPED
    for language in SHIPPED:
        assert isinstance(load(language, LOCALE_ROOT), Catalog), language


def test_every_language_renders_and_parses_a_roll_back_to_canonical_values():
    """The renderer and the parser are built from one string, so they cannot drift.

    Everything downstream of ``parse_rolls`` compares canonical values --
    ``verdict_mismatches`` matches the tool's own ``STR`` and ``critical_success`` -- so
    a language is only wired correctly if its display forms invert exactly.
    """
    for language in SHIPPED:
        catalog = load(language, LOCALE_ROOT)
        for attribute in ("STR", "WIS"):
            for outcome in ("success", "critical_failure"):
                line = catalog.render_roll(
                    name="Rill", attribute=attribute, total=4, target=13, outcome=outcome
                )
                parsed = catalog.parse_rolls(line)
                assert parsed == [
                    {
                        "name": "Rill",
                        "attribute": attribute,
                        "total": 4,
                        "target": 13,
                        "outcome": outcome,
                    }
                ], f"{language} {attribute}/{outcome}: {line}"


def test_a_critical_outcome_is_not_swallowed_by_its_own_substring():
    """"critical success" contains "success"; the alternation must prefer the longer.

    Without longest-first ordering the parser reads a critical as an ordinary result,
    which is a wrong mechanical fact rather than a formatting problem.
    """
    for language in SHIPPED:
        catalog = load(language, LOCALE_ROOT)
        line = catalog.render_roll(
            name="Rill", attribute="STR", total=1, target=13, outcome="critical_success"
        )
        assert catalog.parse_rolls(line)[0]["outcome"] == "critical_success", language


def test_an_untranslated_language_falls_back_to_english():
    """A table with no catalog plays in English rather than failing to start."""
    assert load("xx-XX", LOCALE_ROOT).language == load(SOURCE_LANGUAGE, LOCALE_ROOT).language


def test_a_regional_tag_resolves_to_its_base_language():
    assert load("fr-CA", LOCALE_ROOT).language == "fr-FR"


# ---------------------------------------------------------------------------
# What must NOT load. Each of these reached a player as a defect in some form.
# ---------------------------------------------------------------------------


def test_a_translation_that_drops_a_placeholder_does_not_load(tmp_path: Path):
    """The failure gettext could not catch, and the reason this validates at load.

    ``decision_refused`` names the refused mechanic through ``$refusal``. A translation
    without it renders a sentence that says the rules engine refused nothing.
    """
    root = _mutated(
        tmp_path,
        lambda d: d["notices"].__setitem__("decision_refused", "Rien n'a été résolu."),
    )
    with pytest.raises(LocaleError, match=r"must use exactly \['refusal'\]"):
        load("fr", root)


def test_a_translation_that_drops_the_roll_target_does_not_load(tmp_path: Path):
    """A die result with no target is meaningless in a roll-under game."""
    root = _mutated(
        tmp_path,
        lambda d: d["roll"].__setitem__(
            "announcement", "$name lance $attribute : $total, $outcome."
        ),
    )
    with pytest.raises(LocaleError, match="roll.announcement"):
        load("fr", root)


def test_a_translated_recovery_control_does_not_load(tmp_path: Path):
    """``/retry`` is syntax. ``narrator.delivery`` arms a control by finding its name in
    the posted text, so translating it leaves a notice advertising a command the channel
    then answers "Unknown command" to."""
    root = _mutated(
        tmp_path,
        lambda d: d["notices"].__setitem__(
            "decision_fault", d["notices"]["decision_fault"].replace("/retry", "/reessayer")
        ),
    )
    with pytest.raises(LocaleError, match="recovery controls"):
        load("fr", root)


def test_a_missing_key_does_not_load(tmp_path: Path):
    root = _mutated(tmp_path, lambda d: d["notices"].pop("fault"))
    with pytest.raises(LocaleError, match=r"missing \['notices\.fault'\]"):
        load("fr", root)


def test_two_attributes_sharing_a_display_form_do_not_load(tmp_path: Path):
    """Ambiguous display forms make the parser unable to invert them."""
    root = _mutated(tmp_path, lambda d: d["attributes"].__setitem__("DEX", "FOR"))
    with pytest.raises(LocaleError, match="must be distinct"):
        load("fr", root)


def test_a_missing_canonical_outcome_does_not_load(tmp_path: Path):
    """These keys are looked up by the rules engine's own value; a miss is a crash."""
    root = _mutated(tmp_path, lambda d: d["outcomes"].pop("critical_failure"))
    with pytest.raises(LocaleError, match="must key on exactly"):
        load("fr", root)


# ---------------------------------------------------------------------------
# The wording lives in exactly one place.
# ---------------------------------------------------------------------------


def test_the_config_reads_notices_from_the_catalog_and_holds_no_copy():
    """Drift guard. Before this change the English wording was a dataclass default in
    ``config.py`` *and* would have been in the catalog; a translator fixing one would
    not have fixed the other. The notices are properties now, and this asserts the
    module carries no second copy of the text.
    """
    source = (REPO_ROOT / "src" / "narrator" / "config.py").read_text(encoding="utf-8")
    catalog = load(SOURCE_LANGUAGE, LOCALE_ROOT)
    for key, text in catalog.notices.items():
        # A distinctive fragment, long enough that an accidental match is implausible.
        fragment = text.split(".")[0][:40]
        assert fragment not in source, (
            f"config.py still contains the wording for '{key}'. Player-facing text "
            "belongs in locale/, not in code."
        )


def test_the_config_switches_language():
    for language, expected in (("en-US", "en-US"), ("fr-FR", "fr-FR"), ("ja-JP", "ja-JP")):
        config = NarratorConfig(campaign_root=REPO_ROOT, language=language)
        assert config.catalog.language == expected
        assert config.fault_notice == load(language.split("-")[0], LOCALE_ROOT).notice("fault")


def test_the_config_switches_language_mid_session():
    """``/language`` acts through ``set_language``: the frozen config carries the
    override as session state (``LanguageState``), so every notice property and
    ``config.catalog`` read follows at call time with no rewiring."""
    config = NarratorConfig(campaign_root=REPO_ROOT)
    assert config.active_language == "en-US"
    english = config.fault_notice

    assert config.set_language("fr") == "fr"
    assert config.active_language == "fr"
    assert config.fault_notice == load("fr", LOCALE_ROOT).notice("fault")
    assert config.fault_notice != english

    with pytest.raises(ValueError):
        config.set_language("xx")
    assert config.active_language == "fr", "a refused switch must leave the language standing"

    # A regional tag applies by its base language, exactly like the launch path.
    assert config.set_language("ja-JP") == "ja-JP"
    assert config.fault_notice == load("ja", LOCALE_ROOT).notice("fault")


def test_required_locales_resolve_to_their_own_language():
    """The coverage contract: every required market resolves to a catalog in its own
    language, in both domains, never silently to the English fallback (except English
    itself). English US and UK, French, German, Korean, Simplified Chinese (Mandarin),
    Spanish for Spain and Mexico, Portuguese for Portugal and Brazil, Finnish, Swedish,
    Norwegian, Lithuanian, Dutch and Russian; Japanese predates the list and stays."""
    required = {
        "en-US": "en-US", "en-GB": "en-GB",
        "fr": "fr-FR", "fr-FR": "fr-FR",
        "de": "de-DE", "de-DE": "de-DE",
        "ko": "ko-KR", "ko-KR": "ko-KR",
        "zh": "zh-CN", "zh-CN": "zh-CN", "zh-Hans": "zh-CN",
        "es": "es-ES", "es-ES": "es-ES", "es-MX": "es-MX",
        "pt": "pt-PT", "pt-PT": "pt-PT", "pt-BR": "pt-BR",
        "fi": "fi-FI", "fi-FI": "fi-FI",
        "sv": "sv-SE", "sv-SE": "sv-SE",
        "no": "nb-NO", "no-NO": "nb-NO",
        "lt": "lt-LT", "lt-LT": "lt-LT",
        "nl": "nl-NL", "nl-NL": "nl-NL",
        "ru": "ru-RU", "ru-RU": "ru-RU",
        "ja": "ja-JP", "ja-JP": "ja-JP",
    }
    for tag, expected in required.items():
        for domain in ("narrator", "terminal"):
            assert load(tag, LOCALE_ROOT, domain=domain).language == expected, (tag, domain)


def test_a_switch_is_session_state_not_configuration():
    first = NarratorConfig(campaign_root=REPO_ROOT)
    second = NarratorConfig(campaign_root=REPO_ROOT)
    first.set_language("de")
    assert second.active_language == "en-US", "a switch must not leak between configs"
    assert first == second, "session state must stay out of config equality"


def test_every_notice_the_config_exposes_exists_in_every_language():
    """A property reading a key no translation carries would raise mid-turn."""
    english = load(SOURCE_LANGUAGE, LOCALE_ROOT)
    for language in SHIPPED:
        catalog = load(language, LOCALE_ROOT)
        assert set(catalog.notices) == set(english.notices), language
        assert set(catalog.confirmations) == set(english.confirmations), language
        assert set(catalog.options) == set(english.options), language


def test_catalog_comments_carry_the_roll_under_constraint():
    """Every translator must be told not to unlabel the numbers.

    The constraint is not inferable from the string: it was learned from a live table
    that could not tell the die from the target. YAML comments are why this catalog
    format was chosen over JSON, so this asserts the comment is actually there.
    """
    for language in SHIPPED:
        text = (LOCALE_ROOT / language / "narrator.yaml").read_text(encoding="utf-8")
        roll_section = text[text.index("roll:"):text.index("attributes:")]
        assert re.search(r"^\s*#", roll_section, re.M), (
            f"{language}: the roll block carries no translator comment"
        )


# ---------------------------------------------------------------------------
# The terminal domain.
# ---------------------------------------------------------------------------

TERMINAL_SHIPPED = sorted(
    p.name for p in LOCALE_ROOT.iterdir() if (p / "terminal.yaml").is_file()
)


def test_every_language_ships_both_domains():
    """A language with a narrator catalog and no terminal catalog would play in its own
    language and then render its status line and character sheet in English."""
    assert TERMINAL_SHIPPED == SHIPPED


def test_available_languages_lists_every_shipped_catalog():
    """The vocabulary ``/language`` offers is the filesystem, like ``SHIPPED`` is."""
    from narrator.locale import available_languages

    assert available_languages(LOCALE_ROOT) == tuple(SHIPPED)
    assert available_languages(LOCALE_ROOT, domain="terminal") == tuple(TERMINAL_SHIPPED)
    assert available_languages(LOCALE_ROOT / "absent") == ()


def test_the_terminal_catalog_covers_every_path_the_adapter_reads():
    """Drift guard between the adapter and its catalog.

    ``Catalog.text`` raises ``KeyError`` on a missing path, deliberately, so a typo
    surfaces here rather than rendering an empty string mid-session. This walks the
    paths the adapter actually asks for.
    """
    used = (
        "status.no_character", "status.hp", "status.doom", "status.combat",
        "status.round", "status.turn", "status.day",
        "sheet.backgrounds", "sheet.stories", "sheet.hp", "sheet.doom", "sheet.armour",
        "sheet.weapons", "sheet.equipment", "sheet.coins", "sheet.languages",
        "sheet.resources", "sheet.doses", "sheet.gifts", "sheet.spells", "sheet.powers",
        "sheet.runic_weapon", "sheet.conditions", "sheet.scars", "sheet.notes",
        "sheet.none", "sheet.shield", "sheet.level", "sheet.unlinked",
        "commands.help", "commands.recovery_help", "commands.revise_after",
        "commands.revise_now", "commands.no_decision_open", "commands.unknown",
        "commands.slash_required",
        "commands.language_show", "commands.language_set",
        "commands.language_unknown", "commands.language_unavailable",
        "decision.prompt", "decision.enter_number", "decision.help",
        "decision.choose_listed", "decision.describe_approach",
        "decision.custom_grammar", "decision.custom_plain", "decision.custom_example",
        "errors.invalid", "errors.conflict", "errors.stale", "errors.cancelled",
        "errors.rejected", "thinking",
    )
    placeholders = {
        "status.hp": {"hp": 7, "hp_max": 9}, "status.doom": {"die": "d6"},
        "status.round": {"round": 2}, "status.turn": {"name": "rill"},
        "status.day": {"day": 3}, "sheet.level": {"level": 1},
        "commands.unknown": {"command": "/foo"}, "decision.prompt": {"name": "Rill"},
        "decision.custom_grammar": {"label": "x", "marker": "2.", "example": "y"},
        "commands.language_show": {"language": "fr", "languages": "en, fr"},
        "commands.language_set": {"language": "fr"},
        "commands.language_unknown": {"language": "xx", "languages": "en, fr"},
    }
    for language in TERMINAL_SHIPPED:
        catalog = load(language, LOCALE_ROOT, domain="terminal")
        for path in used:
            assert catalog.text(path, **placeholders.get(path, {})), f"{language}:{path}"


def test_the_terminal_keeps_its_slash_commands_untranslated():
    """``/help`` and its siblings are bytes the terminal parses, not words.

    The loader enforces this per string against the English source; this asserts the
    set actually present, so a language that drops a whole command-bearing string
    rather than translating it is caught too.
    """
    from narrator.channels.base import LOCAL_COMMANDS, RECOVERY_CONTROLS

    for language in TERMINAL_SHIPPED:
        catalog = load(language, LOCALE_ROOT, domain="terminal")
        blob = catalog.text("commands.help") + catalog.text("commands.recovery_help")
        for name in RECOVERY_CONTROLS | LOCAL_COMMANDS - {"revise"}:
            assert f"/{name}" in blob or name == "character", f"{language}: /{name}"


def test_the_terminal_wraps_a_space_free_cjk_line_by_display_width():
    """Japanese has no spaces, so a greedy word wrapper would emit one endless row.

    It does not: ``_word_wrap`` measures with ``prompt_toolkit``'s ``get_cwidth`` and
    breaks inside a token that cannot fit, so a CJK run wraps at the column limit and
    each row stays within the terminal's real width -- double-width characters counted
    as two columns, not one.
    """
    from prompt_toolkit.utils import get_cwidth

    from narrator.channels.terminal_ui import _word_wrap

    japanese = "霧が波止場を包み込み、今夜は鐘がまったく鳴らないまま夜が更けていくのだった"
    rows = _word_wrap([("", japanese)], 30)
    assert len(rows) > 1, "a space-free CJK line must still wrap"
    for row in rows:
        columns = sum(get_cwidth(char) for _, char in row)
        assert columns <= 30, f"row overflows the terminal width: {columns}"
    assert "".join(text for row in rows for _, text in row) == japanese
