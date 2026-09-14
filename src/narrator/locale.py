"""The one mechanism for player-facing text in the narrator.

Every string a player reads that this engine authors -- the notices posted instead of
narration, the table-safety confirmations, the decision option labels, the roll
announcement -- lives in ``locale/<language>/narrator.yaml`` and is read through this
module. Nothing else localizes, and nothing here is read by the model.

Scope, which is a design boundary rather than a convenience:

- **In.** Text the *engine* writes and a *player* reads. That text must be in the
  table's language, because no model stands between the engine and the player to
  translate it.
- **Out.** Text the *model* reads: tool descriptions, skills, prompts, and the
  ``narration_facts`` and error messages a tool returns. Those are instructions and
  source data, not interface copy. The narrator was measured translating English tool
  results into fluent French narration, so translating the instructions themselves buys
  nothing and risks changing behavior -- the whole mechanical contract is written in
  them.
- **Out.** Authored content under ``world/`` and the prose fields of ``rules/*.json``.
  Those have an authoring lifecycle, not a translation one, and ``Manifest.world_root``
  already points a campaign at a different tree.

Why one catalog per domain and not one per tool or per skill. The Model Context
Protocol tools are twenty-four functions in a single module that ships, versions and
fails as one unit; splitting their strings twenty-four ways would create files that
always change together and never separately. Skills are model-facing prose with a
behavioral contract in them, so they take the ``world_root`` treatment -- point a
campaign at a translated directory -- rather than a string catalog. What does divide
cleanly is deployment ownership: the narrator engine owns one catalog, and each
channel adapter owns its own, so shipping a Discord adapter adds
``locale/<lang>/discord.yaml`` instead of growing a shared file.

The English catalog is the schema. A translation is validated against it at load:
same keys, the same ``$placeholders`` in every string, and the same recovery controls.
A catalog that fails any of those does not load, because the alternative is a
malformed mechanical statement reaching a table -- and unlike a missing translation,
which degrades to readable English, a malformed one has no safe reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from string import Template

import yaml

#: The locale shipped as the source of every other, and the fallback for all of them.
SOURCE_LANGUAGE = "en"

#: Sections whose values are flat ``key -> text`` maps of player-facing strings.
_TEXT_SECTIONS = ("notices", "confirmations", "options")

#: Sections whose keys are the rules engine's own canonical values. Only the values
#: translate; a missing or invented key is a load failure rather than a fallback,
#: because these are looked up by canonical value and a miss is a crash at render time.
_VOCABULARIES = {
    "attributes": ("STR", "DEX", "CON", "INT", "WIS", "CHA"),
    "outcomes": ("success", "failure", "critical_success", "critical_failure"),
}

_PLACEHOLDER = re.compile(r"\$(\w+)")

#: A recovery control as a notice spells it. ``narrator.delivery`` arms a control by
#: finding this exact form in the posted text, so a translation that drops or renames
#: one silently disarms it -- the notice would tell a player to type something the
#: channel then refuses. Validated per string, not per catalog.
_CONTROL = re.compile(r"/[a-z]+")


class LocaleError(RuntimeError):
    """A catalog that cannot be trusted to render. Raised at load, never at render."""


@dataclass(frozen=True)
class Catalog:
    """One language's player-facing text, validated against the English source."""

    language: str
    notices: dict[str, str]
    confirmations: dict[str, str]
    options: dict[str, str]
    announcement: str
    attributes: dict[str, str]
    outcomes: dict[str, str]
    #: The whole tree, for domains whose shape this class does not model as fields.
    data: dict = field(default_factory=dict)

    # -- reading ----------------------------------------------------------------

    def notice(self, key: str) -> str:
        return self.notices[key]

    def confirmation(self, category: str) -> str:
        """The confirmation for a hazard category, or the uncategorised wording.

        An unknown category takes the uncategorised sentence rather than raising: it
        names all three consequences, so it is true whatever the category turned out to
        be, and a confirmation that fails to render is a hazard that reaches narration.
        """
        return self.confirmations.get(category) or self.confirmations["uncategorized"]

    def option(self, key: str) -> str:
        return self.options[key]

    def text(self, path: str, **values: object) -> str:
        """One string by dotted path, substituted if it carries placeholders.

        The accessor for domains other than ``narrator``. ``KeyError`` on a missing
        path is deliberate: load-time validation guarantees every path the English
        source defines exists in every language, so a miss here is a caller typo, not a
        translation gap, and should surface in a test rather than silently render "".
        """
        node: object = self.data
        for part in path.split("."):
            node = node[part]  # type: ignore[index]
        return self.render(str(node), **values) if values else str(node)

    def render(self, template: str, **values: object) -> str:
        """Substitute into one catalog string.

        ``Template.substitute`` rather than ``safe_substitute``: a missing value is a
        programming error and should raise here, not emit a half-written sentence to a
        table. Load-time validation is what makes this safe -- every placeholder the
        template carries is one the English source carries, so the caller supplying the
        English set always supplies enough.
        """
        return Template(template).substitute(**values)

    # -- the roll announcement, rendered and parsed from one string ---------------

    def render_roll(self, *, name: str, attribute: str, total: int, target: int, outcome: str) -> str:
        """One roll announcement in this language, from canonical inputs."""
        return self.render(
            self.announcement,
            name=name,
            attribute=self.attributes[attribute],
            total=total,
            target=target,
            outcome=self.outcomes[outcome],
        )

    def roll_pattern(self) -> re.Pattern[str]:
        """The parser for ``render_roll``'s output, built from the same string.

        Renderer and parser cannot drift, because a language is one file and both are
        derived from its ``roll.announcement``. Longest-first alternation on the outcome
        vocabulary matters: "critical success" must win against "success", which is a
        substring of it.
        """
        attrs = "|".join(re.escape(v) for v in self.attributes.values())
        outs = "|".join(
            re.escape(v) for v in sorted(self.outcomes.values(), key=len, reverse=True)
        )
        groups = {
            "name": r"(?P<name>\S[^\n]*?)",
            "attribute": f"(?P<attribute>{attrs})",
            "total": r"(?P<total>\d{1,3})",
            "target": r"(?P<target>\d{1,3})",
            "outcome": f"(?P<outcome>{outs})",
        }
        pattern = re.escape(self.announcement)
        for slot, group in groups.items():
            pattern = pattern.replace(re.escape(f"${slot}"), group)
        return re.compile(pattern)

    def parse_rolls(self, text: str) -> list[dict]:
        """Every announcement in ``text``, back in canonical form.

        The display vocabularies are inverted here, so a French ``FOR`` and ``succès
        critique`` reach the caller as ``STR`` and ``critical_success``. Everything
        downstream compares canonical values and never learns which language the table
        speaks.
        """
        by_attribute = {v: k for k, v in self.attributes.items()}
        by_outcome = {v: k for k, v in self.outcomes.items()}
        return [
            {
                "name": match.group("name"),
                "attribute": by_attribute[match.group("attribute")],
                "total": int(match.group("total")),
                "target": int(match.group("target")),
                "outcome": by_outcome[match.group("outcome")],
            }
            for match in self.roll_pattern().finditer(text)
        ]


def _walk(node, prefix: str = "") -> dict[str, str]:
    """Every leaf string in a catalog tree, keyed by its dotted path.

    Validation is structural rather than per-domain: whatever shape a domain's English
    file has, a translation must carry the same leaves with the same placeholders. That
    is what lets a new domain -- a Discord adapter's strings, say -- be added as a file
    without also adding a validator for it.
    """
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            found.update(_walk(value, f"{prefix}{key}." if not prefix else f"{prefix}{key}."))
    elif isinstance(node, str):
        found[prefix.rstrip(".")] = node
    return found


def _read(root: Path, language: str, domain: str) -> dict:
    path = root / language / f"{domain}.yaml"
    if not path.is_file():
        raise LocaleError(f"no catalog at {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise LocaleError(f"{path} is not readable YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise LocaleError(f"{path} must be a mapping")
    return loaded


def _validate(language: str, raw: dict, source: dict | None, domain: str) -> None:
    """Check one catalog against the English source. ``None`` source means this is it.

    Two layers. The structural layer applies to every domain: same leaves, same
    ``$placeholders``, same recovery controls. The narrator domain adds its own
    requirement that the two vocabularies key on the rules engine's canonical values,
    because those are looked up by canonical value and a miss is a crash at render time
    rather than a missing word.
    """
    if domain != "narrator":
        if source is not None:
            _validate_leaves(language, raw, source)
        return
    for section in (*_TEXT_SECTIONS, *_VOCABULARIES):
        if not isinstance(raw.get(section), dict):
            raise LocaleError(f"{language}: section '{section}' is missing or not a mapping")
    if not isinstance(raw.get("roll", {}).get("announcement"), str):
        raise LocaleError(f"{language}: roll.announcement is missing")

    for section, canonical in _VOCABULARIES.items():
        if set(raw[section]) != set(canonical):
            missing = sorted(set(canonical) - set(raw[section]))
            extra = sorted(set(raw[section]) - set(canonical))
            raise LocaleError(
                f"{language}: '{section}' must key on exactly {list(canonical)}; "
                f"missing {missing}, unexpected {extra}"
            )
        values = list(raw[section].values())
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise LocaleError(
                f"{language}: '{section}' display forms must be distinct so the parser "
                f"is unambiguous; repeated {repeated}"
            )

    if source is None:
        return
    _validate_leaves(language, raw, source)

    pairs: list[tuple[str, str, str]] = [("roll.announcement",
                                          source["roll"]["announcement"],
                                          raw["roll"]["announcement"])]
    for section in _TEXT_SECTIONS:
        missing = sorted(set(source[section]) - set(raw[section]))
        if missing:
            raise LocaleError(f"{language}: '{section}' is missing {missing}")
        extra = sorted(set(raw[section]) - set(source[section]))
        if extra:
            raise LocaleError(
                f"{language}: '{section}' has {extra}, which the English source does not"
            )
        pairs += [(f"{section}.{k}", source[section][k], raw[section][k]) for k in source[section]]

    for key, english, translated in pairs:
        if not isinstance(translated, str) or not translated.strip():
            raise LocaleError(f"{language}: '{key}' is empty")
        want = set(_PLACEHOLDER.findall(english))
        got = set(_PLACEHOLDER.findall(translated))
        if want != got:
            raise LocaleError(
                f"{language}: '{key}' must use exactly {sorted(want)}; "
                f"missing {sorted(want - got)}, unexpected {sorted(got - want)}"
            )
        want_controls = set(_CONTROL.findall(english))
        got_controls = set(_CONTROL.findall(translated))
        if want_controls != got_controls:
            raise LocaleError(
                f"{language}: '{key}' must name the same recovery controls as the "
                f"English source {sorted(want_controls)}; found {sorted(got_controls)}. "
                "A control is syntax, not vocabulary -- translating it disarms it."
            )


def _validate_leaves(language: str, raw: dict, source: dict) -> None:
    """Same leaves, same placeholders, same recovery controls as the English source."""
    english, translated = _walk(source), _walk(raw)
    missing = sorted(set(english) - set(translated))
    if missing:
        raise LocaleError(f"{language}: missing {missing}")
    extra = sorted(set(translated) - set(english))
    if extra:
        raise LocaleError(f"{language}: has {extra}, which the English source does not")
    for key, text in english.items():
        other = translated[key]
        if not other.strip():
            raise LocaleError(f"{language}: '{key}' is empty")
        want, got = set(_PLACEHOLDER.findall(text)), set(_PLACEHOLDER.findall(other))
        if want != got:
            raise LocaleError(
                f"{language}: '{key}' must use exactly {sorted(want)}; "
                f"missing {sorted(want - got)}, unexpected {sorted(got - want)}"
            )
        want_c, got_c = set(_CONTROL.findall(text)), set(_CONTROL.findall(other))
        if want_c != got_c:
            raise LocaleError(
                f"{language}: '{key}' must name the same recovery controls as the "
                f"English source {sorted(want_c)}; found {sorted(got_c)}. A control is "
                "syntax, not vocabulary -- translating it disarms it."
            )


def available_languages(locale_root: Path | str, domain: str = "narrator") -> tuple[str, ...]:
    """The language directories shipping a catalog for ``domain``, sorted.

    The vocabulary a runtime language switch may name. Read from the filesystem
    rather than hard-coded so a new translation joins by existing, the same way
    the test suite's ``SHIPPED`` list discovers it.
    """
    root = Path(locale_root)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(entry.name for entry in root.iterdir() if (entry / f"{domain}.yaml").is_file())
    )


@cache
def load(language: str, locale_root: Path | str, domain: str = "narrator") -> Catalog:
    """Load one language, validated against the English source.

    Cached per ``(language, locale_root)``: a catalog is immutable once validated, and
    every ``NarratorConfig`` would otherwise re-read and re-validate the same files.

    A language with no catalog falls back to English rather than failing: a table
    that has not been translated yet should play in English, not refuse to start. A
    catalog that exists and is *wrong* does fail, because a malformed mechanical
    statement has no safe reading.
    """
    root = Path(locale_root)
    source = _read(root, SOURCE_LANGUAGE, domain)
    _validate(SOURCE_LANGUAGE, source, None, domain)

    requested = (language or SOURCE_LANGUAGE).replace("_", "-")
    for candidate in (requested, requested.split("-")[0], SOURCE_LANGUAGE):
        if (root / candidate / f"{domain}.yaml").is_file():
            raw = source if candidate == SOURCE_LANGUAGE else _read(root, candidate, domain)
            _validate(
                candidate, raw, None if candidate == SOURCE_LANGUAGE else source, domain
            )
            break
    else:  # pragma: no cover - the English catalog is checked above
        raise LocaleError(f"no catalog for {language} and no English fallback")

    if domain != "narrator":
        return Catalog(
            language=str(raw.get("language", candidate)),
            notices={}, confirmations={}, options={},
            announcement="", attributes={}, outcomes={},
            data=raw,
        )
    return Catalog(
        language=str(raw.get("language", candidate)),
        notices=dict(raw["notices"]),
        confirmations=dict(raw["confirmations"]),
        options=dict(raw["options"]),
        announcement=raw["roll"]["announcement"],
        attributes=dict(raw["attributes"]),
        outcomes=dict(raw["outcomes"]),
        data=raw,
    )
