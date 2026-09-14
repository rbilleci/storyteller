"""Runtime configuration for the narrator service.

The Discord-shaped keys do not appear. Channel behaviour that generalises, such as
mention gating and history backfill, lives here as channel-neutral settings; anything
that is genuinely Discord-specific belongs to the Discord adapter when it ships.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

#: The channel agent's thinking levels, in ascending order of budget. ``off`` sends the
#: request the narrator has always sent; every other level asks the endpoint to let the
#: model think before it answers, capped at that level's token budget. These names are
#: syntax: a player types one after ``/thinking`` at the terminal, so they stay English
#: in every locale the same way ``/help`` does.
THINKING_LEVELS: tuple[str, ...] = ("off", "low", "medium", "high")


DEFAULT_THINKING_BUDGETS: Mapping[str, int] = {"low": 250, "medium": 1000, "high": 2500}


DEFAULT_THINKING_ANSWER_TOKENS = 1500


class LanguageState:
    """The table's live language override: session state, not configuration.

    ``/language`` at the terminal writes here through ``NarratorConfig.set_language``.
    A mutable cell carried *by* the frozen config, rather than engine state like
    ``_turn_thinking_level``, because every consumer of player-facing text -- the
    notice properties below, ``narrator.delivery``, the engine's announcement scrub --
    already reads ``config.catalog`` at call time through the one config object the
    process shares. Holding the override where those reads converge means a switch
    reaches all of them with no rewiring; holding it anywhere else means finding and
    re-threading each one. Empty means no override: the launch ``language`` stands.
    """

    __slots__ = ("tag",)

    def __init__(self) -> None:
        self.tag = ""

    def __repr__(self) -> str:
        return f"LanguageState(tag={self.tag!r})"


@dataclass(frozen=True)
class NarratorConfig:
    """One campaign's narrator settings."""

    campaign_root: Path


    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    #: The local vLLM endpoint. Requirement ``IR-NARRATOR-ENDPOINT`` names
    #: ``the model server launcher`` as the script that serves it.
    base_url: str = "http://localhost:8000/v1"
    model_id: str = "google/gemma-4-26B-A4B-it"


    max_tokens: int = 8192


    temperature: float = 0.4

    #: Tool calls allowed inside one turn. Strands exposes no per-turn ceiling on a
    #: single agent, so the engine enforces this with a ``BeforeToolCallEvent`` hook.
    max_tool_calls_per_turn: int = 8


    turn_thinking_level: str = "low"

    #: Thought budget per level, in tokens, keyed by every level but ``off``. Read by
    #: ``turn_thinking_budget``; the defaults and their evidence are at
    #: ``DEFAULT_THINKING_BUDGETS``. ``BSH_TURN_THINKING_BUDGET_LOW``, ``_MEDIUM`` and
    #: ``_HIGH`` override one each at launch.
    turn_thinking_budgets: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_THINKING_BUDGETS)
    )

    #: See ``DEFAULT_THINKING_ANSWER_TOKENS``. Bounds one thinking call's output at
    #: budget plus this; never touches a thinking-off call.
    turn_thinking_answer_tokens: int = DEFAULT_THINKING_ANSWER_TOKENS

    #: The game-master address designator. A player turn opening with ``@`` followed by
    #: this token (any casing) is out-of-fiction discussion with the game master: it
    #: routes past the risk floor and the decision planner, and the engine refuses
    #: every tool outside ``narrator.policy.GM_DISCUSSION_TOOLS`` for its duration.
    #: The token is configured syntax, never vocabulary -- ``@MJ`` for a French table,
    #: ``@SL`` for a German one, any script -- which is what keeps the lane
    #: language-independent where a keyword lexicon cannot be. Detection lives in
    #: ``narrator.interactions.gm_discussion_remainder`` and reads nothing but this
    #: token and the turn's leading characters.
    gm_address: str = "GM"


    max_settle_attempts: int = 2


    settle_timeout_seconds: float = 30.0


    sweep_timeout_seconds: float = 30.0
    sweep_max_tokens: int = 1900


    max_decision_rounds: int = 2

    #: Session-zero policy defaults to disabled. A table must opt in explicitly.
    romance_escalation_enabled: bool = False

    #: The decision planner uses the same tool-less guided-decoding model path as the
    #: settler. This bound contains malformed schema loops before a player waits.
    decision_max_tokens: int = 900


    #: The table's own directory context changed between this decision's proposal and
    #: its answer -- a rename, a departure, a new arrival -- so the answer cannot bind
    #: safely. Distinct from ``decision_fault_notice``: nothing was malformed, and a
    #: retry needs no different input, only a moment for the state to settle.
    # ``decision_stale_context_notice`` is a property below, read from the locale catalog.

    #: An answer arrived for an action the session no longer recognises as the one it
    #: proposed -- the declaration moved on, by a new segment or a revision, before the
    #: answer landed. Distinct from ``decision_declined_notice``: nothing here was
    #: actually declined, so this notice never claims that it was.
    # ``decision_stale_confirmation_notice`` is a property below, read from the locale catalog.

    #: A view boundary never narrates or performs an action. The retained typed state
    #: stays available when a channel begins its next decision segment with ``/continue``.
    # ``decision_segment_notice`` is a property below, read from the locale catalog.

    #: A declined risk confirmation terminates the action before narration or mechanics.
    # ``decision_declined_notice`` is a property below, read from the locale catalog.

    #: A committed hazard never falls through to narration when confirmation is unavailable.
    # ``risk_confirmation_notice`` is a property below, read from the locale catalog.


    #: Consent policy is enforced before model narration and never explains private consent state.
    # ``romance_boundary_notice`` is a property below, read from the locale catalog.

    #: A purchase cannot become narration when its authenticated confirmation is unavailable.
    # ``trade_confirmation_notice`` is a property below, read from the locale catalog.

    #: This service-authored receipt follows the only successful atomic purchase path.
    # ``trade_completed_notice`` is a property below, read from the locale catalog.


    settle_max_tokens: int = 900

    #: The hazard assessor shares the settler's request ceiling deliberately: same
    #: model, same endpoint, one tool-less guided-decoding request per firing, and it
    #: fires at most once per risk-flagged turn. It fails closed toward the
    #: confirmation, so these two numbers cap what a broken assessor can cost a turn:
    #: 30 seconds, once, after which the table simply sees the confirmation it would
    #: have seen before the assessor existed.
    assess_timeout_seconds: float = 30.0
    #: ``narrator.assess.HazardAssessment`` is two bounded fields -- a four-value
    #: enum and a 240-character reason -- so 300 tokens covers every schema-legal
    #: object with margin; the budget's job, as with the settler's, is bounding a
    #: degenerate generation.
    assess_max_tokens: int = 300


    classify_timeout_seconds: float = 30.0


    classify_max_tokens: int = 400


    canon_scene_max_chars: int = 8000
    canon_location_max_chars: int = 2600
    canon_npcs_max_chars: int = 1200

    #: Replaces ``discord.history_backfill_limit``. Messages since the previous reply
    #: that ride along with the next turn.
    backfill_limit: int = 50

    #: Conversation turns held in process. Campaign files are canon, so this window only
    #: has to carry immediate context; ``campaign_status`` recovers the rest.
    window_size: int = 40


    #: Posted when this turn's story text is gone — the model call raised, the settle
    #: failed, or the reply scrubbed away to nothing. An audit rejected the first
    #: draft ("Nothing was lost; repeat the action") by execution: a failed settle
    #: proves mechanics already committed inside locked transactions, so dice and
    #: consequences STAND, the narration is exactly what was lost, and a repeated
    #: action re-rolls into a second debt entry because no idempotency layer exists.
    #: The copy therefore says so, and directs continuation rather than repetition.
    # ``fault_notice`` is a property below, read from the locale catalog.

    #: Empty means derive the portable launch from ``campaign_root``. The narrator and
    #: the Model Context Protocol server run under different interpreters, because
    #: ``strands-agents`` requires ``mcp<2.0.0`` and this project pins ``mcp[cli]>=2,<3``.
    server_command: tuple[str, ...] = field(default_factory=tuple)

    #: The table's language, as a BCP-47 tag. Selects the catalog under
    #: ``locale_root`` that every player-facing string below is read from. A language
    #: with no catalog falls back to English rather than refusing to start; a
    #: catalog that exists and is malformed does refuse, because a half-rendered
    #: mechanical statement has no safe reading. See ``narrator.locale``.
    language: str = "en-US"

    #: Where the locale catalogs live. Separate from ``repo_root`` for the same
    #: reason ``campaign_root`` is: a sandbox campaign may point elsewhere while the
    #: code still ships from the checkout.
    locale_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "locale"
    )

    #: The live mid-session override for ``language`` -- see ``LanguageState``.
    #: ``compare=False`` keeps session state out of config equality and hashing:
    #: two configs built alike stay equal whatever their tables later switched to.
    language_state: LanguageState = field(default_factory=LanguageState, compare=False)

    def __post_init__(self) -> None:
        validate_turn_thinking(self.turn_thinking_level, self.turn_thinking_budgets)
        if isinstance(self.turn_thinking_answer_tokens, bool) or not isinstance(
            self.turn_thinking_answer_tokens, int
        ) or self.turn_thinking_answer_tokens <= 0:
            raise ValueError("turn_thinking_answer_tokens must be a positive integer")

    def turn_thinking_budget(self, level: str | None = None) -> int:
        """The thought ceiling for ``level`` (default: the configured level); 0 for off."""
        chosen = self.turn_thinking_level if level is None else level
        if chosen == "off":
            return 0
        return int(self.turn_thinking_budgets[chosen])

    @property
    def active_language(self) -> str:
        """The language in force: the session override when set, else the launch tag."""
        return self.language_state.tag or self.language

    def set_language(self, language: str) -> str:
        """Apply a mid-session language switch; returns the tag as applied.

        The write half of ``LanguageControl``. Fails closed on anything it cannot
        apply: a tag no shipped catalog covers raises ``ValueError``, and so does a
        catalog that exists but refuses validation, because a switch that half-lands
        would leave the table's mechanical statements unreadable. The old language
        stands in either case. The tag is validated by loading it, so the failure
        surfaces here at the player's command rather than mid-turn in a notice.
        """
        from narrator.locale import LocaleError, available_languages, load

        requested = (language or "").strip().replace("_", "-")
        known = available_languages(self.locale_root)
        if not requested or (
            requested not in known and requested.split("-")[0] not in known
        ):
            raise ValueError(f"unknown language: {language!r}")
        try:
            load(requested, self.locale_root)
        except LocaleError as error:
            raise ValueError(str(error)) from error
        self.language_state.tag = requested
        return requested

    @property
    def catalog(self):
        """This table's player-facing text. Cached in ``narrator.locale.load``."""
        from narrator.locale import load

        return load(self.active_language, self.locale_root)

    # Every notice below reads the catalog rather than holding its own copy, so the
    # English wording lives in exactly one place -- ``locale/en/narrator.yaml`` -- and a
    # translation cannot silently diverge from it. The comments above each one record
    # why that notice exists and what it may not claim; the wording itself, and the
    # translator guidance that goes with it, are in the catalog.

    @property
    def decision_fault_notice(self) -> str:
        return self.catalog.notice("decision_fault")

    @property
    def decision_stale_context_notice(self) -> str:
        return self.catalog.notice("decision_stale_context")

    @property
    def decision_stale_confirmation_notice(self) -> str:
        return self.catalog.notice("decision_stale_confirmation")

    @property
    def decision_segment_notice(self) -> str:
        return self.catalog.notice("decision_segment")

    @property
    def decision_declined_notice(self) -> str:
        return self.catalog.notice("decision_declined")

    @property
    def risk_confirmation_notice(self) -> str:
        return self.catalog.notice("risk_confirmation")

    @property
    def classifier_fault_notice(self) -> str:
        return self.catalog.notice("classifier_fault")

    @property
    def romance_boundary_notice(self) -> str:
        return self.catalog.notice("romance_boundary")

    @property
    def trade_confirmation_notice(self) -> str:
        return self.catalog.notice("trade_confirmation")

    @property
    def trade_completed_notice(self) -> str:
        return self.catalog.notice("trade_completed")

    @property
    def withheld_notice(self) -> str:
        return self.catalog.notice("withheld")

    @property
    def fault_notice(self) -> str:
        return self.catalog.notice("fault")


def validate_turn_thinking(level: str, budgets: Mapping[str, int]) -> None:
    """Refuse a level outside ``THINKING_LEVELS`` or a budget set that cannot serve one.

    Every level but ``off`` must carry a positive integer budget, because the engine
    sends that number to the endpoint as the thought ceiling and a missing or
    non-positive one would either crash the first turn or ask for an unbounded thought
    under a name that promised a bound.
    """
    if level not in THINKING_LEVELS:
        raise ValueError(
            f"turn thinking level must be one of {', '.join(THINKING_LEVELS)}; got {level!r}"
        )
    budgeted = tuple(name for name in THINKING_LEVELS if name != "off")
    if not isinstance(budgets, Mapping) or set(budgets) != set(budgeted):
        raise ValueError(
            f"turn thinking budgets must key on exactly {list(budgeted)}; "
            f"got {sorted(budgets) if isinstance(budgets, Mapping) else budgets!r}"
        )
    for name in budgeted:
        value = budgets[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"turn thinking budget for {name} must be a positive integer; got {value!r}")


def turn_thinking_from_env(environ: Mapping[str, str] | None = None) -> tuple[str, dict[str, int]]:
    """Resolve the launch thinking level and budgets from ``BSH_TURN_THINKING*``.

    Shared by ``load_config`` and ``scripts/narrator_serve.py``, which builds its
    ``NarratorConfig`` directly, so the two production paths cannot drift. Unset
    variables keep the dataclass defaults; a malformed one raises ``ValueError`` with
    the variable's name, so a launch fails at the shell rather than on the first turn.
    """
    source = os.environ if environ is None else environ
    level = (source.get("BSH_TURN_THINKING") or "low").strip().lower()
    if level not in THINKING_LEVELS:
        raise ValueError(f"BSH_TURN_THINKING must be one of {', '.join(THINKING_LEVELS)}")
    budgets = dict(DEFAULT_THINKING_BUDGETS)
    for name in budgets:
        raw = source.get(f"BSH_TURN_THINKING_BUDGET_{name.upper()}")
        if raw is None or not raw.strip():
            continue
        try:
            value = int(raw.strip())
        except ValueError as error:
            raise ValueError(f"BSH_TURN_THINKING_BUDGET_{name.upper()} must be a positive integer") from error
        if value <= 0:
            raise ValueError(f"BSH_TURN_THINKING_BUDGET_{name.upper()} must be a positive integer")
        budgets[name] = value
    return level, budgets


def load_config(campaign_root: Path | str | None = None) -> NarratorConfig:
    """Build a configuration from the campaign root and the environment.

    Precedence matches ``src/bsh_mcp/store.py`` ``discover_root``: an explicit argument,
    then ``BSH_CAMPAIGN_ROOT``, then the repository this file ships in.
    """
    if campaign_root:
        root = Path(campaign_root).expanduser().resolve()
    elif os.environ.get("BSH_CAMPAIGN_ROOT"):
        root = Path(os.environ["BSH_CAMPAIGN_ROOT"]).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2]

    decision_rounds = int(os.environ.get("BSH_DECISION_ROUNDS", "2"))
    if decision_rounds < 0:
        raise ValueError("BSH_DECISION_ROUNDS must be nonnegative")
    romance_policy = os.environ.get("BSH_ROMANCE_ESCALATION", "0")
    if romance_policy not in {"0", "1"}:
        raise ValueError("BSH_ROMANCE_ESCALATION must be 0 or 1")
    language = os.environ.get("BSH_LANGUAGE", "en-US").strip() or "en-US"
    gm_address = os.environ.get("BSH_GM_ADDRESS", "GM").strip()
    if not gm_address or any(char.isspace() for char in gm_address) or "@" in gm_address:
        raise ValueError("BSH_GM_ADDRESS must be one token without whitespace or @")
    thinking_level, thinking_budgets = turn_thinking_from_env()
    return NarratorConfig(
        campaign_root=root,
        repo_root=Path(__file__).resolve().parents[2],
        base_url=os.environ.get("BSH_LLM_BASE_URL", "http://localhost:8000/v1"),
        model_id=os.environ.get("BSH_LLM_MODEL", "google/gemma-4-26B-A4B-it"),
        max_decision_rounds=decision_rounds,
        romance_escalation_enabled=romance_policy == "1",
        gm_address=gm_address,
        language=language,
        turn_thinking_level=thinking_level,
        turn_thinking_budgets=thinking_budgets,
    )
