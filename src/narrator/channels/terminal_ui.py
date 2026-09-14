"""Prompt-toolkit presentation for the terminal channel.

The channel adapter owns terminal semantics. This module owns interactive rendering,
editable input, and terminal-state restoration.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable
from typing import Protocol

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import FormattedText, fragment_list_width
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from narrator.channels.base import MentionCandidate

#: A ``/`` or ``@`` token opening the word under the cursor -- start of line or
#: preceded by whitespace, so a path or an email address typed mid-line never
#: triggers a menu. Mirrors the boundary rule ``TerminalAdapter._select_option``
#: already uses to keep command syntax out of free-text decision answers.
_SIGIL_TOKEN = re.compile(r"(?:^|\s)([@/][\w-]*)$")


class SigilCompleter(Completer):
    """Typing-aid completion for ``/`` commands and ``@``-mentions.

    Knows nothing about recovery controls, PCs or NPCs -- it renders whatever the
    two callables it is given return, so the narrator-domain vocabulary stays in
    ``TerminalAdapter`` and this module stays presentation-only. Accepting a
    completion inserts text exactly as if the player had typed it; it carries no
    other effect.
    """

    def __init__(
        self,
        *,
        commands: Callable[[], tuple[tuple[str, str], ...]],
        mentions: Callable[[], tuple[MentionCandidate, ...]],
    ) -> None:
        self._commands = commands
        self._mentions = mentions

    def get_completions(self, document, complete_event):
        match = _SIGIL_TOKEN.search(document.text_before_cursor)
        if not match:
            return
        token = match.group(1)
        prefix = token[1:].casefold()
        if token.startswith("/"):
            for name, description in self._commands():
                if name.casefold().startswith(prefix):
                    yield Completion(
                        f"/{name}", start_position=-len(token), display_meta=description
                    )
        else:
            for candidate in self._mentions():
                if prefix in candidate.sigil.casefold():
                    yield Completion(f"@{candidate.sigil}", start_position=-len(token))


class _TerminalUI(Protocol):
    """The presentation operations the terminal channel needs."""

    async def read_line(self, prompt_prefix: str, speaker: str = "") -> str:
        """Read one editable line or raise a terminal control exception.

        ``speaker`` names whoever's line this is in the echoed transcript once it is
        accepted (e.g. a character's own name), bold and followed by ``"> "`` -- ready
        for a multiplayer table where more than one player's lines can appear one
        after another. Empty by default: no name, no separator, matching how a
        single-player echo already reads.
        """
        ...

    async def read_decision_line(self, prompt_prefix: str, speaker: str = "") -> str:
        """Read one decision answer without adding it to ordinary input history."""
        ...

    async def render(self, prefix: str, text: str) -> None:
        """Render one prefix and one untrusted plain-text body."""
        ...

    async def set_status(self, left: str, right: str = "") -> None:
        """Update the persistent status line. An empty ``left`` hides it entirely;
        ``right``, when given, is right-justified against the terminal's own current
        width rather than simply appended."""
        ...

    async def close(self) -> None:
        """Restore terminal state without closing process streams."""
        ...


#: The inline spans this renderer styles, in the order it must try them. The two-character
#: delimiters come first, so ``**bold**`` never matches the single-character emphasis rule
#: against its own outer characters.
#:
#: Every rule requires a non-space character beside each delimiter, and the underscore
#: rules additionally require a word boundary outside them. That boundary keeps
#: ``combat_defend_directive`` intact, because an identifier's inner underscores sit
#: between word characters and open no span. An unbalanced delimiter matches nothing and
#: prints as itself, which is the behaviour untrusted narration needs.
_INLINE_MARKUP = re.compile(
    r"\*\*(?P<strong>[^*\s](?:[^*\n]*[^*\s])?)\*\*"
    r"|(?<![\w])__(?P<strong_score>[^_\s](?:[^_\n]*[^_\s])?)__(?![\w])"
    r"|`(?P<code>[^`\n]+)`"
    r"|\*(?P<emphasis>[^*\s](?:[^*\n]*[^*\s])?)\*"
    r"|(?<![\w])_(?P<emphasis_score>[^_\s](?:[^_\n]*[^_\s])?)_(?![\w])"
)

#: A bullet opening a line, and the heading marker Markdown uses. Both need the trailing
#: space, so ``*italic*`` opening a line reads as emphasis rather than as a bullet.
_LIST_MARKER = re.compile(r"^(?P<indent>[ \t]*)[-*+][ \t]+")
_HEADING_MARKER = re.compile(r"^#{1,6}[ \t]+")

_STYLE_FOR_GROUP = {
    "strong": "class:markdown-strong",
    "strong_score": "class:markdown-strong",
    "code": "class:markdown-code",
    "emphasis": "class:markdown-emphasis",
    "emphasis_score": "class:markdown-emphasis",
}


def markdown_fragments(text: str) -> list[tuple[str, str]]:
    """Turn one narration into styled fragments, dropping no character of content.

    The terminal printed the model's Markdown verbatim, so a player read ``**Doom**`` and
    a leading hyphen where emphasis and a bullet belonged. This renders those marks.

    It is deliberately small. Narration is untrusted model output, so the guarantee that
    matters is that unmatched markup survives as literal text rather than vanishing. Every
    branch either styles a matched span or copies the source through unchanged.

    The word-boundary rule protects an identifier's inner underscores, and it does not
    protect a leading and trailing pair. ``__dunder__`` therefore renders as strong
    emphasis with both pairs removed, which is the correct Markdown reading of that text
    and the wrong reading of a Python name. An audit recorded that consequence.
    """
    fragments: list[tuple[str, str]] = []
    for line in text.splitlines(keepends=True):
        body, ending = (line[:-1], line[-1:]) if line.endswith("\n") else (line, "")
        style = ""
        heading = _HEADING_MARKER.match(body)
        if heading:
            body = body[heading.end():]
            style = "class:markdown-strong"
        else:
            bullet = _LIST_MARKER.match(body)
            if bullet:
                fragments.append(("", f"{bullet.group('indent')}• "))
                body = body[bullet.end():]
        fragments.extend(_inline_fragments(body, style))
        if ending:
            fragments.append(("", ending))
    return fragments


def _inline_fragments(body: str, style: str) -> list[tuple[str, str]]:
    """Style the inline spans inside one line, leaving every other character alone."""
    fragments: list[tuple[str, str]] = []
    position = 0
    for match in _INLINE_MARKUP.finditer(body):
        group = match.lastgroup
        if group is None or match.group(group) is None:
            continue
        if match.start() > position:
            fragments.append((style, body[position:match.start()]))
        fragments.append((_STYLE_FOR_GROUP[group], match.group(group)))
        position = match.end()
    if position < len(body):
        fragments.append((style, body[position:]))
    return fragments


#: The margin every printed line gets, narration and player text alike, on both
#: sides -- the one deliberate exception is the persistent status line, which is a
#: prompt_toolkit bottom toolbar rather than anything this module prints. A blank
#: character rather than a drawn rule: the gutter itself stays plain regardless of
#: what styling the framed content inside carries.
_BORDER_CHAR = " "
#: How many columns of ``_BORDER_CHAR`` sit on each side.
_BORDER_WIDTH = 2
_MARGIN = _BORDER_CHAR * _BORDER_WIDTH


def _split_into_lines(fragments):
    """Split one flat fragment stream into a list of lines, dropping the newlines.

    ``markdown_fragments`` isolates every line ending into its own single-character
    ``("", "\\n")`` fragment rather than embedding it in a text-bearing one (see its
    own docstring's line-by-line construction), which is what makes splitting here
    safe: every fragment whose text is exactly one newline is a line boundary, and no
    other fragment ever contains one. A source ending in its own trailing newline
    produces one final empty line by this rule alone; that artifact is dropped so a
    normal, newline-terminated block of text yields exactly as many lines as it
    visually has, not one phantom blank line after the last.
    """
    lines: list[list[tuple[str, str]]] = [[]]
    for style, text in fragments:
        if text == "\n":
            lines.append([])
        else:
            lines[-1].append((style, text))
    if len(lines) > 1 and not lines[-1]:
        lines.pop()
    return lines


def _word_wrap(line: list[tuple[str, str]], max_width: int) -> list[list[tuple[str, str]]]:
    """Break one logical line into as many rows as it takes to fit ``max_width``.

    Needed because a printed logical line that is too long for one terminal row does
    not stop existing at that row's edge -- the *terminal* still wraps it onto the
    next one, and it knows nothing about ``_bordered``'s own margin, so a continuation
    row it creates carries no left margin and no right padding at all: caught live,
    the exact bug this function exists to close. Wrapping first, so no row this
    module ever prints is wider than ``max_width``, means the terminal never has to
    choose a break point itself -- every row it receives already fits, so it never
    wraps anything, and ``_bordered`` can go on treating every row the identical way
    it already treats a row from a real newline.

    Works token by token -- a maximal run of space characters, or a maximal run of
    anything else -- rather than character by character. A first attempt processed
    characters one at a time and updated its break point the instant it saw a space,
    which reads as correct until traced by hand: "aaa bbb ccc ddd" wrapped at width 7
    produced ["aaa", "bbb", "ccc ddd"] instead of ["aaa bbb", "ccc ddd"], because by
    the time the second space forced a decision, the break point on record was still
    the first space -- everything between them had already been committed
    character-by-character with no way to reconsider. Deciding one whole word at a
    time instead (does this word, plus the space before it, still fit?) does not have
    that failure mode, and is the same grain every real greedy word-wrap operates at.
    A spacer is held rather than appended immediately, so one sitting exactly at a
    wrap point is dropped instead of starting the next row -- the same silent "eaten"
    boundary space a terminal's own native wrap produces, not a swallowed character of
    real content. A single word wider than ``max_width`` on its own hard-breaks
    character by character instead of overflowing or looping forever -- rare for
    ordinary prose, but possible for an unbroken run like a long identifier or URL.
    """
    if max_width <= 0:
        return [line]
    chars = [(style, char) for style, text in line for char in text]
    if not chars:
        return [line]

    # Alternating runs of space / non-space characters, each keeping its own
    # characters' original styles, so multiple consecutive spaces survive intact
    # rather than being collapsed the way a display-only wrapper like `textwrap`
    # would normalize them.
    tokens: list[list[tuple[str, str]]] = []
    for style, char in chars:
        is_space = char == " "
        if tokens and (tokens[-1][0][1] == " ") == is_space:
            tokens[-1].append((style, char))
        else:
            tokens.append([(style, char)])

    def token_width(token: list[tuple[str, str]]) -> int:
        return sum(get_cwidth(c) for _, c in token)

    rows: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    row_width = 0
    pending_spacer: list[tuple[str, str]] | None = None

    for token in tokens:
        if token[0][1] == " ":
            pending_spacer = token
            continue
        width = token_width(token)
        spacer_width = token_width(pending_spacer) if pending_spacer else 0
        if row and row_width + spacer_width + width > max_width:
            rows.append(row)
            row, row_width = [], 0
            pending_spacer = None
        elif pending_spacer is not None:
            row.extend(pending_spacer)
            row_width += spacer_width
            pending_spacer = None
        if width > max_width:
            for style, char in token:
                char_width = get_cwidth(char)
                if row and row_width + char_width > max_width:
                    rows.append(row)
                    row, row_width = [], 0
                row.append((style, char))
                row_width += char_width
        else:
            row.extend(token)
            row_width += width
    rows.append(row)
    return [_coalesce(r) for r in rows]


def _coalesce(chars: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge adjacent same-style ``(style, one_char)`` pairs back into text runs.

    ``_word_wrap`` above works one character at a time to find its break points, which
    would otherwise hand ``_bordered`` (and, past it, ``print_formatted_text``) a
    separate fragment per character instead of the same runs ``markdown_fragments``
    or a plain typed line already produced -- correct, since nothing renders wrong
    either way, but needlessly larger and harder to read back out in a test.
    """
    merged: list[tuple[str, str]] = []
    for style, char in chars:
        if merged and merged[-1][0] == style:
            merged[-1] = (style, merged[-1][1] + char)
        else:
            merged.append((style, char))
    return merged


def _bordered(fragments, width: int, *, border_style: str, fill_style: str = "") -> list[tuple[str, str]]:
    """Frame every visual row of ``fragments`` with a ``_BORDER_WIDTH``-wide margin
    on each side.

    Operates on an already-styled fragment stream rather than raw text, so neither
    caller (narration or the player's own echoed line) re-derives line boundaries or
    re-applies per-span styling -- ``_split_into_lines`` above is what makes that
    safe. Each *logical* line (split on a real newline) is further word-wrapped to
    the space between the two margins (``_word_wrap`` above) before being bordered,
    so a paragraph longer than one row gets its margin on every row it actually
    occupies, not only its first. Each row is then padded with ``fill_style`` out to
    the terminal's own width between the two margins, so a caller whose ``fill_style``
    sets a background paints the whole row rather than stopping at the last printed
    character. Every row, including the last,
    gets its own trailing newline, so a caller needs no separate "did this already
    end in one" check the way ``render`` used to.
    """
    result: list[tuple[str, str]] = []
    inner_width = max(width - 2 * _BORDER_WIDTH, 0)
    for logical_line in _split_into_lines(fragments):
        for row in _word_wrap(logical_line, inner_width):
            padding = max(inner_width - fragment_list_width(row), 0)
            result.append((border_style, _MARGIN))
            result.extend(row)
            if padding:
                result.append((fill_style, " " * padding))
            result.append((border_style, _MARGIN))
            result.append(("", "\n"))
    return result


#: The rule character framing the live input line top and bottom, replacing an
#: earlier left/right margin plus background shading -- the same idiom Claude
#: Code's own terminal UI uses for its input box: no fill, just a line above and a
#: line below. Not imported from ``prompt_toolkit.widgets.base``'s own ``Border``
#: (which names the identical character): this module already defines its own
#: constants for the rest of its framing rather than reaching into that submodule,
#: and a one-line definition costs less than the extra import.
_RULE_CHAR = "─"


def _find_split_slot(container, buffer):
    """Find the ``HSplit``-like container and index holding, as a direct child, the
    subtree that contains ``buffer``'s own ``Window``.

    Lets a caller splice new siblings in immediately around that child -- the
    mechanism behind the live input line's top/bottom rule, since neither prompt_toolkit
    nor ``Window`` has a "decoration above/below" concept the way ``left_margins``/
    ``right_margins`` cover the sides; only inserting real sibling rows in the parent
    ``HSplit`` does. Tries every container's own direct children first (returning the
    shallowest match) before descending further, so this finds ``main_input_container``'s
    own ``HSplit`` -- the correct splice point -- rather than some deeper container
    that also happens to contain the buffer.
    """
    children = getattr(container, "children", None)
    if children is not None:
        for index, child in enumerate(children):
            if _find_buffer_window(child, buffer) is not None:
                return container, index
    for child in container.get_children():
        found = _find_split_slot(child, buffer)
        if found is not None:
            return found
    return None


def _find_buffer_window(container, buffer) -> Window | None:
    """Locate the ``Window`` prompt_toolkit built around one specific buffer.

    ``PromptSession`` exposes its assembled ``Layout`` publicly but has no constructor
    argument for the live input line's own background -- confirmed against this
    project's own installed prompt_toolkit (3.0.53) by walking a session's layout and
    finding the default buffer's ``BufferControl`` sitting in an otherwise-unstyled
    ``Window`` (``style == \"\"``). Confirmed separately, from
    ``prompt_toolkit.shortcuts.prompt``'s own ``_split_multiline_prompt``:
    ``has_before_fragments`` is true only when the ``message`` text itself contains
    ``\\n``, and this session's message never does, so the prompt chevron renders
    through ``get_line_prefix`` *inside* this same window rather than a separate one --
    one restyle covers the chevron and the typed text together, matching the boxed echo
    below rather than leaving a gap before it.
    """
    if isinstance(container, Window) and isinstance(container.content, BufferControl):
        if container.content.buffer is buffer:
            return container
    for child in getattr(container, "get_children", lambda: [])():
        found = _find_buffer_window(child, buffer)
        if found is not None:
            return found
    return None


class PromptToolkitTerminalUI:
    """Run one persistent prompt-toolkit application across every line this session reads.

    A naive implementation calls ``PromptSession.prompt_async`` once per line, the way
    this class used to. Each call is its own self-contained ``Application`` run, and
    prompt_toolkit deliberately hides the bottom toolbar (``& ~is_done`` in its own
    container filter) during the still frame it draws right after a line is accepted, so
    the status line visibly vanished the instant a player pressed Enter and only came
    back once the next prompt started -- which, in this game, is the entire narration-
    generation wait, the single moment a status line is most worth seeing. Keeping one
    ``Application`` running for the whole channel lifetime means it is never ``is_done``
    between turns, only at real shutdown.

    This still reuses ``PromptSession`` for its layout, key bindings, history, and CPR
    handling rather than hand-building an ``Application`` -- the one piece of stock
    behaviour it does not want is the default buffer's accept handler calling
    ``app.exit()`` on every accepted line, so that handler is replaced with one that
    hands the line to a queue instead. ``prompt_toolkit.print_formatted_text`` already
    detects a running ``Application`` (via its process-wide current-app session, not a
    per-task contextvar) and interleaves safely above it, so ``render`` needs no change
    to keep working while the application runs continuously.
    """

    def __init__(
        self,
        *,
        input: Input | None = None,
        output: Output | None = None,
        completer: Completer | None = None,
    ) -> None:
        self._style = Style.from_dict(
            {
                "player-prefix": "bold ansicyan",
                "output-prefix": "bold ansimagenta",
                "markdown-strong": "bold",
                "markdown-emphasis": "italic",
                "markdown-code": "ansicyan",
                "player-input": "bold",
                "player-border-active": "bold ansicyan",
                "output-border": "ansimagenta",
            }
        )
        self._input = input
        self._output = output
        self._status_left: str | None = None
        self._status_right: str = ""
        self._prompt_message: str = ""
        self._input_lines: asyncio.Queue[str] = asyncio.Queue()
        self._app_task: asyncio.Task[None] | None = None
        self._session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            input=input,
            output=output,
            style=self._style,
            bottom_toolbar=self._render_status_bar,
            message=lambda: FormattedText([("class:player-prefix", self._prompt_message)]),
            completer=completer,
            complete_while_typing=completer is not None,
            # PromptSession's own reservation (``_get_default_buffer_control_height``)
            # forces the input line's own window to a minimum of ``reserve_space_for_menu``
            # rows the instant ``complete_while_typing`` is true -- i.e. for this whole
            # session, not only while a menu is actually open, since it keys off the
            # filter being enabled rather than ``complete_state``. That is the empty-line
            # bloat this fixes: 0 disables the built-in version, and the callable
            # assigned to the window below replaces it with one keyed on real completion
            # state, sized to the match count.
            reserve_space_for_menu=0,
        )
        self._session.default_buffer.accept_handler = self._on_accept
        # Best-effort: a future prompt_toolkit release could change enough of the
        # layout it builds internally that this lookup stops finding a match, in
        # which case the live input line simply keeps its own default styling rather
        # than the codebase raising over a cosmetic miss.
        input_window = _find_buffer_window(
            self._session.layout.container, self._session.default_buffer
        )
        if input_window is not None:
            input_window.dont_extend_height = to_filter(True)
            # The completion menu is a Float scoped to this window's own small
            # FloatContainer (confirmed by inspecting what prompt_toolkit passes that
            # Float at render time) -- it can only ever render into extra height this
            # specific window reports, never into a sibling window elsewhere in the
            # layout. Reserving that height here, sized to the live match count and
            # zero otherwise, is what makes a menu appear at all while keeping the
            # line back to one row the instant nothing is completing.
            input_window.height = self._completions_reserved_height
        slot = _find_split_slot(self._session.layout.container, self._session.default_buffer)
        if slot is not None:
            parent, index = slot
            rule_above = Window(char=_RULE_CHAR, height=1, style="class:player-border-active")
            rule_below = Window(char=_RULE_CHAR, height=1, style="class:player-border-active")
            parent.children.insert(index, rule_above)
            parent.children.insert(index + 2, rule_below)
        self._closed = False

    def _completions_reserved_height(self) -> Dimension:
        """The input line's own height while a completion menu is open, else one row.

        Sized to the live match count (capped at 16, matching the stock
        ``CompletionsMenu``'s own ``max_height``) rather than a fixed reservation, so
        two matches never reserve room for eight. Keyed on ``complete_state`` -- a
        menu is genuinely open -- not ``complete_while_typing``, which is merely
        enabled for the whole session and would reproduce the original bug.
        """
        state = self._session.default_buffer.complete_state
        if state is None or not state.completions:
            return Dimension()
        rows = 1 + min(len(state.completions), 16)
        return Dimension(min=rows)

    def _on_accept(self, buffer) -> bool:
        """Hand one accepted line to a reader instead of ending the application.

        Returning ``False`` still clears the buffer afterward (``Buffer.validate_and_handle``'s
        own behaviour), exactly as a fresh ``prompt_async`` call used to.
        """
        self._input_lines.put_nowait(buffer.document.text)
        return False

    async def _run_app(self) -> None:
        with patch_stdout(raw=False):
            await self._session.app.run_async()

    async def _next_line(self, prompt_prefix: str, speaker: str = "") -> str:
        if self._closed:
            raise RuntimeError("terminal UI is closed")
        if self._app_task is None:
            self._app_task = asyncio.ensure_future(self._run_app())
        self._prompt_message = prompt_prefix
        self._session.app.invalidate()

        get_line = asyncio.ensure_future(self._input_lines.get())
        try:
            done, _ = await asyncio.wait(
                {get_line, self._app_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if get_line in done:
                text = get_line.result()
                # Keeping one Application alive across turns (4dcc847) means the accept-time
                # redraw that used to leave a finished line in scrollback no longer runs, so
                # nothing else commits this line to real output before the buffer clears it.
                #
                # ``speaker`` replaces the chevron this echo used to carry and then lost
                # (a single-player table has no use naming who's typing when the status
                # line already does) -- multiplayer does, once more than one player's
                # lines can appear one after another with nothing else to tell them
                # apart. The whole line prints bold, speaker included, rather than
                # shaded, so it still reads as a distinct block without a fill style
                # that would otherwise need padding out to the box's own width. A blank
                # line above and below sets the box off from whatever
                # came before it and whatever the reply prints next, neither of which add
                # spacing of their own; the blank lines themselves stay borderless, since
                # they are margin, not a line of text.
                echo_fragments = (
                    [
                        ("class:player-prefix class:player-input", speaker),
                        ("class:player-input", "> "),
                        ("class:player-input", text),
                    ]
                    if speaker
                    else [("class:player-input", text)]
                )
                width = self._session.app.output.get_size().columns
                bordered = _bordered(
                    echo_fragments,
                    width,
                    border_style="class:player-border-active",
                )
                content = [("", "\n"), *bordered, ("", "\n")]
                print_formatted_text(
                    FormattedText(content),
                    end="",
                    style=self._style,
                    output=self._session.app.output,
                )
                return text
            # The application itself ended first -- Ctrl-C, Ctrl-D, or close() -- before
            # a line arrived. Its own key bindings set an EOFError/KeyboardInterrupt as
            # the exit exception for exactly this case; a plain close() sets none, which
            # reads the same as end-of-input to a reader still waiting on a line.
            error = self._app_task.exception()
            if error is not None:
                raise error
            raise EOFError()
        finally:
            if not get_line.done():
                get_line.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await get_line

    async def read_line(self, prompt_prefix: str, speaker: str = "") -> str:
        return await self._next_line(prompt_prefix, speaker)

    async def read_decision_line(self, prompt_prefix: str, speaker: str = "") -> str:
        """Use a one-shot memory history so answers never join normal recall."""
        default_buffer = self._session.default_buffer
        original_history = default_buffer.history
        default_buffer.history = InMemoryHistory()
        try:
            return await self._next_line(prompt_prefix, speaker)
        finally:
            default_buffer.history = original_history
            # A redraw between the decision's own accept and this restore can have
            # already started loading the buffer's working lines from the one-shot
            # history above -- ``Buffer.reset`` only cancels a load in flight, it does
            # not itself re-trigger one, and that load reads whatever ``self.history``
            # was at the moment it started, not whatever it is later reassigned to.
            # Resetting again after the restore discards that (if it happened) so the
            # buffer's next load reads the just-restored, real history instead.
            default_buffer.reset()

    async def render(self, prefix: str, text: str) -> None:
        if self._closed:
            raise RuntimeError("terminal UI is closed")
        fragments = []
        if prefix:
            fragments.append(("class:output-prefix", prefix))
        fragments.extend(markdown_fragments(text))
        width = self._session.app.output.get_size().columns
        bordered = _bordered(fragments, width, border_style="class:output-border")
        print_formatted_text(
            FormattedText(bordered),
            end="",
            style=self._style,
            output=self._session.app.output,
        )

    async def set_status(self, left: str, right: str = "") -> None:
        if self._closed:
            return
        self._status_left = left or None
        self._status_right = right
        self._session.app.invalidate()

    def _render_status_bar(self) -> str | None:
        """Right-justify ``self._status_right`` against ``self._status_left``.

        A plain string, not styled fragments: the bar carries no per-segment styling
        of its own, only prompt_toolkit's default ``bottom-toolbar`` class on the
        whole row, so there is nothing a fragment list would add here. Read at render
        time rather than padded once in ``set_status``, the same reason the boxed
        echo and the live input line both query the terminal's width when they
        actually print rather than when the text is first known: a mid-session
        resize must not leave the right half padded for a width that no longer holds.
        """
        if self._status_left is None:
            return None
        if not self._status_right:
            return self._status_left
        width = self._session.app.output.get_size().columns
        padding = max(width - len(self._status_left) - len(self._status_right), 2)
        return f"{self._status_left}{' ' * padding}{self._status_right}"

    async def close(self) -> None:
        self._closed = True
        if self._app_task is not None:
            if not self._app_task.done():
                self._session.app.exit()
            with contextlib.suppress(BaseException):
                await self._app_task
