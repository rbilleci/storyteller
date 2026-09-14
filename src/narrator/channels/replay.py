"""Replay a scripted channel transcript. The first adapter, and the one that needs no credentials.

The grammar is two rules. A plain `name: text` line is chatter and buffers. A line
starting with `@GM` addresses the narrator and triggers one turn carrying every buffered
line since the previous turn, which reproduces channel history backfill.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from narrator.channels.base import ChannelMessage, ChannelPrincipal, InboundTurn


class TranscriptReplayAdapter:
    """Feed a transcript file through the engine and collect what the narrator says."""

    name = "replay"

    def __init__(
        self,
        transcript: Path | str,
        channel_id: str = "replay",
        backfill_limit: int = 50,
        echo: bool = True,
    ) -> None:
        self.transcript = Path(transcript)
        self.channel_id = channel_id
        self.backfill_limit = backfill_limit
        self.echo = echo
        self.posted: list[str] = []

    async def turns(self) -> AsyncIterator[InboundTurn]:
        """Yield one turn per ``@GM`` line, carrying the chatter that preceded it."""
        buffered: list[ChannelMessage] = []
        for raw in self.transcript.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@GM"):
                # The mention carries an authenticated principal, exactly as a real
                # channel's would: session-zero scripts need ``character_create``'s
                # engine-side account binding to see who spoke, and the fixed
                # ``player`` subject matches the fixed ``player`` author this
                # grammar has always stamped.
                yield InboundTurn(
                    channel_id=self.channel_id,
                    mention=ChannelMessage(
                        "player",
                        line[3:].strip(),
                        principal=ChannelPrincipal("replay", "player", "player"),
                    ),
                    backfill=tuple(buffered[-self.backfill_limit :]),
                )
                buffered = []
                continue
            author, _, text = line.partition(":")
            buffered.append(
                ChannelMessage(author.strip(), text.strip() if text else author.strip())
            )

    async def post(self, channel_id: str, text: str) -> None:
        self.posted.append(text)
        if self.echo:
            print(f"\nGM: {text}\n", flush=True)

    async def close(self) -> None:
        return None
