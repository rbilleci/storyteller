---
name: bsh-session-close
description: "Close a Black Sword Hack session: summarise events, record NPC attitudes and discoveries, award Stories, call session_close, and post a concise recap."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  storyteller:
    tags: [game-master, tabletop, black-sword-hack, session-close, mcp]
    related_skills: [bsh-gm, bsh-session-zero]
---

# Black Sword Hack session close

## Overview

You close the session, write the record, and leave the campaign in a state that any future session can resume from files alone.

## Order of work

### Ask for a title

Ask the table for a session title in one line. If nobody answers within one exchange, choose one drawn from the session's most concrete image. Never choose a title that spoils a hidden truth.

### Summarise what happened

Call `campaign_status` and read the scene, the party, the clocks, and the open hooks. Write four to eight sentences covering what the party did, what changed in the world, what it cost them, and what remains unresolved.

Write only what happened at the table. Do not invent events to make a tidier story.

### Record durable facts

Before closing, call `scene_commit` for anything that happened late in the session and was never committed: a moved party, an alarmed faction, a promise made, a secret learned, or a clock that advanced. The session summary is advisory. `scene_commit` is what survives.

Record NPC attitudes plainly: who now trusts the party, who is hunting them, and who is waiting for an answer.

### Award Stories

Award one Story to a character who did something the campaign will remember: a real risk taken, a bargain struck, a loss accepted, or a problem solved without violence. Award nothing for merely attending. Two Stories in one session is unusual and should be justified in the summary.

A character advances after collecting Stories equal to their current level. The tool reports eligibility; the players make the choices at the start of the next session.

### Call the tool

Call `session_close` with the title, the public summary, the Story awards keyed by character id, the open hooks, and the party's stated next intention.

The tool writes the dated summary, increments the session counter, clears session-length conditions, creates a backup archive, and opens the next session log. Read the returned `advancement_eligible` list and tell each eligible player which level they may take next session.

If it refuses with `uncommitted_fiction_debt`, the message names each rolled outcome still missing from the record. Call `scene_commit` for those, then call `session_close` again. Only pass `accept_uncommitted: true` if nothing more is worth writing into canon — the debt is not erased, just left open.

### Post the recap

Post a recap in the channel, of roughly one hundred and fifty words. Cover what the party did, what it cost, and what waits. End with the open question the next session opens on.

Keep hidden truths hidden. The recap is a player-facing document.

### Optional audio recap

If the profile has text to speech available, offer an audio version only after the text recap is posted and the tool has committed. Never generate audio in place of the written record.

## Closing checks

Confirm that `session_close` returned `ok: true`, that the summary path and backup path both exist in the result, and that the session counter advanced. If any of those is missing, say so in the channel and stop; do not post a recap that claims a save that did not happen.
