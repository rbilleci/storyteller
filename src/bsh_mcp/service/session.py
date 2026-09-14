"""Session close: award Stories, write the session summary, back up, and open
the next session.
"""

from __future__ import annotations

from .. import rules
from ..results import ToolEnvelopeFailure, ToolEnvelopeSuccess
from ..store import CampaignError
from .common import guard, success_after_commit
from .scene import _clean_entries


class SessionCloseResult(ToolEnvelopeSuccess):
    session: int
    next_session: int
    summary_path: str
    backup_path: str
    advancement_eligible: list[dict]
    runic_session_tests: list[dict]
    persons: list[dict]
    accepted_uncommitted_debt: list[dict]


class SessionMixin:
    """Session close: Stories, summary, backup, and the next session's open."""

    @guard
    def session_close(
        self,
        session_title: str,
        public_summary: str,
        character_stories_awarded: dict[str, int] | None = None,
        open_hooks: list[str] | None = None,
        next_intention: str = "",
        accept_uncommitted: bool = False,
    ) -> SessionCloseResult | ToolEnvelopeFailure:
        """Write the session summary, award Stories, back up, and open the next session."""
        session_title = session_title.strip()
        public_summary = public_summary.strip()
        if not session_title or not public_summary:
            raise CampaignError(
                "empty_session_record",
                "a session close needs a title and a public summary. The written record is the "
                "only thing a future session reads.",
                ["Ask the table for a title, summarise what happened, then call again."],
            )
        awards = {key: int(value) for key, value in (character_stories_awarded or {}).items()}

        with self.store.transaction(
            "session_close", actor_id="", reason=session_title
        ) as transaction:


            debts = list(transaction.state.fiction_debt)
            accepted_uncommitted_debt: list[dict] = []
            if debts:
                itemized = [
                    {
                        "seq": debt.seq,
                        "tool": debt.tool,
                        "outcome": debt.outcome,
                        "realized_public_text": debt.realized_public_text(),
                    }
                    for debt in debts
                ]
                if not accept_uncommitted:
                    raise CampaignError(
                        "uncommitted_fiction_debt",
                        f"{len(debts)} outcome(s) remain unratified; closing now would leave "
                        "them out of the session record.\n"
                        + "\n".join(
                            f"[{entry['seq']}] {entry['tool']}: "
                            f"{entry['realized_public_text'] or entry['outcome']}"
                            for entry in itemized
                        ),
                        [
                            "Call scene_commit to ratify the outstanding outcomes, then "
                            "call session_close again.",
                            "Call session_close again with accept_uncommitted=true to close "
                            "anyway; the debt stays open and settleable later.",
                        ],
                    )
                accepted_uncommitted_debt = itemized
                transaction.record(
                    f"closed with {len(itemized)} outcome(s) still unratified (accepted): "
                    + ", ".join(str(entry["seq"]) for entry in itemized)
                )

            session_number = transaction.state.session
            eligible: list[dict] = []
            runic_session_tests: list[dict] = []

            canonical_awards: dict[str, int] = {}
            for character_id, count in awards.items():
                if count < 0:
                    raise CampaignError(
                        "invalid_story_award", "a Story award cannot be negative."
                    )
                character = transaction.character(character_id)
                character.stories += count
                canonical_awards[character.id] = canonical_awards.get(character.id, 0) + count
                transaction.touch_character(character.id)
                transaction.record(f"{character.id}: awarded {count} Stories")
                target = rules.eligible_level(character.stories)
                if target > character.level:
                    eligible.append(
                        {
                            "character_id": character.id,
                            "name": character.name,
                            "current_level": character.level,
                            "eligible_level": target,
                            "stories": character.stories,
                        }
                    )
                    transaction.warn(
                        f"{character.name} has {character.stories} Stories and may advance to "
                        f"level {target}. Apply the level-up choices with the players."
                    )

            for character_id in self.store.character_ids():
                character = transaction.character(character_id)
                kept = [
                    condition
                    for condition in character.conditions
                    if condition.scope != "session"
                ]
                pools_before = dict(character.pools)
                self._reset_pools(character, "session")
                if len(kept) != len(character.conditions) or character.pools != pools_before:
                    character.conditions = kept
                    transaction.touch_character(character_id)
                    transaction.record(f"{character_id}: session conditions and pools refreshed")
                if character.runic_weapon is not None:
                    weapon = character.runic_weapon
                    session_test = rules.resolve_test(self.roller, target=weapon.weapon_int)
                    weapon.kills_helpless = session_test.succeeded
                    character.runic_weapon = weapon
                    transaction.touch_character(character_id)
                    transaction.record(
                        f"{character_id}: runic weapon session INT test "
                        f"{'succeeded' if session_test.succeeded else 'failed'}; "
                        f"kills_helpless={weapon.kills_helpless}"
                    )
                    runic_session_tests.append(
                        {
                            "character_id": character.id,
                            "weapon_name": weapon.name,
                            "weapon_int": weapon.weapon_int,
                            "roll": session_test.roll.as_dict(),
                            "outcome": session_test.outcome,
                            "kills_helpless": weapon.kills_helpless,
                        }
                    )

            summary_lines = [
                f"# Session {session_number:03d}: {session_title}",
                "",
                f"Closed at {self.store.clock().isoformat(timespec='seconds')}.",
                "",
                "## Summary",
                "",
                public_summary,
                "",
                "## Stories awarded",
                "",
            ]
            summary_lines += [
                f"- {character_id}: {count}"
                for character_id, count in sorted(canonical_awards.items())
            ] or ["- None."]
            cleaned_hooks = _clean_entries(open_hooks) if open_hooks is not None else None
            summary_lines += ["", "## Open hooks", ""]
            summary_lines += [
                f"- {hook}"
                for hook in (
                    cleaned_hooks if cleaned_hooks is not None else transaction.state.scene.hooks
                )
            ] or ["- None."]


            persons_left = [
                {
                    "id": person.id,
                    "name": person.name,
                    "source": person.source,
                    "first_event": person.first_event,
                    "last_event": person.last_event,
                }
                for person in sorted(
                    transaction.state.scene.persons.values(),
                    key=lambda person: (-person.last_event, person.id),
                )
            ]
            summary_lines += ["", "## Persons present", ""]
            summary_lines += [
                f"- {person['id']}: {person['name']} (introduced at event "
                f"{person['first_event']}, last mentioned at event {person['last_event']})"
                for person in persons_left
            ] or ["- None."]


            if accepted_uncommitted_debt:
                summary_lines += ["", "## Unresolved outcomes", ""]
                summary_lines += [
                    f"- [{entry['seq']}] {entry['tool']}: "
                    f"{entry['realized_public_text'] or entry['outcome']}"
                    for entry in accepted_uncommitted_debt
                ]
            summary_lines += ["", "## Next intention", "", next_intention.strip() or "Undeclared.", ""]

            if cleaned_hooks is not None:
                scene = transaction.state.scene
                scene.hooks = cleaned_hooks
                transaction.state.scene = scene
                transaction.scene_dirty = True

            transaction.state.session += 1
            manifest = transaction.manifest
            manifest.session = transaction.state.session
            transaction.touch_manifest()

            # Stage both advisory files. commit() writes them only after every
            # read that can raise has already succeeded.
            summary_path = self.store.summaries_dir / f"session-{session_number:03d}.md"
            transaction.stage_file(
                summary_path, "\n".join(line.rstrip() for line in summary_lines)
            )
            next_log = self.store.logs_dir / f"session-{transaction.state.session:03d}.md"
            transaction.stage_file(
                next_log,
                f"# Session {transaction.state.session:03d}\n\nNo entries recorded yet.\n",
                only_if_absent=True,
            )

            transaction.record(f"session {session_number} closed; session {transaction.state.session} opened")
            sequence = transaction.commit(
                {
                    "outcome": "session_closed",
                    "session": session_number,
                    "title": session_title,
                    "stories_awarded": canonical_awards,
                    "runic_session_tests": runic_session_tests,
                    "accepted_uncommitted_debt": accepted_uncommitted_debt,
                }
            )

        archive = self.store.backup(label=f"session-{session_number:03d}")

        return success_after_commit(
            transaction,
            f"Session {session_number:03d} closed: {session_title}.",
            sequence=sequence,
            outcome="session_closed",
            state_changes=transaction.changes + [f"created {archive}"],
            narration_facts=[public_summary],
            session=session_number,
            next_session=session_number + 1,
            summary_path=str(summary_path),
            backup_path=str(archive),
            advancement_eligible=eligible,
            runic_session_tests=runic_session_tests,
            persons=persons_left,
            accepted_uncommitted_debt=accepted_uncommitted_debt,
        )
