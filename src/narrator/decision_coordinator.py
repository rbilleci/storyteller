"""In-memory, replay-safe coordination for structured player decisions.

Pending decisions deliberately live outside campaign state.  Human waits therefore hold
no campaign transaction, survive no process restart, and cannot enter backups or event
logs.  The service cancels this coordinator during shutdown.
"""

from __future__ import annotations

import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

from narrator.decisions import (
    CharacterAudience,
    DecisionDraft,
    DecisionResolution,
    DecisionSubmission,
    DecisionValidationError,
    DecisionView,
    PartyAudience,
    PreparedDecision,
    TargetAssignment,
    public_options,
    resolution_from_submission,
)
from narrator.player_directory import PlayerDirectorySnapshot


class DecisionCoordinatorError(RuntimeError):
    """A lifecycle request cannot safely change pending decision state."""


@dataclass(frozen=True)
class SubmissionResult:
    """A semantic submission disposition without private answer content."""

    status: Literal[
        "accepted", "completed", "idempotent", "invalid", "unknown", "unauthorized",
        "conflict", "stale", "cancelled", "inactive",
    ]
    request_id: str = ""
    failure_category: str = ""


@dataclass
class _Pending:
    prepared: PreparedDecision
    tokens: dict[str, str]
    state: Literal["prepared", "active", "completed", "cancelled", "stale"] = "prepared"
    submissions: dict[str, DecisionSubmission] = field(default_factory=dict)
    resolutions: dict[str, DecisionResolution] = field(default_factory=dict)


@dataclass(frozen=True)
class _Tombstone:
    request_id: str
    state: Literal["completed", "cancelled", "stale"]
    assignments: tuple[TargetAssignment, ...]
    tokens: tuple[str, ...]
    submissions: tuple[tuple[str, DecisionSubmission], ...]
    resolutions: tuple[DecisionResolution, ...]


class DecisionCoordinator:
    """Authorize independent answers with one asyncio lock around each state change."""

    def __init__(self, *, tombstone_limit: int = 64) -> None:
        if tombstone_limit < 1:
            raise ValueError("tombstone_limit must be positive")
        import asyncio

        self._lock = asyncio.Lock()
        self._pending: dict[str, _Pending] = {}
        self._tokens: dict[str, str] = {}
        self._tombstones: OrderedDict[str, _Tombstone] = OrderedDict()
        self._tombstone_limit = tombstone_limit

    async def prepare(
        self,
        *,
        draft: DecisionDraft,
        snapshot: PlayerDirectorySnapshot,
        principal,
        context_fingerprint: str,
    ) -> tuple[PreparedDecision, tuple[DecisionView, ...]]:
        """Resolve the declared audience and create inactive assignment views."""
        assignments = self._resolve_audience(draft, snapshot, principal)
        request_id = secrets.token_urlsafe(24)
        prepared = PreparedDecision(
            request_id=request_id,
            kind=draft.kind,
            assignments=assignments,
            context_fingerprint=context_fingerprint,
            draft=draft,
        )
        tokens = {item.character_id: secrets.token_urlsafe(32) for item in assignments}
        pending = _Pending(prepared=prepared, tokens=tokens)
        async with self._lock:
            self._pending[request_id] = pending
            self._tokens.update({token: request_id for token in tokens.values()})
        return prepared, self._views(pending)

    async def activate(self, request_id: str) -> bool:
        """Make a request answerable only after every assigned view was presented."""
        async with self._lock:
            pending = self._pending.get(request_id)
            if pending is None or pending.state != "prepared":
                return False
            pending.state = "active"
            return True

    async def submit(
        self,
        *,
        principal,
        submission: DecisionSubmission,
        context_fingerprint: str,
    ) -> SubmissionResult:
        """Accept one matching answer, preserving idempotent retry and stale safety."""
        async with self._lock:
            request_id = self._tokens.get(submission.presentation_token)
            if request_id is None:
                return SubmissionResult(status="unknown", failure_category="unknown_token")
            pending = self._pending.get(request_id)
            if pending is None:
                return self._submit_tombstone(request_id, principal, submission)
            assignment = self._assignment_for_token(pending, submission.presentation_token)
            if assignment is None:
                return SubmissionResult(status="unknown", failure_category="unknown_token")
            if (
                getattr(principal, "adapter_name", None) != assignment.adapter_name
                or getattr(principal, "subject_id", None) != assignment.subject_id
            ):
                return SubmissionResult(
                    status="unauthorized", request_id=request_id, failure_category="principal_mismatch"
                )
            if pending.state == "prepared":
                return SubmissionResult(status="inactive", request_id=request_id)
            if pending.state == "cancelled":
                return SubmissionResult(status="cancelled", request_id=request_id)
            if pending.state == "stale":
                return SubmissionResult(status="stale", request_id=request_id)
            if context_fingerprint != pending.prepared.context_fingerprint:
                pending.state = "stale"
                self._tombstone(pending)
                return SubmissionResult(status="stale", request_id=request_id, failure_category="context_changed")
            prior = pending.submissions.get(assignment.character_id)
            if prior is not None:
                if prior == submission:
                    status = "completed" if pending.state == "completed" else "idempotent"
                    return SubmissionResult(status=status, request_id=request_id)
                return SubmissionResult(
                    status="conflict", request_id=request_id, failure_category="conflicting_answer"
                )
            try:
                resolution = resolution_from_submission(
                    pending.prepared.draft, assignment.character_id, submission
                )
            except DecisionValidationError:
                return SubmissionResult(
                    status="invalid", request_id=request_id, failure_category="invalid_answer"
                )
            pending.submissions[assignment.character_id] = submission
            pending.resolutions[assignment.character_id] = resolution
            if len(pending.resolutions) == len(pending.prepared.assignments):
                pending.state = "completed"
                self._tombstone(pending)
                return SubmissionResult(status="completed", request_id=request_id)
            return SubmissionResult(status="accepted", request_id=request_id)

    async def resolutions(self, request_id: str) -> tuple[DecisionResolution, ...]:
        """Return ordered immutable answers only after every assigned player answered."""
        async with self._lock:
            pending = self._pending.get(request_id)
            if pending is not None:
                if pending.state != "completed":
                    return ()
                return tuple(
                    pending.resolutions[item.character_id]
                    for item in pending.prepared.assignments
                )
            tombstone = self._tombstones.get(request_id)
            if tombstone is None or tombstone.state != "completed":
                return ()
            return tombstone.resolutions

    async def is_complete(self, request_id: str) -> bool:
        """Return whether a request has collected every independently authorized answer."""
        return bool(await self.resolutions(request_id))

    async def cancel(
        self,
        request_id: str,
        *,
        reason: Literal["shutdown", "facilitator", "stale", "dismissed"],
    ) -> bool:
        """Cancel a whole request only through an authorized lifecycle cause."""
        async with self._lock:
            pending = self._pending.get(request_id)
            if pending is None or pending.state not in {"prepared", "active"}:
                return False
            pending.state = "stale" if reason == "stale" else "cancelled"
            self._tombstone(pending)
            return True

    async def cancel_all_shutdown(self) -> None:
        """Forget every process-local wait during service shutdown."""
        async with self._lock:
            for pending in tuple(self._pending.values()):
                if pending.state in {"prepared", "active"}:
                    pending.state = "cancelled"
                    self._tombstone(pending)

    async def pending_count(self) -> int:
        async with self._lock:
            return sum(item.state in {"prepared", "active"} for item in self._pending.values())

    def _resolve_audience(self, draft, snapshot, principal) -> tuple[TargetAssignment, ...]:
        adapter_name = getattr(principal, "adapter_name", "")
        subject_id = getattr(principal, "subject_id", "")
        if not adapter_name or not subject_id:
            raise DecisionCoordinatorError("decision principal is absent")
        if draft.audience.kind == "current":
            targets = (snapshot.for_principal(adapter_name, subject_id),)
        elif isinstance(draft.audience, CharacterAudience):
            targets = tuple(
                snapshot.for_character(character_id, adapter_name)
                for character_id in draft.audience.character_ids
            )
        elif isinstance(draft.audience, PartyAudience):
            targets = tuple(
                snapshot.for_character(item.character_id, adapter_name)
                for item in snapshot.players
            )
        else:  # pragma: no cover - Pydantic's discriminated union makes this unreachable
            targets = ()
        if not targets or any(item is None for item in targets):
            raise DecisionCoordinatorError("decision audience is unavailable")
        assignments = tuple(
            TargetAssignment(
                character_id=item.character_id,
                character_name=item.character_name,
                adapter_name=item.adapter_name,
                subject_id=item.subject_id,
            )
            for item in targets
            if item is not None
        )
        if len({item.character_id for item in assignments}) != len(assignments):
            raise DecisionCoordinatorError("decision audience repeats a character")
        return assignments

    def _views(self, pending: _Pending) -> tuple[DecisionView, ...]:
        options = public_options(pending.prepared.draft)
        return tuple(
            DecisionView(
                presentation_token=pending.tokens[item.character_id],
                kind=pending.prepared.kind,
                character_id=item.character_id,
                character_name=item.character_name,
                question=pending.prepared.draft.question,
                context=pending.prepared.draft.context,
                options=options,
            )
            for item in pending.prepared.assignments
        )

    @staticmethod
    def _assignment_for_token(pending: _Pending, token: str) -> TargetAssignment | None:
        for assignment in pending.prepared.assignments:
            if pending.tokens.get(assignment.character_id) == token:
                return assignment
        return None

    def _submit_tombstone(self, request_id: str, principal, submission: DecisionSubmission) -> SubmissionResult:
        tombstone = self._tombstones.get(request_id)
        if tombstone is None:
            return SubmissionResult(status="unknown", failure_category="unknown_request")
        assignment = next(
            (
                item
                for item in tombstone.assignments
                if getattr(principal, "adapter_name", None) == item.adapter_name
                and getattr(principal, "subject_id", None) == item.subject_id
            ),
            None,
        )
        if assignment is None:
            return SubmissionResult(status="unauthorized", request_id=request_id, failure_category="principal_mismatch")
        if tombstone.state == "completed":
            prior = dict(tombstone.submissions).get(assignment.character_id)
            return SubmissionResult(
                status="completed" if prior == submission else "conflict",
                request_id=request_id,
                failure_category="" if prior == submission else "conflicting_answer",
            )
        return SubmissionResult(status=tombstone.state, request_id=request_id)

    def _tombstone(self, pending: _Pending) -> None:
        if pending.state not in {"completed", "cancelled", "stale"}:
            return
        tombstone = _Tombstone(
            request_id=pending.prepared.request_id,
            state=pending.state,
            assignments=pending.prepared.assignments,
            tokens=tuple(pending.tokens.values()),
            submissions=tuple(pending.submissions.items()),
            resolutions=tuple(
                pending.resolutions[item.character_id]
                for item in pending.prepared.assignments
                if item.character_id in pending.resolutions
            ),
        )
        self._tombstones[pending.prepared.request_id] = tombstone
        self._tombstones.move_to_end(pending.prepared.request_id)
        self._pending.pop(pending.prepared.request_id, None)
        while len(self._tombstones) > self._tombstone_limit:
            discarded_id, discarded = self._tombstones.popitem(last=False)
            for token in discarded.tokens:
                self._tokens.pop(token, None)
