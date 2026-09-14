"""Asynchronous diagnostic recording for channel adapters.

The recorder stores structured channel observations outside campaign state. It never
receives model requests, canon, credentials, or adapter configuration.

It receives one redacted record per executed tool: the tool name, its ``ok`` flag,
its error code, and its event id. It never receives a tool's arguments or its result
content. A disposable terminal campaign deletes its logs at exit, so without this the
retained transcript could not answer whether a die rolled or a tool refused; the
redacted disposition closes that gap without exposing player or canon data.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any, Protocol


class DiagnosticRecorder(Protocol):
    """Accept semantic diagnostic records from any channel transport."""

    def record(self, event: str, **fields: Any) -> None:
        """Accept one semantic record without exposing storage ownership."""
        ...


@dataclass(frozen=True)
class TranscriptCloseResult:
    """Describe the final state of one diagnostic transcript target."""

    completed: bool
    target: Path
    partial_target: Path
    failure: str = ""
    dropped_records: int = 0


class DiagnosticTranscriptRecorder:
    """Queue diagnostic records for a JSON Lines target without caller I/O."""

    schema = "storyteller.narrator.diagnostic.v1"

    def __init__(self, target: Path | str, *, queue_size: int = 1_024) -> None:
        self.target = Path(target)
        self.partial_target = self.target.with_name(f"{self.target.name}.partial")
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        if self.target.exists():
            raise FileExistsError(f"diagnostic target already exists: {self.target}")
        if self.partial_target.exists():
            raise FileExistsError(
                f"diagnostic partial target already exists: {self.partial_target}"
            )

        parent = self.partial_target.parent
        missing_parents: list[Path] = []
        candidate = parent
        while not candidate.exists():
            missing_parents.append(candidate)
            candidate = candidate.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        for created_parent in missing_parents:
            created_parent.chmod(0o700)
        descriptor = os.open(
            self.partial_target,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.chmod(self.partial_target, 0o600, follow_symlinks=False)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self._queue: queue.Queue[tuple[int, str, dict[str, Any]] | None] = queue.Queue(
            maxsize=queue_size
        )
        self._sequence = count(1)
        self._lock = threading.Lock()
        self._closed = False
        self._failure = ""
        self._dropped_records = 0
        self._writer = threading.Thread(
            target=self._write_records,
            name="narrator-diagnostic-recorder",
            daemon=True,
        )
        self._writer.start()

    def record(self, event: str, **fields: Any) -> None:
        """Queue one record and return before the writer performs file I/O."""
        with self._lock:
            if self._closed or self._failure:
                return
            record = (next(self._sequence), event, fields)
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped_records += 1

    def _write_records(self) -> None:
        try:
            while True:
                queued = self._queue.get()
                if queued is None:
                    break
                sequence, event, fields = queued
                record = {
                    "schema": self.schema,
                    "sequence": sequence,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "event": event,
                    **fields,
                }
                self._stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                self._stream.write("\n")
                self._stream.flush()
        except (OSError, TypeError, ValueError) as error:
            with self._lock:
                self._failure = f"{type(error).__name__}: {error}"
        finally:
            try:
                self._stream.close()
            except OSError as error:
                with self._lock:
                    self._failure = f"{type(error).__name__}: {error}"

    def close(self, *, complete: bool) -> TranscriptCloseResult:
        """Flush queued records and rename only a complete, lossless session."""
        with self._lock:
            if self._closed:
                return TranscriptCloseResult(
                    completed=False,
                    target=self.target,
                    partial_target=self.partial_target,
                    failure="recorder already closed",
                    dropped_records=self._dropped_records,
                )
            self._closed = True
            dropped_records = self._dropped_records
            loss_record = (
                next(self._sequence),
                "recorder_loss",
                {"dropped_records": dropped_records},
            )
        if dropped_records:
            while self._writer.is_alive():
                try:
                    self._queue.put(loss_record, timeout=0.1)
                    break
                except queue.Full:
                    continue
        while self._writer.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
            except queue.Full:
                continue
        self._writer.join()
        with self._lock:
            failure = self._failure
            dropped_records = self._dropped_records
        completed = complete and not failure and dropped_records == 0
        if completed:
            failure = self._promote_without_clobbering_target()
            completed = not failure
        return TranscriptCloseResult(
            completed=completed,
            target=self.target,
            partial_target=self.partial_target,
            failure=failure,
            dropped_records=dropped_records,
        )

    def _promote_without_clobbering_target(self) -> str:
        """Create the final link without replacing a target another process created."""
        try:
            os.link(self.partial_target, self.target, follow_symlinks=False)
        except OSError as error:
            return f"{type(error).__name__}: {error}"
        try:
            self.partial_target.unlink()
        except OSError as error:
            return f"{type(error).__name__}: {error}"
        return ""
