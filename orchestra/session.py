"""In-process session store with replayable event streams.

Deliberately memory-only: a council run is a single interactive session, and
persisting transcripts would mean owning a retention policy for whatever the
user pasted into the task box. Export the markdown report if you want a record.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    id: str
    task: str
    context: str
    config: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None
    error: str | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    _subscribers: set[asyncio.Queue] = field(default_factory=set)

    async def emit(self, event: dict[str, Any]) -> None:
        event = {"seq": len(self.events), **event}
        self.events.append(event)
        for queue in list(self._subscribers):
            queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        """Late subscribers get the full backlog first, then live events."""
        queue: asyncio.Queue = asyncio.Queue()
        for event in self.events:
            queue.put_nowait(event)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def done(self) -> bool:
        return self.finished.is_set()


class SessionStore:
    def __init__(self, max_sessions: int = 50) -> None:
        self._sessions: dict[str, Session] = {}
        self._order: deque[str] = deque()
        self._max = max_sessions

    def create(self, task: str, context: str, config: dict[str, Any]) -> Session:
        session = Session(id=uuid.uuid4().hex[:12], task=task, context=context, config=config)
        self._sessions[session.id] = session
        self._order.append(session.id)
        while len(self._order) > self._max:
            evicted = self._order.popleft()
            self._sessions.pop(evicted, None)
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for sid in list(self._order)[-limit:][::-1]:
            session = self._sessions[sid]
            out.append(
                {
                    "id": session.id,
                    "task": session.task[:160],
                    "orchestrator": session.config.get("orchestrator"),
                    "done": session.done,
                    "error": session.error,
                }
            )
        return out
