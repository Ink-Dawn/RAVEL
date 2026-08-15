from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Hashable


class ReplayActivityTracker:
    """Track externally arrived requests until their terminal lifecycle event."""

    def __init__(self) -> None:
        self._active: set[Hashable] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def is_idle(self) -> bool:
        return not self._active

    def mark_arrived(self, request_id: Hashable) -> None:
        if request_id in self._active:
            raise ValueError(f"request {request_id!r} is already active")
        self._active.add(request_id)
        self._idle.clear()

    def mark_terminal(self, request_id: Hashable) -> bool:
        if request_id not in self._active:
            return False
        self._active.remove(request_id)
        if not self._active:
            self._idle.set()
        return True

    async def wait_until_idle(self) -> None:
        await self._idle.wait()


@dataclass(frozen=True)
class ReplayDeadline:
    physical_scheduled_at: float
    logical_arrived_at: float
    logical_offset_s: float
    skipped_idle_s: float
    idle_segments: int


class IdleCompressingReplayClock:
    """Map a scaled trace timeline onto wall time while removing idle gaps.

    Logical arrival gaps remain unchanged. Wall time is advanced only when all
    previously arrived requests have reached a terminal state, so removing the
    gap cannot change request overlap or queue contention.
    """

    def __init__(self, now: Callable[[], float] = time.perf_counter) -> None:
        self._now = now
        self.started_at = float(now())
        self.skipped_idle_s = 0.0
        self.idle_segments = 0

    def physical_deadline(self, logical_offset_s: float) -> float:
        return (
            self.started_at
            + max(0.0, float(logical_offset_s))
            - self.skipped_idle_s
        )

    async def wait_until(
        self,
        logical_offset_s: float,
        activity: ReplayActivityTracker,
    ) -> ReplayDeadline:
        logical_offset_s = max(0.0, float(logical_offset_s))
        while True:
            deadline = self.physical_deadline(logical_offset_s)
            now = float(self._now())
            remaining = deadline - now
            if remaining <= 0.0:
                break
            if activity.is_idle:
                self.skipped_idle_s += remaining
                self.idle_segments += 1
                deadline = now
                break
            try:
                await asyncio.wait_for(
                    activity.wait_until_idle(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                deadline = self.physical_deadline(logical_offset_s)
                break

        return ReplayDeadline(
            physical_scheduled_at=deadline,
            logical_arrived_at=self.started_at + logical_offset_s,
            logical_offset_s=logical_offset_s,
            skipped_idle_s=self.skipped_idle_s,
            idle_segments=self.idle_segments,
        )
