from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from dualmap.entities.request import Request


WAITING_MOVABLE = "WAITING_MOVABLE"
COMMITTING = "COMMITTING"
PREFILL_LOCKED = "PREFILL_LOCKED"


@dataclass(frozen=True)
class WaitingMovableEntry:
    """A request body owned by one regional Sidecar before engine admission."""

    request: Request
    replica_id: int
    prefix_hit_tokens: int
    version: int
    enqueued_at: float


class SidecarWaitingMovableRegistry:
    """In-process transport model for Sidecar-owned movable queues.

    The live prototype currently runs all Sidecar clients in one process. This
    registry preserves the distributed ownership contract: a complete request
    exists in exactly one regional queue, while the global scheduler accesses
    queues only through snapshots and versioned transfer/claim operations.
    The lock represents the atomic compare-and-swap endpoint that a deployed
    Sidecar RPC must provide.
    """

    def __init__(self, num_replicas: int) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        self.num_replicas = num_replicas
        self._queues: list[dict[int, WaitingMovableEntry]] = [
            {} for _ in range(num_replicas)
        ]
        self._owners: dict[int, int] = {}
        self._versions: dict[int, int] = {}
        self._lock = threading.RLock()

    def _validate_replica(self, replica_id: int) -> None:
        if not 0 <= replica_id < self.num_replicas:
            raise ValueError(f"invalid replica_id {replica_id}")

    @staticmethod
    def _set_state(
        request: Request,
        state: str,
        replica_id: int,
        version: int,
    ) -> None:
        request._sidecar_waiting_state = state
        request._sidecar_waiting_replica = replica_id
        request._sidecar_waiting_version = version

    def enqueue(
        self,
        replica_id: int,
        request: Request,
        prefix_hit_tokens: int,
    ) -> WaitingMovableEntry:
        self._validate_replica(replica_id)
        request_id = int(request._id)
        with self._lock:
            owner = self._owners.get(request_id)
            if owner is not None:
                raise RuntimeError(
                    f"request {request_id} is already owned by Sidecar {owner}"
                )
            version = self._versions.get(request_id, 0) + 1
            entry = WaitingMovableEntry(
                request=request,
                replica_id=replica_id,
                prefix_hit_tokens=max(0, int(prefix_hit_tokens)),
                version=version,
                enqueued_at=time.perf_counter(),
            )
            self._queues[replica_id][request_id] = entry
            self._owners[request_id] = replica_id
            self._versions[request_id] = version
            self._set_state(request, WAITING_MOVABLE, replica_id, version)
            return entry

    def restore(
        self,
        replica_id: int,
        entry: WaitingMovableEntry,
        prefix_hit_tokens: Optional[int] = None,
    ) -> WaitingMovableEntry:
        """Return a failed engine admission to its original Sidecar queue."""

        hit_tokens = (
            entry.prefix_hit_tokens
            if prefix_hit_tokens is None
            else int(prefix_hit_tokens)
        )
        return self.enqueue(replica_id, entry.request, hit_tokens)

    def owner(self, request_id: int) -> Optional[int]:
        with self._lock:
            return self._owners.get(int(request_id))

    def entry(
        self, replica_id: int, request_id: int
    ) -> Optional[WaitingMovableEntry]:
        self._validate_replica(replica_id)
        with self._lock:
            return self._queues[replica_id].get(int(request_id))

    def entries(self, replica_id: int) -> tuple[WaitingMovableEntry, ...]:
        self._validate_replica(replica_id)
        with self._lock:
            return tuple(self._queues[replica_id].values())

    def requests(self, replica_id: int) -> tuple[Request, ...]:
        return tuple(entry.request for entry in self.entries(replica_id))

    def queue_len(self, replica_id: int) -> int:
        self._validate_replica(replica_id)
        with self._lock:
            return len(self._queues[replica_id])

    def total_count(self) -> int:
        with self._lock:
            return len(self._owners)

    def waiting_tokens(self, replica_id: int) -> int:
        return sum(
            max(1, int(entry.request._num_prefill_tokens))
            for entry in self.entries(replica_id)
        )

    def transfer(
        self,
        source_replica_id: int,
        target_replica_id: int,
        request_id: int,
        *,
        expected_version: int,
        prefix_hit_tokens: int,
    ) -> Optional[WaitingMovableEntry]:
        """Atomically rebind a request if it is still movable at the source."""

        self._validate_replica(source_replica_id)
        self._validate_replica(target_replica_id)
        request_id = int(request_id)
        with self._lock:
            entry = self._queues[source_replica_id].get(request_id)
            if (
                entry is None
                or self._owners.get(request_id) != source_replica_id
                or entry.version != int(expected_version)
            ):
                return None
            if source_replica_id == target_replica_id:
                return entry
            del self._queues[source_replica_id][request_id]
            version = entry.version + 1
            moved = WaitingMovableEntry(
                request=entry.request,
                replica_id=target_replica_id,
                prefix_hit_tokens=max(0, int(prefix_hit_tokens)),
                version=version,
                enqueued_at=entry.enqueued_at,
            )
            self._queues[target_replica_id][request_id] = moved
            self._owners[request_id] = target_replica_id
            self._versions[request_id] = version
            self._set_state(
                entry.request,
                WAITING_MOVABLE,
                target_replica_id,
                version,
            )
            return moved

    def claim(
        self,
        replica_id: int,
        request_id: int,
        *,
        expected_version: int,
    ) -> Optional[WaitingMovableEntry]:
        """Lock placement before the request is submitted to vLLM."""

        self._validate_replica(replica_id)
        request_id = int(request_id)
        with self._lock:
            entry = self._queues[replica_id].get(request_id)
            if (
                entry is None
                or self._owners.get(request_id) != replica_id
                or entry.version != int(expected_version)
            ):
                return None
            del self._queues[replica_id][request_id]
            del self._owners[request_id]
            version = entry.version + 1
            self._versions[request_id] = version
            self._set_state(entry.request, COMMITTING, replica_id, version)
            return WaitingMovableEntry(
                request=entry.request,
                replica_id=replica_id,
                prefix_hit_tokens=entry.prefix_hit_tokens,
                version=version,
                enqueued_at=entry.enqueued_at,
            )

    def mark_prefill_locked(
        self,
        replica_id: int,
        request: Request,
        *,
        expected_version: int,
    ) -> bool:
        """Finalize a successful engine submission at the mobility boundary."""

        self._validate_replica(replica_id)
        request_id = int(request._id)
        with self._lock:
            if (
                request_id in self._owners
                or self._versions.get(request_id) != int(expected_version)
                or getattr(request, "_sidecar_waiting_state", "") != COMMITTING
                or getattr(request, "_sidecar_waiting_replica", -1)
                != replica_id
            ):
                return False
            self._set_state(
                request,
                PREFILL_LOCKED,
                replica_id,
                int(expected_version),
            )
            request._sidecar_prefill_locked_at = time.perf_counter()
            return True


class SidecarWaitingMovableQueueAdapter:
    """Read-mostly compatibility view over Sidecar-owned queues.

    It intentionally stores no requests. Existing RAVEL quote and beam-search
    code can consume queue snapshots without making the global scheduler the
    owner of WAITING_MOVABLE request bodies.
    """

    def __init__(self, registry: SidecarWaitingMovableRegistry) -> None:
        self.registry = registry
        self.num_replicas = registry.num_replicas

    def push(self, replica_id: int, request: Request, prefix_cache_hit_len: int) -> None:
        self.registry.enqueue(replica_id, request, prefix_cache_hit_len)

    def get_queue_len(self, replica_id: int) -> int:
        return self.registry.queue_len(replica_id)

    def is_empty(self, replica_id: int) -> bool:
        return self.get_queue_len(replica_id) == 0

    def get_all_requests(self, replica_id: int) -> list[Request]:
        return list(self.registry.requests(replica_id))

    def get_global_actual_waiting_tokens_count(self, replica_id: int) -> int:
        return self.registry.waiting_tokens(replica_id)

    def get_max_waiting_delay(self, replica_id: int) -> float:
        entries = self.registry.entries(replica_id)
        if not entries:
            return 0.0
        return max(0.0, time.perf_counter() - min(row.enqueued_at for row in entries))

    def del_req(self, replica_id: int, request: Request) -> bool:
        """Reject central mutation; transfer/claim must be versioned Sidecar RPCs."""

        raise RuntimeError(
            "Sidecar WAITING_MOVABLE queues require transfer() or claim(); "
            f"central delete attempted for request {request._id} on {replica_id}"
        )
