"""Wire-compatible priority metadata shared by RAVEL and the engine shim."""
from __future__ import annotations

import math
from typing import Optional


PRIORITY_METADATA_SCALE = 1_000_000
PHASE_METADATA_SCALE = 10_000_000
PHASE_SERVICE_MARKER = 1_000_000
PRIORITY_DEADLINE_HZ = 1_000
PRIORITY_MAGIC = 1 << 62
PRIORITY_PAYLOAD_LIMIT = PRIORITY_MAGIC // 2
PHASE_PRIORITY_MAGIC = 1 << 61
PHASE_PRIORITY_PAYLOAD_LIMIT = PHASE_PRIORITY_MAGIC // 2
PHASE_PRIORITY_DEFERRED_FLAG = PHASE_PRIORITY_PAYLOAD_LIMIT // 2
PHASE_PRIORITY_VALUE_LIMIT = PHASE_PRIORITY_DEFERRED_FLAG


def encode_priority(
    deadline_at_s: float,
    tbt_s: Optional[float],
) -> int:
    """Preserve millisecond EDF order and carry an optional TBT.

    Encoded priorities occupy a reserved negative range. This makes the
    engine adapter inert for ordinary vLLM priorities while keeping values
    inside signed int64 for more than a century of monotonic-clock uptime.
    """
    deadline_tick = max(
        0, int(float(deadline_at_s) * PRIORITY_DEADLINE_HZ)
    )
    metadata = 0
    if tbt_s is not None:
        metadata = max(
            1,
            min(
                PRIORITY_METADATA_SCALE - 1,
                int(float(tbt_s) * 1_000_000),
            ),
        )
    payload = deadline_tick * PRIORITY_METADATA_SCALE + metadata
    if payload >= PRIORITY_PAYLOAD_LIMIT:
        raise OverflowError("deadline is outside the priority wire range")
    return -PRIORITY_MAGIC + payload


def encode_phase_priority(
    deadline_budget_s: float,
    tbt_s: Optional[float],
    *,
    deferred: bool = False,
    service_s: Optional[float] = None,
) -> int:
    """Carry a remaining deadline budget across unsynchronized hosts.

    Unlike :func:`encode_priority`, the first argument is a duration measured
    when the Router dispatches the request. The engine reconstructs a local
    deadline from its own request-arrival clock, so no wall or monotonic clock
    synchronization is required.
    """

    budget_tick = max(
        0, int(float(deadline_budget_s) * PRIORITY_DEADLINE_HZ)
    )
    if tbt_s is not None and service_s is not None:
        raise ValueError("phase priority cannot carry TBT and service demand")
    metadata = 0
    if tbt_s is not None:
        metadata = max(
            1,
            min(
                PHASE_SERVICE_MARKER - 1,
                int(float(tbt_s) * 1_000_000),
            ),
        )
    elif service_s is not None and float(service_s) > 0:
        service_ms = max(
            1,
            min(
                PHASE_METADATA_SCALE - PHASE_SERVICE_MARKER - 1,
                int(math.ceil(float(service_s) * 1_000)),
            ),
        )
        metadata = PHASE_SERVICE_MARKER + service_ms
    payload = budget_tick * PHASE_METADATA_SCALE + metadata
    if payload >= PHASE_PRIORITY_VALUE_LIMIT:
        raise OverflowError("deadline budget is outside the priority range")
    if deferred:
        payload += PHASE_PRIORITY_DEFERRED_FLAG
    return -PHASE_PRIORITY_MAGIC + payload


def _decode_payload(priority: object) -> tuple[Optional[int], bool]:
    if priority is None:
        return None, False
    value = int(priority)
    for magic, limit, phase_aware in (
        (PRIORITY_MAGIC, PRIORITY_PAYLOAD_LIMIT, False),
        (
            PHASE_PRIORITY_MAGIC,
            PHASE_PRIORITY_PAYLOAD_LIMIT,
            True,
        ),
    ):
        payload = value + magic
        if 0 <= payload < limit:
            return payload, phase_aware
    return None, False


def decode_tbt_s(priority: object) -> Optional[float]:
    """Return the encoded TBT, or None for ordinary/legacy priorities."""
    payload, phase_aware = _decode_payload(priority)
    if payload is None:
        return None
    if phase_aware:
        payload %= PHASE_PRIORITY_DEFERRED_FLAG
        metadata = payload % PHASE_METADATA_SCALE
        if metadata >= PHASE_SERVICE_MARKER:
            return None
    else:
        metadata = payload % PRIORITY_METADATA_SCALE
    if metadata <= 0:
        return None
    return metadata / 1_000_000.0


def decode_service_s(priority: object) -> Optional[float]:
    """Return a phase-aware conservative service demand in seconds."""

    payload, phase_aware = _decode_payload(priority)
    if payload is None or not phase_aware:
        return None
    payload %= PHASE_PRIORITY_DEFERRED_FLAG
    metadata = payload % PHASE_METADATA_SCALE
    if metadata < PHASE_SERVICE_MARKER:
        return None
    return (metadata - PHASE_SERVICE_MARKER) / 1_000.0


def decode_deadline_s(priority: object) -> Optional[float]:
    """Return a legacy absolute deadline, never a phase budget."""

    payload, phase_aware = _decode_payload(priority)
    if payload is None or phase_aware:
        return None
    deadline_tick = payload // PRIORITY_METADATA_SCALE
    return deadline_tick / PRIORITY_DEADLINE_HZ


def decode_deadline_budget_s(priority: object) -> Optional[float]:
    """Return a phase-aware remaining deadline budget in seconds."""

    payload, phase_aware = _decode_payload(priority)
    if payload is None or not phase_aware:
        return None
    payload %= PHASE_PRIORITY_DEFERRED_FLAG
    budget_tick = payload // PHASE_METADATA_SCALE
    return budget_tick / PRIORITY_DEADLINE_HZ


def is_deferred_phase_priority(priority: object) -> bool:
    """Whether phase-aware work is opportunistic rather than protected."""

    payload, phase_aware = _decode_payload(priority)
    return bool(
        phase_aware
        and payload is not None
        and payload >= PHASE_PRIORITY_DEFERRED_FLAG
    )


def is_phase_aware_priority(priority: object) -> bool:
    """Whether Prefill order is decoupled from Decode TBT metadata."""

    payload, phase_aware = _decode_payload(priority)
    return payload is not None and phase_aware


def safe_prefill_chunk_tokens(
    tbt_s: float,
    prefill_seconds_per_token: float,
    prefill_intercept_s: float,
    block_size: int,
    max_num_batched_tokens: int,
) -> int:
    """Largest block-aligned Prefill chunk within a profiled TBT period."""
    tbt_s = float(tbt_s)
    slope = float(prefill_seconds_per_token)
    intercept = float(prefill_intercept_s)
    block_size = int(block_size)
    maximum = int(max_num_batched_tokens)
    if tbt_s <= 0 or slope <= 0 or intercept < 0:
        raise ValueError("TBT and service slope must be positive")
    if block_size <= 0 or maximum < block_size:
        raise ValueError("invalid block size or batch-token limit")
    available_s = max(0.0, tbt_s - intercept)
    raw_tokens = math.floor(available_s / slope)
    aligned = (raw_tokens // block_size) * block_size
    return min(maximum, max(block_size, aligned))
