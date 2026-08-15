"""Feature-detected vLLM scheduler patch for dynamic Prefill chunking."""
from __future__ import annotations

import json
import math
import os
import time
from functools import wraps
from pathlib import Path

from .protocol import (
    decode_deadline_budget_s,
    decode_service_s,
    decode_tbt_s,
    is_deferred_phase_priority,
    is_phase_aware_priority,
    safe_prefill_chunk_tokens,
)


_INSTALLED_ATTR = "_ravel_dynamic_prefill_installed"


def _phase_prefill_key(
    seq_group,
    *,
    slope: float,
    intercept: float,
    now_s: float,
) -> tuple[float, ...]:
    """Deadline guard followed by shortest-remaining-Prefill order.

    A non-positive laxity means delaying the request would make its declared
    deadline infeasible even in isolation, so urgent requests use EDF. Safe
    requests use shortest remaining processing time, which minimizes mean
    completion time on a preemptive single service lane.
    """

    deadline_budget_s = decode_deadline_budget_s(
        getattr(seq_group, "priority", None)
    )
    if deadline_budget_s is None:
        raise ValueError("phase-aware priority has no deadline budget")
    deadline_s = float(seq_group.arrival_time) + deadline_budget_s
    remaining_tokens = max(
        1, int(seq_group.get_num_uncomputed_tokens())
    )
    remaining_s = intercept + slope * remaining_tokens
    if is_deferred_phase_priority(
        getattr(seq_group, "priority", None)
    ):
        remaining_s += decode_service_s(
            getattr(seq_group, "priority", None)
        ) or 0.0
    if (
        is_deferred_phase_priority(
            getattr(seq_group, "priority", None)
        )
        and deadline_s > now_s
    ):
        return (
            3.0,
            remaining_s,
            float(seq_group.arrival_time),
        )
    laxity_s = deadline_s - now_s - remaining_s
    if decode_tbt_s(getattr(seq_group, "priority", None)) is not None:
        return (
            0.0,
            deadline_s,
            remaining_s,
            float(seq_group.arrival_time),
        )
    if laxity_s <= 0.0:
        return (
            1.0,
            deadline_s,
            remaining_s,
            float(seq_group.arrival_time),
        )
    return (
        2.0,
        remaining_s,
        deadline_s,
        float(seq_group.arrival_time),
    )


def _phase_prefill_order(
    seq_groups,
    *,
    slope: float,
    intercept: float,
    now_s: float,
):
    """Preserve TTFT/TBT EDF, then reclaim safe completion-SLO slack.

    LATENCY requests form a strict EDF prefix because their first-token and
    per-token deadlines cannot be repaired after Prefill. For completion-only
    requests, the shortest remaining Prefill may move ahead only if executing
    it first leaves the remaining EDF schedule feasible under the conservative
    profiled service bound. Otherwise the order falls back to EDF.
    """

    rows = []
    for seq_group in seq_groups:
        priority = getattr(seq_group, "priority", None)
        budget_s = decode_deadline_budget_s(priority)
        if budget_s is None:
            raise ValueError("phase-aware priority has no deadline budget")
        remaining_s = intercept + slope * max(
            1, int(seq_group.get_num_uncomputed_tokens())
        )
        if is_deferred_phase_priority(priority):
            remaining_s += decode_service_s(priority) or 0.0
        rows.append(
            (
                (
                    math.inf
                    if (
                        is_deferred_phase_priority(priority)
                        and float(seq_group.arrival_time) + budget_s > now_s
                    )
                    else float(seq_group.arrival_time) + budget_s
                ),
                remaining_s,
                float(seq_group.arrival_time),
                seq_group,
                decode_tbt_s(priority) is not None,
            )
        )

    latency = sorted(
        (row for row in rows if row[4]),
        key=lambda row: (row[0], row[2]),
    )
    flexible = [row for row in rows if not row[4]]
    ordered = [row[3] for row in latency]
    cursor_s = now_s + sum(row[1] for row in latency)

    while flexible:
        edf = sorted(flexible, key=lambda row: (row[0], row[2]))
        cumulative_s = 0.0
        minimum_prefix_slack_s = math.inf
        safe = []
        edf_feasible = True
        for row in edf:
            if row[1] <= minimum_prefix_slack_s + 1e-12:
                safe.append(row)
            cumulative_s += row[1]
            slack_s = row[0] - cursor_s - cumulative_s
            minimum_prefix_slack_s = min(
                minimum_prefix_slack_s, slack_s
            )
            if slack_s < 0.0:
                edf_feasible = False

        chosen = (
            min(safe, key=lambda row: (row[1], row[0], row[2]))
            if edf_feasible and safe
            else edf[0]
        )
        ordered.append(chosen[3])
        cursor_s += chosen[1]
        flexible.remove(chosen)

    return ordered


def _partition_unexpired_deferred(seq_groups, *, now_s: float):
    """Keep opportunistic work queued until its latest-start boundary."""

    ready = []
    held = []
    for seq_group in seq_groups:
        priority = getattr(seq_group, "priority", None)
        budget_s = decode_deadline_budget_s(priority)
        if (
            budget_s is not None
            and is_deferred_phase_priority(priority)
            and float(seq_group.arrival_time) + budget_s > now_s
        ):
            held.append(seq_group)
        else:
            ready.append(seq_group)
    return ready, held


def _has_protected_decode(seq_groups) -> bool:
    """Legacy and phase-aware work are protected unless explicitly deferred."""

    return any(
        not is_deferred_phase_priority(
            getattr(seq_group, "priority", None)
        )
        for seq_group in seq_groups
    )


def _required_float(name: str) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        raise RuntimeError(f"{name} is required by the RAVEL vLLM adapter")
    value = float(raw)
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value

def _profile_prefill_bound(profile_path: str) -> tuple[float, float]:
    with Path(profile_path).open("r", encoding="utf-8") as source:
        profile = json.load(source)
    if int(profile.get("schema_version", 0)) != 2:
        raise RuntimeError("RAVEL_SERVICE_PROFILE must use schema_version 2")
    curve = profile.get("prefill_curve")
    if not isinstance(curve, list) or not curve:
        raise RuntimeError("RAVEL_SERVICE_PROFILE has no prefill_curve")
    try:
        slope = max(
            float(row["seconds_per_token"])
            for row in curve
        )
        intercept = max(
            float(row.get("intercept_s", 0.0)) for row in curve
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "RAVEL_SERVICE_PROFILE contains an invalid prefill curve"
        ) from error
    if slope <= 0 or intercept < 0:
        raise RuntimeError(
            "RAVEL_SERVICE_PROFILE prefill bounds must be non-negative"
        )
    return slope, intercept


def _prefill_bound_from_environment() -> tuple[float, float]:
    if os.environ.get("RAVEL_PREFILL_SECONDS_PER_TOKEN", ""):
        slope = _required_float("RAVEL_PREFILL_SECONDS_PER_TOKEN")
        intercept = float(os.environ.get("RAVEL_PREFILL_INTERCEPT_S", "0"))
        if intercept < 0:
            raise RuntimeError("RAVEL_PREFILL_INTERCEPT_S must be non-negative")
        return slope, intercept

    profile_path = os.environ.get("RAVEL_SERVICE_PROFILE", "")
    if profile_path:
        return _profile_prefill_bound(profile_path)
    raise RuntimeError(
        "RAVEL_SERVICE_PROFILE or RAVEL_PREFILL_SECONDS_PER_TOKEN is required"
    )

def install_from_environment() -> None:
    """Install the adapter once; fail fast on unsupported vLLM layouts."""
    from vllm.core.scheduler import Scheduler

    if bool(getattr(Scheduler, _INSTALLED_ATTR, False)):
        return
    original = getattr(Scheduler, "_schedule_chunked_prefill", None)
    if original is None:
        raise RuntimeError(
            "unsupported vLLM: Scheduler._schedule_chunked_prefill is absent"
        )
    original_priority = getattr(Scheduler, "_get_priority", None)
    if original_priority is None:
        raise RuntimeError(
            "unsupported vLLM: Scheduler._get_priority is absent"
        )

    slope, intercept = _prefill_bound_from_environment()

    @wraps(original_priority)
    def get_priority(self, seq_group):
        if (
            seq_group.is_prefill()
            and is_phase_aware_priority(
                getattr(seq_group, "priority", None)
            )
        ):
            return _phase_prefill_key(
                seq_group,
                slope=slope,
                intercept=intercept,
                now_s=time.time(),
            )
        return original_priority(self, seq_group)

    @wraps(original)
    def schedule_chunked_prefill(self):
        running_decodes = [
            seq_group
            for seq_group in self.running
            if not seq_group.is_prefill()
        ]
        running_prefills = [
            seq_group
            for seq_group in self.running
            if seq_group.is_prefill()
        ]

        def run_original():
            protected_decode_active = _has_protected_decode(
                running_decodes
            )
            if not protected_decode_active:
                return original(self)

            ready, held = _partition_unexpired_deferred(
                self.waiting, now_s=time.time()
            )
            if not held:
                return original(self)
            self.waiting = type(self.waiting)(ready)
            self.ravel_decode_guard_events = int(
                getattr(self, "ravel_decode_guard_events", 0)
            ) + 1
            self.ravel_decode_guard_held = int(
                getattr(self, "ravel_decode_guard_held", 0)
            ) + len(held)
            try:
                return original(self)
            finally:
                self.waiting.extend(held)
        if running_prefills and all(
            is_phase_aware_priority(
                getattr(seq_group, "priority", None)
            )
            for seq_group in running_prefills
        ):
            self.running = type(self.running)(running_decodes + list(
                _phase_prefill_order(
                    running_prefills,
                    slope=slope,
                    intercept=intercept,
                    now_s=time.time(),
                )
            ))

        latency_tbt_s = []
        decode_sequences = 0
        for seq_group in self.running:
            if seq_group.is_prefill():
                continue
            decode_sequences += max(
                1, int(seq_group.get_max_num_running_seqs())
            )
            tbt_s = decode_tbt_s(getattr(seq_group, "priority", None))
            if tbt_s is not None:
                latency_tbt_s.append(tbt_s)

        if not latency_tbt_s:
            if self.scheduler_config.policy == "priority":
                waiting = list(self.waiting)
                ordered = (
                    _phase_prefill_order(
                        waiting,
                        slope=slope,
                        intercept=intercept,
                        now_s=time.time(),
                    )
                    if waiting and all(
                        is_phase_aware_priority(
                            getattr(seq_group, "priority", None)
                        )
                        for seq_group in waiting
                    )
                    else sorted(waiting, key=self._get_priority)
                )
                self.waiting = type(self.waiting)(ordered)
            return run_original()

        original_limit = int(
            self.scheduler_config.max_num_batched_tokens
        )
        block_size = int(self.cache_config.block_size)
        prompt_cap = safe_prefill_chunk_tokens(
            min(latency_tbt_s),
            slope,
            intercept,
            block_size,
            original_limit,
        )
        desired_limit = min(
            original_limit,
            int(
                math.ceil(
                    (decode_sequences + prompt_cap) / block_size
                )
                * block_size
            ),
        )
        if desired_limit >= original_limit:
            return run_original()

        self.ravel_dynamic_prefill_events = int(
            getattr(self, "ravel_dynamic_prefill_events", 0)
        ) + 1
        self.ravel_last_prefill_cap_tokens = prompt_cap
        self.scheduler_config.max_num_batched_tokens = desired_limit
        try:
            if self.scheduler_config.policy == "priority":
                waiting = list(self.waiting)
                ordered = (
                    _phase_prefill_order(
                        waiting,
                        slope=slope,
                        intercept=intercept,
                        now_s=time.time(),
                    )
                    if waiting and all(
                        is_phase_aware_priority(
                            getattr(seq_group, "priority", None)
                        )
                        for seq_group in waiting
                    )
                    else sorted(waiting, key=self._get_priority)
                )
                self.waiting = type(self.waiting)(ordered)
            return run_original()
        finally:
            self.scheduler_config.max_num_batched_tokens = original_limit

    Scheduler._get_priority = get_priority
    Scheduler._schedule_chunked_prefill = schedule_chunked_prefill
    setattr(Scheduler, _INSTALLED_ATTR, True)
