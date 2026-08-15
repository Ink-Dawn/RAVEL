from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Optional

from dualmap.cluster.slo import COLLECTIVE, LATENCY, THROUGHPUT
from ravel_engine_adapter.protocol import encode_priority


@dataclass(frozen=True)
class PlannerReplicaSnapshot:
    """Primitive replica state used by the off-loop planner."""

    replica_id: int
    cluster_id: str
    engine_work_tokens: int
    pending_requests: int
    running_requests: int
    fixed_queue_count: int
    fixed_queue_work_tokens: int
    latency_requests: int
    flexible_requests: int
    prefill_tpot_s: float
    prefill_intercept_s: float
    decode_tpot_by_sequence_s: tuple[float, ...]


@dataclass(frozen=True)
class PlannerRequestSnapshot:
    """Router-visible request state with no live ``Request`` reference."""

    request_id: int
    request_type: int
    client_region: str
    arrived_at: float
    deadline_at: float
    limit_s: float
    ttft_limit_s: float
    tbt_limit_s: float
    prompt_tokens: int
    output_tokens_hint: int
    output_tokens_upper_hint: int
    owner_replica: int
    initial_replica: int
    owner_version: int
    attempt: int
    rebind_count: int
    in_flight: bool
    enforce_admission_limit: bool
    rtt_by_replica_s: tuple[float, ...]
    prefix_hit_by_replica: tuple[int, ...]
    residual_guard_by_replica_s: tuple[float, ...]
    decision_guard_by_replica_s: tuple[float, ...]
    risk_ready_by_replica: tuple[bool, ...]
    fixed_work_before_by_replica: tuple[int, ...]
    fixed_requests_before_by_replica: tuple[int, ...]


@dataclass(frozen=True)
class PlannerSnapshot:
    """Complete immutable input for one joint-placement decision."""

    captured_at: float
    replicas: tuple[PlannerReplicaSnapshot, ...]
    requests: tuple[PlannerRequestSnapshot, ...]
    pending_request_limit: int
    beam_width: int
    control_rebind_cost_s: float

from dualmap.entities.request import Request
from dualmap.scheduler.global_scheduler.ravel_base_scheduler import (
    RavelBaseGlobalScheduler,
)


Cost = tuple[int, float, float, float, float, int]
INF_COST: Cost = (1_000_000, math.inf, math.inf, math.inf, math.inf, 1_000_000)


@dataclass(frozen=True)
class Quote:
    replica_id: int
    predicted_ttft_s: float
    point_ttft_s: float
    predicted_objective_s: float
    point_objective_s: float
    predicted_service_s: float
    base_ttft_s: float
    prefix_hit_tokens: int
    prefix_benefit_s: float
    residual_guard_s: float
    risk_calibrated: bool
    feasible: bool


@dataclass(frozen=True)
class Slot:
    replica_id: int
    original_request_id: int
    work_before_tokens: int
    requests_before: int
    original_work_tokens: int


def add_cost(left: Cost, right: Cost) -> Cost:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


class RavelNativeGlobalScheduler(RavelBaseGlobalScheduler):
    """Live port of RAVEL-Native v49 before KV materialization.

    The live controller retains v49's observable semantics: adaptive initial
    placement, one waiting frontier per replica, at most four candidates,
    one rebind per request, cold-prefix rebind safety, lexicographic joint
    assignment, and a current-cohort miss guard. Quotes use the shared live
    linear service profile rather than the offline replay simulator.
    """

    candidate_limit = 4
    frontier_per_replica = 1
    decode_locality_headroom = 2
    prefix_load_headroom = 2
    work_compat_ratio = 2.0
    decode_horizon_tokens = 16
    residual_calibration_enabled = True
    residual_decision_enabled = False
    flexible_locality_headroom = 0
    latency_locality_headroom = 1
    # Latency-class requests may only use remote clusters for initial
    # placement when the best local-cluster quote already consumes a
    # meaningful fraction of the SLO budget (mirrors the mobility gate).
    # Tunable via RAVEL_MOBILE_LATENCY_LOCAL_FRACTION.
    try:
        mobile_latency_local_fraction = float(
            os.environ.get("RAVEL_MOBILE_LATENCY_LOCAL_FRACTION", "0.8")
        )
    except ValueError:
        mobile_latency_local_fraction = 0.8
    # Decode-aware dispersion: even with TTFT slack, latency-class requests may
    # leave the local cluster when the local per-replica pressure (queued +
    # pending + running) reaches this threshold, to avoid decode-concurrency
    # TBT violations.  Tunable via RAVEL_MOBILE_LATENCY_DISPERSE_PRESSURE.
    try:
        mobile_latency_disperse_pressure = int(
            os.environ.get("RAVEL_MOBILE_LATENCY_DISPERSE_PRESSURE", "4")
        )
    except ValueError:
        mobile_latency_disperse_pressure = 4
    # Latency placement watermark: once the local cluster has accumulated
    # replica_cap * n_local_replicas latency-class placements over the run,
    # new latency requests are forced to disperse to remote clusters, bounding
    # per-replica decode concurrency (TBT protection).  Tunable via
    # RAVEL_MOBILE_LATENCY_REPLICA_CAP.
    try:
        mobile_latency_replica_cap = int(
            os.environ.get("RAVEL_MOBILE_LATENCY_REPLICA_CAP", "35")
        )
    except ValueError:
        mobile_latency_replica_cap = 35
    slo_class_headroom_enabled = True
    risk_aware_region_selection = False
    risk_epsilon = 0.05

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self._recent_arrivals: list[tuple[float, str]] = []
        self._recent_prefix_groups: list[str] = []
        self._prefix_diversity_observed = False
        self._first_prefix_group: Optional[str] = None
        self.rebalance_count = 0
        self.rollout_evaluations = 0
        self.rollout_passes = 0
        self._latency_placed_count: Dict[str, int] = {}
        self._placed_count: Dict[int, int] = {}
        self.risk_epsilon = float(
            getattr(args, "ravel_risk_epsilon", self.risk_epsilon)
        )
        if not 0.0 < self.risk_epsilon < 1.0:
            raise ValueError("ravel_risk_epsilon must be in (0, 1)")

    def _limit(self, request: Request) -> float:
        if request._request_type == LATENCY:
            return float(request._slo_constraint[0])
        return float(request._slo_constraint[2])

    def _request_work(self, request: Request, replica_id: int, cold: bool) -> tuple[int, int]:
        """Return conservative work and a non-binding locality hint.

        Without engine-exported cached-token telemetry, the shadow cache cannot
        certify KV residency. Quotes therefore charge the full prompt while the
        estimated hit is used only as a deterministic tie-breaker.
        """
        if cold:
            return max(1, request._num_prefill_tokens), 0
        replica = self.shared_state.replica_budgets[replica_id]
        estimated_recompute = replica.get_num_recompute_token_ids(request._input_ids)
        estimated_hit = max(0, request._num_prefill_tokens - estimated_recompute)
        return max(1, request._num_prefill_tokens), estimated_hit

    @staticmethod
    def _edf_key(request: Request) -> tuple[float, float, int]:
        return (
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )

    def _queued_debt_before(
        self, request: Request, replica_id: int
    ) -> tuple[int, int]:
        request_key = self._edf_key(request)
        total = 0
        count = 0
        for queued in self.global_request_queue.get_all_requests(replica_id):
            if queued is request or self._edf_key(queued) > request_key:
                continue
            work, _ = self._request_work(queued, replica_id, cold=False)
            total += work
            count += 1
        return total, count

    def _engine_work(self, replica_id: int) -> int:
        return self.shared_state.get_num_actual_pending_tokens_replica(replica_id)

    def _pressure(self, replica_id: int) -> int:
        replica = self.shared_state.replica_budgets[replica_id]
        return (
            self.global_request_queue.get_queue_len(replica_id)
            + replica.get_num_pending_req()
            + replica.get_num_running_req()
        )

    def _prospective_sequence_count(
        self, request: Request, replica_id: int
    ) -> int:
        """Upper-bound Decode occupancy after admitting the request.

        Pending and Router-queued requests may enter Decode during the
        newcomer's lifetime. Counting each outstanding request exactly once
        avoids both the old fixed Decode tax and request double counting.
        The service profile clips this count at its measured endpoint.
        """
        replica = self.shared_state.replica_budgets[replica_id]
        request_ids = {
            row._id
            for row in (
                *self.global_request_queue.get_all_requests(replica_id),
                *replica.pending_requests,
                *replica.running_requests,
            )
        }
        already_present = request._id in request_ids
        request_ids.add(request._id)
        counter_bound = (
            self.global_request_queue.get_queue_len(replica_id)
            + replica.get_num_pending_req()
            + replica.get_num_running_req()
            + int(not already_present)
        )
        return max(1, len(request_ids), counter_bound)

    def _residual_bound(
        self, replica_id: int, request_type: int
    ) -> tuple[float, bool]:
        """Return a type-specific conformal-style empirical residual bound.

        The finite-sample rank matches split conformal under exchangeable
        residuals. Because this implementation uses a rolling online window,
        coverage must be validated empirically under drift. If the required
        rank does not exist, the quote is explicitly marked uncalibrated.
        """
        if not self.residual_calibration_enabled:
            return 0.0, True
        replica = self.shared_state.replica_budgets[replica_id]
        if hasattr(replica, "get_objective_residual_history"):
            history = replica.get_objective_residual_history(request_type)
        else:
            histories = getattr(replica, "objective_residual_history", {})
            history = histories.get(
                int(request_type),
                getattr(replica, "ttft_residual_history", ()),
            )
        residuals = list(history)
        if not residuals:
            return 0.0, False
        rank = math.ceil((len(residuals) + 1) * (1.0 - self.risk_epsilon))
        if rank > len(residuals):
            return 0.0, False
        quantile = sorted(float(value) for value in residuals)[rank - 1]
        return max(0.0, quantile), True

    def _residual_guard(self, replica_id: int, request_type: int) -> float:
        return self._residual_bound(replica_id, request_type)[0]

    def _decode_horizon(self, request: Request) -> int:
        return self.decode_horizon_tokens

    def _output_token_bounds(self, request: Request) -> tuple[int, int]:
        """Return separate expected demand and conservative risk demand."""

        expected = max(
            1,
            int(request._routing_output_tokens_hint),
        )
        configured_upper = max(
            expected,
            int(
                getattr(
                    request,
                    "_routing_output_tokens_upper_hint",
                    expected,
                )
            ),
        )
        upper = configured_upper
        if hasattr(self.shared_state, "get_routing_output_hint"):
            upper = max(
                upper,
                int(
                    self.shared_state.get_routing_output_hint(
                        request._request_type,
                        configured_upper,
                    )
                ),
            )
        request._routing_output_tokens_hint_used = expected
        request._routing_output_tokens_upper_hint_used = upper
        return expected, upper

    def _quote(
        self,
        request: Request,
        replica_id: int,
        *,
        work_before_tokens: Optional[int] = None,
        requests_before: Optional[int] = None,
        prospective_sequences: Optional[int] = None,
        cold: bool = False,
    ) -> Quote:
        cluster = self.topology.cluster_for_replica(replica_id)
        request_work, hit_tokens = self._request_work(request, replica_id, cold)
        if work_before_tokens is None:
            queued, queued_requests = self._queued_debt_before(
                request, replica_id
            )
        else:
            queued = max(0, work_before_tokens)
            queued_requests = max(0, int(requests_before or 0))
        engine_work = self._engine_work(replica_id)
        engine_requests = self.shared_state.replica_budgets[
            replica_id
        ].get_num_pending_req()
        elapsed = max(0.0, time.perf_counter() - request._arrived_at)
        rtt = cluster.rtt_s(request._client_region)
        running = self.shared_state.replica_budgets[replica_id].get_num_running_req()

        # The prefill curve is measured under the indicated decode concurrency,
        # so it already contains decode interference. Adding a second decode tax
        # would count the same contention twice.
        prefill_tpot = cluster.prefill_tpot_for(running)
        prefill_intercept = cluster.prefill_intercept_for(running)
        service = (
            (engine_work + queued + request_work) * prefill_tpot
            + (engine_requests + queued_requests + 1) * prefill_intercept
        )
        point_ttft = elapsed + rtt + service
        point_objective = point_ttft
        risk_objective = point_objective
        if request._request_type != LATENCY:
            expected_output, upper_output = self._output_token_bounds(
                request
            )
            decode_occupancy = (
                max(1, int(prospective_sequences))
                if prospective_sequences is not None
                else self._prospective_sequence_count(
                    request, replica_id
                )
            )
            decode_tpot = cluster.decode_tpot_for(decode_occupancy)
            point_objective += (
                max(0, expected_output - 1) * decode_tpot
            )
            risk_objective += (
                max(0, upper_output - 1) * decode_tpot
            )
        residual_guard, calibrated = self._residual_bound(
            replica_id, request._request_type
        )
        decision_guard = residual_guard if self.residual_decision_enabled else 0.0
        predicted_objective = risk_objective + decision_guard
        predicted_ttft = (
            point_ttft + decision_guard
            if request._request_type == LATENCY
            else point_ttft
        )
        risk_ready = calibrated or not self.residual_decision_enabled
        return Quote(
            replica_id=replica_id,
            predicted_ttft_s=predicted_ttft,
            point_ttft_s=point_ttft,
            predicted_objective_s=predicted_objective,
            point_objective_s=point_objective,
            predicted_service_s=service,
            base_ttft_s=(
                rtt + prefill_intercept + request_work * prefill_tpot
            ),
            prefix_hit_tokens=hit_tokens,
            prefix_benefit_s=(
                min(max(0, request_work - 1), hit_tokens) * prefill_tpot
            ),
            residual_guard_s=residual_guard,
            risk_calibrated=risk_ready,
            feasible=predicted_objective <= self._limit(request),
        )

    def _observe_arrival(self, request: Request) -> None:
        now = time.perf_counter()
        history_limit = max(
            2, int(getattr(self, "mobile_regularity_window", 1)) + 1
        )
        self._recent_arrivals.append((now, request._client_region))
        self._recent_arrivals = self._recent_arrivals[-history_limit:]
        prefix_group = request._hash_session_id
        self._recent_prefix_groups.append(prefix_group)
        self._recent_prefix_groups = self._recent_prefix_groups[-history_limit:]
        if self._first_prefix_group is None:
            self._first_prefix_group = prefix_group
        elif prefix_group != self._first_prefix_group:
            self._prefix_diversity_observed = True

    def _arrival_context(self, request: Request, quotes: Dict[int, Quote]) -> tuple[int, int, str]:
        self._observe_arrival(request)
        feasible_count = sum(quote.feasible for quote in quotes.values())
        if len(self._recent_arrivals) < 2:
            return self.decode_locality_headroom, self.prefix_load_headroom, "warmup"
        mean_gap = (
            self._recent_arrivals[-1][0] - self._recent_arrivals[0][0]
        ) / (len(self._recent_arrivals) - 1)
        local_own = [
            max(1e-6, quote.base_ttft_s - self.topology.cluster_for_replica(rid).rtt_s(request._client_region))
            for rid, quote in quotes.items()
            if self.topology.cluster_for_replica(rid).cluster_id == request._client_region
        ]
        pressure_threshold = 1.0 / sum(1.0 / value for value in local_own) if local_own else 0.0
        one_region = len({region for _, region in self._recent_arrivals}) == 1
        one_prefix = len(set(self._recent_prefix_groups)) == 1
        high_pressure = pressure_threshold > 0 and mean_gap <= pressure_threshold + 1e-9
        prefix_scope_matches = one_prefix or self.residual_decision_enabled
        if feasible_count == 0 and one_region and prefix_scope_matches and high_pressure:
            return 0, min(1, self.prefix_load_headroom), "one-sided-infeasible-arrival-pressure"
        return self.decode_locality_headroom, self.prefix_load_headroom, "default"

    def _choose_initial(self, request: Request) -> Quote:
        quotes = {
            replica_id: self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        }
        headroom, prefix_headroom, reason = self._arrival_context(request, quotes)
        if self.slo_class_headroom_enabled and request._request_type == THROUGHPUT:
            headroom = min(headroom, self.flexible_locality_headroom)
            reason = f"{reason}-slo-throughput-balance"
        elif self.slo_class_headroom_enabled and request._request_type == LATENCY:
            headroom = min(headroom, self.latency_locality_headroom)
            reason = f"{reason}-slo-latency-balance"
        feasible_count = sum(quote.feasible for quote in quotes.values())
        recent_count = len(self._recent_arrivals)
        if recent_count >= 2:
            mean_gap = (
                self._recent_arrivals[-1][0] - self._recent_arrivals[0][0]
            ) / (recent_count - 1)
        else:
            mean_gap = math.inf
        local_own = [
            max(1e-6, quote.base_ttft_s - self.topology.cluster_for_replica(rid).rtt_s(request._client_region))
            for rid, quote in quotes.items()
            if self.topology.cluster_for_replica(rid).cluster_id == request._client_region
        ]
        threshold = 1.0 / sum(1.0 / value for value in local_own) if local_own else 0.0
        if feasible_count == 0 and recent_count >= 2 and threshold > 0 and mean_gap <= threshold:
            choice = min(
                quotes.values(),
                key=lambda row: (
                    self._pressure(row.replica_id),
                    row.predicted_service_s,
                    row.predicted_ttft_s,
                    -row.prefix_hit_tokens,
                    row.replica_id,
                ),
            )
            request._cluster_route_reason = "ravel_native_capacity_overload_striping"
            return choice

        local = [
            quote
            for rid, quote in quotes.items()
            if self.topology.cluster_for_replica(rid).cluster_id == request._client_region
            and self.global_request_queue.is_empty(rid)
        ]
        remote = [
            quote
            for rid, quote in quotes.items()
            if self.topology.cluster_for_replica(rid).cluster_id != request._client_region
            and self.global_request_queue.is_empty(rid)
        ]
        feasible_local = [row for row in local if row.feasible]
        feasible_remote = [row for row in remote if row.feasible]
        if request._request_type == LATENCY and feasible_local:
            local_min_objective = min(
                row.predicted_objective_s for row in feasible_local
            )
            local_replicas = [
                rid
                for rid, quote in quotes.items()
                if self.topology.cluster_for_replica(rid).cluster_id
                == request._client_region
            ]
            local_pressure_per_replica = (
                sum(self._pressure(rid) for rid in local_replicas)
                / max(1, len(local_replicas))
            )
            local_cap = self.mobile_latency_replica_cap * max(
                1, len(local_replicas)
            )
            local_count = self._latency_placed_count.get(
                request._client_region, 0
            )
            if (
                local_min_objective
                < self.mobile_latency_local_fraction * self._limit(request)
                and local_pressure_per_replica
                < self.mobile_latency_disperse_pressure
                and local_count < local_cap
            ):
                feasible_remote = []
                remote = []
        calibrated_remote_only = (
            self.risk_aware_region_selection
            and bool(feasible_remote)
            and not feasible_local
            and any(row.residual_guard_s > 0 for row in local)
        )
        if feasible_local or feasible_remote:
            local, remote = feasible_local, feasible_remote
        if calibrated_remote_only:
            reason = f"{reason}-calibrated-remote-feasible"
        if local and remote:
            local_pressure = min(self._pressure(row.replica_id) for row in local)
            remote_pressure = min(self._pressure(row.replica_id) for row in remote)
            available = local if local_pressure <= remote_pressure + headroom else remote
        else:
            available = local or remote
        if not available:
            available = list(quotes.values())
        minimum_pressure = min(self._pressure(row.replica_id) for row in available)
        guarded = [
            row
            for row in available
            if self._pressure(row.replica_id) <= minimum_pressure + prefix_headroom
        ]
        prefix_first = not self._prefix_diversity_observed
        if prefix_first:
            key = lambda row: (
                not row.feasible,
                -row.prefix_hit_tokens,
                row.predicted_service_s,
                row.predicted_ttft_s,
                row.replica_id,
            )
        else:
            key = lambda row: (
                not row.feasible,
                row.predicted_service_s,
                row.predicted_ttft_s,
                -row.prefix_hit_tokens,
                row.replica_id,
            )
        request._cluster_route_reason = f"ravel_native_initial_{reason}"
        return min(guarded, key=key)

    def _record_quote(self, request: Request, quote: Quote) -> None:
        cluster = self.topology.cluster_for_replica(quote.replica_id)
        request._primary_cluster = cluster.cluster_id
        request._primary_replica = quote.replica_id
        request._predicted_ttft_s = quote.predicted_ttft_s
        request._point_predicted_ttft_s = quote.point_ttft_s
        request._predicted_objective_s = quote.predicted_objective_s
        request._point_predicted_objective_s = quote.point_objective_s
        request._ttft_residual_guard_s = (
            quote.residual_guard_s if request._request_type == LATENCY else 0.0
        )
        request._objective_residual_guard_s = quote.residual_guard_s
        request._risk_calibrated = quote.risk_calibrated
        request._ttft_residual_track = self.residual_calibration_enabled
        request._objective_residual_track = self.residual_calibration_enabled
        request._cluster_route_feasible = quote.feasible

    async def _enqueue(self, request: Request) -> None:
        quote = self._choose_initial(request)
        self._record_quote(request, quote)
        if not hasattr(request, "_rebind_count"):
            request._rebind_count = 0
        self.global_request_queue.push(quote.replica_id, request, quote.prefix_hit_tokens)

    def _ordered_queue(self, replica_id: int) -> list[Request]:
        return [item[2] for item in sorted(self.global_request_queue.queues[replica_id])]

    def _candidates(self, trigger: Optional[Request]) -> list[Request]:
        rows = []
        for replica_id in self.topology.replica_ids:
            rows.extend(self._ordered_queue(replica_id)[: self.frontier_per_replica])
        if trigger is not None and any(
            trigger in self.global_request_queue.get_all_requests(rid)
            for rid in self.topology.replica_ids
        ) and trigger not in rows:
            rows.append(trigger)
        rows.sort(
            key=lambda row: (
                0 if row is trigger else 1,
                getattr(row, "_rebind_count", 0),
                row._arrived_at,
                row._id,
            )
        )
        return rows[: self.candidate_limit]

    def _owner(self, request: Request) -> int:
        for replica_id in self.topology.replica_ids:
            if request in self.global_request_queue.get_all_requests(replica_id):
                return replica_id
        raise RuntimeError(f"RAVEL lost queued request {request._id}")

    def _movable_in_flight_count_by_replica(
        self, candidate_ids: set[int]
    ) -> dict[int, int]:
        """Submitted movable requests, absent for non-Sidecar schedulers."""

        del candidate_ids
        return {}

    def _movable_in_flight_work_by_replica(
        self, candidate_ids: set[int]
    ) -> dict[int, float]:
        """Movable prompt debt, absent for non-Sidecar schedulers."""

        del candidate_ids
        return {}

    def _slots(self, candidates: Iterable[Request]) -> list[Slot]:
        candidate_ids = {row._id for row in candidates}
        slots: Dict[int, Slot] = {}
        movable_work = self._movable_in_flight_work_by_replica(candidate_ids)
        for replica_id in self.topology.replica_ids:
            cumulative = self._engine_work(replica_id) - movable_work.get(
                replica_id, 0.0
            )
            queued_count = 0
            for row in self._ordered_queue(replica_id):
                work, _ = self._request_work(row, replica_id, cold=True)
                if row._id in candidate_ids:
                    slots[row._id] = Slot(
                        replica_id, row._id, cumulative, queued_count, work
                    )
                cumulative += work
                queued_count += 1
        return [slots[row._id] for row in candidates]

    def _edge_cost(self, request: Request, source: int, slot: Slot, base_quotes: list[Quote]) -> Cost:
        moved = source != slot.replica_id
        if moved and getattr(request, "_rebind_count", 0) >= 1:
            return INF_COST
        request_work = max(1, request._num_prefill_tokens)
        ratio = max(request_work, slot.original_work_tokens) / max(
            1, min(request_work, slot.original_work_tokens)
        )
        if moved and ratio > self.work_compat_ratio:
            return INF_COST
        quote = self._quote(
            request,
            slot.replica_id,
            work_before_tokens=max(0, slot.work_before_tokens - self._engine_work(slot.replica_id)),
            requests_before=slot.requests_before,
            cold=True,
        )
        miss = int(not quote.feasible)
        lateness = max(0.0, quote.predicted_objective_s - self._limit(request))
        feasible_bases = [row.base_ttft_s for row in base_quotes if row.base_ttft_s <= self._limit(request)]
        weakest = max(feasible_bases, default=quote.base_ttft_s)
        capability_waste = max(0.0, weakest - quote.base_ttft_s) if not miss else 0.0
        mismatch = abs(request_work - slot.original_work_tokens) / max(1, slot.original_work_tokens)
        return (
            miss,
            lateness,
            mismatch,
            capability_waste,
            quote.predicted_ttft_s,
            int(moved),
        )

    def _cohort_misses(self, assignment: Optional[Dict[int, int]] = None) -> int:
        assignment = assignment or {}
        by_replica: Dict[int, list[Request]] = {rid: [] for rid in self.topology.replica_ids}
        for replica_id in self.topology.replica_ids:
            for request in self._ordered_queue(replica_id):
                by_replica[assignment.get(request._id, replica_id)].append(request)
        misses = 0
        for replica_id, requests in by_replica.items():
            cumulative = 0
            request_count = 0
            for request in sorted(requests, key=lambda row: (row._arrived_at, row._id)):
                quote = self._quote(
                    request,
                    replica_id,
                    work_before_tokens=cumulative,
                    requests_before=request_count,
                    cold=True,
                )
                misses += int(not quote.feasible)
                cumulative += max(1, request._num_prefill_tokens)
                request_count += 1
        return misses

    async def _joint_rebalance(self, trigger: Optional[Request]) -> int:
        candidates = self._candidates(trigger)
        if len(candidates) < 2:
            return 0
        sources = [self._owner(row) for row in candidates]
        slots = self._slots(candidates)
        base_quotes = [
            [
                self._quote(
                    row, rid, work_before_tokens=0, requests_before=0, cold=True
                )
                for rid in self.topology.replica_ids
            ]
            for row in candidates
        ]
        costs = [
            [self._edge_cost(row, source, slot, quotes) for slot in slots]
            for row, source, quotes in zip(candidates, sources, base_quotes)
        ]
        identity = tuple(
            next(index for index, slot in enumerate(slots) if slot.original_request_id == row._id)
            for row in candidates
        )
        before: Cost = (0, 0.0, 0.0, 0.0, 0.0, 0)
        for row_index, column in enumerate(identity):
            before = add_cost(before, costs[row_index][column])
        best_cost = before
        best_columns = identity
        for columns in itertools.permutations(range(len(candidates))):
            total: Cost = (0, 0.0, 0.0, 0.0, 0.0, 0)
            for row_index, column in enumerate(columns):
                total = add_cost(total, costs[row_index][column])
            if total < best_cost:
                best_cost, best_columns = total, columns
        assignment = {
            row._id: slots[column].replica_id
            for row, column in zip(candidates, best_columns)
        }
        cross_moves = sum(assignment[row._id] != source for row, source in zip(candidates, sources))
        if cross_moves == 0 or best_cost[0] >= before[0]:
            return 0
        self.rollout_evaluations += 1
        before_misses = self._cohort_misses()
        after_misses = self._cohort_misses(assignment)
        if after_misses >= before_misses:
            return 0
        self.rollout_passes += 1

        for row, source in zip(candidates, sources):
            if not self.global_request_queue.del_req(source, row):
                raise RuntimeError(f"RAVEL failed to remove request {row._id}")
        moved = 0
        for row, source in zip(candidates, sources):
            target = assignment[row._id]
            _, hit = self._request_work(row, target, cold=False)
            self.global_request_queue.push(target, row, hit)
            if target != source:
                assigned_slot = slots[best_columns[candidates.index(row)]]
                assigned_quote = self._quote(
                    row,
                    target,
                    work_before_tokens=max(
                        0, assigned_slot.work_before_tokens - self._engine_work(target)
                    ),
                    requests_before=assigned_slot.requests_before,
                    cold=True,
                )
                self._record_quote(row, assigned_quote)
                row._rebind_count += 1
                row._primary_cluster = self.topology.cluster_for_replica(target).cluster_id
                row._primary_replica = target
                row._cluster_route_reason = "ravel_native_joint_rebind"
                moved += 1
        self.rebalance_count += moved
        return moved

    async def schedule(self, new_request: Optional[Request]) -> int:
        async with self._schedule_lock:
            if new_request is not None:
                await self._enqueue(new_request)
            await self._joint_rebalance(new_request)
            await self._dispatch_schedulable()
            return -1


class RavelNativeV2GlobalScheduler(RavelNativeGlobalScheduler):
    """Frozen v2 behavior used only for paired optimization screens."""

    residual_calibration_enabled = False
    risk_aware_region_selection = False
    slo_class_headroom_enabled = False


class _RavelNativeTokenAwareGlobalScheduler(RavelNativeGlobalScheduler):
    """Screen token/service-aware initial placement with unchanged Recourse."""

    initial_selection_mode = "objective"

    def _choose_initial(self, request: Request) -> Quote:
        quotes = {
            replica_id: self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        }
        _, _, context = self._arrival_context(request, quotes)
        feasible = [row for row in quotes.values() if row.feasible]
        available = feasible or list(quotes.values())

        def primary(row: Quote) -> tuple[float, float]:
            if self.initial_selection_mode == "service":
                return row.predicted_service_s, row.predicted_objective_s
            if (
                self.initial_selection_mode == "mixed"
                and request._request_type != LATENCY
            ):
                return row.predicted_service_s, row.predicted_objective_s
            return row.predicted_objective_s, row.predicted_service_s

        def key(row: Quote) -> tuple[float, float, int, int, int]:
            first, second = primary(row)
            return (
                first,
                second,
                self._pressure(row.replica_id),
                -row.prefix_hit_tokens,
                row.replica_id,
            )

        choice = min(available, key=key)
        suffix = "feasible" if feasible else "no_feasible"
        request._cluster_route_reason = (
            f"ravel_native_token_{self.initial_selection_mode}_{context}_{suffix}"
        )
        return choice


class RavelNativeTokenObjectiveGlobalScheduler(
    _RavelNativeTokenAwareGlobalScheduler
):
    initial_selection_mode = "objective"


class RavelNativeTokenServiceGlobalScheduler(
    _RavelNativeTokenAwareGlobalScheduler
):
    initial_selection_mode = "service"


class RavelNativeTokenMixedGlobalScheduler(
    _RavelNativeTokenAwareGlobalScheduler
):
    initial_selection_mode = "mixed"


class _RavelNativeDecodeQuantumGlobalScheduler(
    _RavelNativeTokenAwareGlobalScheduler
):
    """Charge one or more engine-derived Decode scheduling quanta."""

    initial_selection_mode = "mixed"
    decode_quantum_multiplier = 1

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        block_size = int(args.block_size)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.decode_quantum_tokens = self.shared_state.num_replicas * block_size
        self.decode_horizon_tokens = (
            self.decode_quantum_multiplier
            * self.decode_quantum_tokens
        )


class RavelNativeDecodeQuantum1GlobalScheduler(
    _RavelNativeDecodeQuantumGlobalScheduler
):
    decode_quantum_multiplier = 1


class RavelNativeDecodeQuantum2GlobalScheduler(
    _RavelNativeDecodeQuantumGlobalScheduler
):
    decode_quantum_multiplier = 2


class RavelNativeAdaptiveQuantumGlobalScheduler(
    _RavelNativeDecodeQuantumGlobalScheduler
):
    """Use request-visible SLO semantics to price Decode occupancy."""

    def _decode_horizon(self, request: Request) -> int:
        multiplier = 1 if request._request_type == THROUGHPUT else 2
        return multiplier * self.decode_quantum_tokens


class RavelNativeLoadAdaptiveQuantumGlobalScheduler(
    _RavelNativeDecodeQuantumGlobalScheduler
):
    """Cap Decode pricing during bursts within one cross-region RTT."""

    def _burst_within_rtt(self, request: Request) -> bool:
        if len(self._recent_arrivals) < 2:
            return False
        latest_gap = self._recent_arrivals[-1][0] - self._recent_arrivals[-2][0]
        remote_rtts = [
            cluster.rtt_s(request._client_region)
            for cluster in self.topology.clusters
            if cluster.cluster_id != request._client_region
        ]
        # A second arrival within one remote decision RTT creates an observed,
        # not predicted, opportunity for joint placement.
        return bool(remote_rtts) and latest_gap <= max(remote_rtts)

    def _decode_horizon(self, request: Request) -> int:
        if request._request_type == COLLECTIVE or not self._burst_within_rtt(request):
            return 2 * self.decode_quantum_tokens
        return self.decode_quantum_tokens


class RavelNativeLoadAdaptiveRiskGlobalScheduler(
    RavelNativeLoadAdaptiveQuantumGlobalScheduler
):
    """Correct load-adaptive quotes with fresh replica TTFT residuals."""

    residual_decision_enabled = True
    risk_aware_region_selection = True


class RavelNativeMobileInsertionGlobalScheduler(
    RavelNativeLoadAdaptiveRiskGlobalScheduler
):
    """Keep a short, SLO-safe soft-placement window before KV materialization."""

    mobile_candidate_limit = 16
    mobile_beam_width = 256
    mobile_hold_rtt_fraction = 0.5
    mobile_min_slack_factor = 1.0
    plan_single_mobile_candidate = False

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self._mobile_timer: Optional[asyncio.Task] = None
        self._mobile_timer_deadline = math.inf
        self.direct_rebalance_count = 0
        hold_override = getattr(args, "ravel_mobile_hold_fraction", None)
        if hold_override is not None:
            self.mobile_hold_rtt_fraction = float(hold_override)
        slack_override = getattr(args, "ravel_mobile_slack_factor", None)
        if slack_override is not None:
            self.mobile_min_slack_factor = float(slack_override)

    def _mobile_hold_s(self, request: Request) -> float:
        if not self._burst_within_rtt(request):
            return 0.0
        remote_rtts = [
            cluster.rtt_s(request._client_region)
            for cluster in self.topology.clusters
            if cluster.cluster_id != request._client_region
            and cluster.rtt_s(request._client_region) > 0
        ]
        if not remote_rtts:
            return 0.0
        hold_s = self.mobile_hold_rtt_fraction * min(remote_rtts)
        quotes = [
            self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        ]
        feasible_clusters = {
            self.topology.cluster_for_replica(quote.replica_id).cluster_id
            for quote in quotes
            if quote.feasible
        }
        best_objective = min(quote.predicted_objective_s for quote in quotes)
        slack = self._limit(request) - best_objective
        if len(feasible_clusters) < 2:
            return 0.0
        if slack < self.mobile_min_slack_factor * hold_s:
            return 0.0
        return hold_s

    async def _enqueue(self, request: Request) -> None:
        quote = self._choose_initial(request)
        self._record_quote(request, quote)
        request._rebind_count = 0
        request._ravel_initial_replica = quote.replica_id
        request._ravel_soft_moves = 0
        hold_s = self._mobile_hold_s(request)
        request._ravel_mobile_hold_s = hold_s
        request._ravel_mobile_until = time.perf_counter() + hold_s
        self._placed_count[quote.replica_id] = (
            self._placed_count.get(quote.replica_id, 0) + 1
        )
        if request._request_type == LATENCY:
            cluster_id = self.topology.cluster_for_replica(
                quote.replica_id
            ).cluster_id
            self._latency_placed_count[cluster_id] = (
                self._latency_placed_count.get(cluster_id, 0) + 1
            )
        self.global_request_queue.push(
            quote.replica_id,
            request,
            quote.prefix_hit_tokens,
        )

    def _queued_owner(self, request: Request) -> int:
        return self._owner(request)

    def _mobile_candidates(self) -> list[Request]:
        rows = []
        for replica_id in self.topology.replica_ids:
            rows.extend(self.global_request_queue.get_all_requests(replica_id))
        rows.sort(
            key=lambda row: (
                row._arrived_at + self._limit(row),
                row._arrived_at,
                row._id,
            )
        )
        return rows[: self.mobile_candidate_limit]

    def _admission_overflow_cost(
        self, request: Request, overflow: int
    ) -> float:
        return float(overflow)

    def _movable_in_flight_count_by_replica(
        self, candidate_ids: set[int]
    ) -> dict[int, int]:
        """Movable submitted-but-unlocked requests per replica (default: none)."""
        return {}

    def _movable_in_flight_work_by_replica(
        self, candidate_ids: set[int]
    ) -> dict[int, float]:
        """Prompt-token work of movable in-flight candidates per replica."""
        return {}


    @staticmethod
    def _planner_quote(
        snapshot: PlannerSnapshot,
        request: PlannerRequestSnapshot,
        replica: PlannerReplicaSnapshot,
        replica_index: int,
        planned_work_tokens: int,
        planned_request_count: int,
    ) -> Quote:
        queue_work = (
            request.fixed_work_before_by_replica[replica_index]
            + planned_work_tokens
        )
        queue_count = (
            request.fixed_requests_before_by_replica[replica_index]
            + planned_request_count
        )
        service_s = (
            (
                replica.engine_work_tokens
                + queue_work
                + request.prompt_tokens
            )
            * replica.prefill_tpot_s
            + (
                replica.pending_requests
                + queue_count
                + 1
            )
            * replica.prefill_intercept_s
        )
        elapsed_s = max(0.0, snapshot.captured_at - request.arrived_at)
        rtt_s = request.rtt_by_replica_s[replica_index]
        point_ttft_s = elapsed_s + rtt_s + service_s
        point_objective_s = point_ttft_s
        risk_objective_s = point_ttft_s

        prospective_sequences = (
            replica.pending_requests
            + replica.running_requests
            + replica.fixed_queue_count
            + planned_request_count
            + 1
        )
        decode_curve = replica.decode_tpot_by_sequence_s
        decode_index = min(
            max(0, prospective_sequences),
            len(decode_curve) - 1,
        )
        decode_tpot_s = decode_curve[decode_index]
        if request.request_type != LATENCY:
            point_objective_s += (
                max(0, request.output_tokens_hint - 1)
                * decode_tpot_s
            )
            risk_objective_s += (
                max(0, request.output_tokens_upper_hint - 1)
                * decode_tpot_s
            )

        decision_guard_s = request.decision_guard_by_replica_s[replica_index]
        predicted_objective_s = risk_objective_s + decision_guard_s
        predicted_ttft_s = (
            point_ttft_s + decision_guard_s
            if request.request_type == LATENCY
            else point_ttft_s
        )
        feasible = predicted_objective_s <= request.limit_s
        if request.request_type == LATENCY:
            tbt_as_ttft_risk_s = (
                request.ttft_limit_s
                * decode_tpot_s
                / max(1e-9, request.tbt_limit_s)
            )
            predicted_objective_s = max(
                predicted_objective_s,
                tbt_as_ttft_risk_s,
            )
            feasible = (
                predicted_ttft_s <= request.ttft_limit_s
                and decode_tpot_s <= request.tbt_limit_s
            )

        return Quote(
            replica_id=replica.replica_id,
            predicted_ttft_s=predicted_ttft_s,
            point_ttft_s=point_ttft_s,
            predicted_objective_s=predicted_objective_s,
            point_objective_s=point_objective_s,
            predicted_service_s=service_s,
            base_ttft_s=(
                rtt_s
                + replica.prefill_intercept_s
                + request.prompt_tokens * replica.prefill_tpot_s
            ),
            prefix_hit_tokens=request.prefix_hit_by_replica[replica_index],
            prefix_benefit_s=(
                min(
                    max(0, request.prompt_tokens - 1),
                    request.prefix_hit_by_replica[replica_index],
                )
                * replica.prefill_tpot_s
            ),
            residual_guard_s=(
                request.residual_guard_by_replica_s[replica_index]
            ),
            risk_calibrated=request.risk_ready_by_replica[replica_index],
            feasible=feasible,
        )

    @classmethod
    def _plan_snapshot(
        cls,
        snapshot: PlannerSnapshot,
    ) -> Dict[int, tuple[int, Quote]]:
        """Minimize cohort SLO loss under a bounded convex load guard."""

        replica_count = len(snapshot.replicas)
        base_pressure = tuple(
            replica.pending_requests
            + replica.running_requests
            + replica.fixed_queue_count
            for replica in snapshot.replicas
        )
        base_pending = tuple(
            replica.pending_requests + replica.fixed_queue_count
            for replica in snapshot.replicas
        )
        zero_work = tuple(0 for _ in range(replica_count))
        zero_count = tuple(0 for _ in range(replica_count))
        zero_cost: Cost = (0, 0.0, 0.0, 0.0, 0.0, 0)
        beam = [(zero_cost, zero_work, zero_count, tuple())]

        for request in snapshot.requests:
            expanded = []
            limit_s = max(1e-9, request.limit_s)
            for cost, planned_work, planned_count, placements in beam:
                for index, replica in enumerate(snapshot.replicas):
                    moved = replica.replica_id != request.owner_replica
                    if moved and request.rebind_count >= 1:
                        continue
                    quote = cls._planner_quote(
                        snapshot,
                        request,
                        replica,
                        index,
                        planned_work[index],
                        planned_count[index],
                    )
                    pending_after = (
                        base_pending[index] + planned_count[index] + 1
                    )
                    overflow = (
                        max(
                            0,
                            pending_after - snapshot.pending_request_limit,
                        )
                        if request.enforce_admission_limit
                        else 0
                    )
                    lateness = max(
                        0.0,
                        quote.predicted_objective_s - limit_s,
                    ) / limit_s
                    pressure = base_pressure[index] + planned_count[index]
                    # This is exactly (p + 1)^2 - p^2, so the accumulated
                    # term minimizes the convex sum-of-squares load potential.
                    congestion = float(2 * pressure + 1)
                    move_cost = (
                        2
                        if moved and request.in_flight
                        else int(moved)
                    )
                    edge: Cost = (
                        int(not quote.feasible),
                        float(overflow),
                        lateness,
                        congestion,
                        quote.predicted_objective_s / limit_s,
                        move_cost,
                    )
                    next_work = list(planned_work)
                    next_work[index] += request.prompt_tokens
                    next_count = list(planned_count)
                    next_count[index] += 1
                    expanded.append(
                        (
                            tuple(
                                left + right
                                for left, right in zip(cost, edge)
                            ),
                            tuple(next_work),
                            tuple(next_count),
                            placements + ((replica.replica_id, quote),),
                        )
                    )

            if not expanded:
                raise RuntimeError(
                    f"planner has no legal placement for request "
                    f"{request.request_id}"
                )
            expanded.sort(
                key=lambda state: (
                    state[0],
                    max(
                        base + count
                        for base, count in zip(base_pressure, state[2])
                    ),
                    state[2],
                    state[1],
                    tuple(target for target, _quote in state[3]),
                )
            )
            deduplicated = {}
            for state in expanded:
                state_key = (state[1], state[2])
                if state_key not in deduplicated:
                    deduplicated[state_key] = state
                if len(deduplicated) >= snapshot.beam_width:
                    break
            beam = list(deduplicated.values())

        best = min(
            beam,
            key=lambda state: (
                state[0],
                max(
                    base + count
                    for base, count in zip(base_pressure, state[2])
                ),
                state[2],
                state[1],
                tuple(target for target, _quote in state[3]),
            ),
        )
        return {
            request.request_id: placement
            for request, placement in zip(snapshot.requests, best[3])
        }
    def _plan_mobile_assignment(
        self, candidates: list[Request], snapshot: Optional[PlannerSnapshot] = None
    ) -> Dict[int, tuple[int, Quote]]:
        if snapshot is not None:
            return self._plan_snapshot(snapshot)

        replica_ids = tuple(self.topology.replica_ids)
        candidate_ids = {row._id for row in candidates}
        base_pressure = []
        base_pending = []
        fixed_rows_by_replica: list[list[Request]] = []
        for replica_id in replica_ids:
            if snapshot is not None:
                fixed_rows = list(snapshot.fixed_rows.get(replica_id, []))
                fixed_rows_by_replica.append(fixed_rows)
                fixed_queued = len(fixed_rows)
                pending_net = int(snapshot.pending_net.get(replica_id, 0))
                running = int(snapshot.running.get(replica_id, 0))
                base_pending.append(pending_net + fixed_queued)
                base_pressure.append(pending_net + running + fixed_queued)
                continue
            replica = self.shared_state.replica_budgets[replica_id]
            fixed_rows = [
                row
                for row in self.global_request_queue.get_all_requests(replica_id)
                if row._id not in candidate_ids
            ]
            fixed_rows_by_replica.append(fixed_rows)
            fixed_queued = len(fixed_rows)
            pending = replica.get_num_pending_req()
            # Movable in-flight requests are already inside replica.pending;
            # they are re-counted exactly once via the candidate beam below.
            movable_in_flight = self._movable_in_flight_count_by_replica(
                candidate_ids
            ).get(replica_id, 0)
            pending_net = max(0, pending - movable_in_flight)
            # Fixed Router-queued requests consume future admission slots even
            # when they are not movable candidates.
            base_pending.append(pending_net + fixed_queued)
            base_pressure.append(
                pending_net
                + replica.get_num_running_req()
                + fixed_queued
            )
        base_pressure = tuple(base_pressure)
        base_pending = tuple(base_pending)

        zero_work = tuple(0 for _ in replica_ids)
        zero_count = tuple(0 for _ in replica_ids)
        zero_cost = (0, 0.0, 0.0, 0.0, 0.0, 0)
        beam = [(zero_cost, zero_work, zero_count, tuple())]

        for row in candidates:
            limit = max(1e-9, self._limit(row))
            if snapshot is not None:
                initial = int(snapshot.candidate_initial.get(int(row._id), -1))
                if initial < 0:
                    initial = int(snapshot.candidate_owner.get(int(row._id), -1))
            else:
                initial_value = getattr(row, "_ravel_initial_replica", None)
                initial = (
                    int(initial_value)
                    if initial_value is not None
                    else int(self._queued_owner(row))
                )
                if initial < 0:
                    initial = int(self._queued_owner(row))
            fixed_work_before = []
            fixed_requests_before = []
            row_key = self._edf_key(row)
            for index, replica_id in enumerate(replica_ids):
                work = 0
                count = 0
                for fixed in fixed_rows_by_replica[index]:
                    if self._edf_key(fixed) > row_key:
                        continue
                    if snapshot is not None:
                        fixed_work = float(
                            snapshot.fixed_work.get(
                                (replica_id, int(fixed._id)), 0.0
                            )
                        )
                    else:
                        fixed_work, _ = self._request_work(
                            fixed, replica_id, cold=False
                        )
                    work += fixed_work
                    count += 1
                fixed_work_before.append(work)
                fixed_requests_before.append(count)
            expanded = []
            for cost, planned_work, planned_count, placements in beam:
                for index, replica_id in enumerate(replica_ids):
                    quote = self._quote(
                        row,
                        replica_id,
                        work_before_tokens=(
                            fixed_work_before[index] + planned_work[index]
                        ),
                        requests_before=(
                            fixed_requests_before[index]
                            + planned_count[index]
                        ),
                        prospective_sequences=(
                            base_pressure[index] + planned_count[index] + 1
                        ),
                        cold=False,
                    )
                    lateness = max(
                        0.0,
                        quote.predicted_objective_s - limit,
                    ) / limit
                    current_pressure = (
                        base_pressure[index] + planned_count[index]
                    )
                    congestion = float(2 * current_pressure + 1)
                    pending_after = (
                        base_pending[index] + planned_count[index] + 1
                    )
                    admission_overflow = self._admission_overflow_cost(
                        row,
                        max(0, pending_after - self.pending_request_limit),
                    )
                    edge = (
                        int(not quote.feasible),
                        admission_overflow,
                        lateness,
                        congestion,
                        quote.predicted_objective_s / limit,
                        int(replica_id != initial),
                    )
                    new_cost = tuple(
                        left + right for left, right in zip(cost, edge)
                    )
                    request_work, _ = self._request_work(
                        row,
                        replica_id,
                        cold=False,
                    )
                    new_work = list(planned_work)
                    new_work[index] += request_work
                    new_count = list(planned_count)
                    new_count[index] += 1
                    expanded.append(
                        (
                            new_cost,
                            tuple(new_work),
                            tuple(new_count),
                            placements + ((replica_id, quote),),
                        )
                    )

            expanded.sort(
                key=lambda state: (
                    state[0],
                    max(
                        base + count
                        for base, count in zip(base_pressure, state[2])
                    ),
                    state[2],
                    state[1],
                    tuple(target for target, _ in state[3]),
                )
            )
            deduplicated = {}
            for state in expanded:
                state_key = (state[1], state[2])
                if state_key not in deduplicated:
                    deduplicated[state_key] = state
                if len(deduplicated) >= self.mobile_beam_width:
                    break
            beam = list(deduplicated.values())

        best = min(
            beam,
            key=lambda state: (
                state[0],
                max(
                    base + count
                    for base, count in zip(base_pressure, state[2])
                ),
                state[2],
                state[1],
                tuple(target for target, _ in state[3]),
            ),
        )
        return {
            row._id: placement
            for row, placement in zip(candidates, best[3])
        }

    async def _direct_reassign_mobile(self) -> int:
        candidates = self._mobile_candidates()
        if not candidates or (
            len(candidates) < 2
            and not self.plan_single_mobile_candidate
        ):
            return 0
        owners = {row._id: self._queued_owner(row) for row in candidates}
        assignment = self._plan_mobile_assignment(candidates)
        movers = [
            row
            for row in candidates
            if assignment[row._id][0] != owners[row._id]
        ]
        for row in movers:
            if not self.global_request_queue.del_req(owners[row._id], row):
                raise RuntimeError(
                    f"RAVEL failed to remove mobile request {row._id}"
                )
        for row in movers:
            target, _ = assignment[row._id]
            _, hit = self._request_work(row, target, cold=False)
            self.global_request_queue.push(target, row, hit)
            row._ravel_soft_moves = (
                int(getattr(row, "_ravel_soft_moves", 0)) + 1
            )
        for row in candidates:
            _, quote = assignment[row._id]
            self._record_quote(row, quote)
            if row in movers:
                row._cluster_route_reason = "ravel_mobile_soft_replan"
        self.direct_rebalance_count += len(movers)
        return len(movers)

    async def _wake_mobile_timer(self, deadline: float) -> None:
        try:
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        except asyncio.CancelledError:
            return
        self._mobile_timer = None
        self._mobile_timer_deadline = math.inf
        await self.schedule(None)

    def _arm_mobile_timer(self, deadline: float) -> None:
        if not math.isfinite(deadline):
            return
        if self._mobile_timer is not None and not self._mobile_timer.done():
            if self._mobile_timer_deadline <= deadline + 1e-6:
                return
            self._mobile_timer.cancel()
        self._mobile_timer_deadline = deadline
        self._mobile_timer = asyncio.create_task(
            self._wake_mobile_timer(deadline)
        )

    def _can_admit_to_engine(
        self, request: Request, replica_id: int
    ) -> bool:
        """Return whether the request may cross the reversible frontier."""
        return True

    async def _dispatch_schedulable(self) -> None:
        now = time.perf_counter()
        next_deadline = math.inf
        for replica_id in self.topology.replica_ids:
            if self.global_request_queue.is_empty(replica_id):
                continue
            replica = self.shared_state.replica_budgets[replica_id]
            current_budget, _, _, pending_requests, _ = (
                await replica.get_load_states()
            )
            if current_budget <= 0 or pending_requests >= self.pending_request_limit:
                continue
            max_pop = max(0, self.pending_request_limit - pending_requests)
            eligible = []
            for row in self.global_request_queue.get_all_requests(replica_id):
                deadline = float(
                    getattr(row, "_ravel_mobile_until", row._arrived_at)
                )
                if deadline <= now:
                    eligible.append(row)
                else:
                    next_deadline = min(next_deadline, deadline)
            eligible.sort(
                key=lambda row: (
                    row._arrived_at + self._limit(row),
                    row._arrived_at,
                    row._id,
                )
            )
            dispatched = 0
            for row in eligible:
                if dispatched >= max_pop:
                    break
                request_work, hit = self._request_work(
                    row,
                    replica_id,
                    cold=False,
                )
                if not self._can_admit_to_engine(row, replica_id):
                    continue
                if request_work > current_budget:
                    continue
                if not self.global_request_queue.del_req(replica_id, row):
                    continue
                initial = int(
                    getattr(row, "_ravel_initial_replica", replica_id)
                )
                row._rebind_count = int(replica_id != initial)
                if row._rebind_count:
                    row._cluster_route_reason = "ravel_mobile_cohort_rebind"
                elif int(getattr(row, "_ravel_soft_moves", 0)) > 0:
                    row._cluster_route_reason = "ravel_mobile_cohort_restored"
                row._enforce_prefill_budget = True
                admitted = await self.shared_state.add_posting_request_tasks(
                    replica_id,
                    row,
                )
                if not admitted:
                    self.global_request_queue.push(replica_id, row, hit)
                    continue
                current_budget -= request_work
                dispatched += 1
        self._arm_mobile_timer(next_deadline)

    async def schedule(self, new_request: Optional[Request]) -> int:
        async with self._schedule_lock:
            if new_request is not None:
                await self._enqueue(new_request)
            await self._direct_reassign_mobile()
            await self._dispatch_schedulable()
            return -1


class RavelNativeMobileNaturalGlobalScheduler(
    RavelNativeMobileInsertionGlobalScheduler
):
    mobile_hold_rtt_fraction = 0.0


class RavelNativeMobileHold10GlobalScheduler(
    RavelNativeMobileInsertionGlobalScheduler
):
    mobile_hold_rtt_fraction = 0.1


class RavelNativeMobileHold25GlobalScheduler(
    RavelNativeMobileInsertionGlobalScheduler
):
    mobile_hold_rtt_fraction = 0.25


class RavelNativeMobileSLOAwareGlobalScheduler(
    RavelNativeMobileHold25GlobalScheduler
):
    def _mobile_hold_s(self, request: Request) -> float:
        if request._request_type == LATENCY:
            return 0.0
        return super()._mobile_hold_s(request)

    def _admission_overflow_cost(
        self, request: Request, overflow: int
    ) -> float:
        if request._request_type == LATENCY:
            return float(overflow)
        mobile_until = float(
            getattr(request, "_ravel_mobile_until", request._arrived_at)
        )
        return (
            0.0
            if time.perf_counter() < mobile_until
            else float(overflow)
        )


class RavelNativeMobileTriggeredGlobalScheduler(
    RavelNativeMobileSLOAwareGlobalScheduler
):

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self.mobile_candidate_limit = int(
            getattr(args, "ravel_mobile_candidate_limit", self.mobile_candidate_limit)
        )
        self.mobile_beam_width = int(
            getattr(args, "ravel_mobile_beam_width", self.mobile_beam_width)
        )
        if self.mobile_candidate_limit <= 0 or self.mobile_beam_width <= 0:
            raise ValueError("mobile candidate limit and beam width must be positive")

    def _choose_initial(self, request: Request) -> Quote:
        """Solve the per-arrival risk-constrained placement problem.

        First filter replicas by the conservative SLO bound, then minimize
        expected completion time within that feasible set. If no replica is
        feasible, minimize predicted violation before expected cost. Prefix
        locality is only a final deterministic tie-breaker.
        """
        quotes = {
            replica_id: self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        }
        self._observe_arrival(request)
        calibrated = [
            row for row in quotes.values()
            if row.feasible and row.risk_calibrated
        ]
        feasible = [row for row in quotes.values() if row.feasible]
        if calibrated:
            candidates = calibrated
            reason = "risk_feasible_min_objective"
        elif feasible:
            candidates = feasible
            reason = "uncalibrated_feasible_min_objective"
        else:
            candidates = list(quotes.values())
            reason = "no_feasible_min_violation"
        choice = min(
            candidates,
            key=lambda row: (
                max(0.0, row.predicted_objective_s - self._limit(request)),
                row.predicted_objective_s,
                row.point_objective_s,
                -row.prefix_hit_tokens,
                row.replica_id,
            ),
        )
        request._cluster_route_reason = f"ravel_mobile_{reason}"
        return choice

    def _mobile_hold_s(self, request: Request) -> float:
        if request._request_type == LATENCY or not self._burst_within_rtt(request):
            return 0.0
        remote_rtts = [
            cluster.rtt_s(request._client_region)
            for cluster in self.topology.clusters
            if cluster.cluster_id != request._client_region
            and cluster.rtt_s(request._client_region) > 0
        ]
        if not remote_rtts:
            return 0.0
        quotes = [self._quote(request, rid) for rid in self.topology.replica_ids]
        feasible_clusters = {
            self.topology.cluster_for_replica(quote.replica_id).cluster_id
            for quote in quotes
            if quote.feasible
        }
        if len(feasible_clusters) < 2:
            return 0.0
        slack = max(0.0, self._limit(request) - min(q.predicted_objective_s for q in quotes))
        hold_s = min(remote_rtts)
        # The frozen MobileTriggered baseline exposes one observed remote RTT;
        # Hold10/Hold25 apply only to their explicit scheduler variants.
        return min(hold_s, slack)

    def _latency_mobile_allowed(self, request: Request) -> bool:
        quotes = [self._quote(request, rid) for rid in self.topology.replica_ids]
        return any(
            quote.feasible
            and self.topology.cluster_for_replica(quote.replica_id).cluster_id
            != request._client_region
            for quote in quotes
        )

    def _mobile_candidates(self) -> list[Request]:
        rows = []
        for replica_id in self.topology.replica_ids:
            rows.extend(self.global_request_queue.get_all_requests(replica_id))
        rows = [
            row
            for row in rows
            if row._request_type != LATENCY or self._latency_mobile_allowed(row)
        ]
        rows.sort(
            key=lambda row: (
                row._arrived_at + self._limit(row),
                row._arrived_at,
                row._id,
            )
        )
        return rows[: self.mobile_candidate_limit]

    async def _wake_mobile_timer(self, deadline: float) -> None:
        try:
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        except asyncio.CancelledError:
            return
        self._mobile_timer = None
        self._mobile_timer_deadline = math.inf
        async with self._schedule_lock:
            await self._direct_reassign_mobile()
            await self._dispatch_schedulable()

    async def schedule(self, new_request: Optional[Request]) -> int:
        async with self._schedule_lock:
            should_replan = new_request is None
            if new_request is not None:
                await self._enqueue(new_request)
                should_replan = (
                    new_request._request_type == LATENCY
                    or float(
                        getattr(
                            new_request, "_ravel_mobile_hold_s", 0.0
                        )
                    ) <= 0
                )
            if should_replan:
                await self._direct_reassign_mobile()
            await self._dispatch_schedulable()
            return -1


class RavelNativeMobileSLOCompleteGlobalScheduler(
    RavelNativeMobileTriggeredGlobalScheduler
):
    """Apply both TTFT and TBT feasibility for latency-class requests."""

    def _quote(
        self,
        request: Request,
        replica_id: int,
        *,
        work_before_tokens: Optional[int] = None,
        requests_before: Optional[int] = None,
        prospective_sequences: Optional[int] = None,
        cold: bool = False,
    ) -> Quote:
        quote = super()._quote(
            request,
            replica_id,
            work_before_tokens=work_before_tokens,
            requests_before=requests_before,
            prospective_sequences=prospective_sequences,
            cold=cold,
        )
        if request._request_type != LATENCY:
            return quote

        cluster = self.topology.cluster_for_replica(replica_id)
        active_sequences = (
            max(1, int(prospective_sequences))
            if prospective_sequences is not None
            else self._prospective_sequence_count(request, replica_id)
        )
        predicted_tbt_s = cluster.decode_tpot_for(active_sequences)
        ttft_limit_s = max(1e-9, float(request._slo_constraint[0]))
        tbt_limit_s = max(1e-9, float(request._slo_constraint[1]))
        tbt_as_ttft_risk_s = ttft_limit_s * predicted_tbt_s / tbt_limit_s
        return replace(
            quote,
            predicted_objective_s=max(
                quote.predicted_objective_s,
                tbt_as_ttft_risk_s,
            ),
            feasible=(
                quote.predicted_ttft_s <= ttft_limit_s
                and predicted_tbt_s <= tbt_limit_s
            ),
        )


class RavelNativeCenterYieldGlobalScheduler(
    RavelNativeMobileSLOCompleteGlobalScheduler
):
    """Keep excess work movable behind a profiled engine frontier.

    The center queue is the reversible tier. Each engine receives at most one
    complete Chunked-Prefill quantum plus the prompt work it can drain during
    one worst-case inter-region control RTT. This keeps the engine busy while
    preventing a deep, irrevocable HTTP queue from hiding future SLO-yield
    decisions. No workload identity or realized output length is used.
    """


    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self.control_horizon_s = max(
            destination.rtt_s(source.cluster_id)
            for source in self.topology.clusters
            for destination in self.topology.clusters
            if source.cluster_id != destination.cluster_id
        )
        quantum_tokens = int(getattr(args, "max_num_batched_tokens", 0))
        if quantum_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        self._engine_frontier_tokens: Dict[int, int] = {}
        for replica_id in self.topology.replica_ids:
            cluster = self.topology.cluster_for_replica(replica_id)
            fastest_prefill_tpot_s = float(cluster.prefill_tpot_for(0))
            if fastest_prefill_tpot_s <= 0:
                raise ValueError("prefill service rate must be positive")
            refill_cover_tokens = math.ceil(
                self.control_horizon_s / fastest_prefill_tpot_s
            )
            self._engine_frontier_tokens[replica_id] = (
                quantum_tokens + refill_cover_tokens
            )

    @staticmethod
    def _yield_dispatch_key(request: Request) -> tuple[float, float, int]:
        return (
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )

    def _set_engine_priority(self, request: Request) -> None:
        request._vllm_priority = max(
            0,
            int((request._arrived_at + self._limit(request)) * 1_000_000),
        )

    def _refresh_yield_state(self, request: Request) -> None:
        request._ravel_yield_deferred = not bool(
            getattr(request, "_cluster_route_feasible", False)
        )
        self._set_engine_priority(request)

    async def _enqueue(self, request: Request) -> None:
        await super()._enqueue(request)
        self._refresh_yield_state(request)

    async def _direct_reassign_mobile(self) -> int:
        moved = await super()._direct_reassign_mobile()
        for replica_id in self.topology.replica_ids:
            for request in self.global_request_queue.get_all_requests(
                replica_id
            ):
                self._refresh_yield_state(request)
        return moved

    async def _dispatch_schedulable(self) -> None:
        now = time.perf_counter()
        next_deadline = math.inf
        for replica_id in self.topology.replica_ids:
            if self.global_request_queue.is_empty(replica_id):
                continue
            replica = self.shared_state.replica_budgets[replica_id]
            current_budget, _, _, _pending_requests, _ = (
                await replica.get_load_states()
            )
            frontier_headroom = max(
                0,
                self._engine_frontier_tokens[replica_id]
                - self._engine_work(replica_id),
            )
            current_budget = min(current_budget, frontier_headroom)
            if current_budget <= 0:
                continue

            eligible = []
            for request in self.global_request_queue.get_all_requests(
                replica_id
            ):
                deadline = float(
                    getattr(
                        request,
                        "_ravel_mobile_until",
                        request._arrived_at,
                    )
                )
                if deadline <= now:
                    eligible.append(request)
                else:
                    next_deadline = min(next_deadline, deadline)
            eligible.sort(key=self._yield_dispatch_key)

            for request in eligible:
                request_work, hit = self._request_work(
                    request,
                    replica_id,
                    cold=False,
                )
                if not self._can_admit_to_engine(request, replica_id):
                    continue
                if request_work > current_budget:
                    continue
                if not self.global_request_queue.del_req(
                    replica_id, request
                ):
                    continue
                initial = int(
                    getattr(request, "_ravel_initial_replica", replica_id)
                )
                request._rebind_count = int(replica_id != initial)
                if request._rebind_count:
                    request._cluster_route_reason = "ravel_center_yield_rebind"
                elif int(getattr(request, "_ravel_soft_moves", 0)) > 0:
                    request._cluster_route_reason = "ravel_center_yield_restored"
                self._set_engine_priority(request)
                request._enforce_prefill_budget = True
                admitted = await self.shared_state.add_posting_request_tasks(
                    replica_id,
                    request,
                )
                if not admitted:
                    self.global_request_queue.push(replica_id, request, hit)
                    continue
                current_budget -= request_work
        self._arm_mobile_timer(next_deadline)


class RavelNativeCenterWorkYieldGlobalScheduler(
    RavelNativeCenterYieldGlobalScheduler
):
    """Fast center placement using profiled service work, not request count.

    Candidates are considered in absolute-deadline order. For each request,
    the planner first preserves SLO feasibility, then minimizes normalized
    lateness and predicted SLO consumption. A convex marginal-work term breaks
    remaining ties, balancing heterogeneous prompt/decode service without a
    fitted score weight. The planner is linear in the current cohort and
    replica count, so arrivals cannot self-throttle behind a beam search.
    """

    _DEFERRED_PRIORITY_OFFSET_US = 1_000_000_000_000_000
    marginal_work_first = False
    # The released SLO-first policy balances virtual service debt before
    # comparing latency among otherwise feasible placements. Experimental
    # latency-balanced policies reverse only those secondary terms; the
    # feasibility, overflow, and normalized-lateness ordering remains intact.
    flow_time_before_virtual_balance = False
    # Unified enables this for completion-only traffic so heterogeneous
    # service rates are tracked across successive mobile planning cohorts.
    completion_virtual_service_enabled = False

    @staticmethod
    def _within_one_job_guard(
        projected_finish: float,
        minimum_finish: float,
        own_service_s: float,
    ) -> bool:
        """Classic list-scheduling bound for one online admission."""

        return projected_finish <= minimum_finish + own_service_s

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self._latency_class_observed = False
        # Sum of admitted LATENCY requests' normalized one-token demand,
        # d_r(1) / TBT_i. This virtual admission ledger is independent of
        # lifecycle callback timing and does not estimate engine backlog.
        self._latency_virtual_load = {
            replica_id: 0.0
            for replica_id in self.topology.replica_ids
        }
        # Greedy list-scheduling load for mixed SLO traffic. Each admitted
        # request contributes its profiled own service time. Feasibility and
        # normalized lateness remain earlier lexicographic constraints.
        self._service_virtual_load = {
            replica_id: 0.0
            for replica_id in self.topology.replica_ids
        }

    def _set_engine_priority(self, request: Request) -> None:
        # Collective-only workloads have E2E deadlines. Reordering their decode
        # streams inside vLLM is not justified by a TTFT objective.
        if not self._latency_class_observed:
            request._vllm_priority = 0
            return
        deadline_us = max(
            0,
            int((request._arrived_at + self._limit(request)) * 1_000_000),
        )
        deferred = bool(
            getattr(request, "_ravel_yield_deferred", False)
        )
        request._vllm_priority = deadline_us + (
            self._DEFERRED_PRIORITY_OFFSET_US if deferred else 0
        )

    def _refresh_yield_state(self, request: Request) -> None:
        # A point-model miss is not strong enough evidence to defer work during
        # calibration warm-up.
        request._ravel_yield_deferred = bool(
            getattr(request, "_risk_calibrated", False)
            and not bool(
                getattr(request, "_cluster_route_feasible", False)
            )
        )
        self._set_engine_priority(request)

    @staticmethod
    def _yield_dispatch_key(
        request: Request,
    ) -> tuple[int, float, float, int]:
        return (
            int(
                bool(
                    getattr(
                        request, "_ravel_yield_deferred", False
                    )
                )
            ),
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )

    async def _enqueue(self, request: Request) -> None:
        if request._request_type == LATENCY:
            self._latency_class_observed = True
        await super()._enqueue(request)
        owner = int(self._queued_owner(request))
        selected_quote = getattr(
            request, "_ravel_virtual_selected_quote", None
        )
        self._register_service_virtual_admission(
            request, owner_hint=owner, quote=selected_quote
        )
        self._register_latency_virtual_admission(
            request, owner_hint=owner
        )
        if hasattr(request, "_ravel_virtual_selected_quote"):
            del request._ravel_virtual_selected_quote

    def _own_profiled_service_s(
        self,
        request: Request,
        replica_id: int,
        quote: Quote,
    ) -> float:
        cluster = self.topology.cluster_for_replica(replica_id)
        own_prefill_s = max(
            0.0,
            quote.base_ttft_s
            - cluster.rtt_s(request._client_region),
        )
        own_decode_s = max(
            0.0,
            quote.point_objective_s - quote.point_ttft_s,
        )
        return max(1e-9, own_prefill_s + own_decode_s)

    def _planning_key(
        self, request: Request
    ) -> tuple[float, float, int]:
        return self._edf_key(request)

    @staticmethod
    def _latency_virtual_weight(request: object) -> float:
        if getattr(request, "_request_type", None) != LATENCY:
            return 0.0
        slo = getattr(request, "_slo_constraint", None)
        if not slo or len(slo) < 2:
            return 0.0
        return 1.0 / max(1e-9, float(slo[1]))

    def _latency_virtual_charge(
        self, request: object, replica_id: int
    ) -> float:
        weight = self._latency_virtual_weight(request)
        if weight <= 0.0:
            return 0.0
        cluster = self.topology.cluster_for_replica(replica_id)
        return cluster.decode_tpot_for(1) * weight

    def _register_latency_virtual_admission(
        self,
        request: object,
        owner_hint: Optional[int] = None,
    ) -> None:
        if self._latency_virtual_weight(request) <= 0.0:
            return
        if hasattr(request, "_ravel_latency_virtual_owner"):
            return
        owner = (
            int(owner_hint)
            if owner_hint is not None
            else int(self._queued_owner(request))
        )
        charge = self._latency_virtual_charge(request, owner)
        self._latency_virtual_load[owner] += charge
        request._ravel_latency_virtual_owner = owner
        request._ravel_latency_virtual_charge = charge

    def _service_virtual_charge(
        self,
        request: Request,
        replica_id: int,
        quote: Quote,
    ) -> float:
        return self._own_profiled_service_s(
            request, replica_id, quote
        )

    @staticmethod
    def _normalized_ttft_lateness(
        request: Request, quote: Quote
    ) -> float:
        ttft_slo = max(
            1e-9, float(request._slo_constraint[0])
        )
        return max(
            0.0, quote.predicted_ttft_s - ttft_slo
        ) / ttft_slo

    def _register_service_virtual_admission(
        self,
        request: Request,
        owner_hint: Optional[int] = None,
        quote: Optional[Quote] = None,
    ) -> None:
        if not (
            self._latency_class_observed
            or self.completion_virtual_service_enabled
        ):
            return
        if hasattr(request, "_ravel_service_virtual_owner"):
            return
        owner = (
            int(owner_hint)
            if owner_hint is not None
            else int(self._queued_owner(request))
        )
        selected_quote = (
            quote if quote is not None else self._quote(request, owner)
        )
        charge = self._service_virtual_charge(
            request, owner, selected_quote
        )
        self._service_virtual_load[owner] += charge
        request._ravel_service_virtual_owner = owner
        request._ravel_service_virtual_charge = charge

    def _choose_initial(self, request: Request) -> Quote:
        baseline = super()._choose_initial(request)
        if not self._latency_class_observed:
            return baseline

        limit_s = max(1e-9, self._limit(request))
        quotes = {
            replica_id: (
                baseline
                if replica_id == baseline.replica_id
                else self._quote(request, replica_id)
            )
            for replica_id in self.topology.replica_ids
        }
        projected_service_finish = {
            replica_id: (
                self._service_virtual_load[replica_id]
                + self._service_virtual_charge(
                    request, replica_id, quote
                )
            )
            for replica_id, quote in quotes.items()
        }
        minimum_service_finish = min(
            projected_service_finish.values()
        )

        def key(quote: Quote) -> tuple:
            replica_id = quote.replica_id
            replica = self.shared_state.replica_budgets[replica_id]
            pending_after = (
                replica.get_num_pending_req()
                + self.global_request_queue.get_queue_len(replica_id)
                + 1
            )
            overflow = self._admission_overflow_cost(
                request,
                max(0, pending_after - self.pending_request_limit),
            )
            lateness = max(
                0.0, quote.predicted_objective_s - limit_s
            ) / limit_s
            ttft_lateness = self._normalized_ttft_lateness(
                request, quote
            )
            virtual_finish = (
                self._latency_virtual_load[replica_id]
                + self._latency_virtual_charge(request, replica_id)
                if request._request_type == LATENCY
                else 0.0
            )
            service_finish = (
                projected_service_finish[replica_id]
            )
            primary = (
                int(not quote.feasible),
                overflow,
                lateness,
                ttft_lateness,
            )
            if self.flow_time_before_virtual_balance:
                own_service_s = self._service_virtual_charge(
                    request, replica_id, quote
                )
                secondary = (
                    int(
                        not self._within_one_job_guard(
                            service_finish,
                            minimum_service_finish,
                            own_service_s,
                        )
                    ),
                    quote.predicted_ttft_s
                    / max(
                        1e-9,
                        float(request._slo_constraint[0]),
                    ),
                    quote.predicted_objective_s / limit_s,
                    virtual_finish,
                    service_finish,
                )
            else:
                secondary = (
                    virtual_finish,
                    service_finish,
                    quote.predicted_objective_s / limit_s,
                )
            return (
                *primary,
                *secondary,
                self._pressure(replica_id),
                -quote.prefix_hit_tokens,
                replica_id,
            )

        chosen = min(quotes.values(), key=key)
        request._cluster_route_reason = (
            "ravel_work_yield_mixed_virtual_admission"
        )
        request._ravel_virtual_selected_quote = chosen
        return chosen

    def _plan_mobile_assignment(
        self,
        candidates: list[Request],
        snapshot: Optional[PlannerSnapshot] = None,
    ) -> Dict[int, tuple[int, Quote]]:
        if snapshot is not None:
            return super()._plan_mobile_assignment(
                candidates, snapshot
            )

        replica_ids = tuple(self.topology.replica_ids)
        candidate_ids = {row._id for row in candidates}
        for replica_id in replica_ids:
            replica = self.shared_state.replica_budgets[replica_id]
            known = (
                *self.global_request_queue.get_all_requests(replica_id),
                *replica.pending_requests,
                *replica.running_requests,
            )
            for row in known:
                self._register_latency_virtual_admission(
                    row, owner_hint=replica_id
                )
                self._register_service_virtual_admission(
                    row, owner_hint=replica_id
                )
        fixed_entries: Dict[
            int, list[tuple[tuple[float, float, int], int]]
        ] = {}
        fixed_cursor = {replica_id: 0 for replica_id in replica_ids}
        fixed_prefix_work = {
            replica_id: 0 for replica_id in replica_ids
        }
        fixed_prefix_count = {
            replica_id: 0 for replica_id in replica_ids
        }
        base_pressure: Dict[int, int] = {}
        base_virtual_load = dict(self._latency_virtual_load)
        base_virtual_service_load = dict(
            self._service_virtual_load
        )
        base_pending: Dict[int, int] = {}
        base_service_s: Dict[int, float] = {}
        planned_work = {
            replica_id: 0 for replica_id in replica_ids
        }
        planned_count = {
            replica_id: 0 for replica_id in replica_ids
        }
        planned_virtual_load = {
            replica_id: 0.0 for replica_id in replica_ids
        }
        planned_virtual_service_load = {
            replica_id: 0.0 for replica_id in replica_ids
        }
        planned_service_s = {
            replica_id: 0.0 for replica_id in replica_ids
        }

        for replica_id in replica_ids:
            replica = self.shared_state.replica_budgets[replica_id]
            rows = [
                row
                for row in self.global_request_queue.get_all_requests(
                    replica_id
                )
                if row._id not in candidate_ids
            ]
            entries = sorted(
                (
                    self._edf_key(row),
                    self._request_work(
                        row, replica_id, cold=False
                    )[0],
                )
                for row in rows
            )
            fixed_entries[replica_id] = entries
            fixed_work = sum(work for _key, work in entries)
            pending = replica.get_num_pending_req()
            running = replica.get_num_running_req()
            base_pending[replica_id] = pending + len(rows)
            base_pressure[replica_id] = (
                pending + running + len(rows)
            )
            cluster = self.topology.cluster_for_replica(
                replica_id
            )
            prefill_tpot = cluster.prefill_tpot_for(running)
            prefill_intercept = cluster.prefill_intercept_for(
                running
            )
            base_service_s[replica_id] = (
                (
                    self._engine_work(replica_id)
                    + fixed_work
                )
                * prefill_tpot
                + base_pressure[replica_id]
                * prefill_intercept
            )

        # Current mobile candidates are replanned, so remove their old virtual
        # charges from the fixed prefix and add exactly one chosen charge below.
        for request in candidates:
            owner = getattr(request, "_ravel_latency_virtual_owner", None)
            if owner is not None:
                owner = int(owner)
                old_charge = float(
                    getattr(
                        request,
                        "_ravel_latency_virtual_charge",
                        0.0,
                    )
                )
                base_virtual_load[owner] = max(
                    0.0, base_virtual_load[owner] - old_charge
                )
            service_owner = getattr(
                request, "_ravel_service_virtual_owner", None
            )
            if service_owner is None:
                continue
            service_owner = int(service_owner)
            old_service_charge = float(
                getattr(
                    request,
                    "_ravel_service_virtual_charge",
                    0.0,
                )
            )
            base_virtual_service_load[service_owner] = max(
                0.0,
                base_virtual_service_load[service_owner]
                - old_service_charge,
            )

        assignment: Dict[int, tuple[int, Quote]] = {}
        for request in sorted(candidates, key=self._planning_key):
            limit_s = max(1e-9, self._limit(request))
            initial_value = getattr(
                request, "_ravel_initial_replica", None
            )
            initial = (
                int(initial_value)
                if initial_value is not None
                else int(self._queued_owner(request))
            )
            request_key = self._edf_key(request)
            edges = []
            for replica_id in replica_ids:
                moved = replica_id != initial
                if (
                    moved
                    and int(
                        getattr(request, "_rebind_count", 0)
                    )
                    >= 1
                ):
                    continue
                entries = fixed_entries[replica_id]
                cursor = fixed_cursor[replica_id]
                while (
                    cursor < len(entries)
                    and entries[cursor][0] <= request_key
                ):
                    fixed_prefix_work[replica_id] += (
                        entries[cursor][1]
                    )
                    fixed_prefix_count[replica_id] += 1
                    cursor += 1
                fixed_cursor[replica_id] = cursor

                quote = self._quote(
                    request,
                    replica_id,
                    work_before_tokens=(
                        fixed_prefix_work[replica_id]
                        + planned_work[replica_id]
                    ),
                    requests_before=(
                        fixed_prefix_count[replica_id]
                        + planned_count[replica_id]
                    ),
                    prospective_sequences=(
                        base_pressure[replica_id]
                        + planned_count[replica_id]
                        + 1
                    ),
                    cold=False,
                )
                pending_after = (
                    base_pending[replica_id]
                    + planned_count[replica_id]
                    + 1
                )
                overflow = self._admission_overflow_cost(
                    request,
                    max(
                        0,
                        pending_after
                        - self.pending_request_limit,
                    ),
                )
                lateness = max(
                    0.0,
                    quote.predicted_objective_s - limit_s,
                ) / limit_s
                ttft_lateness = self._normalized_ttft_lateness(
                    request, quote
                )
                own_service_s = self._own_profiled_service_s(
                    request, replica_id, quote
                )
                current_service_s = (
                    base_service_s[replica_id]
                    + planned_service_s[replica_id]
                )
                marginal_work = (
                    (current_service_s + own_service_s) ** 2
                    - current_service_s**2
                ) / (limit_s**2)
                virtual_charge = self._latency_virtual_charge(
                    request, replica_id
                )
                # For LATENCY traffic, virtual load is cumulative normalized
                # one-token utilization sum(d_r(1) / TBT_i). It is compared
                # only after own feasibility and lateness, so balancing cannot
                # turn a feasible request into an avoidable SLO miss. Flexible
                # traffic pays no class-isolation term; its measured effect is
                # already represented by the quote and marginal-work terms.
                latency_virtual_finish = (
                    base_virtual_load[replica_id]
                    + planned_virtual_load[replica_id]
                    + virtual_charge
                    if request._request_type == LATENCY
                    else 0.0
                )
                service_virtual_charge = (
                    self._service_virtual_charge(
                        request, replica_id, quote
                    )
                    if self._latency_class_observed
                    else 0.0
                )
                service_virtual_finish = (
                    base_virtual_service_load[replica_id]
                    + planned_virtual_service_load[replica_id]
                    + service_virtual_charge
                )
                if self.marginal_work_first:
                    secondary_terms = (
                        marginal_work,
                        quote.predicted_objective_s / limit_s,
                    )
                else:
                    secondary_terms = (
                        quote.predicted_objective_s / limit_s,
                        marginal_work,
                    )
                if self.flow_time_before_virtual_balance:
                    decision_terms = (
                        quote.predicted_ttft_s
                        / max(
                            1e-9,
                            float(request._slo_constraint[0]),
                        ),
                        quote.predicted_objective_s / limit_s,
                        latency_virtual_finish,
                        service_virtual_finish,
                        marginal_work,
                    )
                else:
                    decision_terms = (
                        latency_virtual_finish,
                        service_virtual_finish,
                        *secondary_terms,
                    )
                edges.append(
                    (
                        int(not quote.feasible),
                        overflow,
                        lateness,
                        ttft_lateness,
                        *decision_terms,
                        int(moved),
                        replica_id,
                        quote,
                        own_service_s,
                        virtual_charge,
                        service_virtual_charge,
                    )
                )
            if not edges:
                raise RuntimeError(
                    "RAVEL has no legal placement for request "
                    f"{request._id}"
                )
            eligible_edges = edges
            if self.flow_time_before_virtual_balance:
                best_primary = min(edge[:4] for edge in edges)
                eligible_edges = [
                    edge for edge in edges
                    if edge[:4] == best_primary
                ]

                def projected_finish(edge: tuple) -> float:
                    target = int(edge[-5])
                    return (
                        base_virtual_service_load[target]
                        + planned_virtual_service_load[target]
                        + float(edge[-1])
                    )

                minimum_finish = min(
                    projected_finish(edge)
                    for edge in eligible_edges
                )
                guarded_edges = [
                    edge
                    for edge in eligible_edges
                    if self._within_one_job_guard(
                        projected_finish(edge),
                        minimum_finish,
                        float(edge[-3]),
                    )
                ]
                if guarded_edges:
                    eligible_edges = guarded_edges
            chosen = min(
                eligible_edges, key=lambda edge: edge[:-4]
            )
            replica_id = int(chosen[-5])
            quote = chosen[-4]
            own_service_s = float(chosen[-3])
            virtual_charge = float(chosen[-2])
            service_virtual_charge = float(chosen[-1])
            request_work, _ = self._request_work(
                request, replica_id, cold=False
            )
            planned_work[replica_id] += request_work
            planned_count[replica_id] += 1
            planned_service_s[replica_id] += own_service_s
            planned_virtual_load[replica_id] += virtual_charge
            planned_virtual_service_load[
                replica_id
            ] += service_virtual_charge
            assignment[request._id] = (replica_id, quote)

        for request in candidates:
            old_owner = getattr(
                request, "_ravel_latency_virtual_owner", None
            )
            if old_owner is None:
                continue
            old_owner = int(old_owner)
            old_charge = float(
                getattr(request, "_ravel_latency_virtual_charge", 0.0)
            )
            new_owner = int(assignment[request._id][0])
            new_charge = self._latency_virtual_charge(
                request, new_owner
            )
            self._latency_virtual_load[old_owner] = max(
                0.0,
                self._latency_virtual_load[old_owner] - old_charge,
            )
            self._latency_virtual_load[new_owner] += new_charge
            request._ravel_latency_virtual_owner = new_owner
            request._ravel_latency_virtual_charge = new_charge
        for request in candidates:
            old_owner = getattr(
                request, "_ravel_service_virtual_owner", None
            )
            if old_owner is None:
                continue
            old_owner = int(old_owner)
            old_charge = float(
                getattr(
                    request,
                    "_ravel_service_virtual_charge",
                    0.0,
                )
            )
            new_owner = int(assignment[request._id][0])
            new_quote = assignment[request._id][1]
            new_charge = self._service_virtual_charge(
                request, new_owner, new_quote
            )
            self._service_virtual_load[old_owner] = max(
                0.0,
                self._service_virtual_load[old_owner] - old_charge,
            )
            self._service_virtual_load[new_owner] += new_charge
            request._ravel_service_virtual_owner = new_owner
            request._ravel_service_virtual_charge = new_charge
        return assignment


class RavelNativeCenterWorkBalancedGlobalScheduler(
    RavelNativeCenterWorkYieldGlobalScheduler
):
    """Minimize predicted flow time inside the SLO-equivalent set.

    Feasibility, queue-overflow cost, and normalized lateness remain strict
    primary objectives. Only placements tied on those safety terms compare
    predicted TTFT/E2E before the virtual load ledgers. This exposes a
    parameter-free Pareto point without combining unlike metrics in a fitted
    weighted score.
    """

    flow_time_before_virtual_balance = True


class RavelNativeCenterWorkEDFGlobalScheduler(
    RavelNativeCenterWorkYieldGlobalScheduler
):
    """Balance profiled work before applying pure mixed-class EDF."""

    marginal_work_first = True

    def _set_engine_priority(self, request: Request) -> None:
        if not self._latency_class_observed:
            request._vllm_priority = 0
            return
        request._vllm_priority = max(
            0,
            int(
                (request._arrived_at + self._limit(request))
                * 1_000_000
            ),
        )

    @staticmethod
    def _yield_dispatch_key(
        request: Request,
    ) -> tuple[float, float, int]:
        return (
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )


class RavelNativeCenterLatencyFirstGlobalScheduler(
    RavelNativeCenterWorkEDFGlobalScheduler
):
    """Protect TTFT/TBT requests before flexible TTLT-only work."""

    _FLEXIBLE_PRIORITY_OFFSET_US = 1_000_000_000_000_000

    @staticmethod
    def _semantic_deadline_key(
        request: Request,
    ) -> tuple[int, float, float, int]:
        return (
            int(request._request_type != LATENCY),
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )

    def _planning_key(
        self, request: Request
    ) -> tuple[int, float, float, int]:
        return self._semantic_deadline_key(request)

    def _set_engine_priority(self, request: Request) -> None:
        if not self._latency_class_observed:
            request._vllm_priority = 0
            return
        deadline_us = max(
            0,
            int(
                (request._arrived_at + self._limit(request))
                * 1_000_000
            ),
        )
        request._vllm_priority = deadline_us + (
            0
            if request._request_type == LATENCY
            else self._FLEXIBLE_PRIORITY_OFFSET_US
        )

    @staticmethod
    def _yield_dispatch_key(
        request: Request,
    ) -> tuple[int, float, float, int]:
        return (
            int(request._request_type != LATENCY),
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )


class RavelNativeCenterLatencyFairGlobalScheduler(
    RavelNativeCenterLatencyFirstGlobalScheduler
):
    """Use class priority while preserving fairness within Decode classes."""

    def _set_engine_priority(self, request: Request) -> None:
        if not self._latency_class_observed:
            request._vllm_priority = 0
            return
        request._vllm_priority = int(
            request._request_type != LATENCY
        )


class RavelNativeCenterSelectedAdmissionGlobalScheduler(
    RavelNativeCenterWorkEDFGlobalScheduler
):
    """Admit the predicted feasible set first, then use engine EDF."""

    def _refresh_yield_state(self, request: Request) -> None:
        request._ravel_yield_deferred = not bool(
            getattr(request, "_cluster_route_feasible", False)
        )
        self._set_engine_priority(request)

    @staticmethod
    def _yield_dispatch_key(
        request: Request,
    ) -> tuple[int, float, float, int]:
        return (
            int(
                bool(
                    getattr(
                        request, "_ravel_yield_deferred", False
                    )
                )
            ),
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )


class RavelNativeCenterDynamicChunkGlobalScheduler(
    RavelNativeCenterSelectedAdmissionGlobalScheduler
):
    """Export EDF and TBT metadata to the local dynamic-chunk adapter."""

    def _set_engine_priority(self, request: Request) -> None:
        if not self._latency_class_observed:
            request._vllm_priority = 0
            return
        deadline_at_s = request._arrived_at + self._limit(request)
        tbt_s = (
            float(request._slo_constraint[1])
            if request._request_type == LATENCY
            else None
        )
        request._vllm_priority = encode_priority(
            deadline_at_s,
            tbt_s,
        )


class RavelNativeCenterSLOEnvelopeGlobalScheduler(
    RavelNativeCenterSelectedAdmissionGlobalScheduler
):
    """Protect active Decode SLOs from non-preemptible Prefill chunks.

    Chunked Prefill bounds the number of prompt tokens issued in one engine
    iteration, but that iteration can still last longer than a running
    request's TBT deadline.  The sidecar therefore admits a prompt only when
    the profiled duration of the prospective next iteration fits the tightest
    active LATENCY request.  No realized output length or workload identity is
    used, and the engine remains work-conserving when no TBT obligation exists.
    """

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self._max_num_batched_tokens = int(
            getattr(args, "max_num_batched_tokens", 0)
        )
        if self._max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        self.envelope_deferrals = 0

    @staticmethod
    def _latency_obligation_alive(
        request: Request, now: float
    ) -> bool:
        if request._request_type != LATENCY:
            return False
        if not bool(getattr(request, "_ravel_ttft_alive", True)):
            return False
        if not bool(getattr(request, "_ravel_tbt_alive", True)):
            return False
        last_token_at = float(
            getattr(request, "_ravel_last_token_at", 0.0)
        )
        tbt_limit_s = max(
            1e-9, float(request._slo_constraint[1])
        )
        return last_token_at <= 0.0 or now - last_token_at <= tbt_limit_s

    def _protected_tbt_s(self, replica_id: int) -> Optional[float]:
        replica = self.shared_state.replica_budgets[replica_id]
        now = time.perf_counter()
        limits = [
            float(row._slo_constraint[1])
            for row in replica.running_requests
            if self._latency_obligation_alive(row, now)
        ]
        return min(limits) if limits else None

    def _prospective_prefill_gap(
        self, request: Request, replica_id: int
    ) -> tuple[float, Optional[float], int]:
        protected_tbt_s = self._protected_tbt_s(replica_id)
        if protected_tbt_s is None:
            return 0.0, None, 0

        replica = self.shared_state.replica_budgets[replica_id]
        running = replica.get_num_running_req()
        pending_tokens = sum(
            max(
                1,
                int(
                    getattr(
                        row,
                        "_actual_num_prefill_tokens",
                        row._num_prefill_tokens,
                    )
                ),
            )
            for row in replica.pending_requests
        )
        request_work = max(1, int(request._num_prefill_tokens))
        decode_tokens = min(running, self._max_num_batched_tokens - 1)
        prompt_budget = max(
            1, self._max_num_batched_tokens - decode_tokens
        )
        prompt_chunk_tokens = min(
            prompt_budget, pending_tokens + request_work
        )
        cluster = self.topology.cluster_for_replica(replica_id)
        profiled_prefill_gap_s = (
            cluster.prefill_intercept_for(running)
            + prompt_chunk_tokens * cluster.prefill_tpot_for(running)
        )
        predicted_gap_s = max(
            profiled_prefill_gap_s,
            cluster.decode_tpot_for(max(1, running)),
        )
        return predicted_gap_s, protected_tbt_s, prompt_chunk_tokens

    def _at_risk_victim_count(
        self, replica_id: int, predicted_gap_s: float
    ) -> int:
        replica = self.shared_state.replica_budgets[replica_id]
        now = time.perf_counter()
        return sum(
            self._latency_obligation_alive(row, now)
            and predicted_gap_s > float(row._slo_constraint[1])
            for row in replica.running_requests
        )

    def _envelope_safe(
        self, request: Request, replica_id: int, *, record: bool
    ) -> bool:
        gap_s, protected_tbt_s, prompt_chunk_tokens = (
            self._prospective_prefill_gap(request, replica_id)
        )
        victim_count = self._at_risk_victim_count(
            replica_id, gap_s
        )
        # Unit-weight SLO goodput: admitting one viable newcomer may trade at
        # most one incumbent success, but must not destroy multiple live SLOs.
        safe = victim_count == 0
        if record:
            request._ravel_prefill_gap_s = gap_s
            request._ravel_protected_tbt_s = protected_tbt_s or 0.0
            request._ravel_prefill_chunk_tokens = prompt_chunk_tokens
            request._ravel_admission_victims = victim_count
            request._ravel_admission_envelope_safe = safe
        return safe

    def _can_admit_to_engine(
        self, request: Request, replica_id: int
    ) -> bool:
        no_live_victim = self._envelope_safe(
            request, replica_id, record=True
        )
        own_quote = super()._quote(request, replica_id)
        # Miss isolation: never delay a newcomer that still has a feasible
        # end-to-end schedule.  Only predicted misses may yield to live SLOs.
        safe = own_quote.feasible or no_live_victim
        request._ravel_admission_envelope_safe = safe
        if not safe:
            request._ravel_admission_deferrals = int(
                getattr(request, "_ravel_admission_deferrals", 0)
            ) + 1
            request._ravel_max_admission_victims = max(
                int(
                    getattr(
                        request, "_ravel_max_admission_victims", 0
                    )
                ),
                int(getattr(request, "_ravel_admission_victims", 0)),
            )
            self.envelope_deferrals += 1
        return safe

class RavelNativeCenterTBTGlobalScheduler(
    RavelNativeCenterSelectedAdmissionGlobalScheduler
):
    """Approximate per-token deadlines with rate-monotonic priority.

    A LATENCY request renews its service obligation after every emitted token;
    its relative deadline is therefore its TBT SLO, not the one-shot TTFT
    deadline used by EDF placement. Smaller TBT periods receive higher fixed
    priority, while equal-period requests retain vLLM's stable FCFS order.
    Flexible TTLT-only work remains FCFS below the latency class.
    """

    _FLEXIBLE_PRIORITY = 1_000_000_000_000_000
    _TBT_PRIORITY_SCALE = 1_000_000_000

    def _set_engine_priority(self, request: Request) -> None:
        if request._request_type == LATENCY:
            request._vllm_priority = max(
                1,
                int(
                    float(request._slo_constraint[1])
                    * self._TBT_PRIORITY_SCALE
                ),
            )
            return
        request._vllm_priority = self._FLEXIBLE_PRIORITY

    @staticmethod
    def _yield_dispatch_key(request: Request) -> tuple:
        is_latency = request._request_type == LATENCY
        return (
            int(
                bool(
                    getattr(
                        request,
                        "_ravel_yield_deferred",
                        False,
                    )
                )
            ),
            int(not is_latency),
            (
                float(request._slo_constraint[1])
                if is_latency
                else request._arrived_at
                + float(request._slo_constraint[2])
            ),
            request._arrived_at,
            request._id,
        )


class RavelNativeCenterObjectiveEDFGlobalScheduler(
    RavelNativeCenterSelectedAdmissionGlobalScheduler
):
    """Order every admitted request by its applicable absolute SLO deadline.

    ``CenterSelectedAdmission`` previously disabled vLLM priority when a
    workload contained no LATENCY requests. That is valid for a TTFT-only
    policy, but not for COLLECTIVE/THROUGHPUT requests whose success is defined
    by TTLT. Their absolute TTLT deadline is request-visible and gives EDF a
    meaningful ordering without predicting the realized output length.
    """

    def _set_engine_priority(self, request: Request) -> None:
        request._vllm_priority = max(
            0,
            int(
                (request._arrived_at + self._limit(request))
                * 1_000_000
            ),
        )


class RavelNativeCenterDensityGlobalScheduler(
    RavelNativeCenterSelectedAdmissionGlobalScheduler
):
    """Order the certified set by JITServe's profiled service-gain density."""

    _JITSERVE_DECODE_GAIN = 8
    _DENSITY_PRIORITY_RESOLUTION = 1_000

    def _set_engine_priority(self, request: Request) -> None:
        replica_id = int(request._primary_replica)
        replica = self.shared_state.replica_budgets[replica_id]
        cluster = self.topology.cluster_for_replica(replica_id)
        prompt_tokens = max(1, int(request._num_prefill_tokens))
        output_tokens = max(
            1,
            int(request._routing_output_tokens_hint_used),
        )
        own_service_s = (
            prompt_tokens
            * cluster.prefill_tpot_for(replica.get_num_running_req())
            + max(0, output_tokens - 1)
            * cluster.decode_tpot_for(
                self._prospective_sequence_count(request, replica_id)
            )
        )
        predicted_finish_s = max(
            own_service_s,
            float(request._predicted_objective_s),
            1e-9,
        )
        deadline_discount = min(
            1.0,
            self._limit(request) / predicted_finish_s,
        )
        service_gain = (
            prompt_tokens
            + self._JITSERVE_DECODE_GAIN * output_tokens
        )
        density = (
            service_gain
            * deadline_discount
            / max(own_service_s, 1e-9)
        )
        request._vllm_priority = -int(
            round(density * self._DENSITY_PRIORITY_RESOLUTION)
        )

    @staticmethod
    def _yield_dispatch_key(
        request: Request,
    ) -> tuple[int, int, float, int]:
        return (
            int(bool(getattr(request, "_ravel_yield_deferred", False))),
            int(getattr(request, "_vllm_priority", 0)),
            request._arrived_at
            + (
                float(request._slo_constraint[0])
                if request._request_type == LATENCY
                else float(request._slo_constraint[2])
            ),
            request._id,
        )


class RavelNativeCenterCohortGlobalScheduler(
    RavelNativeCenterSelectedAdmissionGlobalScheduler
):
    """Expose one observed-arrival RTT before Latency placement commits."""

    def _mobile_hold_s(self, request: Request) -> float:
        if request._request_type != LATENCY:
            return super()._mobile_hold_s(request)
        if not self._burst_within_rtt(request):
            return 0.0
        remote_rtts = [
            cluster.rtt_s(request._client_region)
            for cluster in self.topology.clusters
            if cluster.cluster_id != request._client_region
            and cluster.rtt_s(request._client_region) > 0
        ]
        if not remote_rtts:
            return 0.0
        quotes = [
            self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        ]
        feasible_clusters = {
            self.topology.cluster_for_replica(quote.replica_id).cluster_id
            for quote in quotes
            if quote.feasible
        }
        if len(feasible_clusters) < 2:
            return 0.0
        hold_s = min(remote_rtts)
        slack_s = self._limit(request) - min(
            quote.predicted_objective_s for quote in quotes
        )
        return hold_s if slack_s >= hold_s else 0.0


class RavelNativeCenterOnTimeSetGlobalScheduler(
    RavelNativeCenterSelectedAdmissionGlobalScheduler
):
    """Maintain a maximum-cardinality approximation of on-time work.

    Requests are scanned by absolute deadline. When no replica can admit the
    next request on time, the planner replaces the largest profiled job if
    doing so preserves the predicted on-time count while freeing capacity.
    This is Moore-Hodgson on one machine and a deterministic approximation on
    the heterogeneous replicas. Removed requests are deferred, never rejected.
    """

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self._collective_stage_min = math.inf
        self._collective_stage_expected_output_tokens: Dict[int, int] = {}
        self._collective_stage_upper_output_tokens: Dict[int, int] = {}
        self._collective_context_expected_output_tokens: Dict[
            tuple[int, int], int
        ] = {}
        self._collective_context_upper_output_tokens: Dict[tuple[int, int], int] = {}
        output_profile_path = str(
            getattr(args, "ravel_collective_stage_output_profile", "")
        ) or os.environ.get(
            "RAVEL_COLLECTIVE_STAGE_OUTPUT_PROFILE", ""
        )
        if output_profile_path:
            with open(output_profile_path, "r", encoding="utf-8") as source:
                output_profile = json.load(source)
            if (
                int(output_profile.get("schema_version", 0)) not in {1, 2}
                or output_profile.get("kind")
                != "semantic-collective-stage-output-profile"
            ):
                raise ValueError("invalid collective stage output profile")
            selector = output_profile.get("selector", {})
            if int(selector.get("request_type", -1)) != COLLECTIVE:
                raise ValueError(
                    "output profile must select COLLECTIVE requests"
                )
            self._collective_stage_min = int(
                selector.get("min_stage_num", 2)
            )
            stages = output_profile.get("stages", {})
            self._collective_stage_expected_output_tokens = {
                int(stage_id): max(
                    1, int(round(float(values["mean_tokens"])))
                )
                for stage_id, values in stages.items()
                if int(values.get("samples", 0)) > 0
                and float(values.get("mean_tokens", 0)) > 0
            }
            self._collective_stage_upper_output_tokens = {
                int(stage_id): int(values["q95_tokens"])
                for stage_id, values in stages.items()
                if int(values.get("samples", 0)) > 0
                and int(values.get("q95_tokens", 0)) > 0
            }

            contexts = output_profile.get("contexts", {})
            if any(
                len(str(context_id).split(":")) != 2
                for context_id in contexts
            ):
                raise ValueError("invalid stage context in output profile")
            self._collective_context_expected_output_tokens = {
                tuple(int(part) for part in str(context_id).split(":")): max(
                    1, int(round(float(values["mean_tokens"])))
                )
                for context_id, values in contexts.items()
                if int(values.get("samples", 0)) > 0
                and float(values.get("mean_tokens", 0)) > 0
            }
            self._collective_context_upper_output_tokens = {
                tuple(int(part) for part in str(context_id).split(":")): int(
                    values["q95_tokens"]
                )
                for context_id, values in contexts.items()
                if int(values.get("samples", 0)) > 0
                and int(values.get("q95_tokens", 0)) > 0
            }

    def _collective_stage_profile_values(
        self, request: Request
    ) -> tuple[Optional[int], Optional[int]]:
        context = (int(request._stage_id), int(request._stage_num))
        expected = self._collective_context_expected_output_tokens.get(
            context,
            self._collective_stage_expected_output_tokens.get(context[0]),
        )
        upper = self._collective_context_upper_output_tokens.get(
            context,
            self._collective_stage_upper_output_tokens.get(context[0]),
        )
        return expected, upper

    def _apply_collective_stage_output_profile(
        self, request: Request
    ) -> None:
        if (
            request._request_type != COLLECTIVE
            or int(request._stage_num) < self._collective_stage_min
        ):
            return
        expected, upper = self._collective_stage_profile_values(request)
        if expected is None or upper is None:
            return
        request._routing_output_tokens_hint = expected
        request._routing_output_tokens_hint_used = expected
        request._routing_output_tokens_upper_hint = max(expected, upper)
        request._routing_output_tokens_upper_hint_used = max(
            expected, upper
        )

    async def _enqueue(self, request: Request) -> None:
        self._apply_collective_stage_output_profile(request)
        await super()._enqueue(request)

    @staticmethod
    def _request_tbt_limit(request: Request) -> float:
        if request._request_type != LATENCY:
            return math.inf
        return max(1e-9, float(request._slo_constraint[1]))

    def _plan_mobile_assignment(
        self,
        candidates: list[Request],
        snapshot: Optional[PlannerSnapshot] = None,
    ) -> Dict[int, tuple[int, Quote]]:
        if snapshot is not None:
            return super()._plan_mobile_assignment(candidates, snapshot)

        replica_ids = tuple(self.topology.replica_ids)
        candidate_ids = {request._id for request in candidates}
        candidate_by_id = {
            request._id: request for request in candidates
        }
        completion_tracking_enabled = bool(
            self.completion_virtual_service_enabled
        )
        completion_balance = bool(
            completion_tracking_enabled and not self._latency_class_observed
        )
        if completion_balance:
            for replica_id in self.topology.replica_ids:
                replica = self.shared_state.replica_budgets[replica_id]
                for row in (
                    *self.global_request_queue.get_all_requests(replica_id),
                    *replica.pending_requests,
                    *replica.running_requests,
                ):
                    self._register_service_virtual_admission(
                        row, owner_hint=replica_id
                    )
        base_virtual_service_load = dict(self._service_virtual_load)
        if completion_balance:
            for request in candidates:
                owner = getattr(
                    request, "_ravel_service_virtual_owner", None
                )
                if owner is None:
                    continue
                owner = int(owner)
                charge = float(
                    getattr(
                        request, "_ravel_service_virtual_charge", 0.0
                    )
                )
                base_virtual_service_load[owner] = max(
                    0.0, base_virtual_service_load[owner] - charge
                )
        # The baseline call is used only to freeze the feasible-set cardinality.
        # Disable the persistent completion ledger for this dry planning pass;
        # the final OnTimeSet assignment commits it exactly once below.
        self.completion_virtual_service_enabled = False
        try:
            baseline_assignment = super()._plan_mobile_assignment(candidates)
        finally:
            self.completion_virtual_service_enabled = (
                completion_tracking_enabled
            )
        selected_budget = sum(
            int(quote.feasible)
            for _replica_id, quote in baseline_assignment.values()
        )
        fixed_entries: Dict[
            int, list[tuple[tuple[float, float, int], int]]
        ] = {}
        fixed_total_work: Dict[int, int] = {}
        fixed_total_count: Dict[int, int] = {}
        base_pressure: Dict[int, int] = {}
        base_service_s: Dict[int, float] = {}
        base_tbt_limit: Dict[int, float] = {}

        for replica_id in replica_ids:
            replica = self.shared_state.replica_budgets[replica_id]
            fixed_rows = [
                request
                for request in self.global_request_queue.get_all_requests(
                    replica_id
                )
                if request._id not in candidate_ids
                and not bool(
                    getattr(request, "_ravel_yield_deferred", False)
                )
            ]
            entries = sorted(
                (
                    self._edf_key(request),
                    self._request_work(
                        request, replica_id, cold=False
                    )[0],
                )
                for request in fixed_rows
            )
            fixed_entries[replica_id] = entries
            fixed_total_work[replica_id] = sum(
                work for _key, work in entries
            )
            fixed_total_count[replica_id] = len(entries)
            pending = replica.get_num_pending_req()
            running = replica.get_num_running_req()
            base_pressure[replica_id] = (
                pending + running + len(fixed_rows)
            )
            cluster = self.topology.cluster_for_replica(replica_id)
            prefill_tpot = cluster.prefill_tpot_for(running)
            prefill_intercept = cluster.prefill_intercept_for(running)
            base_service_s[replica_id] = (
                (
                    self._engine_work(replica_id)
                    + fixed_total_work[replica_id]
                )
                * prefill_tpot
                + base_pressure[replica_id] * prefill_intercept
            )
            engine_rows = (
                *getattr(replica, "pending_requests", ()),
                *getattr(replica, "running_requests", ()),
                *fixed_rows,
            )
            base_tbt_limit[replica_id] = min(
                (
                    self._request_tbt_limit(request)
                    for request in engine_rows
                ),
                default=math.inf,
            )

        selected_by_replica: Dict[int, list[int]] = {
            replica_id: [] for replica_id in replica_ids
        }
        # id -> (replica, quote, prompt work, profiled service)
        selected: Dict[int, tuple[int, Quote, int, float]] = {}
        deferred_ids: set[int] = set()
        planned_work = {replica_id: 0 for replica_id in replica_ids}
        planned_count = {replica_id: 0 for replica_id in replica_ids}
        planned_service_s = {
            replica_id: 0.0 for replica_id in replica_ids
        }
        planned_virtual_service_s = {
            replica_id: 0.0 for replica_id in replica_ids
        }

        def initial_replica(request: Request) -> int:
            value = getattr(request, "_ravel_initial_replica", None)
            if value is not None:
                return int(value)
            return int(self._queued_owner(request))

        def legal_target(request: Request, replica_id: int) -> bool:
            return not (
                replica_id != initial_replica(request)
                and int(getattr(request, "_rebind_count", 0)) >= 1
            )

        def fixed_prefix(
            replica_id: int, request: Request
        ) -> tuple[int, int]:
            key = self._edf_key(request)
            work = 0
            count = 0
            for fixed_key, fixed_work in fixed_entries[replica_id]:
                if fixed_key > key:
                    break
                work += fixed_work
                count += 1
            return work, count

        def tbt_set_feasible(
            replica_id: int,
            prospective_sequences: int,
            request: Request,
            victim_id: Optional[int] = None,
        ) -> bool:
            limit = min(
                base_tbt_limit[replica_id],
                self._request_tbt_limit(request),
            )
            for request_id in selected_by_replica[replica_id]:
                if request_id == victim_id:
                    continue
                limit = min(
                    limit,
                    self._request_tbt_limit(
                        candidate_by_id[request_id]
                    ),
                )
            if not math.isfinite(limit):
                return True
            cluster = self.topology.cluster_for_replica(replica_id)
            return (
                cluster.decode_tpot_for(prospective_sequences)
                <= limit
            )

        def add_edges(request: Request):
            if len(selected) >= selected_budget:
                return []
            edges = []
            limit_s = max(1e-9, self._limit(request))
            for replica_id in replica_ids:
                if not legal_target(request, replica_id):
                    continue
                prefix_work, prefix_count = fixed_prefix(
                    replica_id, request
                )
                prospective_sequences = (
                    base_pressure[replica_id]
                    + planned_count[replica_id]
                    + 1
                )
                quote = self._quote(
                    request,
                    replica_id,
                    work_before_tokens=(
                        prefix_work + planned_work[replica_id]
                    ),
                    requests_before=(
                        prefix_count + planned_count[replica_id]
                    ),
                    prospective_sequences=prospective_sequences,
                    cold=False,
                )
                if not quote.feasible or not tbt_set_feasible(
                    replica_id, prospective_sequences, request
                ):
                    continue
                own_service_s = self._own_profiled_service_s(
                    request, replica_id, quote
                )
                current_service_s = (
                    base_service_s[replica_id]
                    + planned_service_s[replica_id]
                )
                marginal_work = (
                    (current_service_s + own_service_s) ** 2
                    - current_service_s**2
                ) / (limit_s**2)
                balance_finish = (
                    base_virtual_service_load[replica_id]
                    + planned_virtual_service_s[replica_id]
                    + own_service_s
                )
                request_work, _ = self._request_work(
                    request, replica_id, cold=False
                )
                edges.append(
                    (
                        balance_finish if completion_balance else marginal_work,
                        (
                            marginal_work
                            if completion_balance
                            else quote.predicted_objective_s / limit_s
                        ),
                        int(replica_id != initial_replica(request)),
                        replica_id,
                        quote,
                        request_work,
                        own_service_s,
                    )
                )
            return edges

        def select_request(request: Request, edge) -> None:
            replica_id = int(edge[3])
            quote = edge[4]
            request_work = int(edge[5])
            own_service_s = float(edge[6])
            selected_by_replica[replica_id].append(request._id)
            selected[request._id] = (
                replica_id,
                quote,
                request_work,
                own_service_s,
            )
            planned_work[replica_id] += request_work
            planned_count[replica_id] += 1
            planned_service_s[replica_id] += own_service_s
            planned_virtual_service_s[replica_id] += own_service_s
            deferred_ids.discard(request._id)

        for request in sorted(candidates, key=self._planning_key):
            edges = add_edges(request)
            if edges:
                select_request(
                    request, min(edges, key=lambda edge: edge[:4])
                )
                continue

            replacement_edges = []
            limit_s = max(1e-9, self._limit(request))
            for replica_id in replica_ids:
                if (
                    not legal_target(request, replica_id)
                    or not selected_by_replica[replica_id]
                ):
                    continue
                victim_id = max(
                    selected_by_replica[replica_id],
                    key=lambda request_id: (
                        selected[request_id][3],
                        request_id,
                    ),
                )
                (
                    _victim_replica,
                    _victim_quote,
                    victim_work,
                    victim_service_s,
                ) = selected[victim_id]
                prefix_work, prefix_count = fixed_prefix(
                    replica_id, request
                )
                prospective_sequences = (
                    base_pressure[replica_id]
                    + planned_count[replica_id]
                )
                quote = self._quote(
                    request,
                    replica_id,
                    work_before_tokens=max(
                        0,
                        prefix_work
                        + planned_work[replica_id]
                        - victim_work,
                    ),
                    requests_before=max(
                        0,
                        prefix_count
                        + planned_count[replica_id]
                        - 1,
                    ),
                    prospective_sequences=prospective_sequences,
                    cold=False,
                )
                if not quote.feasible or not tbt_set_feasible(
                    replica_id,
                    prospective_sequences,
                    request,
                    victim_id=victim_id,
                ):
                    continue
                request_work, _ = self._request_work(
                    request, replica_id, cold=False
                )
                own_service_s = self._own_profiled_service_s(
                    request, replica_id, quote
                )
                freed_service_s = victim_service_s - own_service_s
                if freed_service_s <= 1e-9:
                    continue
                balance_finish = (
                    base_virtual_service_load[replica_id]
                    + planned_virtual_service_s[replica_id]
                    - victim_service_s
                    + own_service_s
                )
                replacement_edges.append(
                    (
                        (
                            balance_finish
                            if completion_balance
                            else -freed_service_s
                        ),
                        (
                            -freed_service_s
                            if completion_balance
                            else quote.predicted_objective_s / limit_s
                        ),
                        int(replica_id != initial_replica(request)),
                        replica_id,
                        victim_id,
                        quote,
                        request_work,
                        own_service_s,
                    )
                )
            if not replacement_edges:
                deferred_ids.add(request._id)
                continue

            replacement = min(
                replacement_edges, key=lambda edge: edge[:4]
            )
            replica_id = int(replacement[3])
            victim_id = int(replacement[4])
            (
                _victim_replica,
                _victim_quote,
                victim_work,
                victim_service_s,
            ) = selected.pop(victim_id)
            selected_by_replica[replica_id].remove(victim_id)
            planned_work[replica_id] -= victim_work
            planned_count[replica_id] -= 1
            planned_service_s[replica_id] -= victim_service_s
            planned_virtual_service_s[replica_id] -= victim_service_s
            deferred_ids.add(victim_id)
            select_request(
                request,
                (
                    replacement[0],
                    replacement[1],
                    replacement[2],
                    replacement[3],
                    replacement[5],
                    replacement[6],
                    replacement[7],
                ),
            )

        # A replacement may expose a feasible slot on another replica.
        for request in sorted(
            (
                candidate_by_id[request_id]
                for request_id in tuple(deferred_ids)
            ),
            key=self._planning_key,
        ):
            edges = add_edges(request)
            if edges:
                select_request(
                    request, min(edges, key=lambda edge: edge[:4])
                )

        assignment: Dict[int, tuple[int, Quote]] = {
            request_id: (row[0], row[1])
            for request_id, row in selected.items()
        }
        post_work = dict(planned_work)
        post_count = dict(planned_count)
        post_service_s = dict(planned_service_s)
        post_virtual_service_s = dict(planned_virtual_service_s)
        for request in sorted(
            (
                candidate_by_id[request_id]
                for request_id in deferred_ids
            ),
            key=self._planning_key,
        ):
            limit_s = max(1e-9, self._limit(request))
            edges = []
            for replica_id in replica_ids:
                if not legal_target(request, replica_id):
                    continue
                quote = self._quote(
                    request,
                    replica_id,
                    work_before_tokens=(
                        fixed_total_work[replica_id]
                        + post_work[replica_id]
                    ),
                    requests_before=(
                        fixed_total_count[replica_id]
                        + post_count[replica_id]
                    ),
                    prospective_sequences=(
                        base_pressure[replica_id]
                        + post_count[replica_id]
                        + 1
                    ),
                    cold=False,
                )
                own_service_s = self._own_profiled_service_s(
                    request, replica_id, quote
                )
                current_service_s = (
                    base_service_s[replica_id]
                    + post_service_s[replica_id]
                )
                marginal_work = (
                    (current_service_s + own_service_s) ** 2
                    - current_service_s**2
                ) / (limit_s**2)
                balance_finish = (
                    base_virtual_service_load[replica_id]
                    + post_virtual_service_s[replica_id]
                    + own_service_s
                )
                request_work, _ = self._request_work(
                    request, replica_id, cold=False
                )
                lateness = max(
                    0.0, quote.predicted_objective_s - limit_s
                ) / limit_s
                edges.append(
                    (
                        balance_finish if completion_balance else lateness,
                        lateness if completion_balance else marginal_work,
                        (
                            marginal_work
                            if completion_balance
                            else quote.predicted_objective_s / limit_s
                        ),
                        int(replica_id != initial_replica(request)),
                        replica_id,
                        quote,
                        request_work,
                        own_service_s,
                    )
                )
            if not edges:
                raise RuntimeError(
                    "RAVEL has no legal deferred placement for request "
                    f"{request._id}"
                )
            edge = min(edges, key=lambda row: row[:5])
            replica_id = int(edge[4])
            quote = replace(edge[5], feasible=False)
            request_work = int(edge[6])
            own_service_s = float(edge[7])
            assignment[request._id] = (replica_id, quote)
            post_work[replica_id] += request_work
            post_count[replica_id] += 1
            post_service_s[replica_id] += own_service_s
            post_virtual_service_s[replica_id] += own_service_s

        if completion_balance:
            for request in candidates:
                old_owner = getattr(
                    request, "_ravel_service_virtual_owner", None
                )
                old_charge = float(
                    getattr(
                        request, "_ravel_service_virtual_charge", 0.0
                    )
                )
                new_owner, new_quote = assignment[request._id]
                new_owner = int(new_owner)
                new_charge = self._service_virtual_charge(
                    request, new_owner, new_quote
                )
                if old_owner is not None:
                    old_owner = int(old_owner)
                    self._service_virtual_load[old_owner] = max(
                        0.0,
                        self._service_virtual_load[old_owner] - old_charge,
                    )
                self._service_virtual_load[new_owner] += new_charge
                request._ravel_service_virtual_owner = new_owner
                request._ravel_service_virtual_charge = new_charge

        if assignment.keys() != candidate_ids:
            missing = sorted(candidate_ids - assignment.keys())
            raise RuntimeError(
                f"RAVEL on-time planner lost requests: {missing}"
            )
        return assignment


class RavelDiagnosticExactCompletionOracleGlobalScheduler(
    RavelNativeCenterOnTimeSetGlobalScheduler
):
    """Offline-only execution oracle with realized output information.

    This scheduler is deliberately unavailable unless the diagnostic guard is
    enabled. It measures whether exact job sizes plus local on-time-set/EDF
    control have real-system headroom; it must never be reported as an online
    RAVEL policy.
    """

    _DEFERRED_PRIORITY_OFFSET_US = 1_000_000_000_000_000

    def __init__(self, num_replicas, shared_state, args):
        if os.environ.get(
            "RAVEL_DIAGNOSTIC_ALLOW_REALIZED_OUTPUT", "0"
        ) != "1":
            raise RuntimeError(
                "exact completion oracle requires "
                "RAVEL_DIAGNOSTIC_ALLOW_REALIZED_OUTPUT=1"
            )
        if os.environ.get("RAVEL_COLLECTIVE_STAGE_OUTPUT_PROFILE", ""):
            raise RuntimeError(
                "exact completion oracle cannot use an output profile"
            )
        self._completion_order = os.environ.get(
            "RAVEL_DIAGNOSTIC_COMPLETION_ORDER", "edf"
        )
        if self._completion_order not in {"edf", "spt"}:
            raise ValueError(
                "RAVEL_DIAGNOSTIC_COMPLETION_ORDER must be edf or spt"
            )
        super().__init__(num_replicas, shared_state, args)

    @staticmethod
    def _apply_realized_output(request: Request) -> None:
        if request._request_type != COLLECTIVE:
            raise RuntimeError(
                "exact completion oracle only supports COLLECTIVE requests"
            )
        exact_tokens = max(1, int(request._output_len))
        request._routing_output_tokens_hint = exact_tokens
        request._routing_output_tokens_hint_used = exact_tokens
        request._routing_output_tokens_upper_hint = exact_tokens
        request._routing_output_tokens_upper_hint_used = exact_tokens

    async def _enqueue(self, request: Request) -> None:
        self._apply_realized_output(request)
        # Skip OnTimeSet's optional stage profile: exact output is the sole
        # diagnostic difference from the online planner.
        await RavelNativeCenterSelectedAdmissionGlobalScheduler._enqueue(
            self, request
        )

    def _refresh_yield_state(self, request: Request) -> None:
        if self._completion_order == "spt":
            request._ravel_yield_deferred = False
            self._set_engine_priority(request)
            return
        super()._refresh_yield_state(request)

    def _yield_dispatch_key(self, request: Request) -> tuple:
        if self._completion_order == "spt":
            return (
                int(request._routing_output_tokens_hint_used),
                request._arrived_at + self._limit(request),
                request._arrived_at,
                request._id,
            )
        return (
            RavelNativeCenterSelectedAdmissionGlobalScheduler
            ._yield_dispatch_key(request)
        )

    def _set_engine_priority(self, request: Request) -> None:
        if self._completion_order == "spt":
            output_tokens = max(
                1, int(request._routing_output_tokens_hint_used)
            )
            relative_deadline_us = max(
                0, int(self._limit(request) * 1_000_000)
            )
            request._vllm_priority = (
                output_tokens * 100_000_000
                + relative_deadline_us
            )
            return
        deadline_us = max(
            0,
            int(
                (request._arrived_at + self._limit(request))
                * 1_000_000
            ),
        )
        deferred = bool(
            getattr(request, "_ravel_yield_deferred", False)
        )
        request._vllm_priority = deadline_us + (
            self._DEFERRED_PRIORITY_OFFSET_US if deferred else 0
        )


class RavelNativeSLOFlowGlobalScheduler(
    RavelNativeCenterOnTimeSetGlobalScheduler
):
    """Adapt local control to the active request-visible SLO semantics.

    Completion-only cohorts use the maximum on-time-set planner. Once a
    TTFT/TBT obligation is observed, routing returns to the selected-admission
    planner and exports EDF/TBT metadata for profile-derived Prefill chunks.
    """

    def _set_engine_priority(self, request: Request) -> None:
        if not self._latency_class_observed:
            request._vllm_priority = 0
            return
        if request._request_type != LATENCY:
            # Flexible completion traffic keeps stable FCFS order within its
            # class. Encoding every TTLT deadline as EDF can indefinitely
            # postpone later-deadline requests during a sustained arrival
            # stream, inflating TTFT without improving their declared TTLT.
            request._vllm_priority = 0
            return
        deadline_at_s = request._arrived_at + self._limit(request)
        request._vllm_priority = encode_priority(
            deadline_at_s,
            float(request._slo_constraint[1]),
        )

    async def _dispatch_schedulable(self) -> None:
        if not self._latency_class_observed:
            await (
                RavelNativeMobileInsertionGlobalScheduler
                ._dispatch_schedulable(self)
            )
            return
        await RavelNativeCenterYieldGlobalScheduler._dispatch_schedulable(
            self
        )

    def _plan_mobile_assignment(
        self,
        candidates: list[Request],
        snapshot: Optional[PlannerSnapshot] = None,
    ) -> Dict[int, tuple[int, Quote]]:
        if self._latency_class_observed:
            return (
                RavelNativeCenterWorkYieldGlobalScheduler
                ._plan_mobile_assignment(self, candidates, snapshot)
            )
        return super()._plan_mobile_assignment(candidates, snapshot)


class RavelNativeMobileAdaptiveGlobalScheduler(
    RavelNativeMobileTriggeredGlobalScheduler
):
    """Expose one more arrival when the observed stream makes it worthwhile.

    The extension is based only on request-visible SLO type, recent arrival
    gaps, and topology RTT. It does not use a workload label or future
    arrivals. Irregular streams retain MobileTriggered's operating point;
    regular streams and collective requests may use a longer, still
    RTT-bounded reversible-placement window.
    """

    mobile_collective_hold_rtt_fraction = 0.5
    mobile_regular_hold_rtt_fraction = 0.5
    mobile_regularity_min_gaps = 4
    mobile_regularity_window = 8
    mobile_regularity_cv_threshold = 0.15
    mobile_regular_gap_headroom = 1.05

    def _recent_same_region_gaps(self, request: Request) -> list[float]:
        arrivals = [
            timestamp
            for timestamp, region in self._recent_arrivals
            if region == request._client_region
        ]
        return [
            later - earlier
            for earlier, later in zip(arrivals, arrivals[1:])
            if later > earlier
        ][-self.mobile_regularity_window :]

    def _safe_extended_hold(
        self,
        request: Request,
        candidate_s: float,
    ) -> float:
        if candidate_s <= 0:
            return 0.0
        quotes = [
            self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        ]
        feasible_clusters = {
            self.topology.cluster_for_replica(quote.replica_id).cluster_id
            for quote in quotes
            if quote.feasible
        }
        if len(feasible_clusters) < 2:
            return 0.0
        best_objective = min(quote.predicted_objective_s for quote in quotes)
        if (
            self._limit(request) - best_objective
            < self.mobile_min_slack_factor * candidate_s
        ):
            return 0.0
        return candidate_s

    def _mobile_hold_s(self, request: Request) -> float:
        base_hold_s = super()._mobile_hold_s(request)
        request._ravel_replan_on_flexible_arrival = False
        if request._request_type == LATENCY or base_hold_s <= 0:
            return base_hold_s

        remote_rtts = [
            cluster.rtt_s(request._client_region)
            for cluster in self.topology.clusters
            if cluster.cluster_id != request._client_region
            and cluster.rtt_s(request._client_region) > 0
        ]
        if not remote_rtts:
            return base_hold_s

        candidate_s = base_hold_s
        if request._request_type == COLLECTIVE:
            candidate_s = max(
                candidate_s,
                self.mobile_collective_hold_rtt_fraction * min(remote_rtts),
            )

        gaps = self._recent_same_region_gaps(request)
        if len(gaps) >= self.mobile_regularity_min_gaps:
            mean_gap = sum(gaps) / len(gaps)
            variance = sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)
            coefficient_of_variation = math.sqrt(variance) / max(
                mean_gap, 1e-9
            )
            max_regular_hold = (
                self.mobile_regular_hold_rtt_fraction * max(remote_rtts)
            )
            if (
                mean_gap <= max(remote_rtts)
                and coefficient_of_variation
                <= self.mobile_regularity_cv_threshold
            ):
                candidate_s = max(
                    candidate_s,
                    min(
                        self.mobile_regular_gap_headroom * mean_gap,
                        max_regular_hold,
                    ),
                )

        extended_s = self._safe_extended_hold(request, candidate_s)
        if extended_s > base_hold_s + 1e-9:
            request._ravel_replan_on_flexible_arrival = True
        elif request._request_type == COLLECTIVE:
            request._ravel_replan_on_flexible_arrival = True
        return max(base_hold_s, extended_s)

    def _has_revealed_mobile_cohort(self) -> bool:
        now = time.perf_counter()
        mobile = 0
        for replica_id in self.topology.replica_ids:
            for request in self.global_request_queue.get_all_requests(replica_id):
                if float(
                    getattr(request, "_ravel_mobile_until", request._arrived_at)
                ) > now:
                    mobile += 1
                    if mobile >= 2:
                        return True
        return False

    async def schedule(self, new_request: Optional[Request]) -> int:
        async with self._schedule_lock:
            should_replan = new_request is None
            if new_request is not None:
                await self._enqueue(new_request)
                hold_s = float(
                    getattr(new_request, "_ravel_mobile_hold_s", 0.0)
                )
                should_replan = (
                    new_request._request_type == LATENCY
                    or hold_s <= 0
                    or (
                        bool(
                            getattr(
                                new_request,
                                "_ravel_replan_on_flexible_arrival",
                                False,
                            )
                        )
                        and self._has_revealed_mobile_cohort()
                    )
                )
            if should_replan:
                await self._direct_reassign_mobile()
            await self._dispatch_schedulable()
            return -1


class _RavelNativeSurgicalGlobalScheduler(RavelNativeGlobalScheduler):
    """Keep Native placement, overriding only strong token-work disagreements."""

    objective_override_s = 0.05
    pressure_override_headroom = 2

    def _choose_initial(self, request: Request) -> Quote:
        baseline = super()._choose_initial(request)
        quotes = {
            replica_id: self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        }
        feasible = [row for row in quotes.values() if row.feasible]
        available = feasible or list(quotes.values())
        candidate = min(
            available,
            key=lambda row: (
                row.predicted_objective_s,
                row.predicted_service_s,
                self._pressure(row.replica_id),
                -row.prefix_hit_tokens,
                row.replica_id,
            ),
        )
        objective_gain = (
            baseline.predicted_objective_s - candidate.predicted_objective_s
        )
        pressure_ok = self._pressure(candidate.replica_id) <= (
            self._pressure(baseline.replica_id)
            + self.pressure_override_headroom
        )
        if (
            candidate.replica_id != baseline.replica_id
            and objective_gain >= self.objective_override_s
            and pressure_ok
        ):
            request._cluster_route_reason = (
                "ravel_native_surgical_"
                f"{int(1000 * self.objective_override_s)}ms_override"
            )
            return candidate
        return baseline


class RavelNativeSurgical25GlobalScheduler(
    _RavelNativeSurgicalGlobalScheduler
):
    objective_override_s = 0.025


class RavelNativeSurgical50GlobalScheduler(
    _RavelNativeSurgicalGlobalScheduler
):
    objective_override_s = 0.05


class RavelNativeSurgical100GlobalScheduler(
    _RavelNativeSurgicalGlobalScheduler
):
    objective_override_s = 0.10


class RavelNativeFlexibleSurgical50GlobalScheduler(
    RavelNativeSurgical50GlobalScheduler
):
    """Apply the 50 ms override only to non-latency requests."""

    def _choose_initial(self, request: Request) -> Quote:
        if request._request_type == LATENCY:
            return RavelNativeGlobalScheduler._choose_initial(self, request)
        return super()._choose_initial(request)


class _RavelNativeClassIsolationGlobalScheduler(RavelNativeGlobalScheduler):
    """Avoid mixing latency commitments with flexible Decode work."""

    class_service_guard_s = 0.10

    def _class_counts(self, replica_id: int) -> tuple[int, int]:
        replica = self.shared_state.replica_budgets[replica_id]
        rows = [
            *self.global_request_queue.get_all_requests(replica_id),
            *replica.pending_requests,
            *replica.running_requests,
        ]
        unique = {row._id: row for row in rows}
        latency = sum(
            row._request_type == LATENCY for row in unique.values()
        )
        return latency, len(unique) - latency

    def _choose_initial(self, request: Request) -> Quote:
        baseline = super()._choose_initial(request)
        quotes = {
            replica_id: self._quote(request, replica_id)
            for replica_id in self.topology.replica_ids
        }
        feasible = [row for row in quotes.values() if row.feasible]
        available = feasible or list(quotes.values())
        minimum_service = min(row.predicted_service_s for row in available)
        guarded = [
            row
            for row in available
            if row.predicted_service_s
            <= minimum_service + self.class_service_guard_s
        ]

        def interference(row: Quote) -> int:
            latency, flexible = self._class_counts(row.replica_id)
            return flexible if request._request_type == LATENCY else latency

        candidate = min(
            guarded,
            key=lambda row: (
                interference(row),
                row.predicted_objective_s,
                row.predicted_service_s,
                self._pressure(row.replica_id),
                -row.prefix_hit_tokens,
                row.replica_id,
            ),
        )
        if (
            candidate.replica_id != baseline.replica_id
            and interference(candidate) < interference(baseline)
        ):
            request._cluster_route_reason = (
                "ravel_native_class_isolation_"
                f"{int(1000 * self.class_service_guard_s)}ms"
            )
            return candidate
        return baseline


class RavelNativeClassIsolation50GlobalScheduler(
    _RavelNativeClassIsolationGlobalScheduler
):
    class_service_guard_s = 0.05


class RavelNativeClassIsolation200GlobalScheduler(
    _RavelNativeClassIsolationGlobalScheduler
):
    class_service_guard_s = 0.20


class RavelHindsightFixedGlobalScheduler(RavelNativeGlobalScheduler):
    """Replay an offline RAVEL assignment only for upper-bound validation."""

    _REPLICA_IDS = {
        "local-gpu0": 0,
        "local-gpu1": 1,
        "remote-gpu0": 2,
        "remote-gpu1": 3,
    }

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        raw_path = str(getattr(args, "ravel_assignment_file", ""))
        if not raw_path:
            raise ValueError("ravel_assignment_file is required")
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = str(getattr(args, "ravel_assignment_arm", "future_initial"))
        if isinstance(payload, list):
            values = payload
        else:
            if arm == "selected":
                arm = str(payload["decomposition"]["selected_upper_arm"])
            values = payload[arm]["guarded"]["assignments"]
        self._fixed_assignments = tuple(
            self._REPLICA_IDS[str(value)]
            if str(value) in self._REPLICA_IDS
            else int(value)
            for value in values
        )
        expected = int(getattr(args, "request_num", len(values)))
        if len(self._fixed_assignments) < expected:
            raise ValueError(
                "RAVEL upper assignment is shorter than the request trace"
            )

    def _choose_initial(self, request: Request) -> Quote:
        try:
            replica_id = self._fixed_assignments[int(request._id)]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                f"missing fixed assignment for request {request._id}"
            ) from error
        if replica_id not in self.topology.replica_ids:
            raise ValueError(f"invalid fixed replica {replica_id}")
        quote = self._quote(request, replica_id)
        request._cluster_route_reason = "ravel_hindsight_fixed_upper"
        return quote

    async def _joint_rebalance(self, trigger: Optional[Request]) -> int:
        return 0


class RavelHindsightJointPlanGlobalScheduler(
    RavelHindsightFixedGlobalScheduler
):
    """Replay a future-aware assignment/admission plan for diagnostics only.

    The plan may use future arrivals and realized output lengths. Requiring an
    explicit environment guard prevents this scheduler from being confused
    with an online RAVEL policy. It never rejects requests: predicted misses
    remain in the central queue and are released after the protected cohort.
    """

    _FUTURE_PLAN_GUARD = "RAVEL_DIAGNOSTIC_ALLOW_FUTURE_PLAN"
    _HOLD_FRACTION = "RAVEL_DIAGNOSTIC_PLAN_HOLD_FRACTION"

    def __init__(self, num_replicas, shared_state, args):
        if os.environ.get(self._FUTURE_PLAN_GUARD, "") != "1":
            raise RuntimeError(
                f"{self._FUTURE_PLAN_GUARD}=1 is required for the "
                "future-aware diagnostic scheduler"
            )
        RavelNativeGlobalScheduler.__init__(
            self, num_replicas, shared_state, args
        )
        raw_path = str(getattr(args, "ravel_assignment_file", ""))
        if not raw_path:
            raise ValueError("ravel_assignment_file is required")
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        required = {
            "best_assignments",
            "best_admission_rank",
            "best_not_before_ms",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "joint hindsight plan is missing fields: "
                + ", ".join(sorted(missing))
            )

        self._fixed_assignments = tuple(
            int(value) for value in payload["best_assignments"]
        )
        self._admission_rank = {
            int(request_id): int(rank)
            for request_id, rank in payload["best_admission_rank"].items()
        }
        self._not_before_ms = {
            int(request_id): float(not_before_ms)
            for request_id, not_before_ms
            in payload["best_not_before_ms"].items()
        }
        expected = int(
            getattr(args, "request_num", len(self._fixed_assignments))
        )
        expected_ids = set(range(expected))
        if len(self._fixed_assignments) < expected:
            raise ValueError(
                "joint hindsight assignments are shorter than the trace"
            )
        if not expected_ids.issubset(self._admission_rank):
            raise ValueError(
                "joint hindsight admission rank does not cover the trace"
            )
        invalid_replicas = {
            replica_id
            for replica_id in self._fixed_assignments[:expected]
            if replica_id not in self.topology.replica_ids
        }
        if invalid_replicas:
            raise ValueError(
                f"joint hindsight plan has invalid replicas: "
                f"{sorted(invalid_replicas)}"
            )

        try:
            self._hold_fraction = float(
                os.environ.get(self._HOLD_FRACTION, "1.0")
            )
        except ValueError as error:
            raise ValueError(
                f"{self._HOLD_FRACTION} must be a number"
            ) from error
        if not 0.0 <= self._hold_fraction <= 1.0:
            raise ValueError(
                f"{self._HOLD_FRACTION} must be in [0, 1]"
            )

        self._selected_request_ids = expected_ids.difference(
            self._not_before_ms
        )
        self._seen_selected_request_ids: set[int] = set()
        self._first_arrival_at: Optional[float] = None
        self._last_selected_arrival_at: Optional[float] = None
        self._release_at: Optional[float] = None
        self._release_task: Optional[asyncio.Task] = None
        self._full_release_relative_s = max(
            self._not_before_ms.values(), default=0.0
        ) / 1000.0

    def _choose_initial(self, request: Request) -> Quote:
        quote = super()._choose_initial(request)
        request._cluster_route_reason = "ravel_hindsight_joint_plan"
        request._vllm_priority = self._admission_rank[int(request._id)]
        return quote

    async def _enqueue(self, request: Request) -> None:
        await RavelNativeGlobalScheduler._enqueue(self, request)
        now = float(request._arrived_at)
        if self._first_arrival_at is None:
            self._first_arrival_at = now
        request_id = int(request._id)
        if request_id in self._selected_request_ids:
            self._seen_selected_request_ids.add(request_id)
            self._last_selected_arrival_at = max(
                now,
                self._last_selected_arrival_at
                if self._last_selected_arrival_at is not None
                else now,
            )
        self._maybe_fix_release_time()

    def _maybe_fix_release_time(self) -> None:
        if (
            self._release_at is not None
            or not self._not_before_ms
            or self._first_arrival_at is None
            or self._last_selected_arrival_at is None
            or self._seen_selected_request_ids
            != self._selected_request_ids
        ):
            return
        full_release_at = (
            self._first_arrival_at + self._full_release_relative_s
        )
        protected_interval_s = max(
            0.0, full_release_at - self._last_selected_arrival_at
        )
        self._release_at = (
            self._last_selected_arrival_at
            + self._hold_fraction * protected_interval_s
        )
        delay_s = max(0.0, self._release_at - time.perf_counter())
        if delay_s > 0:
            self._release_task = asyncio.create_task(
                self._wake_at_release(delay_s)
            )

    async def _wake_at_release(self, delay_s: float) -> None:
        try:
            await asyncio.sleep(delay_s)
            await self.schedule(None)
        finally:
            self._release_task = None

    def _eligible_for_admission(
        self, request: Request, now: float
    ) -> bool:
        if int(request._id) not in self._not_before_ms:
            return True
        return self._release_at is not None and now >= self._release_at

    async def _dispatch_schedulable(self) -> None:
        now = time.perf_counter()
        for replica_id in self.topology.replica_ids:
            queued = self.global_request_queue.get_all_requests(replica_id)
            if not queued:
                continue
            replica = self.shared_state.replica_budgets[replica_id]
            (
                current_budget,
                _,
                _,
                pending_requests,
                _,
            ) = await replica.get_load_states()
            open_slots = max(
                0, self.pending_request_limit - pending_requests
            )
            if current_budget <= 0 or open_slots <= 0:
                continue

            remaining_budget = current_budget
            ordered = sorted(
                queued,
                key=lambda request: (
                    self._admission_rank[int(request._id)],
                    request._arrived_at,
                    request._id,
                ),
            )
            admitted_count = 0
            for request in ordered:
                if admitted_count >= open_slots:
                    break
                if not self._eligible_for_admission(request, now):
                    continue
                work = max(1, int(request._num_prefill_tokens))
                if work > remaining_budget:
                    continue
                if not self.global_request_queue.del_req(
                    replica_id, request
                ):
                    continue
                request._vllm_priority = self._admission_rank[
                    int(request._id)
                ]
                request._enforce_prefill_budget = True
                admitted = (
                    await self.shared_state.add_posting_request_tasks(
                        replica_id, request
                    )
                )
                if not admitted:
                    self.global_request_queue.push(
                        replica_id,
                        request,
                        int(
                            getattr(
                                request,
                                "_estimated_prefix_hit_tokens",
                                0,
                            )
                        ),
                    )
                    continue
                admitted_count += 1
                remaining_budget -= work
