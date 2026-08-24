from __future__ import annotations

import math
import time
from typing import Dict, Optional

from dualmap.cluster.slo import COLLECTIVE, LATENCY
from dualmap.entities.request import Request
from ravel_engine_adapter.protocol import encode_phase_priority
from dualmap.scheduler.global_scheduler.ravel_native_global_scheduler import (
    PlannerSnapshot,
    Quote,
    RavelNativeCenterSelectedAdmissionGlobalScheduler,
    RavelNativeCenterWorkBalancedGlobalScheduler,
    RavelNativeCenterWorkYieldGlobalScheduler,
    RavelNativeCenterYieldGlobalScheduler,
    RavelNativeSLOFlowGlobalScheduler,
)


class RavelUnifiedGlobalScheduler(RavelNativeSLOFlowGlobalScheduler):
    """Production RAVEL after the Campaign-B mechanism ablation.

    RAVEL keeps causal completion-pressure detection, the held-out semantic
    output profile, revisable placement, risk-aware quotes, and completion
    virtual-service accounting. Candidate placement always uses WorkYield and
    dispatch always uses the ordinary reversible CenterYield frontier.

    The protected OnTimeSet cohort and Router-side soft-admission withholding
    were removed. On DeepResearch 16x they reduced SLO attainment from 75.0%
    on this path to a 63.04% five-run Full mean by increasing queue wait from
    1.011 seconds to 7.893 seconds.
    """

    completion_virtual_service_enabled = True
    completion_virtual_release_fraction = 1.0

    @property
    def plan_single_mobile_candidate(self) -> bool:
        """Replan lone arrivals while completion pressure remains active."""

        return bool(getattr(self, "_completion_pressure_active", False))

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self._total_sequence_capacity = (
            int(self.pending_request_limit) * int(num_replicas)
        )
        self._workflow_completion_observed = False
        self._completion_arrival_count = 0
        self._completion_arrival_first_at = math.inf
        self._completion_arrival_latest_at = -math.inf
        self._completion_arrival_min_limit_s = math.inf
        self._arrival_pressure_observed = False
        self._arrival_pressure_min_samples = max(
            4, min(16, int(self.pending_request_limit) // 4)
        )
        self._non_workflow_observed = False
        self._completion_pressure_active = False
        self.completion_pressure_activations = 0
        self._completion_virtual_requests: Dict[int, Request] = {}
        if self.completion_virtual_service_enabled:
            shared_state.set_request_terminal_callback(
                self.on_request_terminal
            )

    @staticmethod
    def _is_workflow_completion(request: Request) -> bool:
        return (
            request._request_type == COLLECTIVE
            and int(getattr(request, "_stage_num", 1)) > 1
        )

    def _outstanding_count(self) -> int:
        queued = sum(
            self.global_request_queue.get_queue_len(replica_id)
            for replica_id in self.topology.replica_ids
        )
        engine = sum(
            self.shared_state.replica_budgets[
                replica_id
            ].get_num_pending_req()
            + self.shared_state.replica_budgets[
                replica_id
            ].get_num_running_req()
            for replica_id in self.topology.replica_ids
        )
        return queued + engine

    def _observe_completion_arrival(self, request: Request) -> None:
        arrived_at = float(request._arrived_at)
        self._completion_arrival_count += 1
        self._completion_arrival_first_at = min(
            self._completion_arrival_first_at, arrived_at
        )
        self._completion_arrival_latest_at = max(
            self._completion_arrival_latest_at, arrived_at
        )
        self._completion_arrival_min_limit_s = min(
            self._completion_arrival_min_limit_s,
            max(1e-6, self._limit(request)),
        )
        if self._completion_arrival_count < self._arrival_pressure_min_samples:
            return
        span_s = (
            self._completion_arrival_latest_at
            - self._completion_arrival_first_at
        )
        if span_s <= 0:
            self._arrival_pressure_observed = True
            return
        observed_qps = (self._completion_arrival_count - 1) / span_s
        pressure_window_s = max(
            1.0, 0.5 * self._completion_arrival_min_limit_s
        )
        activation_qps = self._total_sequence_capacity / pressure_window_s
        if observed_qps >= activation_qps:
            self._arrival_pressure_observed = True

    def _update_completion_pressure(self) -> bool:
        outstanding = self._outstanding_count()
        if outstanding <= 0:
            self._completion_pressure_active = False
            return False
        eligible = (
            self._workflow_completion_observed
            and not self._non_workflow_observed
            and self._arrival_pressure_observed
        )
        if eligible and not self._completion_pressure_active:
            self._completion_pressure_active = True
            self.completion_pressure_activations += 1
        return eligible

    def _apply_completion_output_profile(self, request: Request) -> None:
        """Load a held-out semantic profile without exposing true output."""

        super()._apply_collective_stage_output_profile(request)
        if not self._is_workflow_completion(request):
            return
        expected, upper = self._collective_stage_profile_values(request)
        if expected is None or upper is None:
            return
        request._ravel_profile_output_expected = max(1, int(expected))
        request._ravel_profile_output_upper = max(
            int(expected), int(upper)
        )

    def _output_token_bounds(self, request: Request) -> tuple[int, int]:
        """Use the semantic mean for completion-pressure routing decisions."""

        profiled_expected = int(
            getattr(request, "_ravel_profile_output_expected", 0)
        )
        if self._update_completion_pressure() and profiled_expected > 0:
            expected = profiled_expected
            request._routing_output_tokens_hint_used = expected
            request._routing_output_tokens_upper_hint_used = expected
            return expected, expected
        return super()._output_token_bounds(request)

    async def _enqueue(self, request: Request) -> None:
        if self._is_workflow_completion(request):
            self._workflow_completion_observed = True
            self._observe_completion_arrival(request)
            self._apply_completion_output_profile(request)
        else:
            self._non_workflow_observed = True
        await RavelNativeCenterSelectedAdmissionGlobalScheduler._enqueue(
            self, request
        )
        if (
            self.completion_virtual_service_enabled
            and hasattr(request, "_ravel_service_virtual_owner")
        ):
            self._completion_virtual_requests[int(request._id)] = request
        self._update_completion_pressure()

    async def on_request_terminal(
        self, request_id: int, attempt: int
    ) -> None:
        """Release completed work from the active completion ledger."""

        request = self._completion_virtual_requests.pop(
            int(request_id), None
        )
        if request is None:
            return
        owner = getattr(request, "_ravel_service_virtual_owner", None)
        charge = float(
            getattr(request, "_ravel_service_virtual_charge", 0.0)
        )
        if owner is not None:
            owner = int(owner)
            self._service_virtual_load[owner] = max(
                0.0,
                self._service_virtual_load[owner]
                - charge * self.completion_virtual_release_fraction,
            )
        for attribute in (
            "_ravel_service_virtual_owner",
            "_ravel_service_virtual_charge",
        ):
            if hasattr(request, attribute):
                delattr(request, attribute)
        replan = getattr(self.shared_state, "request_replan", None)
        if callable(replan):
            replan()

    def _mobile_candidates(self) -> list[Request]:
        rows = []
        for replica_id in self.topology.replica_ids:
            rows.extend(
                self.global_request_queue.get_all_requests(replica_id)
            )
        rows = [
            row
            for row in rows
            if row._request_type != LATENCY
            or self._latency_mobile_allowed(row)
        ]
        rows.sort(
            key=lambda row: (
                row._arrived_at + self._limit(row),
                row._arrived_at,
                row._id,
            )
        )
        limit = (
            self._total_sequence_capacity
            if self._update_completion_pressure()
            else self.mobile_candidate_limit
        )
        return rows[:limit]

    def _plan_mobile_assignment(
        self,
        candidates: list[Request],
        snapshot: Optional[PlannerSnapshot] = None,
    ) -> Dict[int, tuple[int, Quote]]:
        for request in candidates:
            self._apply_completion_output_profile(request)
        return RavelNativeCenterWorkYieldGlobalScheduler._plan_mobile_assignment(
            self, candidates, snapshot
        )

    async def _dispatch_schedulable(self) -> None:
        await RavelNativeCenterYieldGlobalScheduler._dispatch_schedulable(
            self
        )

    async def schedule(self, new_request: Optional[Request]) -> int:
        async with self._schedule_lock:
            if new_request is not None:
                await self._enqueue(new_request)
            should_replan = (
                self._update_completion_pressure()
                or new_request is None
                or (
                    new_request is not None
                    and (
                        new_request._request_type == LATENCY
                        or float(
                            getattr(
                                new_request,
                                "_ravel_mobile_hold_s",
                                0.0,
                            )
                        )
                        <= 0.0
                    )
                )
            )
            if should_replan:
                await self._direct_reassign_mobile()
            await self._dispatch_schedulable()
            return -1


class RavelUnifiedBalancedGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Latency-balanced Pareto point without protected admission.

    Mixed traffic uses WorkBalanced placement. Completion-pressure traffic
    stays on the production WorkYield path. Both paths dispatch immediately
    through CenterYield.
    """

    flow_time_before_virtual_balance = True

    def _set_engine_priority(self, request: Request) -> None:
        replica_id = int(request._primary_replica)
        cluster = self.topology.cluster_for_replica(replica_id)
        remaining_budget_s = (
            request._arrived_at
            + self._limit(request)
            - time.perf_counter()
            - cluster.rtt_s(request._client_region)
        )
        decode_service_s = 0.0
        conservative_service_s = 0.0
        if request._request_type != LATENCY:
            output_tokens = max(
                1,
                int(
                    getattr(
                        request,
                        "_routing_output_tokens_upper_hint_used",
                        request._routing_output_tokens_upper_hint,
                    )
                ),
            )
            decode_service_s = max(
                0, output_tokens - 1
            ) * cluster.decode_tpot_for(
                self._prospective_sequence_count(request, replica_id)
            )
            remaining_budget_s -= decode_service_s
            conservative_output_tokens = max(
                output_tokens,
                int(
                    getattr(
                        request,
                        "_ravel_profile_output_upper",
                        output_tokens,
                    )
                ),
            )
            conservative_service_s = max(
                0, conservative_output_tokens - 1
            ) * cluster.decode_tpot_for(
                self._prospective_sequence_count(request, replica_id)
            )
        request._ravel_prefill_deadline_budget_s = max(
            0.0, remaining_budget_s
        )
        tbt_s = (
            float(request._slo_constraint[1])
            if request._request_type == LATENCY
            else None
        )
        request._vllm_priority = encode_phase_priority(
            request._ravel_prefill_deadline_budget_s,
            tbt_s,
            deferred=False,
            service_s=(
                conservative_service_s
                if request._request_type != LATENCY
                else None
            ),
        )

    def _plan_mobile_assignment(
        self,
        candidates: list[Request],
        snapshot: Optional[PlannerSnapshot] = None,
    ) -> Dict[int, tuple[int, Quote]]:
        if (
            not self._update_completion_pressure()
            and self._non_workflow_observed
        ):
            return (
                RavelNativeCenterWorkBalancedGlobalScheduler
                ._plan_mobile_assignment(self, candidates, snapshot)
            )
        return super()._plan_mobile_assignment(candidates, snapshot)


class RavelUnifiedNoLedgerGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Ablation: disable active completion virtual-service accounting."""

    completion_virtual_service_enabled = False


class RavelUnifiedBoundedLedgerGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Retain half of completed service history to stabilize region shares."""

    completion_virtual_release_fraction = 0.5


class RavelUnifiedFlowPlacementGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Ablation: flow-time tie breaking with production engine priorities."""

    flow_time_before_virtual_balance = True


class RavelUnifiedPhasePriorityGlobalScheduler(
    RavelUnifiedBalancedGlobalScheduler
):
    """Ablation: production placement with phase-aware engine priorities."""

    flow_time_before_virtual_balance = False
