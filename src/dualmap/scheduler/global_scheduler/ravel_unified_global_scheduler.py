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
    RavelNativeCenterOnTimeSetGlobalScheduler,
    RavelNativeCenterSelectedAdmissionGlobalScheduler,
    RavelNativeCenterWorkBalancedGlobalScheduler,
    RavelNativeCenterWorkYieldGlobalScheduler,
    RavelNativeCenterYieldGlobalScheduler,
    RavelNativeSLOFlowGlobalScheduler,
)


class RavelUnifiedGlobalScheduler(RavelNativeSLOFlowGlobalScheduler):
    """Production RAVEL policy for mixed and workflow-completion SLOs.

    Mixed TTFT/TBT traffic follows the profile-derived Dynamic-Chunk path.
    Multi-stage completion traffic enables protected soft admission when the
    observed arrival rate would fill sequence capacity within half of the
    tightest observed completion deadline. Mode selection uses only causal
    request state, never a workload name, future arrival, realized output
    length, or a timing-sensitive instantaneous outstanding threshold.

    Predicted misses are delayed before KV materialization while a protected
    cohort remains in the target engine, but no later than the request's own
    objective deadline. Every request is eventually submitted; this is not
    request rejection or admission dropping.
    """

    completion_virtual_service_enabled = True
    completion_virtual_release_fraction = 1.0
    # Soft admission stamps every candidate with a planning generation.
    # A lone late arrival must still be planned; otherwise protected dispatch
    # rejects its stale generation forever after the burst has drained.
    @property
    def plan_single_mobile_candidate(self) -> bool:
        return bool(getattr(self, "_soft_epoch_active", False))

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        self._soft_admission_enabled = bool(
            getattr(args, "ravel_soft_admission_enabled", True)
        )
        self._soft_release_policy = str(
            getattr(args, "ravel_soft_admission_release_policy", "slo_deadline")
        )
        if self._soft_release_policy != "slo_deadline":
            raise ValueError(
                "ravel_soft_admission_release_policy must be slo_deadline"
            )

        configured_reserve = int(
            getattr(args, "ravel_soft_admission_reserve_sequences", 0)
        )
        if configured_reserve < 0:
            raise ValueError(
                "ravel_soft_admission_reserve_sequences cannot be negative"
            )
        self._total_sequence_capacity = (
            int(self.pending_request_limit) * int(num_replicas)
        )
        # Zero means one replica-equivalent planning cohort. The value follows
        # the deployed max_num_seqs instead of a hardware-specific constant.
        self._reserve_sequences = (
            configured_reserve
            if configured_reserve > 0
            else int(self.pending_request_limit)
        )
        self._reserve_sequences = min(
            self._reserve_sequences,
            max(1, self._total_sequence_capacity - 1),
        )
        self._activation_outstanding = max(
            1, self._total_sequence_capacity - self._reserve_sequences
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
        self._soft_epoch_active = False
        self._soft_plan_generation = 0
        self.soft_admission_activations = 0
        self.soft_admission_protected = 0
        self.soft_admission_deferred = 0
        self.soft_admission_aged_releases = 0
        self.soft_admission_slack_releases = 0
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

    def _update_soft_epoch(self) -> bool:
        outstanding = self._outstanding_count()
        if outstanding <= 0:
            self._soft_epoch_active = False
            return False
        eligible = (
            self._soft_admission_enabled
            and self._workflow_completion_observed
            and not self._non_workflow_observed
            and self._arrival_pressure_observed
        )
        if eligible and not self._soft_epoch_active:
            self._soft_epoch_active = True
            self.soft_admission_activations += 1
        return eligible

    def _apply_soft_output_profile(self, request: Request) -> None:
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
        """Use the semantic mean for expected-goodput admission decisions.

        The held-out q95 remains recorded for diagnostics. Using it as a hard
        feasibility demand rejected most long-stage requests before the
        active-slot externality was considered. End-to-end residuals still
        record model error after completion.
        """

        profiled_expected = int(
            getattr(request, "_ravel_profile_output_expected", 0)
        )
        if self._update_soft_epoch() and profiled_expected > 0:
            expected = profiled_expected
            request._routing_output_tokens_hint_used = expected
            request._routing_output_tokens_upper_hint_used = expected
            return expected, expected
        return super()._output_token_bounds(request)

    async def _enqueue(self, request: Request) -> None:
        if self._is_workflow_completion(request):
            self._workflow_completion_observed = True
            self._observe_completion_arrival(request)
            self._apply_soft_output_profile(request)
        else:
            self._non_workflow_observed = True
        # Apply the semantic profile once, then reuse common queue accounting.
        await RavelNativeCenterSelectedAdmissionGlobalScheduler._enqueue(
            self, request
        )
        if (
            self.completion_virtual_service_enabled
            and hasattr(request, "_ravel_service_virtual_owner")
        ):
            self._completion_virtual_requests[int(request._id)] = request
        self._update_soft_epoch()

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
            if self._update_soft_epoch()
            else self.mobile_candidate_limit
        )
        return rows[:limit]

    def _plan_mobile_assignment(
        self,
        candidates: list[Request],
        snapshot: Optional[PlannerSnapshot] = None,
    ) -> Dict[int, tuple[int, Quote]]:
        if not self._update_soft_epoch():
            if self._non_workflow_observed:
                return (
                    RavelNativeCenterWorkYieldGlobalScheduler
                    ._plan_mobile_assignment(self, candidates, snapshot)
                )
            return RavelNativeSLOFlowGlobalScheduler._plan_mobile_assignment(
                self, candidates, snapshot
            )

        for request in candidates:
            self._apply_soft_output_profile(request)
        assignment = (
            RavelNativeCenterOnTimeSetGlobalScheduler
            ._plan_mobile_assignment(self, candidates, snapshot)
        )
        self._soft_plan_generation += 1
        generation = self._soft_plan_generation
        for request in candidates:
            request._ravel_soft_plan_generation = generation
            _replica_id, quote = assignment[request._id]
            request._ravel_soft_selected = bool(quote.feasible)
        return assignment

    def _soft_release_at(self, request: Request) -> float:
        return request._arrived_at + self._limit(request)

    @staticmethod
    def _engine_has_protected(replica) -> bool:
        return any(
            bool(
                getattr(
                    request, "_ravel_soft_admission_protected", False
                )
            )
            for request in (
                *replica.pending_requests,
                *replica.running_requests,
            )
        )

    def _protected_cohort_allows(
        self,
        request: Request,
        replica_id: int,
        queued: list[Request],
        now: float,
    ) -> bool:
        """Preserve released Unified's deadline-only hold semantics."""

        return False

    def _engine_standby_enabled(self) -> bool:
        return False

    async def _dispatch_protected(self) -> None:
        now = time.perf_counter()
        next_deadline = math.inf
        generation = self._soft_plan_generation

        for replica_id in self.topology.replica_ids:
            queued = self.global_request_queue.get_all_requests(replica_id)
            if not queued:
                continue
            replica = self.shared_state.replica_budgets[replica_id]
            (
                current_budget,
                _last_prefill,
                _last_ttft,
                pending_requests,
                _qps,
            ) = await replica.get_load_states()
            open_slots = max(
                0, self.pending_request_limit - pending_requests
            )
            if current_budget <= 0 or open_slots <= 0:
                continue

            eligible = []
            for request in queued:
                mobile_until = float(
                    getattr(
                        request,
                        "_ravel_mobile_until",
                        request._arrived_at,
                    )
                )
                if mobile_until > now:
                    next_deadline = min(next_deadline, mobile_until)
                    continue
                if int(
                    getattr(
                        request, "_ravel_soft_plan_generation", -1
                    )
                ) != generation:
                    continue
                eligible.append(request)

            eligible.sort(
                key=lambda request: (
                    int(
                        bool(
                            getattr(
                                request,
                                "_ravel_yield_deferred",
                                not bool(
                                    getattr(
                                        request,
                                        "_cluster_route_feasible",
                                        False,
                                    )
                                ),
                            )
                        )
                    ),
                    request._arrived_at + self._limit(request),
                    int(
                        getattr(
                            request,
                            "_ravel_profile_output_expected",
                            request._routing_output_tokens_hint_used,
                        )
                    ),
                    request._arrived_at,
                    request._id,
                )
            )

            protected_in_engine = self._engine_has_protected(replica)
            protected_in_queue = any(
                not bool(
                    getattr(
                        request,
                        "_ravel_yield_deferred",
                        not bool(
                            getattr(
                                request, "_cluster_route_feasible", False
                            )
                        ),
                    )
                )
                for request in eligible
            )
            dispatched = 0
            standby_dispatch_limit = max(
                0,
                open_slots - int(self._engine_standby_enabled()),
            )
            for request in eligible:
                if dispatched >= open_slots:
                    break
                deferred = bool(
                    getattr(
                        request,
                        "_ravel_yield_deferred",
                        not bool(
                            getattr(
                                request, "_cluster_route_feasible", False
                            )
                        ),
                    )
                )
                release_at = self._soft_release_at(request)
                aged = now >= release_at
                slack_safe = False
                if (
                    deferred
                    and (protected_in_engine or protected_in_queue)
                    and not aged
                ):
                    slack_safe = self._protected_cohort_allows(
                        request,
                        replica_id,
                        queued,
                        now,
                    )
                    if not slack_safe:
                        request._ravel_soft_admission_release_at = release_at
                        next_deadline = min(next_deadline, release_at)
                        continue
                    if dispatched >= standby_dispatch_limit:
                        continue

                request_work, hit = self._request_work(
                    request, replica_id, cold=False
                )
                if request_work > current_budget:
                    continue
                if not self.global_request_queue.del_req(
                    replica_id, request
                ):
                    continue

                initial = int(
                    getattr(
                        request, "_ravel_initial_replica", replica_id
                    )
                )
                request._rebind_count = int(replica_id != initial)
                request._ravel_soft_admission_active = True
                request._ravel_soft_admission_protected = not deferred
                request._ravel_soft_admission_release_at = release_at
                if deferred:
                    self.soft_admission_deferred += 1
                    if aged:
                        self.soft_admission_aged_releases += 1
                    if slack_safe:
                        self.soft_admission_slack_releases += 1
                        request._ravel_soft_admission_slack_release = True
                        request._cluster_route_reason = (
                            "ravel_unified_soft_slack_release"
                        )
                    else:
                        request._cluster_route_reason = (
                            "ravel_unified_soft_aged_release"
                            if aged
                            else "ravel_unified_soft_deferred_release"
                        )
                else:
                    self.soft_admission_protected += 1
                    protected_in_engine = True
                    request._cluster_route_reason = (
                        "ravel_unified_soft_protected"
                    )
                self._set_engine_priority(request)
                request._enforce_prefill_budget = True
                admitted = await self.shared_state.add_posting_request_tasks(
                    replica_id, request
                )
                if not admitted:
                    self.global_request_queue.push(
                        replica_id, request, hit
                    )
                    continue
                current_budget -= request_work
                dispatched += 1
                if not deferred:
                    protected_in_queue = any(
                        row is not request
                        and not bool(
                            getattr(
                                row,
                                "_ravel_yield_deferred",
                                not bool(
                                    getattr(
                                        row,
                                        "_cluster_route_feasible",
                                        False,
                                    )
                                ),
                            )
                        )
                        for row in eligible
                        if row in self.global_request_queue.get_all_requests(
                            replica_id
                        )
                    )
        self._arm_mobile_timer(next_deadline)

    async def _dispatch_schedulable(self) -> None:
        if not self._update_soft_epoch():
            if self._non_workflow_observed:
                await RavelNativeCenterYieldGlobalScheduler._dispatch_schedulable(
                    self
                )
                return
            await RavelNativeSLOFlowGlobalScheduler._dispatch_schedulable(
                self
            )
            return
        await self._dispatch_protected()

    async def schedule(self, new_request: Optional[Request]) -> int:
        async with self._schedule_lock:
            if new_request is not None:
                await self._enqueue(new_request)
            should_replan = (
                self._update_soft_epoch()
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
    """Latency-balanced Pareto point with the same SLO safety ordering.

    The policy preserves Unified's request-visible mode selection, dynamic
    Chunked-Prefill contract, and workflow soft admission. For mixed traffic,
    it invokes the same WorkYield planner with flow-time terms ahead of virtual
    balancing only after feasibility and lateness are tied. Completion-only
    traffic exports protected Prefill deadlines and marks predicted misses as
    opportunistic engine work, allowing local slack stealing without changing
    the Router's protected-set admission semantics.
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
        standby_service_s = 0.0
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
            standby_output_tokens = max(
                output_tokens,
                int(
                    getattr(
                        request,
                        "_ravel_profile_output_upper",
                        output_tokens,
                    )
                ),
            )
            standby_service_s = max(
                0, standby_output_tokens - 1
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
        # A predicted miss is not permanently low-priority work. Once its
        # Router hold expires, delaying it again in the engine only creates a
        # second admission queue. The wire-level deferred class is reserved
        # for work released *before* that boundary under a certified
        # protected-cohort slack check.
        deferred = bool(
            not self._non_workflow_observed
            and request._request_type != LATENCY
            and getattr(
                request, "_ravel_soft_admission_slack_release", False
            )
        )
        request._vllm_priority = encode_phase_priority(
            request._ravel_prefill_deadline_budget_s,
            tbt_s,
            deferred=deferred,
            service_s=(
                standby_service_s
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
            not self._update_soft_epoch()
            and self._non_workflow_observed
        ):
            return (
                RavelNativeCenterWorkBalancedGlobalScheduler
                ._plan_mobile_assignment(self, candidates, snapshot)
            )
        return super()._plan_mobile_assignment(candidates, snapshot)


class RavelUnifiedNoLedgerGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Ablation: pre-ledger Unified behavior on current service profiles."""

    completion_virtual_service_enabled = False


class RavelUnifiedBoundedLedgerGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Retain half of completed service history to stabilize region shares."""

    completion_virtual_release_fraction = 0.5


class RavelUnifiedAlwaysSoftGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Ablation: enable soft planning for every completion-only arrival."""

    def _update_soft_epoch(self) -> bool:
        outstanding = self._outstanding_count()
        if outstanding <= 0:
            self._soft_epoch_active = False
            return False
        eligible = (
            self._soft_admission_enabled
            and self._workflow_completion_observed
            and not self._non_workflow_observed
        )
        if eligible and not self._soft_epoch_active:
            self._soft_epoch_active = True
            self.soft_admission_activations += 1
        return eligible


class RavelUnifiedFlowPlacementGlobalScheduler(RavelUnifiedGlobalScheduler):
    """Ablation: Balanced placement with Unified engine priorities."""

    flow_time_before_virtual_balance = True


class RavelUnifiedPhasePriorityGlobalScheduler(
    RavelUnifiedBalancedGlobalScheduler
):
    """Ablation: Unified placement with Balanced engine priorities."""

    flow_time_before_virtual_balance = False


class RavelUnifiedStandbyGlobalScheduler(
    RavelUnifiedBalancedGlobalScheduler
):
    """Experimental engine-standby Pareto point.

    Predicted misses may enter the engine before their Router release time.
    This policy is intentionally separate from the validated Balanced policy:
    current experiments lower flow time but do not preserve maximum SLO
    attainment on completion-heavy traffic.
    """

    def _engine_standby_enabled(self) -> bool:
        return True

    def _protected_cohort_allows(
        self,
        request: Request,
        replica_id: int,
        queued: list[Request],
        now: float,
    ) -> bool:
        return True
