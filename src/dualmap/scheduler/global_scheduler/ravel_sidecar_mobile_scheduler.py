from __future__ import annotations

import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import math
import multiprocessing
import os
import time
from typing import Optional

from dualmap.cluster.slo import COLLECTIVE, LATENCY
from dualmap.entities.request import Request
from dualmap.logger import init_logger
from dualmap.scheduler.global_scheduler.ravel_native_global_scheduler import (
    PlannerReplicaSnapshot,
    PlannerRequestSnapshot,
    PlannerSnapshot,
    Quote,
    RavelNativeMobileSLOCompleteGlobalScheduler,
)
from dualmap.sidecar.waiting_movable import (
    SidecarWaitingMovableQueueAdapter,
    SidecarWaitingMovableRegistry,
)

logger = init_logger(__name__)

_PLANNER_EXECUTOR: ProcessPoolExecutor | None = None


def _planner_process_ready() -> bool:
    return True


def _plan_frozen_sidecar_snapshot(
    snapshot: PlannerSnapshot,
) -> dict[int, tuple[int, Quote]]:
    return (
        RavelNativeSidecarMobileSLOCompleteGlobalScheduler
        ._plan_snapshot_greedy(snapshot)
    )


def _get_sidecar_planner_executor() -> ProcessPoolExecutor:
    global _PLANNER_EXECUTOR
    if _PLANNER_EXECUTOR is None:
        _PLANNER_EXECUTOR = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        _PLANNER_EXECUTOR.submit(_planner_process_ready).result(timeout=30.0)
    return _PLANNER_EXECUTOR


class RavelNativeSidecarMobileSLOCompleteGlobalScheduler(
    RavelNativeMobileSLOCompleteGlobalScheduler
):
    """Sidecar-owned reversible placement before observed KV materialization.

    Request bodies remain in regional ``WAITING_MOVABLE`` queues. The Router
    receives primitive snapshots, computes a bounded joint assignment, and
    applies it through versioned transfer/claim operations. The mobility window
    is one measured inter-region decision RTT when the request has enough SLO
    slack; no workload label or future request is used.
    """

    mobile_hold_rtt_fraction = 1.0

    def __init__(self, num_replicas, shared_state, args):
        super().__init__(num_replicas, shared_state, args)
        registry = getattr(shared_state, "sidecar_waiting_movable", None)
        if registry is None:
            registry = SidecarWaitingMovableRegistry(num_replicas)
            shared_state.sidecar_waiting_movable = registry
        if registry.num_replicas != num_replicas:
            raise ValueError("Sidecar queue count does not match replicas")
        # Replace the empty central queue created by the legacy base class with
        # a stateless snapshot adapter. All request storage remains in registry.
        self._sidecar_waiting = registry
        self.global_request_queue = SidecarWaitingMovableQueueAdapter(registry)
        self.sidecar_transfer_attempts = 0
        self.sidecar_transfer_successes = 0
        self.sidecar_claim_conflicts = 0
        # A decision epoch may contain more than the historical fixed cohort
        # of 16 requests. Bound joint planning by the engine's configured
        # sequence capacity so this limit follows serving configuration.
        self.mobile_candidate_limit = max(
            2,
            int(self.pending_request_limit),
        )
        self._collective_stage_min = math.inf
        self._collective_stage_expected_output_tokens: dict[int, int] = {}
        self._collective_stage_upper_output_tokens: dict[int, int] = {}
        self._collective_context_expected_output_tokens: dict[
            tuple[int, int], int
        ] = {}
        self._collective_context_upper_output_tokens: dict[tuple[int, int], int] = {}
        output_profile_path = os.environ.get(
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
                raise ValueError("output profile must select COLLECTIVE requests")
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
                tuple(int(part) for part in str(context_id).split(":")): int(values["q95_tokens"])
                for context_id, values in contexts.items()
            }
        # Request-visible SLO semantics, rather than workload identity, decide
        # whether mixed-service traffic benefits from deadline ordering.
        self._latency_class_observed = False
        self._deadline_ordering_active = False
        # Unmodified vLLM has no exact Prefill-start boundary. Formal RAVEL
        # therefore moves only Sidecar-owned WAITING_MOVABLE requests.
        self.mobile_abort_hysteresis_s = math.inf
        if os.environ.get("RAVEL_SIDECAR_ENABLE_ABORT", "0") == "1":
            positive_remote_rtts = [
                cluster.rtt_s(self.topology.default_client_region)
                for cluster in self.topology.clusters
                if cluster.cluster_id != self.topology.default_client_region
                and cluster.rtt_s(self.topology.default_client_region) > 0
            ]
            self.mobile_abort_hysteresis_s = (
                min(positive_remote_rtts)
                if positive_remote_rtts
                else math.inf
            )
        # Scan every configured region pair. This is the actual bounded
        # control horizon; no legacy 100 ms constant is used.
        self.control_horizon_s = max(
            destination.rtt_s(source.cluster_id)
            for source in self.topology.clusters
            for destination in self.topology.clusters
            if source.cluster_id != destination.cluster_id
        )
        # Keep one complete vLLM scheduling quantum plus the prompt work that
        # the fastest calibrated prefill path can drain during one control RTT.
        # Excess work stays movable at the Sidecar without starving the engine.
        self._engine_buffer_tokens: dict[int, int] = {}
        if os.environ.get("RAVEL_SIDECAR_ENGINE_BUFFER", "0") == "1":
            quantum_tokens = int(getattr(args, "max_num_batched_tokens", 0))
            if quantum_tokens <= 0:
                raise ValueError("max_num_batched_tokens must be positive")
            for replica_id in self.topology.replica_ids:
                cluster = self.topology.cluster_for_replica(replica_id)
                fastest_prefill_tpot_s = float(cluster.prefill_tpot_for(0))
                if fastest_prefill_tpot_s <= 0:
                    raise ValueError("prefill service rate must be positive")
                control_cover_tokens = math.ceil(
                    self.control_horizon_s / fastest_prefill_tpot_s
                )
                self._engine_buffer_tokens[replica_id] = (
                    quantum_tokens + control_cover_tokens
                )
        self._planner_executor = _get_sidecar_planner_executor()
        self._replan_event = asyncio.Event()
        self._planner_generation = 0
        self._planner_runs = 0
        self._planner_total_s = 0.0
        self._planner_max_s = 0.0
        self._last_arrival_id = -1
        self._replan_task = asyncio.create_task(self._replan_worker())

        # Submitted to an engine but not yet observed at first token. A hard
        # move is best effort because unmodified vLLM exposes no prefill-start
        # event; attempt and version isolate all stale callbacks.
        self._in_flight: dict[
            tuple[int, int], tuple[int, object, bool]
        ] = {}
        self._aborted_attempts: dict[int, set[int]] = {}
        self._request_by_id: dict[int, Request] = {}
        shared_state.set_prefill_started_callback(self.on_prefill_started)
        shared_state.set_request_terminal_callback(self.on_request_terminal)

    def _ordered_queue(self, replica_id: int) -> list[Request]:
        entries = self._sidecar_waiting.entries(replica_id)
        return [
            entry.request
            for entry in sorted(
                entries,
                key=lambda entry: (
                    -entry.prefix_hit_tokens,
                    entry.request._arrived_at,
                    entry.request._id,
                ),
            )
        ]

    def _owner(self, request: Request) -> int:
        rid = int(request._id)
        key = (rid, int(getattr(request, "_attempt", 0)))
        if key in self._in_flight:
            return self._in_flight[key][0]
        return super()._owner(request)


    def _mobile_hold_s(self, request: Request) -> float:
        """Use the bounded SLO-aware hold defined by the base controller."""

        return super()._mobile_hold_s(request)

    _DEFERRED_PRIORITY_OFFSET_US = 1_000_000_000_000_000
    # JITServe's published service-gain model values a Decode token at eight
    # Prefill-token units. The scale below only preserves integer ordering for
    # vLLM's priority API; it is not a policy weight.
    _JITSERVE_DECODE_GAIN = 8
    _DENSITY_PRIORITY_RESOLUTION = 1_000

    def _apply_collective_stage_output_profile(
        self, request: Request
    ) -> None:
        if (
            request._request_type != COLLECTIVE
            or int(request._stage_num) < self._collective_stage_min
        ):
            return
        context = (int(request._stage_id), int(request._stage_num))
        expected = self._collective_context_expected_output_tokens.get(
            context,
            self._collective_stage_expected_output_tokens.get(context[0]),
        )
        upper = self._collective_context_upper_output_tokens.get(
            context,
            self._collective_stage_upper_output_tokens.get(context[0]),
        )
        if expected is None or upper is None:
            return
        request._routing_output_tokens_hint = expected
        request._routing_output_tokens_hint_used = expected
        request._routing_output_tokens_upper_hint = max(expected, upper)
        request._routing_output_tokens_upper_hint_used = max(
            expected, upper
        )

    def _within_fast_recourse_window(self, request: Request) -> bool:
        """Return whether the latest same-region arrival is still reversible."""

        arrivals = [
            timestamp
            for timestamp, region in self._recent_arrivals[-2:]
            if region == request._client_region
        ]
        if len(arrivals) < 2:
            return False
        remote_rtts = [
            cluster.rtt_s(request._client_region)
            for cluster in self.topology.clusters
            if cluster.cluster_id != request._client_region
            and cluster.rtt_s(request._client_region) > 0
        ]
        if not remote_rtts:
            return False
        return arrivals[-1] - arrivals[-2] <= min(remote_rtts)

    def _set_engine_priority(self, request: Request) -> None:
        """Encode the selected engine ordering without tuned weights."""

        mode = os.environ.get(
            "RAVEL_ENGINE_PRIORITY_MODE", "selected_edf"
        )
        if mode == "slo_semantic_adaptive":
            if not self._deadline_ordering_active:
                request._vllm_priority = 0
                return
            mode = "edf"
        if mode == "jitserve_density":
            replica_id = int(request._primary_replica)
            replica = self.shared_state.replica_budgets[replica_id]
            cluster = self.topology.cluster_for_replica(replica_id)
            prompt_tokens = max(1, int(request._num_prefill_tokens))
            output_tokens = max(
                1,
                int(request._routing_output_tokens_hint_used),
            )
            decode_intervals = max(0, output_tokens - 1)
            own_service_s = (
                prompt_tokens
                * cluster.prefill_tpot_for(replica.get_num_running_req())
                + decode_intervals
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
                service_gain * deadline_discount / max(own_service_s, 1e-9)
            )
            request._vllm_priority = -int(
                round(density * self._DENSITY_PRIORITY_RESOLUTION)
            )
            return
        if mode == "latency_triggered_edf":
            if not self._latency_class_observed:
                request._vllm_priority = 0
                return
            mode = "edf"
        if mode == "slo_yield":
            limit_s = max(1e-9, self._limit(request))
            predicted_s = max(
                0.0,
                float(getattr(request, "_predicted_objective_s", limit_s)),
            )
            # Integer scaling controls only score resolution. Ordering is the
            # dimensionless predicted SLO consumption predicted_s / limit_s.
            request._vllm_priority = int(
                predicted_s / limit_s * 1_000_000_000
            )
            return
        if mode not in {"edf", "selected_edf"}:
            raise ValueError(f"unknown engine priority mode: {mode}")

        deadline_us = max(
            0,
            int((request._arrived_at + self._limit(request)) * 1_000_000),
        )
        deferred = mode == "selected_edf" and bool(
            getattr(
                request,
                "_ravel_yield_deferred",
                not bool(getattr(request, "_cluster_route_feasible", True)),
            )
        )
        request._vllm_priority = deadline_us + (
            self._DEFERRED_PRIORITY_OFFSET_US if deferred else 0
        )

    async def _enqueue(self, request: Request) -> None:
        self._apply_collective_stage_output_profile(request)
        await super()._enqueue(request)
        if (
            request._request_type != COLLECTIVE
            and self._within_fast_recourse_window(request)
        ):
            self._deadline_ordering_active = True
        if request._request_type == LATENCY:
            self._latency_class_observed = True
        self._set_engine_priority(request)
    def _movable_in_flight_count_by_replica(
        self, candidate_ids: set[int]
    ) -> dict[int, int]:
        counts: dict[int, int] = {}
        for (rid, _attempt), (replica_id, _claimed, _aborting) in self._in_flight.items():
            if rid not in candidate_ids:
                continue
            counts[replica_id] = counts.get(replica_id, 0) + 1
        return counts

    def _movable_in_flight_work_by_replica(
        self, candidate_ids: set[int]
    ) -> dict[int, float]:
        work_by_replica: dict[int, float] = {}
        for (rid, _attempt), (replica_id, claimed, _aborting) in self._in_flight.items():
            if rid not in candidate_ids:
                continue
            request_work, _hit = self._request_work(
                claimed.request, replica_id, cold=False
            )
            work_by_replica[replica_id] = (
                work_by_replica.get(replica_id, 0.0) + float(request_work)
            )
        return work_by_replica

    def _mobile_candidates(self) -> list[Request]:
        by_id: dict[int, Request] = {}
        for replica_id in self.topology.replica_ids:
            for row in self.global_request_queue.get_all_requests(replica_id):
                by_id[int(row._id)] = row

        rows = sorted(
            by_id.values(),
            key=lambda row: (
                row._arrived_at + self._limit(row),
                row._arrived_at,
                row._id,
            ),
        )
        selected = rows[: self.mobile_candidate_limit]
        latest = by_id.get(self._last_arrival_id)
        if latest is not None and latest not in selected:
            if len(selected) >= self.mobile_candidate_limit:
                selected[-1] = latest
            else:
                selected.append(latest)
            selected.sort(
                key=lambda row: (
                    row._arrived_at + self._limit(row),
                    row._arrived_at,
                    row._id,
                )
            )
        return selected

    def _hard_rebind_candidates(self) -> list[Request]:
        """Return submitted requests that remain before the observed KV boundary.

        These requests are deliberately excluded from ``_mobile_candidates``:
        engine-pending recourse must not perturb the proven Sidecar soft-placement
        cohort. They enter a separate rescue pass only after one measured remote
        RTT, and each request can be rebound at most once.
        """

        if (
            os.environ.get("RAVEL_ENGINE_PRIORITY_MODE")
            == "latency_triggered_edf"
            and self._latency_class_observed
        ):
            return []
        if not math.isfinite(self.mobile_abort_hysteresis_s):
            return []
        now = time.perf_counter()
        rows = []
        for (_request_id, attempt), (
            _replica_id,
            claimed,
            aborting,
        ) in self._in_flight.items():
            row = claimed.request
            if (
                aborting
                or int(getattr(row, "_attempt", 0)) != attempt
                or int(getattr(row, "_rebind_count", 0)) >= 1
                or now
                - float(getattr(row, "_sidecar_dispatched_at", now))
                < self.mobile_abort_hysteresis_s
            ):
                continue
            rows.append(row)
        rows.sort(key=self._edf_key)
        return rows[: self.mobile_candidate_limit]

    @staticmethod
    def _quote_with_transfer_cost(
        quote: Quote,
        transfer_s: float,
        limit_s: float,
    ) -> Quote:
        """Charge a hard rebind for the measured source-to-target RTT."""

        transfer = max(0.0, float(transfer_s))
        objective = quote.predicted_objective_s + transfer
        return replace(
            quote,
            predicted_ttft_s=quote.predicted_ttft_s + transfer,
            point_ttft_s=quote.point_ttft_s + transfer,
            predicted_objective_s=objective,
            point_objective_s=quote.point_objective_s + transfer,
            feasible=quote.feasible and objective <= limit_s,
        )

    def _plan_hard_slo_rescues(
        self,
        snapshot: PlannerSnapshot,
    ) -> dict[int, tuple[int, Quote]]:
        """Move only requests whose visible SLO feasibility can be restored.

        The engine state in ``snapshot`` excludes every candidate exactly once.
        We add the current candidate owners back as shadow pending work, then
        evaluate each request sequentially. A move is legal only when the source
        is infeasible and a destination remains feasible after the measured
        inter-region transfer RTT. This prevents objective-only churn.
        """

        replica_index = {
            replica.replica_id: index
            for index, replica in enumerate(snapshot.replicas)
        }
        owners = {
            request.request_id: request.owner_replica
            for request in snapshot.requests
        }
        assignment: dict[int, tuple[int, Quote]] = {}

        def quote_at(
            request: PlannerRequestSnapshot,
            target: int,
        ) -> Quote:
            other = [
                row
                for row in snapshot.requests
                if row.request_id != request.request_id
                and owners[row.request_id] == target
            ]
            index = replica_index[target]
            return self._planner_quote(
                snapshot,
                request,
                snapshot.replicas[index],
                index,
                sum(row.prompt_tokens for row in other),
                len(other),
            )

        moved = 0
        for request in sorted(
            snapshot.requests,
            key=lambda row: (
                row.deadline_at,
                row.arrived_at,
                row.request_id,
            ),
        ):
            source = owners[request.request_id]
            source_quote = quote_at(request, source)
            if source_quote.feasible:
                continue

            source_cluster = self.topology.cluster_for_replica(source)
            edges = []
            for target in replica_index:
                if target == source:
                    continue
                target_quote = quote_at(request, target)
                target_cluster = self.topology.cluster_for_replica(target)
                effective = self._quote_with_transfer_cost(
                    target_quote,
                    target_cluster.rtt_s(source_cluster.cluster_id),
                    request.limit_s,
                )
                if not effective.feasible:
                    continue
                edges.append(
                    (
                        effective.predicted_objective_s
                        / max(1e-9, request.limit_s),
                        effective.predicted_objective_s,
                        target,
                        effective,
                    )
                )
            if not edges:
                continue
            choice = min(edges)
            target = int(choice[2])
            assignment[request.request_id] = (target, choice[3])
            owners[request.request_id] = target
            moved += 1
            if moved >= self.max_rebalances_per_event:
                break
        return assignment

    async def _apply_hard_slo_rescues(self) -> int:
        candidates = self._hard_rebind_candidates()
        if not candidates:
            return 0
        self._planner_generation += 1
        snapshot = self._build_planner_snapshot(candidates)
        if not snapshot.requests:
            return 0
        assignment = self._plan_hard_slo_rescues(snapshot)
        if not assignment:
            return 0
        return await self._apply_mobile_assignment(
            candidates,
            assignment,
            snapshot,
        )

    @classmethod
    def _plan_snapshot_greedy(
        cls,
        snapshot: PlannerSnapshot,
    ) -> dict[int, tuple[int, Quote]]:
        """Maximize predicted on-time completions before tardy work.

        This is a parallel-machine Moore-Hodgson adaptation. Requests enter an
        EDF on-time set. When a request cannot meet its SLO, it may replace a
        longer accepted request if that preserves predicted yield while
        reducing occupied service. Tardy requests are still served, after the
        on-time set. Inputs are limited to visible SLOs and measured profiles.
        Runtime is O(K^2 R) for a K-request cohort.
        """

        replica_count = len(snapshot.replicas)
        base_pressure = [
            replica.pending_requests
            + replica.running_requests
            + replica.fixed_queue_count
            for replica in snapshot.replicas
        ]
        accepted: list[
            list[tuple[PlannerRequestSnapshot, Quote, float]]
        ] = [[] for _ in range(replica_count)]
        accepted_work = [0] * replica_count
        accepted_count = [0] * replica_count
        deferred: list[PlannerRequestSnapshot] = []
        assignment: dict[int, tuple[int, Quote]] = {}

        def service_size(
            request: PlannerRequestSnapshot,
            replica: PlannerReplicaSnapshot,
            _quote: Quote,
        ) -> float:
            # Moore-Hodgson compares each job's own processing requirement.
            # Queue and engine work affect completion feasibility, but are
            # common predecessor work and must not inflate the victim size.
            size = (
                replica.prefill_intercept_s
                + request.prompt_tokens * replica.prefill_tpot_s
            )
            size -= min(size, _quote.prefix_benefit_s)
            if request.request_type != LATENCY:
                decode_curve = replica.decode_tpot_by_sequence_s
                decode_index = min(
                    max(0, replica.running_requests + 1),
                    len(decode_curve) - 1,
                )
                size += (
                    max(0, request.output_tokens_hint - 1)
                    * decode_curve[decode_index]
                )
            return max(0.0, float(size))

        for request in snapshot.requests:
            limit_s = max(1e-9, request.limit_s)
            edges = []
            for index, replica in enumerate(snapshot.replicas):
                moved = replica.replica_id != request.owner_replica
                if moved and request.rebind_count >= 1:
                    continue
                quote = cls._planner_quote(
                    snapshot,
                    request,
                    replica,
                    index,
                    accepted_work[index],
                    accepted_count[index],
                )
                edges.append(
                    (
                        int(not quote.feasible),
                        max(
                            0.0,
                            quote.predicted_objective_s - request.limit_s,
                        )
                        / limit_s,
                        max(
                            0.0,
                            quote.point_objective_s
                            - quote.prefix_benefit_s,
                        )
                        / limit_s,
                        int(moved),
                        base_pressure[index] + accepted_count[index],
                        replica.replica_id,
                        index,
                        quote,
                    )
                )
            if not edges:
                raise RuntimeError(
                    f"planner has no legal placement for request "
                    f"{request.request_id}"
                )

            chosen = min(edges)
            index = chosen[-2]
            quote = chosen[-1]
            if quote.feasible:
                size = service_size(
                    request, snapshot.replicas[index], quote
                )
                accepted[index].append((request, quote, size))
                accepted_work[index] += request.prompt_tokens
                accepted_count[index] += 1
                assignment[request.request_id] = (
                    snapshot.replicas[index].replica_id,
                    quote,
                )
                continue

            replacements = []
            for replacement_index, replica in enumerate(
                snapshot.replicas
            ):
                moved = replica.replica_id != request.owner_replica
                if moved and request.rebind_count >= 1:
                    continue
                for victim_offset, (
                    victim,
                    _victim_quote,
                    victim_size,
                ) in enumerate(accepted[replacement_index]):
                    candidate_work = (
                        accepted_work[replacement_index]
                        - victim.prompt_tokens
                    )
                    candidate_count = (
                        accepted_count[replacement_index] - 1
                    )
                    candidate_quote = cls._planner_quote(
                        snapshot,
                        request,
                        replica,
                        replacement_index,
                        candidate_work,
                        candidate_count,
                    )
                    candidate_size = service_size(
                        request, replica, candidate_quote
                    )
                    if (
                        not candidate_quote.feasible
                        or candidate_size >= victim_size
                    ):
                        continue
                    replacements.append(
                        (
                            candidate_size - victim_size,
                            max(
                                0.0,
                                candidate_quote.point_objective_s
                                - candidate_quote.prefix_benefit_s,
                            )
                            / limit_s,
                            int(moved),
                            replica.replica_id,
                            replacement_index,
                            victim_offset,
                            victim.request_id,
                            victim,
                            candidate_quote,
                            candidate_size,
                        )
                    )
            if not replacements:
                deferred.append(request)
                continue

            replacement = min(replacements)
            replacement_index = replacement[4]
            victim_offset = replacement[5]
            victim = replacement[7]
            candidate_quote = replacement[8]
            candidate_size = replacement[9]
            accepted[replacement_index].pop(victim_offset)
            accepted_work[replacement_index] += (
                request.prompt_tokens - victim.prompt_tokens
            )
            accepted[replacement_index].append(
                (request, candidate_quote, candidate_size)
            )
            assignment.pop(victim.request_id, None)
            deferred.append(victim)
            assignment[request.request_id] = (
                snapshot.replicas[replacement_index].replica_id,
                candidate_quote,
            )

        deferred_work = [0] * replica_count
        deferred_count = [0] * replica_count
        for request in deferred:
            edges = []
            for index, replica in enumerate(snapshot.replicas):
                moved = replica.replica_id != request.owner_replica
                if moved and request.rebind_count >= 1:
                    continue
                quote = cls._planner_quote(
                    snapshot,
                    request,
                    replica,
                    index,
                    accepted_work[index] + deferred_work[index],
                    accepted_count[index] + deferred_count[index],
                )
                edges.append(
                    (
                        accepted_work[index] + deferred_work[index],
                        base_pressure[index]
                        + accepted_count[index]
                        + deferred_count[index],
                        max(
                            0.0,
                            quote.point_objective_s
                            - quote.prefix_benefit_s,
                        )
                        / max(1e-9, request.limit_s),
                        int(moved),
                        replica.replica_id,
                        index,
                        quote,
                    )
                )
            chosen = min(edges)
            index = chosen[-2]
            quote = replace(chosen[-1], feasible=False)
            deferred_work[index] += request.prompt_tokens
            deferred_count[index] += 1
            assignment[request.request_id] = (
                snapshot.replicas[index].replica_id,
                quote,
            )

        return assignment

    def _plan_mobile_assignment(
        self,
        candidates: list[Request],
        snapshot: Optional[PlannerSnapshot] = None,
    ) -> dict[int, tuple[int, Quote]]:
        if snapshot is not None:
            return self._plan_snapshot_greedy(snapshot)
        return super()._plan_mobile_assignment(candidates, snapshot)

    def _build_planner_snapshot(
        self,
        candidates: list[Request],
    ) -> PlannerSnapshot:
        """Freeze visible Sidecar queues and engine state for one decision."""

        captured_at = time.perf_counter()
        replica_ids = tuple(self.topology.replica_ids)
        located: list[tuple[Request, int, int, bool]] = []
        for row in candidates:
            request_id = int(row._id)
            owner = self._sidecar_waiting.owner(request_id)
            if owner is not None:
                entry = self._sidecar_waiting.entry(owner, request_id)
                if entry is not None:
                    located.append(
                        (row, int(owner), int(entry.version), False)
                    )
                continue

            attempt = int(getattr(row, "_attempt", 0))
            in_flight = self._in_flight.get((request_id, attempt))
            if in_flight is None or in_flight[2]:
                continue
            replica_id, claimed, _aborting = in_flight
            located.append(
                (row, int(replica_id), int(claimed.version), True)
            )

        candidate_ids = {
            int(row._id)
            for row, _owner, _version, _in_flight in located
        }
        fixed_queue: dict[int, tuple[Request, ...]] = {}
        for replica_id in replica_ids:
            fixed_queue[replica_id] = tuple(
                entry.request
                for entry in self._sidecar_waiting.entries(replica_id)
                if int(entry.request._id) not in candidate_ids
            )

        movable_in_flight_count = (
            self._movable_in_flight_count_by_replica(candidate_ids)
        )
        movable_in_flight_work = (
            self._movable_in_flight_work_by_replica(candidate_ids)
        )
        replica_snapshots = []
        for replica_id in replica_ids:
            replica = self.shared_state.replica_budgets[replica_id]
            running = int(replica.get_num_running_req())
            pending = max(
                0,
                int(replica.get_num_pending_req())
                - movable_in_flight_count.get(replica_id, 0),
            )
            engine_work = max(
                0,
                int(
                    self._engine_work(replica_id)
                    - movable_in_flight_work.get(replica_id, 0.0)
                ),
            )
            outstanding_rows = [
                *replica.pending_requests,
                *replica.running_requests,
                *fixed_queue[replica_id],
            ]
            outstanding = {
                int(row._id): row
                for row in outstanding_rows
                if int(row._id) not in candidate_ids
            }
            latency_requests = sum(
                int(getattr(row, "_request_type", -1)) == LATENCY
                for row in outstanding.values()
            )
            flexible_requests = len(outstanding) - latency_requests
            cluster = self.topology.cluster_for_replica(replica_id)
            max_sequences = (
                pending
                + running
                + len(fixed_queue[replica_id])
                + len(located)
                + 1
            )
            replica_snapshots.append(
                PlannerReplicaSnapshot(
                    replica_id=int(replica_id),
                    cluster_id=str(cluster.cluster_id),
                    engine_work_tokens=engine_work,
                    pending_requests=pending,
                    running_requests=running,
                    fixed_queue_count=len(fixed_queue[replica_id]),
                    fixed_queue_work_tokens=sum(
                        max(1, int(row._num_prefill_tokens))
                        for row in fixed_queue[replica_id]
                    ),
                    latency_requests=latency_requests,
                    flexible_requests=flexible_requests,
                    prefill_tpot_s=float(
                        cluster.prefill_tpot_for(running)
                    ),
                    prefill_intercept_s=float(
                        cluster.prefill_intercept_for(running)
                    ),
                    decode_tpot_by_sequence_s=tuple(
                        float(cluster.decode_tpot_for(count))
                        for count in range(max_sequences + 1)
                    ),
                )
            )

        request_snapshots = []
        for row, owner, version, is_in_flight in located:
            request_id = int(row._id)
            row_key = self._edf_key(row)
            fixed_work_before = []
            fixed_requests_before = []
            rtts = []
            prefix_hits = []
            residual_guards = []
            decision_guards = []
            risk_ready = []
            for replica_id in replica_ids:
                preceding = [
                    fixed
                    for fixed in fixed_queue[replica_id]
                    if self._edf_key(fixed) <= row_key
                ]
                fixed_work_before.append(
                    sum(
                        max(1, int(fixed._num_prefill_tokens))
                        for fixed in preceding
                    )
                )
                fixed_requests_before.append(len(preceding))
                cluster = self.topology.cluster_for_replica(replica_id)
                rtts.append(
                    float(cluster.rtt_s(row._client_region))
                )
                _work, hit = self._request_work(
                    row,
                    replica_id,
                    cold=False,
                )
                prefix_hits.append(int(hit))
                residual, calibrated = self._residual_bound(
                    replica_id,
                    row._request_type,
                )
                residual_guards.append(float(residual))
                decision_guards.append(
                    float(residual)
                    if self.residual_decision_enabled
                    else 0.0
                )
                risk_ready.append(
                    bool(
                        calibrated
                        or not self.residual_decision_enabled
                    )
                )
            output_hint, output_upper_hint = self._output_token_bounds(
                row
            )
            limit_s = float(self._limit(row))
            request_snapshots.append(
                PlannerRequestSnapshot(
                    request_id=request_id,
                    request_type=int(row._request_type),
                    client_region=str(row._client_region),
                    arrived_at=float(row._arrived_at),
                    deadline_at=float(row._arrived_at) + limit_s,
                    limit_s=limit_s,
                    ttft_limit_s=float(row._slo_constraint[0]),
                    tbt_limit_s=float(row._slo_constraint[1]),
                    prompt_tokens=max(
                        1,
                        int(row._num_prefill_tokens),
                    ),
                    output_tokens_hint=max(1, output_hint),
                    output_tokens_upper_hint=max(
                        1, output_upper_hint
                    ),
                    owner_replica=owner,
                    initial_replica=int(
                        getattr(
                            row,
                            "_ravel_initial_replica",
                            owner,
                        )
                    ),
                    owner_version=version,
                    attempt=int(getattr(row, "_attempt", 0)),
                    rebind_count=int(
                        getattr(row, "_rebind_count", 0)
                    ),
                    in_flight=is_in_flight,
                    # max_num_seqs limits concurrent execution inside vLLM;
                    # it is not a bound on the engine's waiting queue.
                    enforce_admission_limit=False,
                    rtt_by_replica_s=tuple(rtts),
                    prefix_hit_by_replica=tuple(prefix_hits),
                    residual_guard_by_replica_s=tuple(
                        residual_guards
                    ),
                    decision_guard_by_replica_s=tuple(
                        decision_guards
                    ),
                    risk_ready_by_replica=tuple(risk_ready),
                    fixed_work_before_by_replica=tuple(
                        fixed_work_before
                    ),
                    fixed_requests_before_by_replica=tuple(
                        fixed_requests_before
                    ),
                )
            )

        return PlannerSnapshot(
            captured_at=captured_at,
            replicas=tuple(replica_snapshots),
            requests=tuple(request_snapshots),
            pending_request_limit=int(self.pending_request_limit),
            beam_width=int(self.mobile_beam_width),
            control_rebind_cost_s=self.control_horizon_s,
        )

    async def _rebind_in_flight(
        self,
        row: Request,
        planned: PlannerRequestSnapshot,
        target: int,
        target_quote: Quote,
        snapshot: PlannerSnapshot,
    ) -> bool:
        """Atomically cancel and requeue one pre-first-token attempt."""

        source = int(planned.owner_replica)
        if target == source or planned.rebind_count >= 1:
            return False
        key = (int(planned.request_id), int(planned.attempt))
        current = self._in_flight.get(key)
        if current is None:
            self.sidecar_claim_conflicts += 1
            return False
        current_replica, claimed, aborting = current
        if (
            aborting
            or int(current_replica) != source
            or int(claimed.version) != planned.owner_version
            or int(getattr(row, "_attempt", 0)) != planned.attempt
        ):
            self.sidecar_claim_conflicts += 1
            return False

        if (
            not target_quote.feasible
            or target_quote.predicted_objective_s > self._limit(row)
        ):
            return False

        # CAS to ABORTING before yielding to cancellation callbacks.
        self._in_flight[key] = (source, claimed, True)
        self.sidecar_transfer_attempts += 1
        row._abort_attempts = int(getattr(row, "_abort_attempts", 0)) + 1
        self._aborted_attempts.setdefault(int(row._id), set()).add(
            int(planned.attempt)
        )
        aborted = await self.shared_state.abort_in_flight(
            row,
            source,
            int(planned.attempt),
        )
        if not aborted:
            aborted_attempts = self._aborted_attempts.get(int(row._id))
            if aborted_attempts is not None:
                aborted_attempts.discard(int(planned.attempt))
                if not aborted_attempts:
                    del self._aborted_attempts[int(row._id)]
            self._in_flight.pop(key, None)
            self._sidecar_waiting.mark_prefill_locked(
                source,
                row,
                expected_version=int(claimed.version),
            )
            self.sidecar_claim_conflicts += 1
            return False

        self._in_flight.pop(key, None)
        _, hit = self._request_work(row, target, cold=False)
        self._sidecar_waiting.restore(target, claimed, hit)
        row._attempt = int(planned.attempt) + 1
        row._rebind_count = int(getattr(row, "_rebind_count", 0)) + 1
        row._ravel_mobile_until = snapshot.captured_at
        row._ravel_yield_defer_until = row._arrived_at
        row._ravel_planned_generation = self._planner_generation
        row._cluster_route_reason = "ravel_sidecar_engine_pending_rebind"
        self.sidecar_transfer_successes += 1
        return True

    async def _apply_mobile_assignment(
        self,
        candidates: list[Request],
        assignment: dict[int, tuple[int, Quote]],
        snapshot: PlannerSnapshot,
    ) -> int:
        """Apply an expired plan through versioned Sidecar transfers."""

        moved = 0
        rows = {int(row._id): row for row in candidates}
        replica_index = {
            replica.replica_id: index
            for index, replica in enumerate(snapshot.replicas)
        }
        for planned in snapshot.requests:
            row = rows.get(planned.request_id)
            decision = assignment.get(planned.request_id)
            if row is None or decision is None:
                continue
            mobile_deadline = float(
                getattr(row, "_ravel_mobile_until", row._arrived_at)
            )
            if snapshot.captured_at >= mobile_deadline:
                row._ravel_planned_generation = self._planner_generation
            if int(getattr(row, "_attempt", 0)) != planned.attempt:
                self.sidecar_claim_conflicts += 1
                continue
            source = planned.owner_replica
            target, target_quote = decision
            if target == source or planned.rebind_count >= 1:
                continue
            if snapshot.captured_at < float(
                getattr(
                    row,
                    "_ravel_mobile_until",
                    row._arrived_at,
                )
            ):
                continue
            if planned.in_flight:
                if await self._rebind_in_flight(
                    row,
                    planned,
                    int(target),
                    target_quote,
                    snapshot,
                ):
                    moved += 1
                continue

            current_owner = self._sidecar_waiting.owner(
                planned.request_id
            )
            entry = (
                self._sidecar_waiting.entry(
                    source,
                    planned.request_id,
                )
                if current_owner == source
                else None
            )
            if (
                entry is None
                or int(entry.version) != planned.owner_version
            ):
                self.sidecar_claim_conflicts += 1
                continue
            _, hit = self._request_work(
                row,
                target,
                cold=False,
            )
            self.sidecar_transfer_attempts += 1
            transferred = self._sidecar_waiting.transfer(
                source,
                target,
                planned.request_id,
                expected_version=planned.owner_version,
                prefix_hit_tokens=hit,
            )
            if transferred is None:
                self.sidecar_claim_conflicts += 1
                continue
            self.sidecar_transfer_successes += 1
            row._rebind_count = (
                int(getattr(row, "_rebind_count", 0)) + 1
            )
            row._ravel_soft_moves = (
                int(getattr(row, "_ravel_soft_moves", 0)) + 1
            )
            row._cluster_route_reason = (
                "ravel_sidecar_mobile_transfer"
            )
            moved += 1

        for planned in snapshot.requests:
            row = rows.get(planned.request_id)
            decision = assignment.get(planned.request_id)
            if row is None or decision is None:
                continue
            planned_target, planned_quote = decision
            actual = self._sidecar_waiting.owner(
                planned.request_id
            )
            if actual is None:
                continue
            if int(actual) == planned_target:
                quote = planned_quote
            else:
                index = replica_index[int(actual)]
                quote = self._planner_quote(
                    snapshot,
                    planned,
                    snapshot.replicas[index],
                    index,
                    0,
                    0,
                )
            row._routing_output_tokens_hint_used = (
                planned.output_tokens_hint
            )
            row._routing_output_tokens_upper_hint_used = (
                planned.output_tokens_upper_hint
            )
            self._record_quote(row, quote)
            row._ravel_yield_deferred = not quote.feasible
            if quote.feasible:
                row._ravel_yield_defer_until = row._arrived_at
            elif not hasattr(row, "_ravel_yield_defer_until"):
                # Tardy work yields one measured control RTT, then runs.
                row._ravel_yield_defer_until = (
                    snapshot.captured_at + self.control_horizon_s
                )

        self.direct_rebalance_count += moved
        return moved
    async def _replan_worker(self) -> None:
        """Coalesced snapshot-plan-CAS loop; no fixed polling interval."""

        while True:
            await self._replan_event.wait()
            self._replan_event.clear()
            try:
                async with self._schedule_lock:
                    candidates = self._mobile_candidates()
                    if len(candidates) < 2:
                        await self._apply_hard_slo_rescues()
                        await self._dispatch_schedulable(
                            allow_unplanned_mobile=True
                        )
                        continue
                    self._planner_generation += 1
                    snapshot = self._build_planner_snapshot(candidates)
                if len(snapshot.requests) < 2:
                    async with self._schedule_lock:
                        await self._apply_hard_slo_rescues()
                        await self._dispatch_schedulable(
                            allow_unplanned_mobile=True
                        )
                    continue
                started_at = time.perf_counter()
                loop = asyncio.get_running_loop()
                assignment = await loop.run_in_executor(
                    self._planner_executor,
                    _plan_frozen_sidecar_snapshot,
                    snapshot,
                )
                planning_s = time.perf_counter() - started_at
                self._planner_runs += 1
                self._planner_total_s += planning_s
                self._planner_max_s = max(self._planner_max_s, planning_s)
                async with self._schedule_lock:
                    await self._apply_mobile_assignment(
                        candidates,
                        assignment,
                        snapshot,
                    )
                    await self._apply_hard_slo_rescues()
                    await self._dispatch_schedulable()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Sidecar replan worker failed")

    async def schedule(self, new_request: Optional[Request]) -> int:
        if new_request is not None:
            async with self._schedule_lock:
                await self._enqueue(new_request)
                self._last_arrival_id = int(new_request._id)
                await self._dispatch_schedulable()
        self._replan_event.set()
        return -1

    async def _wake_mobile_timer(self, deadline: float) -> None:
        try:
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        except asyncio.CancelledError:
            return
        self._mobile_timer = None
        self._mobile_timer_deadline = math.inf
        # Replan before committing expired rows. Dispatching first collapses the
        # WAITING_MOVABLE window and turns every useful move into an HTTP abort.
        self._replan_event.set()

    def _dispatch_priority(self, entry) -> tuple:
        """Serve the predicted on-time set in EDF order before tardy work."""

        row = entry.request
        return (
            int(bool(getattr(row, "_ravel_yield_deferred", False))),
            row._arrived_at + self._limit(row),
            row._arrived_at,
            row._id,
        )

    async def _dispatch_schedulable(
        self, *, allow_unplanned_mobile: bool = False
    ) -> None:
        now = time.perf_counter()
        next_deadline = math.inf
        needs_replan = False
        for replica_id in self.topology.replica_ids:
            entries = self._sidecar_waiting.entries(replica_id)
            if not entries:
                continue
            replica = self.shared_state.replica_budgets[replica_id]
            current_budget, _, _, _pending_requests, _ = (
                await replica.get_load_states()
            )
            buffer_target = self._engine_buffer_tokens.get(replica_id)
            if buffer_target is not None:
                current_budget = min(
                    current_budget,
                    max(0, buffer_target - self._engine_work(replica_id)),
                )
            if current_budget <= 0:
                continue
            eligible = []
            for entry in entries:
                row = entry.request
                mobile_deadline = float(
                    getattr(row, "_ravel_mobile_until", row._arrived_at)
                )
                yield_deadline = float(
                    getattr(
                        row,
                        "_ravel_yield_defer_until",
                        row._arrived_at,
                    )
                )
                deadline = max(mobile_deadline, yield_deadline)
                requires_plan = float(
                    getattr(row, "_ravel_mobile_hold_s", 0.0)
                ) > 0.0
                is_planned = int(
                    getattr(row, "_ravel_planned_generation", -1)
                ) >= 0
                if deadline <= now:
                    if (
                        requires_plan
                        and not is_planned
                        and not allow_unplanned_mobile
                    ):
                        needs_replan = True
                        continue
                    eligible.append(entry)
                else:
                    next_deadline = min(next_deadline, deadline)
            eligible.sort(key=self._dispatch_priority)
            for entry in eligible:
                row = entry.request
                request_work, hit = self._request_work(
                    row,
                    replica_id,
                    cold=False,
                )
                if request_work > current_budget:
                    continue
                claimed = self._sidecar_waiting.claim(
                    replica_id,
                    row._id,
                    expected_version=entry.version,
                )
                if claimed is None:
                    self.sidecar_claim_conflicts += 1
                    continue
                initial = int(
                    getattr(row, "_ravel_initial_replica", replica_id)
                )
                row._rebind_count = max(
                    int(getattr(row, "_rebind_count", 0)),
                    int(replica_id != initial),
                )
                if row._rebind_count:
                    row._cluster_route_reason = (
                        "ravel_sidecar_mobile_committed_rebind"
                    )
                elif int(getattr(row, "_ravel_soft_moves", 0)) > 0:
                    row._cluster_route_reason = (
                        "ravel_sidecar_mobile_committed_restored"
                    )
                self._set_engine_priority(row)
                row._enforce_prefill_budget = True
                try:
                    admitted = await self.shared_state.add_posting_request_tasks(
                        replica_id,
                        row,
                    )
                except Exception:
                    self._sidecar_waiting.restore(replica_id, claimed, hit)
                    raise
                if not admitted:
                    self._sidecar_waiting.restore(replica_id, claimed, hit)
                    continue
                row._sidecar_dispatched_at = time.perf_counter()
                self._request_by_id[int(row._id)] = row
                self._in_flight[
                    (int(row._id), int(getattr(row, "_attempt", 0)))
                ] = (replica_id, claimed, False)
                current_budget -= request_work
        if needs_replan:
            self._replan_event.set()
        self._arm_mobile_timer(next_deadline)

    async def on_prefill_started(
        self, request_id: int, attempt: int
    ) -> None:
        key = (int(request_id), int(attempt))
        entry = self._in_flight.get(key)
        if entry is None:
            # Stale callback from a previous attempt: ignore.
            return
        replica_id, claimed, aborting = entry
        if aborting:
            # Abort decided before first token: do not lock; rebind proceeds.
            return
        self._in_flight.pop(key, None)
        self._sidecar_waiting.mark_prefill_locked(
            replica_id,
            claimed.request,
            expected_version=claimed.version,
        )

    async def on_request_terminal(
        self, request_id: int, attempt: int
    ) -> None:
        rid = int(request_id)
        key = (rid, int(attempt))
        entry = self._in_flight.get(key)
        if entry is not None and not entry[2]:
            self._in_flight.pop(key, None)
        aborted = self._aborted_attempts.get(rid)
        if aborted and int(attempt) in aborted:
            aborted.discard(int(attempt))
            if not aborted:
                del self._aborted_attempts[rid]
            row = self._request_by_id.get(rid)
            if row is not None:
                row._abort_completed = (
                    int(getattr(row, "_abort_completed", 0)) + 1
                )
        still_in_flight = any(
            active_request_id == rid
            for active_request_id, _active_attempt in self._in_flight
        )
        still_waiting = self._sidecar_waiting.owner(rid) is not None
        if (
            not still_in_flight
            and not still_waiting
            and not self._aborted_attempts.get(rid)
        ):
            self._request_by_id.pop(rid, None)

    async def close(self) -> None:
        tasks = []
        if self._mobile_timer is not None and not self._mobile_timer.done():
            self._mobile_timer.cancel()
            tasks.append(self._mobile_timer)
        if self._replan_task is not None and not self._replan_task.done():
            self._replan_task.cancel()
            tasks.append(self._replan_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        mean_ms = (
            1000.0 * self._planner_total_s / self._planner_runs
            if self._planner_runs
            else 0.0
        )
        logger.info(
            "ravel_sidecar planner_runs=%d mean_ms=%.3f max_ms=%.3f "
            "transfer_attempts=%d transfer_successes=%d conflicts=%d "
            "control_rtt_ms=%.3f",
            self._planner_runs,
            mean_ms,
            1000.0 * self._planner_max_s,
            self.sidecar_transfer_attempts,
            self.sidecar_transfer_successes,
            self.sidecar_claim_conflicts,
            1000.0 * self.control_horizon_s,
        )

    def waiting_movable_count(self) -> int:
        return self._sidecar_waiting.total_count()
