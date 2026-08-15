from __future__ import annotations

import asyncio
from typing import Dict, Optional

from dualmap.cluster.routing import (
    RavelInitialPlacementPolicy,
    ReplicaRoutingState,
)
from dualmap.cluster.topology import ClusterTopology
from dualmap.entities.request import Request
from dualmap.logger import init_logger
from dualmap.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from dualmap.scheduler.utils.double_hash_global_scheduler_utils import GlobalRequestQueue
from dualmap.scheduler.utils.shared import SharedState


logger = init_logger(__name__)


class RavelBaseGlobalScheduler(BaseGlobalScheduler):
    """Internal cross-cluster routing base for RAVEL.

    The central queue owns requests until they are admitted to a regional
    replica. Consequently, rebalancing never migrates materialized KV state.
    """

    def __init__(self, num_replicas: int, shared_state: SharedState, args):
        super().__init__(num_replicas)
        self.shared_state = shared_state
        self.topology = ClusterTopology.from_json(args.cluster_topology)
        self.topology.validate_replica_count(num_replicas)
        self.policy = RavelInitialPlacementPolicy(
            self.topology,
            overload_fraction=float(getattr(args, "cluster_overload_fraction", 0.5)),
        )
        self.global_request_queue = GlobalRequestQueue(num_replicas)
        self._schedule_lock = asyncio.Lock()
        self.pending_request_limit = int(
            getattr(args, "dh_replica_pending_req_threshold", 1)
        )
        self.rebalance_token_threshold = int(
            getattr(args, "dh_rebalance_thredhold", 0)
        )
        self.rebalance_wait_s = float(
            getattr(args, "dh_rebalance_waiting_latency_thredhold", 0.5)
        )
        self.rebalance_hysteresis_s = float(
            getattr(args, "cluster_rebalance_hysteresis_s", 0.02)
        )
        self.max_rebalances_per_event = int(
            getattr(args, "cluster_max_rebalances_per_event", 8)
        )
        self.rebalance_count = 0
        self.shared_state.set_scheduler_callback(self.schedule)

    def _build_states(
        self,
        request: Request,
        exclude_queued_replica: Optional[int] = None,
    ) -> Dict[int, ReplicaRoutingState]:
        states: Dict[int, ReplicaRoutingState] = {}
        for replica_id in self.topology.replica_ids:
            replica = self.shared_state.replica_budgets[replica_id]
            uncached_tokens = replica.get_num_recompute_token_ids(request._input_ids)
            queued_tokens = self.global_request_queue.get_global_actual_waiting_tokens_count(
                replica_id
            )
            if replica_id == exclude_queued_replica:
                queued_tokens = max(0, queued_tokens - uncached_tokens)
            states[replica_id] = ReplicaRoutingState(
                replica_id=replica_id,
                queued_prefill_tokens=queued_tokens,
                engine_pending_prefill_tokens=(
                    self.shared_state.get_num_actual_pending_tokens_replica(replica_id)
                ),
                uncached_prompt_tokens=uncached_tokens,
                prefix_hit_tokens=max(0, len(request._input_ids) - uncached_tokens),
                running_requests=replica.get_num_running_req(),
                queued_requests=max(
                    0,
                    self.global_request_queue.get_queue_len(replica_id)
                    - int(replica_id == exclude_queued_replica),
                ),
                engine_pending_requests=replica.get_num_pending_req(),
            )
        return states

    def _record_decision(self, request: Request, decision) -> None:
        request._primary_cluster = decision.cluster_id
        request._second_cluster = decision.alternate_cluster_id
        request._primary_replica = decision.replica_id
        request._second_replica = decision.alternate_replica_id
        request._predicted_ttft_s = decision.predicted_ttft_s
        request._predicted_objective_s = decision.predicted_objective_s
        request._cluster_route_reason = decision.reason
        request._cluster_route_feasible = decision.feasible

    async def _enqueue(self, request: Request) -> None:
        states = self._build_states(request)
        decision = self.policy.choose(request.routing_view(), states)
        self._record_decision(request, decision)
        state = states[decision.replica_id]
        self.global_request_queue.push(
            decision.replica_id,
            request,
            state.prefix_hit_tokens,
        )
        logger.info(
            "cluster_route request=%s cluster=%s replica=%s alternate_cluster=%s "
            "predicted_ttft=%.4f objective=%.4f feasible=%s reason=%s",
            request._id,
            decision.cluster_id,
            decision.replica_id,
            decision.alternate_cluster_id,
            decision.predicted_ttft_s,
            decision.predicted_objective_s,
            decision.feasible,
            decision.reason,
        )

    async def _rebalance_source(self, source_replica_id: int) -> int:
        source_cluster_id = self.topology.cluster_for_replica(source_replica_id).cluster_id
        moved = 0
        candidates = list(self.global_request_queue.get_all_requests(source_replica_id))
        candidates.sort(key=lambda request: request._arrived_at)
        for request in candidates:
            if moved >= self.max_rebalances_per_event:
                break
            states = self._build_states(request, exclude_queued_replica=source_replica_id)
            source = self.policy.choose(
                request.routing_view(), states, allowed_cluster_ids=(source_cluster_id,)
            )
            target_cluster_ids = tuple(
                cluster.cluster_id
                for cluster in self.topology.clusters
                if cluster.cluster_id != source_cluster_id
            )
            target = self.policy.choose(
                request.routing_view(), states, allowed_cluster_ids=target_cluster_ids
            )
            improves_feasibility = target.feasible and not source.feasible
            improves_objective = (
                target.predicted_objective_s + self.rebalance_hysteresis_s
                < source.predicted_objective_s
            )
            if not improves_feasibility and not improves_objective:
                continue
            if not self.global_request_queue.del_req(source_replica_id, request):
                continue

            target_state = states[target.replica_id]
            self.global_request_queue.push(
                target.replica_id,
                request,
                target_state.prefix_hit_tokens,
            )
            target = type(target)(
                cluster_id=target.cluster_id,
                replica_id=target.replica_id,
                alternate_cluster_id=source.cluster_id,
                alternate_replica_id=source.replica_id,
                predicted_ttft_s=target.predicted_ttft_s,
                predicted_objective_s=target.predicted_objective_s,
                feasible=target.feasible,
                reason="cluster_pending_rebalance",
            )
            self._record_decision(request, target)
            moved += 1
            self.rebalance_count += 1
        return moved

    async def _rebalance(self) -> int:
        moved = 0
        for replica_id in self.topology.replica_ids:
            queued_tokens = self.global_request_queue.get_global_actual_waiting_tokens_count(
                replica_id
            )
            max_wait_s = self.global_request_queue.get_max_waiting_delay(replica_id)
            if (
                queued_tokens > self.rebalance_token_threshold
                or max_wait_s >= self.rebalance_wait_s
            ):
                moved += await self._rebalance_source(replica_id)
        return moved

    async def _dispatch_schedulable(self) -> None:
        for replica_id in self.topology.replica_ids:
            if self.global_request_queue.is_empty(replica_id):
                continue
            replica = self.shared_state.replica_budgets[replica_id]
            current_budget, _, _, pending_requests, _ = await replica.get_load_states()
            if current_budget <= 0 or pending_requests >= self.pending_request_limit:
                continue
            max_pop = max(0, self.pending_request_limit - pending_requests)
            requests = self.global_request_queue.pop_schedulable(
                replica_id, current_budget, max_pop
            )
            for request in requests:
                request._enforce_prefill_budget = True
                admitted = await self.shared_state.add_posting_request_tasks(
                    replica_id, request
                )
                if not admitted:
                    self.global_request_queue.push(
                        replica_id,
                        request,
                        int(getattr(request, "_estimated_prefix_hit_tokens", 0)),
                    )

    async def schedule(self, new_request: Optional[Request]) -> int:
        async with self._schedule_lock:
            if new_request is not None:
                await self._enqueue(new_request)
            await self._rebalance()
            await self._dispatch_schedulable()
            # This RAVEL base dispatches queued requests
            # itself and prevents a second dispatch in RequestGenerator.
            return -1

    def finish_request(self, func_output=None, text: str = None, input_ids=None):
        return None
