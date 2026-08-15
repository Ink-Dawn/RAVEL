from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from dualmap.cluster.slo import LATENCY, RequestSLO
from dualmap.cluster.topology import ClusterSpec, ClusterTopology


@dataclass(frozen=True)
class RoutingRequestView:
    """Only fields the central router is allowed to observe."""

    request_id: int
    prefix_key: str
    prompt_tokens: int
    request_type: int
    slo: RequestSLO
    client_region: str
    output_tokens_hint: int


@dataclass(frozen=True)
class ReplicaRoutingState:
    replica_id: int
    queued_prefill_tokens: int
    engine_pending_prefill_tokens: int
    uncached_prompt_tokens: int
    prefix_hit_tokens: int
    running_requests: int = 0
    queued_requests: int = 0
    engine_pending_requests: int = 0

    @property
    def total_prefill_tokens(self) -> int:
        return (
            self.queued_prefill_tokens
            + self.engine_pending_prefill_tokens
            + self.uncached_prompt_tokens
        )

    @property
    def total_prefill_requests(self) -> int:
        return self.queued_requests + self.engine_pending_requests + 1


@dataclass(frozen=True)
class ReplicaDecision:
    cluster_id: str
    replica_id: int
    predicted_ttft_s: float
    predicted_objective_s: float
    prefix_hit_tokens: int
    feasible: bool


@dataclass(frozen=True)
class ClusterDecision:
    cluster_id: str
    replica_id: int
    alternate_cluster_id: str
    alternate_replica_id: int
    predicted_ttft_s: float
    predicted_objective_s: float
    feasible: bool
    reason: str


class RavelInitialPlacementPolicy:
    """RAVEL's deterministic initial-candidate placement policy.

    Each candidate cluster first evaluates two prefix-hashed replicas. The
    cluster-level decision keeps the cache-affine choice while it is SLO
    feasible and otherwise selects the candidate with the lowest objective.
    """

    def __init__(
        self,
        topology: ClusterTopology,
        overload_fraction: float = 0.5,
    ) -> None:
        if not 0 < overload_fraction <= 1:
            raise ValueError("overload_fraction must be in (0, 1]")
        self.topology = topology
        self.overload_fraction = overload_fraction

    @staticmethod
    def _hash_index(key: str, salt: str, size: int) -> int:
        digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % size

    def candidate_clusters(self, prefix_key: str) -> Tuple[ClusterSpec, ClusterSpec]:
        clusters = self.topology.clusters
        first_index = self._hash_index(prefix_key, "cluster-primary", len(clusters))
        second_index = self._hash_index(prefix_key, "cluster-secondary", len(clusters))
        if second_index == first_index:
            second_index = (first_index + 1) % len(clusters)
        return clusters[first_index], clusters[second_index]

    def candidate_replicas(self, cluster: ClusterSpec, prefix_key: str) -> Tuple[int, ...]:
        replicas = cluster.replica_ids
        first_index = self._hash_index(prefix_key, "replica-primary", len(replicas))
        if len(replicas) == 1:
            return (replicas[first_index],)
        second_index = self._hash_index(prefix_key, "replica-secondary", len(replicas))
        if second_index == first_index:
            second_index = (first_index + 1) % len(replicas)
        return replicas[first_index], replicas[second_index]

    @staticmethod
    def _objective_s(
        request: RoutingRequestView,
        cluster: ClusterSpec,
        predicted_ttft_s: float,
        running_requests: int = 0,
    ) -> float:
        if request.request_type == LATENCY:
            return predicted_ttft_s
        # The hint is a public, fixed/configured cap. It must not be the trace's
        # true output length. This keeps throughput routing non-anticipating.
        return predicted_ttft_s + request.output_tokens_hint * cluster.decode_tpot_for(running_requests + 1)

    def _evaluate_replica(
        self,
        request: RoutingRequestView,
        cluster: ClusterSpec,
        state: ReplicaRoutingState,
    ) -> ReplicaDecision:
        predicted_ttft_s = (
            cluster.rtt_s(request.client_region)
            + state.total_prefill_requests
            * cluster.prefill_intercept_for(state.running_requests)
            + state.total_prefill_tokens
            * cluster.prefill_tpot_for(state.running_requests)
        )
        objective_s = self._objective_s(
            request, cluster, predicted_ttft_s, state.running_requests
        )
        slo_limit = request.slo.ttft_s if request.request_type == LATENCY else request.slo.ttlt_s
        feasible = objective_s <= slo_limit
        return ReplicaDecision(
            cluster_id=cluster.cluster_id,
            replica_id=state.replica_id,
            predicted_ttft_s=predicted_ttft_s,
            predicted_objective_s=objective_s,
            prefix_hit_tokens=state.prefix_hit_tokens,
            feasible=feasible,
        )

    def _choose_within_cluster(
        self,
        request: RoutingRequestView,
        cluster: ClusterSpec,
        states: Mapping[int, ReplicaRoutingState],
    ) -> ReplicaDecision:
        candidates = [
            self._evaluate_replica(request, cluster, states[replica_id])
            for replica_id in self.candidate_replicas(cluster, request.prefix_key)
        ]
        cache_choice = min(
            candidates,
            key=lambda item: (-item.prefix_hit_tokens, item.predicted_objective_s, item.replica_id),
        )
        slo_limit = request.slo.ttft_s if request.request_type == LATENCY else request.slo.ttlt_s
        cache_guard = slo_limit * self.overload_fraction
        if cache_choice.predicted_objective_s <= cache_guard:
            return cache_choice
        return min(
            candidates,
            key=lambda item: (
                not item.feasible,
                item.predicted_objective_s,
                -item.prefix_hit_tokens,
                item.replica_id,
            ),
        )

    def choose(
        self,
        request: RoutingRequestView,
        states: Mapping[int, ReplicaRoutingState],
        allowed_cluster_ids: Optional[Iterable[str]] = None,
    ) -> ClusterDecision:
        candidate_clusters = self.candidate_clusters(request.prefix_key)
        if allowed_cluster_ids is not None:
            allowed = set(allowed_cluster_ids)
            candidate_clusters = tuple(
                cluster for cluster in candidate_clusters if cluster.cluster_id in allowed
            )
        if not candidate_clusters:
            raise ValueError("no eligible cluster candidates")

        cluster_choices = [
            self._choose_within_cluster(request, cluster, states)
            for cluster in candidate_clusters
        ]
        cache_choice = min(
            cluster_choices,
            key=lambda item: (-item.prefix_hit_tokens, item.predicted_objective_s, item.cluster_id),
        )
        slo_limit = request.slo.ttft_s if request.request_type == LATENCY else request.slo.ttlt_s
        if cache_choice.predicted_objective_s <= slo_limit * self.overload_fraction:
            primary = cache_choice
            reason = "cluster_prefix_affinity_within_guard"
        else:
            primary = min(
                cluster_choices,
                key=lambda item: (
                    not item.feasible,
                    item.predicted_objective_s,
                    -item.prefix_hit_tokens,
                    item.cluster_id,
                ),
            )
            reason = "cluster_slo_objective"

        alternates = [item for item in cluster_choices if item.cluster_id != primary.cluster_id]
        alternate = min(
            alternates or [item for item in cluster_choices if item.replica_id != primary.replica_id] or [primary],
            key=lambda item: (not item.feasible, item.predicted_objective_s, item.replica_id),
        )
        return ClusterDecision(
            cluster_id=primary.cluster_id,
            replica_id=primary.replica_id,
            alternate_cluster_id=alternate.cluster_id,
            alternate_replica_id=alternate.replica_id,
            predicted_ttft_s=primary.predicted_ttft_s,
            predicted_objective_s=primary.predicted_objective_s,
            feasible=primary.feasible,
            reason=reason,
        )
