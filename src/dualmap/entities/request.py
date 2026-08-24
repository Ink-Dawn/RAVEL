from typing import List, Optional, Tuple

from dualmap.cluster.routing import RoutingRequestView
from dualmap.cluster.slo import RequestSLO

class Request():
    def __init__(
        self,
        request_id: int,
        dataset_type: str,
        native_session_id: int,
        session_id: str,
        hash_session_id: str,
        round_id: int, 
        prompts: str,
        input_ids: Optional[List[int]],
        num_prefill_tokens: int,
        actual_num_prefill_tokens: int,
        output_len: int,
        over_flow: bool,
        n: int,
        temperature: int,
        top_p: int,
        max_tokens: int,
        stream:bool,
        arrived_at: float,
        time_interval:float,
        hash_prefix_len:int,
        request_type: int = 0,
        collection_id: int = 0,
        slo_constraint: Tuple[float, float, float] = (0.8, 0.08, 8.0),
        client_region: str = "local",
        routing_output_tokens_hint: int = 256,
        stage_id: int = 0,
        stage_num: int = 1,
        task_branch_count: int = 1,
    ):
        self._id = request_id
        self._dataset_type = dataset_type
        self._native_session_id = native_session_id
        self._session_id = session_id
        self._hash_session_id = hash_session_id
        self._round_id = round_id
        self._prompts = prompts
        self._input_ids = input_ids
        self._num_prefill_tokens = num_prefill_tokens # len(self._prefill_tokens)
        self._actual_num_prefill_tokens = actual_num_prefill_tokens  
        self._output_len = output_len
        self._over_flow = over_flow
        self._n= n
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._stream = stream
        self._arrived_at = arrived_at
        self._time_interval = time_interval
        self._attempt = 0
        self._abort_attempts = 0
        self._abort_completed = 0
        self._primary_replica = -1 # for double_hash
        self._second_replica = -1 # for double_hash
        self._rounting_cache_hit_max = -1 # for double_hash
        self._is_dh_cache_affinity = 0 # for double_hash
        self._is_dh_least_loaded = 0 # for double_hash
        self._is_dh_cache_affinity_least_loaded = 0 # for double_hash
        self._hash_prefix_len = hash_prefix_len
        self._request_type = int(request_type)
        self._collection_id = int(collection_id)
        self._slo_constraint = tuple(float(value) for value in slo_constraint)
        self._client_region = client_region
        self._routing_output_tokens_hint = int(routing_output_tokens_hint)
        self._routing_output_tokens_hint_used = int(routing_output_tokens_hint)
        self._routing_output_tokens_upper_hint = int(routing_output_tokens_hint)
        self._routing_output_tokens_upper_hint_used = int(
            routing_output_tokens_hint
        )
        self._estimated_prefix_hit_tokens = 0
        self._prefix_hit_source = "router_shadow_locality_hint"
        self._enforce_prefill_budget = False
        self._stage_id = int(stage_id)
        self._stage_num = int(stage_num)
        self._task_branch_count = max(1, int(task_branch_count))
        self._primary_cluster = ""
        self._second_cluster = ""
        self._predicted_ttft_s = 0.0
        self._point_predicted_ttft_s = 0.0
        self._predicted_objective_s = 0.0
        self._point_predicted_objective_s = 0.0
        self._ttft_residual_guard_s = 0.0
        self._objective_residual_guard_s = 0.0
        self._risk_calibrated = False
        self._ttft_prediction_residual_s = 0.0
        self._objective_prediction_residual_s = 0.0
        self._ttft_residual_track = False  # Compatibility alias.
        self._objective_residual_track = False
        self._cluster_route_reason = ""
        self._cluster_route_feasible = False
        self._network_rtt_s = 0.0
        self._inject_network_delay = True
        self._ravel_prefill_gap_s = 0.0
        self._ravel_protected_tbt_s = 0.0
        self._ravel_prefill_chunk_tokens = 0
        self._ravel_admission_envelope_safe = True
        self._ravel_admission_victims = 0
        self._ravel_ttft_alive = True
        self._ravel_admission_deferrals = 0
        self._ravel_max_admission_victims = 0
        self._ravel_tbt_alive = True
        self._ravel_last_token_at = 0.0
        self._ravel_first_token_at = 0.0
        self._ravel_soft_admission_active = False
        self._ravel_soft_admission_protected = False
        self._ravel_soft_admission_release_at = 0.0
        self._ravel_soft_admission_slack_release = False
        self._ravel_soft_admission_min_slack_s = 0.0
        self._ravel_soft_admission_increment_s = 0.0
        self._ravel_soft_admission_victims = 0
        self._ravel_soft_plan_generation = -1
        self._ravel_soft_selected = False
        # Mechanism-level observability for protected/deferred admission.
        # These fields are write-only diagnostics and never enter placement.
        self._ravel_soft_admission_first_deferred_at = 0.0
        self._ravel_soft_admission_deferral_s = 0.0
        self._ravel_soft_admission_deadline_release = False
        self._ravel_soft_admission_dispatch_at = 0.0
        # The planner continuously refreshes this tentative quote.  The
        # handoff callback freezes the last replica-matching value immediately
        # before the request is materialized in the engine.
        self._ravel_last_quote_replica_id = -1
        self._ravel_last_quote_timestamp_s = 0.0
        self._ravel_last_quote_predicted_ttft_s = 0.0
        self._ravel_last_quote_predicted_tbt_s = 0.0
        self._ravel_last_quote_point_completion_s = 0.0
        self._ravel_last_quote_risk_completion_s = 0.0
        self._ravel_last_quote_feasible = False
        self._ravel_quote_handoff_replica_id = -1
        self._ravel_quote_handoff_timestamp_s = 0.0
        self._ravel_quote_handoff_age_s = 0.0
        self._ravel_quote_handoff_predicted_ttft_s = 0.0
        self._ravel_quote_handoff_predicted_tbt_s = 0.0
        self._ravel_quote_handoff_point_completion_s = 0.0
        self._ravel_quote_handoff_risk_completion_s = 0.0
        self._ravel_quote_handoff_feasible = False
        self._ravel_quote_handoff_valid = False
        self._ravel_profile_output_expected = 0
        self._ravel_profile_output_upper = 0

    def __lt__(self, other: object) -> bool:
        return int(self._id) < int(getattr(other, "_id", other))

    def routing_view(self) -> RoutingRequestView:
        """Return a router-safe view that excludes true output length."""
        return RoutingRequestView(
            request_id=self._id,
            prefix_key=self._hash_session_id,
            prompt_tokens=self._num_prefill_tokens,
            request_type=self._request_type,
            slo=RequestSLO(*self._slo_constraint),
            client_region=self._client_region,
            output_tokens_hint=self._routing_output_tokens_hint,
        )
