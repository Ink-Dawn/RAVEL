import asyncio
from collections import defaultdict, deque
from dataclasses import replace
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from dualmap.entities.replica import Replica
from dualmap.cache_manager.kvcache_store.prefix_kvcache_lru import PrefixLRUCache
from dualmap.entities.request import Request
from dualmap.scheduler.utils.shared import SharedState
from dualmap.scheduler.global_scheduler.ravel_native_global_scheduler import (
    Quote,
    RavelNativeAdaptiveQuantumGlobalScheduler,
    RavelNativeLoadAdaptiveQuantumGlobalScheduler,
    RavelNativeLoadAdaptiveRiskGlobalScheduler,
    RavelNativeMobileInsertionGlobalScheduler,
    RavelNativeMobileAdaptiveGlobalScheduler,
    RavelNativeMobileSLOCompleteGlobalScheduler,
    RavelNativeCenterYieldGlobalScheduler,
    RavelNativeCenterWorkBalancedGlobalScheduler,
    RavelNativeCenterWorkYieldGlobalScheduler,
    RavelNativeCenterWorkEDFGlobalScheduler,
    RavelNativeCenterLatencyFirstGlobalScheduler,
    RavelNativeCenterLatencyFairGlobalScheduler,
    RavelNativeCenterSelectedAdmissionGlobalScheduler,
    RavelNativeCenterDynamicChunkGlobalScheduler,
    RavelNativeCenterSLOEnvelopeGlobalScheduler,
    RavelNativeCenterTBTGlobalScheduler,
    RavelNativeCenterObjectiveEDFGlobalScheduler,
    RavelNativeCenterDensityGlobalScheduler,
    RavelNativeCenterCohortGlobalScheduler,
    RavelNativeCenterOnTimeSetGlobalScheduler,
    RavelDiagnosticExactCompletionOracleGlobalScheduler,
    RavelHindsightJointPlanGlobalScheduler,
    RavelNativeSLOFlowGlobalScheduler,
    RavelNativeMobileSLOAwareGlobalScheduler,
    RavelNativeMobileTriggeredGlobalScheduler,
    RavelNativeDecodeQuantum1GlobalScheduler,
    RavelNativeGlobalScheduler,
    RavelNativeV2GlobalScheduler,
)
from ravel_engine_adapter.protocol import decode_tbt_s
from dualmap.scheduler.global_scheduler.ravel_sidecar_mobile_scheduler import (
    RavelNativeSidecarMobileSLOCompleteGlobalScheduler,
)


class FakeReplica:
    def __init__(self, pending_tokens=0, pending_count=0):
        self.pending_tokens = pending_tokens
        self.pending_count = pending_count
        self.ttft_residual_samples = 0
        self.ttft_residual_updated_at = -1.0
        self.ttft_residual_history = []
        self.pending_requests = [
            SimpleNamespace(_id=-(index + 1))
            for index in range(pending_count)
        ]
        self.running_requests = []

    def get_num_recompute_token_ids(self, input_ids):
        return len(input_ids)

    def get_num_pending_req(self):
        return self.pending_count

    def get_num_running_req(self):
        return 0

    async def get_load_states(self):
        return 65536, 0.0, 0.0, 0, 0.0


class FakeSharedState:
    def __init__(self, local_work=0, remote_work=0, local_count=0, remote_count=0):
        self.replica_budgets = {
            0: FakeReplica(local_work, local_count),
            1: FakeReplica(remote_work, remote_count),
        }
        self.num_replicas = 2
        self.callback = None

    def set_scheduler_callback(self, callback):
        self.callback = callback


    def set_prefill_started_callback(self, callback):
        self.prefill_started_callback = callback

    def set_request_terminal_callback(self, callback):
        self.request_terminal_callback = callback

    def get_routing_output_hint(self, request_type, fallback_tokens):
        return int(fallback_tokens)
    def get_num_actual_pending_tokens_replica(self, replica_id):
        return self.replica_budgets[replica_id].pending_tokens


def make_request(request_id=1, request_type=0, slo_constraint=(0.8, 0.08, 8.0)):
    return Request(
        request_id,
        "jitserve",
        0,
        "session",
        f"prefix-{request_id}",
        0,
        "prompt",
        list(range(100)),
        100,
        100,
        64,
        False,
        1,
        0,
        1,
        256,
        True,
        time.perf_counter(),
        0.0,
        0,
        request_type=request_type,
        slo_constraint=slo_constraint,
        client_region="region_a",
        routing_output_tokens_hint=256,
    )


class RavelNativeSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        topology = {
            "default_client_region": "region_a",
            "clusters": [
                {
                    "id": "region_a",
                    "replica_ids": [0],
                    "rtt_ms_by_client_region": {"region_a": 0.5},
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.001,
                    "decode_curve": [
                        {"active_sequences": 0, "tpot_s_median": 0.001},
                        {"active_sequences": 81, "tpot_s_median": 0.1},
                    ],
                },
                {
                    "id": "region_b",
                    "replica_ids": [1],
                    "rtt_ms_by_client_region": {"region_a": 100.0},
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.001,
                    "decode_curve": [
                        {"active_sequences": 0, "tpot_s_median": 0.001},
                        {"active_sequences": 81, "tpot_s_median": 0.1},
                    ],
                },
            ],
        }
        topology_path = Path(self.directory.name) / "topology.json"
        topology_path.write_text(json.dumps(topology), encoding="utf-8")
        self.args = SimpleNamespace(
            cluster_topology=str(topology_path),
            cluster_overload_fraction=0.5,
            dh_replica_pending_req_threshold=8,
            dh_rebalance_thredhold=32768,
            dh_rebalance_waiting_latency_thredhold=0.5,
            cluster_rebalance_hysteresis_s=0.02,
            cluster_max_rebalances_per_event=8,
            block_size=16,
            max_num_batched_tokens=16384,
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_calibrated_quote_includes_fresh_one_sided_residual(self):
        shared = FakeSharedState()
        replica = shared.replica_budgets[0]
        replica.ttft_residual_history = [0.3] * 19
        replica.ttft_residual_samples = len(replica.ttft_residual_history)
        replica.ttft_residual_updated_at = time.perf_counter()
        scheduler = RavelNativeGlobalScheduler(2, shared, self.args)

        guarded = scheduler._quote(make_request(), 0)
        frozen = RavelNativeV2GlobalScheduler(2, shared, self.args)._quote(
            make_request(2), 0
        )

        self.assertAlmostEqual(guarded.residual_guard_s, 0.3, delta=0.01)
        self.assertTrue(guarded.risk_calibrated)
        self.assertEqual(frozen.residual_guard_s, 0.0)
        self.assertAlmostEqual(
            guarded.predicted_ttft_s, frozen.predicted_ttft_s, delta=0.01
        )

    def test_residual_observation_does_not_change_placement(self):
        shared = FakeSharedState(local_work=5000, remote_work=0)
        local = shared.replica_budgets[0]
        local.ttft_residual_history = [0.3] * 19
        local.ttft_residual_samples = len(local.ttft_residual_history)
        local.ttft_residual_updated_at = time.perf_counter()
        optimized = RavelNativeGlobalScheduler(2, shared, self.args)
        frozen = RavelNativeV2GlobalScheduler(2, shared, self.args)

        optimized_choice = optimized._choose_initial(make_request())
        frozen_choice = frozen._choose_initial(make_request(2))

        self.assertEqual(optimized_choice.replica_id, 0)
        self.assertEqual(frozen_choice.replica_id, 0)


    def test_lower_local_rtt_does_not_override_pressure_striping(self):
        shared = FakeSharedState(local_count=3, remote_count=0)
        scheduler = RavelNativeGlobalScheduler(2, shared, self.args)

        choice = scheduler._choose_initial(make_request(request_type=1))

        self.assertEqual(choice.replica_id, 1)


    def test_flexible_requests_use_zero_locality_headroom(self):
        shared = FakeSharedState(local_count=2, remote_count=0)
        optimized = RavelNativeGlobalScheduler(2, shared, self.args)
        frozen = RavelNativeV2GlobalScheduler(2, shared, self.args)

        optimized_choice = optimized._choose_initial(make_request(request_type=1))
        frozen_choice = frozen._choose_initial(make_request(2, request_type=1))

        self.assertEqual(optimized_choice.replica_id, 1)
        self.assertEqual(frozen_choice.replica_id, 0)


    def test_latency_initial_placement_uses_remote_only_under_local_pressure(self):
        shared = FakeSharedState(local_work=7000, local_count=2)
        scheduler = RavelNativeGlobalScheduler(2, shared, self.args)

        choice = scheduler._choose_initial(make_request(request_type=0))

        self.assertEqual(choice.replica_id, 1)

    def test_latency_initial_placement_stays_local_when_local_has_slack(self):
        shared = FakeSharedState()
        scheduler = RavelNativeGlobalScheduler(2, shared, self.args)

        choice = scheduler._choose_initial(make_request(request_type=0))

        self.assertEqual(choice.replica_id, 0)

    def test_latency_initial_placement_disperses_under_decode_pressure(self):
        shared = FakeSharedState(local_count=6)
        scheduler = RavelNativeGlobalScheduler(2, shared, self.args)

        choice = scheduler._choose_initial(make_request(request_type=0))

        self.assertEqual(choice.replica_id, 1)

    def test_mobile_triggered_latency_stays_local_when_quiet(self):
        shared = FakeSharedState()
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(2, shared, self.args)

        choice = scheduler._choose_initial(make_request(request_type=0))

        self.assertEqual(choice.replica_id, 0)


    def test_mobile_triggered_initial_choice_has_no_count_threshold(self):
        shared = FakeSharedState(local_count=6)
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(2, shared, self.args)
        request = make_request(request_type=0)

        choice = scheduler._choose_initial(request)

        self.assertEqual(choice.replica_id, 0)
        self.assertIn("min_objective", request._cluster_route_reason)


    def test_decode_quantum_horizon_scales_with_replicas_and_block_size(self):
        shared = FakeSharedState()
        self.args.block_size = 32

        scheduler = RavelNativeDecodeQuantum1GlobalScheduler(2, shared, self.args)

        self.assertEqual(scheduler.decode_horizon_tokens, 64)

    def test_adaptive_decode_quantum_uses_request_slo_type(self):
        shared = FakeSharedState()
        self.args.block_size = 32
        scheduler = RavelNativeAdaptiveQuantumGlobalScheduler(2, shared, self.args)

        self.assertEqual(scheduler._decode_horizon(make_request(request_type=1)), 64)
        self.assertEqual(scheduler._decode_horizon(make_request(request_type=0)), 128)
        self.assertEqual(scheduler._decode_horizon(make_request(request_type=2)), 128)

    def test_load_adaptive_quantum_uses_topology_rtt_as_burst_boundary(self):
        shared = FakeSharedState()
        self.args.block_size = 32
        scheduler = RavelNativeLoadAdaptiveQuantumGlobalScheduler(2, shared, self.args)

        scheduler._recent_arrivals = [(0.0, "region_a"), (0.05, "region_a")]
        self.assertEqual(scheduler._decode_horizon(make_request(request_type=0)), 64)
        self.assertEqual(scheduler._decode_horizon(make_request(request_type=2)), 128)

        scheduler._recent_arrivals = [(0.0, "region_a"), (0.20, "region_a")]
        self.assertEqual(scheduler._decode_horizon(make_request(request_type=0)), 128)

    def test_load_adaptive_risk_enables_fresh_residual_decisions(self):
        scheduler = RavelNativeLoadAdaptiveRiskGlobalScheduler(
            2, FakeSharedState(), self.args
        )

        self.assertTrue(scheduler.residual_decision_enabled)
        self.assertTrue(scheduler.risk_aware_region_selection)

    def test_mobile_hold_is_bounded_by_nearest_remote_rtt(self):
        scheduler = RavelNativeMobileInsertionGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        request = make_request(request_type=0)

        scheduler._recent_arrivals = [(0.0, "region_a"), (0.05, "region_a")]
        self.assertAlmostEqual(scheduler._mobile_hold_s(request), 0.05)

        scheduler._recent_arrivals = [(0.0, "region_a"), (0.20, "region_a")]
        self.assertEqual(scheduler._mobile_hold_s(request), 0.0)

    def test_mobile_insertion_can_use_an_empty_remote_replica(self):
        shared = FakeSharedState()
        scheduler = RavelNativeMobileInsertionGlobalScheduler(
            2, shared, self.args
        )
        first = make_request(1, request_type=2)
        second = make_request(2, request_type=2)
        for row in (first, second):
            row._input_ids = list(range(5000))
            row._num_prefill_tokens = 5000
            row._rebind_count = 0
            scheduler.global_request_queue.push(0, row, 0)

        assignment = scheduler._plan_mobile_assignment([first, second])

        self.assertEqual(
            {target for target, _ in assignment.values()},
            {0, 1},
        )

    def test_mobile_beam_accounts_for_fixed_edf_prompt_debt(self):
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        fixed = make_request(1, request_type=0)
        fixed._input_ids = list(range(600))
        fixed._num_prefill_tokens = 600
        candidate = make_request(2, request_type=1)
        scheduler.global_request_queue.push(0, fixed, 0)
        scheduler.global_request_queue.push(0, candidate, 0)

        observed = []
        original_quote = scheduler._quote

        def capture_quote(request, replica_id, **kwargs):
            if request is candidate:
                observed.append((replica_id, dict(kwargs)))
            return original_quote(request, replica_id, **kwargs)

        scheduler._quote = capture_quote
        scheduler._plan_mobile_assignment([candidate])

        local = next(kwargs for replica_id, kwargs in observed if replica_id == 0)
        self.assertEqual(local["work_before_tokens"], 600)
        self.assertEqual(local["requests_before"], 1)

    def test_slo_aware_mobile_window_only_holds_flexible_requests(self):
        scheduler = RavelNativeMobileSLOAwareGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._recent_arrivals = [
            (0.0, "region_a"),
            (0.05, "region_a"),
        ]

        self.assertEqual(
            scheduler._mobile_hold_s(make_request(request_type=0)),
            0.0,
        )
        self.assertAlmostEqual(
            scheduler._mobile_hold_s(make_request(2, request_type=1)),
            0.025,
        )

    def test_latency_without_slack_is_excluded_from_mobile_candidates(self):
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        tight = make_request(
            1, request_type=0, slo_constraint=(0.02, 0.08, 8.0)
        )
        flexible = make_request(2, request_type=1)
        for row in (tight, flexible):
            scheduler.global_request_queue.push(0, row, 0)

        candidates = scheduler._mobile_candidates()

        self.assertNotIn(tight, candidates)
        self.assertIn(flexible, candidates)

    def test_latency_with_large_slack_and_local_pressure_is_mobile(self):
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(
            2, FakeSharedState(local_work=7000), self.args
        )
        roomy = make_request(1, request_type=0)
        scheduler.global_request_queue.push(0, roomy, 0)

        self.assertIn(roomy, scheduler._mobile_candidates())

    def test_latency_with_remote_feasibility_is_mobile(self):
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        slacky = make_request(1, request_type=0)
        scheduler.global_request_queue.push(0, slacky, 0)

        self.assertIn(slacky, scheduler._mobile_candidates())

    def test_flexible_admission_overflow_becomes_hard_at_expiry(self):
        scheduler = RavelNativeMobileSLOAwareGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        request = make_request(request_type=1)
        request._ravel_mobile_until = time.perf_counter() + 1.0
        self.assertEqual(
            scheduler._admission_overflow_cost(request, 2),
            0.0,
        )

        request._ravel_mobile_until = time.perf_counter() - 1.0
        self.assertEqual(
            scheduler._admission_overflow_cost(request, 2),
            2.0,
        )

    def test_adaptive_mobile_window_extends_collective_hold(self):
        scheduler = RavelNativeMobileAdaptiveGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._recent_arrivals = [(0.0, "region_a"), (0.05, "region_a")]
        request = make_request(request_type=2)

        self.assertAlmostEqual(scheduler._mobile_hold_s(request), 0.1)
        self.assertTrue(request._ravel_replan_on_flexible_arrival)

    def test_adaptive_mobile_window_uses_regular_observed_gap(self):
        scheduler = RavelNativeMobileAdaptiveGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._recent_arrivals = [
            (0.00, "region_a"),
            (0.05, "region_a"),
            (0.10, "region_a"),
            (0.15, "region_a"),
            (0.20, "region_a"),
        ]
        request = make_request(request_type=1)

        self.assertAlmostEqual(scheduler._mobile_hold_s(request), 0.1)
        self.assertFalse(request._ravel_replan_on_flexible_arrival)

    def test_adaptive_mobile_window_keeps_irregular_operating_point(self):
        scheduler = RavelNativeMobileAdaptiveGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._recent_arrivals = [
            (0.000, "region_a"),
            (0.005, "region_a"),
            (0.095, "region_a"),
            (0.105, "region_a"),
            (0.200, "region_a"),
        ]
        request = make_request(request_type=1)

        self.assertAlmostEqual(scheduler._mobile_hold_s(request), 0.1)
        self.assertFalse(request._ravel_replan_on_flexible_arrival)

    def test_quote_uses_hypothetical_decode_occupancy(self):
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        request = make_request(request_type=1)

        low = scheduler._quote(request, 0, prospective_sequences=1)
        high = scheduler._quote(request, 0, prospective_sequences=81)

        self.assertGreater(high.point_objective_s, low.point_objective_s + 20.0)

    def test_output_objective_counts_only_post_first_token_intervals(self):
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        request = make_request(request_type=1)
        request._routing_output_tokens_hint = 256

        quote = scheduler._quote(request, 0, prospective_sequences=1)
        decode_tpot = scheduler.topology.cluster_for_replica(
            0
        ).decode_tpot_for(1)

        self.assertAlmostEqual(
            quote.point_objective_s - quote.point_ttft_s,
            255 * decode_tpot,
        )

        request._routing_output_tokens_hint = 1
        one_token = scheduler._quote(
            request, 0, prospective_sequences=1
        )
        self.assertAlmostEqual(
            one_token.point_objective_s,
            one_token.point_ttft_s,
        )

    def test_output_risk_bound_is_separate_from_expected_cost(self):
        scheduler = RavelNativeMobileTriggeredGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        request = make_request(request_type=1)
        request._routing_output_tokens_hint = 100
        request._routing_output_tokens_upper_hint = 400

        quote = scheduler._quote(
            request, 0, prospective_sequences=1
        )
        decode_tpot = scheduler.topology.cluster_for_replica(
            0
        ).decode_tpot_for(1)

        self.assertAlmostEqual(
            quote.point_objective_s - quote.point_ttft_s,
            99 * decode_tpot,
        )
        self.assertAlmostEqual(
            quote.predicted_objective_s - quote.point_ttft_s,
            399 * decode_tpot,
        )
        self.assertEqual(
            request._routing_output_tokens_hint_used, 100
        )
        self.assertEqual(
            request._routing_output_tokens_upper_hint_used, 400
        )

    def test_slo_complete_quote_enforces_latency_tbt_limit(self):
        shared = FakeSharedState()
        shared.replica_budgets[0].get_num_running_req = lambda: 80
        scheduler = RavelNativeMobileSLOCompleteGlobalScheduler(
            2, shared, self.args
        )
        request = make_request(request_type=0)

        local = scheduler._quote(request, 0)
        remote = scheduler._quote(request, 1)

        self.assertFalse(local.feasible)
        self.assertTrue(remote.feasible)
        self.assertGreater(local.predicted_objective_s, remote.predicted_objective_s)

    def test_slo_complete_quote_does_not_change_throughput_semantics(self):
        shared = FakeSharedState()
        shared.replica_budgets[0].get_num_running_req = lambda: 7
        scheduler = RavelNativeMobileSLOCompleteGlobalScheduler(
            2, shared, self.args
        )
        baseline = RavelNativeMobileTriggeredGlobalScheduler(
            2, shared, self.args
        )
        request = make_request(request_type=1)

        candidate = scheduler._quote(request, 0)
        reference = baseline._quote(request, 0)
        self.assertAlmostEqual(
            candidate.predicted_objective_s,
            reference.predicted_objective_s,
            delta=0.001,
        )
        self.assertEqual(candidate.feasible, reference.feasible)


class CenterYieldSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        topology = {
            "default_client_region": "region_a",
            "clusters": [
                {
                    "id": "region_a",
                    "replica_ids": [0],
                    "rtt_ms_by_client_region": {
                        "region_a": 0.5,
                        "region_b": 34.1,
                        "default": 34.1,
                    },
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.001,
                },
                {
                    "id": "region_b",
                    "replica_ids": [1],
                    "rtt_ms_by_client_region": {
                        "region_a": 34.1,
                        "region_b": 0.5,
                        "default": 34.1,
                    },
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.001,
                },
            ],
        }
        topology_path = Path(self.directory.name) / "topology.json"
        topology_path.write_text(json.dumps(topology), encoding="utf-8")
        self.topology_path = topology_path
        self.args = SimpleNamespace(
            cluster_topology=str(topology_path),
            cluster_overload_fraction=0.5,
            dh_replica_pending_req_threshold=64,
            dh_rebalance_thredhold=32768,
            dh_rebalance_waiting_latency_thredhold=0.5,
            cluster_rebalance_hysteresis_s=0.02,
            cluster_max_rebalances_per_event=8,
            block_size=16,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            routing_output_tokens_hint=256,
        )
        self.shared = FakeSharedState()
        self.scheduler = RavelNativeCenterYieldGlobalScheduler(
            2,
            self.shared,
            self.args,
        )

    async def asyncTearDown(self):
        timer = getattr(self.scheduler, "_mobile_timer", None)
        if timer is not None:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
        self.directory.cleanup()

    async def test_frontier_is_derived_from_quantum_profile_and_rtt(self):
        expected = 16384 + math.ceil(0.0341 / 0.0001)
        self.assertEqual(
            self.scheduler._engine_frontier_tokens,
            {0: expected, 1: expected},
        )

    async def test_absolute_deadline_orders_dispatch_and_engine(self):
        arrived_at = time.perf_counter()
        tight = make_request(
            101,
            request_type=2,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        loose = make_request(
            102,
            request_type=2,
            slo_constraint=(0.8, 0.08, 16.0),
        )
        tight._arrived_at = arrived_at
        loose._arrived_at = arrived_at
        tight._cluster_route_feasible = False
        loose._cluster_route_feasible = True

        self.scheduler._refresh_yield_state(tight)
        self.scheduler._refresh_yield_state(loose)

        self.assertLess(
            self.scheduler._yield_dispatch_key(tight),
            self.scheduler._yield_dispatch_key(loose),
        )
        self.assertLess(tight._vllm_priority, loose._vllm_priority)
        self.assertTrue(tight._ravel_yield_deferred)

    async def test_frontier_holds_excess_then_refills_on_headroom(self):
        request = make_request(103, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._rebind_count = 0
        request._cluster_route_feasible = True
        self.scheduler._refresh_yield_state(request)
        self.scheduler.global_request_queue.push(0, request, 0)
        self.scheduler._engine_frontier_tokens = {0: 100, 1: 100}
        local = self.shared.replica_budgets[0]
        local.pending_tokens = 100
        posted = []

        async def record_post(replica_id, row):
            posted.append((replica_id, row._id))
            return True

        self.shared.add_posting_request_tasks = record_post
        await self.scheduler._dispatch_schedulable()
        self.assertEqual(posted, [])
        self.assertIn(
            request,
            self.scheduler.global_request_queue.get_all_requests(0),
        )

        local.pending_tokens = 0
        await self.scheduler._dispatch_schedulable()
        self.assertEqual(posted, [(0, request._id)])
        self.assertNotIn(
            request,
            self.scheduler.global_request_queue.get_all_requests(0),
        )


    async def test_work_yield_defers_only_calibrated_misses(self):
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        scheduler._latency_class_observed = True
        request = make_request(104, request_type=0)
        request._cluster_route_feasible = False
        request._risk_calibrated = False

        scheduler._refresh_yield_state(request)
        self.assertFalse(request._ravel_yield_deferred)

        request._risk_calibrated = True
        scheduler._refresh_yield_state(request)
        self.assertTrue(request._ravel_yield_deferred)
        self.assertGreater(
            request._vllm_priority,
            scheduler._DEFERRED_PRIORITY_OFFSET_US,
        )

    async def test_work_yield_collective_keeps_native_engine_order(self):
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(105, request_type=2)
        request._cluster_route_feasible = True
        request._risk_calibrated = True

        scheduler._refresh_yield_state(request)

        self.assertEqual(request._vllm_priority, 0)

    async def test_work_yield_routes_away_from_profiled_service_debt(self):
        self.shared.replica_budgets[0].pending_tokens = 5000
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(106, request_type=1)
        request._ravel_initial_replica = 0
        request._rebind_count = 0
        scheduler.global_request_queue.push(0, request, 0)

        assignment = scheduler._plan_mobile_assignment([request])

        self.assertEqual(assignment[request._id][0], 1)

    async def test_balanced_uses_flow_time_only_after_safety_ties(self):
        request = make_request(1061, request_type=1)
        fast = Quote(
            replica_id=0,
            predicted_ttft_s=0.10,
            point_ttft_s=0.10,
            predicted_objective_s=1.0,
            point_objective_s=1.0,
            predicted_service_s=0.10,
            base_ttft_s=0.10,
            prefix_hit_tokens=0,
            prefix_benefit_s=0.0,
            residual_guard_s=0.0,
            risk_calibrated=True,
            feasible=True,
        )
        balanced = RavelNativeCenterWorkBalancedGlobalScheduler(
            2,
            FakeSharedState(),
            self.args,
        )
        balanced._latency_class_observed = True
        # The faster replica carries more debt, but remains within one current
        # job of the least-loaded replica. Flow time may break this safety tie.
        balanced._service_virtual_load[0] = 0.05
        balanced._service_virtual_load[1] = 0.0
        slower = replace(
            fast,
            replica_id=1,
            predicted_ttft_s=0.20,
            point_ttft_s=0.20,
            predicted_objective_s=2.0,
            point_objective_s=2.0,
            base_ttft_s=0.20,
        )
        with (
            patch.object(
                RavelNativeCenterYieldGlobalScheduler,
                "_choose_initial",
                return_value=fast,
            ),
            patch.object(
                balanced,
                "_quote",
                side_effect=lambda _request, replica_id: (
                    fast if replica_id == 0 else slower
                ),
            ),
        ):
            chosen = balanced._choose_initial(request)

        self.assertEqual(chosen.replica_id, 0)

        unsafe_fast = replace(fast, feasible=False)
        with (
            patch.object(
                RavelNativeCenterYieldGlobalScheduler,
                "_choose_initial",
                return_value=unsafe_fast,
            ),
            patch.object(
                balanced,
                "_quote",
                side_effect=lambda _request, replica_id: (
                    unsafe_fast if replica_id == 0 else slower
                ),
            ),
        ):
            chosen = balanced._choose_initial(request)

        self.assertEqual(chosen.replica_id, 1)

    def test_balanced_one_job_guard_bounds_low_rtt_hotspot(self):
        guard = (
            RavelNativeCenterWorkBalancedGlobalScheduler
            ._within_one_job_guard
        )

        self.assertTrue(guard(2.0, 1.0, 1.0))
        self.assertFalse(guard(2.001, 1.0, 1.0))

    async def test_work_yield_balances_virtual_latency_admissions(self):
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            FakeSharedState(),
            self.args,
        )
        owners = []
        for request_id in range(1061, 1067):
            request = make_request(request_id, request_type=0)
            await scheduler._enqueue(request)
            owner = scheduler._queued_owner(request)
            owners.append(owner)
            scheduler.global_request_queue.del_req(owner, request)

        self.assertLessEqual(
            abs(owners.count(0) - owners.count(1)), 1
        )

    async def test_tighter_tbt_has_larger_virtual_charge(self):
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            FakeSharedState(),
            self.args,
        )
        loose = make_request(
            1067,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        tight = make_request(
            1068,
            request_type=0,
            slo_constraint=(0.8, 0.04, 8.0),
        )

        self.assertAlmostEqual(
            scheduler._latency_virtual_charge(tight, 0),
            2.0 * scheduler._latency_virtual_charge(loose, 0),
        )

    async def test_virtual_latency_replan_is_idempotent(self):
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            FakeSharedState(),
            self.args,
        )
        scheduler._latency_class_observed = True
        request = make_request(1069, request_type=0)
        request._ravel_initial_replica = 0
        request._rebind_count = 0
        scheduler.global_request_queue.push(0, request, 0)

        scheduler._plan_mobile_assignment([request])
        first_total = sum(scheduler._latency_virtual_load.values())
        first_service_total = sum(
            scheduler._service_virtual_load.values()
        )
        scheduler._plan_mobile_assignment([request])
        second_total = sum(scheduler._latency_virtual_load.values())
        second_service_total = sum(
            scheduler._service_virtual_load.values()
        )

        self.assertAlmostEqual(first_total, second_total)
        self.assertAlmostEqual(
            first_service_total, second_service_total
        )

    async def test_mixed_service_virtual_load_balances_flexible_initial(self):
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            FakeSharedState(),
            self.args,
        )
        scheduler._latency_class_observed = True
        owners = []
        for request_id in range(1074, 1080):
            request = make_request(request_id, request_type=1)
            await scheduler._enqueue(request)
            owner = scheduler._queued_owner(request)
            owners.append(owner)
            scheduler.global_request_queue.del_req(owner, request)

        self.assertLessEqual(
            abs(owners.count(0) - owners.count(1)), 1
        )

    async def test_mixed_initial_avoids_predicted_ttft_miss(self):
        shared = FakeSharedState(local_work=12000)
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            shared,
            self.args,
        )
        scheduler._latency_class_observed = True
        request = make_request(1080, request_type=1)

        await scheduler._enqueue(request)

        self.assertEqual(scheduler._queued_owner(request), 1)

    async def test_flexible_work_does_not_create_latency_lane(self):
        scheduler = RavelNativeCenterWorkYieldGlobalScheduler(
            2,
            FakeSharedState(),
            self.args,
        )
        for request_id in range(1070, 1073):
            latency = make_request(request_id, request_type=0)
            scheduler._register_latency_virtual_admission(
                latency, owner_hint=0
            )
        before = dict(scheduler._latency_virtual_load)
        flexible = make_request(1073, request_type=1)
        flexible._ravel_initial_replica = 0
        flexible._rebind_count = 0
        scheduler.global_request_queue.push(0, flexible, 0)

        assignment = scheduler._plan_mobile_assignment([flexible])

        self.assertEqual(assignment[flexible._id][0], 0)
        self.assertEqual(before, scheduler._latency_virtual_load)



    async def test_work_edf_uses_marginal_work_and_no_deferred_offset(self):
        scheduler = RavelNativeCenterWorkEDFGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        scheduler._latency_class_observed = True
        request = make_request(107, request_type=0)
        request._cluster_route_feasible = False
        request._risk_calibrated = True

        scheduler._refresh_yield_state(request)

        self.assertTrue(scheduler.marginal_work_first)
        self.assertTrue(request._ravel_yield_deferred)
        self.assertLess(
            request._vllm_priority,
            scheduler._DEFERRED_PRIORITY_OFFSET_US,
        )


    async def test_latency_first_precedes_flexible_work(self):
        scheduler = RavelNativeCenterLatencyFirstGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        scheduler._latency_class_observed = True
        latency = make_request(108, request_type=0)
        flexible = make_request(109, request_type=1)
        flexible._arrived_at = latency._arrived_at - 10.0

        scheduler._set_engine_priority(latency)
        scheduler._set_engine_priority(flexible)

        self.assertLess(
            scheduler._planning_key(latency),
            scheduler._planning_key(flexible),
        )
        self.assertLess(
            latency._vllm_priority,
            flexible._vllm_priority,
        )


    async def test_latency_fair_uses_equal_priority_within_class(self):
        scheduler = RavelNativeCenterLatencyFairGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        scheduler._latency_class_observed = True
        first = make_request(110, request_type=0)
        second = make_request(111, request_type=0)
        flexible = make_request(112, request_type=1)

        for request in (first, second, flexible):
            scheduler._set_engine_priority(request)

        self.assertEqual(first._vllm_priority, second._vllm_priority)
        self.assertLess(first._vllm_priority, flexible._vllm_priority)


    async def test_selected_admission_does_not_offset_engine_edf(self):
        scheduler = RavelNativeCenterSelectedAdmissionGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        scheduler._latency_class_observed = True
        request = make_request(113, request_type=0)
        request._cluster_route_feasible = False

        scheduler._refresh_yield_state(request)

        self.assertTrue(request._ravel_yield_deferred)
        self.assertLess(
            request._vllm_priority,
            scheduler._DEFERRED_PRIORITY_OFFSET_US,
        )



    def test_dynamic_chunk_priority_carries_latency_tbt(self):
        scheduler = RavelNativeCenterDynamicChunkGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        latency = make_request(
            132,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        flexible = make_request(133, request_type=1)

        scheduler._latency_class_observed = True
        scheduler._set_engine_priority(latency)
        scheduler._set_engine_priority(flexible)

        self.assertAlmostEqual(
            decode_tbt_s(latency._vllm_priority),
            0.08,
        )
        self.assertIsNone(decode_tbt_s(flexible._vllm_priority))

    def test_slo_envelope_is_work_conserving_without_active_latency(self):
        scheduler = RavelNativeCenterSLOEnvelopeGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(124, request_type=1)
        request._num_prefill_tokens = 12000

        self.assertTrue(scheduler._can_admit_to_engine(request, 0))
        self.assertEqual(request._ravel_protected_tbt_s, 0.0)

    def test_slo_envelope_defers_multi_victim_prefill_chunk(self):
        scheduler = RavelNativeCenterSLOEnvelopeGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        active = make_request(
            125,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        active_peer = make_request(
            129,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        self.shared.replica_budgets[0].running_requests = [
            active,
            active_peer,
        ]
        request = make_request(126, request_type=1)
        request._num_prefill_tokens = 1000
        request._arrived_at -= 10.0

        self.assertFalse(scheduler._can_admit_to_engine(request, 0))
        self.assertAlmostEqual(request._ravel_prefill_gap_s, 0.1)
        self.assertEqual(request._ravel_protected_tbt_s, 0.08)
        self.assertEqual(request._ravel_prefill_chunk_tokens, 1000)
        self.assertEqual(request._ravel_admission_victims, 2)

    def test_slo_envelope_allows_one_for_one_yield_trade(self):
        scheduler = RavelNativeCenterSLOEnvelopeGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        active = make_request(
            130,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        self.shared.replica_budgets[0].running_requests = [active]
        request = make_request(131, request_type=1)
        request._num_prefill_tokens = 1000

        self.assertTrue(scheduler._can_admit_to_engine(request, 0))
        self.assertEqual(request._ravel_admission_victims, 1)

    def test_slo_envelope_admits_profiled_short_chunk(self):
        scheduler = RavelNativeCenterSLOEnvelopeGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        active = make_request(
            127,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        self.shared.replica_budgets[0].running_requests = [active]
        request = make_request(128, request_type=1)

        self.assertTrue(scheduler._can_admit_to_engine(request, 0))
        self.assertAlmostEqual(request._ravel_prefill_gap_s, 0.01)


    async def test_center_tbt_uses_rate_monotonic_priority(self):
        scheduler = RavelNativeCenterTBTGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        tight = make_request(
            120,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        tight_peer = make_request(
            121,
            request_type=0,
            slo_constraint=(1.6, 0.08, 16.0),
        )
        loose = make_request(
            122,
            request_type=0,
            slo_constraint=(3.2, 0.32, 32.0),
        )
        flexible = make_request(123, request_type=1)

        for request in (tight, tight_peer, loose, flexible):
            scheduler._set_engine_priority(request)

        self.assertEqual(tight._vllm_priority, tight_peer._vllm_priority)
        self.assertLess(tight._vllm_priority, loose._vllm_priority)
        self.assertLess(loose._vllm_priority, flexible._vllm_priority)


    async def test_objective_edf_prioritizes_collective_ttlt(self):
        scheduler = RavelNativeCenterObjectiveEDFGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(
            119,
            request_type=2,
            slo_constraint=(0.8, 0.08, 24.0),
        )

        scheduler._set_engine_priority(request)

        self.assertEqual(
            request._vllm_priority,
            int((request._arrived_at + 24.0) * 1_000_000),
        )


    async def test_center_density_uses_profiled_service_gain(self):
        scheduler = RavelNativeCenterDensityGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(118, request_type=1)
        request._primary_replica = 0
        request._predicted_objective_s = 1.0
        request._routing_output_tokens_hint_used = 256

        scheduler._set_engine_priority(request)

        self.assertLess(request._vllm_priority, 0)


    async def test_center_cohort_holds_latency_for_one_remote_rtt(self):
        scheduler = RavelNativeCenterCohortGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(117, request_type=0)
        now = time.perf_counter()
        scheduler._recent_arrivals = [
            (now - 0.001, request._client_region),
            (now, request._client_region),
        ]

        self.assertAlmostEqual(
            scheduler._mobile_hold_s(request),
            0.0341,
            places=4,
        )


    async def test_on_time_set_replaces_largest_profiled_job(self):
        scheduler = RavelNativeCenterOnTimeSetGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        arrived_at = time.perf_counter()
        first_long = make_request(
            114,
            request_type=2,
            slo_constraint=(0.8, 0.08, 0.75),
        )
        second_long = make_request(
            115,
            request_type=2,
            slo_constraint=(0.8, 0.08, 0.75),
        )
        short = make_request(
            116,
            request_type=2,
            slo_constraint=(0.8, 0.08, 0.76),
        )
        for request in (first_long, second_long, short):
            request._arrived_at = arrived_at
            request._routing_output_tokens_hint = 1
            request._routing_output_tokens_upper_hint = 1
            request._ravel_initial_replica = 0
            request._rebind_count = 0
            scheduler.global_request_queue.push(0, request, 0)
        first_long._num_prefill_tokens = 7000
        second_long._num_prefill_tokens = 7000
        short._num_prefill_tokens = 1000

        assignment = scheduler._plan_mobile_assignment(
            [first_long, second_long, short]
        )

        self.assertTrue(assignment[short._id][1].feasible)
        self.assertEqual(
            sum(
                assignment[request._id][1].feasible
                for request in (first_long, second_long)
            ),
            1,
        )
        self.assertEqual(
            sum(quote.feasible for _replica, quote in assignment.values()),
            2,
        )

    def test_exact_completion_oracle_is_guarded_and_uses_realized_output(self):
        with patch.dict(
            "os.environ",
            {"RAVEL_DIAGNOSTIC_ALLOW_REALIZED_OUTPUT": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "RAVEL_DIAGNOSTIC"):
                RavelDiagnosticExactCompletionOracleGlobalScheduler(
                    2, self.shared, self.args
                )

        with patch.dict(
            "os.environ",
            {
                "RAVEL_DIAGNOSTIC_ALLOW_REALIZED_OUTPUT": "1",
                "RAVEL_COLLECTIVE_STAGE_OUTPUT_PROFILE": "",
            },
            clear=False,
        ):
            scheduler = (
                RavelDiagnosticExactCompletionOracleGlobalScheduler(
                    2, self.shared, self.args
                )
            )
        request = make_request(139, request_type=2)
        request._output_len = 713
        scheduler._apply_realized_output(request)

        self.assertEqual(request._routing_output_tokens_hint_used, 713)
        self.assertEqual(request._routing_output_tokens_upper_hint_used, 713)

        request._cluster_route_feasible = False
        scheduler._refresh_yield_state(request)
        deferred_priority = request._vllm_priority
        request._cluster_route_feasible = True
        scheduler._refresh_yield_state(request)
        self.assertLess(request._vllm_priority, deferred_priority)

        scheduler._completion_order = "spt"
        request._cluster_route_feasible = False
        scheduler._refresh_yield_state(request)
        self.assertFalse(request._ravel_yield_deferred)
        self.assertEqual(
            request._vllm_priority,
            713 * 100_000_000 + 8_000_000,
        )

    def test_slo_flow_switches_planner_with_slo_semantics(self):
        scheduler = RavelNativeSLOFlowGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(136, request_type=2)

        with patch.object(
            RavelNativeCenterOnTimeSetGlobalScheduler,
            "_plan_mobile_assignment",
            return_value={},
        ) as on_time, patch.object(
            RavelNativeCenterWorkYieldGlobalScheduler,
            "_plan_mobile_assignment",
            return_value={},
        ) as mixed:
            scheduler._plan_mobile_assignment([request])
            on_time.assert_called_once()
            mixed.assert_not_called()

            scheduler._latency_class_observed = True
            scheduler._plan_mobile_assignment([request])
            mixed.assert_called_once()

    def test_slo_flow_exports_tbt_only_after_latency_is_observed(self):
        scheduler = RavelNativeSLOFlowGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        latency = make_request(
            137,
            request_type=0,
            slo_constraint=(0.8, 0.08, 8.0),
        )

        scheduler._set_engine_priority(latency)
        self.assertEqual(latency._vllm_priority, 0)

        scheduler._latency_class_observed = True
        scheduler._set_engine_priority(latency)
        self.assertAlmostEqual(
            decode_tbt_s(latency._vllm_priority),
            0.08,
        )

        flexible = make_request(138, request_type=1)
        scheduler._set_engine_priority(flexible)
        self.assertEqual(flexible._vllm_priority, 0)

    async def test_slo_flow_switches_engine_admission_frontier(self):
        scheduler = RavelNativeSLOFlowGlobalScheduler(
            2,
            self.shared,
            self.args,
        )

        with patch.object(
            RavelNativeMobileInsertionGlobalScheduler,
            "_dispatch_schedulable",
            new_callable=AsyncMock,
        ) as work_conserving, patch.object(
            RavelNativeCenterYieldGlobalScheduler,
            "_dispatch_schedulable",
            new_callable=AsyncMock,
        ) as bounded:
            await scheduler._dispatch_schedulable()
            work_conserving.assert_awaited_once()
            bounded.assert_not_awaited()

            scheduler._latency_class_observed = True
            await scheduler._dispatch_schedulable()
            bounded.assert_awaited_once()

    async def test_slo_flow_completion_only_dispatch_is_work_conserving(self):
        scheduler = RavelNativeSLOFlowGlobalScheduler(
            2,
            self.shared,
            self.args,
        )
        request = make_request(138, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._rebind_count = 0
        scheduler.global_request_queue.push(0, request, 0)
        posted = []

        async def record_post(replica_id, row):
            posted.append((replica_id, row._id))
            return True

        self.shared.add_posting_request_tasks = record_post
        await scheduler._dispatch_schedulable()

        self.assertEqual(posted, [(0, request._id)])
        self.assertNotIn(
            request,
            scheduler.global_request_queue.get_all_requests(0),
        )


class SidecarPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        topology = {
            "default_client_region": "region_a",
            "clusters": [
                {
                    "id": "region_a",
                    "replica_ids": [0],
                    "rtt_ms_by_client_region": {
                        "region_a": 0.5,
                        "region_b": 34.1,
                        "default": 34.1,
                    },
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.001,
                },
                {
                    "id": "region_b",
                    "replica_ids": [1],
                    "rtt_ms_by_client_region": {
                        "region_a": 34.1,
                        "region_b": 0.5,
                        "default": 34.1,
                    },
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.001,
                },
            ],
        }
        topology_path = Path(self.directory.name) / "topology.json"
        topology_path.write_text(json.dumps(topology), encoding="utf-8")
        self.args = SimpleNamespace(
            cluster_topology=str(topology_path),
            cluster_overload_fraction=0.5,
            dh_replica_pending_req_threshold=64,
            dh_rebalance_thredhold=32768,
            dh_rebalance_waiting_latency_thredhold=0.5,
            cluster_rebalance_hysteresis_s=0.02,
            cluster_max_rebalances_per_event=8,
            block_size=16,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            routing_output_tokens_hint=256,
        )
        self.shared = FakeSharedState(local_work=700, local_count=1)
        self.scheduler = (
            RavelNativeSidecarMobileSLOCompleteGlobalScheduler(
                2,
                self.shared,
                self.args,
            )
        )

    async def asyncTearDown(self):
        await self.scheduler.close()
        self.directory.cleanup()

    async def test_snapshot_is_pure_and_does_not_double_count_in_flight(self):
        first = make_request(1, request_type=2)
        second = make_request(2, request_type=2)
        fixed = make_request(3, request_type=2)
        for row in (first, second, fixed):
            row._ravel_initial_replica = 0
            row._ravel_mobile_until = time.perf_counter() + 1.0
            row._rebind_count = 0
            self.scheduler._sidecar_waiting.enqueue(0, row, 0)

        snapshot = self.scheduler._build_planner_snapshot([first, second])
        local = snapshot.replicas[0]
        self.assertEqual(local.pending_requests, 1)
        self.assertEqual(local.engine_work_tokens, 700)
        self.assertEqual(local.fixed_queue_count, 1)
        self.assertEqual(local.fixed_queue_work_tokens, 100)
        self.assertEqual(local.latency_requests, 0)
        self.assertEqual(local.flexible_requests, 2)

        def fail_live_read(*_args, **_kwargs):
            raise AssertionError("planner read live scheduler state")

        self.scheduler._quote = fail_live_read
        self.scheduler._request_work = fail_live_read
        assignment = self.scheduler._plan_mobile_assignment(
            [first, second],
            snapshot,
        )
        self.assertEqual(set(assignment), {1, 2})

    async def test_latency_request_commits_without_a_hold(self):
        self.scheduler._recent_arrivals = [
            (0.0, "region_a"),
            (0.01, "region_a"),
        ]
        request = make_request(4, request_type=0)
        self.assertEqual(self.scheduler._mobile_hold_s(request), 0.0)

    async def test_formal_mode_bounds_cohort_by_engine_capacity(self):
        self.assertEqual(
            self.scheduler.mobile_candidate_limit,
            self.scheduler.pending_request_limit,
        )
        self.assertEqual(
            self.scheduler.mobile_abort_hysteresis_s,
            float("inf"),
        )

    async def test_control_rtt_uses_current_topology(self):
        self.assertAlmostEqual(self.scheduler.control_horizon_s, 0.0341)


    async def test_engine_buffer_is_derived_from_quantum_profile_and_rtt(self):
        shared = FakeSharedState()
        with patch.dict(
            "os.environ", {"RAVEL_SIDECAR_ENGINE_BUFFER": "1"}
        ):
            scheduler = RavelNativeSidecarMobileSLOCompleteGlobalScheduler(
                2,
                shared,
                self.args,
            )
        try:
            expected = 16384 + math.ceil(0.0341 / 0.0001)
            self.assertEqual(
                scheduler._engine_buffer_tokens,
                {0: expected, 1: expected},
            )
        finally:
            await scheduler.close()

    async def test_engine_priority_is_absolute_visible_slo_deadline(self):
        arrived_at = time.perf_counter()
        tight = make_request(
            19,
            request_type=2,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        loose = make_request(
            20,
            request_type=2,
            slo_constraint=(0.8, 0.08, 16.0),
        )
        tight._arrived_at = arrived_at
        loose._arrived_at = arrived_at
        await self.scheduler._enqueue(tight)
        await self.scheduler._enqueue(loose)

        self.assertLess(tight._vllm_priority, loose._vllm_priority)
        self.assertEqual(
            loose._vllm_priority - tight._vllm_priority,
            8_000_000,
        )
        tight._ravel_yield_deferred = True
        self.scheduler._set_engine_priority(tight)
        self.assertGreater(tight._vllm_priority, loose._vllm_priority)
        self.assertEqual(
            tight._vllm_priority
            - int((arrived_at + 8.0) * 1_000_000),
            self.scheduler._DEFERRED_PRIORITY_OFFSET_US,
        )
    async def test_engine_slo_yield_priority_orders_normalized_demand(self):
        tight = make_request(
            21,
            request_type=2,
            slo_constraint=(0.8, 0.08, 8.0),
        )
        loose = make_request(
            22,
            request_type=2,
            slo_constraint=(0.8, 0.08, 16.0),
        )
        tight._predicted_objective_s = 4.0
        loose._predicted_objective_s = 4.0
        with patch.dict(
            "os.environ", {"RAVEL_ENGINE_PRIORITY_MODE": "slo_yield"}
        ):
            self.scheduler._set_engine_priority(tight)
            self.scheduler._set_engine_priority(loose)

        self.assertEqual(tight._vllm_priority, 500_000_000)
        self.assertEqual(loose._vllm_priority, 250_000_000)
        self.assertLess(loose._vllm_priority, tight._vllm_priority)

    async def test_jitserve_density_uses_profiled_service_rate(self):
        short = make_request(23, request_type=2)
        long = make_request(24, request_type=2)
        short._num_prefill_tokens = 100
        long._num_prefill_tokens = 1000
        for row in (short, long):
            row._primary_replica = 0
            row._predicted_objective_s = 4.0
            row._routing_output_tokens_hint_used = 256

        with patch.dict(
            "os.environ",
            {"RAVEL_ENGINE_PRIORITY_MODE": "jitserve_density"},
        ):
            self.scheduler._set_engine_priority(short)
            self.scheduler._set_engine_priority(long)

        self.assertLess(long._vllm_priority, short._vllm_priority)

    async def test_latency_triggered_edf_uses_only_observed_slo_types(self):
        collective_before = make_request(
            23,
            request_type=2,
            slo_constraint=(0.8, 0.08, 16.0),
        )
        latency = make_request(
            24,
            request_type=0,
            slo_constraint=(0.8, 0.08, 16.0),
        )
        collective_after = make_request(
            25,
            request_type=2,
            slo_constraint=(0.8, 0.08, 16.0),
        )
        with patch.dict(
            "os.environ",
            {"RAVEL_ENGINE_PRIORITY_MODE": "latency_triggered_edf"},
        ):
            await self.scheduler._enqueue(collective_before)
            await self.scheduler._enqueue(latency)
            await self.scheduler._enqueue(collective_after)

        self.assertEqual(collective_before._vllm_priority, 0)
        self.assertGreater(latency._vllm_priority, 0)
        self.assertGreater(collective_after._vllm_priority, 0)
        self.assertTrue(self.scheduler._latency_class_observed)

    async def test_semantic_adaptive_edf_uses_fast_rtt_and_slo_type(self):
        slow_gap = make_request(27, request_type=0)
        now = time.perf_counter()
        self.scheduler._recent_arrivals = [(now - 0.040, "region_a")]
        with patch.dict(
            "os.environ",
            {"RAVEL_ENGINE_PRIORITY_MODE": "slo_semantic_adaptive"},
        ):
            await self.scheduler._enqueue(slow_gap)
        self.assertFalse(self.scheduler._deadline_ordering_active)
        self.assertEqual(slow_gap._vllm_priority, 0)

        fast_gap = make_request(28, request_type=1)
        now = time.perf_counter()
        self.scheduler._recent_arrivals = [(now - 0.010, "region_a")]
        with patch.dict(
            "os.environ",
            {"RAVEL_ENGINE_PRIORITY_MODE": "slo_semantic_adaptive"},
        ):
            await self.scheduler._enqueue(fast_gap)
        self.assertTrue(self.scheduler._deadline_ordering_active)
        self.assertGreater(fast_gap._vllm_priority, 0)

        self.scheduler._deadline_ordering_active = False
        collective = make_request(29, request_type=2)
        now = time.perf_counter()
        self.scheduler._recent_arrivals = [(now - 0.001, "region_a")]
        with patch.dict(
            "os.environ",
            {"RAVEL_ENGINE_PRIORITY_MODE": "slo_semantic_adaptive"},
        ):
            await self.scheduler._enqueue(collective)
        self.assertFalse(self.scheduler._deadline_ordering_active)
        self.assertEqual(collective._vllm_priority, 0)


    async def test_mobile_candidates_exclude_submitted_requests(self):
        waiting = make_request(5, request_type=2)
        submitted = make_request(6, request_type=2)
        for row in (waiting, submitted):
            row._ravel_initial_replica = 0
            row._ravel_mobile_until = time.perf_counter() + 1.0
            row._rebind_count = 0
            self.scheduler._sidecar_waiting.enqueue(0, row, 0)

        submitted_entry = self.scheduler._sidecar_waiting.entry(0, 6)
        claimed = self.scheduler._sidecar_waiting.claim(
            0,
            6,
            expected_version=submitted_entry.version,
        )
        self.scheduler._in_flight[(6, 0)] = (0, claimed, False)

        self.assertEqual(
            [row._id for row in self.scheduler._mobile_candidates()],
            [5],
        )
    async def test_abort_mode_includes_submitted_unstarted_request(self):
        request = make_request(23, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._sidecar_dispatched_at = time.perf_counter() - 1.0
        request._rebind_count = 0
        entry = self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        claimed = self.scheduler._sidecar_waiting.claim(
            0,
            request._id,
            expected_version=entry.version,
        )
        self.scheduler._in_flight[(request._id, 0)] = (
            0,
            claimed,
            False,
        )
        self.scheduler.mobile_abort_hysteresis_s = 0.0

        self.assertEqual(self.scheduler._mobile_candidates(), [])
        self.assertEqual(
            [row._id for row in self.scheduler._hard_rebind_candidates()],
            [request._id],
        )

    async def test_latency_triggered_edf_disables_hard_rebind_churn(self):
        request = make_request(26, request_type=2)
        request._ravel_initial_replica = 0
        request._sidecar_dispatched_at = time.perf_counter() - 1.0
        request._rebind_count = 0
        entry = self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        claimed = self.scheduler._sidecar_waiting.claim(
            0,
            request._id,
            expected_version=entry.version,
        )
        self.scheduler._in_flight[(request._id, 0)] = (
            0,
            claimed,
            False,
        )
        self.scheduler.mobile_abort_hysteresis_s = 0.0
        self.scheduler._latency_class_observed = True

        with patch.dict(
            "os.environ",
            {"RAVEL_ENGINE_PRIORITY_MODE": "latency_triggered_edf"},
        ):
            self.assertEqual(self.scheduler._hard_rebind_candidates(), [])

    async def test_hard_rebind_plan_requires_source_infeasible_target_feasible(self):
        request = make_request(
            26,
            request_type=2,
            slo_constraint=(0.8, 0.08, 0.5),
        )
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._sidecar_dispatched_at = time.perf_counter() - 1.0
        request._rebind_count = 0
        entry = self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        claimed = self.scheduler._sidecar_waiting.claim(
            0,
            request._id,
            expected_version=entry.version,
        )
        self.scheduler._in_flight[(request._id, 0)] = (
            0,
            claimed,
            False,
        )
        local = self.shared.replica_budgets[0]
        local.pending_count = 2
        local.pending_tokens = 10100
        local.pending_requests = [request, SimpleNamespace(_id=-26)]

        snapshot = self.scheduler._build_planner_snapshot([request])
        assignment = self.scheduler._plan_hard_slo_rescues(snapshot)

        self.assertEqual(assignment[request._id][0], 1)
        self.assertTrue(assignment[request._id][1].feasible)

        local.pending_count = 1
        local.pending_tokens = request._num_prefill_tokens
        local.pending_requests = [request]
        snapshot = self.scheduler._build_planner_snapshot([request])
        self.assertEqual(
            self.scheduler._plan_hard_slo_rescues(snapshot),
            {},
        )

    async def test_snapshot_counts_in_flight_candidate_exactly_once(self):
        request = make_request(24, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._sidecar_dispatched_at = time.perf_counter() - 1.0
        request._rebind_count = 0
        entry = self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        claimed = self.scheduler._sidecar_waiting.claim(
            0,
            request._id,
            expected_version=entry.version,
        )
        self.scheduler._in_flight[(request._id, 0)] = (
            0,
            claimed,
            False,
        )
        local = self.shared.replica_budgets[0]
        local.pending_count = 1
        local.pending_tokens = request._num_prefill_tokens
        local.pending_requests = [request]

        snapshot = self.scheduler._build_planner_snapshot([request])
        self.assertTrue(snapshot.requests[0].in_flight)
        self.assertEqual(snapshot.replicas[0].pending_requests, 0)
        self.assertEqual(snapshot.replicas[0].engine_work_tokens, 0)

    async def test_in_flight_rebind_is_exactly_once_and_attempt_scoped(self):
        request = make_request(25, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._sidecar_dispatched_at = time.perf_counter() - 1.0
        request._rebind_count = 0
        entry = self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        claimed = self.scheduler._sidecar_waiting.claim(
            0,
            request._id,
            expected_version=entry.version,
        )
        self.scheduler._in_flight[(request._id, 0)] = (
            0,
            claimed,
            False,
        )
        self.scheduler._request_by_id[request._id] = request
        local = self.shared.replica_budgets[0]
        local.pending_count = 1
        local.pending_tokens = request._num_prefill_tokens
        local.pending_requests = [request]
        calls = []

        async def abort_in_flight(row, replica_id, attempt):
            calls.append((row._id, replica_id, attempt))
            return True

        self.shared.abort_in_flight = abort_in_flight
        snapshot = self.scheduler._build_planner_snapshot([request])
        planned = snapshot.requests[0]
        target_quote = self.scheduler._planner_quote(
            snapshot,
            planned,
            snapshot.replicas[1],
            1,
            0,
            0,
        )

        moved = await self.scheduler._apply_mobile_assignment(
            [request],
            {request._id: (1, target_quote)},
            snapshot,
        )

        self.assertEqual(moved, 1)
        self.assertEqual(calls, [(request._id, 0, 0)])
        self.assertEqual(request._attempt, 1)
        self.assertEqual(request._rebind_count, 1)
        self.assertEqual(request._abort_attempts, 1)
        self.assertEqual(self.scheduler._sidecar_waiting.owner(request._id), 1)
        self.assertNotIn((request._id, 0), self.scheduler._in_flight)

        await self.scheduler.on_request_terminal(request._id, 0)
        self.assertEqual(request._abort_completed, 1)
        self.assertNotIn(request._id, self.scheduler._aborted_attempts)


    async def test_planner_does_not_treat_max_num_seqs_as_admission_limit(self):
        request = make_request(7, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() + 1.0
        request._rebind_count = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        local = self.shared.replica_budgets[0]
        local.pending_count = self.scheduler.pending_request_limit
        local.pending_requests = [
            SimpleNamespace(_id=-(index + 1))
            for index in range(local.pending_count)
        ]

        snapshot = self.scheduler._build_planner_snapshot([request])
        first = self.scheduler._plan_snapshot_greedy(snapshot)
        second = self.scheduler._plan_snapshot_greedy(snapshot)

        self.assertEqual(first, second)
        self.assertFalse(snapshot.requests[0].enforce_admission_limit)

    async def test_engine_token_buffer_holds_only_excess_prompt_work(self):
        request = make_request(27, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._rebind_count = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        self.scheduler._engine_buffer_tokens = {0: 100, 1: 100}
        local = self.shared.replica_budgets[0]
        local.pending_tokens = 100
        posted = []

        async def record_post(replica_id, row):
            posted.append((replica_id, row._id))
            return True

        self.shared.add_posting_request_tasks = record_post
        await self.scheduler._dispatch_schedulable()
        self.assertEqual(posted, [])
        self.assertEqual(self.scheduler._sidecar_waiting.owner(request._id), 0)

        local.pending_tokens = 0
        await self.scheduler._dispatch_schedulable()
        self.assertEqual(posted, [(0, request._id)])
        self.assertIsNone(self.scheduler._sidecar_waiting.owner(request._id))

    async def test_engine_pending_count_does_not_block_sidecar_dispatch(self):
        request = make_request(12, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._rebind_count = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)

        async def engine_queue_is_deep():
            return (
                65536,
                0.0,
                0.0,
                self.scheduler.pending_request_limit,
                0.0,
            )

        posted = []

        async def record_post(replica_id, row):
            posted.append((replica_id, row._id))
            return True

        self.shared.replica_budgets[0].get_load_states = engine_queue_is_deep
        self.shared.add_posting_request_tasks = record_post
        await self.scheduler._dispatch_schedulable()

        self.assertIsNone(self.scheduler._sidecar_waiting.owner(request._id))
        self.assertEqual(posted, [(0, request._id)])

    async def test_expired_mobile_waits_for_completed_joint_plan(self):
        self.scheduler._replan_task.cancel()
        await asyncio.gather(
            self.scheduler._replan_task,
            return_exceptions=True,
        )
        request = make_request(13, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_hold_s = 0.0166
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._rebind_count = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        posted = []

        async def record_post(replica_id, row):
            posted.append((replica_id, row._id))
            return True

        self.shared.add_posting_request_tasks = record_post
        await self.scheduler._dispatch_schedulable()

        self.assertEqual(self.scheduler._sidecar_waiting.owner(request._id), 0)
        self.assertEqual(posted, [])
        self.assertTrue(self.scheduler._replan_event.is_set())

        request._ravel_planned_generation = 1
        await self.scheduler._dispatch_schedulable()

        self.assertIsNone(self.scheduler._sidecar_waiting.owner(request._id))
        self.assertEqual(posted, [(0, request._id)])


    async def test_greedy_prioritizes_slo_then_marginal_pressure(self):
        request = make_request(11, request_type=0)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() + 1.0
        request._rebind_count = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)

        snapshot = self.scheduler._build_planner_snapshot([request])
        pressured = replace(
            snapshot,
            replicas=(
                replace(snapshot.replicas[0], pending_requests=10),
                replace(snapshot.replicas[1], pending_requests=0),
            ),
        )
        self.assertEqual(
            self.scheduler._plan_snapshot_greedy(pressured)[11][0],
            1,
        )

        local_only_request = replace(
            pressured.requests[0],
            rtt_by_replica_s=(0.0005, 2.0),
        )
        local_only = replace(
            pressured,
            requests=(local_only_request,),
        )
        self.assertEqual(
            self.scheduler._plan_snapshot_greedy(local_only)[11][0],
            0,
        )

    async def test_yield_planner_replaces_long_job_with_shorter_job(self):
        long_request = make_request(
            14,
            request_type=2,
            slo_constraint=(0.1, 0.1, 1.0),
        )
        short_request = make_request(
            15,
            request_type=2,
            slo_constraint=(0.1, 0.1, 1.0),
        )
        long_request._num_prefill_tokens = 900
        short_request._num_prefill_tokens = 200
        for row in (long_request, short_request):
            row._ravel_initial_replica = 0
            row._ravel_mobile_until = time.perf_counter() + 1.0
            row._rebind_count = 0
            self.scheduler._sidecar_waiting.enqueue(0, row, 0)

        snapshot = self.scheduler._build_planner_snapshot(
            [long_request, short_request]
        )
        local = replace(
            snapshot.replicas[0],
            engine_work_tokens=0,
            pending_requests=0,
            running_requests=0,
            fixed_queue_count=0,
            fixed_queue_work_tokens=0,
            prefill_tpot_s=0.001,
            prefill_intercept_s=0.0,
            decode_tpot_by_sequence_s=(0.001, 0.001, 0.001),
        )

        def local_only(request_snapshot, prompt_tokens):
            return replace(
                request_snapshot,
                limit_s=1.0,
                ttft_limit_s=0.1,
                tbt_limit_s=0.1,
                prompt_tokens=prompt_tokens,
                output_tokens_hint=1,
                rtt_by_replica_s=(0.0,),
                prefix_hit_by_replica=(0,),
                residual_guard_by_replica_s=(0.0,),
                decision_guard_by_replica_s=(0.0,),
                risk_ready_by_replica=(True,),
                fixed_work_before_by_replica=(0,),
                fixed_requests_before_by_replica=(0,),
            )

        one_replica = replace(
            snapshot,
            replicas=(local,),
            requests=(
                local_only(snapshot.requests[0], 900),
                local_only(snapshot.requests[1], 200),
            ),
        )
        plan = self.scheduler._plan_snapshot_greedy(one_replica)

        self.assertFalse(plan[14][1].feasible)
        self.assertTrue(plan[15][1].feasible)

    async def test_transfer_occurs_only_once_at_hold_expiry(self):
        request = make_request(10, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() + 1.0
        request._rebind_count = 0
        request._ravel_soft_moves = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)

        early = self.scheduler._build_planner_snapshot([request])
        remote_quote = self.scheduler._planner_quote(
            early,
            early.requests[0],
            early.replicas[1],
            1,
            0,
            0,
        )
        moved = await self.scheduler._apply_mobile_assignment(
            [request],
            {10: (1, remote_quote)},
            early,
        )
        self.assertEqual(moved, 0)
        self.assertEqual(self.scheduler._sidecar_waiting.owner(10), 0)

        request._ravel_mobile_until = time.perf_counter() - 1.0
        expired = self.scheduler._build_planner_snapshot([request])
        remote_quote = self.scheduler._planner_quote(
            expired,
            expired.requests[0],
            expired.replicas[1],
            1,
            0,
            0,
        )
        moved = await self.scheduler._apply_mobile_assignment(
            [request],
            {10: (1, remote_quote)},
            expired,
        )
        self.assertEqual(moved, 1)
        self.assertEqual(self.scheduler._sidecar_waiting.owner(10), 1)
        self.assertEqual(request._rebind_count, 1)
        self.assertEqual(request._ravel_soft_moves, 1)

    async def test_dispatch_priority_protects_predicted_slo_yield(self):
        feasible = make_request(8, request_type=2)
        infeasible = make_request(9, request_type=2)
        feasible._ravel_yield_deferred = False
        infeasible._ravel_yield_deferred = True

        self.assertLess(
            self.scheduler._dispatch_priority(
                SimpleNamespace(request=feasible)
            ),
            self.scheduler._dispatch_priority(
                SimpleNamespace(request=infeasible)
            ),
        )

    async def test_yield_deferral_is_bounded_by_one_control_horizon(self):
        request = make_request(16, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() - 1.0
        request._ravel_planned_generation = 1
        request._ravel_yield_deferred = True
        request._ravel_yield_defer_until = time.perf_counter() + 1.0
        request._rebind_count = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        posted = []

        async def record_post(replica_id, row):
            posted.append((replica_id, row._id))
            return True

        self.shared.add_posting_request_tasks = record_post
        await self.scheduler._dispatch_schedulable()
        self.assertEqual(self.scheduler._sidecar_waiting.owner(request._id), 0)
        self.assertEqual(posted, [])

        request._ravel_yield_defer_until = time.perf_counter() - 1.0
        await self.scheduler._dispatch_schedulable()
        self.assertIsNone(self.scheduler._sidecar_waiting.owner(request._id))
        self.assertEqual(posted, [(0, request._id)])


    async def test_prefix_benefit_breaks_safe_ties_without_relaxing_bound(self):
        request = make_request(30, request_type=2)
        request._ravel_initial_replica = 0
        request._ravel_mobile_until = time.perf_counter() + 1.0
        request._rebind_count = 0
        self.scheduler._sidecar_waiting.enqueue(0, request, 0)
        snapshot = self.scheduler._build_planner_snapshot([request])
        first = replace(
            snapshot.replicas[0],
            replica_id=0,
            engine_work_tokens=0,
            pending_requests=0,
            running_requests=0,
            fixed_queue_count=0,
            fixed_queue_work_tokens=0,
        )
        second = replace(first, replica_id=1, cluster_id="region_b")
        planned = replace(
            snapshot.requests[0],
            rtt_by_replica_s=(0.0, 0.0),
            prefix_hit_by_replica=(0, 64),
            residual_guard_by_replica_s=(0.0, 0.0),
            decision_guard_by_replica_s=(0.0, 0.0),
            risk_ready_by_replica=(True, True),
            fixed_work_before_by_replica=(0, 0),
            fixed_requests_before_by_replica=(0, 0),
        )
        tied = replace(
            snapshot,
            replicas=(first, second),
            requests=(planned,),
        )
        cold = self.scheduler._planner_quote(tied, planned, first, 0, 0, 0)
        warm = self.scheduler._planner_quote(tied, planned, second, 1, 0, 0)

        self.assertEqual(cold.feasible, warm.feasible)
        self.assertAlmostEqual(
            cold.predicted_objective_s,
            warm.predicted_objective_s,
        )
        self.assertGreater(warm.prefix_benefit_s, cold.prefix_benefit_s)
        self.assertEqual(
            self.scheduler._plan_snapshot_greedy(tied)[request._id][0],
            1,
        )

    async def test_collective_stage_profile_uses_semantics_not_dataset(self):
        self.scheduler._collective_stage_min = 2
        self.scheduler._collective_stage_expected_output_tokens = {1: 658}
        self.scheduler._collective_stage_upper_output_tokens = {1: 1024}
        workflow = make_request(31, request_type=2)
        workflow._stage_id = 1
        workflow._stage_num = 6
        workflow._output_len = 37
        self.scheduler._apply_collective_stage_output_profile(workflow)
        self.assertEqual(workflow._routing_output_tokens_hint, 658)
        self.assertEqual(
            workflow._routing_output_tokens_upper_hint, 1024
        )
        self.assertEqual(workflow._output_len, 37)

        single_stage = make_request(32, request_type=2)
        single_stage._stage_id = 1
        single_stage._stage_num = 1
        self.scheduler._apply_collective_stage_output_profile(single_stage)
        self.assertEqual(single_stage._routing_output_tokens_hint, 256)

        latency = make_request(33, request_type=0)
        latency._stage_id = 1
        latency._stage_num = 6
        self.scheduler._apply_collective_stage_output_profile(latency)
        self.assertEqual(latency._routing_output_tokens_hint, 256)

class PrefixAccountingTests(unittest.TestCase):
    def test_full_shadow_hit_still_charges_one_token(self):
        replica = Replica.__new__(Replica)
        replica.block_size = 16
        replica.get_num_cached_tokens = lambda _tokens: 32
        self.assertEqual(replica.get_num_recompute_token_ids(list(range(32))), 1)

    def test_shadow_query_refreshes_lru_recency(self):
        cache = PrefixLRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.query(1), 1)
        evicted, _ = cache.pop()
        self.assertEqual(evicted, 2)


class OutputDemandTests(unittest.TestCase):
    def test_output_hint_never_drops_below_public_fallback(self):
        state = SharedState.__new__(SharedState)
        state._output_quantile = 0.75
        state._output_min_samples = 4
        state._output_history = defaultdict(list)
        state._output_history[1].extend([10, 20, 30])

        self.assertEqual(state.get_routing_output_hint(1, 256), 256)
        state._output_history[1].append(40)
        self.assertEqual(state.get_routing_output_hint(1, 256), 256)
        state._output_history[1].extend([400, 400, 400, 400])
        self.assertEqual(state.get_routing_output_hint(1, 256), 400)


class ReplicaResidualTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_token_updates_residual_without_output_information(self):
        replica = Replica.__new__(Replica)
        request = make_request()
        request._point_predicted_ttft_s = 0.2
        request._point_predicted_objective_s = 0.2
        request._predicted_ttft_s = 0.4  # Includes a prior 0.2s guard.
        request._ttft_residual_track = True
        replica.id = 0
        replica.replica_slo_budget = 65536
        replica.current_budget = 65436
        replica.pending_requests = [request]
        replica.running_requests = []
        replica.lock = asyncio.Lock()
        replica.ttft_residual_samples = 0
        replica.ttft_residual_updated_at = -1.0
        replica.ttft_residual_history = deque(maxlen=256)
        replica.save_prefill_token_ids = lambda request_id, token_ids: 0

        success = await replica.complete_request_prefill(0, request, 0.5)

        self.assertTrue(success)
        self.assertEqual(replica.ttft_residual_samples, 1)
        self.assertEqual(list(replica.ttft_residual_history), [0.3])
        self.assertAlmostEqual(request._ttft_prediction_residual_s, 0.3)

    async def test_completion_updates_e2e_residual_by_request_type(self):
        replica = Replica.__new__(Replica)
        replica.lock = asyncio.Lock()
        replica.objective_residual_history = {
            0: deque(maxlen=256),
            1: deque(maxlen=256),
            2: deque(maxlen=256),
        }
        request = make_request(request_type=1)
        request._ttft_residual_track = True
        request._point_predicted_objective_s = 1.0

        await replica.complete_request_objective(request, 1.5)

        self.assertEqual(list(replica.objective_residual_history[1]), [0.5])
        self.assertEqual(list(replica.objective_residual_history[0]), [])
        self.assertAlmostEqual(request._objective_prediction_residual_s, 0.5)


class HindsightJointPlanGuardTests(unittest.TestCase):
    def test_future_plan_requires_explicit_diagnostic_guard(self):
        with patch.dict(
            "os.environ",
            {"RAVEL_DIAGNOSTIC_ALLOW_FUTURE_PLAN": ""},
        ):
            with self.assertRaisesRegex(RuntimeError, "future-aware"):
                RavelHindsightJointPlanGlobalScheduler(0, None, None)

if __name__ == "__main__":
    unittest.main()
