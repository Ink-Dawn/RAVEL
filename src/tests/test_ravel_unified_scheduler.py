from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from unittest.mock import patch
from dualmap.cluster.slo import COLLECTIVE
from dualmap.entities.request import Request
from dualmap.scheduler.global_scheduler.ravel_native_global_scheduler import (
    RavelNativeCenterWorkBalancedGlobalScheduler,
    RavelNativeCenterWorkYieldGlobalScheduler,
    RavelNativeCenterYieldGlobalScheduler,
)
from dualmap.scheduler.global_scheduler.ravel_unified_global_scheduler import (
    RavelUnifiedBalancedGlobalScheduler,
    RavelUnifiedBoundedLedgerGlobalScheduler,
    RavelUnifiedFlowPlacementGlobalScheduler,
    RavelUnifiedGlobalScheduler,
    RavelUnifiedNoLedgerGlobalScheduler,
    RavelUnifiedPhasePriorityGlobalScheduler,
)
from ravel_engine_adapter.protocol import (
    decode_deadline_budget_s,
    decode_service_s,
    decode_tbt_s,
    is_deferred_phase_priority,
    is_phase_aware_priority,
)


class FakeReplica:
    def __init__(self, pending_count: int = 0):
        self.pending_count = pending_count
        self.pending_requests = [
            SimpleNamespace(_id=-(index + 1))
            for index in range(pending_count)
        ]
        self.running_requests = []
        self.ttft_residual_history = []

    def get_num_pending_req(self) -> int:
        return self.pending_count

    def get_num_running_req(self) -> int:
        return len(self.running_requests)

    def get_num_recompute_token_ids(self, input_ids):
        return len(input_ids)


class FakeSharedState:
    def __init__(self, pending_per_replica: int = 0):
        self.num_replicas = 2
        self.replica_budgets = {
            0: FakeReplica(pending_per_replica),
            1: FakeReplica(pending_per_replica),
        }
        self.callback = None
        self.terminal_callback = None

    def set_scheduler_callback(self, callback):
        self.callback = callback

    def set_request_terminal_callback(self, callback):
        self.terminal_callback = callback

    def get_num_actual_pending_tokens_replica(self, replica_id: int) -> int:
        return 0

    def get_routing_output_hint(
        self, request_type: int, fallback_tokens: int
    ) -> int:
        return fallback_tokens


def make_request(
    *,
    request_id: int = 1,
    stage_id: int = 1,
    stage_num: int = 6,
    output_len: int = 1024,
) -> Request:
    return Request(
        request_id=request_id,
        dataset_type="opaque",
        native_session_id=0,
        session_id="session",
        hash_session_id=f"prefix-{request_id}",
        round_id=stage_id,
        prompts="prompt",
        input_ids=list(range(100)),
        num_prefill_tokens=100,
        actual_num_prefill_tokens=100,
        output_len=output_len,
        over_flow=False,
        n=1,
        temperature=0,
        top_p=1,
        max_tokens=256,
        stream=True,
        arrived_at=time.perf_counter(),
        time_interval=0.0,
        hash_prefix_len=0,
        request_type=COLLECTIVE,
        slo_constraint=(0.8, 0.08, 16.0),
        client_region="region_a",
        routing_output_tokens_hint=256,
        stage_id=stage_id,
        stage_num=stage_num,
    )


class RavelUnifiedSchedulerTests(unittest.TestCase):
    def setUp(self):
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
                    },
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.001,
                },
            ],
        }
        topology_path = Path(self.directory.name) / "topology.json"
        topology_path.write_text(json.dumps(topology), encoding="utf-8")
        profile = {
            "schema_version": 2,
            "kind": "semantic-collective-stage-output-profile",
            "selector": {"request_type": COLLECTIVE, "min_stage_num": 2},
            "stages": {
                "1": {
                    "samples": 100,
                    "mean_tokens": 100.0,
                    "q95_tokens": 900,
                }
            },
            "contexts": {
                "1:6": {
                    "samples": 60,
                    "mean_tokens": 120.0,
                    "q95_tokens": 700,
                }
            },
        }
        profile_path = Path(self.directory.name) / "output-profile.json"
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
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
            ravel_collective_stage_output_profile=str(profile_path),
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_completion_pressure_uses_causal_arrival_rate(self):
        shared = FakeSharedState()
        scheduler = RavelUnifiedGlobalScheduler(2, shared, self.args)
        scheduler._workflow_completion_observed = True
        base = time.perf_counter()
        requests = []
        for index in range(scheduler._arrival_pressure_min_samples):
            request = make_request(request_id=100 + index)
            request._arrived_at = base + index * 0.1
            scheduler._observe_completion_arrival(request)
            requests.append(request)
        scheduler.global_request_queue.push(0, requests[-1], 0)

        self.assertEqual(scheduler._total_sequence_capacity, 16)
        self.assertLess(scheduler._outstanding_count(), 8)
        self.assertTrue(scheduler._arrival_pressure_observed)
        self.assertTrue(scheduler._update_completion_pressure())
        self.assertTrue(scheduler._completion_pressure_active)

    def test_low_arrival_pressure_keeps_default_profile_mode(self):
        scheduler = RavelUnifiedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._workflow_completion_observed = True
        base = time.perf_counter()
        requests = []
        for index in range(scheduler._arrival_pressure_min_samples):
            request = make_request(request_id=200 + index)
            request._arrived_at = base + index * 1.0
            scheduler._observe_completion_arrival(request)
            requests.append(request)
        scheduler.global_request_queue.push(0, requests[-1], 0)

        self.assertFalse(scheduler._arrival_pressure_observed)
        self.assertFalse(scheduler._update_completion_pressure())

    def test_completion_pressure_replans_a_lone_candidate(self):
        scheduler = RavelUnifiedNoLedgerGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._workflow_completion_observed = True
        scheduler._arrival_pressure_observed = True
        scheduler._completion_pressure_active = True
        request = make_request(request_id=499)
        scheduler.global_request_queue.push(0, request, 0)

        self.assertTrue(scheduler.plan_single_mobile_candidate)

    def test_mixed_slo_semantics_disable_completion_pressure_mode(self):
        shared = FakeSharedState(pending_per_replica=8)
        scheduler = RavelUnifiedGlobalScheduler(2, shared, self.args)
        scheduler._workflow_completion_observed = True
        scheduler._non_workflow_observed = True

        self.assertFalse(scheduler._update_completion_pressure())

    def test_mixed_semantics_use_dynamic_chunk_planner(self):
        scheduler = RavelUnifiedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._non_workflow_observed = True
        with patch.object(
            RavelNativeCenterWorkYieldGlobalScheduler,
            "_plan_mobile_assignment",
            autospec=True,
            return_value={},
        ) as dynamic:
            result = scheduler._plan_mobile_assignment([])

        self.assertEqual(result, {})
        dynamic.assert_called_once_with(scheduler, [], None)

    def test_completion_pressure_uses_work_yield_planner(self):
        scheduler = RavelUnifiedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._workflow_completion_observed = True
        scheduler._arrival_pressure_observed = True
        request = make_request(request_id=777)
        scheduler.global_request_queue.push(0, request, 0)
        with patch.object(
            RavelNativeCenterWorkYieldGlobalScheduler,
            "_plan_mobile_assignment",
            autospec=True,
            return_value={},
        ) as work_yield:
            result = scheduler._plan_mobile_assignment([request])

        self.assertEqual(result, {})
        work_yield.assert_called_once_with(scheduler, [request], None)
        self.assertEqual(request._ravel_soft_plan_generation, -1)
        self.assertFalse(request._ravel_soft_admission_active)

    def test_full_dispatches_through_center_yield(self):
        scheduler = RavelUnifiedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        with patch.object(
            RavelNativeCenterYieldGlobalScheduler,
            "_dispatch_schedulable",
            autospec=True,
        ) as center_yield:
            asyncio.run(scheduler._dispatch_schedulable())

        center_yield.assert_awaited_once_with(scheduler)
        self.assertFalse(hasattr(scheduler, "_dispatch_protected"))

    def test_balanced_mixed_semantics_use_flow_time_planner(self):
        scheduler = RavelUnifiedBalancedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._non_workflow_observed = True
        with patch.object(
            RavelNativeCenterWorkBalancedGlobalScheduler,
            "_plan_mobile_assignment",
            autospec=True,
            return_value={},
        ) as balanced:
            result = scheduler._plan_mobile_assignment([])

        self.assertEqual(result, {})
        self.assertTrue(scheduler.flow_time_before_virtual_balance)
        balanced.assert_called_once_with(scheduler, [], None)

    def test_release_unified_keeps_slo_first_secondary_order(self):
        scheduler = RavelUnifiedGlobalScheduler(
            2, FakeSharedState(), self.args
        )

        self.assertFalse(scheduler.flow_time_before_virtual_balance)

    def test_active_completion_ledger_releases_terminal_work(self):
        shared = FakeSharedState()
        scheduler = RavelUnifiedGlobalScheduler(2, shared, self.args)
        request = make_request(request_id=91)
        scheduler._register_service_virtual_admission(
            request, owner_hint=0
        )
        scheduler._completion_virtual_requests[request._id] = request
        charge = request._ravel_service_virtual_charge
        self.assertAlmostEqual(scheduler._service_virtual_load[0], charge)

        asyncio.run(scheduler.on_request_terminal(request._id, 0))

        self.assertEqual(scheduler._service_virtual_load[0], 0.0)
        self.assertFalse(hasattr(request, "_ravel_service_virtual_owner"))

    def test_bounded_completion_ledger_retains_half_history(self):
        shared = FakeSharedState()
        scheduler = RavelUnifiedBoundedLedgerGlobalScheduler(
            2, shared, self.args
        )
        request = make_request(request_id=92)
        scheduler._register_service_virtual_admission(
            request, owner_hint=0
        )
        scheduler._completion_virtual_requests[request._id] = request
        charge = request._ravel_service_virtual_charge

        asyncio.run(scheduler.on_request_terminal(request._id, 0))

        self.assertAlmostEqual(scheduler._service_virtual_load[0], charge / 2)

    def test_no_ledger_ablation_disables_completion_history(self):
        scheduler = RavelUnifiedNoLedgerGlobalScheduler(
            2, FakeSharedState(), self.args
        )

        self.assertFalse(scheduler.completion_virtual_service_enabled)

    def test_flow_placement_ablation_changes_only_secondary_order(self):
        scheduler = RavelUnifiedFlowPlacementGlobalScheduler(
            2, FakeSharedState(), self.args
        )

        self.assertTrue(scheduler.flow_time_before_virtual_balance)
        self.assertIs(
            scheduler._set_engine_priority.__func__,
            RavelUnifiedGlobalScheduler._set_engine_priority,
        )

    def test_phase_priority_ablation_keeps_unified_secondary_order(self):
        scheduler = RavelUnifiedPhasePriorityGlobalScheduler(
            2, FakeSharedState(), self.args
        )

        self.assertFalse(scheduler.flow_time_before_virtual_balance)
        self.assertIs(
            scheduler._set_engine_priority.__func__,
            RavelUnifiedBalancedGlobalScheduler._set_engine_priority,
        )

    def test_balanced_latency_priority_is_phase_aware(self):
        scheduler = RavelUnifiedBalancedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._non_workflow_observed = True
        request = make_request()
        request._request_type = 0
        request._primary_replica = 0

        scheduler._set_engine_priority(request)

        self.assertTrue(
            is_phase_aware_priority(request._vllm_priority)
        )
        self.assertAlmostEqual(
            decode_tbt_s(request._vllm_priority),
            request._slo_constraint[1],
        )
        self.assertGreater(
            decode_deadline_budget_s(request._vllm_priority),
            0.0,
        )

    def test_balanced_flexible_priority_carries_no_tbt(self):
        scheduler = RavelUnifiedBalancedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._non_workflow_observed = True
        request = make_request()
        request._request_type = 1
        request._primary_replica = 0

        scheduler._set_engine_priority(request)

        self.assertTrue(
            is_phase_aware_priority(request._vllm_priority)
        )
        self.assertIsNone(decode_tbt_s(request._vllm_priority))
        self.assertGreater(
            decode_deadline_budget_s(request._vllm_priority),
            0.0,
        )

    def test_workflow_semantics_enable_completion_pressure_mode(self):
        scheduler = RavelUnifiedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._workflow_completion_observed = True
        scheduler._arrival_pressure_observed = True
        request = make_request(request_id=8)
        scheduler.global_request_queue.push(0, request, 0)

        self.assertTrue(scheduler._update_completion_pressure())

    def test_completion_virtual_service_debt_avoids_hotspot(self):
        scheduler = RavelUnifiedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        scheduler._workflow_completion_observed = True
        scheduler._arrival_pressure_observed = True
        scheduler._service_virtual_load[0] = 10.0
        request = make_request(output_len=4096)
        request._ravel_initial_replica = 0
        request._rebind_count = 0
        scheduler.global_request_queue.push(0, request, 0)

        assignment = scheduler._plan_mobile_assignment([request])

        self.assertEqual(assignment[request._id][0], 1)
        self.assertEqual(request._ravel_service_virtual_owner, 1)
        self.assertTrue(scheduler.completion_virtual_service_enabled)
        self.assertNotEqual(
            request._ravel_service_virtual_charge, request._output_len
        )

    def test_profile_context_does_not_read_realized_output(self):
        shared = FakeSharedState(pending_per_replica=4)
        scheduler = RavelUnifiedGlobalScheduler(2, shared, self.args)
        scheduler._workflow_completion_observed = True
        scheduler._arrival_pressure_observed = True
        request = make_request(output_len=1024)
        scheduler._apply_completion_output_profile(request)

        expected, upper = scheduler._output_token_bounds(request)

        self.assertEqual((expected, upper), (120, 120))
        self.assertEqual(request._ravel_profile_output_upper, 700)
        self.assertNotEqual(expected, request._output_len)

    def test_profile_falls_back_to_stage_when_context_is_unknown(self):
        shared = FakeSharedState(pending_per_replica=4)
        scheduler = RavelUnifiedGlobalScheduler(2, shared, self.args)
        scheduler._workflow_completion_observed = True
        scheduler._arrival_pressure_observed = True
        request = make_request(stage_num=4)
        scheduler._apply_completion_output_profile(request)

        expected, upper = scheduler._output_token_bounds(request)

        self.assertEqual((expected, upper), (100, 100))
        self.assertEqual(request._ravel_profile_output_upper, 900)

    def test_missing_stage_profile_uses_declared_hint(self):
        shared = FakeSharedState(pending_per_replica=4)
        scheduler = RavelUnifiedGlobalScheduler(2, shared, self.args)
        scheduler._workflow_completion_observed = True
        scheduler._arrival_pressure_observed = True
        request = make_request(stage_id=99)
        scheduler._apply_completion_output_profile(request)

        expected, upper = scheduler._output_token_bounds(request)

        self.assertEqual((expected, upper), (256, 256))
        self.assertEqual(request._ravel_profile_output_expected, 0)

    def test_balanced_completion_priority_is_never_deferred(self):
        scheduler = RavelUnifiedBalancedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        request = make_request()
        request._primary_replica = 0

        scheduler._set_engine_priority(request)

        self.assertTrue(is_phase_aware_priority(request._vllm_priority))
        self.assertFalse(
            is_deferred_phase_priority(request._vllm_priority)
        )
        self.assertGreater(
            decode_deadline_budget_s(request._vllm_priority), 0.0
        )

    def test_balanced_priority_charges_profile_upper_decode_demand(self):
        scheduler = RavelUnifiedBalancedGlobalScheduler(
            2, FakeSharedState(), self.args
        )
        request = make_request()
        request._primary_replica = 0
        request._ravel_profile_output_expected = 120
        request._ravel_profile_output_upper = 700
        request._routing_output_tokens_upper_hint_used = 120

        scheduler._set_engine_priority(request)

        expected_service_s = 699 * scheduler.topology.cluster_for_replica(
            0
        ).decode_tpot_for(
            scheduler._prospective_sequence_count(request, 0)
        )
        self.assertAlmostEqual(
            decode_service_s(request._vllm_priority),
            expected_service_s,
            delta=0.001,
        )



if __name__ == "__main__":
    unittest.main()
