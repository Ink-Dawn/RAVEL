import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dualmap.entities.request import Request
from dualmap.scheduler.global_scheduler.ravel_base_scheduler import (
    RavelBaseGlobalScheduler,
)


class FakeReplica:
    def __init__(self, replica_id):
        self.replica_id = replica_id

    def get_num_recompute_token_ids(self, input_ids):
        return len(input_ids)

    def get_num_running_req(self):
        return 0

    def get_num_pending_req(self):
        return 0

    async def get_load_states(self):
        return 10000, 0.0, 0.0, 0, 0.0


class FakeSharedState:
    def __init__(self):
        self.replica_budgets = {0: FakeReplica(0), 1: FakeReplica(1)}
        self.pending_tokens = {0: 9000, 1: 0}
        self.posted = []
        self.callback = None

    def set_scheduler_callback(self, callback):
        self.callback = callback

    def get_num_actual_pending_tokens_replica(self, replica_id):
        return self.pending_tokens[replica_id]

    async def add_posting_request_tasks(self, replica_id, request):
        self.posted.append((replica_id, request._id))
        return True


class ClusterSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_dispatches_to_selected_cluster_replica(self):
        topology = {
            "default_client_region": "a",
            "clusters": [
                {
                    "id": "a",
                    "replica_ids": [0],
                    "rtt_ms_by_client_region": {"a": 0.0},
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.01,
                },
                {
                    "id": "b",
                    "replica_ids": [1],
                    "rtt_ms_by_client_region": {"a": 100.0},
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.01,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            topology_path = Path(directory) / "topology.json"
            topology_path.write_text(json.dumps(topology), encoding="utf-8")
            args = SimpleNamespace(
                cluster_topology=str(topology_path),
                cluster_overload_fraction=0.5,
                dh_replica_pending_req_threshold=1,
                dh_rebalance_thredhold=100000,
                dh_rebalance_waiting_latency_thredhold=10.0,
                cluster_rebalance_hysteresis_s=0.02,
                cluster_max_rebalances_per_event=8,
            )
            shared = FakeSharedState()
            scheduler = RavelBaseGlobalScheduler(2, shared, args)
            request = Request(
                1, "jitserve", 0, "s", "prefix", 0, "p", list(range(100)),
                100, 100, 900, False, 1, 0, 1, 256, True, 0.0, 0.0, 0,
                request_type=0,
                slo_constraint=(0.8, 0.08, 8.0),
                client_region="a",
                routing_output_tokens_hint=256,
            )
            result = await scheduler.schedule(request)

        self.assertEqual(result, -1)
        self.assertEqual(request._primary_cluster, "b")
        self.assertEqual(shared.posted, [(1, 1)])


if __name__ == "__main__":
    unittest.main()
