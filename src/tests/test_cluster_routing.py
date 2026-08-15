import unittest
import json
import tempfile
from pathlib import Path

from dualmap.cluster.routing import (
    RavelInitialPlacementPolicy,
    ReplicaRoutingState,
    RoutingRequestView,
)
from dualmap.cluster.slo import (
    RequestSLO,
    build_request_slo,
    collective_task_deadline_s,
)
from dualmap.cluster.topology import ClusterTopology
from dualmap.entities.request import Request


def topology():
    return ClusterTopology.from_dict(
        {
            "default_client_region": "a",
            "clusters": [
                {
                    "id": "a",
                    "replica_ids": [0],
                    "rtt_ms_by_client_region": {"a": 0.0, "default": 100.0},
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.01,
                },
                {
                    "id": "b",
                    "replica_ids": [1],
                    "rtt_ms_by_client_region": {"a": 100.0, "default": 100.0},
                    "prefill_tpot_s": 0.0001,
                    "decode_tpot_s": 0.01,
                },
            ],
        }
    )


class ClusterRoutingTests(unittest.TestCase):

    def test_topology_loads_and_interpolates_service_profile(self):
        profile = {
            "schema_version": 2,
            "engine_fingerprint": "test-engine",
            "prefill_tpot_s": 0.001,
            "decode_base_tpot_s": 0.01,
            "prefill_curve": [
                {
                    "active_sequences": 0,
                    "tpot_s_median": 0.001,
                    "intercept_s": 0.01,
                },
                {
                    "active_sequences": 10,
                    "tpot_s_median": 0.003,
                    "intercept_s": 0.03,
                },
            ],
            "decode_curve": [
                {"active_sequences": 0, "tpot_s_median": 0.01},
                {"active_sequences": 10, "tpot_s_median": 0.03},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            topology_path = root / "topology.json"
            topology_path.write_text(
                json.dumps(
                    {
                        "default_client_region": "a",
                        "clusters": [
                            {
                                "id": "a",
                                "replica_ids": [0],
                                "rtt_ms_by_client_region": {"a": 0},
                                "service_profile": "profile.json",
                            },
                            {
                                "id": "b",
                                "replica_ids": [1],
                                "rtt_ms_by_client_region": {"a": 100},
                                "service_profile": "profile.json",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded = ClusterTopology.from_json(topology_path)
        self.assertAlmostEqual(loaded.cluster("a").prefill_tpot_for(5), 0.002)
        self.assertAlmostEqual(
            loaded.cluster("a").prefill_intercept_for(5), 0.02
        )
        self.assertAlmostEqual(loaded.cluster("a").decode_tpot_for(5), 0.02)

    def test_formal_topology_rejects_inline_service_rates(self):
        loaded = topology()
        with self.assertRaisesRegex(ValueError, "external schema-v2 service profile"):
            loaded.validate_service_profiles()
        loaded.validate_service_profiles(allow_inline=True)

    def test_external_service_profile_passes_formal_validation(self):
        profile = {
            "schema_version": 2,
            "engine_fingerprint": "test-engine",
            "prefill_tpot_s": 0.001,
            "decode_base_tpot_s": 0.01,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
            topology_path = root / "topology.json"
            topology_path.write_text(
                json.dumps(
                    {
                        "clusters": [
                            {
                                "id": "a",
                                "replica_ids": [0],
                                "rtt_ms_by_client_region": {"default": 0},
                                "service_profile": "profile.json",
                            },
                            {
                                "id": "b",
                                "replica_ids": [1],
                                "rtt_ms_by_client_region": {"default": 100},
                                "service_profile": "profile.json",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loaded = ClusterTopology.from_json(topology_path)
            loaded.validate_service_profiles()

    def test_jitserve_slo_multiplier(self):
        self.assertEqual(build_request_slo("paper_e2e", 0, 0).as_tuple(), (0.8, 0.08, 8.0))
        self.assertEqual(build_request_slo("paper_e2e", 3, 1).as_tuple(), (3.2, 0.32, 32.0))
        branch_slo = build_request_slo(
            "paper_collective", 2, 2, stage_num=6
        )
        self.assertEqual(branch_slo.ttlt_s, 24.0)
        self.assertEqual(
            collective_task_deadline_s(
                branch_slo.ttlt_s,
                collection_id=2,
                stage_num=6,
            ),
            144.0,
        )

    def test_overloaded_local_cluster_routes_remote(self):
        policy = RavelInitialPlacementPolicy(topology())
        request = RoutingRequestView(1, "shared", 100, 0, RequestSLO(0.8, 0.08, 8), "a", 256)
        states = {
            0: ReplicaRoutingState(0, 9000, 0, 100, 0),
            1: ReplicaRoutingState(1, 0, 0, 100, 0),
        }
        decision = policy.choose(request, states)
        self.assertEqual(decision.cluster_id, "b")
        self.assertTrue(decision.feasible)

    def test_router_view_does_not_expose_true_output_length(self):
        request = Request(
            1, "jitserve", 1, "s", "p", 0, "hello", [1, 2], 2, 2,
            997, False, 1, 0, 1, 256, True, 0.0, 0.0, 0,
            request_type=1,
            routing_output_tokens_hint=256,
        )
        view = request.routing_view()
        self.assertEqual(view.output_tokens_hint, 256)
        self.assertFalse(hasattr(view, "output_len"))
        self.assertNotIn(997, view.__dict__.values())


if __name__ == "__main__":
    unittest.main()
