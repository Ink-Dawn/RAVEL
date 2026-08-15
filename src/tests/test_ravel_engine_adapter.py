import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest

from ravel_engine_adapter.vllm_patch import (
    _has_protected_decode,
    _partition_unexpired_deferred,
    _phase_prefill_key,
    _phase_prefill_order,
    _profile_prefill_bound,
)
from ravel_engine_adapter.protocol import (
    decode_deadline_budget_s,
    decode_deadline_s,
    decode_service_s,
    decode_tbt_s,
    encode_phase_priority,
    encode_priority,
    is_deferred_phase_priority,
    is_phase_aware_priority,
    safe_prefill_chunk_tokens,
)


class PriorityProtocolTests(unittest.TestCase):
    def test_edf_order_is_preserved(self):
        earlier = encode_priority(10.0, 0.32)
        later = encode_priority(10.1, 0.08)
        self.assertLess(earlier, later)
        self.assertLess(earlier, 0)
        self.assertGreaterEqual(earlier, -(2**63))

    def test_tbt_round_trip_and_legacy_detection(self):
        self.assertAlmostEqual(
            decode_tbt_s(encode_priority(10.0, 0.08)),
            0.08,
        )
        self.assertIsNone(decode_tbt_s(encode_priority(10.0, None)))
        self.assertIsNone(decode_tbt_s(0))
        self.assertIsNone(decode_tbt_s(123_456_789))
        self.assertIsNone(decode_tbt_s(-1))

    def test_phase_aware_range_preserves_tbt_without_aliasing_legacy(self):
        phase = encode_phase_priority(10.0, 0.08)
        legacy = encode_priority(10.0, 0.08)

        self.assertNotEqual(phase, legacy)
        self.assertTrue(is_phase_aware_priority(phase))
        self.assertFalse(is_phase_aware_priority(legacy))
        self.assertAlmostEqual(decode_tbt_s(phase), 0.08)
        self.assertIsNone(decode_deadline_s(phase))
        self.assertAlmostEqual(decode_deadline_budget_s(phase), 10.0)

    def test_deferred_phase_flag_preserves_budget_and_lowers_priority(self):
        protected = encode_phase_priority(10.0, None)
        deferred = encode_phase_priority(10.0, None, deferred=True)

        self.assertTrue(is_phase_aware_priority(deferred))
        self.assertTrue(is_deferred_phase_priority(deferred))
        self.assertFalse(is_deferred_phase_priority(protected))
        self.assertAlmostEqual(
            decode_deadline_budget_s(deferred), 10.0
        )
        self.assertGreater(deferred, protected)

    def test_phase_service_demand_round_trip_is_not_tbt(self):
        priority = encode_phase_priority(
            10.0, None, deferred=True, service_s=12.3451
        )

        self.assertIsNone(decode_tbt_s(priority))
        self.assertAlmostEqual(decode_service_s(priority), 12.346)
        self.assertAlmostEqual(decode_deadline_budget_s(priority), 10.0)

    def test_phase_prefill_key_is_edf_when_laxity_is_exhausted(self):
        early = SimpleNamespace(
            priority=encode_phase_priority(10.0, 0.08),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 2000,
        )
        late = SimpleNamespace(
            priority=encode_phase_priority(11.0, 0.08),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 2000,
        )

        early_key = _phase_prefill_key(
            early, slope=0.001, intercept=0.0, now_s=109.0
        )
        late_key = _phase_prefill_key(
            late, slope=0.001, intercept=0.0, now_s=109.0
        )

        self.assertEqual(early_key[0], 0.0)
        self.assertLess(early_key, late_key)

    def test_phase_prefill_key_is_shortest_remaining_when_safe(self):
        short = SimpleNamespace(
            priority=encode_phase_priority(20.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 100,
        )
        long = SimpleNamespace(
            priority=encode_phase_priority(20.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1000,
        )

        short_key = _phase_prefill_key(
            short, slope=0.001, intercept=0.0, now_s=102.0
        )
        long_key = _phase_prefill_key(
            long, slope=0.001, intercept=0.0, now_s=102.0
        )

        self.assertEqual(short_key[0], 2.0)
        self.assertLess(short_key, long_key)

    def test_phase_prefill_order_only_steals_feasible_slack(self):
        urgent = SimpleNamespace(
            priority=encode_phase_priority(4.5, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 4,
        )
        short = SimpleNamespace(
            priority=encode_phase_priority(10.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1,
        )

        ordered = _phase_prefill_order(
            [short, urgent],
            slope=1.0,
            intercept=0.0,
            now_s=100.0,
        )

        self.assertEqual(ordered, [urgent, short])

    def test_phase_prefill_order_uses_spt_when_edf_remains_feasible(self):
        earlier = SimpleNamespace(
            priority=encode_phase_priority(6.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 4,
        )
        short = SimpleNamespace(
            priority=encode_phase_priority(10.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1,
        )

        ordered = _phase_prefill_order(
            [earlier, short],
            slope=1.0,
            intercept=0.0,
            now_s=100.0,
        )

        self.assertEqual(ordered, [short, earlier])

    def test_deferred_prefill_cannot_consume_insufficient_protected_slack(self):
        protected = SimpleNamespace(
            priority=encode_phase_priority(4.5, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 4,
        )
        deferred = SimpleNamespace(
            priority=encode_phase_priority(
                10.0, None, deferred=True
            ),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1,
        )

        ordered = _phase_prefill_order(
            [deferred, protected],
            slope=1.0,
            intercept=0.0,
            now_s=100.0,
        )

        self.assertEqual(ordered, [protected, deferred])

    def test_deferred_decode_occupancy_is_charged_to_protected_slack(self):
        protected = SimpleNamespace(
            priority=encode_phase_priority(6.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 4,
        )
        deferred = SimpleNamespace(
            priority=encode_phase_priority(
                10.0, None, deferred=True, service_s=2.0
            ),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1,
        )

        ordered = _phase_prefill_order(
            [deferred, protected],
            slope=1.0,
            intercept=0.0,
            now_s=100.0,
        )

        self.assertEqual(ordered, [protected, deferred])

    def test_deferred_prefill_may_consume_certified_protected_slack(self):
        protected = SimpleNamespace(
            priority=encode_phase_priority(6.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 4,
        )
        deferred = SimpleNamespace(
            priority=encode_phase_priority(
                10.0, None, deferred=True
            ),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1,
        )

        ordered = _phase_prefill_order(
            [deferred, protected],
            slope=1.0,
            intercept=0.0,
            now_s=100.0,
        )

        self.assertEqual(ordered, [deferred, protected])

    def test_deferred_prefill_is_promoted_when_budget_expires(self):
        protected = SimpleNamespace(
            priority=encode_phase_priority(10.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 4,
        )
        deferred = SimpleNamespace(
            priority=encode_phase_priority(
                1.0, None, deferred=True
            ),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1,
        )

        ordered = _phase_prefill_order(
            [protected, deferred],
            slope=1.0,
            intercept=0.0,
            now_s=102.0,
        )

        self.assertEqual(ordered, [deferred, protected])

    def test_decode_guard_holds_only_unexpired_deferred(self):
        protected = SimpleNamespace(
            priority=encode_phase_priority(10.0, None),
            arrival_time=100.0,
        )
        unexpired = SimpleNamespace(
            priority=encode_phase_priority(
                10.0, None, deferred=True, service_s=2.0
            ),
            arrival_time=100.0,
        )
        expired = SimpleNamespace(
            priority=encode_phase_priority(
                1.0, None, deferred=True, service_s=2.0
            ),
            arrival_time=100.0,
        )

        ready, held = _partition_unexpired_deferred(
            [protected, unexpired, expired], now_s=102.0
        )

        self.assertEqual(ready, [protected, expired])
        self.assertEqual(held, [unexpired])

    def test_decode_guard_treats_legacy_work_as_protected(self):
        legacy = SimpleNamespace(priority=0)
        protected_phase = SimpleNamespace(
            priority=encode_phase_priority(10.0, None)
        )
        deferred = SimpleNamespace(
            priority=encode_phase_priority(
                10.0, None, deferred=True, service_s=2.0
            )
        )

        self.assertTrue(_has_protected_decode([legacy]))
        self.assertTrue(_has_protected_decode([protected_phase]))
        self.assertFalse(_has_protected_decode([deferred]))

    def test_latency_prefill_is_strictly_before_flexible_prefill(self):
        latency = SimpleNamespace(
            priority=encode_phase_priority(10.0, 0.08),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 4,
        )
        flexible = SimpleNamespace(
            priority=encode_phase_priority(2.0, None),
            arrival_time=100.0,
            get_num_uncomputed_tokens=lambda: 1,
        )

        ordered = _phase_prefill_order(
            [flexible, latency],
            slope=1.0,
            intercept=0.0,
            now_s=100.0,
        )

        self.assertEqual(ordered, [latency, flexible])

    def test_profile_derived_cap_is_block_aligned(self):
        cap = safe_prefill_chunk_tokens(
            tbt_s=0.08,
            prefill_seconds_per_token=0.0005928750101447713,
            prefill_intercept_s=0.0,
            block_size=16,
            max_num_batched_tokens=16384,
        )
        self.assertEqual(cap, 128)

    def test_profile_bound_covers_every_calibrated_curve(self):
        profile = {
            "schema_version": 2,
            "prefill_curve": [
                {
                    "seconds_per_token": 0.001,
                    "intercept_s": 0.02,
                },
                {
                    "seconds_per_token": 0.002,
                    "intercept_s": 0.01,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            slope, intercept = _profile_prefill_bound(str(path))
        self.assertEqual((slope, intercept), (0.002, 0.02))


if __name__ == "__main__":
    unittest.main()
