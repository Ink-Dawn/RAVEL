import json
import tempfile
import unittest
from pathlib import Path

from dualmap.request_generator.jitserve_trace import (
    assign_request_types,
    load_jitserve_records,
    stable_prefix_block_hashes,
)


class JITServeTraceTests(unittest.TestCase):
    def test_mixed_ratio_is_deterministic(self):
        records = [{"prompt": str(index)} for index in range(100)]
        first = assign_request_types(records, (3, 5, 2), 42)
        second = assign_request_types(records, (3, 5, 2), 42)
        self.assertEqual(
            [record["request_type"] for record in first],
            [record["request_type"] for record in second],
        )
        self.assertEqual(len(first), 100)

    def test_released_jitserve_ratio_rule_has_frozen_request_counts(self):
        records = [{"prompt": str(index)} for index in range(500)]
        assigned = assign_request_types(records, (3, 5, 2), 42)
        counts = {
            request_type: sum(
                record["request_type"] == request_type for record in assigned
            )
            for request_type in (0, 1, 2)
        }
        self.assertEqual(counts, {0: 181, 1: 301, 2: 18})

    def test_deepresearch_trace_flattens_stages(self):
        raw = {
            "stages": [
                [{"prompt": "a", "output_tokens": 7, "start_time": 1, "timestamp": 2}],
                [{"prompt": "b", "output_tokens": 9, "start_time": 2, "timestamp": 4}],
            ],
            "stage_num": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            records = load_jitserve_records(path)
        self.assertEqual([record["output_len"] for record in records], [7, 9])
        self.assertTrue(all(record["request_type"] == 2 for record in records))


    def test_locked_collective_types_survive_ratio_assignment(self):
        records = [
            {"prompt": "collective", "request_type": 2, "request_type_locked": True},
            {"prompt": "ordinary"},
        ]
        assigned = assign_request_types(records, (3, 5, 2), 42)
        self.assertEqual(assigned[0]["request_type"], 2)

    def test_prefix_hashes_are_content_stable(self):
        self.assertEqual(
            stable_prefix_block_hashes(list(range(32)), 16),
            stable_prefix_block_hashes(list(range(32)), 16),
        )
        self.assertNotEqual(
            stable_prefix_block_hashes(list(range(32)), 16)[0],
            stable_prefix_block_hashes([99] + list(range(1, 32)), 16)[0],
        )


if __name__ == "__main__":
    unittest.main()
