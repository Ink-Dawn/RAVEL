import statistics
import unittest

from scripts.calibrate_service_profile import validate_decode_curve_quality


def decode_point(active_sequences, values):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return {
        "active_sequences": active_sequences,
        "tpot_s_mean": mean,
        "tpot_s_median": statistics.median(values),
        "tpot_s_stdev": stdev,
        "tpot_s_cv": stdev / mean,
    }


class ServiceCalibrationTests(unittest.TestCase):
    def test_decode_quality_accepts_stable_near_monotone_curve(self):
        curve = [
            decode_point(1, [0.0100, 0.0101, 0.0099, 0.0100, 0.0101]),
            decode_point(9, [0.0110, 0.0112, 0.0109, 0.0111, 0.0110]),
            decode_point(17, [0.0105, 0.0107, 0.0106, 0.0105, 0.0106]),
        ]

        quality = validate_decode_curve_quality(curve)

        self.assertTrue(quality["passed"])

    def test_decode_quality_rejects_phase_unstable_samples(self):
        curve = [
            decode_point(1, [0.0100, 0.0101, 0.0099, 0.0100, 0.0101]),
            decode_point(9, [0.0100, 0.0200, 0.0300, 0.0400, 0.0500]),
        ]

        with self.assertRaisesRegex(RuntimeError, "coefficient of variation"):
            validate_decode_curve_quality(curve)

    def test_decode_quality_records_stable_concurrency_drop(self):
        curve = [
            decode_point(1, [0.0200, 0.0201, 0.0199, 0.0200, 0.0201]),
            decode_point(9, [0.0120, 0.0121, 0.0119, 0.0120, 0.0121]),
        ]

        quality = validate_decode_curve_quality(curve)

        self.assertTrue(quality["passed"])
        self.assertGreater(quality["max_relative_drop"], 0.10)
        self.assertFalse(quality["monotonicity_required"])


if __name__ == "__main__":
    unittest.main()
