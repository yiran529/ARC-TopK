"""CPU checks for strict timing aggregation and experiment acceptance."""

import unittest
import tempfile
from pathlib import Path

from c4 import table_v_timing as timing
from c4.scripts.run_strict_hook_abcd import accepted_result, comparisons
from c4.scripts import run_strict_hook_abcd as controller


class StrictTimingTests(unittest.TestCase):
    def test_partial_result_json_is_retained_as_failure(self):
        read_result = getattr(controller, "read_result", None)
        self.assertIsNotNone(read_result, "Partial-result handling is missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_results.json"
            path.write_text('{"status":')
            result, error = read_result(path)
            self.assertIsNone(result)
            self.assertIn("JSONDecodeError", error)

    def test_multibucket_result_cannot_enter_the_single_bucket_comparison(self):
        cell = dict(compressor="none", measured=100, warmup=50)
        result = dict(status="completed", optimizer="muon", compressor="none",
                      strict_blocking_communication=True, measured_steps=100,
                      first_measured_update=51, last_measured_update=150,
                      observed_bucket_counts=[1], mean_blocking_hook_seconds=0.1,
                      mean_iteration_seconds=0.2)
        self.assertTrue(accepted_result(result, cell, 0))
        result["observed_bucket_counts"] = [2]
        self.assertFalse(accepted_result(result, cell, 0))
        result["observed_bucket_counts"] = [1]
        self.assertFalse(accepted_result(result, cell, 1))

    def test_failed_or_different_repeat_dense_is_not_used_as_a_pair(self):
        rows = [dict(repeat=1, group="A", model="60m", arm="dense", status="failed"),
                dict(repeat=2, group="A", model="60m", arm="dense", status="completed",
                     run_id="dense-2", mean_iteration_seconds=4, mean_blocking_hook_seconds=2),
                dict(repeat=1, group="A", model="60m", arm="arctopk", status="completed",
                     run_id="arc-1", mean_iteration_seconds=2, mean_blocking_hook_seconds=1)]
        self.assertEqual(comparisons(rows), [])
        rows[-1]["repeat"] = 2
        pair = comparisons(rows)[0]
        self.assertEqual(pair["dense_run_id"], "dense-2")
        self.assertEqual(pair["iter_speedup"], 2)
        self.assertEqual(pair["hook_speedup"], 2)

    def test_hook_slowest_rank_is_independent_of_iteration_slowest_rank(self):
        summarize = getattr(timing, "summarize_blocking_hook_timing", None)
        self.assertIsNotNone(summarize, "Strict hook aggregation is missing")
        result = summarize([[0.2, 0.4], [0.5, 0.3]], [[1, 1], [1, 1]],
                           [[[128], [128]], [[128], [128]]], 2)
        self.assertAlmostEqual(result["mean_blocking_hook_seconds"], 0.4)
        self.assertEqual(result["blocking_hook_slowest_rank"], 1)
        self.assertEqual(result["observed_bucket_counts"], [1])

    def test_incomplete_or_invalid_hook_window_is_rejected(self):
        summarize = getattr(timing, "summarize_blocking_hook_timing", None)
        self.assertIsNotNone(summarize, "Strict hook aggregation is missing")
        for times, counts, sizes in (
            ([[0.1]], [[1]], [[[128]]]),
            ([[0.1, float("nan")]], [[1, 1]], [[[128], [128]]]),
            ([[0.1, 0.2]], [[1, 2]], [[[128], [128]]]),
        ):
            with self.subTest(times=times), self.assertRaises(ValueError):
                summarize(times, counts, sizes, 2)


if __name__ == "__main__":
    unittest.main()
