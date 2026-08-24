'''CPU-only contracts for the Kitchen QKV sweep driver.'''

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'run_kitchen_qkv_sweep.py'
)
SPEC = importlib.util.spec_from_file_location('run_kitchen_qkv_sweep', SCRIPT)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)

ORDERING_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'run_kitchen_qkv_ordering.py'
)
ORDERING_SPEC = importlib.util.spec_from_file_location(
    'run_kitchen_qkv_ordering', ORDERING_SCRIPT
)
ORDERING = importlib.util.module_from_spec(ORDERING_SPEC)
ORDERING_SPEC.loader.exec_module(ORDERING)


class KitchenQKVSweepTests(unittest.TestCase):
    def test_full_arm_is_one_aligned_chunk(self):
        self.assertEqual(
            SWEEP.parse_chunks('1024,full,full', sequence=54006),
            (1024, 54016),
        )

    def test_rejects_unaligned_rows(self):
        with self.assertRaises(ValueError):
            SWEEP.parse_chunks('4097', sequence=54006)

    def test_repeat_order_is_reversed(self):
        chunks = (1024, 2048, 4096)
        self.assertEqual(SWEEP.arm_order(chunks, 0), [1024, 2048, 4096])
        self.assertEqual(SWEEP.arm_order(chunks, 1), [4096, 2048, 1024])

    def test_ordering_fingerprint_delta(self):
        self.assertEqual(
            ORDERING.fingerprint_delta(
                [[1.0, 2.0], [3.0]],
                [[1.0, 2.25], [3.0]],
            ),
            0.25,
        )


if __name__ == '__main__':
    unittest.main()
