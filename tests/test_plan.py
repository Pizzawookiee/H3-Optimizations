'''Pure tests for the order-independent two-node optimization plan.'''

import math
from pathlib import Path
import sys
import unittest

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    MemoryRequest,
    SparseRequest,
)


class PlanTests(unittest.TestCase):
    def test_memory_request_defaults_to_four_thousand_rows(self):
        self.assertEqual(MemoryRequest().chunk_rows, 4096)

    def test_sparse_request_defaults_to_thirty_percent_video_budget(self):
        self.assertEqual(SparseRequest().video_budget, 0.3)

    def test_node_order_does_not_change_the_plan(self):
        memory = MemoryRequest()
        sparse = SparseRequest(video_budget=0.5)
        first = H3OptimizationPlan().with_memory(memory).with_sparse(sparse)
        second = H3OptimizationPlan().with_sparse(sparse).with_memory(memory)
        self.assertEqual(first, second)
        self.assertEqual(first.signature, second.signature)

    def test_identical_requests_are_idempotent(self):
        memory = MemoryRequest()
        sparse = SparseRequest()
        plan = H3OptimizationPlan().with_memory(memory).with_sparse(sparse)
        self.assertEqual(plan.with_memory(memory), plan)
        self.assertEqual(plan.with_sparse(sparse), plan)

    def test_conflicting_duplicate_requests_fail(self):
        plan = H3OptimizationPlan().with_memory(MemoryRequest())
        with self.assertRaisesRegex(ValueError, 'different H3 Memory'):
            plan.with_memory(MemoryRequest(fused_qkv='off'))
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        with self.assertRaisesRegex(ValueError, 'different H3 Sparse'):
            plan.with_sparse(SparseRequest(video_budget=0.4))

    def test_validation_boundaries(self):
        MemoryRequest(chunk_rows=256)
        MemoryRequest(chunk_rows=65_536)
        for chunk_rows in (255, 257, 65_792):
            with self.assertRaises(ValueError):
                MemoryRequest(chunk_rows=chunk_rows)
        for budget in (0.0, 1.01, math.inf, math.nan):
            with self.assertRaises(ValueError):
                SparseRequest(video_budget=budget)


if __name__ == '__main__':
    unittest.main()
