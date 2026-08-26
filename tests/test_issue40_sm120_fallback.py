'''CPU contracts for issue #40 SM120 fallback behavior.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse import triton_sparse  # noqa: E402
from h3_optimizations.attention.sparse.triton_kitchen_sm120 import (  # noqa: E402
    _route_groups,
)
from h3_optimizations.native import selftest  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class Issue40SM120FallbackTests(unittest.TestCase):
    def test_sm120_selects_static_loop_triton_backend(self):
        sentinel = object()
        with mock.patch.object(
            triton_sparse.torch.cuda,
            'get_device_capability',
            return_value=(12, 0),
        ), mock.patch.object(
            triton_sparse,
            'SM120TritonKitchenBackend',
            return_value=sentinel,
        ) as backend:
            actual = triton_sparse.TritonSparseBackend('config', projector='p')
        self.assertIs(actual, sentinel)
        backend.assert_called_once_with('config', projector='p')

    def test_non_sm120_keeps_normal_kitchen_parity_backend(self):
        sentinel = object()
        with mock.patch.object(
            triton_sparse.torch.cuda,
            'get_device_capability',
            return_value=(8, 9),
        ), mock.patch.object(
            triton_sparse,
            'TritonKitchenBackend',
            return_value=sentinel,
        ) as backend:
            actual = triton_sparse.TritonSparseBackend('config', projector='p')
        self.assertIs(actual, sentinel)
        backend.assert_called_once_with('config', projector='p')

    def test_triton_preflight_keeps_legacy_capability_contract(self):
        sentinel = object()
        with mock.patch.object(
            triton_sparse._legacy,
            'preflight_triton_sparse',
            return_value=sentinel,
        ):
            actual = triton_sparse.preflight_triton_sparse(
                cuda_available=lambda: True,
                capability_getter=lambda: (12, 0),
            )
        self.assertIs(actual, sentinel)

    def test_sm120_route_groups_restore_fixed_dense_and_sparse_counts(self):
        metadata = {
            'dense_q_tiles': 3,
            'sparse_q_tiles': 5,
            'pure_video_kv_tiles': 7,
            'retained_video_kv_tiles': 2,
        }
        self.assertEqual(
            _route_groups(metadata, q_tiles=8, kv_blocks=11),
            [(0, 3, 11), (3, 5, 6)],
        )

    def test_healthy_geometries_ignore_global_and_bit_identity_failures(self):
        detail = {
            'dense_int8_passed': True,
            'full_route_passed': {
                '128x128': False,
                '128x64': True,
                '64x64': True,
            },
            'full_route_bit_identical': {
                '128x128': False,
                '128x64': False,
                '64x64': False,
            },
        }
        with mock.patch.object(selftest, '_load_result', return_value=(False, detail)):
            self.assertTrue(selftest.sparse_geometry_check(64, 64, 'cuda'))
            self.assertTrue(selftest.sparse_geometry_check(128, 64, 'cuda'))
            self.assertFalse(selftest.sparse_geometry_check(128, 128, 'cuda'))

    def test_geometry_is_rejected_when_common_dense_carrier_gate_fails(self):
        detail = {
            'dense_int8_passed': False,
            'full_route_passed': {'64x64': True},
        }
        with mock.patch.object(selftest, '_load_result', return_value=(False, detail)):
            self.assertFalse(selftest.sparse_geometry_check(64, 64, 'cuda'))

    def test_lse_probe_is_not_part_of_normal_kitchen_gate(self):
        detail = {
            'dense_int8_passed': True,
            'full_route_passed': {'64x64': True},
        }
        with mock.patch.object(selftest, '_load_result', return_value=(True, detail)):
            self.assertTrue(selftest.sparse_geometry_check(64, 64, 'cuda'))

        selftest._lse_results.clear()
        with mock.patch.object(
            selftest.torch.cuda, 'is_available', return_value=True
        ), mock.patch.object(
            selftest.loader, 'is_available', return_value=True
        ), mock.patch.object(
            selftest, '_cache_key', return_value='device'
        ), mock.patch.object(
            selftest, 'run_lse', return_value=(False, {'passed': False})
        ) as run_lse:
            self.assertFalse(selftest.sparse_lse_check('cuda'))
        run_lse.assert_called_once_with('cuda')


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
