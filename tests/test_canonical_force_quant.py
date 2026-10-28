"""CPU contracts for one-time Force quant canonicalization."""

import copy
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

from h3_optimizations import canonical_quant  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    FUSED_QKV_FORCE_QUANT,
    H3OptimizationPlan,
    MLP_MEMORY_FORCE_QUANT,
    MemoryRequest,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def make_block():
    return SimpleNamespace(
        attn=SimpleNamespace(
            qkv_proj=torch.nn.Linear(8, 12, bias=False, dtype=torch.bfloat16),
            out_proj=torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16),
        ),
        mlp=SimpleNamespace(
            fc1=torch.nn.Linear(8, 16, bias=False, dtype=torch.bfloat16),
            fc2=torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16),
        ),
    )


class FakePatcher:
    def __init__(self, blocks):
        self.blocks = blocks
        self.model_options = {}
        self.load_device = torch.device('cuda')
        self.offload_device = torch.device('cpu')
        self.model = SimpleNamespace(model_loaded_weight_memory=123)
        self.size = 456
        self.clone_calls = []
        self.cached_patcher_init = object()

    def clone(self, **kwargs):
        self.clone_calls.append(kwargs)
        child = FakePatcher(copy.deepcopy(self.blocks) if kwargs else self.blocks)
        child.model_options = copy.deepcopy(self.model_options)
        return child


class CanonicalForceQuantTests(unittest.TestCase):
    def force_plan(self):
        return H3OptimizationPlan(
            memory=MemoryRequest(
                fused_qkv=FUSED_QKV_FORCE_QUANT,
                mlp_memory=MLP_MEMORY_FORCE_QUANT,
            )
        )

    def test_force_requests_all_four_h3_linears(self):
        self.assertEqual(
            [
                name
                for name, _getter in canonical_quant._requested_targets(
                    self.force_plan()
                )
            ],
            ['attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2'],
        )

    def test_conversion_owns_a_fresh_model_and_releases_its_float_weights(self):
        source = FakePatcher([make_block(), make_block()])
        source_weights = [
            source.blocks[0].attn.qkv_proj.weight,
            source.blocks[0].attn.out_proj.weight,
            source.blocks[0].mlp.fc1.weight,
            source.blocks[0].mlp.fc2.weight,
        ]

        def fake_convert(module, _device, _offload_device):
            before = module.weight.nbytes
            replacement = torch.nn.Parameter(
                torch.empty(module.weight.shape, dtype=torch.int8),
                requires_grad=False,
            )
            module.register_parameter('weight', replacement)
            return before, replacement.nbytes

        with patch.object(
            canonical_quant,
            'get_h3_blocks',
            side_effect=lambda patcher: patcher.blocks,
        ), patch.object(
            canonical_quant,
            '_conversion_device',
            return_value=torch.device('cpu'),
        ), patch.object(
            canonical_quant,
            '_convert_weight',
            side_effect=fake_convert,
        ), patch.object(
            canonical_quant.comfy.model_management,
            'soft_empty_cache',
        ):
            converted = canonical_quant.clone_with_force_quant(
                source,
                self.force_plan(),
            )

        self.assertEqual(source.clone_calls, [{'force_deepcopy': True}])
        self.assertIsNot(converted.blocks, source.blocks)
        self.assertTrue(
            all(weight.dtype == torch.bfloat16 for weight in source_weights)
        )
        self.assertTrue(
            all(
                module.weight.dtype == torch.int8
                for block in converted.blocks
                for module in (
                    block.attn.qkv_proj,
                    block.attn.out_proj,
                    block.mlp.fc1,
                    block.mlp.fc2,
                )
            )
        )
        report = converted.model_options[canonical_quant.CANONICAL_QUANT_KEY]
        self.assertEqual(report['converted_count'], 8)
        self.assertGreater(report['released_bytes'], 0)
        self.assertEqual(converted.size, 0)
        self.assertEqual(converted.model.model_loaded_weight_memory, 0)

    def test_non_force_plan_uses_the_normal_shared_clone(self):
        source = FakePatcher([make_block()])
        with patch.object(
            canonical_quant,
            'get_h3_blocks',
            side_effect=lambda patcher: patcher.blocks,
        ):
            cloned = canonical_quant.clone_with_force_quant(
                source,
                H3OptimizationPlan(memory=MemoryRequest()),
            )
        self.assertEqual(source.clone_calls, [{}])
        self.assertIs(cloned.blocks, source.blocks)


if __name__ == '__main__':
    unittest.main()
