'''Execution-policy contracts for automatic H3 MLP quantization.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from h3_optimizations import apply_policy  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    MLP_MEMORY_AUTO,
    MLP_MEMORY_FORCE_QUANT,
)


class MLPAutoPolicyTests(unittest.TestCase):
    def _resolve(self, *, capability, plain_float):
        inventory = SimpleNamespace(mlp_plain_float=plain_float)
        sentinel = object()
        with (
            mock.patch.object(
                apply_policy,
                '_current_capability',
                return_value=capability,
            ),
            mock.patch.object(
                apply_policy,
                '_BASE_MLP_RESOLVER',
                return_value=sentinel,
            ) as resolver,
        ):
            result = apply_policy.resolve_mlp_provider(
                inventory,
                request=MLP_MEMORY_AUTO,
                fp8_available=True,
            )
        return result, resolver

    def test_nvidia_auto_runtime_quantizes_plain_float_mlp(self):
        result, resolver = self._resolve(
            capability=(8, 9),
            plain_float=True,
        )
        self.assertIsNotNone(result)
        resolver.assert_called_once()
        self.assertEqual(
            resolver.call_args.kwargs['request'],
            MLP_MEMORY_FORCE_QUANT,
        )

    def test_native_quantized_mlp_keeps_auto_request(self):
        _result, resolver = self._resolve(
            capability=(8, 9),
            plain_float=False,
        )
        self.assertEqual(
            resolver.call_args.kwargs['request'],
            MLP_MEMORY_AUTO,
        )

    def test_non_nvidia_plain_float_mlp_keeps_auto_request(self):
        _result, resolver = self._resolve(
            capability=None,
            plain_float=True,
        )
        self.assertEqual(
            resolver.call_args.kwargs['request'],
            MLP_MEMORY_AUTO,
        )


if __name__ == '__main__':
    unittest.main()
