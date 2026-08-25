'''Contracts for the attention-side memory experiments.

Three independent changes, each opt-in and each with its own way of being
wrong:

* sequence-major output storage -- must keep the BHND logical contract, and
  must make the caller's flatten a view rather than a copy;
* releasing the Kitchen carriers before out_proj -- must actually drop the
  references, and must refuse rather than silently skip when the prepared
  object cannot release;
* strided Q/K chunk input -- must be taken only when the kernel's own
  alignment predicate holds, and must be refused loudly when the resolved
  producer has no such support.
'''

import os
from pathlib import Path
import sys
import unittest

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations import attention_forward  # noqa: E402
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
)
from h3_optimizations.attention.sparse.kitchen_sparse import (  # noqa: E402
    PreparedSparseKitchen,
    SparseKitchenBackend,
)
from h3_optimizations.attention_forward import (  # noqa: E402
    flatten_attention_output,
)
from h3_optimizations.kitchen_qkv import (  # noqa: E402
    ChunkedKitchenQKVProjector,
    FusedQKVError,
    _qk_chunk_kwargs,
)
from h3_optimizations.native import int8_attention  # noqa: E402
from h3_optimizations.native import producer as native_producer  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def carrier(batch=1, heads=4, sequence=512, head_dim=128, cta_k=128):
    return int8_attention.PrequantizedInt8Attention(
        q=torch.empty(batch, heads, sequence, head_dim, dtype=torch.int8),
        k=torch.empty(batch, heads, sequence, head_dim, dtype=torch.int8),
        v=torch.empty(batch * heads * head_dim, sequence, dtype=torch.int8),
        q_scale=torch.empty(1),
        k_scale=torch.empty(1),
        v_scale=torch.empty(1),
        original_head_dim=head_dim,
        input_dtype=torch.bfloat16,
        attention_scale=head_dim ** -0.5,
        cta_k=cta_k,
    )


class OutputLayoutTests(unittest.TestCase):
    def test_both_layouts_return_the_same_logical_shape(self):
        quantized = carrier()
        hnd, _, _, _ = int8_attention._attention_geometry(
            quantized, int8_attention.OUTPUT_HND
        )
        nhd, _, _, _ = int8_attention._attention_geometry(
            quantized, int8_attention.OUTPUT_NHD
        )
        self.assertEqual(tuple(hnd.shape), (1, 4, 512, 128))
        self.assertEqual(tuple(nhd.shape), tuple(hnd.shape))

    def test_sequence_major_storage_makes_the_flatten_a_view(self):
        '''The entire point: the caller's reshape must stop copying.'''
        quantized = carrier()
        hnd, _, _, _ = int8_attention._attention_geometry(
            quantized, int8_attention.OUTPUT_HND
        )
        nhd, _, _, _ = int8_attention._attention_geometry(
            quantized, int8_attention.OUTPUT_NHD
        )
        flat_hnd = hnd.transpose(1, 2).reshape(1, 512, 4 * 128)
        flat_nhd = nhd.transpose(1, 2).reshape(1, 512, 4 * 128)
        self.assertNotEqual(flat_hnd.data_ptr(), hnd.data_ptr())
        self.assertEqual(flat_nhd.data_ptr(), nhd.data_ptr())

    def test_the_strides_handed_to_the_kernel_describe_the_storage(self):
        quantized = carrier()
        nhd, _, _, strides = int8_attention._attention_geometry(
            quantized, int8_attention.OUTPUT_NHD
        )
        stride_bz_o, stride_seq_o, stride_h_o = strides[-3:]
        self.assertEqual(stride_bz_o, 512 * 4 * 128)
        self.assertEqual(stride_seq_o, 4 * 128)
        self.assertEqual(stride_h_o, 128)
        # The kernel indexes O[b*bz + h*h + q*seq + d]; that has to land on the
        # same element the logical view names.
        for head in (0, 3):
            for row in (0, 511):
                offset = head * stride_h_o + row * stride_seq_o
                self.assertEqual(
                    offset,
                    (
                        nhd.stride(1) * head + nhd.stride(2) * row
                    ),
                )

    def test_head_major_strides_are_unchanged(self):
        quantized = carrier()
        _, _, _, strides = int8_attention._attention_geometry(
            quantized, int8_attention.OUTPUT_HND
        )
        self.assertEqual(strides[-3:], (4 * 512 * 128, 128, 512 * 128))

    def test_the_head_dimension_slice_stays_a_view_in_both_layouts(self):
        quantized = carrier(head_dim=128)
        for layout in int8_attention.OUTPUT_LAYOUTS:
            output, _, _, _ = int8_attention._attention_geometry(
                quantized, layout
            )
            self.assertEqual(output[..., :128].data_ptr(), output.data_ptr())

    def test_an_unknown_layout_is_refused(self):
        with self.assertRaises(ValueError):
            int8_attention._attention_geometry(carrier(), 'ndh')


class FlattenTests(unittest.TestCase):
    def test_the_flatten_is_one_expression_for_both_layouts(self):
        module = type('M', (), {'heads': 4, 'head_dim': 128})()
        quantized = carrier()
        for layout in int8_attention.OUTPUT_LAYOUTS:
            output, _, _, _ = int8_attention._attention_geometry(
                quantized, layout
            )
            flat = flatten_attention_output(module, output, 'test')
            self.assertEqual(tuple(flat.shape), (1, 512, 512))

    def test_a_rank_three_output_is_refused(self):
        from h3_optimizations.attention_forward import flatten_attention_output

        module = type('M', (), {'heads': 4, 'head_dim': 128})()
        with self.assertRaises(RuntimeError):
            flatten_attention_output(module, torch.empty(1, 512, 512), 'test')


class CarrierReleaseTests(unittest.TestCase):
    def test_release_drops_the_carriers_for_every_holder(self):
        quantized = carrier()
        prepared = PreparedSparseKitchen(
            quantized=quantized,
            route=object(),
            original_head_dim=128,
            layer_index=0,
            metadata={},
        )
        also_held = prepared
        prepared.release()
        self.assertIsNone(prepared.quantized)
        self.assertIsNone(prepared.route)
        self.assertIsNone(also_held.quantized)

    def test_the_backend_reports_whether_it_releases(self):
        kitchen = type('K', (), {'__version__': 'test'})()
        for release in (False, True):
            backend = SparseKitchenBackend(
                HybridSparseConfig(video_budget=0.3),
                kitchen=kitchen,
                allow_cpu_for_tests=True,
                release_carrier_before_out_proj=release,
            )
            self.assertEqual(
                backend.as_status()['release_carrier_before_out_proj'], release
            )

    def test_the_installation_signature_separates_the_variants(self):
        from h3_optimizations.attention.sparse.kitchen_sparse import (
            SparseKitchenBackend,
        )
        from h3_optimizations.attention.sparse.config import HybridSparseConfig

        kitchen = type('K', (), {'__version__': 'test'})()

        def build(**options):
            return SparseKitchenBackend(
                HybridSparseConfig(video_budget=0.3),
                kitchen=kitchen,
                allow_cpu_for_tests=True,
                **options,
            ).installation_signature

        baseline = build()
        self.assertNotEqual(baseline, build(output_layout='nhd'))
        self.assertNotEqual(baseline, build(release_carrier_before_out_proj=True))
        self.assertNotEqual(baseline, build(score_chunk_tiles=64))

    def test_releasing_without_support_raises_rather_than_skipping(self):
        module = type('M', (), {'heads': 4, 'head_dim': 128})()
        module.out_proj = lambda value: value

        class Backend:
            name = 'test'
            release_carrier_before_out_proj = True

            def execute(self, prepared):
                del prepared
                return torch.empty(1, 4, 8, 128)

        with self.assertRaises(RuntimeError):
            attention_forward._finish_projected(module, Backend(), object())


class StridedChunkInputTests(unittest.TestCase):
    def _hnd_view(self, rows=512, heads=4, head_dim=128, dtype=torch.bfloat16):
        projected = torch.empty(rows, 3 * heads * head_dim, dtype=dtype)
        q, _k, _v = projected.split(heads * head_dim, dim=-1)
        return q.view(rows, heads, head_dim).transpose(0, 1).unsqueeze(0)

    def test_the_producer_view_is_accepted_without_a_copy(self):
        view = self._hnd_view()
        self.assertFalse(view.is_contiguous())
        self.assertTrue(native_producer._kernel_accepts_strided(view, 128))
        self.assertIs(
            native_producer._prepare_chunk_input(view, 128, True), view
        )

    def test_the_copy_still_happens_when_it_is_not_requested(self):
        view = self._hnd_view()
        prepared = native_producer._prepare_chunk_input(view, 128, False)
        self.assertIsNot(prepared, view)
        self.assertTrue(prepared.is_contiguous())

    def test_a_wrong_head_width_is_never_taken_strided(self):
        view = self._hnd_view(head_dim=64)
        self.assertFalse(native_producer._kernel_accepts_strided(view, 128))

    def test_a_non_contiguous_head_dimension_is_refused(self):
        base = torch.empty(1, 4, 512, 256, dtype=torch.bfloat16)
        view = base[..., ::2]
        self.assertFalse(native_producer._kernel_accepts_strided(view, 128))

    def test_a_stride_breaking_vector_alignment_is_refused(self):
        '''Mirrors the launcher's own 4-element predicate.'''
        base = torch.empty(1, 4, 512, 130, dtype=torch.bfloat16)
        view = base[..., :128]
        self.assertEqual(view.stride(2), 130)
        self.assertFalse(native_producer._kernel_accepts_strided(view, 128))

    def test_a_single_element_extent_is_exempt(self):
        view = self._hnd_view()
        self.assertEqual(view.shape[0], 1)
        self.assertTrue(native_producer._kernel_accepts_strided(view, 128))

    def test_the_producer_advertises_the_capability(self):
        self.assertTrue(native_producer.SUPPORTS_STRIDED_QK_CHUNK)

    def test_the_package_resolve_kitchen_returns_advertises_it_too(self):
        '''resolve_kitchen hands back the package, not the producer module.'''
        from h3_optimizations import native

        self.assertTrue(getattr(native, 'SUPPORTS_STRIDED_QK_CHUNK', False))
        self.assertIn('SUPPORTS_STRIDED_QK_CHUNK', native.__all__)

    def test_requesting_strided_input_from_a_producer_without_it_aborts(self):
        legacy = type('Kitchen', (), {'__name__': 'comfy_kitchen'})()
        self.assertEqual(_qk_chunk_kwargs(legacy, False), {})
        with self.assertRaises(FusedQKVError):
            _qk_chunk_kwargs(legacy, True)
        self.assertEqual(
            _qk_chunk_kwargs(native_producer, True),
            {'allow_strided_input': True},
        )


class ProjectorOptionTests(unittest.TestCase):
    def test_the_projector_signature_separates_every_variant(self):
        baseline = ChunkedKitchenQKVProjector(
            routing_summaries=True
        ).installation_signature
        variants = (
            {'v_mode': 'two_pass'},
            {'strided_qk_input': True},
            {'stream_output': True},
            {'chunk_rows': 1024},
        )
        for options in variants:
            other = ChunkedKitchenQKVProjector(
                routing_summaries=True, **options
            ).installation_signature
            self.assertNotEqual(baseline, other, str(options))

    def test_an_unknown_v_mode_is_refused(self):
        with self.assertRaises(ValueError):
            ChunkedKitchenQKVProjector(v_mode='streamed')


class ProductionDefaultTests(unittest.TestCase):
    """The measured-free variant has to be what production actually builds."""

    def test_apply_enables_the_measured_defaults_on_the_sparse_route(self):
        source = (
            PACK / 'h3_optimizations' / 'apply.py'
        ).read_text(encoding='utf-8')
        sparse = source.split('_resolve_kitchen_sparse', 1)[1]
        self.assertIn('output_layout=OUTPUT_NHD', sparse)
        self.assertIn('release_carrier_before_out_proj=True', sparse)
        self.assertIn('strided_qk_input=True', sparse)
        self.assertIn('stream_output=True', sparse)

    def test_the_dense_route_takes_strided_qk_but_not_the_layout(self):
        """The layout was measured on the sparse route only."""
        source = (
            PACK / 'h3_optimizations' / 'apply.py'
        ).read_text(encoding='utf-8')
        dense = source.split('def _resolve_dense', 1)[1].split('def ', 1)[0]
        self.assertIn('strided_qk_input=True', dense)
        self.assertNotIn('output_layout=', dense)

    def test_the_dense_backend_never_asks_for_a_release_it_cannot_do(self):
        """PreparedChunkedKitchenQKV has no release(); asking would raise."""
        from h3_optimizations.kitchen_qkv import (
            ChunkedKitchenAttentionBackend,
            PreparedChunkedKitchenQKV,
        )

        backend = ChunkedKitchenAttentionBackend()
        self.assertFalse(
            getattr(backend, 'release_carrier_before_out_proj', False)
        )
        self.assertFalse(hasattr(PreparedChunkedKitchenQKV, 'release'))

    def test_the_sparse_prepared_carrier_can_release(self):
        self.assertTrue(hasattr(PreparedSparseKitchen, 'release'))

    def test_two_pass_v_is_not_a_production_default(self):
        """Measured at 0 MiB and +82 ms; it must stay opt-in."""
        source = (
            PACK / 'h3_optimizations' / 'apply.py'
        ).read_text(encoding='utf-8')
        self.assertNotIn('v_mode=', source)
        self.assertNotIn('two_pass', source)
        self.assertEqual(
            ChunkedKitchenQKVProjector(routing_summaries=True).v_mode,
            'retain',
        )

    def test_the_shipped_projector_defaults_stay_opt_in(self):
        """Constructing without arguments must not pick up the new options."""
        projector = ChunkedKitchenQKVProjector()
        self.assertFalse(projector.strided_qk_input)
        self.assertFalse(projector.stream_output)
        self.assertEqual(projector.v_mode, 'retain')


if __name__ == '__main__':
    unittest.main()
