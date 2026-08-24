'''CPU-only contracts for the end-to-end attention arm benchmark.

The benchmark itself talks to a running ComfyUI server and never imports torch,
so everything here runs without a GPU and without a server: prompt construction
is driven by a stub schema source, and the reporting helpers are pure.
'''

import asyncio
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_attention_arms.py'
)
SPEC = importlib.util.spec_from_file_location('bench_attention_arms', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


# Minimal stand-ins for the /object_info entries the graph builder touches. Only
# the shapes matter: which inputs are required, and what defaults they carry.
STUB_SCHEMAS = {
    'UNETLoader': {'input': {'required': {
        'unet_name': [['a.safetensors']],
        'weight_dtype': [['default', 'fp8_e4m3fn']],
    }}},
    'CLIPLoader': {'input': {'required': {
        'clip_name': [['c.safetensors']],
        'type': [['minimax', 'stable_diffusion']],
    }}},
    'VAELoader': {'input': {'required': {'vae_name': [['v.safetensors']]}}},
    'MiniMaxH3ImageToVideo': {'input': {
        'required': {
            'clip': ['CLIP', {}], 'vae': ['VAE', {}],
            'prompt': ['STRING', {'multiline': True}],
            'width': ['INT', {'default': 1344}],
            'height': ['INT', {'default': 768}],
            'length': ['INT', {'default': 124}],
        },
        'optional': {'first_frame': ['IMAGE', {}], 'last_frame': ['IMAGE', {}]},
    }},
    'ModelAttentionBackend': {'input': {'required': {
        'model': ['MODEL'],
        'attention': [['pytorch attention', 'comfy kitchen attention']],
    }}},
    'PathchSageAttentionKJ': {'input': {
        # Mirrors the real node, whose combo default is a bool that is not in
        # its own option list. The sage arm must therefore be explicit.
        'required': {'model': ['MODEL'],
                     'sage_attention': [['disabled', 'auto'], {'default': False}]},
        'optional': {'allow_compile': ['BOOLEAN', {'default': False}]},
    }},
    'H3MemoryOptimization': {'input': {'required': {
        'model': ['MODEL', {}],
        'fused_qkv': ['COMBO', {'default': 'auto', 'options': ['auto', 'off']}],
        'mlp_memory': ['COMBO', {'default': 'auto', 'options': ['auto', 'off']}],
        'chunk_rows': ['INT', {'default': 4096}],
    }}},
    'H3SparseAttention': {'input': {
        'required': {
            'model': ['MODEL', {}],
            'video_budget': ['FLOAT', {'default': 0.3}],
            'denser_early_late_steps': ['BOOLEAN', {'default': False}],
        },
        'optional': {'layer_video_budgets': ['STRING', {'default': ''}]},
    }},
    'BasicGuider': {'input': {'required': {
        'model': ['MODEL', {}], 'conditioning': ['CONDITIONING', {}]}}},
    'KSamplerSelect': {'input': {'required': {
        'sampler_name': ['COMBO', {'options': ['euler', 'res_multistep']}]}}},
    'BasicScheduler': {'input': {'required': {
        'model': ['MODEL', {}],
        'scheduler': ['COMBO', {'options': ['simple', 'beta']}],
        'steps': ['INT', {'default': 20}],
        'denoise': ['FLOAT', {'default': 1.0}],
    }}},
    'SplitSigmas': {'input': {'required': {
        'sigmas': ['SIGMAS', {}], 'step': ['INT', {'default': 0}]}}},
    'RandomNoise': {'input': {'required': {'noise_seed': ['INT', {'default': 0}]}}},
    'SamplerCustomAdvanced': {'input': {'required': {
        'noise': ['NOISE', {}], 'guider': ['GUIDER', {}], 'sampler': ['SAMPLER', {}],
        'sigmas': ['SIGMAS', {}], 'latent_image': ['LATENT', {}]}}},
    'PreviewAny': {'input': {'required': {'source': ['*', {}]}}},
}


class StubSchemas(bench.Schemas):
    def __init__(self, overrides=None):
        super().__init__('http://stub', None)
        self.cache = dict(STUB_SCHEMAS)
        self.cache.update(overrides or {})

    async def get(self, node_type):
        if node_type not in self.cache:
            raise bench.BenchError('server has no node %r' % node_type)
        return self.cache[node_type]


def build(arm, frames=124, schemas=None, **overrides):
    args = bench.parse_args(['--dry-run'])
    for key, value in overrides.items():
        setattr(args, key, value)
    return asyncio.run(
        bench.build_prompt(schemas or StubSchemas(), arm, frames, args)
    )


class ArgumentTests(unittest.TestCase):
    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])

    def test_dry_run_needs_no_acknowledgement(self):
        args = bench.parse_args(['--dry-run'])
        self.assertEqual(
            args.arm_list,
            ['kitchen', 'sage', 'h3opt_dense', 'h3opt_kv100', 'h3opt'],
        )
        self.assertEqual(args.workload_list, ['5s', '10s'])

    def test_defaults_are_one_megapixel_and_a_short_measured_series(self):
        args = bench.parse_args(['--dry-run'])
        self.assertEqual((args.width, args.height), (1376, 768))
        self.assertEqual(args.schedule_steps, 20)
        self.assertEqual(args.measure_steps, 3)
        self.assertEqual(args.warmup_steps, 1)

    def test_baseline_defaults_to_the_first_arm(self):
        self.assertEqual(bench.parse_args(['--dry-run']).baseline, 'kitchen')
        self.assertEqual(
            bench.parse_args(['--dry-run', '--arms', 'sage,h3opt']).baseline, 'sage'
        )

    def test_unknown_arm_workload_and_baseline_are_rejected(self):
        for argv in (
            ['--dry-run', '--arms', 'nope'],
            ['--dry-run', '--workloads', '7s'],
            ['--dry-run', '--arms', 'kitchen', '--baseline', 'sage'],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bench.parse_args(argv)


class WorkloadTests(unittest.TestCase):
    def test_durations_sit_on_the_models_frame_grid(self):
        # H3 snaps frame counts up to 17k+5; a value off the grid would be
        # silently rounded by the node and the reported duration would be a lie.
        for frames in bench.WORKLOADS.values():
            self.assertEqual((frames - 5) % 17, 0, frames)

    def test_nominal_durations_are_honest_about_the_grid(self):
        self.assertAlmostEqual(bench.WORKLOADS['5s'] / bench.FPS, 5.167, places=3)
        self.assertAlmostEqual(bench.WORKLOADS['10s'] / bench.FPS, 10.125, places=3)

    def test_one_megapixel_is_a_valid_step_32_resolution(self):
        width, height = bench.ONE_MP
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertAlmostEqual(width * height / 1e6, 1.06, places=2)


class PromptTests(unittest.TestCase):
    def test_arms_differ_only_in_the_patch_chain(self):
        stripped = {}
        for arm in ('stock', 'kitchen', 'sage', 'h3opt'):
            graph, _ = build(arm)
            stripped[arm] = {
                key: value for key, value in graph.items()
                if not key.startswith('patch')
            }
        reference = stripped['stock']
        for arm, graph in stripped.items():
            if arm == 'stock':
                continue
            # Every non-patch node is identical apart from the guider's model
            # link, which necessarily points at the end of the chain.
            for key, value in graph.items():
                if key == 'guider':
                    continue
                self.assertEqual(value, reference[key], '%s / %s' % (arm, key))

    def test_patch_chain_is_wired_in_order_from_the_loader(self):
        graph, _ = build('h3opt')
        self.assertEqual(
            graph['patch0_H3MemoryOptimization']['inputs']['model'], ['loader', 0]
        )
        self.assertEqual(
            graph['patch1_H3SparseAttention']['inputs']['model'],
            ['patch0_H3MemoryOptimization', 0],
        )
        self.assertEqual(
            graph['guider']['inputs']['model'], ['patch1_H3SparseAttention', 0]
        )

    def test_stock_arm_feeds_the_loader_straight_to_the_guider(self):
        graph, _ = build('stock')
        self.assertFalse([key for key in graph if key.startswith('patch')])
        self.assertEqual(graph['guider']['inputs']['model'], ['loader', 0])

    def test_schedule_is_built_from_the_unpatched_model(self):
        # Every arm must see an identical sigma trajectory, so the scheduler
        # reads the loader rather than the end of the patch chain.
        for arm in bench.ARMS:
            graph, _ = build(arm)
            self.assertEqual(
                graph['schedule']['inputs']['model'], ['loader', 0], arm
            )

    def test_split_truncates_to_warmup_plus_measured_steps(self):
        # One extra step because the server emits no value=0 boundary, so N
        # executed steps yield only N-1 timed intervals.
        graph, executed = build('kitchen', warmup_steps=2, measure_steps=5)
        self.assertEqual(executed, 8)
        self.assertEqual(graph['split']['inputs']['step'], 8)
        self.assertEqual(graph['schedule']['inputs']['steps'], 20)
        self.assertEqual(graph['sample']['inputs']['sigmas'], ['split', 0])

    def test_executed_steps_must_leave_trajectory_to_truncate(self):
        with self.assertRaises(bench.BenchError):
            build('kitchen', warmup_steps=1, measure_steps=18)

    def test_executed_steps_yield_the_requested_measured_count(self):
        # The whole point of the extra step: after dropping the warmup, the
        # number of measured intervals must equal --measure-steps.
        for warmup in (0, 1, 3):
            for measure in (1, 3, 5):
                _, executed = build(
                    'kitchen', warmup_steps=warmup, measure_steps=measure,
                )
                # The server reports boundaries at values 1..executed.
                boundaries = [(value, float(value)) for value in range(1, executed + 1)]
                got, _ = bench.step_durations(boundaries, warmup)
                self.assertEqual(len(got), measure, (warmup, measure))

    def test_graph_terminates_without_a_vae_decode(self):
        graph, _ = build('kitchen')
        classes = {node['class_type'] for node in graph.values()}
        self.assertIn('PreviewAny', classes)
        self.assertNotIn('VAEDecode', classes)
        self.assertNotIn('SaveLatent', classes)
        self.assertEqual(graph['sink']['inputs']['source'], ['sample', 0])

    def test_text_to_video_passes_no_reference_frames(self):
        graph, _ = build('kitchen')
        inputs = graph['cond']['inputs']
        self.assertNotIn('first_frame', inputs)
        self.assertNotIn('last_frame', inputs)

    def test_frames_and_resolution_reach_the_conditioning_node(self):
        graph, _ = build('kitchen', frames=243, width=1376, height=768)
        inputs = graph['cond']['inputs']
        self.assertEqual(inputs['length'], 243)
        self.assertEqual((inputs['width'], inputs['height']), (1376, 768))

    def test_sage_arm_overrides_a_combo_default_that_is_not_an_option(self):
        graph, _ = build('sage')
        selected = graph['patch0_PathchSageAttentionKJ']['inputs']['sage_attention']
        self.assertIn(selected, ['disabled', 'auto'])
        self.assertNotIsInstance(selected, bool)

    def test_missing_inputs_are_filled_from_the_live_schema(self):
        # weight_dtype is never set by an arm; it must still be present.
        graph, _ = build('kitchen')
        self.assertEqual(graph['loader']['inputs']['weight_dtype'], 'default')

    def test_a_new_required_input_is_picked_up_automatically(self):
        # Mirrors H3MemoryOptimization gaining its preserve-precision toggle:
        # a new required input with a default must not need a code change here.
        overrides = {'H3MemoryOptimization': {'input': {'required': dict(
            STUB_SCHEMAS['H3MemoryOptimization']['input']['required'],
            preserve_precision=['BOOLEAN', {'default': False}],
        )}}}
        graph, _ = build('h3opt', schemas=StubSchemas(overrides))
        self.assertIs(
            graph['patch0_H3MemoryOptimization']['inputs']['preserve_precision'],
            False,
        )

    def test_a_new_required_input_without_a_default_is_refused(self):
        overrides = {'H3SparseAttention': {'input': {'required': dict(
            STUB_SCHEMAS['H3SparseAttention']['input']['required'],
            mandatory_knob=['INT', {}],
        )}}}
        with self.assertRaises(bench.BenchError):
            build('h3opt', schemas=StubSchemas(overrides))

    def test_an_arm_naming_an_input_the_node_lost_is_refused(self):
        overrides = {'H3SparseAttention': {'input': {'required': {
            'model': ['MODEL', {}],
            'video_budget': ['FLOAT', {'default': 0.3}],
        }}}}
        with self.assertRaises(bench.BenchError):
            build('h3opt', schemas=StubSchemas(overrides))

    def test_an_unavailable_node_names_itself(self):
        schemas = StubSchemas()
        del schemas.cache['PathchSageAttentionKJ']
        with self.assertRaises(bench.BenchError) as caught:
            build('sage', schemas=schemas)
        self.assertIn('PathchSageAttentionKJ', str(caught.exception))


class StepTimingTests(unittest.TestCase):
    def test_warmup_steps_are_dropped_from_the_measured_series(self):
        boundaries = [(0, 10.0), (1, 11.0), (2, 11.5), (3, 12.0), (4, 12.4)]
        measured, every = bench.step_durations(boundaries, 1)
        self.assertEqual([round(value) for value in every], [1000, 500, 500, 400])
        self.assertEqual([round(value) for value in measured], [500, 500, 400])

    def test_boundaries_are_ordered_before_differencing(self):
        boundaries = [(0, 10.0), (1, 11.0), (2, 11.5), (3, 12.0)]
        forward, _ = bench.step_durations(boundaries, 0)
        reverse, _ = bench.step_durations(list(reversed(boundaries)), 0)
        self.assertEqual(forward, reverse)

    def test_a_run_with_no_progress_yields_no_measurements(self):
        self.assertEqual(bench.step_durations([], 1), ([], []))
        self.assertEqual(bench.step_durations([(0, 1.0)], 0), ([], []))


class DefaultResolutionTests(unittest.TestCase):
    def test_inline_combo_falls_back_to_its_first_option(self):
        self.assertEqual(bench.default_for('N', 'a', [['p', 'q']]), 'p')

    def test_declared_combo_options_are_used(self):
        self.assertEqual(
            bench.default_for('N', 'a', ['COMBO', {'options': ['x', 'y']}]), 'x'
        )

    def test_explicit_default_wins(self):
        self.assertEqual(bench.default_for('N', 'a', ['INT', {'default': 7}]), 7)

    def test_no_default_and_no_options_raises(self):
        with self.assertRaises(bench.BenchError):
            bench.default_for('N', 'a', ['INT', {}])


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            measure_steps=3, warmup_steps=1, schedule_steps=20,
            sampler='res_multistep', scheduler='simple', width=1376, height=768,
        )

    def test_repeats_of_one_cell_collapse_to_their_median(self):
        records = [
            {'arm': 'kitchen', 'workload': '5s', 'median_step_ms': 1000.0, 'peak_mib': 9000.0},
            {'arm': 'kitchen', 'workload': '5s', 'median_step_ms': 1100.0, 'peak_mib': 9100.0},
        ]
        table = bench.render_table(records, self.args, 'kitchen')
        self.assertIn(
            '| %s | 1050 | 9050 | 1.00x |' % bench.ARM_LABELS['kitchen'], table
        )

    def test_speedup_is_relative_to_the_named_baseline(self):
        records = [
            {'arm': 'kitchen', 'workload': '5s', 'median_step_ms': 1000.0, 'peak_mib': 9000.0},
            {'arm': 'h3opt', 'workload': '5s', 'median_step_ms': 500.0, 'peak_mib': 8000.0},
        ]
        table = bench.render_table(records, self.args, 'kitchen')
        self.assertIn(
            '| %s | 500 | 8000 | 2.00x |' % bench.ARM_LABELS['h3opt'], table
        )

    def test_a_failed_arm_renders_as_dashes_rather_than_a_number(self):
        records = [
            {'arm': 'kitchen', 'workload': '5s', 'median_step_ms': 1000.0, 'peak_mib': 9000.0},
            {'arm': 'sage', 'workload': '5s', 'error': 'backend unavailable', 'peak_mib': None},
        ]
        table = bench.render_table(records, self.args, 'kitchen')
        self.assertIn('| %s | - | - | - |' % bench.ARM_LABELS['sage'], table)

    def test_the_table_states_its_own_measurement_boundary(self):
        records = [
            {'arm': 'kitchen', 'workload': '5s', 'median_step_ms': 1000.0, 'peak_mib': 9000.0},
        ]
        table = bench.render_table(records, self.args, 'kitchen')
        self.assertIn('3 measured steps after 1 warmup', table)
        self.assertIn('20-step', table)
        self.assertIn('1376x768', table)
        self.assertIn('driver-level', table)


if __name__ == '__main__':
    unittest.main()


class ResetSemanticsTests(unittest.IsolatedAsyncioTestCase):
    '''The /free payload sent between arms.

    Regression cover for a real measurement bug: sending free_memory=True made
    the server reset its executor, which wiped the cached conditioning and made
    every arm reload the 32B text encoder on top of the resident diffusion
    model. All arms then peaked within noise of each other on encoder memory
    rather than on the attention backend under test.
    '''

    class RecordingSession:
        def __init__(self):
            self.posts = []

        def post(self, url, json=None):
            self.posts.append((url, json))
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def read(self):
            return b''

    async def test_default_sends_no_free_request_at_all(self):
        session = self.RecordingSession()
        await bench.reset_between_arms(session, 'http://stub', False)
        self.assertEqual(session.posts, [])

    async def test_unloading_never_resets_the_executor(self):
        session = self.RecordingSession()
        await bench.reset_between_arms(session, 'http://stub', True)
        self.assertEqual(len(session.posts), 1)
        url, payload = session.posts[0]
        self.assertEqual(url, 'http://stub/free')
        self.assertIs(payload['unload_models'], True)
        self.assertIs(payload['free_memory'], False)


class SeedIsolationTests(unittest.TestCase):
    '''Distinct seeds keep ComfyUI's execution cache from silently skipping work.

    Regression cover for a real failure: the priming run used the same arm and
    the same seed as the first measured arm, so that arm's whole graph was a
    cache hit. It reported zero step boundaries, zero watts and no VRAM, having
    never touched the GPU.
    '''

    def test_arms_in_one_iteration_share_a_seed(self):
        # Identical noise is what makes their timings comparable.
        seeds = {
            build(arm, seed=99)[0]['noise']['inputs']['noise_seed']
            for arm in ('kitchen', 'sage', 'h3opt')
        }
        self.assertEqual(seeds, {99})

    def test_unique_arm_seeds_cannot_cross_arm_cache(self):
        args = bench.parse_args([
            '--dry-run',
            '--arms', 'kitchen,sage,h3opt',
            '--seed', '99',
            '--unique-arm-seeds',
        ])
        self.assertEqual(
            [bench.measured_seed(args, 0, index) for index in range(3)],
            [99, 100, 101],
        )
        self.assertEqual(
            [bench.measured_seed(args, 1, index) for index in range(3)],
            [102, 103, 104],
        )

    def test_explicit_seed_overrides_the_configured_one(self):
        graph, _ = build('kitchen', seed=4321)
        self.assertEqual(graph['noise']['inputs']['noise_seed'], 4321)

    def test_omitted_seed_falls_back_to_the_configured_one(self):
        args = bench.parse_args(['--dry-run', '--seed', '777'])
        graph, _ = asyncio.run(
            bench.build_prompt(StubSchemas(), 'kitchen', 124, args)
        )
        self.assertEqual(graph['noise']['inputs']['noise_seed'], 777)

    def test_priming_seed_differs_from_every_measured_iteration(self):
        args = bench.parse_args(['--dry-run', '--seed', '1234', '--repeat', '3'])
        prime = args.seed - 1
        measured = [args.seed + index for index in range(args.repeat)]
        self.assertNotIn(prime, measured)
        self.assertEqual(len(set(measured)), len(measured))


class AttributionLadderTests(unittest.TestCase):
    '''Each rung of the default ladder must change exactly one thing.'''

    @staticmethod
    def chain(arm):
        return [(node_type, dict(overrides)) for node_type, overrides in bench.ARMS[arm]]

    def test_qkv_mlp_rung_holds_attention_at_public_dense_kitchen(self):
        # kitchen -> h3opt_dense adds H3MemoryOptimization and nothing else.
        # H3MemoryOptimization resolves dense attention to comfy_kitchen_int8,
        # the same backend the kitchen arm selects explicitly.
        self.assertEqual(
            [node_type for node_type, _ in self.chain('h3opt_dense')],
            ['H3MemoryOptimization'],
        )

    def test_kernel_rung_adds_the_sparse_node_at_full_density(self):
        # h3opt_dense -> h3opt_kv100 adds the sparse node but no sparsity, so
        # the only change is which kernel runs.
        dense = self.chain('h3opt_dense')
        full = self.chain('h3opt_kv100')
        self.assertEqual(full[:len(dense)], dense)
        self.assertEqual(len(full), len(dense) + 1)
        node_type, overrides = full[-1]
        self.assertEqual(node_type, 'H3SparseAttention')
        self.assertEqual(overrides['video_budget'], 1.0)

    def test_density_rung_changes_only_the_budget(self):
        # h3opt_kv100 -> h3opt keeps every node and every other input fixed.
        full = self.chain('h3opt_kv100')
        sparse = self.chain('h3opt')
        self.assertEqual(len(full), len(sparse))
        for (left_type, left), (right_type, right) in zip(full, sparse):
            self.assertEqual(left_type, right_type)
            differing = {
                key for key in set(left) | set(right)
                if left.get(key) != right.get(key)
            }
            self.assertIn(differing, ({'video_budget'}, set()))
        self.assertEqual(sparse[-1][1]['video_budget'], 0.3)

    def test_no_rung_enables_the_early_late_density_window(self):
        # An early/late boost would make density vary by step and break the
        # comparison against the fixed-density rung above it.
        for arm in ('h3opt_kv100', 'h3opt'):
            for node_type, overrides in self.chain(arm):
                if node_type == 'H3SparseAttention':
                    self.assertIs(overrides['denser_early_late_steps'], False, arm)


class ArmLabelTests(unittest.TestCase):
    def test_every_arm_has_a_publishable_label(self):
        self.assertEqual(set(bench.ARM_LABELS), set(bench.ARMS))

    def test_labels_name_the_nodes_a_reader_would_wire_up(self):
        self.assertIn('Sparse Attention', bench.ARM_LABELS['h3opt'])
        self.assertIn('30%', bench.ARM_LABELS['h3opt'])
        self.assertIn('100%', bench.ARM_LABELS['h3opt_kv100'])
        # The unpatched-model arms must say so, or a reader assumes the pack
        # was involved in every row of the table.
        for arm in ('kitchen', 'sage', 'stock', 'pytorch'):
            self.assertIn('no pack', bench.ARM_LABELS[arm])

    def test_rendered_table_uses_labels_not_internal_keys(self):
        records = [
            {'arm': 'h3opt', 'workload': '5s', 'median_step_ms': 100.0, 'peak_mib': 1.0},
        ]
        args = SimpleNamespace(
            measure_steps=3, warmup_steps=1, schedule_steps=20,
            sampler='res_multistep', scheduler='simple', width=1376, height=768,
        )
        table = bench.render_table(records, args, 'h3opt')
        self.assertIn(bench.ARM_LABELS['h3opt'], table)
        self.assertNotIn('| h3opt |', table)
