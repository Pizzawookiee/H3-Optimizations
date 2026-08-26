'''CPU-only contracts for the end-to-end attention/memory benchmark.'''

import asyncio
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


BENCHMARK = Path(__file__).resolve().parents[1] / 'benchmarks' / 'bench_attention_arms.py'
SPEC = importlib.util.spec_from_file_location('bench_attention_arms', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


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
    'H3BenchmarkForceQKVConfig0': {'input': {'required': {
        'model': ['MODEL', {}],
    }}},
    'H3BenchmarkAssertRoute': {'input': {'required': {
        'model': ['MODEL', {}],
        'attention': ['STRING', {}],
        'backend': ['STRING', {}],
        'qkv': ['STRING', {}],
        'projector': ['STRING', {}],
    }}},
    'H3AIMDOResidencyLimiter': {'input': {'required': {
        'model': ['MODEL', {}],
        'residency': ['COMBO', {
            'default': '2 blocks',
            'options': ['stock', '0 blocks', '1 block', '2 blocks', '4 blocks'],
        }],
    }}},
    'ModelAttentionBackend': {'input': {'required': {
        'model': ['MODEL'],
        'attention': [['pytorch attention', 'comfy kitchen attention']],
    }}},
    'PathchSageAttentionKJ': {'input': {
        'required': {
            'model': ['MODEL'],
            'sage_attention': [['disabled', 'auto'], {'default': False}],
        },
        'optional': {'allow_compile': ['BOOLEAN', {'default': False}]},
    }},
    'H3MemoryOptimization': {'input': {'required': {
        'model': ['MODEL', {}],
        'fused_qkv': ['COMBO', {'default': 'auto', 'options': ['auto', 'off']}],
        'mlp_memory': ['COMBO', {'default': 'auto', 'options': ['auto', 'off']}],
        'chunk_rows': ['INT', {'default': 4096}],
        'preserve_precision': ['BOOLEAN', {'default': True}],
        'precision_mode': ['COMBO', {
            'default': 'Auto',
            'options': ['Auto', 'BF16', 'Preserve native', 'Force quant'],
        }],
        'qkv_streaming_mode': ['COMBO', {
            'default': 'Auto',
            'options': ['Off', 'Auto', 'Forced'],
        }],
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
    return asyncio.run(bench.build_prompt(schemas or StubSchemas(), arm, frames, args))


def patch_nodes(graph):
    return [
        graph[key]
        for key in sorted(
            (key for key in graph if key.startswith('patch')),
            key=lambda value: int(value.split('_', 1)[0][5:]),
        )
    ]


class ArgumentTests(unittest.TestCase):
    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])

    def test_default_ladder(self):
        args = bench.parse_args(['--dry-run'])
        self.assertEqual(
            args.arm_list,
            ['kitchen', 'sage', 'sage_memory', 'h3opt_kv100', 'h3opt'],
        )
        self.assertEqual(args.baseline, 'kitchen')
        self.assertEqual(args.workload_list, ['5s', '10s'])

    def test_defaults_are_one_megapixel_and_short_measurement(self):
        args = bench.parse_args(['--dry-run'])
        self.assertEqual((args.width, args.height), (1376, 768))
        self.assertEqual(args.schedule_steps, 20)
        self.assertEqual(args.measure_steps, 3)
        self.assertEqual(args.warmup_steps, 1)


class ServerPreflightTests(unittest.IsolatedAsyncioTestCase):
    class Response:
        def __init__(self, argv):
            self.status = 200
            self.argv = argv

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def json(self):
            return {'system': {'argv': self.argv}}

    class Session:
        def __init__(self, argv):
            self.argv = argv

        def get(self, _url):
            return ServerPreflightTests.Response(self.argv)

    async def test_sage_arms_require_command_line_sage(self):
        await bench.require_command_line_sage(
            self.Session(['main.py', '--use-sage-attention']),
            'http://stub',
            ['sage_memory'],
        )
        with self.assertRaisesRegex(bench.BenchError, '--use-sage-attention'):
            await bench.require_command_line_sage(
                self.Session(['main.py']),
                'http://stub',
                ['sage'],
            )

    async def test_non_sage_arms_do_not_require_command_line_sage(self):
        await bench.require_command_line_sage(
            self.Session(['main.py']),
            'http://stub',
            ['kitchen', 'h3opt'],
        )


class PromptTests(unittest.TestCase):
    def test_every_arm_forces_config0_and_zero_aimdo(self):
        for arm in bench.ARMS:
            with self.subTest(arm=arm):
                graph, _ = build(arm)
                patches = patch_nodes(graph)
                self.assertEqual(patches[0]['class_type'], 'H3BenchmarkForceQKVConfig0')
                self.assertEqual(patches[-1]['class_type'], 'H3AIMDOResidencyLimiter')
                self.assertEqual(patches[-1]['inputs']['residency'], '0 blocks')

    def test_common_controls_are_wired_around_arm_specific_nodes(self):
        graph, _ = build('h3opt')
        patches = patch_nodes(graph)
        self.assertEqual(
            [node['class_type'] for node in patches],
            [
                'H3BenchmarkForceQKVConfig0',
                'H3MemoryOptimization',
                'H3SparseAttention',
                'H3AIMDOResidencyLimiter',
            ],
        )
        self.assertEqual(patches[0]['inputs']['model'], ['loader', 0])
        for index in range(1, len(patches)):
            prior_id = [
                key for key, value in graph.items()
                if value is patches[index - 1]
            ][0]
            self.assertEqual(patches[index]['inputs']['model'], [prior_id, 0])

    def test_sage_memory_preserves_default_sage_then_adds_memory_node(self):
        graph, _ = build('sage_memory')
        patches = patch_nodes(graph)
        self.assertEqual(
            [node['class_type'] for node in patches],
            [
                'H3BenchmarkForceQKVConfig0',
                'H3MemoryOptimization',
                'H3BenchmarkAssertRoute',
                'H3AIMDOResidencyLimiter',
            ],
        )
        self.assertEqual(patches[1]['inputs']['precision_mode'], 'Preserve native')
        self.assertEqual(patches[1]['inputs']['qkv_streaming_mode'], 'Off')
        self.assertEqual(patches[1]['inputs']['mlp_memory'], 'auto')
        self.assertEqual(patches[1]['inputs']['chunk_rows'], 4096)
        self.assertEqual(patches[2]['inputs']['attention'], 'sage')
        self.assertEqual(patches[2]['inputs']['backend'], 'sage_mem_eff')
        self.assertEqual(patches[2]['inputs']['qkv'], 'convrot_int8_dense_sage')
        self.assertEqual(
            patches[2]['inputs']['projector'],
            'streamed_dense_sage_qkv',
        )

    def test_full_density_streamed_arm_is_100_percent(self):
        graph, _ = build('h3opt_kv100')
        sparse = next(
            node for node in patch_nodes(graph)
            if node['class_type'] == 'H3SparseAttention'
        )
        self.assertEqual(sparse['inputs']['video_budget'], 1.0)
        self.assertIs(sparse['inputs']['denser_early_late_steps'], False)

    def test_default_streamed_arm_is_safe_30_percent(self):
        graph, _ = build('h3opt')
        sparse = next(
            node for node in patch_nodes(graph)
            if node['class_type'] == 'H3SparseAttention'
        )
        self.assertEqual(sparse['inputs']['video_budget'], 0.3)
        self.assertIs(sparse['inputs']['denser_early_late_steps'], False)

    def test_schedule_is_built_from_unpatched_model(self):
        for arm in bench.ARMS:
            graph, _ = build(arm)
            self.assertEqual(graph['schedule']['inputs']['model'], ['loader', 0])

    def test_arms_differ_only_in_patch_chain(self):
        stripped = {}
        for arm in ('stock', 'kitchen', 'sage', 'sage_memory', 'h3opt'):
            graph, _ = build(arm)
            stripped[arm] = {
                key: value for key, value in graph.items()
                if not key.startswith('patch') and key != 'guider'
            }
        reference = stripped['stock']
        for arm, graph in stripped.items():
            self.assertEqual(graph, reference, arm)

    def test_graph_terminates_without_vae_decode(self):
        graph, _ = build('kitchen')
        classes = {node['class_type'] for node in graph.values()}
        self.assertIn('PreviewAny', classes)
        self.assertNotIn('VAEDecode', classes)
        self.assertNotIn('SaveLatent', classes)

    def test_split_yields_requested_measurement_count(self):
        for warmup in (0, 1, 3):
            for measure in (1, 3, 5):
                _, executed = build('kitchen', warmup_steps=warmup, measure_steps=measure)
                boundaries = [(value, float(value)) for value in range(1, executed + 1)]
                got, _ = bench.step_durations(boundaries, warmup)
                self.assertEqual(len(got), measure)


class LadderTests(unittest.TestCase):
    @staticmethod
    def chain(arm):
        return [(node_type, dict(overrides)) for node_type, overrides in bench.ARMS[arm]]

    def test_sage_memory_adds_memory_and_fail_closed_route_assertion(self):
        sage = self.chain('sage')
        memory = self.chain('sage_memory')
        self.assertEqual(sage, [])
        self.assertEqual(memory[0][0], 'H3MemoryOptimization')
        self.assertEqual(memory[0][1]['qkv_streaming_mode'], 'Off')
        self.assertEqual(
            memory[0][1]['precision_mode'],
            'Preserve native',
        )
        self.assertEqual(memory[1][0], 'H3BenchmarkAssertRoute')

    def test_streamed_100_and_30_arms_differ_only_in_density(self):
        full = self.chain('h3opt_kv100')
        default = self.chain('h3opt')
        self.assertEqual(len(full), len(default))
        for (left_type, left), (right_type, right) in zip(full, default):
            self.assertEqual(left_type, right_type)
            differences = {
                key for key in set(left) | set(right)
                if left.get(key) != right.get(key)
            }
            self.assertIn(differences, (set(), {'video_budget'}))
        self.assertEqual(full[-1][1]['video_budget'], 1.0)
        self.assertEqual(default[-1][1]['video_budget'], 0.3)

    def test_every_arm_has_publishable_label(self):
        self.assertEqual(set(bench.ARM_LABELS), set(bench.ARMS))
        self.assertIn('64x64', bench.ARM_LABELS['h3opt'])
        self.assertIn('30%', bench.ARM_LABELS['h3opt'])
        self.assertIn('100%', bench.ARM_LABELS['h3opt_kv100'])


class StepTimingTests(unittest.TestCase):
    def test_warmup_steps_are_dropped(self):
        boundaries = [(0, 10.0), (1, 11.0), (2, 11.5), (3, 12.0), (4, 12.4)]
        measured, every = bench.step_durations(boundaries, 1)
        self.assertEqual([round(value) for value in every], [1000, 500, 500, 400])
        self.assertEqual([round(value) for value in measured], [500, 500, 400])


class ReportTests(unittest.TestCase):
    def test_table_states_fairness_controls(self):
        args = SimpleNamespace(
            measure_steps=3, warmup_steps=1, schedule_steps=20,
            sampler='res_multistep', scheduler='simple', width=1376, height=768,
        )
        records = [
            {'arm': 'kitchen', 'workload': '5s', 'median_step_ms': 1000.0, 'peak_mib': 9000.0},
            {'arm': 'h3opt', 'workload': '5s', 'median_step_ms': 500.0, 'peak_mib': 8000.0},
        ]
        table = bench.render_table(records, args, 'kitchen')
        self.assertIn('CUTLASS config 0', table)
        self.assertIn('AIMDO residency to 0 blocks', table)
        self.assertIn('2.00x', table)


class ResetSemanticsTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_unloading_never_resets_executor(self):
        session = self.RecordingSession()
        await bench.reset_between_arms(session, 'http://stub', True)
        _, payload = session.posts[0]
        self.assertIs(payload['unload_models'], True)
        self.assertIs(payload['free_memory'], False)


if __name__ == '__main__':
    unittest.main()


class SageLaunchFlagTests(unittest.IsolatedAsyncioTestCase):
    '''The Sage arms come from a launch flag, so the benchmark must verify it.

    Neither Sage row puts an attention node in the graph. On a server started
    without --use-sage-attention they would silently measure the default
    backend and still be reported as SageAttention.
    '''

    class Session:
        def __init__(self, argv, status=200):
            self.argv = argv
            self.status = status
            self.requested = []

        def get(self, url):
            self.requested.append(url)
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {'system': {'argv': self.argv}}

    async def test_missing_flag_is_refused(self):
        session = self.Session(['main.py', '--port', '8189'])
        with self.assertRaises(bench.BenchError) as caught:
            await bench.require_command_line_sage(
                session, 'http://stub', ['kitchen', 'sage']
            )
        self.assertIn('--use-sage-attention', str(caught.exception))

    async def test_present_flag_passes(self):
        session = self.Session(['main.py', '--use-sage-attention'])
        await bench.require_command_line_sage(
            session, 'http://stub', ['sage', 'sage_memory']
        )

    async def test_no_sage_arms_skips_the_check_entirely(self):
        session = self.Session([])
        await bench.require_command_line_sage(
            session, 'http://stub', ['kitchen', 'h3opt']
        )
        self.assertEqual(session.requested, [])

    async def test_unreadable_system_stats_is_refused(self):
        session = self.Session([], status=503)
        with self.assertRaises(bench.BenchError):
            await bench.require_command_line_sage(
                session, 'http://stub', ['sage']
            )
