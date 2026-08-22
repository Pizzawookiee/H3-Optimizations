'''Runtime lifecycle for the post-RoPE H3 attention ordering experiment.'''

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
import time

import torch

from .attention_ordering import ORDERINGS, analyze_orderings, packed_permutation
from .mlp_sharing.config import parse_layers
from .model import get_h3_blocks, is_minimax_h3
from .patch import OWNER_MARKER, key_for
from .runtime.context import (
    H3RuntimeSession,
    RUNTIME_SESSION_KEY,
    get_runtime_snapshot,
    install_runtime_wrapper,
)


ORDERING_OBSERVER_KEY = 'h3_optimizations_attention_ordering_observer'
ORDERING_STATUS_KEY = 'h3_optimizations_attention_ordering_probe'
ORDERING_WRAPPER_KEY = 'h3_optimizations_attention_ordering_request'
DEFAULT_LAYERS = (0, 24, 49)
DEFAULT_STEPS = (0,)
DEFAULT_BUDGETS = (0.2, 0.3, 0.5)
LOG_PREFIX = '[H3 attention ordering probe]'


class AttentionOrderingProbeError(RuntimeError):
    pass


def _parse_indices(value, name):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
        if not parts:
            raise ValueError('%s must contain at least one index' % name)
        values = tuple(int(part) for part in parts)
    else:
        values = tuple(int(part) for part in value)
    if len(set(values)) != len(values) or min(values) < 0:
        raise ValueError('%s must contain unique non-negative indices' % name)
    return tuple(sorted(values))


def parse_budgets(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
    else:
        parts = tuple(value)
    budgets = []
    for part in parts:
        budget = float(part)
        if budget > 1.0:
            budget /= 100.0
        if not 0.0 < budget <= 1.0:
            raise ValueError('video budgets must lie in (0, 100]')
        budgets.append(budget)
    if not budgets:
        raise ValueError('at least one video budget is required')
    return tuple(sorted(set(budgets)))


@dataclass(frozen=True)
class AttentionOrderingConfig:
    layers: tuple = DEFAULT_LAYERS
    steps: tuple = DEFAULT_STEPS
    budgets: tuple = DEFAULT_BUDGETS
    query_samples: int = 64
    head_chunk: int = 2
    capture_uncond: bool = False
    run_tag: str = 'attention-ordering'

    def __post_init__(self):
        object.__setattr__(self, 'layers', parse_layers(self.layers))
        object.__setattr__(self, 'steps', _parse_indices(self.steps, 'steps'))
        object.__setattr__(self, 'budgets', parse_budgets(self.budgets))
        if isinstance(self.query_samples, bool) or not 1 <= int(self.query_samples) <= 1024:
            raise ValueError('query_samples must be in [1, 1024]')
        if isinstance(self.head_chunk, bool) or not 1 <= int(self.head_chunk) <= 56:
            raise ValueError('head_chunk must be in [1, 56]')
        tag = str(self.run_tag)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', tag):
            raise ValueError(
                'run_tag must start with a letter or number and contain only '
                'letters, numbers, dot, underscore, or dash'
            )

    @property
    def signature(self):
        return (
            tuple(self.layers),
            tuple(self.steps),
            tuple(self.budgets),
            int(self.query_samples),
            int(self.head_chunk),
            bool(self.capture_uncond),
            str(self.run_tag),
        )


class AttentionOrderingSession:
    def __init__(self, config):
        self.config = config
        self.request_serial = -1
        self.started = None
        self.records = []
        self.observed = set()
        self.permutations = {}
        self.last_report_directory = None

    def begin_request(self):
        if self.started is not None:
            raise AttentionOrderingProbeError('concurrent requests share one ordering probe')
        self.request_serial += 1
        self.started = time.perf_counter()
        self.records.clear()
        self.observed.clear()
        self.permutations.clear()

    def observe_attention(self, layer_index, transformer_options, q, k, v):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise AttentionOrderingProbeError(
                'the ordering probe requires the H3 sampler-step and packed-layout runtime'
            )
        layer_index = int(layer_index)
        step_index = int(snapshot.step_index)
        if layer_index not in self.config.layers or step_index not in self.config.steps:
            return
        branch_values = transformer_options.get('cond_or_uncond') or [0]
        branch = int(branch_values[0])
        if branch != 0 and not self.config.capture_uncond:
            return
        key = (step_index, layer_index, branch)
        if key in self.observed:
            return
        self.observed.add(key)
        layout_key = (
            int(snapshot.layout.seq_len),
            tuple(int(value) for value in snapshot.layout.video_range),
            tuple(int(value) for value in snapshot.layout.video_shape),
        )
        permutations = self.permutations.get(layout_key)
        if permutations is None:
            permutations = {
                name: packed_permutation(snapshot.layout, name)
                for name in ORDERINGS
            }
            self.permutations[layout_key] = permutations
        result = analyze_orderings(
            q,
            k,
            v,
            snapshot.layout,
            budgets=self.config.budgets,
            query_samples=self.config.query_samples,
            head_chunk=self.config.head_chunk,
            permutations=permutations,
        )
        result.update(
            {
                'request_id': int(snapshot.request_id),
                'step_index': step_index,
                'total_steps': int(snapshot.total_steps),
                'layer_index': layer_index,
                'branch': branch,
            }
        )
        self.records.append(result)

    def _output_directory(self):
        try:
            import folder_paths
            root = folder_paths.get_output_directory()
        except ImportError:
            root = os.path.abspath('output')
        stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        name = '%s_%s_request%d' % (self.config.run_tag, stamp, self.request_serial)
        return os.path.join(root, 'h3_token_ordering', name)

    @staticmethod
    def _report(records, error):
        lines = [
            'H3 post-RoPE target-video ordering experiment',
            'Production token ordering changed: no',
            'Projected QKV while armed: bypassed for observable floating Q/K/V',
            'Orderings: %s' % ', '.join(ORDERINGS),
            '',
        ]
        if error is not None:
            lines.append('Run error: %s: %s' % (type(error).__name__, error))
        for record in records:
            lines.append(
                'Layer %d, step %d, branch %d, video %s, sampled queries %d'
                % (
                    record['layer_index'],
                    record['step_index'],
                    record['branch'],
                    'x'.join(str(value) for value in record['video_shape']),
                    record['query_samples'],
                )
            )
            lines.append(
                '  ordering       Q cosine   K cosine   Q variance  K variance'
            )
            for name in ORDERINGS:
                row = record['orderings'][name]
                lines.append(
                    '  %-13s %9.6f  %9.6f  %10.6f  %10.6f'
                    % (
                        name,
                        row['q_block_cosine_similarity'],
                        row['k_block_cosine_similarity'],
                        row['q_block_variance'],
                        row['k_block_variance'],
                    )
                )
            lines.append('')
            for budget_index, budget in enumerate(record['orderings']['native']['budgets']):
                lines.append(
                    '  Video KV density %.1f%% (%d/%d tiles)'
                    % (
                        100.0 * budget['actual_video_tile_density'],
                        budget['retained_video_kv_tiles'],
                        budget['pure_video_kv_tiles'],
                    )
                )
                lines.append('    ordering       relative L1   relative L2   retained mass')
                for name in ORDERINGS:
                    row = record['orderings'][name]['budgets'][budget_index]
                    lines.append(
                        '    %-13s %11.6f   %11.6f   %11.6f'
                        % (
                            name,
                            row['relative_l1_error'],
                            row['relative_l2_error'],
                            row['retained_dense_attention_mass']['mean'],
                        )
                    )
            lines.append('')
        return '\n'.join(lines)

    def end_request(self, error=None):
        seconds = None if self.started is None else time.perf_counter() - self.started
        self.started = None
        directory = self._output_directory()
        os.makedirs(directory, exist_ok=False)
        with open(os.path.join(directory, 'results.json'), 'w', encoding='utf-8') as handle:
            json.dump(self.records, handle, indent=2, sort_keys=True)
        summary = {
            'completed': error is None,
            'error': None if error is None else '%s: %s' % (type(error).__name__, error),
            'seconds': seconds,
            'request_serial': int(self.request_serial),
            'post_rope': True,
            'visual_tokens_only': True,
            'output_alignment': 'native query indices via inverse permutation',
            'production_order_changed': False,
            'projected_qkv_bypassed': True,
            'layers': list(self.config.layers),
            'steps': list(self.config.steps),
            'budgets': list(self.config.budgets),
            'query_samples': int(self.config.query_samples),
            'records': len(self.records),
        }
        with open(os.path.join(directory, 'summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with open(os.path.join(directory, 'report.txt'), 'w', encoding='utf-8') as handle:
            handle.write(self._report(self.records, error))
        self.last_report_directory = directory
        logging.info('%s wrote %s', LOG_PREFIX, directory)


def observe_attention(layer_index, transformer_options, q, k, v):
    if not transformer_options:
        return
    observer = transformer_options.get(ORDERING_OBSERVER_KEY)
    if observer is not None:
        observer.observe_attention(layer_index, transformer_options, q, k, v)


def has_ordering_observer(transformer_options):
    return bool(transformer_options and transformer_options.get(ORDERING_OBSERVER_KEY) is not None)


def _make_request_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        session.begin_request()
        error = None
        try:
            return executor(*args, **kwargs)
        except Exception as exc:
            error = exc
            raise
        finally:
            session.end_request(error)

    return wrapper


def install_ordering_probe(model_patcher, config):
    if not isinstance(config, AttentionOrderingConfig):
        raise TypeError('config must be AttentionOrderingConfig')
    if not is_minimax_h3(model_patcher):
        raise AttentionOrderingProbeError('the ordering probe only supports MiniMax H3')
    blocks = len(get_h3_blocks(model_patcher))
    existing = getattr(model_patcher, 'object_patches', {})
    missing = [
        key_for(index)
        for index in range(blocks)
        if not getattr(existing.get(key_for(index)), OWNER_MARKER, False)
    ]
    if missing:
        raise AttentionOrderingProbeError(
            'place H3 Attention Ordering Probe after H3 Memory Optimization or '
            'H3 Sparse Attention; %s is not owned by H3-Optimizations' % missing[0]
        )

    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    if options.get('h3_optimizations_attention_backend') == 'comfy_kitchen_int8_prequantized':
        raise AttentionOrderingProbeError(
            'place H3 Attention Ordering Probe after H3 Sparse Attention; '
            'the dense chunked Kitchen backend cannot consume floating Q/K/V'
        )
    current = options.get(ORDERING_OBSERVER_KEY)
    if current is not None:
        if isinstance(current, AttentionOrderingSession) and current.config == config:
            return current
        raise AttentionOrderingProbeError('another H3 attention ordering probe is installed')
    runtime = options.get(RUNTIME_SESSION_KEY)
    if runtime is None:
        runtime = H3RuntimeSession(strict_layout=True)
        install_runtime_wrapper(model_patcher, runtime)
        options = model_patcher.model_options['transformer_options']
    elif not isinstance(runtime, H3RuntimeSession):
        raise AttentionOrderingProbeError('foreign H3 runtime session is already installed')
    else:
        runtime.strict_layout = True

    session = AttentionOrderingSession(config)
    options[ORDERING_OBSERVER_KEY] = session
    options[ORDERING_STATUS_KEY] = {
        'post_rope': True,
        'visual_tokens_only': True,
        'output_alignment': 'native query indices via inverse permutation',
        'production_order_changed': False,
        'projected_qkv_bypassed': True,
        'layers': list(config.layers),
        'steps': list(config.steps),
        'budgets': list(config.budgets),
    }
    import comfy.patcher_extension
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        ORDERING_WRAPPER_KEY,
        _make_request_wrapper(session),
        model_patcher.model_options,
        is_model_options=True,
    )
    logging.info(
        '%s armed: layers=%s steps=%s budgets=%s queries=%d',
        LOG_PREFIX,
        ','.join(str(value) for value in config.layers),
        ','.join(str(value) for value in config.steps),
        ','.join('%.0f%%' % (100.0 * value) for value in config.budgets),
        config.query_samples,
    )
    return session
