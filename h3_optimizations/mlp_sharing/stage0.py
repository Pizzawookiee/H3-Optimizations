'''Stage 0: one dense-MLP run that scores every downstream design choice.

"Dense" here means the exact MLP, not dense attention. Sparse attention runs at
its normal budget so the attention route exists; a step whose route came back
fully dense is recorded as such and takes no selector statistics, which is the
conservative `attention budget 100% -> MLP budget 100%` policy.

Two tiers of measurement share the run:

Tier A covers every target-video KV tile at every measured layer and step. It
compares each attention selector's top-k against the true MLP-update oracle
``|gate_mlp * y|^2`` aggregated to the same 64-row tiles.

Tier B/C runs on a small fixed sample of 128-row video blocks. It measures the
cross-step SwiGLU delta structure, how much of that delta AdaLN modulation
explains, and whether an FP8 cache preserves the values and the group ranking.

Nothing here changes the denoiser result: the exact MLP output is what the
block uses, and every diagnostic evaluation runs on copies.
'''

from __future__ import annotations

import json
import logging
import os
import time

import torch

from ..memory.observer import OBSERVER_KEY
from ..memory.patch import OWNER_MARKER, key_for
from ..memory.sharing import SHARING_KEY
from ..model import get_h3_blocks, is_minimax_h3
from ..runtime.context import (
    H3RuntimeSession,
    RUNTIME_SESSION_KEY,
    get_runtime_snapshot,
    install_runtime_wrapper,
)
from .config import STAGE0_SAMPLE_ROWS, Stage0Config
from .route import ROUTE_KEY, AttentionRouteRecorder, get_route_recorder
from . import stage0_metrics as metrics

STAGE0_STATUS_KEY = 'h3_optimizations_mlp_stage0'
STAGE0_WRAPPER_KEY = 'h3_optimizations_mlp_stage0_request'
LOG_PREFIX = '[H3 MLP Stage 0]'


class Stage0Error(RuntimeError):
    pass


def _mod_rows(values, selector, rows):
    '''Modulation rows for a chunk-local row subset.'''
    if torch.is_tensor(selector):
        picked = selector.to(device=values.device).index_select(0, rows)
        return values.index_select(0, picked.long())
    # Left un-expanded: a scalar segment selector broadcasts, and the held
    # sample would otherwise store one full-width copy per sampled row.
    return values[int(selector)].unsqueeze(0)


def _unmodulate(h, shift_rows, scale_rows):
    '''Recover RMSNorm(x) from the modulated MLP input.'''
    denominator = 1.0 + scale_rows.float()
    guard = torch.copysign(
        torch.full_like(denominator, 1.0e-3),
        denominator,
    )
    denominator = torch.where(denominator.abs() < 1.0e-3, guard, denominator)
    return (h.float() - shift_rows.float()) / denominator


def _modulate(normalized, shift_rows, scale_rows, dtype):
    return (
        normalized * (1.0 + scale_rows.float()) + shift_rows.float()
    ).to(dtype)


def sample_blocks(layout, kv_tile, count):
    '''Evenly spaced 128-row target-video blocks, stable across steps.

    Blocks start on a 128-row boundary at or after the first pure-video KV
    tile, so both the 64-row and 128-row sharing granularities are meaningful
    and every sampled row is one Sparse Sage routes sparsely.
    '''
    video_start, video_stop = (int(value) for value in layout.video_range)
    first_pure = ((video_start + kv_tile - 1) // kv_tile) * kv_tile
    origin = ((first_pure + STAGE0_SAMPLE_ROWS - 1) // STAGE0_SAMPLE_ROWS)
    origin *= STAGE0_SAMPLE_ROWS
    available = (video_stop - origin) // STAGE0_SAMPLE_ROWS
    if available <= 0:
        return ()
    count = min(int(count), available)
    return tuple(
        origin + ((index * available) // count) * STAGE0_SAMPLE_ROWS
        for index in range(count)
    )


class Stage0Session:
    '''Collect Stage 0 statistics for one sampler request.'''

    def __init__(self, config):
        self.config = config
        self.request_serial = -1
        self.schedule = ()
        self.selection_records = []
        self.cache_records = []
        self.notes = []
        self.pending = {}
        self.cached = {}
        self.started = None
        self.last_report_directory = None

    # -- request lifecycle --------------------------------------------------

    def begin_request(self, sigmas):
        if self.started is not None:
            raise Stage0Error('concurrent requests share one Stage 0 session')
        self.request_serial += 1
        self.selection_records.clear()
        self.cache_records.clear()
        self.notes.clear()
        self.pending.clear()
        self.cached.clear()
        self.started = time.perf_counter()
        if torch.is_tensor(sigmas):
            self.schedule = tuple(
                float(value)
                for value in sigmas.detach().flatten().float().cpu().tolist()
            )
        else:
            self.schedule = ()

    def _sigma(self, step_index):
        index = int(step_index)
        if 0 <= index < len(self.schedule):
            return self.schedule[index]
        return None

    def _note(self, message):
        if message not in self.notes:
            self.notes.append(message)

    # -- per-chunk observation ---------------------------------------------

    def observe_exact_mlp(
        self,
        layer_index,
        transformer_options,
        *,
        h,
        y,
        residual,
        gate,
        selector,
        chunk_start,
        chunk_stop,
        evaluate_mlp,
        shift=None,
        scale=None,
        evaluate_activation=None,
        apply_fc2=None,
        mlp_path=None,
        **unused,
    ):
        del residual, evaluate_mlp, unused
        if int(layer_index) not in self.config.layers:
            return
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise Stage0Error(
                'Stage 0 requires the H3 sampler-step and packed-layout runtime'
            )
        recorder = get_route_recorder(transformer_options)
        if recorder is None:
            raise Stage0Error('Stage 0 requires its attention route recorder')
        route = recorder.route(layer_index)
        if route is None or not route.has_selection:
            self._note(
                'layer %d step %d ran without a sparse attention route; enable '
                'H3 Sparse Attention so hit_rate exists'
                % (int(layer_index), int(snapshot.step_index))
            )
            return

        pending = self._pending(snapshot, layer_index, route, mlp_path)
        self._observe_selection(pending, route, y, gate, selector, chunk_start)
        if self.config.measure_cache:
            self._observe_cache(
                pending,
                snapshot,
                layer_index,
                h=h,
                y=y,
                gate=gate,
                shift=shift,
                scale=scale,
                selector=selector,
                chunk_start=chunk_start,
                chunk_stop=chunk_stop,
                evaluate_activation=evaluate_activation,
                apply_fc2=apply_fc2,
            )

    def _pending(self, snapshot, layer_index, route, mlp_path):
        key = (int(snapshot.step_index), int(layer_index))
        record = self.pending.get(key)
        if record is None:
            record = {
                'request_id': int(snapshot.request_id),
                'step_index': int(snapshot.step_index),
                'total_steps': int(snapshot.total_steps),
                'layer': int(layer_index),
                'sigma': self._sigma(snapshot.step_index),
                'kv_tile': int(route.kv_tile),
                'video_kv_start': int(route.video_kv_start),
                'video_kv_tiles': int(route.video_kv_tiles),
                'route_dense': bool(route.dense),
                'route_draws': int(route.draws),
                'mlp_path': mlp_path,
                'chunks': 0,
                'oracle': torch.zeros(
                    int(route.video_kv_tiles),
                    dtype=torch.float64,
                    device=route.counts.device
                    if route.counts is not None
                    else None,
                ),
            }
            self.pending[key] = record
        return record

    def _observe_selection(self, pending, route, y, gate, selector, chunk_start):
        pending['chunks'] += 1
        energy = metrics.row_update_energy(y, gate, selector)
        destination = pending['oracle']
        if destination.device != energy.device:
            destination = destination.to(energy.device)
            pending['oracle'] = destination
        metrics.accumulate_tile_energy(
            destination,
            energy,
            chunk_start,
            route.kv_tile,
            route.video_kv_start,
            route.video_kv_tiles,
        )

    # -- Tier B/C ----------------------------------------------------------

    def _observe_cache(
        self,
        pending,
        snapshot,
        layer_index,
        *,
        h,
        y,
        gate,
        shift,
        scale,
        selector,
        chunk_start,
        chunk_stop,
        evaluate_activation,
        apply_fc2,
    ):
        step = int(snapshot.step_index)
        if step < int(self.config.start_step):
            return
        if evaluate_activation is None or shift is None or scale is None:
            self._note(
                'the MLP path did not expose a SwiGLU activation seam; cache '
                'measurements were skipped'
            )
            return
        blocks = sample_blocks(
            snapshot.layout,
            pending['kv_tile'],
            self.config.sample_blocks,
        )
        inside = [
            start for start in blocks
            if start >= int(chunk_start)
            and start + STAGE0_SAMPLE_ROWS <= int(chunk_stop)
        ]
        if not inside:
            return
        local = torch.cat([
            torch.arange(
                start - int(chunk_start),
                start - int(chunk_start) + STAGE0_SAMPLE_ROWS,
                dtype=torch.int64,
                device=h.device,
            )
            for start in inside
        ])
        rows = tuple(inside)

        h_rows = h.index_select(0, local)
        y_rows = y.index_select(0, local)
        gate_rows = _mod_rows(gate, selector, local)
        shift_rows = _mod_rows(shift, selector, local)
        scale_rows = _mod_rows(scale, selector, local)
        activation = evaluate_activation(h_rows)
        normalized = _unmodulate(h_rows, shift_rows, scale_rows)

        previous = self.cached.get((int(layer_index), rows))
        measured = None
        if previous is not None and previous['step'] < step:
            measured = self._measure_cache(
                previous,
                step=step,
                layer_index=layer_index,
                pending=pending,
                blocks=rows,
                h_rows=h_rows,
                y_rows=y_rows,
                gate_rows=gate_rows,
                shift_rows=shift_rows,
                scale_rows=scale_rows,
                activation=activation,
                normalized=normalized,
                evaluate_activation=evaluate_activation,
                apply_fc2=apply_fc2,
            )
        if measured is not None:
            self.cache_records.append(measured)

        if not (step - int(self.config.start_step)) % int(
            self.config.cache_step_stride
        ):
            self.cached[(int(layer_index), rows)] = {
                'step': step,
                # Held exactly: BF16 and FP8 are what we are measuring, so
                # they must not also be the floor of the measurement.
                'activation': activation.detach().to('cpu', torch.float32),
                'output': y_rows.detach().to('cpu', torch.float32),
                'normalized': normalized.detach().to('cpu', torch.float32),
                'shift': shift_rows.detach().to('cpu', torch.float32),
                'scale': scale_rows.detach().to('cpu', torch.float32),
            }

    def _measure_cache(
        self,
        previous,
        *,
        step,
        layer_index,
        pending,
        blocks,
        h_rows,
        y_rows,
        gate_rows,
        shift_rows,
        scale_rows,
        activation,
        normalized,
        evaluate_activation,
        apply_fc2,
    ):
        device = h_rows.device
        dtype = h_rows.dtype
        previous_activation = previous['activation'].to(device)
        previous_output = previous['output'].to(device)
        previous_normalized = previous['normalized'].to(device)
        previous_shift = previous['shift'].to(device)
        previous_scale = previous['scale'].to(device)

        delta_total = activation.float() - previous_activation.float()
        group_energy = metrics.group_magnitudes(delta_total)

        # Counterfactual inputs: new modulation on old state, and the reverse.
        modulation_only = evaluate_activation(
            _modulate(previous_normalized, shift_rows, scale_rows, dtype)
        )
        state_only = evaluate_activation(
            _modulate(normalized, previous_shift, previous_scale, dtype)
        )
        decomposition = metrics.delta_decomposition(
            delta_total=delta_total,
            delta_modulation=(
                modulation_only.float() - previous_activation.float()
            ),
            delta_state=state_only.float() - previous_activation.float(),
        )

        record = {
            'step_index': step,
            'previous_step': int(previous['step']),
            'layer': int(layer_index),
            'sigma': self._sigma(step),
            'sampled_rows': int(h_rows.shape[0]),
            'sampled_block_starts': list(blocks),
            'mlp_path': pending['mlp_path'],
            'group_concentration': metrics.group_concentration(group_energy),
            'pairwise_group_jaccard': {
                str(k): metrics.pairwise_group_jaccard(
                    group_energy,
                    k,
                    seed=int(self.config.selector_seed) + step * 31 + layer_index,
                )
                for k in metrics.GROUP_BUDGETS
            },
            'delta_decomposition': decomposition,
            'cache_representations': metrics.cache_representation_report(
                y_cached=previous_output,
                gate_rows=gate_rows,
                a_current=activation.float(),
                a_cached=previous_activation,
                apply_fc2=apply_fc2,
            ),
        }
        return record

    # -- block and request completion --------------------------------------

    def end_mlp_block(self, layer_index, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            return
        key = (int(snapshot.step_index), int(layer_index))
        record = self.pending.pop(key, None)
        if record is None:
            return
        recorder = get_route_recorder(transformer_options)
        route = None if recorder is None else recorder.route(layer_index)
        oracle = record.pop('oracle').to(torch.float32)
        record['oracle_total_energy'] = float(oracle.sum())
        if route is not None and not route.dense:
            scores = metrics.selector_scores(
                hit_rate=route.hit_rate(),
                attention_update_energy=route.attention_update_energy(),
                oracle=oracle,
                seed=(
                    int(self.config.selector_seed)
                    + (int(snapshot.step_index) + 1) * 10_007
                    + (int(layer_index) + 1) * 101
                ),
            )
            record['selectors'] = metrics.selector_capture_table(scores, oracle)
            record['available_signals'] = sorted(scores)
        else:
            # Conservative policy: a dense attention route means a dense MLP.
            record['selectors'] = []
            record['available_signals'] = []
        self.selection_records.append(record)

    def _output_directory(self):
        try:
            import folder_paths
            root = folder_paths.get_output_directory()
        except ImportError:
            root = os.path.abspath('output')
        stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        name = '%s_%s_request%d' % (
            self.config.run_tag,
            stamp,
            self.request_serial,
        )
        return os.path.join(root, 'h3_mlp_stage0', name)

    def _write(self, directory, name, records):
        with open(
            os.path.join(directory, name),
            'w',
            encoding='utf-8',
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + '\n')

    def end_request(self, error=None):
        seconds = None if self.started is None else time.perf_counter() - self.started
        self.started = None
        if self.pending and error is None:
            error = Stage0Error('request ended with unfinished Stage 0 blocks')
        directory = self._output_directory()
        os.makedirs(directory, exist_ok=False)
        self._write(directory, 'selection.jsonl', self.selection_records)
        self._write(directory, 'cache.jsonl', self.cache_records)
        summary = {
            'output_exact': True,
            'completed': error is None,
            'error': None if error is None else '%s: %s' % (
                type(error).__name__,
                error,
            ),
            'seconds': seconds,
            'request_serial': int(self.request_serial),
            'layers': list(self.config.layers),
            'measure_cache': bool(self.config.measure_cache),
            'sample_blocks': int(self.config.sample_blocks),
            'sample_rows': int(self.config.sample_rows),
            'cache_step_stride': int(self.config.cache_step_stride),
            'start_step': int(self.config.start_step),
            'selector_seed': int(self.config.selector_seed),
            'schedule': list(self.schedule),
            'selection_records': len(self.selection_records),
            'cache_records': len(self.cache_records),
            'notes': list(self.notes),
            'pooled': pool_summary(self.selection_records, self.cache_records),
        }
        with open(
            os.path.join(directory, 'summary.json'),
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        self.last_report_directory = directory
        self.pending.clear()
        self.cached.clear()
        logging.info('%s wrote %s', LOG_PREFIX, directory)


def _mean(values):
    values = [value for value in values if value == value]
    return sum(values) / len(values) if values else None


def pool_summary(selection_records, cache_records):
    '''Pool per-layer/step statistics into the headline Stage 0 answers.'''
    selectors = {}
    for record in selection_records:
        for row in record.get('selectors', ()):
            key = (row['selector'], row['tile_budget'])
            bucket = selectors.setdefault(key, {'captured': [], 'overlap': []})
            bucket['captured'].append(row['oracle_mass_captured'])
            bucket['overlap'].append(row['oracle_topk_overlap'])
    selection = [
        {
            'selector': name,
            'tile_budget': budget,
            'oracle_mass_captured': _mean(bucket['captured']),
            'oracle_topk_overlap': _mean(bucket['overlap']),
            'samples': len(bucket['captured']),
        }
        for (name, budget), bucket in sorted(selectors.items())
    ]

    concentration = {}
    for record in cache_records:
        for row in record.get('group_concentration', ()):
            key = (row['granularity'], row['groups'])
            bucket = concentration.setdefault(key, {'captured': [], 'ratio': []})
            bucket['captured'].append(row['delta_mass_captured'])
            bucket['ratio'].append(row['fraction_of_token_oracle'])
    groups = [
        {
            'granularity': granularity,
            'groups': count,
            'delta_mass_captured': _mean(bucket['captured']),
            'fraction_of_token_oracle': _mean(bucket['ratio']),
            'samples': len(bucket['captured']),
        }
        for (granularity, count), bucket in sorted(concentration.items())
    ]

    decomposition = {
        field: _mean([
            record['delta_decomposition'].get(field, float('nan'))
            for record in cache_records
            if 'delta_decomposition' in record
        ])
        for field in (
            'modulation_only_fraction',
            'state_only_fraction',
            'modulation_cosine',
            'state_cosine',
            'residual_after_modulation_fraction',
        )
    }

    cache = {}
    for dtype in metrics.CACHE_DTYPES:
        present = [
            record['cache_representations'][dtype]
            for record in cache_records
            if dtype in record.get('cache_representations', {})
        ]
        entry = {
            'output_relative_error': _mean([
                item['output'].get('mean', float('nan')) for item in present
            ]),
            'activation_delta_relative_error': _mean([
                item['activation']['delta_relative_error'].get(
                    'mean', float('nan')
                )
                for item in present
            ]),
            'fc2_relative_error': _mean([
                item['activation']['fc2_relative_error'].get(
                    'mean', float('nan')
                )
                for item in present
                if 'fc2_relative_error' in item['activation']
            ]),
        }
        for k in metrics.GROUP_BUDGETS:
            field = 'group_rank_jaccard_top%d' % k
            entry[field] = _mean([
                item['activation'].get(field, float('nan'))
                for item in present
            ])
        cache[dtype] = entry
    token_pairs = {
        'group_rank_jaccard_top%d' % k: _mean([
            record['pairwise_group_jaccard'].get(str(k), float('nan'))
            for record in cache_records
            if 'pairwise_group_jaccard' in record
        ])
        for k in metrics.GROUP_BUDGETS
    }

    return {
        'selection': selection,
        'group_sharing': groups,
        'delta_decomposition': decomposition,
        'cache': cache,
        'token_pair_group_overlap': token_pairs,
    }


def _make_request_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        sigmas = args[3] if len(args) > 3 else kwargs.get('sigmas')
        session.begin_request(sigmas)
        error = None
        try:
            return executor(*args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            session.end_request(error)

    return wrapper


def install_stage0(model_patcher, config):
    if not isinstance(config, Stage0Config):
        raise TypeError('config must be Stage0Config')
    if not is_minimax_h3(model_patcher):
        raise Stage0Error('Stage 0 only supports MiniMax H3')
    blocks = get_h3_blocks(model_patcher)
    existing = getattr(model_patcher, 'object_patches', {})
    missing = [
        key_for(index)
        for index in range(len(blocks))
        if not getattr(existing.get(key_for(index)), OWNER_MARKER, False)
    ]
    if missing:
        raise Stage0Error(
            'Stage 0 requires H3 Memory Optimization; %s is not owned by the '
            'exact chunked MLP path' % missing[0]
        )

    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    if options.get(SHARING_KEY) is not None:
        raise Stage0Error('Stage 0 cannot run beside executable MLP sharing')
    current = options.get(OBSERVER_KEY)
    if current is not None:
        if isinstance(current, Stage0Session) and current.config == config:
            return current
        raise Stage0Error('a different MLP observer is already installed')

    runtime = options.get(RUNTIME_SESSION_KEY)
    if runtime is None:
        runtime = H3RuntimeSession(strict_layout=True)
        install_runtime_wrapper(model_patcher, runtime)
        options = model_patcher.model_options['transformer_options']
    elif not isinstance(runtime, H3RuntimeSession):
        raise Stage0Error('foreign H3 runtime session is already installed')
    else:
        runtime.strict_layout = True

    session = Stage0Session(config)
    options[OBSERVER_KEY] = session
    options[ROUTE_KEY] = AttentionRouteRecorder()
    options[STAGE0_STATUS_KEY] = {
        'output_exact': True,
        'layers': list(config.layers),
        'measure_cache': bool(config.measure_cache),
        'sample_blocks': int(config.sample_blocks),
        'sample_rows': int(config.sample_rows),
        'cache_step_stride': int(config.cache_step_stride),
        'start_step': int(config.start_step),
    }

    import comfy.patcher_extension
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        STAGE0_WRAPPER_KEY,
        _make_request_wrapper(session),
        model_patcher.model_options,
        is_model_options=True,
    )
    logging.info(
        '%s armed: layers=%s cache=%s sample_rows=%d',
        LOG_PREFIX,
        ','.join(str(layer) for layer in config.layers),
        'on' if config.measure_cache else 'off',
        config.sample_rows,
    )
    return session
