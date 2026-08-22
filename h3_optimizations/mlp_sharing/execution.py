'''Executable target-video MLP output sharing and request reports.'''

from __future__ import annotations

import json
import logging
import os
import threading
import time

import torch
import torch.nn.functional as F

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
from .config import MLPSharingConfig, removal_option
from .metrics import (
    PAIR_COMPLEMENT,
    PAIR_LEFT,
    PAIR_RIGHT,
    filter_cells_by_modulation,
    spatial_cells_1x2x2,
    spatial_cells_1x2x4,
)

MLP_SHARING_STATUS_KEY = 'h3_optimizations_mlp_sharing'
MLP_SHARING_WRAPPER_KEY = 'h3_optimizations_mlp_sharing_request'
LOG_PREFIX = '[H3 MLP sharing]'
_ACTIVE = threading.local()


class MLPSharingError(RuntimeError):
    pass


def _hash_slots(count, modulus, seed, device):
    values = torch.arange(count, dtype=torch.int64, device=device)
    values.mul_(1_103_515_245).add_(int(seed) & 0x7FFFFFFF)
    values.bitwise_and_(0x7FFFFFFF)
    return values.remainder_(int(modulus))


def _cosine_edges(h_cells):
    left = torch.tensor(PAIR_LEFT, dtype=torch.long, device=h_cells.device)
    right = torch.tensor(PAIR_RIGHT, dtype=torch.long, device=h_cells.device)
    return 1.0 - F.cosine_similarity(
        h_cells.index_select(1, left).float(),
        h_cells.index_select(1, right).float(),
        dim=-1,
        eps=1.0e-12,
    )


def _medoid_slots(h_cells):
    normalized = F.normalize(h_cells.float(), dim=-1, eps=1.0e-12)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    return similarity.sum(dim=-1).argmax(dim=-1)


def _pair_rows(cell_indices, edge_slots, include_complement):
    left = torch.tensor(PAIR_LEFT, dtype=torch.long, device=cell_indices.device)
    right = torch.tensor(PAIR_RIGHT, dtype=torch.long, device=cell_indices.device)
    representative_slots = left.index_select(0, edge_slots)
    removed_slots = right.index_select(0, edge_slots)
    representatives = cell_indices.gather(1, representative_slots[:, None]).reshape(-1)
    removed = cell_indices.gather(1, removed_slots[:, None]).reshape(-1)
    if not include_complement:
        return representatives, removed

    complement = torch.tensor(
        PAIR_COMPLEMENT.tolist(),
        dtype=torch.long,
        device=cell_indices.device,
    ).index_select(0, edge_slots)
    complement_representatives = left.index_select(0, complement)
    complement_removed = right.index_select(0, complement)
    representatives = torch.stack((
        representatives,
        cell_indices.gather(1, complement_representatives[:, None]).reshape(-1),
    ), dim=1).reshape(-1)
    removed = torch.stack((
        removed,
        cell_indices.gather(1, complement_removed[:, None]).reshape(-1),
    ), dim=1).reshape(-1)
    return representatives, removed


def _group_rows(cell_indices, representative_slots):
    group_size = int(cell_indices.shape[1])
    representatives = cell_indices.gather(
        1,
        representative_slots[:, None],
    )
    slots = torch.arange(group_size, device=cell_indices.device)[None, :]
    keep = slots != representative_slots[:, None]
    removed = cell_indices[keep].reshape(-1)
    sources = representatives.expand(-1, group_size - 1).reshape(-1)
    return sources, removed


def _selection_rows(h, cell_indices, config, seed):
    fraction = float(config.removal_fraction)
    cells = int(cell_indices.shape[0])
    group_size = int(cell_indices.shape[1])
    h_cells = h.index_select(0, cell_indices.reshape(-1)).reshape(
        cells,
        group_size,
        h.shape[-1],
    )

    if fraction in (0.25, 0.5):
        if config.selector == 'input_cosine':
            edge_slots = _cosine_edges(h_cells).argmin(dim=1)
        else:
            edge_slots = _hash_slots(cells, len(PAIR_LEFT), seed, h.device)
        return _pair_rows(cell_indices, edge_slots, fraction == 0.5)

    if config.selector == 'input_cosine':
        representative_slots = _medoid_slots(h_cells)
    else:
        representative_slots = _hash_slots(cells, group_size, seed, h.device)
    return _group_rows(cell_indices, representative_slots)


def _video_rows(layout, chunk_start, chunk_stop):
    video_start, video_stop = (int(value) for value in layout.video_range)
    return max(0, min(int(chunk_stop), video_stop) - max(int(chunk_start), video_start))


class MLPSharingSession:
    def __init__(self, config):
        self.config = config
        self.request_serial = -1
        self.schedule = ()
        self.records = []
        self.pending = {}
        self.evaluations = {}
        self.started = None
        self.last_report_directory = None

    def begin_request(self, sigmas):
        if self.started is not None:
            raise MLPSharingError('concurrent requests share one MLP sharing session')
        self.request_serial += 1
        self.records.clear()
        self.pending.clear()
        self.evaluations.clear()
        self.started = time.perf_counter()
        if torch.is_tensor(sigmas):
            self.schedule = tuple(
                float(value)
                for value in sigmas.detach().flatten().float().cpu().tolist()
            )
        else:
            self.schedule = ()
        _ACTIVE.session = self

    def _sigma(self, step_index):
        index = int(step_index)
        if 0 <= index < len(self.schedule):
            return self.schedule[index]
        return None

    def _pending_record(self, snapshot, layer_index):
        key = (int(snapshot.step_index), int(layer_index))
        record = self.pending.get(key)
        if record is None:
            record = {
                'request_id': int(snapshot.request_id),
                'step_index': int(snapshot.step_index),
                'total_steps': int(snapshot.total_steps),
                'layer': int(layer_index),
                'chunks': 0,
                'total_mlp_rows': 0,
                'target_video_rows': 0,
                'eligible_video_rows': 0,
                'evaluated_mlp_rows': 0,
                'evaluated_video_rows': 0,
                'removed_video_rows': 0,
                'active': False,
            }
            self.pending[key] = record
        return record

    def evaluate_chunk(
        self,
        layer_index,
        transformer_options,
        *,
        h,
        selector,
        chunk_start,
        chunk_stop,
        evaluate_mlp,
    ):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise MLPSharingError(
                'executable MLP sharing requires sampler-step and packed-layout runtime'
            )
        record = self._pending_record(snapshot, layer_index)
        target_rows = _video_rows(snapshot.layout, chunk_start, chunk_stop)
        record['chunks'] += 1
        record['total_mlp_rows'] += int(h.shape[0])
        record['target_video_rows'] += target_rows

        active = (
            int(layer_index) in self.config.layers
            and int(snapshot.step_index) >= int(self.config.start_after_step)
            and float(self.config.removal_fraction) > 0.0
        )
        if not active or not target_rows:
            out, expanded, path = evaluate_mlp(h)
            record['evaluated_mlp_rows'] += int(h.shape[0])
            record['evaluated_video_rows'] += target_rows
            return out, expanded, path

        cells, _total_video_tokens = (
            spatial_cells_1x2x4(snapshot.layout, chunk_start, chunk_stop)
            if self.config.geometry == (1, 2, 4)
            else spatial_cells_1x2x2(snapshot.layout, chunk_start, chunk_stop)
        )
        cell_indices = filter_cells_by_modulation(cells, selector, h.device)
        eligible_rows = int(cell_indices.numel())
        record['eligible_video_rows'] += eligible_rows
        record['active'] = True
        if not eligible_rows:
            out, expanded, path = evaluate_mlp(h)
            record['evaluated_mlp_rows'] += int(h.shape[0])
            record['evaluated_video_rows'] += target_rows
            return out, expanded, path

        evaluation_index = self.evaluations.get(
            (int(snapshot.step_index), int(layer_index)),
            0,
        )
        seed = (
            int(self.config.selector_seed)
            + (int(snapshot.step_index) + 1) * 10_007
            + (int(layer_index) + 1) * 101
            + int(evaluation_index) * 17
            + int(chunk_start)
        )
        representatives, removed = _selection_rows(
            h,
            cell_indices,
            self.config,
            seed,
        )
        source_by_row = torch.arange(h.shape[0], dtype=torch.long, device=h.device)
        source_by_row.index_copy_(0, removed, representatives)
        evaluated_rows, inverse = torch.unique(
            source_by_row,
            sorted=True,
            return_inverse=True,
        )
        reduced_h = h.index_select(0, evaluated_rows)
        reduced_out, expanded, path = evaluate_mlp(reduced_h)
        out = reduced_out.index_select(0, inverse)
        removed_rows = int(removed.numel())
        record['evaluated_mlp_rows'] += int(evaluated_rows.numel())
        record['evaluated_video_rows'] += target_rows - removed_rows
        record['removed_video_rows'] += removed_rows
        return out, expanded, path

    def end_mlp_block(self, layer_index, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            return
        key = (int(snapshot.step_index), int(layer_index))
        record = self.pending.pop(key, None)
        if record is None:
            return
        evaluation_index = self.evaluations.get(key, 0)
        self.evaluations[key] = evaluation_index + 1
        record['evaluation_index'] = evaluation_index
        record['sigma'] = self._sigma(record['step_index'])
        target = int(record['target_video_rows'])
        eligible = int(record['eligible_video_rows'])
        removed = int(record['removed_video_rows'])
        total = int(record['total_mlp_rows'])
        record['requested_removal_fraction'] = float(self.config.removal_fraction)
        record['realized_target_video_removal_fraction'] = (
            0.0 if not target else float(removed) / float(target)
        )
        record['eligible_target_video_fraction'] = (
            0.0 if not target else float(eligible) / float(target)
        )
        record['whole_mlp_row_removal_fraction'] = (
            0.0 if not total else float(removed) / float(total)
        )
        record['selector'] = self.config.selector
        record['geometry'] = list(self.config.geometry)
        self.records.append(record)

    def callback_metadata(self):
        if not self.records:
            return {
                'h3_vector_mlp_sharing': True,
                'h3_vector_mlp_requested_removal': float(self.config.removal_fraction),
                'h3_vector_mlp_start_after_step': int(self.config.start_after_step),
            }
        step = max(int(record['step_index']) for record in self.records)
        rows = [record for record in self.records if int(record['step_index']) == step]
        target = sum(int(record['target_video_rows']) for record in rows)
        removed = sum(int(record['removed_video_rows']) for record in rows)
        eligible = sum(int(record['eligible_video_rows']) for record in rows)
        return {
            'h3_vector_mlp_sharing': True,
            'h3_vector_mlp_selector': self.config.selector,
            'h3_vector_mlp_requested_removal': float(self.config.removal_fraction),
            'h3_vector_mlp_actual_removal': 0.0 if not target else removed / target,
            'h3_vector_mlp_eligible_fraction': 0.0 if not target else eligible / target,
            'h3_vector_mlp_start_after_step': int(self.config.start_after_step),
            'h3_vector_mlp_step': step,
            'h3_vector_mlp_active': bool(any(record['active'] for record in rows)),
        }

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
        return os.path.join(root, 'h3_mlp_sharing', name)

    def end_request(self, error=None):
        seconds = None if self.started is None else time.perf_counter() - self.started
        self.started = None
        if self.pending and error is None:
            error = MLPSharingError('request ended with unfinished MLP sharing blocks')
        directory = self._output_directory()
        os.makedirs(directory, exist_ok=False)
        with open(
            os.path.join(directory, 'layer_step_stats.jsonl'),
            'w',
            encoding='utf-8',
        ) as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + '\n')
        summary = {
            'output_exact': float(self.config.removal_fraction) == 0.0,
            'completed': error is None,
            'error': None if error is None else '%s: %s' % (type(error).__name__, error),
            'seconds': seconds,
            'request_serial': int(self.request_serial),
            'selector': self.config.selector,
            'selector_seed': int(self.config.selector_seed),
            'requested_removal_fraction': float(self.config.removal_fraction),
            'removal_option': removal_option(self.config.removal_fraction),
            'start_after_step': int(self.config.start_after_step),
            'protected_steps': list(range(int(self.config.start_after_step))),
            'layers': list(self.config.layers),
            'geometry': list(self.config.geometry),
            'reconstruction': 'representative',
            'schedule': list(self.schedule),
            'records': len(self.records),
        }
        with open(
            os.path.join(directory, 'summary.json'),
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        self.last_report_directory = directory
        self.pending.clear()
        if getattr(_ACTIVE, 'session', None) is self:
            _ACTIVE.session = None
        logging.info('%s wrote %s', LOG_PREFIX, directory)


def current_callback_metadata():
    session = getattr(_ACTIVE, 'session', None)
    if isinstance(session, MLPSharingSession):
        return session.callback_metadata()
    return {}


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


def install_sharing(model_patcher, config):
    if not isinstance(config, MLPSharingConfig):
        raise TypeError('config must be MLPSharingConfig')
    if not is_minimax_h3(model_patcher):
        raise MLPSharingError('MLP sharing only supports MiniMax H3')
    blocks = get_h3_blocks(model_patcher)
    existing = getattr(model_patcher, 'object_patches', {})
    missing = [
        key_for(index)
        for index in range(len(blocks))
        if not getattr(existing.get(key_for(index)), OWNER_MARKER, False)
    ]
    if missing:
        raise MLPSharingError(
            'MLP sharing requires H3 Memory Optimization; %s is not owned by '
            'the exact chunked MLP path' % missing[0]
        )

    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    if options.get(OBSERVER_KEY) is not None:
        raise MLPSharingError('the output-exact MLP probe and executable sharing cannot coexist')
    current = options.get(SHARING_KEY)
    if current is not None:
        if isinstance(current, MLPSharingSession) and current.config == config:
            return current
        raise MLPSharingError('a different executable MLP sharing request is already installed')

    runtime = options.get(RUNTIME_SESSION_KEY)
    if runtime is None:
        runtime = H3RuntimeSession(strict_layout=True)
        install_runtime_wrapper(model_patcher, runtime)
        options = model_patcher.model_options['transformer_options']
    elif not isinstance(runtime, H3RuntimeSession):
        raise MLPSharingError('foreign H3 runtime session is already installed')
    else:
        runtime.strict_layout = True

    session = MLPSharingSession(config)
    options[SHARING_KEY] = session
    options[MLP_SHARING_STATUS_KEY] = {
        'output_exact': float(config.removal_fraction) == 0.0,
        'selector': config.selector,
        'selector_seed': int(config.selector_seed),
        'requested_removal_fraction': float(config.removal_fraction),
        'removal_option': removal_option(config.removal_fraction),
        'start_after_step': int(config.start_after_step),
        'layers': list(config.layers),
        'geometry': list(config.geometry),
        'reconstruction': 'representative',
    }

    import comfy.patcher_extension
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        MLP_SHARING_WRAPPER_KEY,
        _make_request_wrapper(session),
        model_patcher.model_options,
        is_model_options=True,
    )
    logging.info(
        '%s armed: selector=%s removal=%s start_after_step=%d geometry=%s',
        LOG_PREFIX,
        config.selector,
        removal_option(config.removal_fraction),
        config.start_after_step,
        'x'.join(str(value) for value in config.geometry),
    )
    return session
