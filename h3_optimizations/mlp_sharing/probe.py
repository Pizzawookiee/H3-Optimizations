'''Runtime installation and report lifecycle for the MLP sharing oracle.'''

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
from .config import MLPSharingProbeConfig, SELECTORS, TARGET_FRACTIONS
from .metrics import (
    VALUE_COLUMNS,
    measure_chunk,
    spatial_cells_1x2x2,
    summarize_measurements,
)

PROBE_STATUS_KEY = 'h3_optimizations_mlp_sharing_probe'
PROBE_WRAPPER_KEY = 'h3_optimizations_mlp_sharing_probe_request'
LOG_PREFIX = '[H3 MLP sharing probe]'


class MLPSharingProbeError(RuntimeError):
    pass


class MLPSharingProbeSession:
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
            raise MLPSharingProbeError('concurrent requests share one probe session')
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

    def _sigma(self, step_index):
        index = int(step_index)
        if index < 0 or index >= len(self.schedule):
            return None
        return self.schedule[index]

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
        **unused,
    ):
        del unused          # shift/scale and the Stage 0 activation seams
        if int(layer_index) not in self.config.layers:
            return
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise MLPSharingProbeError(
                'the probe requires the H3 sampler-step and packed-layout runtime'
            )
        cells, total_video_tokens = spatial_cells_1x2x2(
            snapshot.layout,
            chunk_start,
            chunk_stop,
        )
        measured = measure_chunk(
            h=h,
            y=y,
            residual=residual,
            gate=gate,
            selector=selector,
            cells=cells,
            evaluate_mlp=evaluate_mlp,
            include_mean_input=self.config.include_mean_input,
            mean_batch_rows=self.config.mean_batch_rows,
        )
        if measured is None:
            return
        key = (int(snapshot.step_index), int(layer_index))
        pending = self.pending.get(key)
        if pending is None:
            pending = {
                'request_id': int(snapshot.request_id),
                'step_index': int(snapshot.step_index),
                'total_steps': int(snapshot.total_steps),
                'layer_index': int(layer_index),
                'total_video_tokens': int(total_video_tokens),
                'chunks': [],
            }
            self.pending[key] = pending
        elif int(pending['total_video_tokens']) != int(total_video_tokens):
            raise MLPSharingProbeError('target-video layout changed inside one block')
        pending['chunks'].append(measured)

    def end_mlp_block(self, layer_index, transformer_options):
        if int(layer_index) not in self.config.layers:
            return
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            return
        key = (int(snapshot.step_index), int(layer_index))
        pending = self.pending.pop(key, None)
        if pending is None:
            return
        values = torch.cat(pending.pop('chunks'), dim=0).detach().float().cpu().numpy()
        evaluation_index = self.evaluations.get(key, 0)
        self.evaluations[key] = evaluation_index + 1
        self.records.extend(
            summarize_measurements(
                values,
                request_id=pending['request_id'],
                step_index=pending['step_index'],
                total_steps=pending['total_steps'],
                sigma=self._sigma(pending['step_index']),
                layer_index=pending['layer_index'],
                evaluation_index=evaluation_index,
                total_video_tokens=pending['total_video_tokens'],
            )
        )

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
        return os.path.join(root, 'h3_mlp_sharing_probe', name)

    def end_request(self, error=None):
        seconds = None
        if self.started is not None:
            seconds = time.perf_counter() - self.started
        self.started = None
        if self.pending and error is None:
            error = MLPSharingProbeError(
                'request ended with unfinished MLP probe blocks'
            )
        directory = self._output_directory()
        os.makedirs(directory, exist_ok=False)
        with open(
            os.path.join(directory, 'layer_step_metrics.jsonl'),
            'w',
            encoding='utf-8',
        ) as handle:
            for row in self.records:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
        summary = {
            'output_exact': True,
            'completed': error is None,
            'error': None if error is None else '%s: %s' % (type(error).__name__, error),
            'seconds': seconds,
            'request_serial': int(self.request_serial),
            'layers': list(self.config.layers),
            'geometry': [1, 2, 2],
            'selectors': list(SELECTORS),
            'target_merge_fractions': list(TARGET_FRACTIONS),
            'include_mean_input': bool(self.config.include_mean_input),
            'mean_batch_rows': int(self.config.mean_batch_rows),
            'schedule': list(self.schedule),
            'records': len(self.records),
            'value_columns': list(VALUE_COLUMNS),
        }
        with open(
            os.path.join(directory, 'summary.json'),
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with open(
            os.path.join(directory, 'report.txt'),
            'w',
            encoding='utf-8',
        ) as handle:
            handle.write('H3 MLP output-sharing Stage 1 oracle\n')
            handle.write('Inference output changed: no\n')
            handle.write('Geometry: target-video 1T x 2Y x 2X\n')
            handle.write('Layer-step metric rows: %d\n' % len(self.records))
            if error is not None:
                handle.write('Run error: %s: %s\n' % (type(error).__name__, error))
        self.last_report_directory = directory
        self.pending.clear()
        logging.info('%s wrote %s', LOG_PREFIX, directory)


def _make_request_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        sigmas = args[3] if len(args) > 3 else kwargs.get('sigmas')
        session.begin_request(sigmas)
        error = None
        try:
            return executor(*args, **kwargs)
        except Exception as exc:
            error = exc
            raise
        finally:
            session.end_request(error)

    return wrapper


def install_probe(model_patcher, config):
    if not isinstance(config, MLPSharingProbeConfig):
        raise TypeError('config must be MLPSharingProbeConfig')
    if not is_minimax_h3(model_patcher):
        raise MLPSharingProbeError('the MLP sharing probe only supports MiniMax H3')
    blocks = get_h3_blocks(model_patcher)
    existing = getattr(model_patcher, 'object_patches', {})
    missing = [
        key_for(index)
        for index in range(len(blocks))
        if not getattr(existing.get(key_for(index)), OWNER_MARKER, False)
    ]
    if missing:
        raise MLPSharingProbeError(
            'place H3 MLP Sharing Probe after H3 Memory Optimization; '
            '%s is not owned by its exact chunked MLP path' % missing[0]
        )

    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    if options.get(SHARING_KEY) is not None:
        raise MLPSharingProbeError(
            'the output-exact MLP probe and executable sharing cannot coexist'
        )
    current = options.get(OBSERVER_KEY)
    if current is not None:
        if isinstance(current, MLPSharingProbeSession) and current.config == config:
            return current
        raise MLPSharingProbeError('another H3 MLP observer is already installed')

    runtime = options.get(RUNTIME_SESSION_KEY)
    if runtime is None:
        runtime = H3RuntimeSession(strict_layout=True)
        install_runtime_wrapper(model_patcher, runtime)
        options = model_patcher.model_options['transformer_options']
    elif not isinstance(runtime, H3RuntimeSession):
        raise MLPSharingProbeError('foreign H3 runtime session is already installed')
    else:
        runtime.strict_layout = True

    session = MLPSharingProbeSession(config)
    options[OBSERVER_KEY] = session
    options[PROBE_STATUS_KEY] = {
        'output_exact': True,
        'layers': list(config.layers),
        'geometry': [1, 2, 2],
        'selectors': list(SELECTORS),
        'include_mean_input': bool(config.include_mean_input),
    }

    import comfy.patcher_extension
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        PROBE_WRAPPER_KEY,
        _make_request_wrapper(session),
        model_patcher.model_options,
        is_model_options=True,
    )
    logging.info(
        '%s armed: layers=%s geometry=1x2x2 mean_input=%s',
        LOG_PREFIX,
        ','.join(str(layer) for layer in config.layers),
        config.include_mean_input,
    )
    return session
