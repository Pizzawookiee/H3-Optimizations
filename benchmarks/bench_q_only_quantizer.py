'''Prove Q-only native quantization matches the coupled production quantizer.

This is an enabling-contract test, not a speed comparison: the production
oracle also quantizes K and V, so timing the two against each other would be an
unequal boundary.  Both rotation regimes and ragged Q tile tails are required.
'''

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import sys


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
for _root in (str(COMFY_ROOT), str(PACK_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)


def parse_cases(value):
    cases = []
    for item in str(value).split(','):
        try:
            q_rows, k_rows = (int(part) for part in item.strip().split(':', 1))
        except ValueError as error:
            raise argparse.ArgumentTypeError('cases must use Q_ROWS:FULL_K_ROWS') from error
        if q_rows <= 0 or k_rows <= 0:
            raise argparse.ArgumentTypeError('case lengths must be positive')
        cases.append((q_rows, k_rows))
    if not cases:
        raise argparse.ArgumentTypeError('at least one case is required')
    return tuple(cases)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Byte-compare the experimental Q-only quantizer with production Q/K.'
    )
    parser.add_argument('--library', default=str(PACK_ROOT / 'native' / 'bin' / 'h3_q_only.dll'))
    parser.add_argument('--cases', type=parse_cases, default=parse_cases('129:256,333:4096'))
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--heads', type=int, default=56)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if min(args.batch, args.heads, args.iterations) <= 0:
        parser.error('batch, heads, and iterations must be positive')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def _load(path):
    library_path = Path(path).resolve()
    if not library_path.is_file():
        raise SystemExit(
            'Q-only side-car is missing: %s\nbuild it with native\\build_q_only.ps1'
            % library_path
        )
    library = ctypes.CDLL(str(library_path))
    library.h3_q_only_abi_version.restype = ctypes.c_int
    library.h3_q_only_last_error.restype = ctypes.c_char_p
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    library.h3_int8_quantize_q_only.restype = integer
    library.h3_int8_quantize_q_only.argtypes = (
        [pointer, pointer, pointer] + [integer] * 5
        + [ctypes.c_int64] * 3 + [integer, ctypes.c_size_t]
    )
    if library.h3_q_only_abi_version() != 1:
        raise SystemExit('Q-only side-car ABI is not 1')
    return library, library_path


def _quantize_q_only(torch, library, q, full_k_rows):
    batch, heads, rows, head_dim = q.shape
    output = torch.empty_like(q, dtype=torch.int8)
    scales = torch.empty(
        batch, heads, ((rows + 127) // 128) * 32,
        dtype=torch.float32, device=q.device,
    )
    status = library.h3_int8_quantize_q_only(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(scales.data_ptr()), batch, heads, rows, head_dim,
        full_k_rows, q.stride(0), q.stride(1), q.stride(2), 2,
        ctypes.c_size_t(torch.cuda.current_stream().cuda_stream),
    )
    if status:
        message = library.h3_q_only_last_error().decode('utf-8', errors='replace')
        raise RuntimeError('Q-only native call failed: %s' % message)
    return output, scales


def main(argv=None):
    args = parse_args(argv)
    import torch
    from h3_optimizations.native import int8_attention

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    library, library_path = _load(args.library)
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(args.seed)
    results = []
    for q_rows, k_rows in args.cases:
        q = torch.randn(
            args.batch, args.heads, q_rows, 128,
            dtype=torch.bfloat16, device=device, generator=generator,
        )
        k = torch.randn(
            args.batch, args.heads, k_rows, 128,
            dtype=torch.bfloat16, device=device, generator=generator,
        )
        v = torch.randn(
            args.batch, args.heads, k_rows, 128,
            dtype=torch.bfloat16, device=device, generator=generator,
        )
        coupled = int8_attention.prequantize_int8_attention(q, k, v)
        q_only, q_scale = _quantize_q_only(torch, library, q, k_rows)
        torch.cuda.synchronize(device)
        bytes_exact = bool(torch.equal(q_only, coupled.q))
        scales_exact = bool(torch.equal(q_scale, coupled.q_scale))

        timings = []
        for _ in range(args.iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            trial_q, trial_scale = _quantize_q_only(torch, library, q, k_rows)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(float(begin.elapsed_time(end)))
            del trial_q, trial_scale
        results.append({
            'q_rows': q_rows,
            'full_k_rows': k_rows,
            'rotation': 4 if k_rows <= 256 else 128,
            'ragged_q_tail': q_rows % 128,
            'q_bytes_exact': bytes_exact,
            'q_scales_exact': scales_exact,
            'q_only_median_ms': statistics.median(timings),
            'q_only_samples_ms': timings,
        })
        del q, k, v, coupled, q_only, q_scale

    report = {
        'schema': 1,
        'experiment': 'native_q_only_byte_parity',
        'library': str(library_path),
        'timing_scope': 'q_only_only; production coupled timing is intentionally not compared',
        'cases': results,
        'gpu': torch.cuda.get_device_name(device),
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + '\n', encoding='utf-8')
    print(serialized if args.json else '\n'.join(
        'Q %d / K %d rotation %d: bytes=%s scales=%s, %.3f ms'
        % (
            row['q_rows'], row['full_k_rows'], row['rotation'],
            'exact' if row['q_bytes_exact'] else 'DIFFERS',
            'exact' if row['q_scales_exact'] else 'DIFFERS',
            row['q_only_median_ms'],
        )
        for row in results
    ))
    return 0 if all(row['q_bytes_exact'] and row['q_scales_exact'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
