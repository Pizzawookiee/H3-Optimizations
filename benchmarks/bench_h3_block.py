'''Compare full Kitchen, chunked Kitchen, and legacy attention in a real H3 block.'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_SEQUENCE = 4096


def load_block_state(checkpoint, block_index):
    from safetensors import safe_open

    prefixes = (
        'model.diffusion_model.blocks.%d.' % int(block_index),
        'diffusion_model.blocks.%d.' % int(block_index),
        'blocks.%d.' % int(block_index),
    )
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        keys = set(handle.keys())
        prefix = next(
            (
                candidate
                for candidate in prefixes
                if candidate + 'attn.qkv_proj.weight' in keys
            ),
            None,
        )
        if prefix is None:
            raise KeyError(
                'checkpoint has no blocks.%d state' % int(block_index)
            )
        state = {
            key[len(prefix):]: handle.get_tensor(key)
            for key in keys
            if key.startswith(prefix)
        }
    return prefix, state


def build_block(torch, checkpoint, block_index, device):
    import comfy.ops
    from comfy.ldm.minimax.model import DiTBlock

    prefix, state = load_block_state(checkpoint, block_index)
    qkv = state['attn.qkv_proj.weight']
    q_norm = state['attn.q_norm.weight']
    fc1 = state['mlp.fc1.weight']
    adaln = state['adaln_proj.linear.weight']
    hidden = int(qkv.shape[1])
    head_dim = int(q_norm.shape[0])
    heads = int(qkv.shape[0]) // (3 * head_dim)
    ffn = int(fc1.shape[0]) // 2
    t_dim = int(adaln.shape[1])
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    block = DiTBlock(
        hidden,
        heads,
        head_dim,
        ffn,
        t_dim,
        1e-5,
        1e-5,
        apply_silu=False,
        adaln_dtype=torch.float32,
        dtype=torch.bfloat16,
        operations=ops,
    )
    block.load_state_dict(state, strict=True)
    del state
    for parameter in block.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
    block.to(device)
    return block, prefix, hidden, t_dim


def finish_attention(module, out_hnd):
    out = out_hnd.transpose(1, 2).reshape(
        out_hnd.shape[0],
        out_hnd.shape[2],
        int(module.heads) * int(module.head_dim),
    )
    return module.out_proj(out.squeeze(0))


def full_kitchen_forward(torch, module, full_attention):
    def forward(x, rope_freqs=None, transformer_options=None):
        del transformer_options
        return finish_attention(
            module,
            full_attention(torch, module, x, rope_freqs),
        )

    return forward


def dense_marker():
    from h3_optimizations.dense_resolver import (
        ATTENTION_COMFY_KITCHEN_INT8,
        OVERRIDE_MARKER,
    )

    def override(*_args, **_kwargs):
        raise AssertionError('the marker is not an executable attention path')

    setattr(override, OVERRIDE_MARKER, ATTENTION_COMFY_KITCHEN_INT8)
    return override


def tensor_error(torch, reference, actual):
    delta = actual.float() - reference.float()
    rmse = delta.square().mean().sqrt()
    reference_rms = reference.float().square().mean().sqrt()
    return {
        'exact': bool(torch.equal(reference, actual)),
        'max_abs': float(delta.abs().max().item()),
        'rmse': float(rmse.item()),
        'relative_rmse': float(
            (rmse / reference_rms.clamp_min(1e-12)).item()
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run three attention paths through one real bounded H3 DiT block.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--kitchen-source', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error('sequence/iteration arguments are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required GPU preflight'
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    comfy_root = Path(__file__).resolve().parents[3]
    pack_root = Path(__file__).resolve().parents[1]
    kitchen_root = Path(args.kitchen_source).resolve()
    if not (kitchen_root / 'comfy_kitchen' / '__init__.py').is_file():
        raise SystemExit('--kitchen-source is not a Comfy Kitchen source checkout')
    sys.path.insert(0, str(kitchen_root))
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(pack_root))

    import torch

    from bench_chunked_kitchen_qkv import (
        benchmark_case,
        build_legacy_fused_backend,
        full_kitchen_attention,
        make_rope,
        resolve_checkpoint,
    )
    from h3_optimizations.attention_forward import make_forward as make_attention
    from h3_optimizations.dense_fused_qkv import DenseFusedQKVProjector
    from h3_optimizations.kitchen_qkv import (
        ChunkedKitchenAttentionBackend,
        ChunkedKitchenQKVProjector,
    )
    from h3_optimizations.memory.config import (
        MODE_CONVROT_2SLICE,
        ActivationMemoryConfig,
    )
    from h3_optimizations.memory.forward import make_forward as make_block_forward

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    checkpoint = resolve_checkpoint(args.checkpoint)
    block, prefix, hidden, t_dim = build_block(
        torch,
        checkpoint,
        args.block,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    t_emb = torch.randn(
        (1, t_dim),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    rope = make_rope(torch, args.sequence, device)
    text_stop = min(512, args.sequence)
    audio_stop = min(text_stop + 256, args.sequence)
    mod_segments = []
    if text_stop:
        mod_segments.append((0, text_stop, 1))
    if audio_stop > text_stop:
        mod_segments.append((text_stop, audio_stop, 2))
    if args.sequence > audio_stop:
        mod_segments.append((audio_stop, args.sequence, 0))

    options = {'optimized_attention_override': dense_marker()}
    full_forward = full_kitchen_forward(
        torch,
        block.attn,
        full_kitchen_attention,
    )
    chunked_forward = make_attention(
        block.attn,
        args.block,
        backend=ChunkedKitchenAttentionBackend(),
        projector=ChunkedKitchenQKVProjector(),
    )
    legacy_forward = make_attention(
        block.attn,
        args.block,
        backend=build_legacy_fused_backend(),
        projector=DenseFusedQKVProjector(),
    )
    block_forward = make_block_forward(
        block,
        args.block,
        ActivationMemoryConfig(
            mode=MODE_CONVROT_2SLICE,
            chunk_rows=2048,
            strict=True,
        ),
        original_forward=block.forward,
    )

    def run(attention_forward):
        block.attn.forward = attention_forward
        return block_forward(
            x.clone(),
            t_emb,
            mod_segments,
            rope,
            transformer_options=options,
        )

    full_output = run(full_forward)
    chunked_output = run(chunked_forward)
    legacy_output = run(legacy_forward)
    parity = {
        'chunked_vs_full_kitchen': tensor_error(
            torch, full_output, chunked_output
        ),
        'legacy_vs_full_kitchen': tensor_error(
            torch, full_output, legacy_output
        ),
    }
    del full_output, chunked_output, legacy_output

    cases = {
        'full_kitchen': benchmark_case(
            torch,
            lambda: run(full_forward),
            args.warmup,
            args.iterations,
            device,
        ),
        'chunked_kitchen': benchmark_case(
            torch,
            lambda: run(chunked_forward),
            args.warmup,
            args.iterations,
            device,
        ),
        'legacy_fused_sage': benchmark_case(
            torch,
            lambda: run(legacy_forward),
            args.warmup,
            args.iterations,
            device,
        ),
    }
    result = {
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': int(args.block),
        'sequence': int(args.sequence),
        'hidden': int(hidden),
        'bounded_mlp_chunk_rows': 2048,
        'cases': cases,
        'output_parity': parity,
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
        },
        'comfy_kitchen_source': str(
            Path(sys.modules['comfy_kitchen'].__file__).resolve()
        ),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for name, details in cases.items():
            print('%s: %.3f ms, peak %.3f GiB' % (
                name,
                details['median_ms'],
                details['peak_allocated_bytes'] / 2**30,
            ))
        print('block output parity:')
        print(json.dumps(parity, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
