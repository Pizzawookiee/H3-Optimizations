'''Shared sequence-chunked H3 QKV projection.'''

from ..attention_forward import project_qkv, to_hnd


def project_chunk_hnd(module, x, rope_freqs, start, end):
    chunk_rope = None if rope_freqs is None else rope_freqs[:, start:end]
    return to_hnd(*project_qkv(module, x[start:end], chunk_rope))
