'''Early release of dead MiniMax H3 embedding-assembly tensors.'''

from __future__ import annotations

import ast
import inspect
import textwrap

from ..model import get_minimax_h3_model


FORWARD_KEY = 'diffusion_model._forward'
OWNER_MARKER = '_h3_optimizations_embedding_memory'
SIGNATURE_MARKER = '_h3_optimizations_embedding_memory_signature'
ORIGINAL_MARKER = '_h3_optimizations_embedding_memory_original'
_DEAD_EMBEDDING_NAMES = (
    'video_embed',
    'audio_embed',
    'all_video_rows',
    'all_audio_rows',
    'video_rows',
    'audio_rows',
    'cond_video_rows',
    'cond_audio_rows',
    'img_update',
    'audio_update',
    'text_states',
)


class H3EmbeddingMemoryPatchError(RuntimeError):
    pass


def _source_function(forward):
    function = getattr(forward, '__func__', forward)
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise H3EmbeddingMemoryPatchError(
            'cannot inspect MiniMax H3 _forward for embedding-memory compatibility'
        ) from exc
    module = ast.parse(source)
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1 or functions[0].name != '_forward':
        raise H3EmbeddingMemoryPatchError(
            'cannot locate MiniMax H3 _forward for embedding-memory compatibility'
        )
    return function, module, functions[0]


def _loaded_names(node):
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _stored_names(node):
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def _assembly_index(function):
    candidates = []
    for index, statement in enumerate(function.body):
        if not isinstance(statement, ast.For):
            continue
        loaded = _loaded_names(statement)
        stored = _stored_names(statement)
        if {
            'layout',
            'video_embed',
            'audio_embed',
            'text_states',
        }.issubset(loaded) and 'h' in loaded and {'a', 'b', 'kind'}.issubset(stored):
            candidates.append(index)
    if len(candidates) != 1:
        raise H3EmbeddingMemoryPatchError(
            'MiniMax H3 embedding assembly changed; refusing early tensor release'
        )
    return candidates[0]


def _validate_lifetime(function, assembly_index):
    before = ast.Module(body=function.body[: assembly_index + 1], type_ignores=[])
    after = ast.Module(body=function.body[assembly_index + 1 :], type_ignores=[])
    missing = set(_DEAD_EMBEDDING_NAMES) - _stored_names(before)
    still_live = set(_DEAD_EMBEDDING_NAMES) & _loaded_names(after)
    if missing or still_live:
        detail = []
        if missing:
            detail.append('missing definitions: %s' % ', '.join(sorted(missing)))
        if still_live:
            detail.append('still used: %s' % ', '.join(sorted(still_live)))
        raise H3EmbeddingMemoryPatchError(
            'MiniMax H3 embedding tensor lifetime changed (%s)' % '; '.join(detail)
        )


def _compile_forward(original_forward):
    function, module, parsed = _source_function(original_forward)
    assembly_index = _assembly_index(parsed)
    _validate_lifetime(parsed, assembly_index)
    parsed.body.insert(
        assembly_index + 1,
        ast.Delete(
            targets=[
                ast.Name(id=name, ctx=ast.Del())
                for name in _DEAD_EMBEDDING_NAMES
            ]
        ),
    )
    ast.fix_missing_locations(module)
    first_line = int(getattr(getattr(function, '__code__', None), 'co_firstlineno', 1))
    ast.increment_lineno(module, first_line - 1)
    globals_dict = dict(getattr(function, '__globals__', {}))
    namespace = {}
    filename = getattr(getattr(function, '__code__', None), 'co_filename', '<h3_embedding>')
    exec(compile(module, filename, 'exec'), globals_dict, namespace)
    return namespace['_forward']


def make_forward(model, original_forward):
    patched_forward = _compile_forward(original_forward)

    def forward(x, timestep, context, transformer_options={}, minimax_payload=None,
                denoise_mask=None, audio_denoise_mask=None, **kwargs):
        return patched_forward(
            model,
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask,
            **kwargs,
        )

    setattr(forward, OWNER_MARKER, True)
    setattr(forward, SIGNATURE_MARKER, 'release')
    setattr(forward, ORIGINAL_MARKER, original_forward)
    return forward


def install(model_patcher, *, force_rebuild=False):
    model = get_minimax_h3_model(model_patcher)
    if model is None:
        raise H3EmbeddingMemoryPatchError(
            'H3 Memory Optimization can only patch MiniMaxH3Model'
        )
    existing = getattr(model_patcher, 'object_patches', {}).get(FORWARD_KEY)
    if existing is not None and not getattr(existing, OWNER_MARKER, False):
        options = model_patcher.model_options['transformer_options'] = (
            model_patcher.model_options.get('transformer_options', {}).copy()
        )
        options['h3_optimizations_preserved_embedding_patch'] = True
        return False
    if existing is not None:
        if not force_rebuild:
            return False
        original = getattr(existing, ORIGINAL_MARKER, None)
        if original is None:
            raise H3EmbeddingMemoryPatchError(
                'installed embedding-memory patch has no recoverable original'
            )
    else:
        original = model._forward
        if getattr(original, OWNER_MARKER, False):
            original = getattr(original, ORIGINAL_MARKER, None)
            if original is None:
                raise H3EmbeddingMemoryPatchError(
                    'installed embedding-memory patch has no recoverable original'
                )
    model_patcher.add_object_patch(FORWARD_KEY, make_forward(model, original))
    options = model_patcher.model_options['transformer_options'] = (
        model_patcher.model_options.get('transformer_options', {}).copy()
    )
    options['h3_optimizations_preserved_embedding_patch'] = False
    return True


def clear(model_patcher):
    patches = getattr(model_patcher, 'object_patches', {})
    current = patches.get(FORWARD_KEY)
    if current is not None and getattr(current, OWNER_MARKER, False):
        patches.pop(FORWARD_KEY)
        return True
    return False
