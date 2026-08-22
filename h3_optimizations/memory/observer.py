'''Optional exact-MLP observation seam for H3 diagnostics.'''

OBSERVER_KEY = 'h3_optimizations_mlp_observer'


def get_mlp_observer(transformer_options=None):
    if not transformer_options:
        return None
    return transformer_options.get(OBSERVER_KEY)


def notify_exact_mlp(layer_index, transformer_options=None, **payload):
    observer = get_mlp_observer(transformer_options)
    if observer is not None:
        observer.observe_exact_mlp(
            int(layer_index),
            transformer_options,
            **payload,
        )


def notify_mlp_block_end(layer_index, transformer_options=None):
    observer = get_mlp_observer(transformer_options)
    if observer is not None:
        observer.end_mlp_block(int(layer_index), transformer_options)
