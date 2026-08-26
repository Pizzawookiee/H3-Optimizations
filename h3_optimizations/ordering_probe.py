'''Compatibility seam for post-RoPE H3 attention observers.'''


ORDERING_OBSERVER_KEY = 'h3_optimizations_attention_ordering_observer'


def observe_attention(layer_index, transformer_options, q, k, v):
    if not transformer_options:
        return
    observer = transformer_options.get(ORDERING_OBSERVER_KEY)
    if observer is not None:
        observer.observe_attention(layer_index, transformer_options, q, k, v)


def has_ordering_observer(transformer_options):
    return bool(
        transformer_options
        and transformer_options.get(ORDERING_OBSERVER_KEY) is not None
    )
