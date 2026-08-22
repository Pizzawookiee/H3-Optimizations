'''Optional executable MLP output-sharing seam for H3 experiments.'''

SHARING_KEY = 'h3_optimizations_mlp_sharing_session'


def get_mlp_sharing(transformer_options=None):
    if not transformer_options:
        return None
    return transformer_options.get(SHARING_KEY)
