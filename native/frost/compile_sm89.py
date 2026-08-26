import os
from pathlib import Path


os.environ.setdefault('CUTE_DSL_ARCH', 'sm_89')
os.environ.setdefault('CUTE_DSL_DUMP_DIR', '/work/artifacts')
os.environ.setdefault('CUTE_DSL_KEEP', 'ptx,cubin')
os.environ.setdefault('CUTE_DSL_NO_CACHE', '1')
os.environ.setdefault('CUTE_DSL_KEEP_PTX', '1')
os.environ.setdefault('CUTE_DSL_KEEP_CUBIN', '1')
os.environ.setdefault('CUTE_DSL_KEEP_SASS', '1')

import cudnn


source_root = Path(os.environ.get('CUDNN_FRONTEND_SOURCE', '/work/cudnn-frontend'))
source_cudnn = source_root / 'python' / 'cudnn'
cudnn.__path__.insert(0, str(source_cudnn))

from cudnn.frost.template_loader import load_template
from cudnn.sdpa.fwd import config_sm80


params = config_sm80.params_for_flavor(
    'llama',
    io_bf16=True,
    tile_m=64,
    num_warps=4,
    tile_n=64,
    has_lse=False,
)
kernel_path = source_cudnn / 'sdpa' / 'fwd' / 'kernels' / 'prefill_f16_sm80.py'
module = load_template(kernel_path, params, tag='h3_sm89_bf16_64x64')
compiled = module.compile(b=1, h=56, h_kv=56, sq=193, skv=193, d=128)
print(compiled)
