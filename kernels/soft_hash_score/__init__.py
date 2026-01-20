# kernels/soft_hash_score/__init__.py
import os
from torch.utils.cpp_extension import load

_this_dir = os.path.dirname(__file__)

soft_hash_score_ext = load(
    name="soft_hash_score_ext",
    sources=[
        os.path.join(_this_dir, "soft_hash_score_ext.cpp"),
        os.path.join(_this_dir, "soft_hash_score_kernel.cu"),
    ],
    extra_cuda_cflags=["--use_fast_math"],
    extra_cflags=["-O3"],
    with_cuda=True,
    verbose=False,   # set True once if debugging build
)
