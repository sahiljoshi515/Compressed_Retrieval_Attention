from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rw_decode_ext",
    ext_modules=[
        CUDAExtension(
            name="rw_decode_ext",
            sources=["rw_decode_ext.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
