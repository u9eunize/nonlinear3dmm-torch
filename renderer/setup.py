from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ZBuffer_cuda',
    ext_modules=[
        CUDAExtension('ZBuffer_cuda', [
            'ZBuffer_cuda.cpp',
            'ZBuffer_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
