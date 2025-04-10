import sys
import warnings
import io
import os
import re
import ast
import glob
import subprocess

from pathlib import Path
from packaging.version import parse, Version
from setuptools import find_packages, setup
import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, CUDA_HOME

__version__ = "1.0.0"

# ninja build does not work unless include_dirs are abs path
current_dir = os.path.dirname(os.path.abspath(__file__))


def read(filename):
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return fd.read()

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )

ext_modules = []

custom_sources = ["csrc/IluvatarKernel/impl/cuda_core_kernel.cu",
                  "csrc/kernelpy.cpp",
                  ]
sources = custom_sources
for item in sources:
    sources[sources.index(item)] = os.path.join(current_dir, item)

include_paths = []
include_paths.extend(cpp_extension.include_paths(cuda=True))    # cuda path
include_paths.append(os.path.join(current_dir, 'csrc/include'))

ext_modules.append(
    CUDAExtension(
        name="ILUVATARKERNEL",
        sources=sources,
        include_dirs=include_paths,
        libraries=['cublas', 'cudart', 'cudnn', 'curand', 'cuinfer', 'nvToolsExt'],
        extra_compile_args={
            "cxx": ['-g',
                    '-std=c++17',
                    # '-U NDEBUG',
                    '-O3',
                    '-fopenmp',
                    '-lgomp'],
            "nvcc": ['-O3',
                     '-std=c++17',
                    ],
        },
        define_macros=[('VERSION_INFO', __version__),
                       # ('_DEBUG_MODE_', None),
                       ]
    )
)

setup(
    name="ILUVATARKERNEL",
    version=__version__,
    author_email="",
    description="pytorch extensive cuda",
    package_dir={"": "python"},
    packages=find_packages("python"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.1",
        "einops",
        "packaging",
        "ninja",
    ],
)
