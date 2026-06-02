"""Build the C++ metadata packer with the installed PyTorch/CUDA toolchain."""
from functools import lru_cache
from pathlib import Path
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def initialize():
    return load(name='runtime_prefix_descriptors',
                sources=[str(Path(__file__).parent / 'csrc' / 'prefix_descriptors.cpp')],
                extra_cflags=['-O3'], with_cuda=True, verbose=False)
