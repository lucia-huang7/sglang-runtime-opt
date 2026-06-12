"""Compile the CPU integer packer for reusable pinned staging."""
from functools import lru_cache
from pathlib import Path
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def initialize():
    return load(name='runtime_integer_staging',
                sources=[str(Path(__file__).parent/'csrc/integer_staging.cpp')],
                extra_cflags=['-O3'],with_cuda=True,verbose=False)
