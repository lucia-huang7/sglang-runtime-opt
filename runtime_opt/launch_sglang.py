"""Initialize the metadata extension before starting the patched SGLang server."""
import os
import runpy
from runtime_opt.descriptor_extension import initialize

if os.environ.get('SGLANG_FUSED_PREFIX_COPY') == '1':
    initialize()
runpy.run_module('sglang.launch_server', run_name='__main__')
