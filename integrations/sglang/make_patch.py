"""Generate the version-specific scheduler patch against the pinned source."""
import argparse
import difflib
import hashlib
from pathlib import Path

SOURCE = 'python/sglang/srt/managers/schedule_batch.py'
PINNED_SHA256 = 'c665923c0b259663da5ad28dadff8ff085fa7ee40c4c3faf66e935cb4d6f5230'


def patched_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != PINNED_SHA256:
        raise ValueError("Source does not match the pinned SGLang revision")
    source = source.replace('import logging\n', 'import logging\nimport os\n', 1)
    source = source.replace('import triton.language as tl\n',
        'import triton.language as tl\n\nfrom runtime_opt.prefix_copy import try_fused_prefix_copy\n\n'
        '_FUSED_PREFIX_COPY = os.environ.get("SGLANG_FUSED_PREFIX_COPY", "0") == "1"\n', 1)
    anchor = '        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):\n'
    assert source.count(anchor) == 1
    addition = '''        prefix_copied = False
        if (
            _FUSED_PREFIX_COPY
            and bs >= 16
            and sum(length > 0 for length in prefix_lens) >= 16
            and not isinstance(self.tree_cache, SWAChunkCache)
        ):
            for req, seq_len, pre_len in zip(reqs, seq_lens, prefix_lens):
                assert seq_len - pre_len == req.extend_input_len
            prefix_copied = try_fused_prefix_copy(
                self.req_to_token_pool.req_to_token,
                req_pool_indices,
                [req.prefix_indices for req in reqs],
            )

'''
    source = source.replace(anchor, addition + anchor, 1)
    anchor = '''                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                )'''
    replacement = '''                if not prefix_copied:
                    self.req_to_token_pool.write(
                        (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
                    )'''
    assert source.count(anchor) == 1
    return source.replace(anchor, replacement, 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--patched-output')
    args = p.parse_args()
    original = Path(args.source).read_text()
    patched = patched_source(original)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                       fromfile='a/' + SOURCE, tofile='b/' + SOURCE))
    Path(args.output).write_text(patch)
    if args.patched_output: Path(args.patched_output).write_text(patched)
