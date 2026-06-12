"""Add guarded packed H2D and owned D2H sequence-length paths after M1/M2."""
import argparse
import difflib
import hashlib
from pathlib import Path


def apply(root, output):
    file=Path(root)/'srt/managers/schedule_batch.py'
    original=file.read_text()
    expected='3292f075b9640f09c2092db9b9ac093bc3c3f8c2a3197c856f1fe695849786e7'
    if expected and hashlib.sha256(original.encode()).hexdigest()!=expected:
        raise RuntimeError('Expected the validated M1/M2 source snapshot')
    if 'SGLANG_PINNED_INPUTS' in original:raise RuntimeError('Pinned transfer patch already applied')
    s=original
    start=s.index('        req_pool_indices_tensor = torch.tensor(',s.index('    def prepare_for_extend('))
    end=s.index('\n        token_type_ids_tensor = None',start)
    block=s[start:end]
    wrapped='''        packed_inputs = None
        if os.getenv("SGLANG_PINNED_INPUTS") == "1" and str(self.device).startswith("cuda"):
            from runtime_opt.pinned_transfers import pack_scheduler_inputs
            packed_inputs = pack_scheduler_inputs(
                [req_pool_indices, sum(input_ids, []), seq_lens, prefix_lens], self.device
            )
        if packed_inputs is not None:
            (req_pool_indices_tensor, input_ids_tensor,
             seq_lens_tensor, prefix_lens_tensor) = packed_inputs
        else:
'''+''.join('    '+line+'\n' for line in block.splitlines())
    s=s[:start]+wrapped+s[end:]
    start=s.index('    def get_model_worker_batch(')
    end=s.index('    def copy(self):',start)
    body=s[start:end]
    first=body.index('        if self.forward_mode.is_decode_or_idle():')
    prefetch='''        length_transfer = None
        if (os.getenv("SGLANG_PINNED_LENGTHS") == "1"
                and seq_lens_cpu_cache is None and self.seq_lens is not None):
            backend = global_server_args_dict[
                "decode_attention_backend" if self.forward_mode.is_decode_or_idle()
                else "prefill_attention_backend"
            ]
            needs_cpu = (backend in ("fa3", "flashmla", "cutlass_mla", "ascend")
                         or (global_server_args_dict["use_mla_backend"] and backend == "flashinfer")
                         or global_server_args_dict["enable_two_batch_overlap"])
            if needs_cpu:
                from runtime_opt.pinned_transfers import begin_sequence_lengths
                length_transfer = begin_sequence_lengths(self.seq_lens)
'''
    body=body[:first]+prefetch+body[first:]
    body=body.replace('                else self.seq_lens.cpu()','                else (None if length_transfer is not None else self.seq_lens.cpu())')
    assert body.count('        return ModelWorkerBatch(')==1
    body=body.replace('        return ModelWorkerBatch(','        worker_batch = ModelWorkerBatch(')
    body=body.rstrip()+'''
        if length_transfer is not None:
            # The returned worker owns its CPU copy beyond ring reuse.
            worker_batch.seq_lens_cpu = length_transfer.consume(owned=True)
        return worker_batch

'''
    # Drain a prefetched transfer if any descriptor construction step raises.
    try_start=body.index('        if self.forward_mode.is_decode_or_idle():')
    body=body[:try_start]+'        try:\n'+''.join('    '+line+'\n' for line in body[try_start:].splitlines())+'''        finally:
            if length_transfer is not None and not length_transfer.released:
                length_transfer.release()

'''
    s=s[:start]+body+s[end:]
    # This module already imports os in the pinned upstream source.
    compile(s,str(file),'exec')
    target=Path(output);target.mkdir(parents=True,exist_ok=True)
    (target/'pinned-transfers.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),s.splitlines(True),fromfile='a/python/sglang/srt/managers/schedule_batch.py',
        tofile='b/python/sglang/srt/managers/schedule_batch.py')))
    file.write_text(s)
    print('Patched packed scheduler H2D and owned sequence-length D2H')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();apply(a.source_root,a.output)
