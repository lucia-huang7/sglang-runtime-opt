"""Exercise the actual Scheduler wrapper with initialized distributed groups."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import sglang.srt.managers.scheduler as scheduler
import sglang.srt.managers.schedule_batch as schedule
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from benchmarks.sglang_prefix_copy import build_case


def main():
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=device)
    gloo=dist.new_group(backend='gloo')
    for key,value in {'attention_backend':'triton','prefill_attention_backend':'triton',
            'decode_attention_backend':'triton','use_mla_backend':False,'enable_two_batch_overlap':False}.items():
        schedule.global_server_args_dict[key]=value
    cases=[('immediate',{},True),('overlap',{},True),('off',{},False),
           ('overlap',{'pp_size':2},False),('overlap',{'is_generation':False},False),
           ('overlap',{'spec_algorithm':SpeculativeAlgorithm.EAGLE},False),
           ('overlap',{'disable_overlap_schedule':False},False)]
    results=[]
    for mode,changes,expected in cases:
        args=SimpleNamespace(dp_size=2,disable_cuda_graph=False,speculative_num_draft_tokens=0,
            enable_two_batch_overlap=False,enable_deepep_moe=False,deepep_mode='normal',
            disable_overlap_schedule=changes.get('disable_overlap_schedule',True),
            enable_dp_attention=True,moe_dense_tp_size=None)
        obj=object.__new__(scheduler.Scheduler)
        obj.server_args=args;obj.attn_tp_size=1
        obj.tp_group=SimpleNamespace(device_group=dist.group.WORLD,cpu_group=gloo,device=device)
        obj.spec_algorithm=changes.get('spec_algorithm',SpeculativeAlgorithm.NONE)
        obj.is_generation=changes.get('is_generation',True);obj.pp_size=changes.get('pp_size',1)
        obj.get_idle_batch=lambda: (_ for _ in ()).throw(AssertionError('Unexpected idle batch'))
        batch,reset=build_case(8+rank,33,device,1.0,True,True);reset();batch.prepare_for_extend()
        if mode=='off':os.environ.pop('SGLANG_ASYNC_METADATA',None)
        else:os.environ['SGLANG_ASYNC_METADATA']=mode
        result=obj.prepare_mlp_sync_batch(batch)
        assert hasattr(obj.tp_group,'_runtime_metadata_gathers')==expected
        assert (getattr(batch,'_pending_worker_descriptor',None) is not None)==(expected and mode=='overlap')
        before=schedule.bid;worker=result.get_model_worker_batch()
        assert schedule.bid-before==(0 if expected and mode=='overlap' else 1)
        assert worker.global_num_tokens==[(8+r)*4 for r in [0,1]]
        results.append(dict(mode=mode,guard_changes={k:str(v) for k,v in changes.items()},optimized_path=expected,status='passed'))
    if rank==0:
        Path('results/async-guard-validation.json').write_text(json.dumps(dict(world_size=2,case_count=len(results),cases=results),indent=2))
        print('Actual scheduler wrapper guards passed:',len(results),flush=True)
    dist.destroy_process_group(gloo);dist.destroy_process_group()


if __name__=='__main__':main()
