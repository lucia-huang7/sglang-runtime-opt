"""Validate real scheduler metadata and worker descriptors on two GPU ranks."""
import argparse
import datetime
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import sglang.srt.managers.schedule_batch as schedule
import sglang.srt.managers.scheduler as scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from benchmarks.async_metadata import source_method
from benchmarks.sglang_prefix_copy import build_case
from runtime_opt.async_metadata import MetadataGather


def run(a):
    rank = int(os.environ['LOCAL_RANK']); device = torch.device('cuda', rank)
    torch.cuda.set_device(device); dist.init_process_group('nccl', device_id=device)
    gloo = dist.new_group(backend='gloo')
    group = SimpleNamespace(device_group=dist.group.WORLD, cpu_group=gloo, device=device)
    for key, value in {'attention_backend':'triton','prefill_attention_backend':'triton',
                       'decode_attention_backend':'triton','use_mla_backend':False,
                       'enable_two_batch_overlap':False}.items(): schedule.global_server_args_dict[key]=value
    baseline = source_method(a.original_source, vars(scheduler))
    patched = scheduler.Scheduler.prepare_mlp_sync_batch_raw
    rows = []
    for backend, pg in [('nccl',dist.group.WORLD),('gloo',gloo)]:
        for width in [1,6,257]:
            pending = MetadataGather(width, pg, device if backend=='nccl' else 'cpu')
            for repeat in range(40):
                values=[(repeat * 13 + rank + j) % 37 if repeat%3 else 0 for j in range(width)]
                pending.start(values)
                values[:] = [-999] * width
                try: pending.start(values)
                except RuntimeError: pass
                else: raise AssertionError('In-flight buffer reuse was accepted')
                got = pending.finish().tolist()
                expected=[[(repeat*13+r+j)%37 if repeat%3 else 0 for j in range(width)] for r in range(2)]
                assert got==expected
                try: pending.finish()
                except RuntimeError: pass
                else: raise AssertionError('Repeated consumption was accepted')
            pending.close()
            rows.append(dict(case='alternating counts, input mutation and reuse guards',backend=backend,width=width,iterations=40,status='passed'))
    for backend, pg in [('nccl',dist.group.WORLD),('gloo',gloo)]:
        pending = MetadataGather(1, pg, device if backend == 'nccl' else 'cpu')
        pending.host_output.fill_(-12345)
        dist.barrier()
        if rank == 1: time.sleep(.05)
        pending.start([rank + 11])
        if rank == 0:
            assert -12345 in pending.host_output.tolist(), 'Delayed peer must expose an unfinished buffer'
        assert pending.finish().tolist() == [[11], [12]]
        dist.barrier()
        rows.append(dict(case='premature-read fault injection with delayed peer',backend=backend,status='passed'))
    common=dict(dp_size=2,attn_tp_size=1,tp_group=group,disable_cuda_graph=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,speculative_num_draft_tokens=0,
                enable_two_batch_overlap=False,enable_deepep_moe=False,
                deepep_mode=scheduler.DeepEPMode.normal,require_mlp_tp_gather=True,
                disable_overlap_schedule=True)
    def make_batch(size, decode=False):
        b,reset=build_case(size,33,device,.5,True,True);reset();b.prepare_for_extend()
        if decode: b.forward_mode=schedule.ForwardMode.DECODE
        return b
    for size in [1,8,32,128]:
        for decode in [False,True]:
            for rank_empty in [False,True]:
                for require_gather in [False,True]:
                    snapshots=[]
                    for mode in ['upstream','immediate','overlap']:
                        b=None if rank_empty and rank==0 else make_batch(size+rank*3,decode)
                        idle=make_batch(1);idle.reqs=[];idle.prepare_for_idle()
                        kwargs=dict(common,get_idle_batch=lambda:idle,require_mlp_tp_gather=require_gather)
                        before=schedule.bid
                        if mode=='upstream': result=baseline(b,**kwargs)
                        else: result=patched(b,**kwargs,async_metadata_mode=mode)
                        worker=result.get_model_worker_batch()
                        assert schedule.bid-before==1
                        assert getattr(result,'_pending_worker_descriptor',None) is None
                        snap={name:getattr(worker,name) for name in
                            ['global_num_tokens','global_num_tokens_for_logprob','is_extend_in_batch',
                             'can_run_dp_cuda_graph','tbo_split_seq_index','global_forward_mode',
                             'lora_paths','extend_seq_lens','extend_prefix_lens','extend_logprob_start_lens']}
                        for name in ['input_ids','req_pool_indices','seq_lens','out_cache_loc']:
                            tensor=getattr(worker,name)
                            snap[name]=tensor.cpu().tolist() if tensor is not None else None
                        snapshots.append(snap)
                    assert snapshots[0]==snapshots[1]==snapshots[2],snapshots
                    rows.append(dict(case='actual scheduler and worker descriptor',batch=size,decode=decode,
                        empty_rank=rank_empty,require_mlp_gather=require_gather,status='passed'))
    for mode in ['upstream','immediate','overlap']:
        kwargs=dict(common,get_idle_batch=lambda: (_ for _ in ()).throw(AssertionError('Unexpected idle batch')))
        result=baseline(None,**kwargs) if mode=='upstream' else patched(None,**kwargs,async_metadata_mode=mode)
        assert result is None
    rows.append(dict(case='all ranks empty',status='passed'))
    # Nondefault current streams must not change the dedicated communication-stream contract.
    stream=torch.cuda.Stream(device=device)
    pending=MetadataGather(6,dist.group.WORLD,device)
    with torch.cuda.stream(stream):
        pending.start([rank]*6)
        assert pending.finish().tolist()==[[0]*6,[1]*6]
    rows.append(dict(case='nondefault caller stream',status='passed'))
    dist.barrier()
    if rank==0:
        Path(a.output).write_text(json.dumps(dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            world_size=2,case_count=len(rows),source_cases=32,metadata_iterations_per_rank=240,
            actual_classes=['Scheduler.prepare_mlp_sync_batch_raw','ScheduleBatch','ModelWorkerBatch','Req','ReqToTokenPool','TokenToKVPoolAllocator','SamplingBatchInfo'],
            model_forward=False,cases=rows),indent=2))
        print('Validation passed:',len(rows),'case groups',flush=True)
    dist.destroy_process_group(gloo);dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-source',default='/workspace/scheduler.original.py');p.add_argument('--output',required=True)
    run(p.parse_args())
