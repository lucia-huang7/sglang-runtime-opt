"""Additional real scheduler shapes with larger prefill input-token payloads."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace
import torch
import torch.distributed as dist
import sglang.srt.managers.schedule_batch as schedule
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from benchmarks.sglang_prefix_copy import snapshot, original_method
from benchmarks.async_metadata import measure


def build_case(size,extend,device):
    length=129
    pool=ReqToTokenPool(size+4,length+extend+21,device,False)
    pool.req_to_token.fill_(-99)
    allocator=TokenToKVPoolAllocator(size*(length+extend+8)+100,torch.int64,device,None)
    assert allocator.alloc(size*length) is not None
    initial=allocator.free_pages;requests=[]
    for i in range(size):
        n=length if i<size//2 else 0
        params=SamplingParams(temperature=1.0,top_k=1,max_new_tokens=8)
        req=schedule.Req(str(i),None,[1]*(n+extend),params,return_logprob=False)
        req.fill_ids=req.origin_input_ids.copy()
        req.prefix_indices=torch.arange(n,device=device,dtype=torch.int64)+i*length+1 if n else []
        req.extend_input_len=extend;req.logprob_start_len=-1;requests.append(req)
    batch=schedule.ScheduleBatch(reqs=requests,req_to_token_pool=pool,token_to_kv_pool_allocator=allocator,
        model_config=SimpleNamespace(vocab_size=32000,is_encoder_decoder=False),device=device,
        return_logprob=False,enable_custom_logit_processor=False)
    def reset():
        pool.free_slots=list(range(pool.size));allocator.free_pages=initial
        for req in requests:req.cached_tokens=0;req.already_computed=0
    return batch,reset


def main(a):
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    torch.cuda.set_device(device)
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[min(rank*4,len(allowed)-1)]})
    dist.init_process_group('nccl',device_id=device)
    for key,value in {'attention_backend':'triton','prefill_attention_backend':'triton',
            'decode_attention_backend':'triton','use_mla_backend':False,'enable_two_batch_overlap':False}.items():
        schedule.global_server_args_dict[key]=value
    schedule._FUSED_PREFIX_COPY=True
    baseline=original_method('/workspace/schedule_batch.m2.py')
    rows=[]
    for size in [8,32]:
        for extend in [512,2048,8192]:
            batch,reset=build_case(size+rank*3,extend,device)
            reset();baseline(batch);expected,state=snapshot(batch)
            os.environ['SGLANG_PINNED_INPUTS']='1';reset();batch.prepare_for_extend()
            actual,actual_state=snapshot(batch)
            for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            assert state==actual_state
            def upstream():reset();baseline(batch);return None
            def pinned():reset();batch.prepare_for_extend();return None
            methods={'upstream':upstream,'pinned':pinned};order=list(methods)
            if a.reverse:order.reverse()
            values={}
            for mode in order+list(reversed(order)):
                values.setdefault(mode,[]).append(measure(methods[mode],50,100,rank,device))
            rows.append(dict(direction='h2d',batch_rank0=size,batch_rank1=size+3,extend_tokens_per_request=extend,
                return_logprob=False,prefix_tokens=129,cache_hit_fraction=.5,correctness='exact',measurements=values))
            if rank==0:print(size,extend,{k:round(statistics.median([b['worst_rank']['median_us'] for b in v]),2) for k,v in values.items()},flush=True)
    if rank==0:Path(a.output).write_text(json.dumps(dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        kind='prefill',world_size=2,model_forward=False,warmup=50,samples_per_block=100,
        order='reverse then forward' if a.reverse else 'forward then reverse',rows=rows),indent=2))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reverse',action='store_true');p.add_argument('--output',required=True);main(p.parse_args())
