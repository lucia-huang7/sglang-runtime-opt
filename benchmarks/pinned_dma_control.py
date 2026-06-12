"""Isolate prepared pinned-buffer DMA from pageable-to-pinned staging costs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist
from benchmarks.async_metadata import measure,busy


def dma_case(direction,count,work_us,device):
    pageable=(torch.arange(count,dtype=torch.int64)%251).to(torch.uint8)
    pinned=pageable.pin_memory()
    source=pageable.to(device)
    destination=torch.empty_like(source)
    ready=torch.cuda.Event()
    def call(host,asynchronous):
        if direction=='h2d':destination.copy_(host,non_blocking=asynchronous)
        else:host.copy_(source,non_blocking=asynchronous)
        if asynchronous:ready.record(torch.cuda.current_stream(device))
        busy(work_us)
        if asynchronous:ready.synchronize()
        return None
    methods={'pageable_blocking':lambda:call(pageable,False),
             'prepared_pinned_blocking':lambda:call(pinned,False),
             'prepared_pinned_overlap':lambda:call(pinned,True)}
    for fn in methods.values():
        fn();torch.cuda.synchronize()
        if direction=='h2d':assert torch.equal(destination.cpu(),pageable)
        else:assert torch.equal(pinned,pageable)
    return methods


def main(a):
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    torch.cuda.set_device(device)
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[min(rank*4,len(allowed)-1)]})
    dist.init_process_group('nccl',device_id=device);rows=[]
    for direction in ['h2d','d2h']:
        for count in [8,1024,65536,1048576,16777216]:
            for work in [0,200,1000]:
                methods=dma_case(direction,count,work,device);order=list(methods)
                if a.reverse:order.reverse()
                measurements={}
                for mode in order+list(reversed(order)):
                    measurements.setdefault(mode,[]).append(measure(methods[mode],50,100,rank,device))
                rows.append(dict(direction=direction,bytes=count,independent_cpu_work_us=work,depth=1,measurements=measurements))
                if rank==0:print(direction,count,work,{k:round(statistics.median([b['worst_rank']['median_us'] for b in v]),2) for k,v in measurements.items()},flush=True)
    if rank==0:Path(a.output).write_text(json.dumps(dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        world_size=2,kind='prepared_dma',producer_buffers_prepared_before_timing=True,
        host_staging_included=False,pinning_allocation_included=False,model_forward=False,
        samples_per_block=100,warmup=50,order='reverse then forward' if a.reverse else 'forward then reverse',rows=rows),indent=2))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reverse',action='store_true');p.add_argument('--output',required=True);main(p.parse_args())
