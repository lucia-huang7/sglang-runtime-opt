"""Two-H100 transfer controls and actual SGLang source-method comparisons."""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
from runtime_opt.pinned_transfers import PinnedTransferRing
from benchmarks.async_metadata import measure, busy


def transfer_case(direction, count, work_us, depth, device):
    source_cpu=torch.arange(count,dtype=torch.int64).remainder_(251).to(torch.uint8)
    source_gpu=source_cpu.to(device)
    destination_gpu=torch.empty_like(source_gpu)
    destination_cpu=torch.empty_like(source_cpu)
    init_start=time.perf_counter_ns()
    ring=PinnedTransferRing(count,torch.uint8,device,depth)
    initialization_us=(time.perf_counter_ns()-init_start)/1000

    def blocking():
        for _ in range(depth):
            if direction=='h2d':destination_gpu.copy_(source_cpu,non_blocking=False)
            else:destination_cpu.copy_(source_gpu,non_blocking=False)
            busy(work_us)
        return None

    def immediate():
        for _ in range(depth):
            if direction=='h2d':ring.upload(source_cpu,destination_gpu).wait()
            else:
                ticket=ring.download(source_gpu);ticket.consume(owned=False);ticket.release()
            busy(work_us)
        return None

    def delayed(owned=False):
        tickets=[]
        for _ in range(depth):
            if direction=='h2d':ticket=ring.upload(source_cpu,destination_gpu)
            else:ticket=ring.download(source_gpu)
            busy(work_us);tickets.append(ticket)
        for ticket in tickets:
            if direction=='h2d':ticket.wait()
            else:
                ticket.consume(owned=owned)
                if not owned:ticket.release()
        return None

    # Separate correctness checks are outside all timing populations.
    for fn in [blocking,immediate,delayed]:
        fn();torch.cuda.synchronize()
        if direction=='h2d':assert torch.equal(destination_gpu.cpu(),source_cpu)
        elif fn==blocking:assert torch.equal(destination_cpu,source_cpu)
    methods={'pageable_blocking':blocking,'pinned_immediate':immediate,'pinned_ring':delayed}
    if direction=='d2h':methods['pinned_ring_owned']=lambda:delayed(True)
    return methods,ring.pinned_bytes,initialization_us


def source_case(size,rank,device,direction):
    import sglang.srt.managers.schedule_batch as schedule
    from benchmarks.sglang_prefix_copy import build_case, snapshot
    for key,value in {'attention_backend':'triton','prefill_attention_backend':'triton',
                     'decode_attention_backend':'triton','use_mla_backend':False,
                     'enable_two_batch_overlap':False}.items():schedule.global_server_args_dict[key]=value
    schedule._FUSED_PREFIX_COPY=True
    os.environ['SGLANG_PINNED_INPUTS']='0'
    tree=ast.parse(Path('/workspace/schedule_batch.m2.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScheduleBatch')
    functions={}
    for name in ['prepare_for_extend','get_model_worker_batch']:
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        namespace=dict(vars(schedule));module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
        exec(compile(module,'/workspace/schedule_batch.m2.py','exec'),namespace);functions[name]=namespace[name]
    batch,reset=build_case(size+rank*3,129,device,.5,True,True)
    if direction=='h2d':
        def call(enabled):
            os.environ['SGLANG_PINNED_INPUTS']='1' if enabled else '0'
            reset()
            if enabled:batch.prepare_for_extend()
            else:functions['prepare_for_extend'](batch)
            return None
    else:
        reset();batch.prepare_for_extend()
        schedule.global_server_args_dict['prefill_attention_backend']='fa3'
        def call(enabled):
            os.environ['SGLANG_PINNED_LENGTHS']='1' if enabled else '0'
            worker=batch.get_model_worker_batch() if enabled else functions['get_model_worker_batch'](batch)
            assert worker.seq_lens_cpu is not None
            return None
    return {'upstream':lambda:call(False),'pinned':lambda:call(True)}


def run(a):
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    torch.cuda.set_device(device)
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[min(rank*4,len(allowed)-1)]})
    dist.init_process_group('nccl',device_id=device)
    rows=[]
    if a.kind=='micro':
        cases=[(direction,count,work,depth) for direction in a.directions for count in a.bytes
               for work in a.work_us for depth in a.depths]
    else:cases=[(direction,size) for direction in a.directions for size in a.batches]
    for case in cases:
        if a.kind=='micro':
            direction,count,work,depth=case
            setup=time.perf_counter_ns();methods,pinned_bytes,initialization_us=transfer_case(direction,count,work,depth,device)
            setup_us=(time.perf_counter_ns()-setup)/1000
            settings=dict(direction=direction,bytes=count,independent_cpu_work_us=work,depth=depth,
                          transfers_per_invocation=depth,pinned_bytes_per_rank=pinned_bytes,ring_initialization_us=initialization_us,setup_and_correctness_us=setup_us)
        else:
            direction,size=case;methods=source_case(size,rank,device,direction)
            settings=dict(direction=direction,batch_rank0=size,batch_rank1=size+3)
        order=list(methods)
        if a.reverse:order.reverse()
        values={}
        for mode in order+list(reversed(order)):
            values.setdefault(mode,[]).append(measure(methods[mode],a.warmup,a.samples,rank,device))
        rows.append(dict(**settings,measurements=values))
        if rank==0:
            print(case,{k:round(statistics.median([b['worst_rank']['median_us'] for b in v]),2) for k,v in values.items()},flush=True)
    if rank==0:
        Path(a.output).write_text(json.dumps(dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            kind=a.kind,world_size=2,model_forward=False,warmup=a.warmup,samples_per_block=a.samples,
            cpu_affinity='one distinct logical CPU per rank',
            order='reverse then forward' if a.reverse else 'forward then reverse',
            source_semantics='fixed M1/M2 snapshot plus M3 toggle',rows=rows),indent=2))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['micro','source'],default='micro')
    p.add_argument('--directions',nargs='+',choices=['h2d','d2h'],default=['h2d','d2h'])
    p.add_argument('--bytes',nargs='+',type=int,default=[8,1024,65536,1048576,16777216])
    p.add_argument('--work-us',nargs='+',type=int,default=[0,50,200,1000])
    p.add_argument('--depths',nargs='+',type=int,default=[1,2,4]);p.add_argument('--batches',nargs='+',type=int,default=[1,32,128,512])
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--samples',type=int,default=100)
    p.add_argument('--reverse',action='store_true');p.add_argument('--output',required=True);run(p.parse_args())
