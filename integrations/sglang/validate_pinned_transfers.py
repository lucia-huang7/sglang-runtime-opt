"""Validate bounded transfers and actual scheduler inputs/CPU-length ownership."""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from runtime_opt.pinned_transfers import PinnedTransferRing, pack_scheduler_inputs


def extract(name,schedule,path):
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScheduleBatch')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(vars(schedule));module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    exec(compile(module,path,'exec'),namespace);return namespace[name]


def main(a):
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=device)
    rows=[]
    for dtype in [torch.uint8,torch.int64,torch.float32]:
        for depth in [1,2,4]:
            ring=PinnedTransferRing(1025,dtype,device,depth)
            for n in [0,1,127,1025]:
                source=(torch.arange(n)%89).to(dtype)
                ticket=ring.upload(source)
                source.fill_(0)
                expected=(torch.arange(n)%89).to(dtype)
                torch.testing.assert_close(ticket.wait().cpu(),expected,rtol=0,atol=0)
                gpu=expected.to(device)
                transfer=ring.download(gpu)
                gpu.fill_(17)
                result=transfer.consume(owned=True)
                torch.testing.assert_close(result,expected,rtol=0,atol=0)
                held=result.clone()
                for repeat in range(depth+2):
                    item=ring.download(torch.full((n,),repeat,dtype=dtype,device=device))
                    item.consume(owned=True)
                torch.testing.assert_close(result,held,rtol=0,atol=0)
            ring.close();rows.append(dict(case='typed sizes, CPU producer mutation, GPU producer mutation and owned lifetime',dtype=str(dtype),depth=depth,status='passed'))
    ring=PinnedTransferRing(8,torch.int64,device,1)
    ticket=ring.download(torch.arange(8,device=device));borrowed=ticket.consume(owned=False)
    try:ring.download(torch.arange(8,device=device))
    except BufferError:pass
    else:raise AssertionError('Borrowed CPU view was overwritten')
    try:ring.close()
    except RuntimeError:pass
    else:raise AssertionError('Unreleased CPU lease accepted at close')
    torch.testing.assert_close(borrowed,torch.arange(8),rtol=0,atol=0)
    ticket.release();ring.close()
    rows.append(dict(case='borrowed-view backpressure and close guards',status='passed'))
    ring=PinnedTransferRing(8,torch.int64,device,1)
    ring.slots[0].host.fill_(-77)
    source=torch.arange(8,device=device)
    torch.cuda._sleep(100_000_000)
    ticket=ring.download(source)
    assert ticket.slot.host[0].item()==-77
    torch.testing.assert_close(ticket.consume(),torch.arange(8),rtol=0,atol=0)
    ring.close();rows.append(dict(case='premature-read fault injection behind delayed producer',status='passed'))
    stream=torch.cuda.Stream(device=device)
    ring=PinnedTransferRing(129,torch.int64,device,2)
    with torch.cuda.stream(stream):
        x=torch.arange(129,device=device)
        ticket=ring.download(x);torch.testing.assert_close(ticket.consume(),torch.arange(129),rtol=0,atol=0)
        uploaded=ring.upload(torch.arange(129)).result
        check=uploaded+3
    stream.synchronize();torch.testing.assert_close(check.cpu(),torch.arange(129)+3,rtol=0,atol=0)
    ring.close();rows.append(dict(case='nondefault producer and consumer stream',status='passed'))
    ring=PinnedTransferRing(8,torch.int64,device,1)
    for invalid in [torch.zeros(9,dtype=torch.int64),torch.zeros(8,dtype=torch.float32),torch.zeros(16,dtype=torch.int64)[::2]]:
        try:ring.upload(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid transfer accepted')
    ring.close();rows.append(dict(case='capacity, dtype and layout guards',status='passed'))
    for depth in [1,2,4]:
        count=16*1024*1024
        source=(torch.arange(count,dtype=torch.int64)%251).to(torch.uint8)
        ring=PinnedTransferRing(count,torch.uint8,device,depth)
        uploaded=ring.upload(source).wait()
        torch.testing.assert_close(uploaded.cpu(),source,rtol=0,atol=0)
        ticket=ring.download(uploaded)
        torch.testing.assert_close(ticket.consume(owned=True),source,rtol=0,atol=0)
        ring.close();rows.append(dict(case='16 MiB H2D and D2H exact payload',depth=depth,status='passed'))
    implicit=PinnedTransferRing(8,torch.int64,'cuda',1)
    destination=torch.empty(8,dtype=torch.int64,device=device)
    implicit.upload(torch.arange(8),destination).wait()
    torch.testing.assert_close(implicit.download(destination).consume(),torch.arange(8),rtol=0,atol=0)
    implicit.close();rows.append(dict(case='implicit CUDA device resolves to caller device',status='passed'))
    held=pack_scheduler_inputs([[1],[2,3],[4],[5]],device)
    assert all(t.data_ptr()%256==0 for t in held)
    for _ in range(8):pack_scheduler_inputs([[11],[12,13],[14],[15]],device)
    torch.cuda.synchronize()
    assert [t.cpu().tolist() for t in held]==[[1],[2,3],[4],[5]]
    held[2].fill_(99);assert held[1].cpu().tolist()==[2,3]
    rows.append(dict(case='packed device ownership and disjoint view mutation',status='passed'))

    import sglang.srt.managers.schedule_batch as schedule
    from benchmarks.sglang_prefix_copy import build_case,snapshot
    for key,value in {'attention_backend':'triton','prefill_attention_backend':'triton',
                     'decode_attention_backend':'triton','use_mla_backend':False,
                     'enable_two_batch_overlap':False}.items():schedule.global_server_args_dict[key]=value
    schedule._FUSED_PREFIX_COPY=True
    baseline=extract('prepare_for_extend',schedule,a.original_source)
    original_worker=extract('get_model_worker_batch',schedule,a.original_source)
    for size in [1,8,32,128]:
        for hit in [0.0,.5,1.0]:
            for logprob in [False,True]:
                batch,reset=build_case(size+rank,129,device,hit,True,logprob)
                reset();baseline(batch);expected,state=snapshot(batch)
                os.environ['SGLANG_PINNED_INPUTS']='1'
                reset();batch.prepare_for_extend();actual,actual_state=snapshot(batch)
                for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
                assert state==actual_state
                rows.append(dict(case='actual prepare_for_extend',batch=size,hit_rate=hit,logprob=logprob,status='passed'))
    for size in [1,32,128,512]:
        for decode in [False,True]:
            batch,reset=build_case(size+rank,33,device,.5,True,True);reset();batch.prepare_for_extend()
            if decode:batch.forward_mode=schedule.ForwardMode.DECODE
            for key in ['prefill_attention_backend','decode_attention_backend']:schedule.global_server_args_dict[key]='fa3'
            baseline_worker=original_worker(batch)
            os.environ['SGLANG_PINNED_LENGTHS']='1'
            worker=batch.get_model_worker_batch()
            torch.testing.assert_close(worker.seq_lens_cpu,baseline_worker.seq_lens_cpu,rtol=0,atol=0)
            saved=worker.seq_lens_cpu.clone()
            for i in range(5):
                batch.seq_lens.add_(1);other=batch.get_model_worker_batch()
                torch.testing.assert_close(other.seq_lens_cpu,batch.seq_lens.cpu(),rtol=0,atol=0)
            torch.testing.assert_close(worker.seq_lens_cpu,saved,rtol=0,atol=0)
            explicit=torch.full_like(saved,7)
            assert batch.get_model_worker_batch(seq_lens_cpu_cache=explicit).seq_lens_cpu is explicit
            rows.append(dict(case='actual worker owned CPU lengths and explicit-cache bypass',batch=size,decode=decode,status='passed'))
    original_class=schedule.ModelWorkerBatch
    def fail_descriptor(**kwargs):raise RuntimeError('Injected descriptor failure')
    schedule.ModelWorkerBatch=fail_descriptor
    try:
        try:batch.get_model_worker_batch()
        except RuntimeError as error:assert 'Injected descriptor failure' in str(error)
        else:raise AssertionError('Descriptor failure was not injected')
    finally:schedule.ModelWorkerBatch=original_class
    import runtime_opt.pinned_transfers as transfers
    assert all(not slot.leased for ring in transfers._LENGTH_RINGS.values() for slot in ring.slots)
    rows.append(dict(case='descriptor exception drains prefetched D2H lease',status='passed'))
    if rank==0:
        Path(a.output).write_text(json.dumps(dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),world_size=2,case_count=len(rows),model_forward=False,cases=rows),indent=2))
        print('Pinned transfer validation passed:',len(rows),'case groups',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-source',default='/workspace/schedule_batch.m2.py');p.add_argument('--output',required=True);main(p.parse_args())
