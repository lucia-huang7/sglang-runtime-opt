"""Two-rank metadata and actual SGLang descriptor-preparation experiments."""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
import torch.distributed as dist
from runtime_opt.async_metadata import MetadataGather


def source_method(path, namespace):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_mlp_sync_batch_raw')
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    env = dict(namespace)
    exec(compile(module, str(path), 'exec'), env)
    return env[node.name]


def busy(us):
    end = time.perf_counter_ns() + us * 1000
    while time.perf_counter_ns() < end:
        pass


def metric(samples):
    return {'samples_us': samples, 'median_us': statistics.median(samples),
            'p95_us': sorted(samples)[int(.95 * (len(samples) - 1))]}


def measure(fn, warmup, samples, rank, device):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    dist.barrier()
    rows = []
    waits = []
    for _ in range(samples):
        dist.barrier()
        start = time.perf_counter_ns()
        wait = fn()
        torch.cuda.synchronize()
        rows.append((time.perf_counter_ns() - start) / 1000)
        waits.append(wait)
    # Match invocations before taking the worst rank; barriers are outside timings.
    reports = [None for _ in range(2)]
    dist.all_gather_object(reports, {'total': rows, 'wait': waits})
    worst = [max(r['total'][i] for r in reports) for i in range(samples)]
    return {'worst_rank': metric(worst), 'ranks': [metric(r['total']) for r in reports],
            'consumer_wait_ranks': [metric(r['wait']) for r in reports] if waits[0] is not None else None}


def micro_case(width, work_us, backend, rank, group, device):
    gpu = backend == 'nccl'
    dev = device if gpu else 'cpu'
    values = [rank + i * 7 for i in range(width)]
    output = torch.empty(2 * width, dtype=torch.int64, device=dev)
    pending = MetadataGather(width, group, dev)
    expected = [[r + i * 7 for i in range(width)] for r in range(2)]

    def blocking():
        begin = time.perf_counter_ns()
        source = torch.tensor(values, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(output, source, group=group)
        result = output.cpu().view(2, width).tolist()
        wait = (time.perf_counter_ns() - begin) / 1000
        busy(work_us)
        assert result == expected
        return wait

    def immediate():
        pending.start(values)
        begin = time.perf_counter_ns(); result = pending.finish().tolist()
        wait = (time.perf_counter_ns() - begin) / 1000
        busy(work_us)
        assert result == expected
        return wait

    def delayed():
        pending.start(values)
        busy(work_us)
        begin = time.perf_counter_ns(); result = pending.finish().tolist()
        wait = (time.perf_counter_ns() - begin) / 1000
        assert result == expected
        return wait
    return {'blocking': blocking, 'immediate': immediate, 'overlap': delayed}


def real_case(size, rank, device, groups, original):
    import sglang.srt.managers.schedule_batch as schedule
    import sglang.srt.managers.scheduler as scheduler
    from benchmarks.sglang_prefix_copy import build_case
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    for key, value in {'attention_backend': 'triton', 'prefill_attention_backend': 'triton',
                       'decode_attention_backend': 'triton', 'use_mla_backend': False,
                       'enable_two_batch_overlap': False}.items():
        schedule.global_server_args_dict[key] = value
    batch, reset = build_case(size + rank * 3, 33, device, 1.0, True, True)
    reset(); batch.prepare_for_extend(); torch.cuda.synchronize()
    group = SimpleNamespace(device_group=groups['nccl'], cpu_group=groups['gloo'], device=device)
    kwargs = dict(dp_size=2, attn_tp_size=1, tp_group=group, get_idle_batch=lambda: None,
        disable_cuda_graph=False, spec_algorithm=SpeculativeAlgorithm.NONE,
        speculative_num_draft_tokens=0, enable_two_batch_overlap=False, enable_deepep_moe=False,
        deepep_mode=scheduler.DeepEPMode.normal, require_mlp_tp_gather=True)
    patched = scheduler.Scheduler.prepare_mlp_sync_batch_raw
    baseline = source_method(original, vars(scheduler))
    expected = None
    def call(mode):
        nonlocal expected
        batch._pending_worker_descriptor = None
        if mode == 'upstream_gloo':
            result = baseline(batch, **kwargs, disable_overlap_schedule=False)
        elif mode == 'upstream_nccl':
            result = baseline(batch, **kwargs, disable_overlap_schedule=True)
        else:
            result = patched(batch, **kwargs, disable_overlap_schedule=True,
                             async_metadata_mode=mode)
        worker = result.get_model_worker_batch()
        fields = {name: getattr(worker, name) for name in
                  ['global_num_tokens', 'global_num_tokens_for_logprob', 'is_extend_in_batch',
                   'can_run_dp_cuda_graph', 'tbo_split_seq_index', 'global_forward_mode']}
        if expected is None: expected = fields
        else: assert fields == expected, (mode, fields, expected)
        assert worker.input_ids is batch.input_ids
        assert worker.sampling_info is batch.sampling_info
        assert batch._pending_worker_descriptor is None
        return None
    return {mode: lambda mode=mode: call(mode) for mode in
            ['upstream_gloo', 'upstream_nccl', 'immediate', 'overlap']}


def run(a):
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    if hasattr(os, 'sched_setaffinity'):
        allowed = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, {allowed[min(rank * 4, len(allowed)-1)]})
    dist.init_process_group('nccl', device_id=device)
    gloo = dist.new_group(backend='gloo')
    groups = {'nccl': dist.group.WORLD, 'gloo': gloo}
    rows = []
    if a.kind == 'micro':
        cases = [(w, us, backend) for w in a.widths for us in a.work_us for backend in ['nccl', 'gloo']]
        for width, work_us, backend in cases:
            methods = micro_case(width, work_us, backend, rank, groups[backend], device)
            order = list(methods)
            if a.reverse: order.reverse()
            measurements = {}
            for mode in order + list(reversed(order)):
                block = measure(methods[mode], a.warmup, a.samples, rank, device)
                measurements.setdefault(mode, []).append(block)
            row = dict(width=width, independent_cpu_work_us=work_us, backend=backend,
                       measurements=measurements)
            rows.append(row)
            if rank == 0: print(width, work_us, backend, {k:round(statistics.median([b['worst_rank']['median_us'] for b in v]),2) for k,v in measurements.items()}, flush=True)
    else:
        for size in a.batches:
            methods = real_case(size, rank, device, groups, a.original_source)
            order = list(methods)
            if a.reverse: order.reverse()
            measurements = {}
            for mode in order + list(reversed(order)):
                measurements.setdefault(mode, []).append(measure(methods[mode], a.warmup, a.samples, rank, device))
            rows.append(dict(batch=size, measurements=measurements))
            if rank == 0: print('source', size, {k:round(statistics.median([b['worst_rank']['median_us'] for b in v]),2) for k,v in measurements.items()}, flush=True)
    if rank == 0:
        Path(a.output).write_text(json.dumps(dict(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            kind=a.kind, original_source=a.original_source, warmup=a.warmup, samples_per_block=a.samples,
            world_size=2, order='forward then reverse' if not a.reverse else 'reverse then forward',
            cpu_affinity='one distinct logical CPU per rank', model_forward=False, rows=rows), indent=2))
    dist.destroy_process_group(gloo); dist.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind', choices=['micro','source'], default='micro')
    p.add_argument('--widths', nargs='+', type=int, default=[1,6])
    p.add_argument('--work-us', nargs='+', type=int, default=[0,50,200,1000])
    p.add_argument('--batches', nargs='+', type=int, default=[1,32,128,512])
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--original-source', default='/workspace/scheduler.original.py')
    p.add_argument('--output', required=True)
    run(p.parse_args())
