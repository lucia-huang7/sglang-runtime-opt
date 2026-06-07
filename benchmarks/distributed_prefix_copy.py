"""Measure the worst-rank scheduler-preparation time on two concurrent GPU ranks."""
import argparse
import datetime
import json
import os
import statistics
import time
from pathlib import Path
import torch
import torch.distributed as dist
import sglang.srt.managers.schedule_batch as schedule
from benchmarks.sglang_prefix_copy import build_case, original_method, snapshot


def run(args):
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    os.sched_setaffinity(0, [4 + rank * 4])
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
    assert dist.get_world_size() == 2
    schedule.global_server_args_dict['attention_backend'] = 'triton'
    schedule._FUSED_PREFIX_COPY = True
    baseline = original_method(args.original_source)
    optimized = schedule.ScheduleBatch.prepare_for_extend
    rows = []
    for size in [8, 32, 128, 512]:
        batch, reset = build_case(size, 512, f'cuda:{rank}')
        reset(); baseline(batch); expected, expected_states = snapshot(batch)
        reset(); batch.req_to_token_pool.req_to_token.fill_(-99); optimized(batch)
        actual, states = snapshot(batch)
        for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert states == expected_states
        functions = {'upstream': baseline, 'fused': optimized}
        measurements = {name: [] for name in functions}
        order = ['upstream', 'fused', 'fused', 'upstream']
        if args.reverse: order = ['fused', 'upstream', 'upstream', 'fused']
        for name in order:
            fn = functions[name]
            for _ in range(args.warmup): reset(); fn(batch)
            torch.cuda.synchronize(); dist.barrier()
            local_samples = []
            for _ in range(args.samples):
                reset(); dist.barrier()
                start = time.perf_counter_ns()
                fn(batch); torch.cuda.synchronize()
                local_samples.append((time.perf_counter_ns() - start) / 1000)
            local = torch.tensor(local_samples, dtype=torch.float64, device=f'cuda:{rank}')
            gathered = [torch.empty_like(local) for _ in range(2)]
            dist.all_gather(gathered, local)
            worst = torch.stack(gathered).amax(0).cpu().tolist()
            measurements[name].extend(worst)
        result = {name: {'samples_us': values, 'median_us': statistics.median(values),
                        'p95_us': sorted(values)[int(.95 * (len(values) - 1))]}
                  for name, values in measurements.items()}
        rows.append({'batch': size, 'max_length': 512, 'measurements': result})
        if rank == 0: print(size, {k: round(v['median_us'], 2) for k, v in result.items()}, flush=True)
    if rank == 0:
        Path(args.output).write_text(json.dumps({'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'mode': 'concurrent_two_rank_prepare_for_extend', 'aggregation': 'maximum rank duration for each matched invocation',
            'synchronization': 'NCCL barrier before each timed scheduler call; barrier excluded',
            'samples_per_block': args.samples, 'warmup': args.warmup, 'model_forward': False, 'rows': rows}, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original-source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--reverse', action='store_true')
    run(p.parse_args())
