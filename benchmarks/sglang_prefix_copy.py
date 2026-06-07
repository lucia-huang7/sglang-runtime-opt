"""Run the actual pinned SGLang scheduler method with real request/pool classes.

This benchmark prepares scheduler metadata without loading a language model.
The token allocator stores indices; no attention or model forward is executed.
"""
import argparse
import datetime
import os
import ast
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import sglang.srt.managers.schedule_batch as schedule
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams


def original_method(path):
    text = Path(path).read_text()
    node = next(n for c in ast.parse(text).body if isinstance(c, ast.ClassDef) and c.name == 'ScheduleBatch'
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_for_extend')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(schedule))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['prepare_for_extend']


def build_case(batch_size, length, device, hit_rate=1.0, irregular=False, logprob=False):
    pool = ReqToTokenPool(batch_size + 4, length + 21, device, False)
    pool.req_to_token.fill_(-99)
    allocator = TokenToKVPoolAllocator(batch_size * (length + 8) + 100, torch.int64, device, None)
    reserved_prefix_slots = allocator.alloc(batch_size * length)
    assert reserved_prefix_slots is not None
    initial_pages = allocator.free_pages
    requests = []
    for i in range(batch_size):
        n = length if i < int(batch_size * hit_rate) else 0
        if irregular and n: n = max(1, n - (i * 13) % n)
        params = SamplingParams(temperature=1.0, top_k=1, max_new_tokens=8)
        req = schedule.Req(str(i), None, [1] * (n + 4), params, return_logprob=logprob)
        req.fill_ids = req.origin_input_ids.copy()
        req.prefix_indices = torch.arange(n, device=device, dtype=torch.int64) + i * length + 1 if n else []
        req.extend_input_len = 4
        req.logprob_start_len = -1
        requests.append(req)
    batch = schedule.ScheduleBatch(reqs=requests, req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator, model_config=SimpleNamespace(vocab_size=32000, is_encoder_decoder=False),
        device=device, return_logprob=logprob, enable_custom_logit_processor=False)
    def reset():
        pool.free_slots = list(range(pool.size))
        allocator.free_pages = initial_pages
        for req in requests:
            req.cached_tokens = 0
            req.already_computed = 0
    return batch, reset


def snapshot(batch):
    tensors = {name: getattr(batch, name).clone() for name in
               ['input_ids', 'req_pool_indices', 'seq_lens', 'out_cache_loc']}
    tensors['table'] = batch.req_to_token_pool.req_to_token.clone()
    tensors['temperatures'] = batch.sampling_info.temperatures.clone()
    if batch.extend_input_logprob_token_ids is not None:
        tensors['logprob_tokens'] = batch.extend_input_logprob_token_ids.clone()
    states = [(r.cached_tokens, r.already_computed, r.extend_logprob_start_len, r.req_pool_idx) for r in batch.reqs]
    return tensors, states


def measure(fn, batch, reset, warmup, samples):
    for _ in range(warmup): reset(); fn(batch)
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        reset()
        start = time.perf_counter_ns()
        fn(batch); torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / 1000)
    return {'samples_us': times, 'median_us': statistics.median(times),
            'p95_us': sorted(times)[int(.95 * (len(times) - 1))]}


def run(args):
    torch.cuda.set_device(args.device)
    schedule.global_server_args_dict['attention_backend'] = 'triton'
    schedule._FUSED_PREFIX_COPY = True
    baseline = original_method(args.original_source)
    optimized = schedule.ScheduleBatch.prepare_for_extend
    rows = []
    for b in args.batches:
        for length in args.lengths:
            batch, reset = build_case(b, length, f'cuda:{args.device}', args.hit_rate, args.irregular, args.logprob)
            reset(); baseline(batch); expected, expected_states = snapshot(batch)
            reset(); batch.req_to_token_pool.req_to_token.fill_(-99); optimized(batch)
            actual, states = snapshot(batch)
            for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            assert states == expected_states
            variants = {'upstream': baseline, 'fused': optimized}
            order = list(variants) if args.order == 'AB' else list(reversed(variants))
            values = {}
            for name in order + list(reversed(order)):
                block = measure(variants[name], batch, reset, args.warmup, args.samples)
                if name not in values:
                    values[name] = block
                    values[name]['block_medians_us'] = [block['median_us']]
                else:
                    target = values[name]
                    target['samples_us'].extend(block['samples_us'])
                    target['block_medians_us'].append(block['median_us'])
                    target['median_us'] = statistics.median(target['samples_us'])
                    target['p95_us'] = sorted(target['samples_us'])[int(.95 * (len(target['samples_us']) - 1))]
            row = {'batch': b, 'max_length': length, 'hit_rate': args.hit_rate,
                   'irregular': args.irregular, 'logprob': args.logprob, 'measurements': values, 'correctness': 'exact'}
            rows.append(row)
            print(b, length, {k: round(v['median_us'], 2) for k, v in values.items()}, flush=True)
    Path(args.output).write_text(json.dumps({'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'order_policy': 'ABBA or BAAB', 'fixture_version': 'reserved_cached_slots_v2', 'mode': 'actual_prepare_for_extend', 'device': args.device,
        'order': args.order, 'samples': args.samples, 'warmup': args.warmup,
        'upstream_source_sha256': hashlib.sha256(Path(args.original_source).read_bytes()).hexdigest(),
        'model_forward': False, 'rows': rows}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original-source', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--batches', nargs='+', type=int, default=[1, 8, 32, 128, 512])
    p.add_argument('--lengths', nargs='+', type=int, default=[32, 512])
    p.add_argument('--hit-rate', type=float, default=1.0)
    p.add_argument('--irregular', action='store_true')
    p.add_argument('--logprob', action='store_true')
    p.add_argument('--samples', type=int, default=200)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--order', choices=['AB', 'BA'], default='AB')
    p.add_argument('--output', required=True)
    run(p.parse_args())
