"""Validate the patched scheduler with real upstream request and pool classes."""
import argparse
import json
from pathlib import Path
import torch
import sglang.srt.managers.schedule_batch as schedule
from benchmarks.sglang_prefix_copy import build_case, original_method, snapshot


def run(source):
    schedule.global_server_args_dict['attention_backend'] = 'triton'
    schedule._FUSED_PREFIX_COPY = True
    baseline = original_method(source)
    optimized = schedule.ScheduleBatch.prepare_for_extend
    cases = []
    for device in range(2):
        torch.cuda.set_device(device)
        for batch_size in [1, 8, 32, 128]:
            for hit_rate in [0.0, 0.25, 1.0]:
                for logprob in [False, True]:
                    batch, reset = build_case(batch_size, 257, f'cuda:{device}', hit_rate, True, logprob)
                    reset(); baseline(batch); expected, expected_states = snapshot(batch)
                    reset(); batch.req_to_token_pool.req_to_token.fill_(-99); optimized(batch)
                    actual, states = snapshot(batch)
                    parts = [r.prefix_indices for r in batch.reqs if len(r.prefix_indices)]
                    cached = torch.cat(parts) if parts else None
                    if cached is not None:
                        assert cached.max().item() < batch.out_cache_loc.min().item()
                    for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    assert states == expected_states
                    cases.append({'device': device, 'batch': batch_size, 'hit_rate': hit_rate, 'logprob': logprob, 'status': 'passed'})

    class ObservingSWA(schedule.SWAChunkCache):
        def __init__(self, pool):
            self.pool, self.observed = pool, []
        def evict_swa(self, req, prefix_length, chunk_size):
            torch.testing.assert_close(self.pool.req_to_token[req.req_pool_idx, :prefix_length],
                                       req.prefix_indices.to(torch.int32), rtol=0, atol=0)
            self.observed.append(req.req_pool_idx)

    batch, reset = build_case(32, 257, 'cuda:1')
    batch.tree_cache = ObservingSWA(batch.req_to_token_pool)
    batch.model_config.attention_chunk_size = 128
    original_fused = schedule.try_fused_prefix_copy
    def forbidden(*args, **kwargs): raise AssertionError('SWA must retain its original ordered path')
    schedule.try_fused_prefix_copy = forbidden
    reset(); optimized(batch)
    assert batch.tree_cache.observed == list(range(32))
    schedule.try_fused_prefix_copy = original_fused
    cases.append({'case': 'SWA copy-before-eviction ordering', 'status': 'passed'})

    batch, reset = build_case(32, 257, 'cuda:1')
    for i, req in enumerate(batch.reqs):
        req.prefix_indices = (torch.arange(514, dtype=torch.int64, device='cuda:1') // 2 + i * 257 + 1)[::2]
    reset(); baseline(batch); expected, expected_states = snapshot(batch)
    reset(); batch.req_to_token_pool.req_to_token.fill_(-99); optimized(batch)
    actual, states = snapshot(batch)
    for key in expected: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert states == expected_states
    cases.append({'case': 'noncontiguous-prefix fallback', 'status': 'passed'})
    return {'fixture_version': 'reserved_cached_slots_v2', 'case_count': len(cases), 'cases': cases, 'model_forward': False,
            'actual_upstream_classes': ['Req', 'ScheduleBatch', 'ReqToTokenPool', 'TokenToKVPoolAllocator', 'SamplingBatchInfo']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original-source', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = run(args.original_source)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print('Scheduler validation passed:', result['case_count'], 'cases')
