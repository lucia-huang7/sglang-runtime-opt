"""Inclusive ordinary-call and prepared CUDA Graph prefix-copy measurements."""
import argparse
import datetime
import os
import json
import statistics
import time
from pathlib import Path
import torch
from runtime_opt.prefix_copy import copy_prefixes, loop_prefix_copy, prepare_prefix_copy, prepare_prefix_copy_cpp


def make_case(batch, length, device, irregular=False):
    lengths = [max(0, length - (i * 13) % (length + 1)) if irregular else length for i in range(batch)]
    table = torch.full((batch + 4, length + 17), -99, dtype=torch.int32, device=device)
    sources = [torch.arange(n, dtype=torch.int64, device=device) + i * 100003 for i, n in enumerate(lengths)]
    return table, list(range(2, batch + 2)), sources


def measure(fn, warmup, samples):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    wall, events = [], []
    for _ in range(samples):
        t = time.perf_counter_ns()
        fn(); torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - t) / 1000)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record(); fn(); stop.record(); stop.synchronize()
        events.append(start.elapsed_time(stop) * 1000)
    return {'wall_us': wall, 'event_us': events, 'median_wall_us': statistics.median(wall),
            'p95_wall_us': sorted(wall)[int(.95 * (len(wall) - 1))],
            'median_event_us': statistics.median(events)}


def run(args):
    torch.cuda.set_device(args.device)
    device = f'cuda:{args.device}'
    rows = []
    for batch in args.batches:
        for length in args.lengths:
            table, row_ids, sources = make_case(batch, length, device, args.irregular)
            baseline = lambda: loop_prefix_copy(table, row_ids, sources)
            fused = lambda: copy_prefixes(table, row_ids, sources, min_segments=0)
            baseline(); reference = table.clone(); table.fill_(-99); fused()
            torch.testing.assert_close(table, reference, rtol=0, atol=0)
            variants = {'loop': baseline, 'fused_inclusive': fused}
            if args.cpp_comparison:
                variants['fused_python_descriptors'] = lambda: copy_prefixes(table, row_ids, sources, min_segments=0, backend='python')
            if args.foreach:
                variants['foreach'] = lambda: torch._foreach_copy_([table[r, :len(x)] for r, x in zip(row_ids, sources)], sources)
            if args.ablation:
                plan = prepare_prefix_copy(table, row_ids, sources)
                variants['descriptor_only'] = lambda: prepare_prefix_copy_cpp(table, row_ids, sources)
                variants['fused_prepared'] = lambda: plan.launch(table)
            if args.graph:
                plan = prepare_prefix_copy(table, row_ids, sources)
                variants = {'loop': baseline, 'fused_prepared': lambda: plan.launch(table)}
                graphs = {}
                for name, fn in variants.items():
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(5): fn()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph): fn()
                    graphs[name] = graph
                variants = {name: graph.replay for name, graph in graphs.items()}
            measurements = {}
            order = list(variants) if args.order == 'AB' else list(reversed(variants))
            for name in order + list(reversed(order)):
                block = measure(variants[name], args.warmup, args.samples)
                if name not in measurements:
                    measurements[name] = block
                    measurements[name]['block_medians_us'] = [block['median_wall_us']]
                else:
                    target = measurements[name]
                    target['wall_us'].extend(block['wall_us'])
                    target['event_us'].extend(block['event_us'])
                    target['block_medians_us'].append(block['median_wall_us'])
                    target['median_wall_us'] = statistics.median(target['wall_us'])
                    target['p95_wall_us'] = sorted(target['wall_us'])[int(.95 * (len(target['wall_us']) - 1))]
                    target['median_event_us'] = statistics.median(target['event_us'])
            row = {'batch': batch, 'max_length': length, 'irregular': args.irregular, 'measurements': measurements}
            rows.append(row)
            print(batch, length, {k: round(v['median_wall_us'], 2) for k, v in measurements.items()}, flush=True)
    report = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'order_policy': 'forward then reverse',
              'device': args.device, 'mode': 'prepared_graph' if args.graph else 'ordinary_inclusive',
              'samples': args.samples, 'warmup': args.warmup, 'order': args.order,
              'torch': torch.__version__, 'rows': rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32, 128, 512])
    p.add_argument('--lengths', type=int, nargs='+', default=[1, 8, 32, 128, 512])
    p.add_argument('--samples', type=int, default=200)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--order', choices=['AB', 'BA'], default='AB')
    p.add_argument('--irregular', action='store_true')
    p.add_argument('--graph', action='store_true')
    p.add_argument('--ablation', action='store_true')
    p.add_argument('--foreach', action='store_true')
    p.add_argument('--cpp-comparison', action='store_true')
    p.add_argument('--output', required=True)
    run(p.parse_args())
