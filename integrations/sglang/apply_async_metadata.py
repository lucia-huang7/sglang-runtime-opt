"""Apply the opt-in metadata patch to the pinned installed SGLang source."""
import argparse
import difflib
import hashlib
from pathlib import Path


def apply(root, output):
    root, output = Path(root), Path(output)
    scheduler = root / 'srt/managers/scheduler.py'
    batch = root / 'srt/managers/schedule_batch.py'
    original_scheduler, original_batch = scheduler.read_text(), batch.read_text()
    if 'SGLANG_ASYNC_METADATA' in original_scheduler:
        raise RuntimeError('Metadata patch already applied')
    if hashlib.sha256(original_scheduler.encode()).hexdigest() != '00901b2a33faac178fa5be225680fd5fef7dcfaf9c39ac54052dce5255c0f765':
        raise RuntimeError("Scheduler source does not match the pinned upstream revision")
    s = original_scheduler
    needle = '            disable_overlap_schedule=self.server_args.disable_overlap_schedule,\n'
    assert s.count(needle) == 1
    s = s.replace(needle, needle + '''            async_metadata_mode=(
                os.getenv("SGLANG_ASYNC_METADATA")
                if self.server_args.disable_overlap_schedule
                and self.spec_algorithm.is_none() and self.is_generation
                and self.pp_size == 1 and not self.server_args.enable_two_batch_overlap
                else None
            ),
''')
    needle = '        disable_overlap_schedule: bool,\n    ):\n'
    assert s.count(needle) == 1
    s = s.replace(needle, '        disable_overlap_schedule: bool,\n        async_metadata_mode=None,\n    ):\n')
    first = s.index('        local_info = torch.tensor(', s.index('    def prepare_mlp_sync_batch_raw('))
    last = s.index('        global_num_tokens = ', first)
    old = s[first:last]
    values_first = old.index('            [')
    values_last = old.index('            ],') + len('            ]')
    values = old[values_first:values_last]
    new = '        local_values = ' + values.strip() + '\n'
    new += '''        descriptor = None
        if async_metadata_mode in ("immediate", "overlap") and disable_overlap_schedule:
            from runtime_opt.async_metadata import (
                MetadataGather, prebuild_worker_batch,
            )
            key = (str(device), len(local_values))
            cache = getattr(tp_group, "_runtime_metadata_gathers", None)
            if cache is None:
                cache = tp_group._runtime_metadata_gathers = {}
            if key not in cache:
                cache[key] = MetadataGather(len(local_values), group, device)
            pending = cache[key].start(local_values)
            try:
                if async_metadata_mode == "overlap":
                    descriptor = prebuild_worker_batch(local_batch)
            finally:
                # Drain the collective even if local descriptor construction fails.
                global_info = pending.finish().view(dp_size, attn_tp_size, 6)
        else:
            local_info = torch.tensor(local_values, dtype=torch.int64, device=device)
            global_info = torch.empty((dp_size, attn_tp_size, 6),
                                      dtype=torch.int64, device=device)
            torch.distributed.all_gather_into_tensor(
                global_info.flatten(), local_info, group=group,
            )
'''
    s = s[:first] + new + s[last:]
    pos = s.index('        return local_batch', first)
    s = s[:pos] + '''        if descriptor is not None:
            from runtime_opt.async_metadata import publish_worker_batch
            publish_worker_batch(local_batch, descriptor)

''' + s[pos:]
    b = original_batch
    needle = '    ) -> ModelWorkerBatch:\n'
    assert b.count(needle) == 1
    b = b.replace(needle, needle + '''        descriptor = getattr(self, "_pending_worker_descriptor", None)
        if descriptor is not None:
            if seq_lens_cpu_cache is not None:
                raise RuntimeError("Cached descriptor does not accept a CPU-length override")
            self._pending_worker_descriptor = None
            return descriptor
''')
    compile(s, str(scheduler), 'exec'); compile(b, str(batch), 'exec')
    output.mkdir(parents=True, exist_ok=True)
    diff = []
    for name, old_text, new_text in [('scheduler.py', original_scheduler, s),
                                     ('schedule_batch.py', original_batch, b)]:
        diff.extend(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                    fromfile='a/python/sglang/srt/managers/' + name,
                    tofile='b/python/sglang/srt/managers/' + name))
    (output / 'async-metadata.patch').write_text(''.join(diff))
    (output / 'async-metadata-source.sha256').write_text(
        hashlib.sha256(original_scheduler.encode()).hexdigest() + '  scheduler.py\n')
    scheduler.write_text(s); batch.write_text(b)
    print('Patched scheduler metadata exchange and descriptor handoff')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); apply(a.source_root, a.output)
