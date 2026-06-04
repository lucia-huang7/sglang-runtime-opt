"""Fuse independent cached-prefix writes into an int32 request mapping table."""
from dataclasses import dataclass
from typing import Sequence
import torch
import triton
import triton.language as tl


@triton.jit
def _prefix_copy(Table, Descriptors, TABLE_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    segment = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    address = tl.load(Descriptors + segment * 4)
    row = tl.load(Descriptors + segment * 4 + 1)
    length = tl.load(Descriptors + segment * 4 + 2)
    is_int64 = tl.load(Descriptors + segment * 4 + 3)
    if is_int64:
        source64 = address.to(tl.pointer_type(tl.int64))
        value = tl.load(source64 + offset, offset < length, other=0).to(tl.int32)
    else:
        source32 = address.to(tl.pointer_type(tl.int32))
        value = tl.load(source32 + offset, offset < length, other=0)
    tl.store(Table + row * TABLE_STRIDE + offset, value, offset < length)


@dataclass
class PrefixCopyPlan:
    descriptors: torch.Tensor
    sources: tuple
    max_length: int
    segment_count: int

    def launch(self, table):
        if self.segment_count:
            with torch.cuda.device(table.device):
                _prefix_copy[(self.segment_count, triton.cdiv(self.max_length, 256))](
                    table, self.descriptors, table.stride(0), 256, num_warps=4
                )


def prepare_prefix_copy(table: torch.Tensor, rows: Sequence[int], sources: Sequence[torch.Tensor], min_segments=0):
    """Return a guarded plan, or None when ordinary PyTorch assignment is required.

    Plan creation includes a fresh pinned descriptor tensor and its H2D transfer.
    All source owners remain attached to the plan, with allocator stream tracking.
    Sources must use PyTorch CUDA storage; callers establish producer dependencies.
    Descriptor preparation falls back during capture. Prepared plans support capture.
    Distinct destination rows are required. Source/table aliasing uses the fallback.
    """
    if table.ndim != 2 or table.dtype != torch.int32 or not table.is_cuda or table.stride(1) != 1 or table.stride(0) < table.shape[1]:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    if len(rows) != len(sources) or len(set(rows)) != len(rows):
        return None
    records, owners = [], []
    storage = table.untyped_storage().data_ptr()
    for row, source in zip(rows, sources):
        if not isinstance(row, int) or not 0 <= row < table.shape[0]:
            return None
        if isinstance(source, (list, tuple)) and not source:
            continue
        if not isinstance(source, torch.Tensor) or source.ndim != 1:
            return None
        if source.device != table.device or source.dtype not in (torch.int32, torch.int64):
            return None
        if not source.is_contiguous() or source.numel() > table.shape[1]:
            return None
        if source.untyped_storage().data_ptr() == storage:
            return None
        if source.numel():
            records.append((source.data_ptr(), row, source.numel(), int(source.dtype == torch.int64)))
            owners.append(source)
    if len(records) < min_segments:
        return None
    if not records:
        return PrefixCopyPlan(torch.empty(0, device=table.device, dtype=torch.int64), (), 0, 0)
    with torch.cuda.device(table.device):
        stream = torch.cuda.current_stream(table.device)
        for source in owners:
            source.record_stream(stream)
        host = torch.tensor(records, dtype=torch.int64, pin_memory=True)
        descriptors = host.to(table.device, non_blocking=True)
    return PrefixCopyPlan(descriptors, tuple(owners), max(r[2] for r in records), len(records))


def prepare_prefix_copy_cpp(table, rows, sources, min_segments=0):
    from runtime_opt.descriptor_extension import initialize
    if table.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    # Empty prefixes can be Python lists in the upstream request class.
    if any(not isinstance(source, torch.Tensor) for source in sources):
        pairs = [(row, source) for row, source in zip(rows, sources) if len(source)]
        if len(rows) != len(sources):
            return None
        rows = [pair[0] for pair in pairs]
        sources = [pair[1] for pair in pairs]
    descriptors, count, maximum = initialize().prepare(table, rows, list(sources), min_segments)
    if count < 0:
        return None
    return PrefixCopyPlan(descriptors, tuple(sources), maximum, count)


def try_fused_prefix_copy(table, rows, sources, min_segments=16, backend='cpp'):
    """Launch the fused operation when eligible; otherwise leave the table unchanged."""
    if len(sources) < min_segments:
        return False
    prepare = prepare_prefix_copy_cpp if backend == 'cpp' else prepare_prefix_copy
    plan = prepare(table, rows, sources, min_segments=min_segments)
    if plan is None:
        return False
    plan.launch(table)
    return True


def loop_prefix_copy(table, rows, sources):
    for row, source in zip(rows, sources):
        if len(source):
            table[row, :len(source)] = source


def copy_prefixes(table, rows, sources, min_segments=16, backend='cpp'):
    """Apply guarded fusion with the original assignment loop as fallback."""
    if not try_fused_prefix_copy(table, rows, sources, min_segments, backend):
        loop_prefix_copy(table, rows, sources)
