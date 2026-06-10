"""Single-owner asynchronous metadata collectives with explicit host completion."""
from contextlib import nullcontext
import torch
import torch.distributed as dist


class MetadataGather:
    """Reuse buffers only after finish; finish is the first legal host consumer.

    finish returns borrowed CPU storage, valid until the next start. This object
    is single-threaded and single-flight. Call close before releasing an active
    instance. CPU-origin metadata is snapshotted before asynchronous submission.
    """

    def __init__(self, width, group, device='cpu'):
        self.group = group
        self.device = torch.device(device)
        self.width = width
        self.world_size = dist.get_world_size(group)
        self.pending = False
        self.host_input = torch.empty(width, dtype=torch.int64,
                                      pin_memory=self.device.type == 'cuda')
        self.host_output = torch.empty(self.world_size * width, dtype=torch.int64,
                                       pin_memory=self.device.type == 'cuda')
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream(device=self.device)
            self.input = torch.empty(width, dtype=torch.int64, device=self.device)
            self.output = torch.empty(self.world_size * width, dtype=torch.int64, device=self.device)
            # Allocations originate on the caller stream; order first use after it.
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            self.input.record_stream(self.stream)
            self.output.record_stream(self.stream)
            self.event = torch.cuda.Event()
        else:
            self.input, self.output = self.host_input, self.host_output

    def start(self, values):
        if self.pending:
            raise RuntimeError('Previous metadata must be consumed before buffer reuse')
        if len(values) != self.width:
            raise ValueError('Metadata width changed')
        # Values are CPU integers; input storage remains owned through finish.
        self.host_input.copy_(torch.as_tensor(values, dtype=torch.int64))
        context = torch.cuda.stream(self.stream) if self.device.type == 'cuda' else nullcontext()
        with context:
            if self.device.type == 'cuda':
                self.input.copy_(self.host_input, non_blocking=True)
            self.work = dist.all_gather_into_tensor(self.output, self.input,
                                                    group=self.group, async_op=True)
            if self.device.type == 'cuda':
                # ProcessGroupNCCL waits insert its completion dependency on this stream.
                self.work.wait()
                self.host_output.copy_(self.output, non_blocking=True)
                self.event.record(self.stream)
        self.pending = True
        return self

    def finish(self):
        if not self.pending:
            raise RuntimeError('No pending metadata collective')
        if self.device.type == 'cuda':
            self.event.synchronize()
        else:
            self.work.wait()
        self.pending = False
        return self.host_output.view(self.world_size, self.width)

    def close(self):
        if self.pending:
            self.finish()


def prebuild_worker_batch(batch):
    """Build the actual upstream descriptor before global metadata is consumed."""
    if batch is None:
        return None
    if getattr(batch, '_pending_worker_descriptor', None) is not None:
        raise RuntimeError('Unconsumed model-worker descriptor')
    return batch.get_model_worker_batch()


def publish_worker_batch(batch, descriptor):
    if descriptor is None:
        return
    for name in ('global_num_tokens', 'global_num_tokens_for_logprob',
                 'is_extend_in_batch', 'can_run_dp_cuda_graph',
                 'tbo_split_seq_index', 'global_forward_mode'):
        setattr(descriptor, name, getattr(batch, name))
    batch._pending_worker_descriptor = descriptor
