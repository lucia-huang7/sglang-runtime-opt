"""Owned, event-guarded snapshots of EAGLE integer verification outputs."""
import torch

class VerificationSnapshot:
    def __init__(self, accept_index, predict, accept_length, stream=None):
        values=(accept_index,predict,accept_length)
        if any(x.device!=accept_index.device or x.dtype!=torch.int32 for x in values):
            raise ValueError('Verification snapshots require same-device int32 tensors')
        if not accept_index.is_cuda:raise ValueError('Verification tensors must be CUDA tensors')
        self.shapes=[tuple(x.shape) for x in values]
        self.sizes=[x.numel() for x in values]
        self.device_payload=torch.cat([x.reshape(-1) for x in values])
        self.host_payload=torch.empty(sum(self.sizes),dtype=torch.int32,pin_memory=True)
        self.ready=torch.cuda.Event()
        current=torch.cuda.current_stream(accept_index.device)
        if stream is None:
            self.host_payload.copy_(self.device_payload,non_blocking=True)
            self.ready.record(current)
        else:
            stream.wait_stream(current)
            self.device_payload.record_stream(stream)
            with torch.cuda.stream(stream):
                self.host_payload.copy_(self.device_payload,non_blocking=True)
                self.ready.record(stream)
        self.consumed=False
    def result(self):
        if self.consumed:raise RuntimeError('Verification snapshot already consumed')
        self.ready.synchronize();self.consumed=True
        result=[];offset=0
        for size,shape in zip(self.sizes,self.shapes):
            result.append(self.host_payload[offset:offset+size].reshape(shape).tolist());offset+=size
        return tuple(result)
