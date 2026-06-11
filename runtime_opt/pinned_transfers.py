"""Bounded pinned staging with explicit completion and CPU-view ownership."""
from dataclasses import dataclass
import torch


@dataclass
class _Slot:
    host: torch.Tensor
    ready: object
    generation: int = 0
    active: bool = False
    leased: bool = False


class TransferTicket:
    def __init__(self, ring, slot, generation, result, count, direction):
        self.ring, self.slot, self.generation = ring, slot, generation
        self.result, self.count, self.direction = result, count, direction
        self.released = False
        self.completed = False

    def wait(self):
        if self.direction == 'd2h' and self.released:
            raise RuntimeError('CPU transfer lease was released')
        if not self.completed:
            if self.slot.generation == self.generation:
                self.slot.ready.synchronize()
                self.slot.active = False
            self.completed = True
        return self.result

    def consume(self, owned=True):
        if self.direction != 'd2h':
            return self.wait()
        self.wait()
        view = self.slot.host[:self.count]
        if owned:
            result = view.clone()
            self.release()
            return result
        return view

    def release(self):
        if self.released:
            raise RuntimeError('Transfer lease already released')
        self.wait()
        if self.direction == 'd2h':
            self.slot.leased = False
        self.released = True


class PinnedTransferRing:
    """Single-threaded staging ring; borrowed D2H views need explicit release.

    H2D destinations are caller-owned or freshly allocated, never ring-owned.
    CPU staging can therefore be reused without overwriting model input tensors.
    D2H owned consumption clones before release; borrowed views expire at release.
    """
    def __init__(self, capacity, dtype, device, depth=2):
        if capacity <= 0 or depth not in (1, 2, 4):
            raise ValueError('Use positive capacity and ring depth 1, 2 or 4')
        self.capacity, self.dtype = capacity, dtype
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda':
            raise ValueError('A CUDA device is required')
        self.stream = torch.cuda.Stream(device=self.device)
        self.slots = [_Slot(torch.empty(capacity, dtype=dtype, pin_memory=True),
                            torch.cuda.Event()) for _ in range(depth)]
        self.next = 0
        self.closed = False

    @property
    def pinned_bytes(self):
        return len(self.slots) * self.capacity * self.slots[0].host.element_size()

    def _acquire(self, count):
        if self.closed: raise RuntimeError('Transfer ring is closed')
        if count < 0 or count > self.capacity: raise ValueError('Transfer exceeds capacity')
        slot = self.slots[self.next]
        if slot.leased: raise BufferError('CPU result must be released before ring reuse')
        if slot.active: slot.ready.synchronize()
        self.next = (self.next + 1) % len(self.slots)
        slot.generation += 1
        return slot

    def upload(self, source, destination=None):
        if source.device.type != 'cpu' or source.dtype != self.dtype or not source.is_contiguous():
            raise ValueError('Expected contiguous CPU input with the ring dtype')
        if destination is not None and (destination.device != self.device or destination.dtype != self.dtype
                or not destination.is_contiguous() or destination.numel() != source.numel()):
            raise ValueError('Invalid caller-owned CUDA destination')
        slot = self._acquire(source.numel())
        slot.host[:source.numel()].copy_(source.reshape(-1))
        target = destination if destination is not None else torch.empty(source.shape, dtype=self.dtype, device=self.device)
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            target.copy_(slot.host[:source.numel()].view(source.shape), non_blocking=True)
            target.record_stream(self.stream)
            slot.ready.record(self.stream)
        current.wait_event(slot.ready)
        slot.active = True
        return TransferTicket(self, slot, slot.generation, target, source.numel(), 'h2d')

    def download(self, source):
        if source.device != self.device or source.dtype != self.dtype or not source.is_contiguous():
            raise ValueError('Expected contiguous CUDA input with the ring dtype')
        slot = self._acquire(source.numel())
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            slot.host[:source.numel()].copy_(source.reshape(-1), non_blocking=True)
            source.record_stream(self.stream)
            slot.ready.record(self.stream)
        # Subsequent mutation on the producer stream must follow this snapshot.
        current.wait_event(slot.ready)
        slot.active = slot.leased = True
        return TransferTicket(self, slot, slot.generation, None, source.numel(), 'd2h')

    def close(self):
        for slot in self.slots:
            if slot.active: slot.ready.synchronize()
        if any(slot.leased for slot in self.slots):
            raise RuntimeError('Release outstanding CPU result leases before closing')
        self.closed = True


_INPUT_RINGS = {}
_LENGTH_RINGS = {}


