"""Transactional reference for tentative draft branches and verified KV ownership."""
from dataclasses import dataclass

@dataclass(frozen=True)
class DraftState:
    tokens: tuple
    kv_slots: tuple
    position: int
    finished: bool = False

@dataclass(frozen=True)
class ReconciledDraft:
    state: DraftState
    released_slots: tuple
    useful_tokens: int


def reconcile(initial, candidates, slots, accepted, eos=None, remaining=None, cancelled=False):
    """Publish only the verified prefix and release every uncommitted slot."""
    if len(candidates) != len(slots) or len(set(slots)) != len(slots):
        raise ValueError('Candidate tokens require unique corresponding KV slots')
    if set(slots).intersection(initial.kv_slots):
        raise ValueError('Tentative KV slots alias committed state')
    if not 0 <= accepted <= len(candidates):
        raise ValueError('Invalid verified prefix length')
    if remaining is not None and remaining < 0:
        raise ValueError('Negative output budget')
    keep=0 if cancelled or initial.finished else accepted
    if remaining is not None:keep=min(keep,remaining)
    if eos is not None:
        for i,token in enumerate(candidates[:keep]):
            if token==eos:keep=i+1;break
    selected=tuple(candidates[:keep]);owned=tuple(slots[:keep])
    finished=(initial.finished or cancelled or (eos is not None and eos in selected)
              or (remaining is not None and keep==remaining))
    return ReconciledDraft(DraftState(initial.tokens+selected,initial.kv_slots+owned,
        initial.position+keep,finished),tuple(slots[keep:]),keep)


class TentativeBranches:
    """Require an explicit state for every selectable verified prefix."""
    def __init__(self, initial, candidates, slots, branches):
        self.initial=initial;self.candidates=tuple(candidates);self.slots=tuple(slots)
        self.branches=dict(branches);self.closed=False
        if set(self.branches)!=set(range(len(candidates)+1)):
            raise ValueError('Missing branch for a possible acceptance length')
        for count,state in self.branches.items():
            expected=reconcile(initial,self.candidates,self.slots,count).state
            if state!=expected:raise ValueError('Tentative branch differs from serial state')
    def commit(self,accepted,**kwargs):
        if self.closed:raise RuntimeError('Transaction already reconciled')
        result=reconcile(self.initial,self.candidates,self.slots,accepted,**kwargs)
        self.closed=True
        return result
