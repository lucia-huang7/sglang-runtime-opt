"""Exhaustive prefix acceptance and completion boundaries for the CPU oracle."""
import unittest
from runtime_opt.speculative_state import DraftState,TentativeBranches,reconcile

class StateTests(unittest.TestCase):
 def test_all_prefixes_and_completion(self):
  cases=0
  for depth in range(9):
   for accepted in range(depth+1):
    for eos_at in range(-1,depth):
     for remaining in [None,0,1,3,9]:
      for cancelled in [False,True]:
       initial=DraftState((7,8),(1,2),2)
       tokens=list(range(10,10+depth));slots=list(range(100,100+depth))
       eos=tokens[eos_at] if eos_at>=0 else None
       branches={n:reconcile(initial,tokens,slots,n).state for n in range(depth+1)}
       result=TentativeBranches(initial,tokens,slots,branches).commit(accepted,eos=eos,remaining=remaining,cancelled=cancelled)
       keep=0 if cancelled else min(accepted,remaining if remaining is not None else accepted)
       if eos_at>=0 and eos_at<keep:keep=eos_at+1
       self.assertEqual(result.state.tokens,initial.tokens+tuple(tokens[:keep]))
       self.assertEqual(result.state.kv_slots,initial.kv_slots+tuple(slots[:keep]))
       self.assertEqual(result.state.position,2+keep)
       self.assertEqual(result.released_slots,tuple(slots[keep:]));cases+=1
  self.assertEqual(cases,2850)
 def test_invalid_and_reuse(self):
  initial=DraftState((),(),0)
  with self.assertRaises(ValueError):TentativeBranches(initial,[1],[2],{})
  with self.assertRaises(ValueError):reconcile(initial,[1,2],[3,3],1)
  with self.assertRaises(ValueError):reconcile(initial,[1],[2],2)
  transaction=TentativeBranches(initial,[],[],{0:initial});transaction.commit(0)
  with self.assertRaises(RuntimeError):transaction.commit(0)
 def test_mixed_batch_is_independent(self):
  rows=[reconcile(DraftState((),(),0),[3,4],[10+i*2,11+i*2],i,eos=4) for i in range(3)]
  self.assertEqual([r.useful_tokens for r in rows],[0,1,2])
  self.assertEqual([r.state.finished for r in rows],[False,False,True])
if __name__=='__main__':unittest.main()
