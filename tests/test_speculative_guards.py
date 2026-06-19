"""Unsupported execution configurations retain serial EAGLE behavior."""
import unittest
from types import SimpleNamespace as NS
from runtime_opt.tentative_draft import eligible

class Model:
 __module__='sglang.srt.models.qwen2_eagle'
 config=NS(num_hidden_layers=1)

class GuardTests(unittest.TestCase):
 def build(self):
  args=NS(enable_dp_attention=False,tp_size=1,pp_size=1,disable_cuda_graph=True,
   disable_overlap_schedule=True,attention_backend='triton',dtype='bfloat16',quantization=None,
   kv_cache_dtype='auto',enable_two_batch_overlap=False,enable_lora=None)
  worker=NS(topk=1,page_size=1,speculative_num_steps=3,speculative_num_draft_tokens=4,
   hot_token_id=None,server_args=args,draft_model_runner=NS(model=Model(),token_to_kv_pool=object()),
   target_worker=NS(model_runner=NS(token_to_kv_pool=object())))
  batch=NS(sampling_info=NS(is_all_greedy=True),has_grammar=False,
   forward_mode=NS(is_idle=lambda:False),reqs=[object()])
  return worker,batch,NS()
 def test_selected_configuration(self):
  self.assertTrue(eligible(*self.build()))
 def test_unsupported_server_configs(self):
  changes={'enable_dp_attention':True,'tp_size':2,'pp_size':2,'disable_cuda_graph':False,
   'disable_overlap_schedule':False,'attention_backend':'flashinfer','dtype':'float16',
   'quantization':'fp8','kv_cache_dtype':'fp8_e4m3','enable_two_batch_overlap':True,'enable_lora':True}
  for name,value in changes.items():
   with self.subTest(name=name):
    worker,batch,spec=self.build();setattr(worker.server_args,name,value)
    self.assertFalse(eligible(worker,batch,spec))
 def test_batch_and_storage_guards(self):
  for case in ['stochastic','grammar','empty','idle','shared_kv','tree','pages','depth']:
   with self.subTest(case=case):
    worker,batch,spec=self.build()
    if case=='stochastic':batch.sampling_info.is_all_greedy=False
    if case=='grammar':batch.has_grammar=True
    if case=='empty':batch.reqs=[]
    if case=='idle':batch.forward_mode.is_idle=lambda:True
    if case=='shared_kv':worker.draft_model_runner.token_to_kv_pool=worker.target_worker.model_runner.token_to_kv_pool
    if case=='tree':worker.topk=2
    if case=='pages':worker.page_size=16
    if case=='depth':worker.speculative_num_steps=4
    self.assertFalse(eligible(worker,batch,spec))
if __name__=='__main__':unittest.main()
