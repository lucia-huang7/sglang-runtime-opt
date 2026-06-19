"""Exact cached-prefix masking for reusable linear EAGLE query prefixes."""
import itertools
import torch
import triton
import triton.language as tl

@triton.jit
def _prefix_mask(mask,base_lengths,mask_offsets,depth:tl.constexpr,BLOCK:tl.constexpr):
 request=tl.program_id(0);offset=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
 base=tl.load(base_lengths+request);width=base+2*depth
 query=offset//width;column=offset%width
 keep=(column<=base+query)|((column>=base+depth)&(column<=base+depth+query))
 start=tl.load(mask_offsets+request)
 tl.store(mask+start+offset,keep,mask=query<depth)


def prepare_metadata(backend,forward,base_cpu,depth):
 from sglang.srt.layers.attention.triton_backend import ForwardMetadata
 from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
 bases=base_cpu.tolist();count=len(bases)
 key_offsets=[0]+list(itertools.accumulate(n+depth for n in bases))
 mask_offsets=[0]+list(itertools.accumulate(depth*(n+2*depth) for n in bases))
 query_offsets=[i*depth for i in range(count+1)]
 flat=bases+key_offsets+mask_offsets+query_offsets
 host=torch.tensor(flat,dtype=torch.int32,pin_memory=True)
 device=host.to(forward.seq_lens.device,non_blocking=True)
 start=count;kv_indptr=device[start:start+count+1];start+=count+1
 mask_indptr=device[start:start+count+1];start+=count+1
 qo_indptr=device[start:start+count+1]
 indices=torch.empty(key_offsets[-1],device=device.device,dtype=torch.int32)
 create_flashinfer_kv_indices_triton[(count,)](backend.req_to_token,forward.req_pool_indices,
    forward.seq_lens,kv_indptr,None,indices,backend.req_to_token.stride(0))
 mask=torch.empty(mask_offsets[-1],device=device.device,dtype=torch.bool)
 _prefix_mask[(count,triton.cdiv(depth*(max(bases)+2*depth),256))](mask,device,mask_indptr,depth,256)
 backend.forward_metadata=ForwardMetadata(attn_logits=None,attn_lse=None,max_extend_len=depth,
  num_kv_splits=None,kv_indptr=kv_indptr,kv_indices=indices,qo_indptr=qo_indptr,
  custom_mask=mask,mask_indptr=mask_indptr,window_kv_indptr=None,window_kv_indices=None,window_num_kv_splits=None)
 # Keep CPU and device descriptors alive through the queued model work.
 return host,device,mask,indices


def prefix_masked_forward(backend,runner,forward):
 original=backend.extend_attention_fwd
 model=runner.model
 original_logits=model.logits_processor.forward
 def hidden_only(input_ids,hidden_states,lm_head,logits_metadata,aux_hidden_states=None):
  from sglang.srt.layers.logits_processor import LogitsProcessorOutput
  if aux_hidden_states is not None:raise RuntimeError('Auxiliary draft features are unsupported')
  return LogitsProcessorOutput(next_token_logits=None,hidden_states=hidden_states)
 def with_prefix_mask(*args,**kwargs):
  arguments=list(args)
  if len(arguments)!=16:raise RuntimeError('Unexpected pinned Triton attention signature')
  # The pinned caller passes its window argument at the skip-prefix-mask position.
  arguments[15]=False
  return original(*arguments,sliding_window_size=-1,**kwargs)
 backend.extend_attention_fwd=with_prefix_mask
 model.logits_processor.forward=hidden_only
 try:return runner.forward(forward,skip_attn_backend_init=True)
 finally:
  backend.extend_attention_fwd=original
  model.logits_processor.forward=original_logits
