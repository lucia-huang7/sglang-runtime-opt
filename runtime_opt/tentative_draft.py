"""Guarded linear-chain EAGLE extension before verification CPU reconciliation.

Only the target-verified causal prefix is reusable. Later draft positions are
scratch work; the original verification path retains KV eviction and EOS logic.
"""
import copy
import json
import os
import threading
import torch

_CONTEXT=threading.local()


def current_worker():
    return getattr(_CONTEXT,'worker',None)


def eligible(worker,batch,spec):
    return (worker is not None and worker.topk==1 and worker.page_size==1
        and worker.speculative_num_steps==3
        and worker.speculative_num_draft_tokens==worker.speculative_num_steps+1
        and worker.hot_token_id is None and not worker.server_args.enable_dp_attention
        and worker.server_args.tp_size==1 and worker.server_args.pp_size==1
        and worker.server_args.disable_cuda_graph and worker.server_args.disable_overlap_schedule
        and worker.server_args.attention_backend=='triton' and worker.server_args.dtype=='bfloat16'
        and worker.server_args.quantization is None and worker.server_args.kv_cache_dtype=='auto'
        and not worker.server_args.enable_two_batch_overlap and not worker.server_args.enable_lora
        and batch.sampling_info.is_all_greedy and not batch.has_grammar
        and not batch.forward_mode.is_idle() and len(batch.reqs)>0
        and worker.draft_model_runner.model.__class__.__module__.endswith('qwen2_eagle')
        and worker.draft_model_runner.model.config.num_hidden_layers==1
        and worker.draft_model_runner.token_to_kv_pool is not worker.target_worker.model_runner.token_to_kv_pool
)


def start_tentative(batch,spec,predict):
    worker=current_worker()
    if not eligible(worker,batch,spec):return False
    worker._runtime_tentative_calls=getattr(worker,'_runtime_tentative_calls',0)+1
    from sglang.srt.speculative.eagle_utils import EagleDraftInput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch,ForwardMode,CaptureHiddenMode
    count=len(batch.reqs);depth=worker.speculative_num_draft_tokens
    temporary=copy.copy(batch)
    temporary.spec_info=EagleDraftInput(hidden_states=spec.hidden_states,
        verified_id=predict.reshape(-1)[:count*depth].clone(),
        accept_length=torch.full((count,),depth-1,dtype=torch.int32,device=predict.device),
        accept_length_cpu=[depth-1]*count,
        seq_lens_for_draft_extend=batch.seq_lens+depth,
        req_pool_indices_for_draft_extend=batch.req_pool_indices)
    temporary.spec_info.num_tokens_per_batch=depth
    temporary.spec_info.num_tokens_for_logprob_per_batch=1
    temporary.spec_info.prepare_extend_after_decode(temporary,worker.speculative_num_steps)
    temporary.spec_info.capture_hidden_mode=CaptureHiddenMode.FULL
    temporary.return_logprob=False;temporary.return_hidden_states=True
    temporary.forward_mode=ForwardMode.DRAFT_EXTEND
    base_cpu=spec.seq_lens_cpu
    if base_cpu is None:
        base_cpu=torch.tensor([len(r.origin_input_ids)+len(r.output_ids)-1 for r in batch.reqs],dtype=torch.int64)
    if os.environ.get('SGLANG_TENTATIVE_DRAFT')=='shadow':
        torch.testing.assert_close(batch.seq_lens,base_cpu.to(batch.seq_lens.device),rtol=0,atol=0)
    cpu_lengths=base_cpu+depth
    temporary.seq_lens_sum=int(cpu_lengths.sum())
    descriptor=temporary.get_model_worker_batch(seq_lens_cpu_cache=cpu_lengths)
    forward=ForwardBatch.init_new(descriptor,worker.draft_model_runner)
    forward.return_logprob=False
    from runtime_opt.tentative_attention import prepare_metadata,prefix_masked_forward
    backend=worker.draft_model_runner.attn_backend
    descriptors=prepare_metadata(backend,forward,base_cpu,depth)
    with torch.profiler.record_function('tentative_draft_full_chain'):
        output,_=prefix_masked_forward(backend,worker.draft_model_runner,forward)
    if output.hidden_states.shape[0]!=count*depth:
        raise RuntimeError('Tentative extension did not capture all causal positions')
    worker._runtime_tentative={'output':output,'forward':forward,'requests':list(batch.reqs),
        'depth':depth,'temporary':temporary,'predict':predict,'descriptors':descriptors}
    return True


def selected_outputs(worker,batch):
    from sglang.srt.layers.logits_processor import LogitsMetadata
    from sglang.srt.speculative.eagle_worker import fast_topk
    pending=worker._runtime_tentative
    active=[i for i,r in enumerate(pending['requests']) if not r.finished()]
    accepted=batch.spec_info.accept_length_cpu
    if len(active)!=len(accepted):raise RuntimeError('Finished-request reconciliation differs')
    indices=torch.tensor([i*pending['depth']+n for i,n in zip(active,accepted)],device=batch.seq_lens.device,dtype=torch.long)
    hidden=pending['output'].hidden_states.index_select(0,indices)
    model=worker.draft_model_runner.model
    metadata=LogitsMetadata.from_forward_batch(pending['forward'])
    logits=model.logits_processor._get_logits(hidden,model.lm_head,metadata)
    probabilities,index=fast_topk(torch.softmax(logits,dim=-1),worker.topk,dim=-1)
    return hidden,probabilities,index


def install_tentative_draft():
    mode=os.environ.get('SGLANG_TENTATIVE_DRAFT')
    if mode not in ('shadow','active'):return
    from sglang.srt.speculative.eagle_worker import EAGLEWorker
    if getattr(EAGLEWorker,'_runtime_tentative_installed',False):return
    verify=EAGLEWorker.verify;extend=EAGLEWorker.forward_draft_extend_after_decode
    def verify_wrapper(self,batch,spec):
        if not hasattr(self,'_runtime_tentative_logged'):
            print('TENTATIVE_GUARDS',self.draft_model_runner.model.__class__.__name__,self.topk,self.page_size,
                  self.server_args.tp_size,self.server_args.disable_cuda_graph,'eligible',eligible(self,batch,spec),flush=True)
            self._runtime_tentative_logged=True
        self._runtime_tentative=None;_CONTEXT.worker=self
        try:
            result=verify(self,batch,spec)
            if batch.spec_info.verified_id.numel()==0:self._runtime_tentative=None
            return result
        except Exception:
            self._runtime_tentative=None
            raise
        finally:_CONTEXT.worker=None
    def extend_wrapper(self,batch):
        if self._runtime_tentative is None:return extend(self,batch)
        hidden,probabilities,index=selected_outputs(self,batch)
        if mode=='shadow':
            pending=self._runtime_tentative
            active=[i for i,r in enumerate(pending['requests']) if not r.finished()]
            locations=pending['temporary'].out_cache_loc.reshape(len(pending['requests']),pending['depth'])
            keep=torch.cat([locations[i,:n+1] for i,n in zip(active,batch.spec_info.accept_length_cpu)])
            pool=self.draft_model_runner.token_to_kv_pool
            key_before=pool.get_key_buffer(0)[keep].clone()
            value_before=pool.get_value_buffer(0)[keep].clone()
            accepted=list(batch.spec_info.accept_length_cpu)
            result=extend(self,batch)
            torch.testing.assert_close(pool.get_key_buffer(0)[keep],key_before,rtol=.02,atol=.03)
            torch.testing.assert_close(pool.get_value_buffer(0)[keep],value_before,rtol=.02,atol=.03)
            difference=(hidden-batch.spec_info.hidden_states).float()
            reference_norm=batch.spec_info.hidden_states.float().norm(dim=-1).clamp_min(1e-6)
            relative_norm=(difference.norm(dim=-1)/reference_norm).max().item()
            if relative_norm>.01:raise AssertionError('Draft feature relative L2 error exceeds 1 percent')
            probability_error=float((probabilities-batch.spec_info.topk_p).abs().max())
            candidate_exact=bool(torch.equal(index,batch.spec_info.topk_index))
            report=os.environ.get('SGLANG_TENTATIVE_REPORT')
            if report:
                with open(report,'a') as output:
                    output.write(json.dumps({'accepted':accepted,'acceptance_cap':os.environ.get('SGLANG_VERIFY_ACCEPTANCE_CAP'),'active_requests':len(active),
                        'original_requests':len(pending['requests']),
                        'hidden_max_abs':float(difference.abs().max()),'hidden_relative_l2_max':relative_norm,
                        'token_indices_exact':candidate_exact,'probability_max_abs':probability_error,'kv_prefix_close':True})+'\n')
            self._runtime_tentative=None
            return result
        spec=batch.spec_info
        lengths=batch.seq_lens.clone();pools=batch.req_pool_indices;return_logprob=batch.return_logprob
        spec.num_tokens_per_batch=self.speculative_num_steps+1
        spec.num_tokens_for_logprob_per_batch=1
        spec.prepare_extend_after_decode(batch,self.speculative_num_steps)
        batch.return_hidden_states=False
        spec.hidden_states=hidden;spec.topk_p=probabilities;spec.topk_index=index
        batch.seq_lens=lengths;batch.req_pool_indices=pools;batch.return_logprob=return_logprob
        self._runtime_tentative=None
    EAGLEWorker.verify=verify_wrapper
    EAGLEWorker.forward_draft_extend_after_decode=extend_wrapper
    EAGLEWorker._runtime_tentative_installed=True
