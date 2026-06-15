"""Explicit serial, upstream and lifetime-guarded shared-expert scheduling."""
import os
import torch


def guarded_dual_stream(module, hidden_states, can_fuse_mlp_allreduce=False):
    """Preserve upstream arithmetic while recording cross-stream allocations."""
    from sglang.srt.models.deepseek_v2 import tensor_model_parallel_all_reduce
    current = torch.cuda.current_stream(hidden_states.device)
    alternate = module.alt_stream
    alternate.wait_stream(current)
    hidden_states.record_stream(alternate)
    shared = module._forward_shared_experts(hidden_states)
    with torch.cuda.stream(alternate):
        logits = module.gate(hidden_states)
        arguments = {"hidden_states": hidden_states}
        if module.topk is not None:
            arguments["topk_output"] = module.topk(hidden_states, logits)
        else:
            arguments["router_logits"] = logits
        routed = module.experts(**arguments)
        routed.record_stream(current)
    current.wait_stream(alternate)
    routed += shared
    if module.tp_size > 1 and not can_fuse_mlp_allreduce:
        routed = tensor_model_parallel_all_reduce(routed)
    return routed


def install_expert_policy():
    """Install an opt-in policy without changing upstream fusion settings."""
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
    if getattr(DeepseekV2MoE, '_runtime_policy_installed', False):
        return
    original = DeepseekV2MoE.forward
    def forward(self, hidden_states, forward_batch=None, can_fuse_mlp_allreduce=False):
        mode = os.environ.get('SGLANG_EXPERT_POLICY', 'upstream')
        if mode not in ('serial', 'upstream', 'guarded'):
            raise ValueError('Unsupported expert policy')
        eligible = (not self._enable_deepep_moe and self.alt_stream is not None
                    and self.num_fused_shared_experts == 0
                    and hasattr(self, 'shared_experts') and hidden_states.is_cuda)
        if not eligible or mode == 'upstream':
            return original(self, hidden_states, forward_batch, can_fuse_mlp_allreduce)
        if mode == 'serial' or hidden_states.shape[0] == 0:
            return self.forward_normal(hidden_states, can_fuse_mlp_allreduce)
        limit = int(os.environ.get('SGLANG_EXPERT_MAX_TOKENS', '1024'))
        if hidden_states.shape[0] > limit:
            return self.forward_normal(hidden_states, can_fuse_mlp_allreduce)
        return guarded_dual_stream(self, hidden_states, can_fuse_mlp_allreduce)
    DeepseekV2MoE.forward = forward
    DeepseekV2MoE._runtime_policy_installed = True
