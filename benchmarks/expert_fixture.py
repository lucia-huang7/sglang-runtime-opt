"""Checkpoint-weight MoE layer controls using the actual pinned SGLang class."""
import argparse,datetime,hashlib,json,os,statistics,time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig
from runtime_opt.expert_scheduling import guarded_dual_stream

MODEL='85864749cd611b4353ce1decdb286193298f64c7'

def load_layer(a):
 from sglang.srt.server_args import ServerArgs
 from sglang.srt.managers.schedule_batch import global_server_args_dict
 from sglang.srt.distributed import parallel_state as parallel
 device=torch.device('cuda',a.device);torch.cuda.set_device(device)
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[min(a.device*4,len(allowed)-1)]})
 global_server_args_dict.update(vars(ServerArgs(model_path=a.model,disable_shared_experts_fusion=True)))
 parallel.init_distributed_environment(1,0,f'tcp://127.0.0.1:{a.port}',a.device)
 parallel.initialize_model_parallel()
 from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
 config=AutoConfig.from_pretrained(a.model,trust_remote_code=True)
 torch.set_default_dtype(torch.bfloat16)
 with torch.device(device):module=DeepseekV2MoE(config,1,alt_stream=torch.cuda.Stream(device=device))
 index=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())['weight_map']
 prefix='model.layers.1.mlp.'
 def read(name):
  key=prefix+name
  with safe_open(str(Path(a.model)/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)
 with torch.no_grad():
  module.gate.weight.copy_(read('gate.weight'))
  module.shared_experts.gate_up_proj.weight.copy_(torch.cat([read('shared_experts.gate_proj.weight'),read('shared_experts.up_proj.weight')]))
  module.shared_experts.down_proj.weight.copy_(read('shared_experts.down_proj.weight'))
  for expert in range(config.n_routed_experts):
   module.experts.w13_weight[expert].copy_(torch.cat([read(f'experts.{expert}.gate_proj.weight'),read(f'experts.{expert}.up_proj.weight')]))
   module.experts.w2_weight[expert].copy_(read(f'experts.{expert}.down_proj.weight'))

 torch.cuda.synchronize()
 return module,config,device
