"""Checkpoint-weight MoE layer controls using the actual pinned SGLang class."""
import argparse,datetime,hashlib,json,os,statistics,time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig
from runtime_opt.expert_scheduling import guarded_dual_stream

MODEL='85864749cd611b4353ce1decdb286193298f64c7'

def main(a):
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
 torch.cuda.synchronize();rows=[];validation=[]
 generator=torch.Generator(device=device).manual_seed(193)
 original_gate=module.gate.forward
 for count in [1,8,32,128,512]:
  for routing in ['natural','skewed']:
   x=torch.randn(count,config.hidden_size,device=device,generator=generator)
   if routing=='skewed':
    def gate(t):
     logits=torch.full((t.shape[0],config.n_routed_experts),-20.,device=t.device,dtype=t.dtype)
     logits[:,:config.num_experts_per_tok]=torch.arange(config.num_experts_per_tok,device=t.device,dtype=t.dtype)
     return logits
    module.gate.forward=gate
   else:module.gate.forward=original_gate
   methods={'serial':lambda:module.forward_normal(x.clone()),'upstream':lambda:module.forward_normal_dual_stream(x.clone()),'guarded':lambda:guarded_dual_stream(module,x.clone())}
   with torch.inference_mode():
    reference=methods['serial']().clone();torch.cuda.synchronize()
    for name,fn in methods.items():
     result=fn();torch.cuda.synchronize();torch.testing.assert_close(result,reference,rtol=.01,atol=.015)
     validation.append({'tokens':count,'routing':routing,'method':name,'max_abs':float((result-reference).abs().max()),'finite':bool(torch.isfinite(result).all())})
    order=list(methods)
    if a.reverse:order.reverse()
    measurements={}
    for name in order+list(reversed(order)):
     fn=methods[name]
     for _ in range(50):fn()
     torch.cuda.synchronize();samples=[]
     for _ in range(100):
      start=time.perf_counter_ns();fn();torch.cuda.synchronize();samples.append((time.perf_counter_ns()-start)/1000)
     measurements.setdefault(name,[]).append({'median_us':statistics.median(samples),'p95_us':sorted(samples)[95],'samples_us':samples})
    # Event timing is separate from inclusive host-wall timing.
    events={}
    for name,fn in methods.items():
     start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     start.record()
     for _ in range(50):fn()
     end.record();end.synchronize();events[name]=start.elapsed_time(end)*1000/50
   row={'tokens':count,'routing':routing,'measurements':measurements,'device_event_us':events}
   rows.append(row)
   Path(a.output).write_text(json.dumps({'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'kind':'checkpoint_moe_layer','model_revision':MODEL,'layer':1,'device':str(device),'weight_source':'checkpoint',
    'fusion':False,'warmup':50,'samples_per_block':100,'validation':validation,'rows':rows},indent=2))
   print(count,routing,{k:round(statistics.median([b['median_us'] for b in v]),2) for k,v in measurements.items()},flush=True)
 torch.distributed.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--device',type=int,default=1);p.add_argument('--port',type=int,default=29990);p.add_argument('--reverse',action='store_true');p.add_argument('--output',required=True);main(p.parse_args())
