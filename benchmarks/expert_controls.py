"""Cyclic-balanced routing, skewed routing and held-out MoE token counts."""
import argparse,datetime,json,statistics,time
from pathlib import Path
import torch
from benchmarks.expert_fixture import load_layer
from runtime_opt.expert_scheduling import guarded_dual_stream


def main(a):
 module,config,device=load_layer(a);torch.manual_seed(193);rows=[];validation=[]
 for count in [1,2,8,16,32,64,128,256,512]:
  for routing in ['cyclic_balanced','skewed']:
   x=torch.randn(count,config.hidden_size,device=device,dtype=torch.bfloat16)
   logits=torch.full((count,config.n_routed_experts),-20.,device=device,dtype=torch.bfloat16)
   expert_ids=(torch.arange(count,device=device)[:,None]*config.num_experts_per_tok+torch.arange(config.num_experts_per_tok,device=device)[None,:])%config.n_routed_experts
   if routing=='skewed':expert_ids=expert_ids[:1].expand(count,-1)
   logits.scatter_(1,expert_ids,torch.arange(config.num_experts_per_tok,device=device,dtype=torch.bfloat16)[None,:].expand(count,-1))
   module.gate.forward=lambda hidden:logits
   methods={'serial':lambda:module.forward_normal(x.clone()),'upstream':lambda:module.forward_normal_dual_stream(x.clone()),'guarded':lambda:guarded_dual_stream(module,x.clone())}
   with torch.inference_mode():
    reference=methods['serial']().clone();torch.cuda.synchronize()
    for name,fn in methods.items():
     output=fn();torch.cuda.synchronize();torch.testing.assert_close(output,reference,rtol=.01,atol=.015)
     validation.append({'tokens':count,'routing':routing,'method':name,'max_abs':float((output-reference).abs().max())})
    values={};order=list(methods)
    if a.reverse:order.reverse()
    if not a.validate_only:
     for name in order+list(reversed(order)):
      fn=methods[name]
      for _ in range(50):fn()
      torch.cuda.synchronize();samples=[]
      for _ in range(100):
       start=time.perf_counter_ns();fn();torch.cuda.synchronize();samples.append((time.perf_counter_ns()-start)/1000)
      values.setdefault(name,[]).append({'median_us':statistics.median(samples),'samples_us':samples})
   rows.append({'tokens':count,'routing':routing,'held_out':count in [2,16,64,256],'measurements':values})
 Path(a.output).write_text(json.dumps({'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,
  'validation':validation,'router_logits_prepared':True,'weight_source':'checkpoint','validate_only':a.validate_only},indent=2))
 torch.distributed.destroy_process_group();print('Expert routing controls passed',len(validation),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--port',type=int,default=29992);p.add_argument('--reverse',action='store_true');p.add_argument('--validate-only',action='store_true');p.add_argument('--output',required=True);main(p.parse_args())
