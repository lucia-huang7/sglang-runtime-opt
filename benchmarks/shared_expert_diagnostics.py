"""Graph replay, branch timing, allocator lifetime and profiler controls."""
import argparse,datetime,gzip,json,statistics,time
from pathlib import Path
from types import SimpleNamespace
import torch
from benchmarks.expert_fixture import load_layer
from runtime_opt.expert_scheduling import guarded_dual_stream


def main(a):
 module,config,device=load_layer(a);rows=[];validation=[]
 for count in [1,8,32,128,512]:
  x=torch.randn(count,config.hidden_size,device=device,dtype=torch.bfloat16)
  methods={'serial':lambda:module.forward_normal(x.clone()),'upstream':lambda:module.forward_normal_dual_stream(x.clone()),'guarded':lambda:guarded_dual_stream(module,x.clone())}
  with torch.inference_mode():
   reference=methods['serial']().clone();torch.cuda.synchronize()
   for name,fn in methods.items():
    stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(20):fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):output=fn()
    graph.replay();torch.cuda.synchronize();torch.testing.assert_close(output,reference,rtol=.01,atol=.015)
    samples=[]
    for _ in range(50):graph.replay()
    torch.cuda.synchronize()
    for _ in range(200):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
     start.record();graph.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000)
    rows.append({'tokens':count,'method':name,'graph_gpu_median_us':statistics.median(samples),'samples_us':samples})
   # Both input and result may outlive their allocation stream.
   for attempt in range(10):
    producer=torch.cuda.Stream(device=device);consumer=torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
     changed=x+attempt*.01;expected=module.forward_normal(changed.clone()).clone()
     output=guarded_dual_stream(module,changed.clone())
    consumer.wait_stream(producer);output.record_stream(consumer);expected.record_stream(consumer)
    with torch.cuda.stream(consumer):torch.testing.assert_close(output,expected,rtol=.01,atol=.015)
    del changed,output,expected
    churn=[torch.empty_like(x) for _ in range(8)];consumer.synchronize();del churn
   validation.append({'tokens':count,'nondefault_stream_churn_cases':10,'status':'passed'})
 # Trace a complete serial and two-stream layer without using trace duration as a benchmark.
 x=torch.randn(128,config.hidden_size,device=device,dtype=torch.bfloat16)
 with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
  for name,fn in [('serial',lambda:module.forward_normal(x.clone())),('upstream',lambda:module.forward_normal_dual_stream(x.clone())),('guarded',lambda:guarded_dual_stream(module,x.clone()))]:
   with torch.profiler.record_function('expert_'+name):fn();torch.cuda.synchronize()
 trace=Path(a.output).with_suffix('.trace.json');profiler.export_chrome_trace(str(trace))
 with gzip.open(str(trace)+'.gz','wb') as f:f.write(trace.read_bytes())
 trace.unlink()
 Path(a.output).write_text(json.dumps({'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'graph_replay':rows,'lifetime_validation':validation,'trace':Path(str(trace)+'.gz').name},indent=2))
 torch.distributed.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--device',type=int,default=1);p.add_argument('--port',type=int,default=29991);p.add_argument('--output',required=True);main(p.parse_args())
