"""Inclusive verification-result transfer controls and ownership validation."""
import argparse,datetime,json,statistics,time
from pathlib import Path
import torch
from runtime_opt.verification_transfer import VerificationSnapshot


def main(a):
 torch.cuda.set_device(a.device);device=torch.device('cuda',a.device);rows=[];validation=[]
 for batch in [0,1,4,16,128,512]:
  for depth in [2,4,8]:
   indices=torch.arange(batch*depth,device=device,dtype=torch.int32).reshape(batch,depth)
   predict=torch.arange(batch*(depth+1),device=device,dtype=torch.int32)+31
   lengths=torch.full((batch,),depth-1,device=device,dtype=torch.int32)
   def serial():return indices.tolist(),predict.tolist(),lengths.tolist()
   def packed():return VerificationSnapshot(indices,predict,lengths).result()
   expected=serial();assert packed()==expected;validation.append({'batch':batch,'depth':depth,'exact':True})
   methods={'serial':serial,'packed':packed};order=list(methods)
   if a.reverse:order.reverse()
   values={}
   for name in order+list(reversed(order)):
    fn=methods[name]
    for _ in range(50):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(100):
     start=time.perf_counter_ns();result=fn();samples.append((time.perf_counter_ns()-start)/1000)
     assert result==expected
    values.setdefault(name,[]).append({'median_us':statistics.median(samples),'p95_us':sorted(samples)[95],'samples_us':samples})
   rows.append({'batch':batch,'depth':depth,'measurements':values})
 # Ownership across a nondefault producer stream and later mutation.
 other=torch.cuda.Stream();copy_stream=torch.cuda.Stream();original=torch.arange(64,device=device,dtype=torch.int32)
 producer=torch.cuda.current_stream()
 with torch.cuda.stream(other):
  other.wait_stream(producer);source=original+9
  ticket=VerificationSnapshot(source.reshape(8,8),source,source[:8],stream=copy_stream)
  source.fill_(-1)
 result=ticket.result();assert result[1]==list(range(9,73))
 try:ticket.result();raise AssertionError('Repeated consumer accepted')
 except RuntimeError:pass
 Path(a.output).write_text(json.dumps({'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'warmup':50,'samples_per_block':100,'rows':rows,'validation':validation,'nondefault_stream_mutation':'passed','repeated_consumer':'rejected'},indent=2))
 print('Verification transfer controls passed',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--reverse',action='store_true');p.add_argument('--output',required=True);main(p.parse_args())
