"""Real EAGLE requests covering mixed lengths, stop tokens and cancellation."""
import argparse,asyncio,datetime,json,time
from pathlib import Path
import aiohttp

async def main(a):
 base=f'http://127.0.0.1:{a.port}';rows=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
  async def generate(index,length,maximum,stop=None,rid=None):
   payload={'input_ids':[100+index]+[210+(j*13+index)%4000 for j in range(length-1)],
    'sampling_params':{'temperature':0,'max_new_tokens':maximum,'ignore_eos':True},'return_logprob':True,'logprob_start_len':-1}
   if stop is not None:
    payload['sampling_params']['stop_token_ids']=[stop]
    payload['sampling_params']['ignore_eos']=False
   if rid is not None:payload['rid']=rid
   async with session.post(base+'/generate',json=payload) as response:
    body=await response.json()
    if response.status!=200:raise RuntimeError(str(body))
   ids=[entry[1] for entry in body['meta_info'].get('output_token_logprobs',[])]
   return {'index':index,'prompt_tokens':length,'maximum':maximum,'stop_token':stop,
    'token_ids':ids,'finish_reason':body['meta_info']['finish_reason'],'completion_tokens':body['meta_info']['completion_tokens']}
  for length in [8,128,1024]:
   rows.extend(await asyncio.gather(*(generate(i,length,maximum) for i,maximum in enumerate([1,2,8,16]))))
  reference=await generate(29,16,8)
  if len(reference['token_ids'])>1:
   stopped=await generate(29,16,16,stop=reference['token_ids'][1])
   assert stopped['completion_tokens']==2,stopped
   rows.append(stopped)
  # Abort is issued through the scheduler API while a longer request is in flight.
  rid=f'validation-cancel-{a.label}'
  pending=asyncio.create_task(generate(80,128,256,rid=rid));await asyncio.sleep(.05)
  async with session.post(base+'/abort_request',json={'rid':rid}) as response:
   abort_status=response.status;await response.read()
  cancelled=await pending;rows.append(cancelled)
  assert cancelled['finish_reason']['type']=='abort',cancelled
 Path(a.output).write_text(json.dumps({'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'label':a.label,'abort_status':abort_status,'rows':rows,'status':'passed'},indent=2))
 print('Speculative request validation passed',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=31002);p.add_argument('--label',required=True);p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
