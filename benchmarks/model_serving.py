"""Deterministic closed-loop HTTP controls with per-request token IDs."""
import argparse, asyncio, datetime, json, statistics, time
from pathlib import Path
import aiohttp

async def main(a):
    if a.kind=='speculative' and a.mode=='target' and a.run>1:
        Path(a.output).write_text(json.dumps({'status':'reference_reused','reference_run':1,'run':a.run,
            'mode':'target','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
        print('Target-only context reference uses run 1',flush=True)
        return
    rows=[]
    tokenizer=None
    if a.prompt_kind=='natural':
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer)
        natural_context=tokenizer.encode('Binary search maintains a sorted interval. It compares the middle element with the requested value and keeps the relevant half. '*1200,add_special_tokens=False)
    def prompt(index,length):
        if tokenizer is None:return [100+index]+[200+(i*17+index)%8000 for i in range(length-1)]
        head=tokenizer.encode(f'<|im_start|>user\nQuestion {index}. Read the context and explain how binary search works. Context: ',add_special_tokens=False)
        tail=tokenizer.encode('\nGive a clear explanation with time complexity.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
        context=natural_context
        assert len(head)+len(tail)<length
        return head+context[:length-len(head)-len(tail)]+tail
    timeout=aiohttp.ClientTimeout(total=240)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base=f'http://127.0.0.1:{a.port}'
        # Start the token manager's response loop before control RPCs on a cold server.
        async with session.post(base+'/generate',json={'input_ids':[100,200],
                'sampling_params':{'temperature':0,'max_new_tokens':1,'ignore_eos':True}}) as response:
            if response.status!=200:raise RuntimeError(await response.text())
            await response.read()
        async with session.get(base+'/get_server_info') as response:
            server=await response.json()
        if a.kind=='speculative':
            expected=None if a.mode=='target' else 'EAGLE'
            if server.get('speculative_algorithm')!=expected:
                raise RuntimeError('Serving method does not match the live server configuration')
        server_identity={key:server.get(key) for key in ['speculative_algorithm','speculative_draft_model_path','dtype','attention_backend','tp_size','disable_cuda_graph']}
        for length in ([4096,1024,128] if tokenizer is not None else [128,1024,4096]):
            natural_prompts={i:prompt(i,length) for i in list(range(a.requests))+list(range(1000,1004))} if tokenizer is not None else None
            for concurrency in [1,4,16]:
                async with session.post(base+'/flush_cache') as response:
                    if response.status!=200:raise RuntimeError(await response.text())
                semaphore=asyncio.Semaphore(concurrency)
                async def request(index,warmup=False):
                    ids=natural_prompts[index] if natural_prompts is not None else prompt(index,length)
                    payload={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':a.tokens,'ignore_eos':True},'return_logprob':True,'logprob_start_len':-1}
                    async with semaphore:
                        start=time.perf_counter()
                        async with session.post(base+'/generate',json=payload) as response:
                            body=await response.json()
                            if response.status!=200:raise RuntimeError(str(body))
                        elapsed=time.perf_counter()-start
                    return {'request':index,'elapsed_s':elapsed,'output_ids':body.get('output_ids'),
                            'text':body.get('text'),'meta_info':body.get('meta_info'),
                            'token_ids':[entry[1] for entry in body.get('meta_info',{}).get('output_token_logprobs',[])]}
                await asyncio.gather(*(request(i+1000,True) for i in range(min(4,concurrency))))
                async with session.post(base+'/flush_cache') as response:await response.read()
                start=time.perf_counter();samples=await asyncio.gather(*(request(i) for i in range(a.requests)));wall=time.perf_counter()-start
                latencies=sorted(s['elapsed_s'] for s in samples)
                row={'input_tokens':length,'concurrency':concurrency,'requests':a.requests,'requested_output_tokens':a.tokens,
                     'wall_s':wall,'median_request_s':statistics.median(latencies),'p95_request_s':latencies[min(len(latencies)-1,int(.95*len(latencies)))],
                     'output_tokens':sum(s['meta_info'].get('completion_tokens',0) for s in samples),'samples':samples}
                row['output_tokens_per_s']=row['output_tokens']/wall;rows.append(row)
                Path(a.output).write_text(json.dumps({'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'kind':a.kind,'mode':a.mode,'run':a.run,'prompt_kind':a.prompt_kind,'server_identity':server_identity,'cuda_graph':False,'attention_backend':'triton',
                    'request_protocol':'closed-loop with deterministic integer token prompts; reduced serving control',
                    'planned_requests':128,'executed_requests':a.requests,'planned_output_tokens':128,'executed_output_tokens':a.tokens,'rows':rows},indent=2))
                print(length,concurrency,round(wall,3),round(row['output_tokens_per_s'],2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prompt-kind',choices=['synthetic','natural'],default='synthetic');p.add_argument('--tokenizer');p.add_argument('--port',type=int,default=31000);p.add_argument('--requests',type=int,default=16);p.add_argument('--tokens',type=int,default=32);p.add_argument('--mode',required=True);p.add_argument('--kind',default='shared-experts');p.add_argument('--run',type=int,default=1);p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
