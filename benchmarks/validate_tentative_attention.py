"""Compare masked cached-prefix attention with per-prefix FP32 oracles."""
import argparse,datetime,json,math
from pathlib import Path
import torch
import triton
from runtime_opt.tentative_attention import _prefix_mask
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd


def main(a):
 torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_device(a.device);device=torch.device('cuda',a.device);rows=[]
 torch.manual_seed(193)
 for base in [0,1,7,129,1024,4096]:
  for depth in [1,2,4]:
   width=base+2*depth
   lengths=torch.tensor([base],device=device,dtype=torch.int32);offsets=torch.tensor([0,depth*width],device=device,dtype=torch.int32)
   mask=torch.empty(depth*width,device=device,dtype=torch.bool)
   _prefix_mask[(1,triton.cdiv(depth*width,256))](mask,lengths,offsets,depth,256)
   cpu=torch.tensor([[column<=base+query or base+depth<=column<=base+depth+query for column in range(width)] for query in range(depth)])
   assert torch.equal(mask.reshape(depth,width).cpu(),cpu)
   heads,dim=4,128
   query=torch.randn(depth,heads,dim,device=device,dtype=torch.bfloat16)
   key=torch.randn(base+depth,heads,dim,device=device,dtype=torch.bfloat16);value=torch.randn_like(key)
   output=torch.empty_like(query)
   qo=torch.tensor([0,depth],device=device,dtype=torch.int32);kv=torch.tensor([0,base+depth],device=device,dtype=torch.int32)
   indices=torch.arange(base+depth,device=device,dtype=torch.int32)
   extend_attention_fwd(query,key[base:].contiguous(),value[base:].contiguous(),output,key,value,qo,kv,indices,mask,True,offsets,depth,skip_prefix_custom_mask=False)
   references=[]
   for position in range(depth):
    keys=torch.cat([key[:base+position+1],key[base:base+position+1]]).float()
    values=torch.cat([value[:base+position+1],value[base:base+position+1]]).float()
    weights=torch.softmax(torch.einsum('hd,khd->hk',query[position].float(),keys)/math.sqrt(dim),dim=-1)
    references.append(torch.einsum('hk,khd->hd',weights,values))
   reference=torch.stack(references)
   torch.testing.assert_close(output.float(),reference,rtol=.03,atol=.03)
   rows.append({'base_tokens':base,'depth':depth,'mask_exact':True,'max_abs':float((output.float()-reference).abs().max())})
 Path(a.output).write_text(json.dumps({'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'status':'passed'},indent=2));print('Masked attention validation passed',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--output',required=True);main(p.parse_args())
