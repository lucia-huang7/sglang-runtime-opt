"""Adapt the checkpoint architecture label without modifying its weight files."""
import argparse,hashlib,json,os
from pathlib import Path

def main(a):
 source=Path(a.source).resolve();target=Path(a.output);target.mkdir(parents=True,exist_ok=True)
 original=(source/'config.json').read_bytes();config=json.loads(original)
 if config['architectures']!=['Qwen2ForCausalLM'] or config['num_hidden_layers']!=1:
  raise ValueError('Unexpected pinned EAGLE checkpoint configuration')
 config['architectures']=['Qwen2ForCausalLMEagle']
 config['max_position_embeddings']=8192
 for path in source.iterdir():
  if path.name=='config.json':continue
  destination=target/path.name
  if not destination.exists():destination.symlink_to(path.resolve())
 patched=json.dumps(config,indent=2)+'\n';(target/'config.json').write_text(patched)
 manifest={'source_revision':'1b9c01aa0c354e16e38ac6dd6b2b0b230be31e9f',
  'original_config_sha256':hashlib.sha256(original).hexdigest(),
  'adapted_config_sha256':hashlib.sha256(patched.encode()).hexdigest(),
  'change':{'architectures':{'before':['Qwen2ForCausalLM'],'after':['Qwen2ForCausalLMEagle']}},
  'reason':'Select the pinned SGLang EAGLE implementation so fc and decoder weights are loaded',
  'weights_modified':False,'rotary_cache_positions':8192,'original_rotary_cache_positions':2048}
 Path(a.manifest).write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);main(p.parse_args())
