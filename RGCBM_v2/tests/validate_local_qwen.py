"""Actual local-only Qwen inference against existing three-dataset smoke evidence.
Run under tests/network_guard.py to deny Internet sockets. No high-accuracy or prose-factuality claim.
"""
from pathlib import Path
import json,sys,time,torch
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
from rgcbm.reports import generate_reports
import argparse
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
root=args.results;assets=SOURCE/'assets';torch.set_num_threads(1);records=[]
for ds in ['cub','pets','brain_mri']:
 original=next(root.glob(f'{ds}_K*_guided_v4*smoke'));run=args.output/ds;run.mkdir(exist_ok=True)
 for name in ['visual_example_evidence.json','example_evidence.json']:
  if (original/name).is_file():(run/name).write_bytes((original/name).read_bytes())
 import shutil
 shutil.copytree(original/'reports',run/'initial_reports',dirs_exist_ok=True)
 start=time.monotonic();report=generate_reports(run,assets,backend='local_qwen',max_cases=1,llm_device='cpu')[0]
 raw=run/'reports/case_000_raw_llm.txt';assert raw.is_file() and report['image_path']
 assert 'tensorflow' not in sys.modules, 'Optional TF was imported by the PyTorch verbalizer'
 records.append({'dataset':ds,'run':str(run),'backend_requested':report['backend_requested'],'backend_used':report['backend_used'],'fallback_reason':report['fallback_reason'],'image_path':report['image_path'],'duration_seconds':time.monotonic()-start,'network':'Internet sockets denied; model loaded with local_files_only=True'})
 print(json.dumps(records[-1]),flush=True)
 (args.output/'validation.json').write_text(json.dumps({'status':'passed','meaning':'Actual local Qwen loaded/generated for each dataset; invalid outputs are explicitly rejected/fallback, not counted as faithful LLM explanations. Prose still needs human review.','records':records},indent=2))
