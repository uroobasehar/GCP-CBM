"""Regression checks for concrete prose failures observed in local LLM inference."""
from pathlib import Path
import argparse,json,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgcbm.reports import template_report,validate_report,write_report_markdown
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);args=p.parse_args()
packet=json.loads(args.evidence.read_text())[0];fixed=template_report(packet);validate_report(fixed,packet)
concepts={r['concept_index']:r for r in packet['concepts']}
a=concepts[fixed['supporting_concept_indices'][0]]['name'];b=concepts[fixed['opposing_concept_indices'][0]]['name']
good={**fixed,'explanation':f'{a} provides model support; {b} provides model opposition.','limitations':'Unverified descriptor association; human review required.'}
validate_report(good,packet,prose_only=True)
for bad in [dict(good,explanation=f'{a} is lower compared to the other class; {b} opposes.'),dict(good,explanation=good['explanation']+' Confidence is 0.95.'),dict(good,explanation=f'{a} provides strong guidance, while {b} opposes.'),dict(good,explanation=f'{a} provides support.'),dict(good,explanation=good['explanation']+' This is a diagnosis.'),dict(good,explanation=good['explanation']+' This indicates malignancy.'),dict(good,explanation=good['explanation']+' It has higher intensity.'),dict(good,limitations=[]),dict(good,confidence=99)]:
 try:validate_report(bad,packet,prose_only=True)
 except ValueError:pass
 else:raise AssertionError('Known invalid output was accepted: '+str(bad))
with tempfile.TemporaryDirectory() as folder:
 write_report_markdown(folder,0,packet,{'result':good,'backend_used':'local_qwen'})
 text=(Path(folder)/'case_000.md').read_text();assert '| Margin contribution |' in text and f"{packet['prediction']['probability']:.4f}" in text
print(json.dumps({'status':'passed','checks':['deterministic template valid','qualitative named support/opposition accepted','reject unsupported class descriptor comparisons','reject invented numbers','reject established-guidance claims with weak grounding','reject missing opposition name','reject diagnosis claims','reject non-string prose','reject modified prediction confidence']}))
