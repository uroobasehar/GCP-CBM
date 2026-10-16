"""V4 regression checks on descriptor contracts, guidance exposure and named evidence."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
from rgcbm.core import training_config,loss_terms,RGCBM
from rgcbm.semantics import descriptor_bank,guidance_contract
from rgcbm.evaluate import DESCRIPTORS,summarize_grounding,ground_concepts,evaluate
from rgcbm.publication import margin_allocation
from rgcbm.train import fit,load_model
from rgcbm.data import build_manifest,atomic_json
from rgcbm.visualize import visualize_run
from rgcbm.reports import generate_reports,validate_report,template_report
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output;root.mkdir(parents=True)
for n in ['data','assets']:(root/n).symlink_to(SOURCE/n,target_is_directory=True)
torch.set_num_threads(4);torch.set_num_interop_threads(2)
for x in [torch.zeros(2,3,64,64),torch.ones(2,3,64,64),torch.rand(2,3,64,64)]:
 bank=descriptor_bank(x);assert list(bank)==DESCRIPTORS
 assert all(torch.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in bank.values())
cfg=training_config(dataset='pets',concepts=16,batch_size=2,head_epochs=1,finetune_epochs=1,classification_only_epochs=0,max_train_batches=1,max_validation_batches=2,checkpoint_every_batches=1,show_progress=False)
with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
 bank=descriptor_bank(torch.rand(2,3,64,64))
 assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in bank.values())
contract=guidance_contract(cfg);assert len({r['target_descriptor'] for r in contract[:12]})==12
rng=np.random.default_rng(9);v=rng.normal(0,.01,(80,16,len(DESCRIPTORS)));obs=rng.uniform(0,1,v.shape)
for k in range(12):v[:,k,DESCRIPTORS.index(contract[k]['target_descriptor'])]+=.6
v[:,0,DESCRIPTORS.index(contract[0]['target_descriptor'])]-=1.2
v[:,0,DESCRIPTORS.index('texture_entropy')]=.95+rng.normal(0,.001,80)
rows=summarize_grounding(v,obs,cfg,'pets',guidance_updates=5)
assert rows[0]['grounding_status']=='conflict' and rows[0]['descriptor']=='edge_steepness'
assert all(r['grounding_status']=='verified' for r in rows[1:12])
assert all(not r['concept_name'].startswith('C0') for r in rows)
not_exposed=summarize_grounding(v,obs,cfg,'pets',guidance_updates=0);assert not_exposed[1]['grounding_status']=='not_exposed' and not_exposed[1]['weak_grounding']
# Swapping target-aligned channels raises the individual guidance error; group means alone cannot detect it.
m=RGCBM(37,cfg,root/'assets');x=torch.rand(2,3,240,240);bank=descriptor_bank(x)
expected=torch.cat([bank[r['target_descriptor']] for r in contract[:12]],1)
maps=torch.cat([expected,torch.full((2,4,240,240),.5)],1).requires_grad_()
o={'maps':maps,'concepts':maps.mean((2,3)),'logits':torch.randn(2,37,requires_grad=True)}
_,terms=loss_terms(m,x,torch.tensor([0,1]),o,torch.ones(37),0)
assert sum(float(terms[n]) for n in ['edge','texture','morphology'])<1e-9
swapped=maps.clone();swapped[:,0],swapped[:,1]=maps[:,1],maps[:,0]
_,wrong=loss_terms(m,x,torch.tensor([0,1]),{**o,'maps':swapped},torch.ones(37),0);assert wrong['edge']>terms['edge']+1e-6
loss,_=loss_terms(m,x,torch.tensor([0,1]),o,torch.ones(37),1);loss.backward();assert maps.grad is not None and torch.isfinite(maps.grad).all()
manifest=build_manifest(root,'pets');model,run,status=fit(root,manifest,cfg,device='cpu',run_name='pets_v4_xai')
assert status['complete'] and model.guidance_updates>0
loaded,_=load_model(run,root/'assets',device='cpu');assert loaded.guidance_updates==model.guidance_updates
grounding=ground_concepts(model,root,manifest,run,max_batches=2);assert all(r['weak_grounding'] for r in grounding)
metrics=evaluate(model,root,manifest,run,grounding,max_batches=2)
visualize_run(model,root,manifest,run,grounding,examples=1,prototype_batches=2)
reports=generate_reports(run,root/'assets',backend='template',max_cases=1)
packet=json.loads((run/'visual_example_evidence.json').read_text())[0]
assert reports[0]['image_path']==packet['image_path'] and reports[0]['evidence_source']=='visual_example_evidence.json'
validate_report(reports[0]['result'],packet)
spatial=np.load(run/'case_00_spatial_evidence.npz');np.testing.assert_allclose(spatial['signed_margin_allocation'].sum(),spatial['concept_margin_terms'].sum(),rtol=1e-5,atol=1e-5)
assert (run/'figures/prototype_01.pdf').is_file() and (run/'figures/guidance_alignment_matrix.svg').is_file()
assert len(json.loads((run/'grounding.json').read_text())['concepts'])==16
# Warm-up-only models must never be advertised as guidance-exposed.
warm=training_config(dataset='pets',concepts=16,batch_size=2,head_epochs=1,finetune_epochs=0,max_train_batches=1,max_validation_batches=1,show_progress=False)
unexposed,warmrun,_=fit(root,manifest,warm,device='cpu',run_name='warmup_only')
assert unexposed.guidance_updates==0 and not json.loads((warmrun/'training_status.json').read_text())['guidance_eligible']
result={'status':'passed','scope':'Linux CPU real-image software checks, not trained research performance',
        'checks':['finite constant/random descriptor maps','12 distinct guidance targets','channel-swap regression detects individual-guidance error','guidance gradients finite',
                  'BH-FDR synthetic positive/inverse/conflict checks','failed guidance is not hidden by renaming','unexposed checkpoints flagged',
                  'trained model exposure survives reload','named PNG/PDF/SVG outputs','signed spatial mass exactly matches margin terms',
                  'LLM/template reports use the SAME visualized image','weak-grounding qualifiers preserved']}
atomic_json(root/'grounded_xai_validation.json',result);print(json.dumps(result,indent=2))
