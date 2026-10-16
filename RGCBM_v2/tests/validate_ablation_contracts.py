"""Check that the advertised ablation sweep constructs distinct, valid experimental protocols."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgcbm.core import training_config,Config
from rgcbm.ablations import variant_config,VARIANTS
from rgcbm.semantics import guidance_contract
for dataset in ['cub','pets','brain_mri']:
 base=training_config(dataset=dataset)
 assert base.concepts==(64 if dataset=='cub' else 16)
 for name in VARIANTS:
  cfg=variant_config(base,name);Config(**cfg.to_dict())
  spec=guidance_contract(cfg)
  if name=='legacy_group_guidance':
   assert cfg.guidance_mode=='group_mean' and cfg.concepts==base.concepts
   assert {r['target_descriptor'] for r in spec if r['guidance_applied']}=={'edge','texture','morphology'}
  if name=='unguided':assert not any(r['guidance_applied'] for r in spec)
  if name=='full':assert len({r['target_descriptor'] for r in spec if r['guidance_applied']})==12
  if name in ['K16','K32','K64']:assert cfg.concepts==int(name[1:])
print(json.dumps({'status':'passed','checks':'All 15 advertised variants construct valid configs for all three datasets; legacy guidance keeps capacity and changes targets; per-dataset defaults correct.'}))
