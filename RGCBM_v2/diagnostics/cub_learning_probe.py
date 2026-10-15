from pathlib import Path
import sys,json,time,random
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import torch,numpy as np
from torch import nn
from rgcbm.core import Config,RGCBM
from rgcbm.data import build_manifest,ImageDataset
from rgcbm.train import seed_all
import argparse
parser=argparse.ArgumentParser(description='Small CUB optimizer diagnostic, not a research benchmark.')
parser.add_argument('--root',type=Path,default=SOURCE)
parser.add_argument('--output',type=Path,default=SOURCE/'results/learning_diagnostics')
args=parser.parse_args();root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.set_num_interop_threads(2)
manifest=build_manifest(root,'cub');cfg=Config(batch_size=24)
seed_all(42);base=RGCBM(200,cfg,root/'assets').eval()
features=[];labels=[];splits=[]
for split,n in [('train',3),('validation',2)]:
 ds=ImageDataset(root,manifest,split,cfg);byclass={k:[] for k in range(200)}
 for i,row in enumerate(ds.rows):byclass[row['label']].append(i)
 inds=[i for k in byclass for i in byclass[k][:n]]
 for start in range(0,len(inds),24):
  batch=[ds[i] for i in inds[start:start+24]];x=torch.stack([b[0] for b in batch])
  with torch.no_grad():f=base.backbone((x-base.image_mean)/base.image_std).cpu()
  features.append(f);labels.extend([b[1] for b in batch]);splits.extend([split]*len(batch))
 print('extracted',split,len(inds),flush=True)
F=torch.cat(features);Y=torch.tensor(labels);train=torch.tensor([s=='train' for s in splits]);valid=~train
print('feature cache',F.shape,flush=True)
records=[]
for variant,lr,norm in [('legacy',4e-4,False),('higher_lr',2e-3,False),('standardized',2e-3,True)]:
 seed_all(42);m=RGCBM(200,Config(pretrained=False,concept_normalization='running_standardize' if norm else 'none'),None);m.backbone=nn.Identity();m.image_mean=torch.zeros(1,1280,1,1);m.image_std=torch.ones(1,1280,1,1)
 opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=1e-5)
 rows=[]
 with torch.no_grad():
  m.eval();initial=m(F[valid])['concepts'];print(variant,'initial concept std',initial.std(0).mean().item(),flush=True)
 for epoch in range(15):
  m.train();order=torch.where(train)[0][torch.randperm(train.sum())];correct=0;loss_sum=0
  for ids in order.split(24):
   opt.zero_grad();o=m(F[ids]);loss=nn.functional.cross_entropy(o['logits'],Y[ids]);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();correct+=(o['logits'].argmax(1)==Y[ids]).sum().item();loss_sum+=float(loss.detach())*len(ids)
  m.eval()
  with torch.no_grad():
   logits=[];cs=[]
   for ids in torch.where(valid)[0].split(24):
    o=m(F[ids]);logits.append(o['logits']);cs.append(o['concepts'])
   logits=torch.cat(logits);cs=torch.cat(cs)
   row={'epoch':epoch+1,'train_accuracy':correct/int(train.sum()),'train_loss':loss_sum/int(train.sum()),'validation_accuracy':float((logits.argmax(1)==Y[valid]).float().mean()),'validation_loss':float(nn.functional.cross_entropy(logits,Y[valid])),'mean_concept_std':float(cs.std(0).mean())};rows.append(row)
  print(variant,json.dumps(row),flush=True)
 records.append({'variant':variant,'head_lr':lr,'concept_standardization':norm,'history':rows})
 (out/'learning_probe.json').write_text(json.dumps({'scope':'Controlled optimizer diagnostic, frozen real ImageNet features; 600 unaugmented CUB training/400 validation images, class-balanced, 200 classes; no test data; not full training or paper accuracy. Auxiliary losses disabled for isolation.','records':records},indent=2))
