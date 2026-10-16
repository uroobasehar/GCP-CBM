"""Diagnostic only: full CUB frozen features, simplified objective, no test evaluation.
Requires roughly 5 GB free RAM for feature caches; 10 CPU head epochs; fixed seed42.
This is not the full RG-CBM training/evaluation protocol.
"""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from torch import nn
from rgcbm.core import Config,RGCBM
from rgcbm.data import ImageDataset,build_manifest
from rgcbm.train import seed_all
from rgcbm.semantics import descriptor_bank,KEYS
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True);root=args.root
torch.set_num_threads(4);torch.set_num_interop_threads(2);seed_all(42)
cfg=Config(batch_size=24);manifest=build_manifest(root,'cub');base=RGCBM(200,cfg,root/'assets').eval()
F=[];Y=[];D=[];splits=[]
for split in ['train','validation']:
 ds=ImageDataset(root,manifest,split,cfg,augment=split=='train')
 for start in range(0,len(ds),24):
  batch=[ds[i] for i in range(start,min(start+24,len(ds)))];x=torch.stack([b[0] for b in batch])
  with torch.no_grad():
   f=base.backbone((x-base.image_mean)/base.image_std);bank=descriptor_bank(x)
   target=torch.cat([torch.nn.functional.interpolate(bank[n],size=f.shape[-2:],mode='bilinear',align_corners=False) for n in KEYS],dim=1)
  F.append(f.half());D.append(target.half());Y.extend([b[1] for b in batch]);splits.extend([split]*len(batch))
  if start%480==0:print('extract',split,start,len(ds),flush=True)
 print('cached',split,len(ds),flush=True)
F=torch.cat(F);D=torch.cat(D);Y=torch.tensor(Y);train=torch.tensor([s=='train' for s in splits]);val=~train;del base
records=[]
for variant,K in [('pooled_feature_baseline',0),('guided16',16),('guided64',64)]:
 seed_all(42)
 if K:
  m=RGCBM(200,Config(pretrained=False,concepts=K,concept_normalization='running_standardize'),None);m.backbone=nn.Identity();m.image_mean=torch.zeros(1,1280,1,1);m.image_std=torch.ones(1,1280,1,1)
 else:m=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(.1),nn.Linear(1280,200))
 opt=torch.optim.AdamW(m.parameters(),lr=.002 if K else .003,weight_decay=1e-5);history=[]
 for epoch in range(10):
  m.train();order=torch.where(train)[0][torch.randperm(int(train.sum()))];correct=0;total=0.
  for ids in order.split(24):
   if len(ids)<2:continue
   opt.zero_grad();output=m(F[ids].float());logits=output['logits'] if K else output
   loss=nn.functional.cross_entropy(logits,Y[ids])
   if K:
    group=K//4;indices=[j*4+(q%4) for j in range(3) for q in range(group)]
    for j,weight in enumerate([.18,.10,.08]):loss=loss+min(1,max(0,epoch-2)/5)*weight*nn.functional.mse_loss(output['maps'][:,j*group:(j+1)*group],D[ids][:,indices[j*group:(j+1)*group]].float())
   loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();correct+=(logits.argmax(1)==Y[ids]).sum().item();total+=float(loss.detach())*len(ids)
  m.eval();vcorrect=0;vloss=0
  with torch.no_grad():
   for ids in torch.where(val)[0].split(24):
    o=m(F[ids].float());z=o['logits'] if K else o;vcorrect+=(z.argmax(1)==Y[ids]).sum().item();vloss+=float(nn.functional.cross_entropy(z,Y[ids],reduction='sum'))
  row={'epoch':epoch+1,'training_accuracy':correct/int(train.sum()),'training_objective':total/int(train.sum()),'validation_accuracy':vcorrect/int(val.sum()),'validation_ce':vloss/int(val.sum())};history.append(row);print(variant,json.dumps(row),flush=True)
 records.append({'variant':variant,'concepts':K,'history':history})
 (out/'full_cub_probe.json').write_text(json.dumps({'scope':'Frozen-feature CPU learning diagnostic: full official training pool split 5394/600, single cached augmentation per training image; 10 head-only epochs; simplified guidance objective, no fine-tuning/stability/diversity; not complete RG-CBM benchmark; no test data.', 'records':records},indent=2))
 del m
