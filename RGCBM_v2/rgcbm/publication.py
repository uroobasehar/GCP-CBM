"""Named concept figures and auditable spatial allocation; no causal saliency claim."""
from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch.nn import functional as F
from .data import atomic_json
from .semantics import ANCHORS, LEGACY, DOMAIN, descriptor_bank
from .evaluate import DESCRIPTORS, weighted_descriptor_values, evidence_packet

COLORS = {'boundary':'#277da8','texture':'#ad7031','region':'#6d569a','unguided':'#657a63'}


def label(row, width=30):
    return textwrap.fill(row['concept_name'], width)


def save(fig, path):
    path=Path(path)
    configuration=path.parent.parent/'configuration.json'
    if configuration.is_file():
        cfg=json.loads(configuration.read_text())['config']
        if cfg.get('max_train_batches') is not None or cfg.get('max_validation_batches') is not None:
            fig.text(.5,.005,'SOFTWARE SMOKE CHECK — not a research result',ha='center',fontsize=8,color='#a23030')
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def margin_allocation(maps, concepts, weights, predicted, runner):
    """Each concept's spatial mass integrates to its EXACT signed logit-margin term.

    Allocation is a visualization convention; this is neither unique localization nor a
    gradient/causal explanation. The classifier bias is displayed separately, never spread.
    """
    terms=concepts*(weights[predicted]-weights[runner])
    mass=maps/maps.sum((-2,-1),keepdim=True).clamp_min(1e-8)
    result=(terms[:,None,None]*mass).sum(0)
    torch.testing.assert_close(result.sum(),terms.sum(),atol=1e-5,rtol=1e-5)
    return result,terms


@torch.no_grad()
def publication_cases(model,root,manifest,run,grounding,test,selected):
    figures=Path(run)/'figures';device=next(model.parameters()).device
    packets=[];rows=[]
    for case,index in enumerate(selected):
        image,truth,_=test[int(index)];x=image[None].to(device);output=model(x);bank=descriptor_bank(x)
        obs=weighted_descriptor_values(output['maps'],bank)[0]
        packet=evidence_packet(model,output['concepts'][0].cpu().numpy(),output['logits'][0].cpu().numpy(),manifest['classes'],grounding,obs)
        packet.update(image_path=test.rows[int(index)]['path'],true_class=manifest['classes'][truth],test_index=int(index),figure_prefix=f'case_{case:02d}')
        packets.append(packet)
        rgb=image.permute(1,2,0).numpy();maps=F.interpolate(output['maps'],size=image.shape[-2:],mode='bilinear',align_corners=False)[0]
        order=output['logits'][0].argsort(descending=True);pred,runner=map(int,order[:2])
        allocation,terms=margin_allocation(maps,output['concepts'][0],model.classifier.weight,pred,runner)
        signed=allocation.cpu().numpy();magnitude=max(float(np.abs(signed).max()),1e-9)
        descending=terms.abs().argsort(descending=True).tolist()
        positive=[k for k in descending if terms[k]>0]
        negative=[k for k in descending if terms[k]<0]
        ranked=positive[:2]+negative[:1]
        ranked += [k for k in descending if k not in ranked][:3-len(ranked)]
        fig,axes=plt.subplots(1,5,figsize=(19,5.6))
        fig.subplots_adjust(top=.63,bottom=.04,wspace=.25)
        axes[0].imshow(rgb);axes[0].set_title('Input image\nTrue: '+manifest['classes'][truth],fontsize=10)
        for ax,k in zip(axes[1:4],ranked):
            ax.imshow(rgb);ax.imshow(maps[k].cpu(),cmap='magma',alpha=.48,vmin=0,vmax=1)
            ax.set_title(label(grounding[k],25)+f'\nMargin term {float(terms[k]):+.4f}',fontsize=8)
        axes[4].imshow(rgb);im=axes[4].imshow(signed,cmap='coolwarm',alpha=.68,vmin=-magnitude,vmax=magnitude)
        axes[4].set_title('Signed spatial margin allocation\nred supports / blue opposes',fontsize=9)
        fig.colorbar(im,ax=axes[4],fraction=.045,label='Allocated logit-margin mass / pixel')
        for ax in axes:ax.axis('off')
        fig.suptitle(f"Predicted {packet['prediction']['class']} vs {packet['prediction']['runner_up']} | uncalibrated p={packet['prediction']['probability']:.3f}\nSpatial sum={signed.sum():+.4f}; separate bias={packet['exact_decomposition']['margin_bias']:+.4f}; localization is a display convention",fontsize=10)
        save(fig,figures/f'case_{case:02d}_explanation_panel')
        np.savez_compressed(Path(run)/f'case_{case:02d}_spatial_evidence.npz',signed_margin_allocation=signed,concept_margin_terms=terms.cpu().numpy(),concept_maps=maps.cpu().numpy(),image_path=packet['image_path'])
        # Show intended guidance beside the actually learned map for the strongest guided patterns.
        guided=sorted([k for k,r in enumerate(grounding) if r.get('guidance_configured',r.get('guidance_applied'))],key=lambda k:abs(float(terms[k])),reverse=True)[:3]
        if guided:
            fig,axes=plt.subplots(len(guided),3,figsize=(10,3.6*len(guided)),squeeze=False)
            fig.subplots_adjust(top=.88,bottom=.05,hspace=.65,wspace=.28)
            for row,k in enumerate(guided):
                target=grounding[k]['intended_descriptor']
                axes[row,0].imshow(rgb);axes[row,0].set_title(label(grounding[k]),fontsize=9)
                axes[row,1].imshow(bank[target][0,0].cpu(),cmap='viridis',vmin=0,vmax=1);axes[row,1].set_title('Image-derived guidance target',fontsize=9)
                axes[row,2].imshow(rgb);axes[row,2].imshow(maps[k].cpu(),cmap='magma',alpha=.5,vmin=0,vmax=1)
                axes[row,2].set_title('Learned response\n'+grounding[k].get('grounding_status','legacy'),fontsize=9)
            for ax in axes.flat:ax.axis('off')
            fig.suptitle('Guidance intention versus learned spatial evidence; target is not a lesion/part annotation',fontsize=10)
            save(fig,figures/f'case_{case:02d}_guidance_comparison')
        rows.append({'case':case,'image_path':packet['image_path'],'test_index':int(index),'true_class':packet['true_class'],'predicted_class':packet['prediction']['class'],'figure':f'figures/case_{case:02d}_explanation_panel.pdf','report':f'reports/case_{case:03d}.md','selection_policy':'evenly spaced indices in the official test listing; not selected for correct prediction'})
    atomic_json(Path(run)/'visual_example_evidence.json',packets)
    pd.DataFrame(rows).to_csv(Path(run)/'illustrative_cases.csv',index=False)


def grounding_figures(run,grounding):
    figures=Path(run)/'figures'
    values=np.array([[r['all_descriptor_statistics'][name]['mean'] for name in DESCRIPTORS] for r in grounding])
    fig,ax=plt.subplots(figsize=(14,max(7,len(grounding)*.42)))
    im=ax.imshow(values,aspect='auto',vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=ax,label='Mean validation spatial correlation')
    ax.set_yticks(range(len(grounding)),[label(r,47) for r in grounding],fontsize=7)
    names={**LEGACY,**ANCHORS}
    ax.set_xticks(range(len(DESCRIPTORS)),[names[n][1] for n in DESCRIPTORS],rotation=60,ha='right',fontsize=8)
    for k,row in enumerate(grounding):
        if row.get('guidance_configured',row.get('guidance_applied')):
            j=DESCRIPTORS.index(row['intended_descriptor']);ax.add_patch(Rectangle((j-.5,k-.5),1,1,fill=False,edgecolor='black',lw=1.8))
    ax.set_title('Per-concept grounding audit; outlined cells are the actual guidance targets\nNames and statistics selected on validation only; black outline does not mean verified learning',fontsize=10)
    save(fig,figures/'guidance_alignment_matrix')
    table=pd.DataFrame([{k:v for k,v in r.items() if k!='all_descriptor_statistics'} for r in grounding])
    table.to_csv(Path(run)/'named_concept_dictionary.csv',index=False)


def prototype_plates(run,grounding,best,train):
    figures=Path(run)/'figures';records=[]
    for k,candidates in enumerate(best):
        fig,axes=plt.subplots(2,3,figsize=(10,6),squeeze=False)
        for rank,candidate in enumerate(sorted(candidates,key=lambda v:v[0],reverse=True)):
            value,index,patch,rgb,bounds,grid,label_id=candidate
            axes[0,rank].imshow(rgb);axes[0,rank].imshow(F.interpolate(torch.tensor(grid)[None,None],size=rgb.shape[:2],mode='bilinear',align_corners=False)[0,0],cmap='magma',alpha=.4,vmin=0,vmax=1)
            x0,y0,x1,y1=bounds;axes[0,rank].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#00d4d8',lw=1.5))
            axes[0,rank].set_title(f'Training image; peak={value:.3f}',fontsize=9)
            axes[1,rank].imshow(patch);axes[1,rank].set_title('Patch at peak response',fontsize=9)
            records.append({'concept_index':k,'concept_name':grounding[k]['concept_name'],'rank':rank+1,'image_path':train.rows[index]['path'],'class_label':label_id,'peak_response':value,'crop_xyxy':list(bounds),'figure':f'figures/prototype_{k+1:02d}.pdf'})
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(textwrap.fill(grounding[k]['concept_name'],90)+'\n'+grounding[k]['training_group']+' | '+grounding[k].get('grounding_status','legacy')+' | training examples are not semantic proof',fontsize=10)
        save(fig,figures/f'prototype_{k+1:02d}')
    pd.DataFrame(records).to_csv(Path(run)/'prototype_provenance.csv',index=False)
    (Path(run)/'PAPER_FIGURE_CAPTIONS.md').write_text('''# Figure interpretation and export guide

All named concept/prototype and explanation panels are exported as 300 dpi PNG plus PDF/SVG.
The alignment matrix compares validation concept maps against measurable image descriptors;
outlined cells identify intended guidance, not evidence that guidance succeeded. Unverified and
conflicting targets remain labeled. The table records n, bootstrap intervals and BH-FDR q values.

Prototype plates show the original training image, model response, peak crop and grounding status.
Paths/ranks/crop coordinates are in prototype_provenance.csv. A prototype is an exemplar of model
response, not proof of a bird part, fur property or tumor anatomy.

Case panels pair named concept heatmaps with exact signed predicted-versus-runner-up contributions.
The final spatial panel allocates each concept term in proportion to its nonnegative spatial map;
its pixel sum equals the concept component of the logit margin. Bias is separate. This allocation
is a visualization convention, not unique localization, Grad-CAM or a causal image explanation.
illustrative_cases.csv links every figure to the SAME image's evidence and LLM report. Examples are
selected by evenly spaced test-list indices, not cherry-picked for successful prediction.

Guidance comparison panels show the image-derived target beside learned response; clinical regions
and anatomical parts are not annotated by these weak maps. Independent localization/annotation and
human review are required for stronger domain claims. Smoke figures are software demonstrations,
not journal experimental findings. Include full-run sample counts, seeds and held-out results.
''',encoding='utf-8')
