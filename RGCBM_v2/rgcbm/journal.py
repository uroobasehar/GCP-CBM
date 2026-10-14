"""Journal-facing XAI audits. All measurements preserve the trained classifier."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from .data import ImageDataset, loader, atomic_json


DOMAIN_SCOPE = {
    'cub': 'Bird species classification. Edge/texture/region associations alone do not establish a wing, bill, plumage color or anatomical part. Attribute-assisted labels require the separately reported annotation audit.',
    'pets': 'Pet breed classification. Texture/region associations alone do not establish fur type, muzzle anatomy or disease. Foreground trimap alignment is localization evidence, not proof of a breed-specific semantic concept.',
    'brain_mri': 'MRI image-label classification. Sequence/acquisition and patient independence are unverified. Brightness is not contrast enhancement; region priors are not tumor masks. Model class labels are not clinical diagnoses.',
}


def concept_cards(manifest, run, grounding):
    run = Path(run)
    W = None
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    W = checkpoint['model']['classifier.weight'].numpy()
    test = json.loads((run / 'test_frozen_alignment.json').read_text())
    test_by_id = {r['concept_index']: r for r in test}
    cards = []
    for row in grounding:
        k = row['concept_index']
        order = np.argsort(-W[:, k])[:3]
        cards.append({'id': f'C{k+1:02d}', 'name': row['concept_name'], 'name_basis': 'validation image-descriptor correlation, not anatomy/part annotation', 'group': row['training_group'], 'validation_correlation': row['mean'], 'validation_image_bootstrap_95ci': [row['ci_low'], row['ci_high']], 'weak_grounding': row['weak_grounding'], 'frozen_descriptor_test_audit': test_by_id[k], 'classes_with_largest_classifier_weights': [{'class': manifest['classes'][int(j)], 'weight': float(W[j, k])} for j in order], 'weight_interpretation': 'Relative classifier use; class weights do not name the visual concept.', 'domain_scope': DOMAIN_SCOPE[manifest['dataset']], 'human_review': 'Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.'})
    atomic_json(run / 'concept_cards.json', cards)
    text = '# Concept evidence cards\n\n' + DOMAIN_SCOPE[manifest['dataset']] + '\n\n'
    for card in cards:
        text += f"## {card['id']}: {card['name']}\n\nGroup: {card['group']}. Weak: {card['weak_grounding']}. Validation correlation: {card['validation_correlation']:.3f}; 95% image-bootstrap interval: {card['validation_image_bootstrap_95ci']}.\n\n{card['name_basis']}. {card['human_review']}\n\n"
    (run / 'concept_cards.md').write_text(text, encoding='utf-8')
    return pd.DataFrame([{k:v for k,v in row.items() if k not in ['classes_with_largest_classifier_weights','frozen_descriptor_test_audit']} for row in cards])


def reliability_and_redundancy(run):
    run = Path(run); figures = run / 'figures'; figures.mkdir(exist_ok=True)
    values = np.load(run / 'test_outputs.npz')
    p, y, c = values['probs'], values['labels'], values['concepts']
    conf, pred = p.max(1), p.argmax(1)
    bins = []
    for lo, hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        use = (conf > lo) & (conf <= hi)
        if use.any(): bins.append({'bin_low':lo,'bin_high':hi,'n':int(use.sum()),'mean_confidence':float(conf[use].mean()),'observed_accuracy':float((pred[use]==y[use]).mean())})
    pd.DataFrame(bins).to_csv(run/'reliability_bins.csv',index=False)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot([0,1],[0,1],'--',color='gray',label='perfect calibration')
    ax.scatter([r['mean_confidence'] for r in bins],[r['observed_accuracy'] for r in bins],s=[min(200,15+r['n']) for r in bins],color='#327da8')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean model confidence',ylabel='Observed accuracy',title='Reliability diagram (uncalibrated model)');ax.legend()
    fig.tight_layout();fig.savefig(figures/'reliability_diagram.png',dpi=180);plt.close(fig)
    if len(c)>=3:
        std=c.std(0);valid=std>1e-8
        matrix=np.full((c.shape[1],c.shape[1]),np.nan)
        if valid.sum()>=2:matrix[np.ix_(valid,valid)]=np.corrcoef(c[:,valid],rowvar=False)
        np.save(run/'concept_redundancy.npy',matrix)
        fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(np.ma.masked_invalid(matrix),vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=ax,label='Pearson correlation across test images')
        ax.set(xlabel='Concept index',ylabel='Concept index',title='Concept redundancy; gray/blank means insufficient variation')
        fig.tight_layout();fig.savefig(figures/'concept_redundancy.png',dpi=180);plt.close(fig)
    return pd.DataFrame(bins)


@torch.no_grad()
def intervention(model, image, grounding, classes, edits):
    """Edit latent values, never represent this as an image/medical intervention."""
    model.eval();device=next(model.parameters()).device
    output=model(image[None].to(device))
    original=output['concepts'].clone();changed=original.clone()
    for index,value in edits.items():
        if not 0<=index<changed.shape[1] or not 0<=float(value)<=1:raise ValueError('Use a valid concept index and activation in [0,1].')
        changed[0,index]=float(value)
    z0,p0=model.predict_from_concepts(original);z1,p1=model.predict_from_concepts(changed)
    expected=(changed-original) @ model.classifier.weight.T
    torch.testing.assert_close(z1-z0,expected,atol=1e-6,rtol=1e-5)
    rows=[{'class':name,'original_probability':float(p0[0,k]),'edited_probability':float(p1[0,k]),'probability_change':float(p1[0,k]-p0[0,k]),'exact_logit_change':float(expected[0,k])} for k,name in enumerate(classes)]
    return pd.DataFrame(rows).sort_values('edited_probability',ascending=False)


def _bootstrap(values,seed=42):
    values=np.asarray(values);g=np.random.default_rng(seed)
    if len(values)<2:return float(values.mean()) if len(values) else None,None,None
    distribution=[g.choice(values,len(values),replace=True).mean() for _ in range(200)]
    low,high=np.percentile(distribution,[2.5,97.5]);return float(values.mean()),float(low),float(high)


@torch.no_grad()
def deletion_curves(model,run,max_k=4):
    """Compare support vs random on exactly the same cases at each removal count."""
    run=Path(run);values=np.load(run/'test_outputs.npz');device=next(model.parameters()).device
    c=torch.tensor(values['concepts'],device=device);z=torch.tensor(values['logits'],device=device)
    order=z.argsort(1,descending=True);pred,runner=order[:,0],order[:,1]
    W=model.classifier.weight;terms=c*(W[pred]-W[runner]);ranking=terms.argsort(1,descending=True)
    original=z.softmax(1)[torch.arange(len(c)),pred]
    g=np.random.default_rng(model.config.seed);rows=[]
    for k in range(1,min(max_k,c.shape[1])+1):
        eligible=(terms>0).sum(1)>=k
        if not eligible.any():continue
        indices=eligible.nonzero().flatten();a=c[eligible].clone();r=a.clone()
        chosen=ranking[eligible,:k]
        a.scatter_(1,chosen,0)
        random_indices=torch.tensor(np.stack([g.choice(c.shape[1],size=k,replace=False) for _ in indices]),device=device)
        r.scatter_(1,random_indices,0)
        za,pa=model.predict_from_concepts(a);zr,pr=model.predict_from_concepts(r)
        predicted=pred[eligible];b=torch.arange(len(a),device=device)
        support_drop=(original[eligible]-pa[b,predicted]).cpu().numpy();random_drop=(original[eligible]-pr[b,predicted]).cpu().numpy()
        sm,sl,sh=_bootstrap(support_drop);rm,rl,rh=_bootstrap(random_drop);difference,dl,dh=_bootstrap(support_drop-random_drop)
        rows.append({'removed_concepts':k,'matched_n':len(indices),'support_probability_drop':sm,'support_ci_low':sl,'support_ci_high':sh,'random_probability_drop':rm,'random_ci_low':rl,'random_ci_high':rh,'paired_difference':difference,'paired_ci_low':dl,'paired_ci_high':dh})
    frame=pd.DataFrame(rows);frame.to_csv(run/'deletion_curves.csv',index=False)
    if rows:
        fig,ax=plt.subplots(figsize=(7,4))
        for prefix,label in [('support','positive margin support'),('random','random removal, matched cases')]:
            ax.plot(frame.removed_concepts,frame[prefix+'_probability_drop'],marker='o',label=label)
            if frame[prefix+'_ci_low'].notna().all():ax.fill_between(frame.removed_concepts,frame[prefix+'_ci_low'],frame[prefix+'_ci_high'],alpha=.15)
        ax.set(xlabel='Number of edited latent concepts',ylabel='Original predicted-class probability drop',title='Model dependence; eligible sample count can vary with k');ax.legend()
        fig.tight_layout();fig.savefig(run/'figures/deletion_curves.png',dpi=180);plt.close(fig)
    return frame


@torch.no_grad()
def profile_inference(model,image,run,repeats=20):
    model.eval();device=next(model.parameters()).device;x=image[None].to(device)
    for _ in range(3):model(x)
    if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    measurements=[]
    for _ in range(repeats):
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();model(x)
        if device.type=='cuda':torch.cuda.synchronize()
        measurements.append((time.perf_counter()-start)*1000)
    result={'device':str(device),'torch':torch.__version__,'cuda':torch.version.cuda,'batch_size':1,'precision':'FP32','repeats':repeats,'median_latency_ms':float(np.median(measurements)),'p95_latency_ms':float(np.percentile(measurements,95)),'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters_current_stage':sum(p.numel() for p in model.parameters() if p.requires_grad),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'includes_llm':False,'note':'Measures warmed classifier inference, not training or text generation.'}
    atomic_json(Path(run)/'inference_profile.json',result);return result


@torch.no_grad()
def cub_attribute_audit(model,root,manifest,run,max_batches=None):
    """Optional annotation-assisted post-training association audit; never updates the model."""
    if manifest['dataset']!='cub':raise ValueError('This audit uses CUB official attribute annotations only.')
    root,run=Path(root),Path(run);device=next(model.parameters()).device;model.eval()
    base=root/Path(manifest['rows'][0]['path']).parent
    annotation_files=list((root/'data/cub').rglob('image_attribute_labels.txt'))
    if len(annotation_files)!=1:raise RuntimeError('CUB original image attribute file is missing/ambiguous.')
    names_file=annotation_files[0].parent/'attributes.txt'
    names={int(line.split(maxsplit=1)[0]):line.split(maxsplit=1)[1] for line in names_file.read_text().splitlines() if line.strip()}
    collected={}
    for split in ['validation','test']:
        data=ImageDataset(root,manifest,split,model.config);values,ids=[],[]
        for batch,(images,_,indices) in enumerate(loader(data)):
            if max_batches is not None and batch>=max_batches:break
            values.append(model(images.to(device))['concepts'].cpu().numpy());ids.extend(data.rows[i]['image_id'] for i in indices.tolist())
        if not values:raise RuntimeError('Attribute audit executed zero images.')
        collected[split]={'ids':ids,'concepts':np.concatenate(values)}
    required=set(collected['validation']['ids']+collected['test']['ids'])
    attributes={index:np.full(312,np.nan,dtype=float) for index in required}
    with annotation_files[0].open() as source:
        for line in source:
            fields=line.split()
            if len(fields)<4:continue
            image_id,attribute_id,present,certainty=map(int,fields[:4])
            if image_id in attributes and certainty>=3:attributes[image_id][attribute_id-1]=present
    for split,value in collected.items():value['attributes']=np.stack([attributes[i] for i in value['ids']])
    validation,test=collected['validation'],collected['test'];rows=[]
    corrected_alpha=.05/(model.config.concepts*312)
    for k in range(model.config.concepts):
        candidates=[]
        for j in range(312):
            select=np.isfinite(validation['attributes'][:,j]);x=validation['concepts'][select,k];y=validation['attributes'][select,j]
            if len(x)<50 or len(np.unique(y))<2 or x.std()<1e-8:continue
            correlation,p_value=pearsonr(x,y);candidates.append((abs(correlation),j,float(correlation),float(p_value),len(x)))
        if not candidates:
            rows.append({'concept_index':k,'provisional_annotation_assisted_name':None,'status':'insufficient validation evidence; keep descriptor name'});continue
        _,j,correlation,p_value,n=max(candidates)
        reliable=abs(correlation)>=.2 and p_value<=corrected_alpha
        select=np.isfinite(test['attributes'][:,j]);tx=test['concepts'][select,k];ty=test['attributes'][select,j]
        auc=float(roc_auc_score(ty,tx*(1 if correlation>=0 else -1))) if len(np.unique(ty))==2 else None
        rows.append({'concept_index':k,'attribute':names[j+1],'validation_correlation':correlation,'validation_p_value':p_value,'multiple_comparison_alpha':corrected_alpha,'validation_n':n,'provisional_annotation_assisted_name':f'{"positive" if correlation>=0 else "inverse"} association with {names[j+1]}' if reliable else None,'status':'provisional validation-supported association' if reliable else 'weak association; keep descriptor name','independent_test_attribute_auc':auc,'test_n':len(ty),'annotation_use':'official attributes used after training only; NOT annotation-free semantic naming'})
    atomic_json(run/'cub_attribute_audit.json',{'names_selected_on':'validation only','test_never_renames':True,'attributes_used_for_training':False,'certainty_rule':'>=3','candidates':312,'concepts':rows,'note':'Association does not prove a spatial bird part. Default descriptor names remain unchanged. Report this separately from annotation-free concept discovery.'})
    return pd.DataFrame(rows)
