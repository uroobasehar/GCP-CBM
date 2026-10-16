from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from .core import priors
from .data import ImageDataset, loader, atomic_json

from .semantics import KEYS, descriptor_bank
DESCRIPTORS = ['edge', 'texture', 'morphology', 'intensity'] + KEYS


def spatial_correlations(maps, descriptors):
    a = maps.float().flatten(2)
    a = a - a.mean(2, keepdim=True)
    results = []
    for name in DESCRIPTORS:
        d = F.interpolate(descriptors[name], size=maps.shape[-2:], mode='bilinear', align_corners=False).flatten(2)
        d = d - d.mean(2, keepdim=True)
        denominator = a.square().sum(2).sqrt() * d.square().sum(2).sqrt()
        correlation = (a * d).sum(2) / denominator.clamp_min(1e-8)
        correlation = correlation.masked_fill(denominator < 1e-8, float('nan'))
        results.append(correlation)
    return torch.stack(results, dim=2).cpu().numpy()


def summarize_correlation(values, seed):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {'mean': 0., 'ci_low': 0., 'ci_high': 0., 'valid_n': len(values)}
    generator = np.random.default_rng(seed)
    # Resample images, never individual spatial pixels, for the uncertainty estimate.
    means = [generator.choice(values, size=len(values), replace=True).mean() for _ in range(200)]
    low, high = np.percentile(means, [2.5, 97.5])
    return {'mean': float(values.mean()), 'ci_low': float(low), 'ci_high': float(high), 'valid_n': len(values)}


def weighted_descriptor_values(maps, descriptors):
    weights = maps.float().flatten(2)
    weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-8)
    bank = torch.cat([F.interpolate(descriptors[name], size=maps.shape[-2:], mode='bilinear', align_corners=False) for name in DESCRIPTORS], dim=1).flatten(2)
    return torch.einsum('bks,bds->bkd', weights, bank).cpu().numpy()


def _bh(pvalues):
    flat = np.asarray(pvalues).ravel(); order = np.argsort(flat)
    adjusted = np.minimum.accumulate((flat[order] * len(flat) / np.arange(1, len(flat)+1))[::-1])[::-1]
    result = np.ones_like(flat); result[order] = np.minimum(1, adjusted)
    return result.reshape(np.shape(pvalues))


def summarize_grounding(values, observations, config, dataset, guidance_updates=None):
    from scipy.stats import ttest_1samp
    from .semantics import guidance_contract, readable_name
    contract = guidance_contract(config)
    pvalues = np.ones(values.shape[1:])
    stats = []
    for k in range(config.concepts):
        record = {}
        for j, name in enumerate(DESCRIPTORS):
            valid = values[:,k,j]; valid = valid[np.isfinite(valid)]
            record[name] = summarize_correlation(valid, config.seed+k*len(DESCRIPTORS)+j)
            if len(valid)>=2 and valid.std()>1e-8:
                pvalues[k,j] = float(ttest_1samp(valid,0).pvalue)
            elif len(valid)>=2 and abs(valid.mean())>1e-8:
                pvalues[k,j] = 0.
        stats.append(record)
    qvalues = _bh(pvalues)
    rows = []
    for k, spec in enumerate(contract):
        summary = stats[k]
        for j, name in enumerate(DESCRIPTORS):summary[name]['fdr_q'] = float(qvalues[k,j])
        best = max(summary, key=lambda name:abs(summary[name]['mean']))
        chosen = spec['target_descriptor'] if spec['guidance_applied'] else best
        result = summary[chosen]
        reliable = result['valid_n']>=50 and abs(result['mean'])>=.10 and result['ci_low']*result['ci_high']>0 and result['fdr_q']<.05
        aligned = reliable and result['mean']>0
        if spec['guidance_applied']:
            status = 'verified' if aligned else 'conflict' if reliable else 'unverified'
            if guidance_updates == 0: status = 'not_exposed'
            elif guidance_updates is None: status = 'exposure_unknown'
        else:status = 'unguided_verified' if reliable else 'unguided_weak'
        direction = 'positive' if result['mean']>=0 else 'inverse'
        name = readable_name(dataset,chosen,direction,status,spec['family'],spec['replicate'])
        sample = observations[:,k,DESCRIPTORS.index(chosen)]
        quantiles = np.quantile(sample,[.25,.5,.75]).tolist()
        rows.append({'concept_index':k,'concept_name':name,'training_group':spec['family']+('-guided' if spec['guidance_applied'] else ' (guidance absent)'),
                     'descriptor':chosen,'direction':direction,'weak_grounding':status not in ['verified','unguided_verified'],
                     'association_reliable':reliable,
                     'grounding_status':status,'guidance_applied':spec['guidance_applied'] and guidance_updates is not None and guidance_updates>0,
                     'guidance_configured':spec['guidance_applied'],'selected_model_guidance_updates':guidance_updates,'guidance_mode':spec['mode'],
                     'intended_descriptor':spec['target_descriptor'],'strongest_observed_descriptor':best,
                     'grounding_strength':abs(result['mean']),**result,'validation_response_quartiles':quantiles,
                     'scope':'Image-level descriptor evidence; lesion/part localization is not established.',
                     'all_descriptor_statistics':summary})
    # Keep semantic stems, but disambiguate actual duplicate learned patterns without C1/C2 labels.
    names = [r['concept_name'] for r in rows]
    for row in rows:
        if names.count(row['concept_name'])>1:
            preceding = sum(n==row['concept_name'] for n in names[:row['concept_index']])
            row['concept_name'] += f' (learned variant {preceding+1})'
    return rows


@torch.no_grad()
def ground_concepts(model, root, manifest, run, max_batches=None):
    from .semantics import descriptor_bank
    device = next(model.parameters()).device
    model.eval(); values=[]; observations=[]
    dataset = ImageDataset(root,manifest,'validation',model.config)
    for i,(images,_,_) in enumerate(loader(dataset)):
        if max_batches is not None and i>=max_batches:break
        images=images.to(device);output=model(images);bank=descriptor_bank(images)
        values.append(spatial_correlations(output['maps'],bank))
        observations.append(weighted_descriptor_values(output['maps'],bank))
    if not values:raise RuntimeError('Grounding executed zero examples.')
    values=np.concatenate(values);observations=np.concatenate(observations)
    rows=summarize_grounding(values,observations,model.config,manifest['dataset'],getattr(model,'guidance_updates',None))
    atomic_json(Path(run)/'grounding.json',{'source_split':'validation','validation_n':len(values),
                'selection_note':'Guided concepts are checked against their intended target, never renamed to hide failed guidance. Unguided naming searches descriptors with BH-FDR correction. Minimum n=50, |mean r|>=.10, 95% image-bootstrap interval excluding zero and q<.05. Test never renames concepts.',
                'concepts':rows})
    atomic_json(Path(run)/'guidance_audit.json',{'validation_n':len(values),'selected_model_guidance_updates':getattr(model,'guidance_updates',None),
                'configured_guided_concepts':sum(r['guidance_configured'] for r in rows),
                'verified_guided_associations':sum(r['grounding_status']=='verified' for r in rows),
                'conflicting_guided_associations':sum(r['grounding_status']=='conflict' for r in rows),
                'unknown_or_weak':sum(r['weak_grounding'] for r in rows),
                'causal_guidance_benefit':'Not established by association alone; use matched unguided/per-family ablations and multiple seeds.'})
    pd.DataFrame([{k:v for k,v in row.items() if k!='all_descriptor_statistics'} for row in rows]).to_csv(Path(run)/'grounding.csv',index=False)
    return rows


def evidence_packet(model, concepts, logits, classes, grounding, observations=None):
    W, bias = model.classifier.weight.detach().cpu().numpy(), model.classifier.bias.detach().cpu().numpy()
    c, z = np.asarray(concepts, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    probabilities = torch.tensor(z).softmax(0).numpy()
    order = np.argsort(-z)
    pred, runner = int(order[0]), int(order[1])
    contribution = c * W[pred]
    margin = c * (W[pred] - W[runner])
    reconstruction = float(bias[pred] + contribution.sum())
    np.testing.assert_allclose(reconstruction, z[pred], atol=1e-5, rtol=1e-5)
    rows = []
    for k, row in enumerate(grounding):
        rows.append({'concept_index': k, 'name': row['concept_name'], 'weak_grounding': row['weak_grounding'], 'descriptor': row['descriptor'], 'grounding_correlation': row['mean'], 'activation': float(c[k]), 'score_definition': 'sigmoid of training-standardized pooled response' if model.config.concept_normalization == 'running_standardize' else 'mean spatial response', 'predicted_class_contribution': float(contribution[k]), 'margin_contribution': float(margin[k])})
    for record, ground in zip(rows, grounding):
        record.update(training_group=ground['training_group'], guidance_status=ground.get('grounding_status','legacy'),
                      intended_descriptor=ground.get('intended_descriptor'), guidance_applied=ground.get('guidance_applied',False))
        if observations is not None and 'validation_response_quartiles' in ground:
            value=float(observations[ground['concept_index'],DESCRIPTORS.index(ground['descriptor'])])
            low,median,high=ground['validation_response_quartiles']
            level='undetermined' if high-low<1e-6 else 'high' if value>high else 'low' if value<low else 'typical'
            record['image_descriptor_evidence']={'weighted_response':value,'relative_to_validation':level,
                'scope':'Descriptor measured over model-responsive image locations, not an annotated lesion/part boundary.'}
    return {'prediction': {'class': classes[pred], 'probability': float(probabilities[pred]), 'runner_up': classes[runner], 'runner_up_probability': float(probabilities[runner])}, 'exact_decomposition': {'bias': float(bias[pred]), 'logit': float(z[pred]), 'reconstructed_logit': reconstruction, 'margin_bias': float(bias[pred] - bias[runner]), 'margin': float(z[pred] - z[runner]), 'reconstructed_margin': float(bias[pred] - bias[runner] + margin.sum())}, 'concepts': rows, 'scope': 'Model-derived image-descriptor associations. No clinical, anatomical, species-part or causal image claim is established by correlation. Probability is uncalibrated unless calibration is separately evaluated.'}


def _foreground(root, row, size):
    if 'mask' in row:
        with Image.open(Path(root) / row['mask']) as image:
            trimap = np.asarray(image.resize((size[1], size[0]), resample=Image.Resampling.NEAREST))
        return trimap == 1, trimap != 3  # Border excluded from independent evaluation.
    if 'bbox' in row:
        with Image.open(Path(root) / row['path']) as image:
            width, height = image.size
        x, y, w, h = row['bbox']
        yy, xx = np.mgrid[:size[0], :size[1]]
        xx = (xx + 0.5) * width / size[1]
        yy = (yy + 0.5) * height / size[0]
        return (xx >= x) & (xx <= x + w) & (yy >= y) & (yy <= y + h), np.ones(size, dtype=bool)
    return None, None


def expected_calibration_error(probs, labels):
    confidence, predicted = probs.max(1), probs.argmax(1)
    error = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        select = (confidence > low) & (confidence <= high)
        if select.any():
            error += select.mean() * abs((predicted[select] == labels[select]).mean() - confidence[select].mean())
    return float(error)


@torch.no_grad()
def evaluate(model, root, manifest, run, grounding, max_batches=None):
    model.eval()
    root, run = Path(root), Path(run)
    device = next(model.parameters()).device
    dataset = ImageDataset(root, manifest, 'test', model.config)
    collected, test_correlations, drops, localizations, stability, evidence = [], [], [], [], [], []
    gen = np.random.default_rng(model.config.seed)
    W = model.classifier.weight
    for i, (images, labels, indices) in enumerate(loader(dataset)):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        output = model(images)
        c, logits = output['concepts'], output['logits']
        p = logits.softmax(1)
        order = logits.argsort(1, descending=True)
        pred, runner = order[:, 0], order[:, 1]
        margin_terms = c * (W[pred] - W[runner])
        predicted_terms = c * W[pred]
        selected = margin_terms.argmax(1)
        valid = margin_terms.max(1).values > 0
        edited = c.clone()
        edited[torch.arange(len(c)), selected] = 0
        _, removed = model.predict_from_concepts(edited)
        random_indices = torch.tensor(gen.integers(0, model.config.concepts, size=len(c)), device=device)
        random_c = c.clone()
        random_c[torch.arange(len(c)), random_indices] = 0
        _, random_probs = model.predict_from_concepts(random_c)
        absolute_c = c.clone()
        absolute_c[torch.arange(len(c)), predicted_terms.abs().argmax(1)] = 0
        _, absolute_probs = model.predict_from_concepts(absolute_c)
        change = model((images * 1.03 + 4 / 255).clamp(0, 1))['concepts']
        mae = (change - c).abs().mean(1).cpu().numpy()
        cosine = F.cosine_similarity(c, change).cpu().numpy()
        bank = descriptor_bank(images)
        correlations = spatial_correlations(output['maps'], bank)
        observations = weighted_descriptor_values(output['maps'], bank)
        test_correlations.append(correlations)
        for b, index in enumerate(indices.tolist()):
            row = dataset.rows[index]
            collected.append({'path': row['path'], 'true': int(labels[b]), 'predicted': int(pred[b]), 'confidence': float(p[b, pred[b]])})
            drops.append({'path': row['path'], 'has_positive_margin_support': bool(valid[b]), 'concept_index': int(selected[b]), 'positive_margin_probability_drop': float(p[b, pred[b]] - removed[b, pred[b]]) if valid[b] else None, 'random_probability_drop': float(p[b, pred[b]] - random_probs[b, pred[b]]), 'legacy_absolute_probability_drop': float(p[b, pred[b]] - absolute_probs[b, pred[b]])})
            stability.append({'path': row['path'], 'mae': float(mae[b]), 'cosine': float(cosine[b])})
            foreground, allowed = _foreground(root, row, output['maps'].shape[-2:])
            if foreground is not None and np.any(foreground) and np.any(allowed):
                maps = output['maps'][b].cpu().numpy()
                baseline = foreground[allowed].mean()
                for k, activation in enumerate(maps):
                    mass = activation[foreground & allowed].sum() / max(activation[allowed].sum(), 1e-8)
                    flat = np.where(allowed, activation, -np.inf)
                    yx = np.unravel_index(flat.argmax(), flat.shape)
                    localizations.append({'path': row['path'], 'concept_index': k, 'foreground_mass': float(mass), 'foreground_area_fraction': float(baseline), 'mass_to_area_ratio': float(mass / max(baseline, 1e-8)), 'peak_in_foreground': bool(foreground[yx]), 'annotation': 'pet trimap' if 'mask' in row else 'bird bounding box'})
            if len(evidence) < 8:
                packet = evidence_packet(model, c[b].cpu().numpy(), logits[b].cpu().numpy(), manifest['classes'], grounding, observations[b])
                packet['image_path'] = row['path']
                packet['true_class'] = manifest['classes'][int(labels[b])]
                packet['test_index'] = index
                evidence.append(packet)
        # Store batch arrays for complete classifier metrics and exact concept exports.
        if i == 0:
            arrays = {'probs': [], 'concepts': [], 'labels': [], 'logits': []}
        for name, value in [('probs', p), ('concepts', c), ('labels', labels), ('logits', logits)]:
            arrays[name].append(value.cpu().numpy())
    if not collected:
        raise RuntimeError('Test evaluation executed zero examples.')
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    probs, labels = arrays['probs'], arrays['labels']
    predicted = probs.argmax(1)
    drops_frame = pd.DataFrame(drops)
    stability_frame = pd.DataFrame(stability)
    score = {'n_test_evaluated': len(labels), 'n_test_available': len(dataset), 'complete_test': len(labels) == len(dataset), 'accuracy': float(accuracy_score(labels, predicted)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)), 'macro_f1': float(f1_score(labels, predicted, labels=np.arange(len(manifest['classes'])), average='macro', zero_division=0)), 'weighted_f1': float(f1_score(labels, predicted, average='weighted', zero_division=0)), 'top5_accuracy': float(np.mean([y in row for y, row in zip(labels, probs.argsort(1)[:, -min(5, len(manifest['classes'])):])])) if len(manifest['classes']) >= 5 else None, 'ece_15_bins': expected_calibration_error(probs, labels), 'multiclass_brier_sum': float(np.square(probs - np.eye(len(manifest['classes']))[labels]).sum(1).mean()), 'n_positive_margin_cases': int(drops_frame.has_positive_margin_support.sum()), 'positive_margin_probability_drop_mean': float(drops_frame.positive_margin_probability_drop.mean()) if drops_frame.positive_margin_probability_drop.notna().any() else None, 'random_probability_drop_mean': float(drops_frame.random_probability_drop.mean()), 'legacy_absolute_probability_drop_mean': float(drops_frame.legacy_absolute_probability_drop.mean()), 'stability_mae': float(stability_frame.mae.mean()), 'stability_cosine': float(stability_frame.cosine.mean()), 'warning': 'Latent concept ablation measures model dependence, not a causal intervention on the image or clinical outcome. Smoke runs are not paper results.'}
    centered = arrays['concepts'] - arrays['concepts'].mean(0, keepdims=True)
    comparable=[r['all_descriptor_statistics'][r['intended_descriptor']]['mean'] for r in grounding if r.get('intended_descriptor')]
    score['intended_target_validation_r_mean']=float(np.mean(comparable)) if comparable else None
    enough=all(r['valid_n']>=50 for r in grounding)
    score['verified_guidance_fraction']=float(np.mean([r.get('grounding_status')=='verified' for r in grounding if r.get('guidance_configured')])) if enough and any(r.get('guidance_configured') for r in grounding) else None
    score['reliable_association_fraction']=float(np.mean([r.get('association_reliable',False) for r in grounding])) if enough else None
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    distribution = energy / max(float(energy.sum()), 1e-12)
    score['concept_effective_rank'] = float(np.exp(-np.sum(distribution * np.log(distribution + 1e-12)))) if energy.sum() > 1e-12 else 0.
    if len(centered) >= 3:
        correlation = np.corrcoef(arrays['concepts'], rowvar=False)
        off = correlation[~np.eye(model.config.concepts, dtype=bool)]
        score['mean_absolute_interconcept_correlation'] = float(np.nanmean(np.abs(off))) if np.isfinite(off).any() else None
    if score['n_positive_margin_cases']:
        valid_drops = drops_frame[drops_frame.has_positive_margin_support]
        score['matched_top_minus_random_drop'] = float((valid_drops.positive_margin_probability_drop - valid_drops.random_probability_drop).mean())
    test_values = np.concatenate(test_correlations)
    test_alignment = []
    for k, row in enumerate(grounding):
        test_alignment.append({'concept_index': k, 'frozen_validation_descriptor': row['descriptor'], **summarize_correlation(test_values[:, k, DESCRIPTORS.index(row['descriptor'])], model.config.seed + k)})
    if localizations:
        frame = pd.DataFrame(localizations)
        score['mean_foreground_mass_to_area_ratio'] = float(frame.mass_to_area_ratio.mean())
        score['peak_in_foreground_rate'] = float(frame.peak_in_foreground.mean())
        frame.to_csv(run / 'independent_localization.csv', index=False)
    atomic_json(run / 'metrics.json', score)
    atomic_json(run / 'test_frozen_alignment.json', test_alignment)
    atomic_json(run / 'example_evidence.json', evidence)
    atomic_json(run / 'classification_report.json', classification_report(labels, predicted, labels=np.arange(len(manifest['classes'])), target_names=manifest['classes'], output_dict=True, zero_division=0))
    np.save(run / 'confusion_matrix.npy', confusion_matrix(labels, predicted, labels=np.arange(len(manifest['classes']))))
    np.savez_compressed(run / 'test_outputs.npz', **arrays)
    pd.DataFrame(collected).to_csv(run / 'predictions.csv', index=False)
    drops_frame.to_csv(run / 'faithfulness.csv', index=False)
    stability_frame.to_csv(run / 'stability.csv', index=False)
    return score
