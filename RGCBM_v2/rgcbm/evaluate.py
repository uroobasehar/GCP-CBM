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

DESCRIPTORS = ['edge', 'texture', 'morphology', 'intensity']


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


@torch.no_grad()
def ground_concepts(model, root, manifest, run, max_batches=None):
    """Select neutral descriptor names on validation only; test cannot rename concepts."""
    device = next(model.parameters()).device
    model.eval()
    dataset = ImageDataset(root, manifest, 'validation', model.config)
    all_values = []
    for i, (images, _, _) in enumerate(loader(dataset)):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        output = model(images)
        all_values.append(spatial_correlations(output['maps'], priors(images)))
    if not all_values:
        raise RuntimeError('Grounding executed zero examples.')
    values = np.concatenate(all_values)
    rows = []
    groups = ['boundary-guided', 'texture-guided', 'region-prior-guided', 'unguided']
    for k in range(model.config.concepts):
        summary = {name: summarize_correlation(values[:, k, j], model.config.seed + k * 4 + j) for j, name in enumerate(DESCRIPTORS)}
        best = max(summary, key=lambda name: abs(summary[name]['mean']))
        stats = summary[best]
        reliable = stats['valid_n'] >= 10 and abs(stats['mean']) >= 0.08 and stats['ci_low'] * stats['ci_high'] > 0
        direction = 'positive' if stats['mean'] >= 0 else 'inverse'
        name = f'C{k + 1:02d}: {direction} {best}-associated pattern' if reliable else f'C{k + 1:02d}: weakly grounded visual pattern'
        group_id = k // (model.config.concepts // 4)
        group_name = groups[group_id]
        if group_id < 3 and getattr(model.config, ['edge', 'texture', 'morphology'][group_id]) == 0:
            group_name += ' (alignment disabled)'
        rows.append({'concept_index': k, 'concept_name': name, 'training_group': group_name, 'descriptor': best, 'direction': direction, 'weak_grounding': not reliable, 'grounding_strength': abs(stats['mean']), **stats, 'all_descriptor_statistics': summary})
    atomic_json(Path(run) / 'grounding.json', {'source_split': 'validation', 'validation_n': len(values), 'selection_note': 'Names describe measured image associations; no part, clinical, shape, or color diagnosis is inferred.', 'concepts': rows})
    pd.DataFrame([{k: v for k, v in row.items() if k != 'all_descriptor_statistics'} for row in rows]).to_csv(Path(run) / 'grounding.csv', index=False)
    return rows


def evidence_packet(model, concepts, logits, classes, grounding):
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
        rows.append({'concept_index': k, 'name': row['concept_name'], 'weak_grounding': row['weak_grounding'], 'descriptor': row['descriptor'], 'grounding_correlation': row['mean'], 'activation': float(c[k]), 'predicted_class_contribution': float(contribution[k]), 'margin_contribution': float(margin[k])})
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
        correlations = spatial_correlations(output['maps'], priors(images))
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
                packet = evidence_packet(model, c[b].cpu().numpy(), logits[b].cpu().numpy(), manifest['classes'], grounding)
                packet['image_path'] = row['path']
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
    score = {'n_test_evaluated': len(labels), 'n_test_available': len(dataset), 'complete_test': len(labels) == len(dataset), 'accuracy': float(accuracy_score(labels, predicted)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)), 'macro_f1': float(f1_score(labels, predicted, labels=np.arange(len(manifest['classes'])), average='macro', zero_division=0)), 'weighted_f1': float(f1_score(labels, predicted, average='weighted', zero_division=0)), 'top5_accuracy': float(np.mean([y in row for y, row in zip(labels, probs.argsort(1)[:, -min(5, len(manifest['classes'])):])])), 'ece_15_bins': expected_calibration_error(probs, labels), 'multiclass_brier_sum': float(np.square(probs - np.eye(len(manifest['classes']))[labels]).sum(1).mean()), 'n_positive_margin_cases': int(drops_frame.has_positive_margin_support.sum()), 'positive_margin_probability_drop_mean': float(drops_frame.positive_margin_probability_drop.mean()) if drops_frame.positive_margin_probability_drop.notna().any() else None, 'random_probability_drop_mean': float(drops_frame.random_probability_drop.mean()), 'legacy_absolute_probability_drop_mean': float(drops_frame.legacy_absolute_probability_drop.mean()), 'stability_mae': float(stability_frame.mae.mean()), 'stability_cosine': float(stability_frame.cosine.mean()), 'warning': 'Latent concept ablation measures model dependence, not a causal intervention on the image or clinical outcome. Smoke runs are not paper results.'}
    centered = arrays['concepts'] - arrays['concepts'].mean(0, keepdims=True)
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
