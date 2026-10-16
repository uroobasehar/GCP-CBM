"""Matched baselines and ablations; full runs are explicit, never fabricated results."""
from pathlib import Path
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from rgcbm.core import training_config
from rgcbm.ablations import variant_config
from rgcbm.data import build_manifest, leakage_audit, ImageDataset, loader, atomic_json
from rgcbm.train import fit
from rgcbm.evaluate import ground_concepts, evaluate, expected_calibration_error
from rgcbm.visualize import visualize_run

parser = argparse.ArgumentParser()
parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
parser.add_argument('--dataset', choices=['cub', 'pets', 'brain_mri'], default='cub')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--variant', choices=['full', 'backbone', 'unguided', 'no_edge', 'no_texture', 'no_region', 'no_diversity', 'no_decorrelation', 'no_stability', 'no_sparsity', 'no_finetune', 'no_standardization', 'legacy_group_guidance', 'K16', 'K32', 'K64'], default='full')
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--batch-size', type=int, default=24)
parser.add_argument('--profile', choices=['guided_v4', 'calibrated_v3', 'legacy'], default='guided_v4')
args = parser.parse_args()
root = Path(args.root)
torch.set_num_threads(4)
cfg = training_config(args.profile, dataset=args.dataset, seed=args.seed, batch_size=args.batch_size)
if args.variant != 'backbone': cfg = variant_config(cfg, args.variant)
if args.smoke:
    cfg.head_epochs = 1
    cfg.finetune_epochs = 0 if args.variant == 'no_finetune' else 1
    cfg.max_train_batches = cfg.max_validation_batches = 1
    cfg.checkpoint_every_batches = 1
manifest = build_manifest(root, args.dataset, args.seed, cfg.validation_fraction)
assert not leakage_audit(root, manifest)['cross_split_exact_duplicates'], 'Resolve leakage before reporting results.'
name = f'{args.dataset}_{args.variant}_K{cfg.concepts}_seed{cfg.seed}_{args.profile}' + ('_smoke' if args.smoke else '')
model, run, _ = fit(root, manifest, cfg, run_name=name, model_type='backbone' if args.variant == 'backbone' else 'rgcbm')
if args.variant == 'backbone':
    dataset = ImageDataset(root, manifest, 'test', cfg)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for index, (images, labels, _) in enumerate(loader(dataset)):
            if args.smoke and index >= 1: break
            ps.append(model(images.to(next(model.parameters()).device))['probs'].cpu().numpy()); ys.append(labels.numpy())
    labels, probs = np.concatenate(ys), np.concatenate(ps)
    predicted = probs.argmax(1)
    score = {'accuracy': float(accuracy_score(labels, predicted)), 'macro_f1': float(f1_score(labels, predicted, labels=np.arange(len(manifest['classes'])), average='macro', zero_division=0)), 'weighted_f1': float(f1_score(labels, predicted, average='weighted', zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)), 'ece_15_bins': expected_calibration_error(probs, labels), 'multiclass_brier_sum': float(np.square(probs - np.eye(len(manifest['classes']))[labels]).sum(1).mean()), 'n_test_evaluated': len(labels), 'complete_test': len(labels) == len(dataset), 'note': 'Matched backbone/pretraining/data/training schedule; CE-only baseline has no concept explanations.'}
    np.save(run / 'confusion_matrix.npy', confusion_matrix(labels, predicted, labels=np.arange(len(manifest['classes']))))
    atomic_json(run / 'classification_report.json', classification_report(labels, predicted, labels=np.arange(len(manifest['classes'])), target_names=manifest['classes'], output_dict=True, zero_division=0))
    atomic_json(run / 'metrics.json', score)
else:
    grounding = ground_concepts(model, root, manifest, run, max_batches=1 if args.smoke else None)
    score = evaluate(model, root, manifest, run, grounding, max_batches=1 if args.smoke else None)
    visualize_run(model, root, manifest, run, grounding, examples=1 if args.smoke else 3, prototype_batches=1 if args.smoke else None)
print(score)
