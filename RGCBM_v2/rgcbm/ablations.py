from pathlib import Path
import gc
import pandas as pd
import torch
from .core import Config
from .data import build_manifest, atomic_json
from .train import fit
from .evaluate import ground_concepts, evaluate

VARIANTS = ['full', 'unguided', 'no_edge', 'no_texture', 'no_region', 'no_diversity', 'no_decorrelation', 'no_stability', 'no_sparsity', 'no_finetune', 'no_standardization', 'legacy_group_guidance', 'K16', 'K32', 'K64']


def variant_config(base, variant, seed=None):
    if variant not in VARIANTS: raise ValueError(variant)
    cfg = Config(**base.to_dict())
    if seed is not None: cfg.seed = seed
    if variant == 'unguided': cfg.edge = cfg.texture = cfg.morphology = 0.
    for name, field in [('no_edge', 'edge'), ('no_texture', 'texture'), ('no_region', 'morphology'), ('no_diversity', 'diversity'), ('no_decorrelation', 'disentangle'), ('no_stability', 'stability')]:
        if variant == name: setattr(cfg, field, 0.)
    if variant == 'no_sparsity': cfg.activation_l1 = cfg.classifier_l1 = 0.
    if variant == 'no_finetune': cfg.finetune_epochs = 0
    if variant == 'no_standardization': cfg.concept_normalization = 'none'
    if variant == 'legacy_group_guidance': cfg.guidance_mode = 'group_mean'
    if variant in ['K16', 'K32', 'K64']: cfg.concepts = int(variant[1:])
    return cfg


def run_ablation_suite(root, base_config, variants=None, seeds=(42,), device=None, sync_callback=None):
    """Same data/preprocessing/budget per matched variant; separate reproducible run identities."""
    root = Path(root)
    variants = VARIANTS if variants is None else list(variants)
    records = []
    profile = 'guided_v4' if base_config.guidance_mode == 'individual_descriptors_v4' else 'calibrated_v3' if base_config.concept_normalization == 'running_standardize' else 'legacy'
    summary = root / 'results' / f'{base_config.dataset}_ablation_summary_{profile}'
    summary.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        manifest = build_manifest(root, base_config.dataset, seed, base_config.validation_fraction)
        for variant in variants:
            cfg = variant_config(base_config, variant, seed)
            smoke = cfg.max_train_batches is not None or cfg.max_validation_batches is not None
            name = f'{cfg.dataset}_ablation_{variant}_K{cfg.concepts}_seed{seed}_{profile}' + ('_smoke' if smoke else '')
            print(f'Running {name}', flush=True)
            model, run, status = fit(root, manifest, cfg, device=device, sync_callback=sync_callback, run_name=name)
            assert status['complete']
            limit = cfg.max_validation_batches if smoke else None
            grounding = ground_concepts(model, root, manifest, run, max_batches=limit)
            metrics = evaluate(model, root, manifest, run, grounding, max_batches=limit)
            row = {'variant': variant, 'seed': seed, 'concepts': cfg.concepts, 'software_smoke_only': smoke, 'run': name, **metrics}
            records.append(row)
            frame = pd.DataFrame(records)
            frame.to_csv(summary / 'ablation_runs.csv', index=False)
            numeric = ['accuracy', 'macro_f1', 'balanced_accuracy', 'ece_15_bins', 'positive_margin_probability_drop_mean', 'random_probability_drop_mean', 'stability_mae', 'stability_cosine', 'intended_target_validation_r_mean','verified_guidance_fraction','reliable_association_fraction']
            frame.groupby('variant')[numeric].agg(['mean', 'std', 'count']).to_csv(summary / 'ablation_mean_std.csv')
            atomic_json(summary / 'protocol.json', {'dataset': cfg.dataset, 'base_config': base_config.to_dict(), 'variants': variants, 'seeds': list(seeds), 'warning': 'Use at least three complete seeds for a paper. Test measurements compare preregistered variants; do not tune variants on these test scores.'})
            if sync_callback:
                sync_callback(run)
                sync_callback(summary)
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pd.DataFrame(records)
