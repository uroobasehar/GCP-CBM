"""Sequential, recoverable dataset runs with one model and complete evidence per dataset."""
from dataclasses import replace
from pathlib import Path
import gc
import json
import traceback
import pandas as pd
import torch
from .core import Config
from .data import prepare_data, build_manifest, leakage_audit, atomic_json, ImageDataset
from .train import fit, _sync
from .evaluate import ground_concepts, evaluate
from .visualize import visualize_run
from .reports import generate_reports
from .journal import concept_cards, reliability_and_redundancy, deletion_curves, profile_inference
from .environment import runtime_manifest


def run_dataset_suite(root, datasets, base_config, *, run_tag='calibrated_v3', device=None,
                      prepare_callback=None, sync_callback=None, llm_backend='template',
                      llm_device='cpu', continue_on_error=True):
    """Resume each fit; reuse finished evidence only for its exact configuration fingerprint.

    prepare_callback(dataset, run_name) handles platform-specific data/Drive restoration.
    Test scores are exported after training; they never change the training configuration.
    """
    root = Path(root)
    datasets = list(datasets)
    if not datasets or len(datasets) != len(set(datasets)) or any(n not in ['cub', 'pets', 'brain_mri'] for n in datasets):
        raise ValueError('Choose a nonempty, unique dataset list from cub, pets, brain_mri.')
    if not run_tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_tag):
        raise ValueError('Use letters, numbers, underscores or hyphens in run_tag.')
    smoke = base_config.max_train_batches is not None or base_config.max_validation_batches is not None
    suffix = '_smoke' if smoke else ''
    summary = root / 'results' / f'dataset_suite_{run_tag}_K{base_config.concepts}_seed{base_config.seed}{suffix}'
    summary.mkdir(parents=True, exist_ok=True)
    protocol = {'base_config': base_config.to_dict(), 'run_tag': run_tag, 'llm_backend': llm_backend}
    protocol_file = summary / 'suite_configuration.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('Suite configuration differs. Choose a new run_tag to preserve the previous summary.')
    atomic_json(protocol_file, protocol)
    # Each invocation revisits the requested queue. Checkpoint identities guard all reuse.
    rows = []
    def persist(active=None):
        atomic_json(summary / 'queue_status.json', {'datasets': datasets, 'active_dataset': active,
                    'complete': len(rows) == len(datasets) and all(r['status'] == 'complete' for r in rows),
                    'runs': rows, 'software_smoke_only': smoke})
        pd.DataFrame(rows).to_csv(summary / 'dataset_summary.csv', index=False)
        _sync(sync_callback, summary)
    persist()
    for index, dataset in enumerate(datasets):
        cfg = replace(base_config, dataset=dataset)
        name = f'{dataset}_K{cfg.concepts}_seed{cfg.seed}_{run_tag}{suffix}'
        model = None
        print(f'\nDataset {index+1}/{len(datasets)}: {dataset} -> {name}', flush=True)
        persist(active=dataset)
        try:
            if prepare_callback is not None:
                prepare_callback(dataset, name)
            else:
                prepare_data(root, dataset, online=False)
            manifest = build_manifest(root, dataset, cfg.seed, cfg.validation_fraction)
            audit = leakage_audit(root, manifest)
            if audit['cross_split_exact_duplicates']:
                raise RuntimeError('Cross-split duplicates found; resolve leakage before estimating results.')
            model, run, status = fit(root, manifest, cfg, device=device, sync_callback=sync_callback, run_name=name)
            if not status['complete']:
                raise RuntimeError('Training did not finish; resume this queue with the same configuration.')
            identity = json.loads((run / 'configuration.json').read_text())['fingerprint']
            done = run / 'pipeline_status.json'
            expected = {'complete': True, 'fingerprint': identity, 'llm_backend': llm_backend, 'protocol': 'calibrated_v3'}
            required = ['metrics.json', 'grounding.json', 'concept_cards.json', 'deletion_curves.csv',
                        'inference_profile.json', 'reports/case_000.json', 'figures/confusion_matrix.png']
            cached = done.is_file() and json.loads(done.read_text()) == expected and all((run / p).is_file() for p in required)
            if cached:
                metrics = json.loads((run / 'metrics.json').read_text())
                print('Reusing completed evaluation and explanation outputs:', run, flush=True)
            else:
                limit = cfg.max_validation_batches if smoke else None
                atomic_json(run / 'dataset_manifest.json', manifest)
                atomic_json(run / 'environment.json', runtime_manifest())
                print('Grounding concepts on validation:', dataset, flush=True)
                grounding = ground_concepts(model, root, manifest, run, max_batches=limit)
                print('Evaluating frozen model on test:', dataset, flush=True)
                metrics = evaluate(model, root, manifest, run, grounding, max_batches=limit)
                print('Saving figures, evidence reports and journal audits:', dataset, flush=True)
                visualize_run(model, root, manifest, run, grounding, examples=1 if smoke else 3, prototype_batches=limit)
                generate_reports(run, root / 'assets', backend=llm_backend, max_cases=1 if smoke else 3, llm_device=llm_device)
                concept_cards(manifest, run, grounding)
                reliability_and_redundancy(run)
                deletion_curves(model, run, max_k=4)
                image = ImageDataset(root, manifest, 'test', cfg)[0][0]
                profile_inference(model, image, run, repeats=3 if smoke else 20)
                atomic_json(done, expected)
                _sync(sync_callback, run)
            row = {'dataset': dataset, 'run': name, 'status': 'complete', 'classes': len(manifest['classes']),
                   'best_validation_accuracy': status['progress']['best_accuracy'],
                   'software_smoke_only': smoke, **metrics}
            rows.append(row)
        except Exception as exc:
            rows.append({'dataset': dataset, 'run': name, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            atomic_json(summary / f'{dataset}_error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            print(f'{dataset} failed; checkpoint files retained. See {summary / (dataset+"_error.json")}', flush=True)
            persist()
            if not continue_on_error:
                raise
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        persist()
    frame = pd.DataFrame(rows)
    if all(row['status'] == 'complete' for row in rows):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(frame.dataset, frame.accuracy * 100)
        ax.set(ylabel='Held-out accuracy (%)', title='Software smoke checks only' if smoke else 'Independent dataset runs (different tasks)')
        fig.tight_layout(); fig.savefig(summary / 'dataset_accuracy.png', dpi=160); plt.close(fig)
        _sync(sync_callback, summary)
    return frame
