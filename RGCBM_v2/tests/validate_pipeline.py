"""Real-data CPU checks, including deterministic interrupted/resumed training."""
from pathlib import Path
import sys
import json
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgcbm.core import Config, priors
from rgcbm.data import build_manifest, ImageDataset, leakage_audit, atomic_json
from rgcbm.train import fit, load_model
from rgcbm.evaluate import ground_concepts, evaluate
from rgcbm.visualize import visualize_run
from rgcbm.reports import generate_reports

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
records = []
for name in ['cub', 'pets', 'brain_mri']:
    manifest = build_manifest(ROOT, name)
    cfg = Config(dataset=name, batch_size=2, head_epochs=1, finetune_epochs=1, max_train_batches=1, max_validation_batches=1, checkpoint_every_batches=1)
    dataset = ImageDataset(ROOT, manifest, 'train', cfg)
    images = torch.stack([dataset[i][0] for i in range(2)])
    descriptor = priors(images)
    assert all(v.shape == (2, 1, 240, 240) and torch.isfinite(v).all() for v in descriptor.values())
    model, run, state = fit(ROOT, manifest, cfg, device='cpu', run_name=f'validation_final_{name}')
    assert state['complete']
    head = torch.load(run / 'head_best.pt', weights_only=True)['model']
    fine = torch.load(run / 'finetune_best.pt', weights_only=True)['model']
    assert not torch.equal(head['backbone.8.0.weight'], fine['backbone.8.0.weight']), 'Fine-tuning did not update the backbone.'
    grounded = ground_concepts(model, ROOT, manifest, run, max_batches=1)
    assert len(grounded) == 16 and all(r['weak_grounding'] for r in grounded)
    scores = evaluate(model, ROOT, manifest, run, grounded, max_batches=1)
    assert scores['n_test_evaluated'] == 2 and not scores['complete_test']
    assert scores['stability_mae'] >= 0
    with torch.no_grad():
        restored, classes = load_model(run, ROOT / 'assets', device='cpu')
        np.testing.assert_allclose(model(images)['logits'].numpy(), restored(images)['logits'].numpy(), atol=1e-6)
    generate_reports(run, ROOT / 'assets', backend='template', max_cases=1)
    visualize_run(model, ROOT, manifest, run, grounded, examples=1, prototype_batches=1)
    records.append({'dataset': name, 'official_manifest_counts': manifest['counts'], 'training_updates': 2, 'test_examples': 2, 'checks': ['head training', 'fine-tuning', 'radiomics priors', 'validation grounding', 'test scoring', 'independent foreground annotations' if name != 'brain_mri' else 'MRI evidence export', 'checkpoint reload', 'reports', 'figures']})

manifest = build_manifest(ROOT, 'pets')
cfg = Config(dataset='pets', batch_size=2, head_epochs=1, finetune_epochs=0, max_train_batches=2, max_validation_batches=1, checkpoint_every_batches=1)
reference, _, state = fit(ROOT, manifest, cfg, device='cpu', run_name='validation_final_uninterrupted')
partial, _, state = fit(ROOT, manifest, cfg, device='cpu', run_name='validation_final_resumed', interrupt_after_batches=1)
assert not state['complete']
resumed, _, state = fit(ROOT, manifest, cfg, device='cpu', run_name='validation_final_resumed')
assert state['complete']
for key in reference.state_dict():
    torch.testing.assert_close(reference.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0, msg=lambda message: f'{key}: {message}')
try:
    incompatible = Config(dataset='pets', batch_size=2, concepts=32, head_epochs=1, finetune_epochs=0, max_train_batches=2, max_validation_batches=1)
    fit(ROOT, manifest, incompatible, device='cpu', run_name='validation_final_resumed')
    raise AssertionError('Incompatible checkpoint was silently accepted.')
except ValueError as error:
    assert 'identity' in str(error)
result = {'status': 'passed', 'hardware': 'CPU only; Windows/CUDA runtime not available here', 'datasets': records, 'deterministic_resume': 'bitwise identical CPU model weights after one interrupted batch', 'incompatible_resume': 'correctly rejected', 'scope': 'Functional validation on actual images; tiny runs do not estimate research accuracy.'}
atomic_json(ROOT / 'docs' / 'validation.json', result)
print(json.dumps(result, indent=2))
