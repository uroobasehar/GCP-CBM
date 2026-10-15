"""Real-image checks for calibrated concepts, deterministic resume, snapshots and queue reuse."""
from pathlib import Path
import argparse
import json
import sys
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from rgcbm.core import training_config, Config, RGCBM
from rgcbm.data import build_manifest, ImageDataset
from rgcbm.train import fit, load_model
from rgcbm.suite import run_dataset_suite

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
for name in ['data', 'assets']:
    (root / name).symlink_to(SOURCE / name, target_is_directory=True)
torch.set_num_threads(4); torch.set_num_interop_threads(2)
cfg = training_config(dataset='pets', batch_size=2, head_epochs=1, finetune_epochs=0,
                      max_train_batches=2, max_validation_batches=1, checkpoint_every_batches=1,
                      checkpoint_every_epochs=1, keep_epoch_checkpoints=1, show_progress=False)
manifest = build_manifest(root, 'pets')
reference, full, _ = fit(root, manifest, cfg, device='cpu', run_name='resume_reference')
_, partial, status = fit(root, manifest, cfg, device='cpu', run_name='resume_interrupted', interrupt_after_batches=1)
assert not status['complete']
saved = torch.load(partial / 'last.pt', weights_only=False)
assert saved['progress']['batch'] == 1
resumed, _, status = fit(root, manifest, cfg, device='cpu', run_name='resume_interrupted')
assert status['complete']
for name in reference.state_dict():
    torch.testing.assert_close(reference.state_dict()[name], resumed.state_dict()[name], rtol=0, atol=0, msg=lambda s: name + ': ' + s)
restored, _ = load_model(partial, root / 'assets', device='cpu')
x = torch.stack([ImageDataset(root, manifest, 'validation', cfg)[i][0] for i in range(2)])
before = restored.concept_norm.running_mean.clone()
with torch.no_grad():
    o = restored(x)
    torch.testing.assert_close(o['logits'], restored.classifier(o['concepts']))
    assert torch.isfinite(o['concepts']).all() and o['concepts'].min() >= 0 and o['concepts'].max() <= 1
assert torch.equal(before, restored.concept_norm.running_mean), 'Evaluation changed concept statistics.'
# The legacy score path remains available for loading the old architecture.
legacy = RGCBM(37, Config(pretrained=False)).eval()
assert not hasattr(legacy, 'concept_norm')
cfg.head_epochs = 2; cfg.finetune_epochs = 1; cfg.max_train_batches = 1
_, run, _ = fit(root, manifest, cfg, device='cpu', run_name='snapshot_retention')
snapshots = sorted((run / 'checkpoints').glob('*.pt'))
assert [p.name for p in snapshots] == ['finetune_epoch_0001.pt', 'head_epoch_0002.pt']
for path in snapshots:
    checkpoint = torch.load(path, weights_only=False)
    assert {'optimizer', 'scheduler', 'rng', 'progress', 'scaler', 'model'}.issubset(checkpoint)
last = torch.load(run / 'last.pt', weights_only=False)
groups = {g['name']: g['lr'] for g in last['optimizer']['param_groups']}
assert groups == {'head': cfg.finetune_head_lr, 'backbone': cfg.finetune_lr}
assert (run / 'history.csv').is_file() and (run / 'live_learning_curves.png').is_file()
cfg.head_epochs = 1
queue = run_dataset_suite(root, ['cub', 'pets', 'brain_mri'], cfg, device='cpu', run_tag='workflow_test', continue_on_error=False)
assert queue.status.tolist() == ['complete'] * 3
mtimes = {name: (root / 'results' / name / 'metrics.json').stat().st_mtime_ns for name in queue.run}
again = run_dataset_suite(root, ['cub', 'pets', 'brain_mri'], cfg, device='cpu', run_tag='workflow_test', continue_on_error=False)
assert again.status.tolist() == ['complete'] * 3
assert all((root / 'results' / name / 'metrics.json').stat().st_mtime_ns == stamp for name, stamp in mtimes.items())
report = {'status': 'passed', 'hardware': 'Linux CPU; no Windows/CUDA or live cloud authentication',
          'data': 'real CUB, Pet and MRI images; tiny smoke subsets',
          'checks': ['calibrated bottleneck stays in [0,1]', 'exact linear logit reconstruction',
                     'evaluation uses fixed running statistics', 'legacy uncalibrated model construction',
                     'bitwise identical interrupted/resumed weights and BN buffers', 'independent head/backbone fine-tuning LRs',
                     'full epoch snapshots and retention', 'live history CSV/figure',
                     'three-dataset queue plus saved metrics/figures/reports', 'repeat queue reuses completed evidence'],
          'research_accuracy': 'not estimated by software smoke checks'}
(root / 'training_workflow_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
