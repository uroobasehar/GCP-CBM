from pathlib import Path
import hashlib
import json
import os
import random
import warnings
import shutil
from datetime import datetime, timezone
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from .core import Config, RGCBM, BackboneClassifier, loss_terms
from .data import ImageDataset, loader, atomic_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Exact GPU bitwise replay can still depend on CUDA/driver/algorithm versions.


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


@torch.no_grad()
def validation(model, batches, device, limit=None, show_progress=False):
    model.eval()
    correct, top5, seen, loss = 0, 0, 0, 0.
    predictions, concept_sum, concept_sq = set(), None, None
    for i, (x, y, _) in enumerate(tqdm(batches, desc='Validation', leave=False, disable=not show_progress)):
        if limit is not None and i >= limit:
            break
        x, y = x.to(device), y.to(device)
        output = model(x)
        logits = output['logits'].float()
        loss += torch.nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        top5 += (logits.topk(min(5, logits.shape[1]), dim=1).indices == y[:, None]).any(1).sum().item()
        predictions.update(logits.argmax(1).cpu().tolist())
        if 'concepts' in output:
            c = output['concepts'].float().detach().cpu()
            concept_sum = c.sum(0) if concept_sum is None else concept_sum + c.sum(0)
            concept_sq = c.square().sum(0) if concept_sq is None else concept_sq + c.square().sum(0)
        seen += len(y)
    if not seen:
        raise RuntimeError('Validation executed zero examples.')
    return {'accuracy': correct / seen, 'top5_accuracy': top5 / seen if logits.shape[1] >= 5 else None, 'loss': loss / seen, 'n': seen,
            'predicted_classes': len(predictions),
            'concept_std_mean': float((concept_sq / seen - (concept_sum / seen).square()).clamp_min(0).sqrt().mean()) if concept_sum is not None else None}


def learning_plot(run, history):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    x = np.arange(1, len(history) + 1)
    axes[0].plot(x, [r.get('train_accuracy', float('nan')) * 100 for r in history], label='Train')
    axes[0].plot(x, [r['validation_accuracy'] * 100 for r in history], label='Validation')
    axes[0].axhline(history[-1]['chance_accuracy'] * 100, ls='--', color='gray', label='Uniform chance')
    axes[0].set(xlabel='Completed epoch (both stages)', ylabel='Accuracy (%)')
    axes[1].plot(x, [r.get('classification_loss', r['train_loss']) for r in history], label='Train class CE')
    axes[1].plot(x, [r['validation_loss'] for r in history], label='Validation CE')
    axes[1].set(xlabel='Completed epoch (both stages)', ylabel='Cross entropy')
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(Path(run) / 'live_learning_curves.png', dpi=140); plt.close(fig)


def _sync(callback, directory):
    if callback is None:
        return
    try:
        callback(Path(directory))
    except Exception as exc:
        # Keep recoverable local checkpoints when the external storage is unavailable.
        atomic_json(Path(directory) / 'remote_sync_error.json', {'error_type': type(exc).__name__, 'message': str(exc)[:400]})
        warnings.warn('Remote sync failed; local checkpoint is saved. See remote_sync_error.json.')


def fit(root, manifest, config=None, device=None, sync_callback=None, run_name=None, interrupt_after_batches=None, model_type='rgcbm'):
    """Resume stage, epoch, batch, optimizer, LR scheduler, scaler and random states."""
    cfg = config or Config(dataset=manifest['dataset'])
    root = Path(root)
    if cfg.dataset != manifest['dataset']:
        raise ValueError('Configuration and dataset manifest differ.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable. Run the GPU diagnostic first.')
    run = root / 'results' / (run_name or f'{cfg.dataset}_K{cfg.concepts}_seed{cfg.seed}')
    run.mkdir(parents=True, exist_ok=True)
    if model_type not in ['rgcbm', 'backbone']:
        raise ValueError(model_type)
    identity = {'config': cfg.to_dict(), 'manifest_hash': manifest['manifest_hash'], 'classes': manifest['classes'], 'backbone_sha256': None, 'model_type': model_type}
    if cfg.pretrained:
        from .data import sha256
        identity['backbone_sha256'] = sha256(root / 'assets' / 'efficientnet_b1_imagenet.pth')
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config_path = run / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Run identity differs. Use a new run_name; do not overwrite or resume an incompatible experiment.')
    atomic_json(config_path, {**identity, 'fingerprint': fingerprint})
    seed_all(cfg.seed)
    model_class = BackboneClassifier if model_type == 'backbone' else RGCBM
    model = model_class(len(manifest['classes']), cfg, root / 'assets').to(device)
    train_data = ImageDataset(root, manifest, 'train', cfg, augment=True)
    val_data = ImageDataset(root, manifest, 'validation', cfg)
    class_counts = np.bincount([r['label'] for r in train_data.rows], minlength=len(manifest['classes']))
    if np.any(class_counts == 0):
        raise RuntimeError('A training class has zero examples.')
    weights = torch.tensor(len(train_data) / (len(class_counts) * class_counts), dtype=torch.float32, device=device)
    last = run / 'last.pt'
    restored = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
    if restored and restored['fingerprint'] != fingerprint:
        raise ValueError('Checkpoint identity mismatch.')
    if restored:
        model.load_state_dict(restored['model'])
        state = restored['progress']
    else:
        state = {'stage_i': 0, 'epoch': 0, 'batch': 0, 'best_accuracy': -1., 'stage_best': -1., 'bad_epochs': 0, 'epoch_loss_sum': 0., 'epoch_n': 0, 'epoch_correct': 0, 'epoch_classification_sum': 0., 'history': []}
    stages = [('head', cfg.head_epochs, cfg.head_lr), ('finetune', cfg.finetune_epochs, cfg.finetune_lr)]
    total_new_batches = 0
    while state['stage_i'] < len(stages):
        stage, epochs, lr = stages[state['stage_i']]
        if state['epoch'] >= epochs:
            state.update(stage_i=state['stage_i'] + 1, epoch=0, batch=0, stage_best=-1., bad_epochs=0, epoch_loss_sum=0., epoch_n=0)
            restored = None
            continue
        model.set_stage(stage)
        if stage == 'finetune' and state['epoch'] == 0 and state['batch'] == 0 and restored is None:
            head_path = run / 'head_best.pt'
            if not head_path.exists():
                raise RuntimeError('Missing head-stage best checkpoint.')
            model.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True)['model'])
        backbone = [p for p in model.backbone.parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone}
        head = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
        head_lr = cfg.finetune_head_lr if stage == 'finetune' and cfg.finetune_head_lr is not None else lr
        groups = [{'params': head, 'lr': head_lr, 'name': 'head'}]
        if backbone: groups.append({'params': backbone, 'lr': lr, 'name': 'backbone'})
        optimizer = AdamW(groups, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        scaler = torch.amp.GradScaler('cuda', enabled=cfg.mixed_precision and device.type == 'cuda')
        if restored and restored['active_stage'] == stage:
            optimizer.load_state_dict(restored['optimizer'])
            # torch restores optimizer tensors to each parameter's device.
            scheduler.load_state_dict(restored['scheduler'])
            scaler.load_state_dict(restored['scaler'])
            restore_rng(restored['rng'])
        restored = None
        def save():
            atomic_torch(last, {'fingerprint': fingerprint, 'active_stage': stage, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'progress': state, 'rng': rng_state()})
            atomic_json(run / 'training_status.json', {'complete': False, 'status': 'training', 'dataset': cfg.dataset,
                        'stage': stage, 'completed_stage_epochs': state['epoch'], 'stage_epochs': epochs,
                        'batch_in_epoch': state['batch'], 'checkpoint': 'last.pt',
                        'updated_utc': datetime.now(timezone.utc).isoformat(), 'best_validation_accuracy': state['best_accuracy']})
        while state['epoch'] < epochs:
            epoch = state['epoch']
            batches = loader(train_data, shuffle=True, epoch=state['stage_i'] * 10000 + epoch)
            if stage == 'head':
                scale = 0. if epoch < cfg.classification_only_epochs else min(1., (epoch - cfg.classification_only_epochs + 1) / cfg.ramp_epochs)
            else:
                scale = 1.
            model.train()
            initial_classifier = model.classifier.weight.detach().clone()
            total_batches = min(len(batches), cfg.max_train_batches) if cfg.max_train_batches is not None else len(batches)
            progress = tqdm(total=total_batches, initial=min(state['batch'], total_batches),
                            desc=f'{cfg.dataset} | {stage} {epoch+1}/{epochs}', unit='batch', disable=not cfg.show_progress)
            for i, (x, y, _) in enumerate(batches):
                if i < state['batch']:
                    continue
                if cfg.max_train_batches is not None and i >= cfg.max_train_batches:
                    break
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    output = model(x)
                    if model_type == 'backbone':
                        loss = torch.nn.functional.cross_entropy(output['logits'].float(), y, weight=weights)
                    else:
                        loss, terms = loss_terms(model, x, y, output, weights, scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss; previous checkpoint remains intact.')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite training gradient.')
                scaler.step(optimizer)
                scaler.update()
                state['epoch_loss_sum'] += loss.item() * len(y)
                state['epoch_n'] += len(y)
                state['epoch_correct'] = state.get('epoch_correct', 0) + (output['logits'].argmax(1) == y).sum().item()
                classification = loss if model_type == 'backbone' else terms['classification']
                state['epoch_classification_sum'] = state.get('epoch_classification_sum', 0.) + classification.item() * len(y)
                state['batch'] = i + 1
                total_new_batches += 1
                progress.update(1)
                progress.set_postfix(loss=f'{loss.item():.3f}', accuracy=f"{100*state['epoch_correct']/state['epoch_n']:.1f}%", refresh=False)
                if state['batch'] % cfg.checkpoint_every_batches == 0:
                    save()
                    _sync(sync_callback, run)
                    progress.write(f'Checkpoint saved: {run / "last.pt"} (batch {state["batch"]})')
                if interrupt_after_batches is not None and total_new_batches >= interrupt_after_batches:
                    save()
                    _sync(sync_callback, run)
                    progress.close()
                    return model, run, {'complete': False, 'progress': state}
            progress.close()
            if not state['epoch_n']:
                raise RuntimeError('Training executed zero examples.')
            save()  # A validation interruption resumes without losing trained batches.
            measured = validation(model, loader(val_data), device, cfg.max_validation_batches, cfg.show_progress)
            scheduler.step(measured['loss'])
            record = {'stage': stage, 'epoch': epoch + 1, 'train_loss': state['epoch_loss_sum'] / state['epoch_n'], 'train_n': state['epoch_n'], 'validation_accuracy': measured['accuracy'], 'validation_loss': measured['loss'], 'validation_n': measured['n'], 'auxiliary_scale': scale, 'lr': optimizer.param_groups[0]['lr'], 'classifier_l2_update': (model.classifier.weight - initial_classifier).norm().item()}
            record.update(train_accuracy=state['epoch_correct'] / state['epoch_n'], classification_loss=state['epoch_classification_sum'] / state['epoch_n'],
                          chance_accuracy=1 / len(manifest['classes']), validation_top5_accuracy=measured['top5_accuracy'],
                          validation_predicted_classes=measured['predicted_classes'], validation_concept_std_mean=measured['concept_std_mean'],
                          learning_rates={g['name']: g['lr'] for g in optimizer.param_groups})
            state['history'].append(record)
            print(json.dumps(record), flush=True)
            checkpoint = {'model': model.state_dict(), 'config': cfg.to_dict(), 'classes': manifest['classes'], 'fingerprint': fingerprint, 'validation_accuracy': measured['accuracy'], 'model_type': model_type}
            if measured['accuracy'] > state['best_accuracy']:
                state['best_accuracy'] = measured['accuracy']
                atomic_torch(run / 'best.pt', checkpoint)
            if measured['accuracy'] > state['stage_best']:
                state['stage_best'] = measured['accuracy']
                state['bad_epochs'] = 0
                atomic_torch(run / (stage + '_best.pt'), checkpoint)
            else:
                state['bad_epochs'] += 1
            state.update(epoch=epoch + 1, batch=0, epoch_loss_sum=0., epoch_n=0, epoch_correct=0, epoch_classification_sum=0.)
            completed_epoch = epoch + 1
            if state['bad_epochs'] >= cfg.patience:
                state['epoch'] = epochs
            atomic_json(run / 'history.json', state['history'])
            import pandas as pd
            pd.DataFrame(state['history']).to_csv(run / 'history.csv', index=False)
            learning_plot(run, state['history'])
            save()
            if completed_epoch % cfg.checkpoint_every_epochs == 0 or state['epoch'] >= epochs:
                snapshots = run / 'checkpoints'; snapshots.mkdir(exist_ok=True)
                target = snapshots / f'{stage}_epoch_{completed_epoch:04d}.pt'
                temporary = target.with_suffix('.tmp'); shutil.copyfile(last, temporary); os.replace(temporary, target)
                if cfg.keep_epoch_checkpoints:
                    for old in sorted(snapshots.glob(f'{stage}_epoch_*.pt'))[:-cfg.keep_epoch_checkpoints]: old.unlink()
                print('Epoch snapshot saved:', target, flush=True)
            if completed_epoch >= 3 and measured['accuracy'] <= 2 / len(manifest['classes']):
                warnings.warn('Validation remains near uniform chance. Inspect training accuracy, concept variation and class coverage; do not treat this as a successful research run.')
            _sync(sync_callback, run)
    # Retain a final resume record even when the last stage has no epochs.
    if last.exists():
        finished = torch.load(last, map_location='cpu', weights_only=False)
        finished['progress'] = state
        atomic_torch(last, finished)
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model'])
    model.eval()
    atomic_json(run / 'training_status.json', {'complete': True, 'status': 'complete', 'best_validation_accuracy': state['best_accuracy'], 'device': str(device), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'research_run': cfg.max_train_batches is None and cfg.max_validation_batches is None})
    _sync(sync_callback, run)
    return model, run, {'complete': True, 'progress': state}


def load_model(run, asset_dir, device=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load(Path(run) / 'best.pt', map_location='cpu', weights_only=True)
    config = Config(**checkpoint['config'])
    model_class = BackboneClassifier if checkpoint.get('model_type') == 'backbone' else RGCBM
    model = model_class(len(checkpoint['classes']), config, asset_dir).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint['classes']
