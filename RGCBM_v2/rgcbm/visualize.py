from pathlib import Path
import heapq
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from .core import priors
from .data import ImageDataset, loader
from .publication import label as semantic_label, publication_cases, grounding_figures, prototype_plates
from .evaluate import evidence_packet


@torch.no_grad()
def visualize_run(model, root, manifest, run, grounding, examples=3, prototype_batches=40):
    root, run = Path(root), Path(run)
    figures = run / 'figures'
    figures.mkdir(exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    if (run / 'history.json').exists():
        history = json.loads((run / 'history.json').read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot([r['train_loss'] for r in history], label='train objective')
        axes[0].plot([r['validation_loss'] for r in history], label='validation cross-entropy')
        axes[1].plot([r['validation_accuracy'] for r in history], label='validation accuracy')
        for ax in axes:
            ax.set_xlabel('Completed epoch'); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(figures / 'learning_curves.png', dpi=180); plt.close(fig)
    if (run / 'confusion_matrix.npy').exists():
        confusion = np.load(run / 'confusion_matrix.npy')
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(confusion, cmap='Blues'); fig.colorbar(im, ax=ax, label='Image count')
        ax.set(xlabel='Predicted class index', ylabel='True class index', title=f'{manifest["dataset"]}: confusion matrix')
        if len(manifest['classes']) <= 40:
            ticks = np.arange(len(manifest['classes']))
            ax.set_xticks(ticks, manifest['classes'], rotation=90, fontsize=7)
            ax.set_yticks(ticks, manifest['classes'], fontsize=7)
        fig.tight_layout(); fig.savefig(figures / 'confusion_matrix.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    y = [r['mean'] for r in grounding]
    error = [[max(0, r['mean'] - r['ci_low']) for r in grounding], [max(0, r['ci_high'] - r['mean']) for r in grounding]]
    ax.bar(np.arange(len(y)), y, yerr=error, color=['#b5bbc3' if r['weak_grounding'] else '#327da8' for r in grounding], capsize=2)
    ax.axhline(0, color='black', lw=.5)
    ax.set_xticks(np.arange(len(y)), [semantic_label(r,22) for r in grounding], rotation=90, fontsize=6)
    ax.set(xlabel='Named concept', ylabel='Validation descriptor correlation', title='Grounding with image-bootstrap 95% intervals')
    fig.tight_layout(); fig.savefig(figures / 'grounding_uncertainty.png', dpi=180); plt.close(fig)
    test = ImageDataset(root, manifest, 'test', model.config)
    selected = np.linspace(0, len(test) - 1, min(examples, len(test)), dtype=int)
    for case, index in enumerate(selected):
        image, label, _ = test[int(index)]
        x = image[None].to(device)
        output = model(x)
        descriptor = priors(x)
        rgb = image.permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        axes[0].imshow(rgb); axes[0].set_title(f'True: {manifest["classes"][label]}', fontsize=9)
        for ax, name in zip(axes[1:], descriptor):
            ax.imshow(descriptor[name][0, 0].cpu(), cmap='magma', vmin=0, vmax=1); ax.set_title(name)
        for ax in axes: ax.axis('off')
        fig.tight_layout(); fig.savefig(figures / f'case_{case:02d}_priors.png', dpi=180); plt.close(fig)
        maps = torch.nn.functional.interpolate(output['maps'], size=x.shape[-2:], mode='bilinear', align_corners=False)[0].cpu().numpy()
        cols = 4; rows = int(np.ceil(len(grounding) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), squeeze=False)
        for k, ax in enumerate(axes.flat):
            if k < len(grounding):
                ax.imshow(rgb); ax.imshow(maps[k], cmap='inferno', alpha=.5, vmin=0, vmax=1)
                ax.set_title(semantic_label(grounding[k],29), fontsize=8)
            ax.axis('off')
        fig.tight_layout(); fig.savefig(figures / f'case_{case:02d}_concept_maps.png', dpi=160); plt.close(fig)
        packet = evidence_packet(model, output['concepts'][0].cpu().numpy(), output['logits'][0].cpu().numpy(), manifest['classes'], grounding)
        terms = np.array([r['margin_contribution'] for r in packet['concepts']])
        order = np.argsort(np.abs(terms))[-min(12, len(terms)):]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh([semantic_label(grounding[k],35) for k in order], terms[order], color=['#327da8' if t > 0 else '#c95454' for t in terms[order]])
        ax.axvline(0, color='black', lw=.7)
        ax.set(xlabel='Exact predicted-versus-runner-up logit contribution', title=f'{packet["prediction"]["class"]} vs {packet["prediction"]["runner_up"]}')
        fig.tight_layout(); fig.savefig(figures / f'case_{case:02d}_contributions.png', dpi=180); plt.close(fig)
    # Each candidate is a distinct training image, selected by its peak concept response.
    train = ImageDataset(root, manifest, 'train', model.config)
    best = [[] for _ in grounding]
    for batch_i, (images, labels, indices) in enumerate(loader(train)):
        if prototype_batches is not None and batch_i >= prototype_batches: break
        response = model(images.to(device))['maps'].cpu().numpy()
        for b, index in enumerate(indices.tolist()):
            rgb = images[b].permute(1, 2, 0).numpy()
            for k in range(len(grounding)):
                grid = response[b, k]
                y, x = np.unravel_index(grid.argmax(), grid.shape)
                center_y, center_x = int((y + .5) * rgb.shape[0] / grid.shape[0]), int((x + .5) * rgb.shape[1] / grid.shape[1])
                patch = rgb[max(0, center_y - 40):center_y + 40, max(0, center_x - 40):center_x + 40].copy()
                bounds = (max(0, center_x-40), max(0, center_y-40), min(rgb.shape[1], center_x+40), min(rgb.shape[0], center_y+40))
                candidate = (float(grid.max()), index, patch, rgb.copy(), bounds, grid.copy(), int(labels[b]))
                if len(best[k]) < 3: heapq.heappush(best[k], candidate)
                elif candidate[0] > best[k][0][0]: heapq.heapreplace(best[k], candidate)
    fig, axes = plt.subplots(len(grounding), 3, figsize=(8, 1.6 * len(grounding)), squeeze=False)
    for k, candidates in enumerate(best):
        for j, ax in enumerate(axes[k]):
            if j < len(candidates):
                value, index, patch, *_ = sorted(candidates, key=lambda v: v[0], reverse=True)[j]
                ax.imshow(patch); ax.set_title(semantic_label(grounding[k],30)+f'\nresponse={value:.3f}', fontsize=7)
            ax.axis('off')
    fig.suptitle('Training prototype patches; descriptor association requires separate grounding', fontsize=11)
    fig.tight_layout(); fig.savefig(figures / 'prototype_patches.png', dpi=150); plt.close(fig)
    grounding_figures(run, grounding)
    publication_cases(model, root, manifest, run, grounding, test, selected)
    prototype_plates(run, grounding, best, train)
    return figures
