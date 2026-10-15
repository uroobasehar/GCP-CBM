# Training diagnosis and automatic dataset runs

The reported CUB validation accuracies, 0.83–1.5%, are low compared with a 0.5% uniform chance
baseline for 200 classes. Chance cross entropy is log(200)≈5.298; the reported validation CE falls
from 5.293 to 5.208. Parameters are updating and the loss is improving, but discrimination remains
weak. Four frozen-backbone epochs do not determine final fine-tuned performance. The logs cannot
by themselves establish a data-label bug, missing pretrained weights or a broken GPU.

## Controlled learning diagnostic

`diagnostics/cub_learning_probe.py` uses real frozen ImageNet features from 600 CUB training images
(three/class) and 400 validation images (two/class). It runs 15 classification-only head epochs,
with augmentation and auxiliary losses disabled to isolate optimization. No test images are used.

| Head variant | Final training accuracy | Best validation accuracy |
|---|---:|---:|
| Original mean-score path, LR 4e−4 | 1.83% | 1.50% |
| Same path, LR 2e−3 | 0.67% | 1.25% |
| Running standardization + sigmoid, LR 2e−3 | 49.50% | 4.25% |

Initial pooled-score SD averaged about 0.020 across concepts. Standardization improves the fitting
signal, while increasing LR alone did not. The large training/validation gap in this small,
unaugmented diagnostic is also evidence of overfitting. **These are diagnostic measurements,
not full CUB benchmark results or evidence of publication-ready accuracy.** Full GPU training,
multiple seeds and independent held-out comparisons remain necessary. Raw histories, plot and
workflow validation reports are under `docs/training_v3/`.

## Explicit new training profile

The `calibrated_v3` profile preserves EfficientNet-B1, 128-dimensional projection, 16 prototypes,
three descriptor-guided groups, the unguided group and the sparse linear bottleneck. It changes:

- Pooled response → non-affine BatchNorm1d → sigmoid. Statistics are learned on training batches
  only and frozen during evaluation. Spatial maps and guidance retain their original definition.
- Head-only training: 8 epochs, LR 2e−3. Fine-tuning: up to 30 epochs with head LR 1e−3 and
  backbone LR 2e−5. The old shared fine-tuning LR 8e−6 slowed adaptation of the randomly initialized head.
- Diagnostics: training accuracy and class CE, validation top-1/top-5, predicted-class count,
  mean per-concept variation, chance baseline and separate optimizer learning rates.

Class logits still equal bias plus the sum of signed concept terms exactly. Standardization is
not probability calibration. A score of 0.5 denotes the estimated training-mean response, not 50%
probability that a named part exists. Concept cards/evidence record this score definition. Zero
edits are latent clamps and may be off the image manifold. Names still require measured descriptor
grounding and do not become anatomical/species concepts through standardization.

The new matched `no_standardization` ablation uses the same new learning rates and schedule. The
CLI's `--profile legacy` permits separately named legacy comparisons; old V2 best checkpoints remain
loadable for inference. **V3 training starts a new run** because normalization adds running state
and the optimization protocol changes. Continue a V2 training job using its original saved code;
do not rename/copy an old last.pt into a V3 folder. Existing checkpoints are preserved.

## Run all datasets automatically

Open the latest notebook, restart its kernel after switching code versions, and Run All:

```python
RUN_ALL_DATASETS = True
DATASETS = ['cub', 'pets', 'brain_mri']
RUN_TAG = 'calibrated_v3'
CHECKPOINT_EVERY_EPOCHS = 5
KEEP_EPOCH_CHECKPOINTS = 3  # per stage; 0 keeps all periodic snapshots
```

Detailed cells preview the first dataset. The automatic queue reuses its completed results, then
prepares/trains/evaluates the others one at a time. It creates separate checkpoints, class maps,
normalization statistics, validation grounding, explanations, figures and journal audit outputs.
The optional ablation cell also loops across DATASETS when enabled. Dataset failures are recorded
and the queue continues; a summary with failed rows is not a successful complete experiment.

Colab restores/copies each dataset archive from/to Drive and writes results directly on mounted
Drive. Kaggle restores/syncs each dataset/run through the existing authenticated Drive API.
Offline Jupyter uses the bundled local data and models. No extra dependency was added.
In a fresh Kaggle session the restore helper recovers training checkpoints first. Missing local
evaluation/figure files are regenerated from the restored best model; completed training is reused.

For K=16 and seed=42, results follow:

```text
results/
  cub_K16_seed42_calibrated_v3/
  pets_K16_seed42_calibrated_v3/
  brain_mri_K16_seed42_calibrated_v3/
  dataset_suite_calibrated_v3_K16_seed42/
    queue_status.json
    dataset_summary.csv
    dataset_accuracy.png
```

The summary chart compares different classification tasks; it is not a matched difficulty ranking.
Smoke runs add `_smoke` to names. Rerun with the identical configuration to resume and reuse completed
evidence. Use a new RUN_TAG when changing an experiment. Preserving old summaries prevents accidental
comparison of mismatched protocols. Saved Drive notebook copies do not update automatically.

## Progress and checkpoints

Batch progress bars show dataset/stage/epoch, loss, training accuracy and ETA. Epoch logs report
both accuracy and CE diagnostics. Each run saves:

- `last.pt`: complete resume state every 50 batches and every epoch. At most the unsaved batches
  repeat after interruption. This is independent of periodic snapshots.
- `checkpoints/head_epoch_XXXX.pt` and `finetune_epoch_XXXX.pt`: full optimizer/scheduler/scaler/RNG
  snapshots every configured interval and at stage end; three per stage retained by default.
- `best.pt`, `head_best.pt`, `finetune_best.pt`: validation-selected inference/model-selection checkpoints.
- `training_status.json`, `history.json`, `history.csv`, `live_learning_curves.png`: persistent progress
  and curves during training. The live plot is a saved file; progress bars/logs appear in cell output.

Checkpoint writes are atomic. Colab writes them on Drive; Kaggle syncs saved files with last.pt last.
Drive/network errors are reported and local checkpoints retained. This cannot keep a cloud runtime
alive past Colab/Kaggle limits; restart and resume the same run when needed.

CPU smoke checks passed on actual images for all datasets, exact interrupted/resumed weights including
normalization buffers, independent optimizer groups, checkpoint retention and completed-queue reuse.
Cloud tests use explicit local substitutions for managed pip/mount/OAuth/Drive services. Native
Windows GPU execution, live cloud authentication and full-dataset research accuracy remain unverified.
