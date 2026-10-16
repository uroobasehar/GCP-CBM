# Guidance V4 evidence and reproducibility

This directory distinguishes software smoke checks from learning diagnostics. None of the included
small-run classification scores are research results or evidence of high final test accuracy.
Concept evidence retains explicit weak/unverified guidance flags; new publication panels carry smoke watermarks.

- `full_cub_probe.json` and `cub_capacity_diagnostic.*`: ten-epoch frozen-feature learning diagnosis,
  5,394 train / 600 validation, one seed, no test images. Best validation: pooled-feature classifier
  72.0%, K16 bottleneck 35.33%, K64 bottleneck 53.33%. This uses cached augmentation, a simplified
  objective and no fine-tuning, not the complete RG-CBM comparison protocol. The portable script is
  `../../diagnostics/full_cub_frozen_probe.py`.
- `grounded_xai_validation.json`: real-image guidance gradients, distinct channel targets,
  positive/inverse synthetic association checks, failed-target handling, actual checkpoint exposure,
  finite descriptors under autocast, named figure exports and exact signed spatial allocation.
- `cloud_notebook_validation.json`: local execution of Colab and Kaggle notebook cells using real
  CUB/Pet/MRI images and automatic three-dataset queues, Python 3.13 / NumPy 2.1.3. Installed pins were
  checked; pip, authenticated Drive/OAuth and managed mounts were substituted. This is not a live
  Colab/Kaggle GPU or authenticated Google Drive test.
- `notebook_validation.json`: all offline Jupyter code cells, real-image three-dataset queue and
  no-fine-tuning / K32 ablations on every dataset. Internet sockets denied externally through
  inherited seccomp; Jupyter uses Unix IPC. Python 3.12 Linux CPU, not native Windows/Blackwell.
- `training_workflow_validation.json`: exact CPU interrupted/resumed weights, frozen inference
  running statistics, stage-specific optimizer groups, periodic checkpoint retention and queue reuse.
- `local_qwen_validation/`: actual bundled local Qwen inference without network on all three domains.
  Initial failed generations are preserved separately. Current reports record genuine LLM use or
  explicitly labeled deterministic fallback and reasons; CUB/Pet drafts in this check were rejected
  for unsupported physical-level claims, while the MRI draft passed the gates. Passing gates does not
  replace human review.
- `v4-ablation-contracts.log`: all 15 advertised variants construct valid configurations for all three
  datasets, including the corrected legacy-group dispatch. CLI and notebook sweeps share that dispatch.
- `cloud_smoke_outputs/`, `offline_smoke_outputs/`, `xai_regression_outputs/`: complete non-checkpoint
  evidence, provenance, tabular exports and figures of these validation runs. Generated smoke model
  checkpoints are omitted from this documentation distribution; production runs save checkpoints
  under results. Do not reuse smoke weights as trained models.

Notebook execution uses K16 and tiny batches to check software paths; the full-pool frozen-feature
learning diagnostic separately exercises K64. Full training defaults to CUB K64 and Pet/MRI K16.
Following the full notebook executions, the isolated legacy-ablation dispatch fix was covered by
configuration checks. The revised qualitative-only LLM prompt and PyTorch-only import guard were
tested separately with actual local inference. A final multiline-label spacing fix was validated by
regenerating all case panels from the same saved models (figure_layout_validation.json). Portable
notebooks were regenerated from that shared source. Package dependencies remain unchanged from the previously audited lock.
Windows setup/native CUDA, driver compatibility, actual cloud authentication, complete multi-seed
accuracy and scientific explanation quality remain target-system/experimental checks.

The original reference notebook's SHA-256 remains
`0b6f7b0e2a964dc851b470a1a540c07356e457f83c4bc597e8ea10fd52e7828d`.
Read ../CONCEPT_GUIDANCE.md and ../METHODOLOGY.md before adapting text or figures for the paper.
