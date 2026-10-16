# RG-CBM multi-dataset research package

This revision preserves the original concept bottleneck while preparing **CUB-200-2011 (200 classes)**,
**Oxford-IIIT Pet (37 breeds)** and the original **MRI dataset (4 classes)**. Windows-native NVIDIA
Blackwell support uses **PyTorch 2.7.1 + CUDA 12.8**, with EfficientNet-B1, 128-dimensional projection,
64 CUB / 16 Pet-MRI concept prototypes, descriptor losses and a sparse linear classifier. The original TensorFlow
notebook is preserved in the repository; old TensorFlow weights and reported paper metrics are
not interchangeable with this implementation.

The current `guided_v4` notebook profile adds individual measurable descriptor targets, exposure-aware
checkpoint selection and named visual evidence. It retains the `calibrated_v3` training improvements:
training-only running standardization of pooled concept scores and separate learning rates.
See [the concept-guidance contract and figure guide](docs/CONCEPT_GUIDANCE.md). The earlier V3 profile added training-only running standardization of pooled
concept scores (followed by sigmoid), an 8-epoch head stage and up to 30 fine-tuning epochs with
independent head/backbone learning rates. This is an explicit methodology revision; class
probabilities remain uncalibrated. See [training diagnosis and automatic runs](docs/TRAINING_AND_DATASET_QUEUE.md).

For the GitHub distribution, first follow [offline ZIP assembly instructions](https://github.com/uroobasehar/GCP-CBM/blob/main/RGCBM_v2/offline_bundle/README.md).

## Start on the offline Windows 11 machine

Read [offline system prerequisites](docs/OFFLINE_PREREQUISITES.md) before permanent disconnection.
It includes the driver, browser, filesystem and certificate checks, plus a native acceptance test.
The [dependency review](docs/OFFLINE_DEPENDENCY_REVIEW.md) records the corrected PyArrow/NumPy
conflict, regenerated lock, package isolation and additional native-library validation.
The Intel NPU is optional; this pipeline uses the NVIDIA GPU and CPU. Shared graphics memory is not
additional dedicated VRAM.

1. Obtain the complete offline bundle, reconstruct its ZIP if distributed in parts, and extract it
   to a writable local folder such as `D:\RGCBM`. Allow at least **60 GB free disk** when retaining
   source ZIPs, installed dependencies, extracted data, checkpoints and distribution parts.
2. Before disconnecting, ensure the machine has a current NVIDIA **RTX PRO Blackwell** driver that
   supports CUDA 12.8. The GPU driver is system-specific and is **not included**. PyTorch includes
   its CUDA runtime; a separate CUDA toolkit/Anaconda CUDA installation is unnecessary.
3. In an Anaconda PowerShell Prompt, from the extracted folder, run
   `powershell -ExecutionPolicy Bypass -File .\offline\setup_windows.ps1`.
   The script validates SHA-256 checksums and the official Microsoft runtime installer signature,
   creates an actual Anaconda environment from bundled Windows conda packages using **conda --offline**,
   installs only local GPU wheels and checks actual GPU execution. The environment is `conda-env`
   inside the extracted folder; your Anaconda base environment is not changed. ExecutionPolicy is
   scoped to that script; signature/checksum checks remain enabled. Administrator approval may be
   required for the Microsoft VC++ runtime.
4. Run `offline\launch_jupyter.cmd` and open `notebooks/RGCBM_Offline.ipynb` with the
   **RG-CBM Anaconda offline GPU (Python 3.11)** kernel. Set the dataset/run parameters and execute cells.
   Archives are extracted locally once; training/reports do not require Internet. The default
   `LLM_BACKEND='local_qwen'` uses the bundled LLM; choose `'template'` for faster deterministic reports.

Your driver version was not supplied and Windows/GPU execution cannot be tested from this CPU Linux
workspace. `offline/gpu_check.py --require-gpu` performs the mandatory driver/kernel/model check on
the actual machine. Resolve driver/Windows certificate prerequisites while still online. If setup
fails, preserve the error; do not bypass signature, checksum or dependency checks.

## Cloud notebooks

Current Colab uses Python 3.13. The notebook automatically selects TensorFlow 2.20.0 there;
cloud Python 3.10–3.12 and the Windows offline lock retain TensorFlow 2.19.1.
See [Python compatibility and fresh Python 3.13 validation](docs/PYTHON_COMPATIBILITY.md).

`notebooks/RGCBM_Colab.ipynb` uses a Google Drive mount and local image extraction.
`notebooks/RGCBM_Kaggle.ipynb` uses Google Drive API uploads/downloads and a securely provided OAuth
token, because Kaggle cannot mount Drive using Colab's API. Both embed the complete shared Python
implementation and can obtain the datasets online. See [cloud storage instructions](docs/CLOUD_STORAGE.md).
Run All processes each benchmark automatically with independent models; edit DATASETS to choose a subset.

All three notebooks default to `RUN_ALL_DATASETS=True` with `DATASETS=['cub','pets','brain_mri']`.
Run All previews the first dataset in detailed cells, then automatically trains/evaluates the full
queue with independent models and saved explanations. Dataset summary CSV/status/chart are stored
alongside the runs and synced to configured Drive storage. Restarting the same configuration resumes
training and reuses completed evidence. `RUN_TAG='guided_v4'` preserves the older run folders.

Each notebook contains an **Ablation study** code block. Set `RUN_ABLATIONS=True` to execute full,
unguided, individual guidance/regularizer removals, no-fine-tuning, legacy group guidance and K=16/32/64 variants with isolated
checkpoints. The new `no_standardization` variant isolates the score-standardization change with
the same learning rates and schedule. In all-dataset mode the sweep repeats on each dataset.
Per-run and mean/std CSVs are saved under `results/<dataset>_ablation_summary_guided_v4` and synced
to configured Drive storage. Defaults leave the expensive sweep disabled. `ABLATION_SEEDS` and
`ABLATION_VARIANTS` are editable; smoke outputs must not be reported as scientific findings.

For the matched feature-classifier baseline, from this folder run
`python experiments.py --dataset cub --variant backbone --batch-size 24`.
The same script supports individual ablations, seeds and `--smoke` checks.

## Additional requested libraries

The environment includes TensorFlow **2.19.1**, NumPy, pandas, Matplotlib, Pillow, scikit-learn,
MedMNIST, KaggleHub (<1), and TensorFlow Datasets (4.9.x), satisfying your requested ranges.
Compatible Protobuf/metadata/API-core versions are pinned and tested. Windows TensorFlow is CPU-only;
PyTorch + CUDA 12.8 still performs Blackwell GPU training. Each notebook checks these versions without
initializing TensorFlow or reserving GPU memory.

## Progress, explanations and outputs

Checkpoints include stage, epoch, batch, model, optimizer, scheduler, mixed-precision scaler and RNG.
Rerun the same configuration/run name to resume; use a new run name for changed settings/ablations.
Progress is committed every 50 batches and each epoch. A crash can require repeating work since the
last committed batch. Full epoch snapshots are retained every five epochs and at stage end, with
three per stage by default. Batch bars, training accuracy/CE, class coverage and concept variation
are visible; live curves, status JSON and history CSV are saved during training.
`last.pt` is a trusted, locally generated full-state checkpoint; do not load
arbitrary untrusted pickle checkpoints. `best.pt` is selected on validation only.

Outputs include history, predictions, class metrics/confusion matrices, calibration, concept arrays,
frozen descriptor grounding and uncertainty, model-dependence/stability/localization audits, exact
evidence JSON, report provenance, and figures: learning curves, confusion matrices, priors, signed
contributions, concept maps and prototype patches. The classifier supplies prediction/indices/signs deterministically; local Qwen supplies only prose.
Descriptor-name/weak-evidence gates record explicit fallback; prose still requires human review.
New named prototype plates and case heatmaps include 300 dpi PNG/PDF/SVG, raw spatial evidence,
source image coordinates and guidance-target comparisons. See docs/CONCEPT_GUIDANCE.md.

Additional journal-facing notebook blocks produce domain-scoped concept evidence cards, reliability
diagrams, redundancy matrices, editable latent interventions, matched deletion curves with paired
bootstrap intervals, and actual-device inference profiling. An optional CUB attribute audit uses
official annotations **after training**, with validation-only name selection, sample/strength thresholds
and multiple-comparison correction, followed by held-out AUC. Its annotation-assisted associations
are disclosed separately; weak evidence keeps the original neutral descriptor name. No unsupported
bird-part, pet-anatomy or clinical label is automatically invented.

The [research review](docs/RESEARCH_REVISION.md) explains the base studies, dataset fit, method limitations,
draft/code mismatches and comparison protocol. [Revised methodology text](docs/METHODOLOGY.md) is ready
to adapt into the paper. The methodology figure is supplied as SVG, PDF and PNG in `docs/`.
Use three full seeds and matched reruns before writing result claims; software smoke tests are not
research results. Descriptor correlations do not establish named parts, clinical findings or causal image mechanisms.

## What was validated

The current Colab and Kaggle notebooks passed all 20 code cells locally, including the automatic
CUB/Pet/MRI queue, on Python 3.13 and NumPy 2.1.3. The offline notebook passed all 18 code cells,
including the three-dataset queue and two selected ablations on each dataset. Pip calls and cloud
mount/OAuth/Drive endpoints were substituted; actual training, checkpointing, figures and journal
blocks executed. See `docs/guidance_v4/` for current evidence and `docs/training_v3/` for the previous protocol. Drive API contract tests cover cold restore,
newer-local-checkpoint preservation and complete LLM assets. See `docs/cloud_notebook_validation.json`.
Live platform GPUs and authenticated Drive sessions still require target-platform validation.

The shared implementation passed actual CUB/Pet/MRI CPU training steps in both stages, checkpoint
reload, classification/explanation exports, independent-annotation evaluation, reports and figures.
Interrupted/resumed training produced bitwise-identical CPU weights and rejected incompatible runs.
Full GPU training, Windows installation, cloud authentication and publication-quality model accuracy
remain to be validated on their target systems. Detailed evidence is in `docs/validation.json` and
illustrative **smoke-only** outputs in `docs/validation_examples`.

The exact supplied MCPNet++ PDF exceeded the attachment-transfer limit. The public CVPR MCPNet paper,
official source README and MCPNet++ DOI metadata were recovered; extension-specific full-text claims
remain unverified. The reviewed empirical base is MCPNet; DCBM supplies the complementary bottleneck
comparison. See the study table before citing results from either version.
