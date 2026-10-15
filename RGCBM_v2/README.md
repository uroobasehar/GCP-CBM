# RG-CBM multi-dataset research package

This revision preserves the original concept bottleneck while preparing **CUB-200-2011 (200 classes)**,
**Oxford-IIIT Pet (37 breeds)** and the original **MRI dataset (4 classes)**. Windows-native NVIDIA
Blackwell support uses **PyTorch 2.7.1 + CUDA 12.8**, with EfficientNet-B1, 128-dimensional projection,
16 concept prototypes, descriptor losses and a sparse linear classifier. The original TensorFlow
notebook is preserved in the repository; old TensorFlow weights and reported paper metrics are
not interchangeable with this implementation.

For the GitHub distribution, first follow [offline ZIP assembly instructions](https://github.com/uroobasehar/GCP-CBM/blob/main/RGCBM_v2/offline_bundle/README.md).

## Start on the offline Windows 11 machine

Read [offline system prerequisites](docs/OFFLINE_PREREQUISITES.md) before permanent disconnection.
It includes the driver, browser, filesystem and certificate checks, plus a native acceptance test.
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
   template is immediate; choose `LLM_BACKEND='local_qwen'` to use the bundled local LLM.

Your driver version was not supplied and Windows/GPU execution cannot be tested from this CPU Linux
workspace. `offline/gpu_check.py --require-gpu` performs the mandatory driver/kernel/model check on
the actual machine. Resolve driver/Windows certificate prerequisites while still online. If setup
fails, preserve the error; do not bypass signature, checksum or dependency checks.

## Cloud notebooks

`notebooks/RGCBM_Colab.ipynb` uses a Google Drive mount and local image extraction.
`notebooks/RGCBM_Kaggle.ipynb` uses Google Drive API uploads/downloads and a securely provided OAuth
token, because Kaggle cannot mount Drive using Colab's API. Both embed the complete shared Python
implementation and can obtain the datasets online. See [cloud storage instructions](docs/CLOUD_STORAGE.md).
Change the dataset selector to repeat the same workflow on each benchmark; train separate models.

Each notebook contains an **Ablation study** code block. Set `RUN_ABLATIONS=True` to execute full,
unguided, individual guidance/regularizer removals, no-fine-tuning and K=32/64 variants with isolated
checkpoints. Per-run and mean/std CSVs are saved under `results/<dataset>_ablation_summary` and synced
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
last committed batch. `last.pt` is a trusted, locally generated full-state checkpoint; do not load
arbitrary untrusted pickle checkpoints. `best.pt` is selected on validation only.

Outputs include history, predictions, class metrics/confusion matrices, calibration, concept arrays,
frozen descriptor grounding and uncertainty, model-dependence/stability/localization audits, exact
evidence JSON, report provenance, and figures: learning curves, confusion matrices, priors, signed
contributions, concept maps and prototype patches. Local Qwen reports validate prediction/indices/
signs and record fallback status; prose still requires human review.

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

The new Colab and Kaggle notebooks also passed complete local real-CUB CPU smoke execution with
NumPy 2.1.3. Pip calls and cloud mount/OAuth/Drive endpoints were substituted; actual training,
checkpointing, figures and journal blocks executed. Drive API contract tests cover cold restore,
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
