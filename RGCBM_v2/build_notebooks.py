from pathlib import Path
import json
import nbformat

ROOT = Path(__file__).resolve().parent
payload = {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8') for p in sorted((ROOT / 'rgcbm').glob('*.py'))}
payload['prepare_assets.py'] = (ROOT / 'prepare_assets.py').read_text()
requirements = [line.strip() for line in (ROOT / 'cloud_requirements.txt').read_text().splitlines() if line.strip() and not line.startswith('#')]
bootstrap = (ROOT / 'cloud_bootstrap.py').read_text()

def markdown(text): return nbformat.v4.new_markdown_cell(text)
def code(text): return nbformat.v4.new_code_cell(text)

config_cell = '''# Run All trains/evaluates these datasets in order, with independent models/results.
RUN_ALL_DATASETS = True
DATASETS = ['cub', 'pets', 'brain_mri']
DATASET = DATASETS[0]           # Detailed figures below preview the first dataset.
RUN_TAG = 'guided_v4'       # New protocol: preserves all older run folders/checkpoints.
SEED = 42                      # Repeat independent full runs with 42, 43, 44.
CONCEPTS_BY_DATASET = {'cub': 64, 'pets': 16, 'brain_mri': 16}
NUM_CONCEPTS = CONCEPTS_BY_DATASET[DATASET]  # Capacity ablations include fixed 16/32/64.
BATCH_SIZE = 24                 # 32 GB GPU: start here; lower if CUDA memory is exhausted.
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 30
HEAD_LR = 2e-3
FINETUNE_BACKBONE_LR = 2e-5
FINETUNE_HEAD_LR = 1e-3
CONCEPT_NORMALIZATION = 'running_standardize'  # 'none' reproduces the old pooling path.
CHECKPOINT_EVERY_EPOCHS = 5
KEEP_EPOCH_CHECKPOINTS = 3      # Per stage; 0 retains all periodic snapshots.
SMOKE_TEST = False              # True executes two real training batches per stage; not paper results.
LLM_BACKEND = 'local_qwen'        # 'local_qwen' (offline-capable) or 'gemini' (online key required).
LLM_DEVICE = 'auto'              # 'cuda' also works; release training tensors before loading Qwen.
RUN_NAME = f'{DATASET}_K{NUM_CONCEPTS}_seed{SEED}_{RUN_TAG}' + ('_smoke' if SMOKE_TEST else '')
print('Dataset:', DATASET, 'run:', RUN_NAME)
'''

for mode in ['Offline', 'Colab', 'Kaggle']:
    cells = [markdown(f'''# RG-CBM: descriptor-grounded visual concept bottleneck — {mode}

CUB-200-2011, Oxford-IIIT Pet and MRI share the original architecture: EfficientNet-B1 →
128-dimensional projection → cosine-matched prototypes → domain-named calibrated concepts (CUB: 64; Pet/MRI: 16) → sparse linear classifier.
The three guided groups model edge, texture and image-derived region priors; the fourth is unguided.
CUB capacity was chosen using a disclosed frozen-feature validation diagnostic, not test accuracy.
Boundary, texture and region channels receive distinct measurable targets. Validation audits each intended target;
unverified/conflicting guidance stays labeled. Names never rotate by index to invent anatomy or bird parts.

Use the supplied README and research methodology. This notebook does not contain fabricated research
results. Full training and independent held-out evaluation must run on your GPU. Resume by rerunning
the same configuration; incompatible checkpoints are rejected. Change RUN_NAME for a new experiment.
Run All executes the first dataset's detailed cells, then the automatic queue processes every
remaining dataset and writes a combined summary. Concept calibration is an explicit protocol
revision; it uses training running statistics fixed at inference, and preserves the linear bottleneck.
'''), code(config_cell)]
    if mode == 'Offline':
        cells += [markdown('Run `offline/setup_windows.ps1` first. No network operation occurs in this notebook unless you explicitly select an online LLM. The bundled datasets and backbone are required. Use the **RG-CBM Anaconda offline GPU (Python 3.11)** kernel.'), code('''from pathlib import Path
import os, sys
ROOT = Path(os.getenv('RGCBM_ROOT', str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))).resolve()
assert (ROOT / 'data').is_dir() and (ROOT / 'assets').is_dir(), 'Set RGCBM_ROOT to the extracted bundle directory.'
ONLINE = False
SYNC_CALLBACK = None
''')]
    else:
        cells += [markdown('Internet access is required for initial preparation. Enable a GPU in runtime settings. Dependencies are pinned; restart the kernel/runtime if an already imported package version changed.'), code(bootstrap + f'''
prepare_cloud_runtime({requirements!r})
from pathlib import Path
import os, shutil
ROOT = Path(os.getenv('RGCBM_ROOT', '/content/rgcbm' if '{mode}' == 'Colab' else '/kaggle/working/rgcbm'))
ROOT.mkdir(parents=True, exist_ok=True)
ONLINE = True
SYNC_CALLBACK = None
''')]
    cells += [markdown('The shared implementation is embedded for portability. The original research notebook is preserved. Working code is materialized in a separate folder; data, assets and results remain outside it.'), code('''import json, sys
CODE = ''' + repr(payload) + '''
CODE_ROOT = ROOT if (ROOT / 'rgcbm' / '__init__.py').exists() else ROOT / '_notebook_code'
if CODE_ROOT != ROOT:
    for relative, content in CODE.items():
        target = CODE_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
'''), markdown('''## Requested environment compatibility checks

The requested TensorFlow/scientific/MedMNIST/KaggleHub/TFDS libraries are included. This check reads
package metadata without initializing TensorFlow or reserving GPU memory. Native Windows TensorFlow
2.19.1 is CPU-only; the revised model uses PyTorch CUDA 12.8 for Blackwell GPU training.
Cloud Python 3.13 selects TensorFlow 2.20.0 because 2.19.1 has no Python 3.13 wheel.
Python 3.10–3.12 and the Windows offline environment retain TensorFlow 2.19.1.
'''), code(f'''from rgcbm.environment import check_requested_libraries
display(check_requested_libraries(profile={('offline' if mode == 'Offline' else 'cloud')!r}))
''')]
    if mode == 'Colab':
        cells += [markdown('Google Drive stores source archives, backbone assets, run manifests, progress, checkpoints and figures. Training images are extracted to local runtime storage for speed. Google Drive authentication is interactive and cannot be pretested here.'), code('''from google.colab import drive
drive.mount('/content/drive', force_remount=False)
DRIVE_ROOT = Path(os.getenv('RGCBM_DRIVE_ROOT', '/content/drive/MyDrive/RGCBM_MultiDataset'))
for folder in ['datasets', 'assets', 'results']:
    (DRIVE_ROOT / folder).mkdir(parents=True, exist_ok=True)
# Atomic checkpoint writes are made directly into this mounted Drive results folder.
local_results = ROOT / 'results'
if local_results.exists() and not local_results.is_symlink():
    if any(local_results.iterdir()):
        raise RuntimeError('Local results already exist. Preserve/copy them before switching storage.')
    local_results.rmdir()
if not local_results.exists():
    local_results.symlink_to(DRIVE_ROOT / 'results', target_is_directory=True)
(ROOT / 'data').mkdir(exist_ok=True)
archive = ROOT / 'data' / f'{DATASET}.zip'
if not archive.exists() and (DRIVE_ROOT / 'datasets' / archive.name).exists():
    shutil.copy2(DRIVE_ROOT / 'datasets' / archive.name, archive)
(ROOT / 'assets').mkdir(exist_ok=True)
weight = ROOT / 'assets' / 'efficientnet_b1_imagenet.pth'
if not weight.exists() and (DRIVE_ROOT / 'assets' / weight.name).exists():
    shutil.copy2(DRIVE_ROOT / 'assets' / weight.name, weight)
if LLM_BACKEND == 'local_qwen':
    for filename in ['llm_manifest.json']:
        if (DRIVE_ROOT / 'assets' / filename).is_file():
            shutil.copy2(DRIVE_ROOT / 'assets' / filename, ROOT / 'assets' / filename)
    if (DRIVE_ROOT / 'assets' / 'qwen2.5-1.5b-instruct').is_dir():
        shutil.copytree(DRIVE_ROOT / 'assets' / 'qwen2.5-1.5b-instruct', ROOT / 'assets' / 'qwen2.5-1.5b-instruct', dirs_exist_ok=True)
''')]
    elif mode == 'Kaggle':
        cells += [markdown('Kaggle cannot mount Google Drive through `google.colab.drive`. Use the supported Drive API with a securely stored user OAuth token. Follow `docs/CLOUD_STORAGE.md`; add Kaggle secret **GDRIVE_TOKEN_JSON**. No credential belongs in the notebook. Remote outages are recorded while local checkpoints remain recoverable.'), code('''from rgcbm.cloud import DriveStore
REQUIRE_DRIVE = True
if REQUIRE_DRIVE:
    DRIVE = DriveStore()
    DRIVE.ensure_dataset(ROOT, DATASET)
    DRIVE.restore_run(ROOT, RUN_NAME)
    SYNC_CALLBACK = DRIVE.sync_run
    DRIVE.restore_assets(ROOT, include_llm=LLM_BACKEND == 'local_qwen')
else:
    DRIVE = None
    print('Drive disabled explicitly: persist /kaggle/working using Kaggle Save Version.')
''')]
    cells += [code('''from rgcbm.data import prepare_data, build_manifest, leakage_audit
from rgcbm.core import Config
import subprocess
if ONLINE:
    command = [sys.executable, str(CODE_ROOT / 'prepare_assets.py'), '--root', str(ROOT), '--datasets', DATASET]
    if LLM_BACKEND == 'local_qwen': command.append('--llm')
    # Embedded helper imports its sibling rgcbm package and prepares authoritative assets.
    subprocess.check_call(command)
else:
    prepare_data(ROOT, DATASET, online=False)
manifest = build_manifest(ROOT, DATASET, SEED)
audit = leakage_audit(ROOT, manifest)
print('Classes:', len(manifest['classes']), 'split counts:', manifest['counts'])
print('Exact cross-split duplicates:', len(audit['cross_split_exact_duplicates']))
assert not audit['cross_split_exact_duplicates'], 'Resolve leakage before estimating research results.'
''')]
    if mode == 'Colab':
        cells += [code('''archive = ROOT / 'data' / f'{DATASET}.zip'
if not (DRIVE_ROOT / 'datasets' / archive.name).exists():
    shutil.copy2(archive, DRIVE_ROOT / 'datasets' / archive.name)
for filename in ['efficientnet_b1_imagenet.pth', 'backbone_manifest.json']:
    shutil.copy2(ROOT / 'assets' / filename, DRIVE_ROOT / 'assets' / filename)
if LLM_BACKEND == 'local_qwen':
    shutil.copytree(ROOT / 'assets' / 'qwen2.5-1.5b-instruct', DRIVE_ROOT / 'assets' / 'qwen2.5-1.5b-instruct', dirs_exist_ok=True)
    shutil.copy2(ROOT / 'assets' / 'llm_manifest.json', DRIVE_ROOT / 'assets' / 'llm_manifest.json')
''')]
    elif mode == 'Kaggle':
        cells += [code('''if DRIVE is not None:
    DRIVE.sync_assets(ROOT, include_llm=LLM_BACKEND == 'local_qwen')
''')]
    cells += [markdown('GPU check: the Windows Blackwell wheelhouse uses PyTorch 2.7.1 + CUDA 12.8. Driver compatibility must be checked on that actual machine. This development machine had only a CPU.'), code('''import torch
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))
else:
    print('CPU mode. Set SMOKE_TEST=True for validation, or enable the intended NVIDIA GPU.')
    if not SMOKE_TEST:
        raise RuntimeError('Enable a GPU for full training, or set SMOKE_TEST=True for CPU validation.')
torch.set_num_threads(min(4, os.cpu_count() or 1))
cfg = Config(dataset=DATASET, seed=SEED, concepts=NUM_CONCEPTS, batch_size=BATCH_SIZE,
             head_lr=HEAD_LR, finetune_lr=FINETUNE_BACKBONE_LR,
             guidance_mode='individual_descriptors_v4', require_guidance_exposure=True,
             finetune_head_lr=FINETUNE_HEAD_LR, concept_normalization=CONCEPT_NORMALIZATION,
             checkpoint_every_epochs=CHECKPOINT_EVERY_EPOCHS,
             keep_epoch_checkpoints=KEEP_EPOCH_CHECKPOINTS,
             head_epochs=1 if SMOKE_TEST else HEAD_EPOCHS,
             finetune_epochs=1 if SMOKE_TEST else FINETUNE_EPOCHS,
             max_train_batches=2 if SMOKE_TEST else None,
             max_validation_batches=2 if SMOKE_TEST else None,
             checkpoint_every_batches=1 if SMOKE_TEST else 50)
print(cfg.to_dict())
'''), markdown('Train or resume. Batch progress bars show loss, accuracy and ETA. last.pt is saved each epoch and every 50 batches; full periodic snapshots are saved every CHECKPOINT_EVERY_EPOCHS epochs and at stage end. live_learning_curves.png, history.csv and training_status.json update during training. New guided_v4 runs use separate names; old checkpoints remain intact. Model selection and scheduler updates use validation only.'), code('''from rgcbm.train import fit
from rgcbm.data import atomic_json
model, RUN_DIR, status = fit(ROOT, manifest, cfg, device=DEVICE, sync_callback=SYNC_CALLBACK, run_name=RUN_NAME)
atomic_json(RUN_DIR / 'dataset_manifest.json', manifest)
from rgcbm.environment import runtime_manifest
atomic_json(RUN_DIR / 'environment.json', runtime_manifest())
assert status['complete']
print('Training complete. Best validation model restored:', RUN_DIR)
'''), markdown('Audit each guided channel against its configured descriptor and name it for the dataset. A positive association needs n>=50, |r|>=0.10, a bootstrap interval excluding zero and BH-FDR q<0.05. Guidance exposure is recorded; weak/conflicting/unexposed targets remain labeled. Unguided names are post-hoc associations. Freeze names before the official test evaluation. CUB boxes and Pet trimaps enter evaluation only, never training or cropping. Smoke mode intentionally evaluates only two test batches and cannot supply paper results.'), code('''from rgcbm.evaluate import ground_concepts, evaluate
expected_pipeline = {'complete': True,
                     'fingerprint': json.loads((RUN_DIR / 'configuration.json').read_text())['fingerprint'],
                     'llm_backend': LLM_BACKEND, 'protocol': 'guided_v4'}
PIPELINE_CACHED = (RUN_DIR / 'pipeline_status.json').is_file() and json.loads((RUN_DIR / 'pipeline_status.json').read_text()) == expected_pipeline
if PIPELINE_CACHED:
    grounding = json.loads((RUN_DIR / 'grounding.json').read_text())['concepts']
    metrics = json.loads((RUN_DIR / 'metrics.json').read_text())
    print('Using previously completed evaluation for this exact run.')
else:
    grounding = ground_concepts(model, ROOT, manifest, RUN_DIR, max_batches=2 if SMOKE_TEST else None)
    metrics = evaluate(model, ROOT, manifest, RUN_DIR, grounding, max_batches=2 if SMOKE_TEST else None)
import pandas as pd
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 90):
    display(pd.DataFrame(grounding)[['concept_name','training_group','intended_descriptor','grounding_status','mean','ci_low','ci_high','fdr_q','selected_model_guidance_updates']])
print('Measured domain names and intended targets are saved in named_concept_dictionary.csv; weak targets are not verified findings.')
display(pd.DataFrame([metrics]).T)
'''), code('''from rgcbm.visualize import visualize_run
from IPython.display import display, Image as DisplayImage
FIGURES = RUN_DIR / 'figures' if PIPELINE_CACHED else visualize_run(model, ROOT, manifest, RUN_DIR, grounding, examples=3,
                        prototype_batches=2 if SMOKE_TEST else None)
for name in ['guidance_alignment_matrix.png', 'case_00_explanation_panel.png', 'case_00_guidance_comparison.png', 'prototype_01.png', 'learning_curves.png', 'confusion_matrix.png', 'grounding_uncertainty.png',
             'case_00_priors.png', 'case_00_contributions.png', 'case_00_concept_maps.png', 'prototype_patches.png']:
    display(DisplayImage(filename=str(FIGURES / name)))
for index in [cfg.concepts//4, 2*cfg.concepts//4, 3*cfg.concepts//4]:
    display(DisplayImage(filename=str(FIGURES / f'prototype_{index+1:02d}.png')))
print('All concept prototype plates and explanation panels: 300 dpi PNG + PDF/SVG in', FIGURES)
print('Figure captions, selection policy and limitations:', RUN_DIR / 'PAPER_FIGURE_CAPTIONS.md')
'''), markdown('Text explanations consume exact model evidence only. The deterministic template always works offline. Local Qwen loads included files with `local_files_only=True`; Gemini requires secure runtime credentials. Structure/sign checks cannot prove that generated prose is hallucination-free.'), code('''from rgcbm.reports import generate_reports
if DEVICE == 'cuda': torch.cuda.empty_cache()
reports = generate_reports(RUN_DIR, ROOT / 'assets', backend=LLM_BACKEND, max_cases=3, llm_device=LLM_DEVICE)
for report in reports:
    print(report['result']['explanation'])
    print('Backend:', report['backend_used'], 'fallback:', report['fallback_reason'])
if SYNC_CALLBACK is not None: SYNC_CALLBACK(RUN_DIR)
print('All saved results:', RUN_DIR)
'''), markdown('''## Research run checklist

Repeat complete experiments with seeds 42/43/44 and report mean ± standard deviation. Re-run
baselines on the same full splits, backbone/pretraining, compute budget and selection rule.
Run guidance/stability/diversity ablations and K=16/32/64 using new run names. Existing manuscript
numbers must not be reused for this implementation. Independent attribute/part or human studies
are needed before describing these low-level descriptors as high-level semantic concepts.
''')]
    cells += [markdown('''## Concept evidence cards: names, uncertainty and domain limits

These cards show each frozen descriptor name, validation confidence interval, held-out association,
classifier use and domain limitations. Weak concepts remain weak. Review several prototype patches
and held-out maps before using a higher-level label; a class weight cannot name a bird part or diagnosis.
'''), code('''from rgcbm.journal import concept_cards, DOMAIN_SCOPE
print(DOMAIN_SCOPE[DATASET])
concept_card_table = concept_cards(manifest, RUN_DIR, grounding)
display(concept_card_table[['name', 'group', 'guidance_status', 'intended_descriptor', 'weak_grounding',
                           'validation_correlation', 'validation_image_bootstrap_95ci']])
print('Saved concept_cards.json and concept_cards.md')
'''), markdown('''## Reliability and concept redundancy

Reliability plots expose confidence errors; confidence is not correctness. Concept-correlation
plots and effective rank identify redundancy/collapse. Blank entries mean insufficient variation.
'''), code('''from rgcbm.journal import reliability_and_redundancy
reliability_table = reliability_and_redundancy(RUN_DIR)
display(reliability_table)
for filename in ['reliability_diagram.png', 'concept_redundancy.png']:
    if (FIGURES / filename).exists(): display(DisplayImage(filename=str(FIGURES / filename)))
'''), markdown('''## Manual latent-concept intervention

Edit the dictionary and rerun. This changes the concept vector, not the physical image, anatomy,
pet appearance or tumor. Exact logit changes are checked. Compare all class probabilities;
margin-supporting concepts need not increase probability relative to every other class.
'''), code('''from rgcbm.data import ImageDataset
from rgcbm.journal import intervention
example_dataset = ImageDataset(ROOT, manifest, 'test', cfg)
EXAMPLE_INDEX = 0
EDITED_CONCEPTS = {0: 0.0}  # ZERO-BASED concept index -> activation in [0,1].
# For standardized scores, 0.5 is the estimated training-mean response; 0 is an
# extreme latent clamp, not evidence that a visible object part has disappeared.
example_image, _, _ = example_dataset[EXAMPLE_INDEX]
edited_table = intervention(model, example_image, grounding, manifest['classes'], EDITED_CONCEPTS)
edited_table.to_csv(RUN_DIR / 'manual_intervention.csv', index=False)
display(edited_table.head(10))
import matplotlib.pyplot as plt
top = edited_table.head(5).set_index('class')
ax = top[['original_probability', 'edited_probability']].plot.barh(figsize=(8,4))
ax.set_title('Latent intervention; image and classifier weights unchanged')
ax.set_xlabel('Uncalibrated model probability')
plt.tight_layout(); plt.savefig(FIGURES / 'manual_intervention.png', dpi=180); plt.show()
'''), markdown('''## Matched concept-deletion curves

Compare positive margin-supporting concepts with random edits on identical eligible images per
removal count. Tables include sample counts and paired bootstrap intervals. This measures model
dependence, not causal image explanation.
'''), code('''from rgcbm.journal import deletion_curves
deletion_table = deletion_curves(model, RUN_DIR, max_k=4)
display(deletion_table)
if (FIGURES / 'deletion_curves.png').exists():
    display(DisplayImage(filename=str(FIGURES / 'deletion_curves.png')))
'''), markdown('''## Optional CUB semantic-attribute audit

This separately disclosed **annotation-assisted, post-training** audit uses original CUB attributes
with certainty >=3. A provisional name requires validation n>=50, |r|>=0.2 and a 312×K multiple-
comparison correction. Its frozen attribute is tested with held-out AUC. Test never renames concepts.
No concept labels train the model. An attribute association does not establish spatial part identity.
If evidence is weak, preserve the descriptor/weak name instead of inventing semantics.
'''), code('''from rgcbm.journal import cub_attribute_audit
RUN_CUB_ATTRIBUTE_AUDIT = False
if RUN_CUB_ATTRIBUTE_AUDIT and DATASET == 'cub':
    attribute_table = cub_attribute_audit(model, ROOT, manifest, RUN_DIR,
                                         max_batches=2 if SMOKE_TEST else None)
    display(attribute_table)
    print('Validation-selected annotation-assisted associations saved separately.')
else:
    print('Optional CUB attribute audit is disabled or not applicable to this dataset.')
'''), markdown('''## Runtime/parameter profile and provenance

Measure on the actual GPU. Warmed classifier inference is separate from training and LLM latency.
Device/version, parameters and peak CUDA memory are saved; full seeds are needed for paper claims.
'''), code('''from rgcbm.journal import profile_inference
profile = profile_inference(model, example_image, RUN_DIR, repeats=20)
display(pd.DataFrame([profile]).T)
atomic_json(RUN_DIR / 'pipeline_status.json', {'complete': True,
            'fingerprint': json.loads((RUN_DIR / 'configuration.json').read_text())['fingerprint'],
            'llm_backend': LLM_BACKEND, 'protocol': 'guided_v4'})
if SYNC_CALLBACK is not None: SYNC_CALLBACK(RUN_DIR)
print('Journal-facing evidence and figures saved:', RUN_DIR)
''')]
    cells += [markdown('''## Automatic sequential dataset queue

With RUN_ALL_DATASETS=True, this cell resumes the completed first model, reuses its saved evidence,
then prepares/trains/evaluates each remaining dataset. Each receives independent checkpoints,
learning curves, concept cards, reports and journal figures. A summary CSV, status JSON and chart
are saved alongside the runs and synchronized to configured Drive storage. Restart with the same
configuration to resume. Dataset failures are recorded and the queue continues; inspect failed rows.
'''), code(f'''from rgcbm.suite import run_dataset_suite
def prepare_queued_dataset(dataset, run_name):
    if '{mode}' == 'Colab':
        target = ROOT / 'data' / f'{{dataset}}.zip'
        source = DRIVE_ROOT / 'datasets' / target.name
        if not target.exists() and source.exists(): shutil.copy2(source, target)
    elif '{mode}' == 'Kaggle' and DRIVE is not None:
        DRIVE.ensure_dataset(ROOT, dataset)
        DRIVE.restore_run(ROOT, run_name)
    if ONLINE:
        command = [sys.executable, str(CODE_ROOT / 'prepare_assets.py'), '--root', str(ROOT), '--datasets', dataset]
        if LLM_BACKEND == 'local_qwen': command.append('--llm')
        subprocess.check_call(command)
    else:
        prepare_data(ROOT, dataset, online=False)
    if '{mode}' == 'Colab':
        archive = ROOT / 'data' / f'{{dataset}}.zip'
        if not (DRIVE_ROOT / 'datasets' / archive.name).exists():
            shutil.copy2(archive, DRIVE_ROOT / 'datasets' / archive.name)
    elif '{mode}' == 'Kaggle' and DRIVE is not None:
        DRIVE.sync_assets(ROOT, include_llm=LLM_BACKEND == 'local_qwen')

if RUN_ALL_DATASETS:
    model.cpu()  # Keep the preview model available, release its GPU allocation for the queue.
    dataset_summary = run_dataset_suite(ROOT, DATASETS, cfg, run_tag=RUN_TAG,
                         device=DEVICE, capacity_map=CONCEPTS_BY_DATASET, prepare_callback=prepare_queued_dataset,
                         sync_callback=SYNC_CALLBACK, llm_backend=LLM_BACKEND,
                         llm_device=LLM_DEVICE, continue_on_error=True)
    display(dataset_summary)
    suite_dir = ROOT / 'results' / (f'dataset_suite_{{RUN_TAG}}_K{{cfg.concepts}}_seed{{cfg.seed}}' + ('_smoke' if SMOKE_TEST else ''))
    print('Combined queue status, CSV and chart:', suite_dir)
    if (suite_dir / 'dataset_accuracy.png').exists():
        display(DisplayImage(filename=str(suite_dir / 'dataset_accuracy.png')))
    if not (dataset_summary.status == 'complete').all():
        print('Some datasets failed. Inspect queue_status.json and dataset error files; successful models are retained.')
else:
    print('Single-dataset mode: only the detailed DATASET run was requested.')
''')]
    cells += [markdown('''## Ablation study — optional experiment sweep

Each variant receives a separate checkpoint and run directory. The dataset, pretrained backbone,
selection policy and training budget are held fixed; each requested seed rebuilds its reproducible
validation split. The no-fine-tuning variant intentionally removes stage two. Test scores are final
measurements, not tuning feedback. Use at least three complete seeds for mean/std claims. Smoke mode
produces software checks only. Set RUN_ABLATIONS=True deliberately to start this potentially long sweep.
'''), code('''from rgcbm.ablations import run_ablation_suite, VARIANTS
RUN_ABLATIONS = False
ABLATION_VARIANTS = ['full', 'unguided', 'no_edge', 'no_texture', 'no_region',
                     'no_diversity', 'no_decorrelation', 'no_stability',
                     'no_sparsity', 'no_finetune', 'no_standardization', 'legacy_group_guidance', 'K16', 'K32', 'K64']
ABLATION_SEEDS = [42, 43, 44]  # Use [42] only for exploratory checks, not uncertainty claims.
ABLATION_DATASETS = DATASETS if RUN_ALL_DATASETS else [DATASET]
if RUN_ABLATIONS:
    from dataclasses import replace
    for ablation_dataset in ABLATION_DATASETS:
        ablation_cfg = replace(cfg, dataset=ablation_dataset, concepts=CONCEPTS_BY_DATASET[ablation_dataset])
        ablation_table = run_ablation_suite(ROOT, ablation_cfg, variants=ABLATION_VARIANTS,
                                       seeds=ABLATION_SEEDS, device=DEVICE,
                                       sync_callback=SYNC_CALLBACK)
        display(ablation_table[['variant', 'seed', 'concepts', 'software_smoke_only',
                            'accuracy', 'macro_f1', 'stability_mae']])
        profile_name = 'guided_v4' if cfg.guidance_mode == 'individual_descriptors_v4' else 'calibrated_v3' if cfg.concept_normalization == 'running_standardize' else 'legacy'
        summary_path = ROOT / 'results' / f'{ablation_dataset}_ablation_summary_{profile_name}'
        print('Per-run and mean/std CSVs saved:', summary_path)
else:
    print('Ablation code is ready. Set RUN_ABLATIONS=True to execute the selected variants.')
''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'RG-CBM Anaconda offline GPU (Python 3.11)' if mode == 'Offline' else 'Python 3', 'language': 'python', 'name': 'rgcbm-offline' if mode == 'Offline' else 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}})
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / 'notebooks' / f'RGCBM_{mode}.ipynb')
    print(mode, len(cells), 'cells generated')
