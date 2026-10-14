from pathlib import Path
import json
import nbformat

ROOT = Path(__file__).resolve().parent
payload = {p.relative_to(ROOT).as_posix(): p.read_text(encoding='utf-8') for p in sorted((ROOT / 'rgcbm').glob('*.py'))}
payload['prepare_assets.py'] = (ROOT / 'prepare_assets.py').read_text()
requirements = '''tensorflow==2.19.1 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 matplotlib==3.9.2 Pillow==11.0.0 medmnist==3.0.2 kagglehub==0.3.14 tensorflow-datasets==4.9.9 tensorflow-metadata==1.16.1 protobuf==5.29.6 googleapis-common-protos==1.69.2 proto-plus==1.26.1 google-api-core==2.24.2 opencv-python-headless==4.10.0.84 tqdm==4.67.1 PyYAML==6.0.2 transformers==4.51.3 accelerate==1.6.0 huggingface-hub==0.30.2 safetensors==0.5.3 google-api-python-client==2.166.0 google-auth-oauthlib==1.2.2 google-genai==1.11.0'''

def markdown(text): return nbformat.v4.new_markdown_cell(text)
def code(text): return nbformat.v4.new_code_cell(text)

config_cell = '''# Select one dataset per run; use all three for the complete research evaluation.
DATASET = 'cub'                 # 'cub', 'pets', 'brain_mri'
SEED = 42                      # Repeat independent full runs with 42, 43, 44.
NUM_CONCEPTS = 16               # Capacity ablation: 16, 32, 64.
BATCH_SIZE = 24                 # 32 GB GPU: start here; lower if CUDA memory is exhausted.
HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 20
SMOKE_TEST = False              # True executes two real training batches per stage; not paper results.
LLM_BACKEND = 'template'        # 'local_qwen' (offline-capable) or 'gemini' (online key required).
LLM_DEVICE = 'cpu'              # 'cuda' also works; release training tensors before loading Qwen.
RUN_NAME = f'{DATASET}_K{NUM_CONCEPTS}_seed{SEED}' + ('_smoke' if SMOKE_TEST else '')
print('Dataset:', DATASET, 'run:', RUN_NAME)
'''

for mode in ['Offline', 'Colab', 'Kaggle']:
    cells = [markdown(f'''# RG-CBM: descriptor-grounded visual concept bottleneck — {mode}

CUB-200-2011, Oxford-IIIT Pet and MRI share the original architecture: EfficientNet-B1 →
128-dimensional projection → cosine-matched prototypes → 16 pooled concepts → sparse linear classifier.
The three guided groups model edge, texture and image-derived region priors; the fourth is unguided.
Names are selected from measured validation associations, without asserting tumor anatomy or bird parts.

Use the supplied README and research methodology. This notebook does not contain fabricated research
results. Full training and independent held-out evaluation must run on your GPU. Resume by rerunning
the same configuration; incompatible checkpoints are rejected. Change RUN_NAME for a new experiment.
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
        cells += [markdown('Internet access is required for initial preparation. Enable a GPU in runtime settings. Dependencies are pinned; restart the kernel/runtime if an already imported package version changed.'), code(f'''import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *{requirements.split()!r}])
import torch
if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 7):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1', '--index-url', 'https://download.pytorch.org/whl/cu128'])
    raise RuntimeError('PyTorch changed: restart this runtime, then rerun the notebook.')
import torchvision
from pathlib import Path
import os, shutil
ROOT = Path('/content/rgcbm' if '{mode}' == 'Colab' else '/kaggle/working/rgcbm')
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
'''), code('''from rgcbm.environment import check_requested_libraries
display(check_requested_libraries())
''')]
    if mode == 'Colab':
        cells += [markdown('Google Drive stores source archives, backbone assets, run manifests, progress, checkpoints and figures. Training images are extracted to local runtime storage for speed. Google Drive authentication is interactive and cannot be pretested here.'), code('''from google.colab import drive
drive.mount('/content/drive', force_remount=False)
DRIVE_ROOT = Path('/content/drive/MyDrive/RGCBM_MultiDataset')
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
''')]
    elif mode == 'Kaggle':
        cells += [markdown('Kaggle cannot mount Google Drive through `google.colab.drive`. Use the supported Drive API with a securely stored user OAuth token. Follow `docs/CLOUD_STORAGE.md`; add Kaggle secret **GDRIVE_TOKEN_JSON**. No credential belongs in the notebook. Remote outages are recorded while local checkpoints remain recoverable.'), code('''from rgcbm.cloud import DriveStore
REQUIRE_DRIVE = True
if REQUIRE_DRIVE:
    DRIVE = DriveStore()
    DRIVE.ensure_dataset(ROOT, DATASET)
    DRIVE.restore_run(ROOT, RUN_NAME)
    SYNC_CALLBACK = DRIVE.sync_run
    (ROOT / 'assets').mkdir(exist_ok=True)
    weight = ROOT / 'assets' / 'efficientnet_b1_imagenet.pth'
    if not weight.exists():
        DRIVE.download(weight.name, weight, 'assets')
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
''')]
    elif mode == 'Kaggle':
        cells += [code('''if DRIVE is not None:
    for filename in ['efficientnet_b1_imagenet.pth', 'backbone_manifest.json']:
        DRIVE.upload(ROOT / 'assets' / filename, 'assets')
''')]
    cells += [markdown('GPU check: the Windows Blackwell wheelhouse uses PyTorch 2.7.1 + CUDA 12.8. Driver compatibility must be checked on that actual machine. This development machine had only a CPU.'), code('''import torch
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))
else:
    print('CPU mode. Set SMOKE_TEST=True for validation, or enable the intended NVIDIA GPU.')
torch.set_num_threads(min(4, os.cpu_count() or 1))
cfg = Config(dataset=DATASET, seed=SEED, concepts=NUM_CONCEPTS, batch_size=BATCH_SIZE,
             head_epochs=1 if SMOKE_TEST else HEAD_EPOCHS,
             finetune_epochs=1 if SMOKE_TEST else FINETUNE_EPOCHS,
             max_train_batches=2 if SMOKE_TEST else None,
             max_validation_batches=2 if SMOKE_TEST else None,
             checkpoint_every_batches=1 if SMOKE_TEST else 50)
print(cfg.to_dict())
'''), markdown('Train or resume. Stop/restart preserves progress at each epoch and every 50 batches; at most the batches since the last committed checkpoint need repeating. Model selection and scheduler updates use validation only.'), code('''from rgcbm.train import fit
from rgcbm.data import atomic_json
model, RUN_DIR, status = fit(ROOT, manifest, cfg, device=DEVICE, sync_callback=SYNC_CALLBACK, run_name=RUN_NAME)
atomic_json(RUN_DIR / 'dataset_manifest.json', manifest)
assert status['complete']
print('Training complete. Best validation model restored:', RUN_DIR)
'''), markdown('Ground names on validation; evaluate the frozen names and classifier on the official test set. CUB boxes and Pet trimaps enter evaluation only, never training or cropping. Smoke mode intentionally evaluates only two test batches and cannot supply paper results.'), code('''from rgcbm.evaluate import ground_concepts, evaluate
grounding = ground_concepts(model, ROOT, manifest, RUN_DIR, max_batches=2 if SMOKE_TEST else None)
metrics = evaluate(model, ROOT, manifest, RUN_DIR, grounding, max_batches=2 if SMOKE_TEST else None)
import pandas as pd
display(pd.DataFrame(grounding).drop(columns=['all_descriptor_statistics']))
display(pd.DataFrame([metrics]).T)
'''), code('''from rgcbm.visualize import visualize_run
from IPython.display import display, Image as DisplayImage
FIGURES = visualize_run(model, ROOT, manifest, RUN_DIR, grounding, examples=3,
                        prototype_batches=2 if SMOKE_TEST else None)
for name in ['learning_curves.png', 'confusion_matrix.png', 'grounding_uncertainty.png',
             'case_00_priors.png', 'case_00_contributions.png', 'case_00_concept_maps.png', 'prototype_patches.png']:
    display(DisplayImage(filename=str(FIGURES / name)))
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
display(concept_card_table[['id', 'name', 'group', 'weak_grounding',
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
if SYNC_CALLBACK is not None: SYNC_CALLBACK(RUN_DIR)
print('Journal-facing evidence and figures saved:', RUN_DIR)
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
                     'no_sparsity', 'no_finetune', 'K32', 'K64']
ABLATION_SEEDS = [42, 43, 44]  # Use [42] only for exploratory checks, not uncertainty claims.
if RUN_ABLATIONS:
    ablation_table = run_ablation_suite(ROOT, cfg, variants=ABLATION_VARIANTS,
                                       seeds=ABLATION_SEEDS, device=DEVICE,
                                       sync_callback=SYNC_CALLBACK)
    display(ablation_table[['variant', 'seed', 'concepts', 'software_smoke_only',
                            'accuracy', 'macro_f1', 'stability_mae']])
    summary_path = ROOT / 'results' / f'{DATASET}_ablation_summary'
    print('Per-run and mean/std CSVs saved:', summary_path)
else:
    print('Ablation code is ready. Set RUN_ABLATIONS=True to execute the selected variants.')
''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'RG-CBM Anaconda offline GPU (Python 3.11)' if mode == 'Offline' else 'Python 3', 'language': 'python', 'name': 'rgcbm-offline' if mode == 'Offline' else 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}})
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / 'notebooks' / f'RGCBM_{mode}.ipynb')
    print(mode, len(cells), 'cells generated')
