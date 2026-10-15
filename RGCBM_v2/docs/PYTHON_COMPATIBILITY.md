# Colab Python 3.13 compatibility

The supplied Colab release notes dated 16 September 2026 report:
"Python upgraded from 3.12.13 to 3.13.15 (all images)."
The previous notebook's Python <=3.12 check rejected this runtime before installation.
TensorFlow 2.19.1 has no CPython 3.13 wheel, so removing that check alone cannot make its old pin work.

| Execution environment | TensorFlow requirement |
|---|---|
| Colab/Kaggle on Python 3.13 | 2.20.0 |
| Colab/Kaggle on Python 3.10–3.12 | 2.19.1 |
| Frozen Windows offline environment, Python 3.11 | 2.19.1, unchanged |

`cloud_requirements.txt` uses Python-version markers to select exactly one version. The bootstrap
accepts Python 3.13, and the cloud notebook's dependency table uses the same cloud profile. Other
scientific pins have compatible CPython 3.13 or stable-ABI wheels. The model/training implementation
uses PyTorch; this TensorFlow compatibility change does not alter the RG-CBM architecture or losses.

## Use the corrected notebook

Open the latest `notebooks/RGCBM_Colab.ipynb` from GitHub in a new Colab runtime and run cells from
the beginning. A notebook previously saved to Drive is a separate copy and does not automatically
update when GitHub changes. If setup reports that an already loaded library changed, restart the
session and rerun from the first cell. Do not mix old embedded code with the new notebook in one
kernel. Existing data/checkpoints saved to Drive remain available for restore.

If TensorFlow **2.19.1 itself** is essential for a cloud experiment, choose a past Colab runtime
that actually reports Python 3.12 using its runtime version selector, where available. A Python
3.13 kernel cannot install the missing 2.19.1 wheel by changing a pip flag.

## Validation

The fix was installed into a fresh CPython 3.13.5 Linux environment, including TensorFlow 2.20.0,
NumPy 2.1.3, PyTorch 2.7.1 CPU and torchvision 0.22.1 CPU. `pip check`, TensorFlow matrix math,
compiled torchvision operators, all cloud dependency checks and offline dataset registries passed.
Both complete cloud notebooks passed their 19 executable cells on real CUB images, including both
training stages, checkpoint saves and explanation outputs. Notebook pip calls and cloud-specific
mount/authentication/Drive transport were substituted after actual dependency installation.
Live Colab GPU and Google Drive authorization are not claimed as tested here.

See `python313_functional_validation.json` and `python313_cloud_notebook_validation.json` for evidence.
