# Colab and Kaggle persistence

**Colab:** upload `notebooks/RGCBM_Colab.ipynb`, enable GPU, run cells in order and approve your own
Google Drive mount. `MyDrive/RGCBM_MultiDataset` stores dataset archives, backbone files, models,
progress, figures and reports. Images are extracted locally for training speed. The results directory
is linked to mounted Drive so checkpoints are written there directly. Restart with the same dataset,
seed and run name to resume. Switch datasets in the configuration cell for the other benchmarks.

**Kaggle:** upload `notebooks/RGCBM_Kaggle.ipynb`, enable Internet and a GPU. Colab's `drive.mount`
does not work on Kaggle. The notebook instead uses the Drive API with a user OAuth refresh token.

1. On an online computer with a browser, create/choose a Google Cloud OAuth **Desktop app** client
   with the Google Drive API enabled. Download its client JSON securely. If the consent screen is in
   testing, add your Google account as a test user; testing-mode tokens may expire and need reauthorization.
2. Install `google-auth-oauthlib`/`google-api-python-client`, then run
   `python authorize_drive.py path/to/private_oauth_client.json` locally. This uses supported loopback
   OAuth authorization. The default `drive.file` scope accesses files created by this OAuth app.
   To reuse a folder initially created through Colab or share the same project across platforms,
   instead run `python authorize_drive.py path/to/private_oauth_client.json --scope drive`. This
   grants broader Drive access; choose it deliberately and keep the token private. It never prints a token.
3. Put the resulting authorized-user JSON securely in Kaggle secret **GDRIVE_TOKEN_JSON**. Enable
   that secret for this notebook. Do not paste credentials into notebook cells, commit them, or place
   them in the distributed ZIP. Restrict/revoke the OAuth credential when no longer needed.
4. Run the notebook. It creates its own `RGCBM_MultiDataset` Drive folder, stores source dataset ZIPs,
   pretrained backbone and run checkpoints, and restores training state after a new Kaggle session.
   With the narrow scope, existing Colab-created files may be invisible. For a shared project,
   use the broader scope above and set `RGCBM_DRIVE_FOLDER_ID` to the project folder ID, or use the
   same uniquely named MyDrive root. Colab can select another mounted location with `RGCBM_DRIVE_ROOT`.
   A missing credential or insufficient scope must be resolved through your own Google authorization.

Checkpoint uploads are resumable. Remote failures record `remote_sync_error.json` and preserve
local `/kaggle/working/rgcbm/results` checkpoints. Inspect that file and rerun sync before ending the
session; a local save is not proof of successful remote storage. Also use Kaggle **Save Version** for
the working outputs. `REQUIRE_DRIVE=False` explicitly permits Kaggle-only storage, but the default
requires your Drive credential as requested.

Colab and Kaggle notebooks were executed end to end with real CUB images on a local Linux CPU
runtime and NumPy 2.1.3. Tests simulated pip calls, Colab mounting and Drive API authorization/transport;
they executed the embedded application code, preparation subprocess, both training stages, metrics,
figures and journal blocks. Installed root requirements were checked before the final tests.
Live cloud GPU runtimes, interactive OAuth and your personal Drive account are unavailable here;
the local simulations are not evidence that a live authenticated session has passed. The package includes no credentials. Optional Gemini keys belong in platform secrets or
environment variables; local Qwen and the deterministic template need no API key.


Cloud bootstrap accepts Python 3.10–3.12 for TensorFlow 2.19.1. It preserves a functional compatible
PyTorch/torchvision pair, exercises compiled operators and available GPU kernels, and repairs both
packages together when required. Repairs require restarting the kernel. Older supported cloud GPUs
use CUDA 12.6 repair wheels; detected Blackwell uses CUDA 12.8. A loaded package change also stops
with an explicit restart instruction instead of continuing with mixed NumPy/protobuf binaries.
Cloud NumPy supports 1.26 through 2.1; the Windows offline lock remains exactly 1.26.4.

Kaggle no longer overwrites a newer local checkpoint with a remote copy on rerun. Unchanged uploads
are skipped, and `last.pt` uploads after companion files. This is not a transactional Drive backup;
keep Kaggle Save Version as a second copy and inspect sync errors. Both notebooks restore and save
Qwen weights/tokenizers and manifests when requested. Existing complete, checksum-verified local
Qwen assets are reused before any model-hub request. Full training on CPU is blocked unless smoke
mode is selected, avoiding accidental multi-day CPU runs. Per-run `environment.json` records versions.
