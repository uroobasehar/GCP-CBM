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
   OAuth authorization and the `drive.file` scope. It never prints a token.
3. Put the resulting authorized-user JSON securely in Kaggle secret **GDRIVE_TOKEN_JSON**. Enable
   that secret for this notebook. Do not paste credentials into notebook cells, commit them, or place
   them in the distributed ZIP. Restrict/revoke the OAuth credential when no longer needed.
4. Run the notebook. It creates its own `RGCBM_MultiDataset` Drive folder, stores source dataset ZIPs,
   pretrained backbone and run checkpoints, and restores training state after a new Kaggle session.
   The narrow scope accesses files created by this application, not arbitrary existing Drive files.

Checkpoint uploads are resumable. Remote failures record `remote_sync_error.json` and preserve
local `/kaggle/working/rgcbm/results` checkpoints. Inspect that file and rerun sync before ending the
session; a local save is not proof of successful remote storage. Also use Kaggle **Save Version** for
the working outputs. `REQUIRE_DRIVE=False` explicitly permits Kaggle-only storage, but the default
requires your Drive credential as requested.

Colab and Kaggle notebooks were generated and syntax-checked; the shared local pipeline and offline
notebook were executed here. Live cloud runtimes, interactive OAuth and your Drive account are not
available in this workspace, so those integrations require their first authenticated run on the
platform. The package includes no credentials. Optional Gemini keys belong in platform secrets or
environment variables; local Qwen and the deterministic template need no API key.
