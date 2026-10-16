# Offline dependency conflict review — 15 October 2026

## Confirmed conflict and fix

The original Windows lock combined **NumPy 1.26.4 with PyArrow 26.0.0**. PyArrow's
`Requires-Dist` metadata omits its NumPy requirement, so dependency resolution and `pip check`
accepted the pair. Actual import failed: `ImportError: pyarrow requires NumPy 2.0 or newer,
found 1.26.4`. The bundled Windows wheel's `pyarrow/lib.pyx` contains the same explicit check.
TensorFlow Datasets depends on PyArrow, so this could also break dataset utilities.

The offline and cloud requirements now pin **PyArrow 19.0.1**. NumPy 1.26.4 and TensorFlow
2.19.1 remain unchanged in the Windows lock. The compatible Windows wheel replaces version 26
in the extractable offline bundle; all 188 wheel hashes and the complete dependency closure
were regenerated. Cloud Python 3.13 continues using TensorFlow 2.20.0 and supports NumPy 2.1.3.

Initial failure evidence is preserved in `offline_dependency_initial_failure.json`.
The passing replacement tests are in `offline_native_library_validation_linux.json` and
`python313_native_library_validation.json`. These are Linux CPU tests, not Windows execution.

## Additional checks and protections

- Rechecked 188 Windows wheel hashes, ZIP integrity, Python requirements and target ABI tags,
  387 active dependency requirements, and 20 conda package hashes with 56 dependency/build constraints.
- No overlapping installation paths were found. All three repeated DLL basenames
  (`asmjit.dll`, `fbgemm.dll`, `cudart64_12.dll`) have identical contents across their copies.
  This does not prove native Windows loading against the machine's installed system libraries.
- Corrected the lock builder's ABI filter. It previously permitted older CPython-specific
  ABIs; it now accepts only CPython 3.11-compatible Windows x64/stable-ABI/pure-Python tags.
  Existing selected wheels were compatible. Regression checks reject wrong Python, OS and architecture.
- Setup and launch now disable user-site packages and inherited Python path/home overrides.
  Setup clears pip target/prefix overrides and ignores external pip configuration. The Jupyter
  launcher uses project-local configuration/data and prioritizes the bundled environment's kernels.
- Setup checks all installed distribution versions and locations against the lock. Acceptance
  adds isolated subprocess probes for both TensorFlow/PyTorch import orders, compiled torchvision
  NMS, SciPy, scikit-learn, Pillow/OpenCV, pandas, h5py, PyArrow, protobuf, Matplotlib and a tiny
  Transformers Qwen forward pass. Crashes or timeouts fail acceptance; no OpenMP error bypass is used.
- The cloud bootstrap detects an already loaded PyArrow version changed by pip and requests a
  kernel restart, preventing stale native modules from remaining loaded after replacement.

## Validation and limits

The replacement passed native CPU probes with Python 3.12/TensorFlow 2.19.1/NumPy 1.26.4 and
Python 3.13/TensorFlow 2.20.0/NumPy 2.1.3, with Internet sockets blocked by an inherited kernel
seccomp rule. `pip check` passed in both Linux environments. Both cloud notebooks were re-executed
on real CUB images after the change; managed pip/mount/OAuth/Drive operations use the documented
local test substitutions. The offline acceptance harness also passed real-image training,
checkpoint reload, explanation figures and its registered Jupyter kernel on Linux CPU.

Full native Windows installation, GPU/DLL loading, live Colab/Kaggle GPUs and real Drive
authorization remain untested because those machines/sessions are unavailable here. No
package-version changes can replace the NVIDIA system driver. Carry its signed standalone
installer separately; see `OFFLINE_PREREQUISITES.md`.

## Apply the update

Obtain the latest complete multipart bundle and its matching `bundle_manifest.json` from the same
Git revision; assemble with `Join-OfflineBundle.ps1`. Close Jupyter and its kernels before installing
or replacing native packages. Extract the updated distribution, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\setup_windows.ps1
.\conda-env\python.exe .\offline\audit_dependencies.py --installed
.\conda-env\python.exe .\offline\validate_runtime.py --require-gpu --dataset cub
```

Repeat acceptance for `pets` and `brain_mri` with networking disconnected on the Windows machine.
`audit_dependencies.py` works without a package index and writes `offline/installed_dependency_audit.json`;
acceptance also writes `runtime/native_library_validation.json` and dataset-specific validation reports.
Use the newly published cloud notebooks; previously saved Drive copies do not update themselves.

## Guidance V4 application update

The twelve descriptor maps, named publication figures and exposure-aware checkpoint selection use
existing locked PyTorch, SciPy, NumPy, pandas and Matplotlib packages. No extra runtime package or
network download is introduced. The same Windows Python 3.11 wheel/conda lock remains authoritative.
Local Qwen explicitly disables optional TensorFlow/Flax discovery before importing Transformers,
unless the caller already set those controls, to keep its PyTorch inference from initializing an
additional framework merely because TensorFlow was installed for the requested compatibility stack.
The local-only inference regression checks that TensorFlow was not imported by the verbalizer.
Current notebooks, source and methodology are included in the rebuilt offline archive; dependency
hashes, ZIP CRC and the immutable first four archive parts are checked during packaging.
