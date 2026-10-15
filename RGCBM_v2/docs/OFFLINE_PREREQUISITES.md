# Before the Windows machine becomes permanently offline

The Python wheelhouse includes the complete 188-package Windows dependency closure, the actual
Python 3.11 conda runtime packages, Microsoft VC++ x64 runtime installer, Jupyter frontend/kernel,
three dataset archives, backbone and local Qwen files. No CUDA toolkit, Visual Studio compiler,
Git, Graphviz, LaTeX, paid LLM API, Intel NPU SDK or extra concept annotation downloads are needed
to run the notebooks. Matplotlib fonts and Jupyter JavaScript are included in their wheels.

The dependency conflict review pinned PyArrow to 19.0.1 for NumPy 1.26.4 compatibility and added
installed-lock and native-library checks. Use the latest complete bundle; see
[the dependency review](OFFLINE_DEPENDENCY_REVIEW.md) for the confirmed failure and replacement tests.

The GPU is an **RTX PRO 4500 Blackwell with 32 GB dedicated VRAM**. Windows shared memory is
system RAM, not additional equivalent CUDA VRAM. The Intel NPU is not used by this implementation.
Choose batch size using dedicated VRAM; the notebook starts at 24 and the acceptance test uses 2.

## System prerequisites that a Python ZIP cannot install automatically

| Item | Required action |
|---|---|
| 64-bit Windows 11 and Anaconda3 | Already requested/installed by the user. Open an initialized 64-bit Anaconda PowerShell Prompt. |
| NVIDIA system driver | Install a current standalone NVIDIA RTX Enterprise/RTX PRO Blackwell driver and reboot. CUDA 12.8's Windows release driver minimum is 570.65; support for the exact GPU must also be present. A newer supported production driver is preferable. |
| Offline driver fallback | Carry the complete NVIDIA- or Dell-signed driver installer for the exact GPU/OS, not NVIDIA App or a web downloader. The installed driver version cannot be inferred from GPU model or Dell branding. |
| Local browser | Windows Edge, Chrome or Firefox must already work. Jupyter runs locally without internet. |
| Windows trust/certificates | Verify the bundled Microsoft installer signature and any driver installer while disconnected. Resolve trust/signature errors before final acceptance; no verification bypass is supplied. |
| Storage and permissions | Use a short writable local NTFS/exFAT folder, preferably D:\RGCBM. Budget at least 60 GB overall and 20 GB free after extraction. Installer elevation and a reboot may be needed. |
| Transfer media | FAT32 cannot hold the assembled ZIP or a >4 GB file. Use NTFS/exFAT, or transfer the individual <2 GB archive parts and assemble them on the local NTFS drive. |
| Power and recovery | Disable sleep during training, keep adequate free space, and back up results/checkpoints to a second local drive. |

The NVIDIA download hosts returned policy-level HTTP 403 in the preparation environment. Therefore
**a standalone NVIDIA driver binary is not included in this ZIP**. Adding a hostname to the saved
configuration did not make that host reachable in the running instance. This remains a separately
collected system prerequisite, not an additional missing pip package. On an online Windows computer,
download the full installer from the official NVIDIA RTX Enterprise driver selector or the Dell
support page for the machine. `offline/collect_driver.ps1 -Installer <downloaded.exe>` checks the
NVIDIA Authenticode signature, copies it to `offline/installers/nvidia_driver.exe`, and records
its SHA-256 in `offline/driver_manifest.json`. Carry that companion alongside the immutable ZIP.
For Dell-wrapped installers, verify the Dell signature using Windows and keep that OEM package
separately; the NVIDIA collector intentionally accepts only NVIDIA-signed packages.

## Acceptance sequence on the actual machine

1. Extract the complete bundle into a new folder and run
   `powershell -ExecutionPolicy Bypass -File .\offline\preflight_windows.ps1`.
   It records actual OS, RAM, filesystem, disk space, driver and installer-signature results.
2. Run `powershell -ExecutionPolicy Bypass -File .\offline\setup_windows.ps1`.
   Conda is offline with the classic solver; pip has no index. Local checksums/signatures remain enabled.
3. Disconnect networking and run
   `.\conda-env\python.exe .\offline\validate_runtime.py --require-gpu --dataset cub`.
   Repeat for `pets` and `brain_mri`. This checks the requested libraries, TensorFlow CPU math,
   real GPU training in both stages, explanation figures, reports, checkpoint reload and the
   registered Jupyter kernel. It also verifies installed versions/locations against the Windows
   lock and probes compiled libraries in both TensorFlow/PyTorch import orders. Results are in
   `runtime/offline_validation.json`, `runtime/native_library_validation.json` and `results/`.
   Optionally add `--llm`; a rejected Qwen response and deterministic fallback are recorded explicitly.
4. Run `offline\launch_jupyter.cmd`, open `notebooks/RGCBM_Offline.ipynb`, select the RG-CBM kernel,
   and execute once with `SMOKE_TEST=True`. Stop/restart a training smoke run and verify resume.
5. Keep `preflight_report.json`, `machine_validation.json` and `runtime/offline_validation.json`.
   Do not approve permanent disconnection until these checks pass on the actual Windows GPU machine.

Linux CPU validation cannot prove Windows DLL loading, certificate trust, administrator access,
driver compatibility or GPU execution. These checks deliberately report those boundaries.
