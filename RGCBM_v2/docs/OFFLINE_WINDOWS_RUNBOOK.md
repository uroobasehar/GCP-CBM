# RG-CBM offline Windows 11 runbook

This is the exact installation sequence for the supplied `RGCBM_Offline.zip` on a
Windows 11 Dell workstation with Anaconda3 already installed. The examples use
`D:\\RGCBM`; a short local NTFS path is recommended. Do not install the project
inside OneDrive, a network share, FAT32 media, or an Anaconda base directory.

### Target machine profile

The supplied target details are recorded as **Windows 11 Pro, build 26200,
Dell Pro Max Tower T2FCT2250**, with an NVIDIA RTX PRO 4500 Blackwell. Before
downloading the driver, confirm the Dell Service Tag and the exact GPU name in
the Dell support page or NVIDIA Control Panel. Download the driver package
matching that Service Tag, Windows 11 x64, and the RTX PRO/Blackwell GPU; do
not substitute a consumer GeForce package.

## What is installed, and what is not

The archive contains the Windows Python 3.11 conda runtime, locked wheels,
Jupyter, datasets, model assets, notebooks, and the offline Qwen files. It does
not replace your Anaconda installation. Your Anaconda base Python 3.13.5 stays
unchanged; the project kernel is **RG-CBM Anaconda offline GPU (Python 3.11)**.

The archive cannot install a Windows NVIDIA driver. Before disconnecting the
computer, install a current NVIDIA RTX PRO Blackwell driver for the exact GPU
and reboot. `nvidia-smi` must work. The package supplies the CUDA runtime used
by PyTorch; a separate CUDA Toolkit, Visual Studio, Git, or Intel NPU SDK is not
required.

## A. Prepare the transfer while the computer is online

Run these commands in a normal PowerShell window on the online computer:

```powershell
git lfs install
git clone https://github.com/uroobasehar/GCP-CBM.git
cd GCP-CBM
powershell -ExecutionPolicy Bypass -File .\RGCBM_v2\offline_bundle\Join-OfflineBundle.ps1
```

The join command creates `RGCBM_Offline.zip` from `.001` through `.006`. Copy
that ZIP to an NTFS/exFAT USB drive. Also copy the complete standalone NVIDIA
driver installer for the RTX PRO 4500 Blackwell. Do not use an NVIDIA App web
installer: it cannot work after the computer is disconnected.

## B. Install the NVIDIA driver (Administrator)

1. Log in with an account that can install device drivers.
2. Right-click the NVIDIA/Dell driver installer and choose **Run as administrator**.
3. Choose the normal/clean installation offered by the signed installer.
4. Reboot when it asks. Do not continue until Windows has restarted.
5. Open **Anaconda PowerShell Prompt** normally (not Administrator) and run:

```powershell
nvidia-smi
```

Confirm that the GPU name, driver version, and approximately 32 GB dedicated
memory are shown. If `nvidia-smi` is missing or fails, stop here and fix the
driver. The Python setup cannot correct a driver problem.

## C. Extract the project (normal user)

Create `D:\\RGCBM`, extract `RGCBM_Offline.zip` there, and confirm these paths
exist:

```text
D:\\RGCBM\\offline\\setup_windows.ps1
D:\\RGCBM\\offline\\preflight_windows.ps1
D:\\RGCBM\\offline\\launch_jupyter.cmd
D:\\RGCBM\\notebooks\\RGCBM_Offline.ipynb
D:\\RGCBM\\data\\
D:\\RGCBM\\assets\\
```

Do not run setup from inside the ZIP and do not rename the extracted folders.

## D. Preflight (normal Anaconda PowerShell Prompt)

Open **Anaconda PowerShell Prompt** from the Start menu. This is important:
`conda` must be initialized in that shell. Run each line separately:

```powershell
D:
cd D:\\RGCBM
conda --version
powershell -ExecutionPolicy Bypass -File .\offline\preflight_windows.ps1
```

Preflight checks 64-bit Windows, Anaconda, disk format/free space, `nvidia-smi`,
the local browser, and the bundled Microsoft VC++ signature. It writes
`D:\\RGCBM\\offline\\preflight_report.json`. Resolve every item reported as
an issue before continuing. Warnings should also be reviewed.

Recommended capacity is at least 60 GB for extraction, datasets, checkpoints,
and generated figures. Keep the project on a local writable NTFS volume.

## E. Create the offline environment (normal Anaconda PowerShell Prompt)

Still in `D:\\RGCBM`, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\setup_windows.ps1
```

This uses `conda --offline` and `pip --no-index`; it must not contact an
internet package index. It creates `D:\\RGCBM\\conda-env`, installs the locked
dependencies, runs `pip check`, audits versions, and registers the Jupyter
kernel. The script also performs the GPU check. If you need to inspect the
environment before fixing a GPU issue, use the explicit diagnostic only:

```powershell
powershell -ExecutionPolicy Bypass -File .\offline\setup_windows.ps1 -SkipGpuCheck
```

Do not use `-SkipGpuCheck` as final acceptance.

## F. Validate with networking disconnected

Turn off Wi-Fi and unplug Ethernet before this acceptance test. Then run:

```powershell
cd D:\\RGCBM
.\conda-env\\python.exe .\offline\\validate_runtime.py --require-gpu --dataset cub
.\conda-env\\python.exe .\offline\\validate_runtime.py --require-gpu --dataset pets
.\conda-env\\python.exe .\offline\\validate_runtime.py --require-gpu --dataset brain_mri
```

Each run checks imports, native DLL loading, GPU forward/backward passes,
checkpoint writing/reloading, figures, reports, and the named concept guidance.
Results are written below `D:\\RGCBM\\runtime` and `D:\\RGCBM\\results`.
Keep the JSON reports for your research records.

## G. Start Jupyter and run the notebook

Run as a normal user:

```powershell
cd D:\\RGCBM
.\offline\\launch_jupyter.cmd
```

The command opens a local Jupyter browser page. Open
`notebooks/RGCBM_Offline.ipynb`, select **RG-CBM Anaconda offline GPU
(Python 3.11)**, and run the first setup cells. For the first check leave
`SMOKE_TEST = True`; this runs a short real training pass. After it succeeds,
set `SMOKE_TEST = False` and choose the datasets through `RUN_ALL_DATASETS`.
The notebook saves independent results, checkpoints, metrics, concept
prototype figures, guidance maps, heatmaps, ablation tables, and explanation
reports for CUB, Oxford-IIIT Pet, and brain MRI.

Do not run the notebook in the Anaconda base Python 3.13.5 kernel. That kernel
does not contain the locked project environment.

## H. Resume, backup, and safe shutdown

Training progress is saved under `results/` and checkpoint folders. To resume,
reopen the same notebook and select the same `RUN_NAME`; the notebook detects
the latest checkpoint. Copy `results/`, `runtime/`, and the notebook outputs
to a second local drive after each completed dataset. Close Jupyter with
`Ctrl+C` in its PowerShell window before shutting down Windows.

## Troubleshooting

| Symptom | Action |
|---|---|
| `conda` is not recognized | Open **Anaconda PowerShell Prompt**, not ordinary PowerShell. |
| `nvidia-smi` missing | Install the signed RTX PRO Blackwell driver, reboot, and repeat preflight. |
| CUDA or DLL error | Confirm the driver is current, reboot, and rerun `gpu_check.py`; do not install random CUDA wheels. |
| Checksum mismatch | Re-copy/reassemble the ZIP; do not bypass checksum verification. |
| Less than 20 GB free | Move the project to a larger local NTFS drive and rerun extraction. |
| Notebook says `data` or `assets` missing | Set the working folder to `D:\\RGCBM` and use the supplied notebook, not an older Drive copy. |
| Jupyter opens the wrong kernel | Select the exact RG-CBM Python 3.11 kernel from Kernel → Change Kernel. |
| Out-of-memory | Lower `BATCH_SIZE` in the notebook; use dedicated VRAM, not Windows shared memory, for sizing. |

## Acceptance evidence and boundaries

The repository contains Linux CPU notebook/dependency/guidance validation
reports and the generated offline bundle manifest. Those checks validate the
archive contents and workflow; they cannot emulate Windows DLL loading,
administrator rights, NVIDIA driver installation, or the physical RTX GPU.
The three commands in section F are therefore mandatory acceptance checks on
the target computer before permanent disconnection.
