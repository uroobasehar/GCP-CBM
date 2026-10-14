# Complete offline Windows / Anaconda / Blackwell package

The large offline ZIP is stored as numbered Git LFS parts because a single archive exceeds
GitHub's per-file limits. All parts form **one standard ZIP**; they are not separate ZIP archives.

While online, download/clone with Git LFS:

```powershell
git lfs install
git clone https://github.com/uroobasehar/GCP-CBM.git
cd GCP-CBM
git lfs pull
powershell -ExecutionPolicy Bypass -File .\RGCBM_v2\offline_bundle\Join-OfflineBundle.ps1
```

Transfer the resulting `RGCBM_Offline.zip` to the offline system and extract it. It includes the
Windows conda runtime package set, 188 hash-locked GPU/Jupyter/LLM dependencies, all three dataset
archives, pretrained backbone, offline Qwen weights, three notebooks, code, figures and documents.
Run `offline/setup_windows.ps1` from an initialized Anaconda PowerShell Prompt, then
`offline/launch_jupyter.cmd`. These commands perform no package downloads. A current NVIDIA
RTX PRO Blackwell driver supporting CUDA 12.8 must already be installed.

The join script verifies every part and the assembled ZIP. A small Git LFS text pointer is not
the content: if sizes do not match, fetch the actual objects before going offline. Keep all part
names intact. Allow at least 60 GB free disk for cloning, assembly, extraction, environment and runs.

The separate `../RGCBM_Code.zip` is a small code/documentation download; it intentionally has no
datasets, Windows wheels or model weights. It is useful for reviewing or running the cloud notebooks.
