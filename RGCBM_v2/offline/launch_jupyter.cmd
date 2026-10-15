@echo off
set "RGCBM_ROOT=%~dp0.."
cd /d "%RGCBM_ROOT%"
set "PATH=%RGCBM_ROOT%\conda-env;%RGCBM_ROOT%\conda-env\Library\bin;%RGCBM_ROOT%\conda-env\Scripts;%PATH%"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_DATASETS_OFFLINE=1"
set "PIP_NO_INDEX=1"
set "JUPYTER_RUNTIME_DIR=%RGCBM_ROOT%\runtime"
set "MPLCONFIGDIR=%RGCBM_ROOT%\runtime\matplotlib"
if not exist "%JUPYTER_RUNTIME_DIR%" mkdir "%JUPYTER_RUNTIME_DIR%"
"%RGCBM_ROOT%\conda-env\python.exe" -m notebook --ServerApp.root_dir="%RGCBM_ROOT%" --ip=127.0.0.1
