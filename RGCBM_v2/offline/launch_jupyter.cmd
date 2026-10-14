@echo off
set "RGCBM_ROOT=%~dp0.."
cd /d "%RGCBM_ROOT%"
set "PATH=%RGCBM_ROOT%\conda-env;%RGCBM_ROOT%\conda-env\Library\bin;%RGCBM_ROOT%\conda-env\Scripts;%PATH%"
"%RGCBM_ROOT%\conda-env\python.exe" -m notebook --ServerApp.root_dir="%RGCBM_ROOT%" --ip=127.0.0.1
