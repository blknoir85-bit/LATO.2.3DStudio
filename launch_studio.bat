@echo off
cd /d "%~dp0"
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv\Scripts\python.exe app.py
pause
