# Windows + AMD ROCm 7.2.1 setup script for LATO.2 3D Studio (Python 3.12 via uv)
$ErrorActionPreference = "Stop"

Write-Host "Creating Python 3.12 virtual environment at .venv..." -ForegroundColor Cyan
uv venv .venv --python 3.12

Write-Host "Installing native Windows PyTorch 2.9.1 + ROCm 7.2.1 wheels..." -ForegroundColor Cyan
uv pip install --python .venv\Scripts\python.exe --no-cache `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl

Write-Host "Installing core LATO.2, TripoSG, Hunyuan3D-2.1, and Gradio Studio dependencies..." -ForegroundColor Cyan
uv pip install --python .venv\Scripts\python.exe --no-upgrade `
  trimesh tqdm ninja psutil opencv-python-headless huggingface_hub open3d `
  plyfile zstandard easydict scipy einops timm transformers safetensors `
  kornia scikit-image diffusers accelerate jaxtyping omegaconf peft typeguard `
  pymeshlab gradio

Write-Host "Verifying AMD ROCm GPU detection..." -ForegroundColor Cyan
.venv\Scripts\python.exe -c "import torch; assert 'rocm' in torch.__version__, torch.__version__; assert torch.cuda.is_available(); print('OK:', torch.__version__, 'on', torch.cuda.get_device_name(0))"

Write-Host "Optional: clone third-party scaffold repos (TripoSG / Hunyuan3D-2.1) into third_party/ if desired." -ForegroundColor Yellow
Write-Host "Next: .venv\Scripts\python.exe scripts/ckpt_download.py" -ForegroundColor Green
Write-Host "Then: .\launch_studio.bat" -ForegroundColor Green
