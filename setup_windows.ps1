param([switch]$SkipDownloads)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv not found. Install uv or add it to PATH before running this script.'
}
if (-not (Test-Path $Python)) {
    & uv venv --python 3.10 (Join-Path $Root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment setup failed.' }
}
& uv pip install --python $Python --index-url https://download.pytorch.org/whl/cu118 'torch==2.5.1+cu118' 'torchvision==0.20.1+cu118'
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& uv pip install --python $Python -r (Join-Path $Root 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
if (-not (Test-Path (Join-Path $Root 'models\yoloe-v8s-coco80.pt'))) {
    & uv pip install --python $Python -r (Join-Path $Root 'requirements-text.txt')
    if ($LASTEXITCODE -ne 0) { throw 'One-time text encoder installation failed.' }
}
if (-not $SkipDownloads) {
    & $Python (Join-Path $Root 'download_assets.py')
    if ($LASTEXITCODE -ne 0) { throw 'Asset download failed. Set HTTPS_PROXY if needed.' }
    if (-not (Test-Path (Join-Path $Root 'models\yoloe-v8s-coco80.pt'))) {
        & $Python (Join-Path $Root 'prepare_model.py')
        if ($LASTEXITCODE -ne 0) { throw 'Offline vocabulary preparation failed.' }
    }
}
& $Python (Join-Path $Root 'benchmark.py') --probe
if ($LASTEXITCODE -ne 0) { throw 'Runtime probe failed.' }
Write-Host 'Setup complete. Driver settings were not changed.'
