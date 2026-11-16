$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    rtk py -3.12 -m venv (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.12 virtual environment.' }
}
# The index's R2 mirror failed its hash check locally; use the official origin.
rtk $VenvPython -c "from importlib.metadata import distributions; raise SystemExit(0 if any(d.metadata['Name']=='torch' and d.version=='2.10.0+cu128' for d in distributions()) else 1)"
if ($LASTEXITCODE -ne 0) {
    rtk $VenvPython -m pip install 'https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=fbde8f6a9ec8c76979a0d14df21c10b9e5cab6f0d106a73ca73e2179bc597cae'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
}
$LockPath = Join-Path $ProjectRoot 'requirements-lock-win.txt'
rtk $VenvPython -m pip install -r $LockPath
if ($LASTEXITCODE -ne 0) { throw 'Core dependency installation failed.' }
rtk $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
rtk $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.ones(64,64,device='cuda'); y=x@x; torch.cuda.synchronize(); assert y[0,0].item()==64; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),'CUDA forward passed')"
if ($LASTEXITCODE -ne 0) { throw 'CUDA forward check failed.' }
rtk $VenvPython (Join-Path $ProjectRoot 'assets\test_models\verify_model.py')
if ($LASTEXITCODE -ne 0) { throw 'Real NAM CPU/CUDA forward check failed.' }
