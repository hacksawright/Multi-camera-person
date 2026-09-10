param(
    [switch]$AcceptInsightFaceNonCommercial,
    [switch]$DownloadEpfl
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$VenvPath = Join-Path $WorkspaceRoot '.venv'
$Python = Join-Path $VenvPath 'Scripts\python.exe'
$CacheRoot = Join-Path $WorkspaceRoot '.cache'
$ModelsRoot = Join-Path $ProjectRoot 'models'
$DataRoot = Join-Path $ProjectRoot 'data'
$ThirdParty = Join-Path $ProjectRoot 'third_party'

if (-not (Test-Path $Python)) {
    throw "Existing venv not found: $Python`nExpected layout:`n$WorkspaceRoot\.venv`n$WorkspaceRoot\.cache`n$ProjectRoot"
}

$env:PIT_WORKSPACE_ROOT = $WorkspaceRoot
$env:PIT_ASSETS_ROOT = $ProjectRoot
$env:HF_HOME = Join-Path $CacheRoot 'huggingface'
$env:TORCH_HOME = Join-Path $CacheRoot 'torch'
$env:YOLO_CONFIG_DIR = Join-Path $CacheRoot 'ultralytics'
$env:PIP_CACHE_DIR = Join-Path $CacheRoot 'pip'
$env:MPLCONFIGDIR = Join-Path $CacheRoot 'matplotlib'
$env:XDG_CACHE_HOME = $CacheRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path $CacheRoot,$ModelsRoot,$DataRoot,$ThirdParty | Out-Null
New-Item -ItemType Directory -Force -Path `
    (Join-Path $ModelsRoot 'adaface'), `
    (Join-Path $ModelsRoot 'scrfd'), `
    (Join-Path $ModelsRoot 'osnet'), `
    (Join-Path $ModelsRoot 'yolo') | Out-Null

Write-Host '==========================================================' -ForegroundColor Cyan
Write-Host ' PERSON IDENTITY TRACKING - CLEAN SETUP' -ForegroundColor Cyan
Write-Host '==========================================================' -ForegroundColor Cyan
Write-Host "Project : $ProjectRoot"
Write-Host "Venv    : $VenvPath"
Write-Host "Cache   : $CacheRoot"
Write-Host "Models  : $ModelsRoot"
Write-Host "Data    : $DataRoot"

& $Python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch/CUDA check failed.' }

Write-Host '[deps] Installing/checking dependencies; existing Torch is preserved...' -ForegroundColor Cyan
& $Python -m pip install --upgrade-strategy only-if-needed -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

# Remove the unrelated PyPI distribution if it exists.
$HasBadTorchreid = (& $Python -c "import importlib.metadata as m; print('1' if any(d.metadata.get('Name','').lower()=='torchreid' for d in m.distributions()) else '0')").Trim()
if ($HasBadTorchreid -eq '1') {
    Write-Host '[deps] Removing unrelated PyPI torchreid package...' -ForegroundColor Yellow
    $saved = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python -m pip uninstall -y torchreid | Out-Host
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $saved
    if ($rc -ne 0) { throw 'Failed to uninstall unrelated PyPI torchreid package.' }
}

# Project-local upstream source. No source is read from any old project folder.
$AdaRepo = Join-Path $ThirdParty 'AdaFace'
if (-not (Test-Path (Join-Path $AdaRepo 'net.py'))) {
    Write-Host '[source] Cloning AdaFace...' -ForegroundColor Cyan
    git clone --depth 1 https://github.com/mk-minchul/AdaFace.git $AdaRepo
    if ($LASTEXITCODE -ne 0) { throw 'Failed to clone AdaFace.' }
}
$ReidRepo = Join-Path $ThirdParty 'deep-person-reid'
if (-not (Test-Path (Join-Path $ReidRepo 'torchreid\__init__.py'))) {
    Write-Host '[source] Cloning deep-person-reid...' -ForegroundColor Cyan
    git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git $ReidRepo
    if ($LASTEXITCODE -ne 0) { throw 'Failed to clone deep-person-reid.' }
}

# AdaFace compatibility patch for recent PyTorch non-contiguous tensors.
$AdaNet = Join-Path $AdaRepo 'net.py'
$AdaText = Get-Content $AdaNet -Raw
$AdaPatched = $AdaText.Replace('return input.view(input.size(0), -1)', 'return input.reshape(input.size(0), -1)').Replace('x = x.view(x.shape[0], -1)', 'x = x.reshape(x.shape[0], -1)')
if ($AdaPatched -ne $AdaText) {
    if (-not (Test-Path "$AdaNet.bak")) { Copy-Item $AdaNet "$AdaNet.bak" }
    $AdaPatched | Set-Content $AdaNet -Encoding UTF8
    Write-Host '[source] Patched AdaFace view -> reshape.' -ForegroundColor Green
}

# Clean stale deep-person-reid .pth files left inside the reused venv, then register only this project.
$SitePackages = (& $Python -c "import site; print(site.getsitepackages()[0])").Trim()
Get-ChildItem $SitePackages -Filter '*.pth' -ErrorAction SilentlyContinue | ForEach-Object {
    $txt = Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue
    if ($txt -and $txt -match 'deep-person-reid') {
        Remove-Item $_.FullName -Force
        Write-Host "[venv] Removed stale path file: $($_.Name)" -ForegroundColor Yellow
    }
}
$PthFile = Join-Path $SitePackages 'person_identity_tracking_deep_person_reid.pth'
$ReidRepo | Set-Content $PthFile -Encoding ASCII
Write-Host "[source] Registered project-local deep-person-reid: $PthFile" -ForegroundColor Green

$AdaDst = Join-Path $ModelsRoot 'adaface\adaface_ir50_webface4m.ckpt'
$Scrfd10Dst = Join-Path $ModelsRoot 'scrfd\det_10g.onnx'
$OsnetDst = Join-Path $ModelsRoot 'osnet\osnet_x1_0_msmt17.pth'
$YoloDst = Join-Path $ModelsRoot 'yolo\yolo11s.pt'

if (-not (Test-Path $AdaDst)) {
    Write-Host '[model] Downloading official AdaFace IR50 WebFace4M checkpoint...' -ForegroundColor Cyan
    & $Python -m gdown '1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN' -O $AdaDst
    if ($LASTEXITCODE -ne 0) { throw 'AdaFace checkpoint download failed.' }
}
if ((Get-Item $AdaDst).Length -lt 500000000) { throw "AdaFace checkpoint looks incomplete: $AdaDst" }

if (-not (Test-Path $OsnetDst)) {
    Write-Host '[model] Downloading OSNet x1.0 MSMT17...' -ForegroundColor Cyan
    $url = 'https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'
    & curl.exe -L --fail --retry 5 --retry-all-errors -C - -o $OsnetDst $url
    if ($LASTEXITCODE -ne 0) { throw 'OSNet download failed.' }
}
if ((Get-Item $OsnetDst).Length -lt 10000000) { throw "OSNet checkpoint looks incomplete: $OsnetDst" }

if (-not (Test-Path $YoloDst)) {
    Write-Host '[model] Downloading YOLO11s...' -ForegroundColor Cyan
    Push-Location (Split-Path -Parent $YoloDst)
    try {
        & $Python -c "from ultralytics import YOLO; YOLO('yolo11s.pt'); print('YOLO11s ready')"
        if ($LASTEXITCODE -ne 0) { throw 'YOLO11s download failed.' }
    } finally { Pop-Location }
}

if (-not (Test-Path $Scrfd10Dst)) {
    if (-not $AcceptInsightFaceNonCommercial) {
        throw "SCRFD 10G is missing. Public InsightFace pretrained weights have separate/non-commercial research terms. For research/evaluation run:`npowershell -ExecutionPolicy Bypass -File .\setup.ps1 -AcceptInsightFaceNonCommercial"
    }
    & (Join-Path $ProjectRoot 'download_scrfd10g.ps1') -AcceptInsightFaceNonCommercial
    if ($LASTEXITCODE -ne 0) { throw 'SCRFD 10G download failed.' }
}

if ($DownloadEpfl) {
    & (Join-Path $ProjectRoot 'download_epfl.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'EPFL download failed.' }
}

& $Python -c "import sys, onnxruntime as ort; sys.path.insert(0, r'$ReidRepo'); import torchreid; from torchreid.utils import FeatureExtractor; print('ORT:', ort.get_available_providers()); print('deep-person-reid:', torchreid.__file__)"
if ($LASTEXITCODE -ne 0) { throw 'Runtime/source verification failed.' }

Write-Host ''
Write-Host '[models]' -ForegroundColor Cyan
Write-Host "AdaFace IR50 : OK  $AdaDst"
Write-Host "SCRFD 10G    : OK  $Scrfd10Dst"
Write-Host "OSNet x1.0   : OK  $OsnetDst"
Write-Host "YOLO11s      : OK  $YoloDst"
Write-Host ''
Write-Host 'SETUP COMPLETE' -ForegroundColor Green
Write-Host 'Next: & ..\.venv\Scripts\python.exe .\check.py' -ForegroundColor Green
