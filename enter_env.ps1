$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ProjectRoot
$env:PIT_WORKSPACE_ROOT = $WorkspaceRoot
$env:PIT_ASSETS_ROOT = $ProjectRoot
$env:HF_HOME = Join-Path $WorkspaceRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $WorkspaceRoot '.cache\torch'
$env:YOLO_CONFIG_DIR = Join-Path $WorkspaceRoot '.cache\ultralytics'
$env:PIP_CACHE_DIR = Join-Path $WorkspaceRoot '.cache\pip'
$env:MPLCONFIGDIR = Join-Path $WorkspaceRoot '.cache\matplotlib'
$env:XDG_CACHE_HOME = Join-Path $WorkspaceRoot '.cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Set-Location $ProjectRoot
$Venv = Join-Path $WorkspaceRoot '.venv'
if (Test-Path (Join-Path $Venv 'Scripts\Activate.ps1')) { & (Join-Path $Venv 'Scripts\Activate.ps1') }
Write-Host 'PERSON IDENTITY TRACKING - CLEAN' -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Venv:    $Venv"
Write-Host "Models:  $(Join-Path $ProjectRoot 'models')"
Write-Host "Cache:   $(Join-Path $WorkspaceRoot '.cache')"
