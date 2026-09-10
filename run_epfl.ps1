param(
    [ValidateRange(1,3)][int]$CameraCount = 1,
    [string]$Gallery = '',
    [string]$Output = ''
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot=Split-Path -Parent $ProjectRoot
$Python=Join-Path $WorkspaceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw "Python not found: $Python" }
$env:PIT_WORKSPACE_ROOT=$WorkspaceRoot
$env:PIT_ASSETS_ROOT=$ProjectRoot
$env:HF_HOME=Join-Path $WorkspaceRoot '.cache\huggingface'
$env:TORCH_HOME=Join-Path $WorkspaceRoot '.cache\torch'
$env:YOLO_CONFIG_DIR=Join-Path $WorkspaceRoot '.cache\ultralytics'
$env:PIP_CACHE_DIR=Join-Path $WorkspaceRoot '.cache\pip'
$env:MPLCONFIGDIR=Join-Path $WorkspaceRoot '.cache\matplotlib'
$env:XDG_CACHE_HOME=Join-Path $WorkspaceRoot '.cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) { $Output=Join-Path $ProjectRoot 'runs\epfl' }
$sources=@()
for($i=0;$i -lt $CameraCount;$i++) {
    $p=Join-Path $ProjectRoot "data\cam$i.avi"
    if (-not (Test-Path $p)) { throw "Video missing: $p. Run .\download_epfl.ps1 first." }
    $sources += $p
}
$args=@('.\infer.py','--sources') + $sources + @('--device','0','--output',$Output)
if (-not [string]::IsNullOrWhiteSpace($Gallery)) { $args += @('--gallery',$Gallery) }
Write-Host "Running clean pipeline with $CameraCount camera(s)..." -ForegroundColor Cyan
& $Python @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done: $Output" -ForegroundColor Green
