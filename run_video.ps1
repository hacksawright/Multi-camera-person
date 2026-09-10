param(
    [Parameter(Mandatory=$true)][string[]]$Source,
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
if ([string]::IsNullOrWhiteSpace($Output)) { $Output=Join-Path $ProjectRoot 'runs\video' }
$args=@('.\infer.py','--sources') + $Source + @('--device','0','--output',$Output)
if (-not [string]::IsNullOrWhiteSpace($Gallery)) { $args += @('--gallery',$Gallery) }
& $Python @args
exit $LASTEXITCODE
