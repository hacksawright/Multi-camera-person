param([switch]$AcceptInsightFaceNonCommercial)
$ErrorActionPreference='Stop'
if (-not $AcceptInsightFaceNonCommercial) {
    throw 'Public InsightFace pretrained model-zoo weights have separate/non-commercial research terms. For research/evaluation, re-run with -AcceptInsightFaceNonCommercial; otherwise provide a separately licensed det_10g.onnx.'
}
$ProjectRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot=Split-Path -Parent $ProjectRoot
$Cache=Join-Path $WorkspaceRoot '.cache\downloads'
$Dst=Join-Path $ProjectRoot 'models\scrfd\det_10g.onnx'
New-Item -ItemType Directory -Force -Path $Cache,(Split-Path -Parent $Dst) | Out-Null
if (Test-Path $Dst) { Write-Host "Already exists: $Dst" -ForegroundColor Green; exit 0 }
$Zip=Join-Path $Cache 'buffalo_l.zip'
Write-Host 'Downloading buffalo_l (contains SCRFD 10G)...' -ForegroundColor Cyan
& curl.exe -L --fail --retry 5 --retry-all-errors -C - -o $Zip 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }
$Tmp=Join-Path $Cache 'buffalo_l_extract'
Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
Expand-Archive -Path $Zip -DestinationPath $Tmp -Force
$Found=Get-ChildItem $Tmp -Recurse -Filter 'det_10g.onnx' | Select-Object -First 1
if (-not $Found) { throw 'det_10g.onnx not found in archive.' }
Copy-Item $Found.FullName $Dst -Force
Remove-Item -Recurse -Force $Tmp
Write-Host "SCRFD 10G ready: $Dst" -ForegroundColor Green
