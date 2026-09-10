$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$Data=Join-Path $ProjectRoot 'data'
New-Item -ItemType Directory -Force -Path $Data | Out-Null
$urls=@(
 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c0.avi',
 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c1.avi',
 'https://documents.epfl.ch/groups/c/cv/cvlab-pom-video1/www/4p-c2.avi'
)
for($i=0;$i -lt 3;$i++) {
    $dst=Join-Path $Data "cam$i.avi"
    Write-Host "Downloading/resuming cam$i -> $dst" -ForegroundColor Cyan
    & curl.exe -L --fail --retry 5 --retry-all-errors -C - -o $dst $urls[$i]
    if ($LASTEXITCODE -ne 0) { throw "Download failed for cam$i" }
}
Write-Host 'EPFL videos ready.' -ForegroundColor Green
