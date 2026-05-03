# scripts/make_submission.ps1
# ----------------------------------------------------------------------------
# Builds a self-contained submission zip for thesis defense.
#
# Contents:
#   • All git-tracked source files (clean, .gitignore respected)
#   • ResNet-50 weights         (utils/model/weights/*.pth)
#   • Sample input + style images (utils/data/in-images/, utils/data/styles/)
#   • Pre-rendered sample experiments, if present (analysis/experiments/)
#   • Dockerfile, docker-compose.yml, README, requirements.txt
#
# Excluded: venv, node_modules, build artefacts, the Thesis/ LaTeX folder,
# the SQLite job DB, and the .claude/ tooling state.
#
# Usage (from repo root):
#   .\scripts\make_submission.ps1
#   .\scripts\make_submission.ps1 -OutputName "Szollos_Marcell_szakdolgozat.zip"
# ----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$OutputName = "Szollos_Marcell_szakdolgozat.zip",
    [string]$StagingDir = "submission_staging"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot

Write-Host "Repo root: $RepoRoot"

# 1) Fresh staging directory
if (Test-Path $StagingDir) { Remove-Item $StagingDir -Recurse -Force }
New-Item -ItemType Directory -Path $StagingDir | Out-Null

# 2) Export tracked files via `git archive` (respects .gitignore automatically)
Write-Host "Exporting tracked files via git archive..."
$archivePath = Join-Path $StagingDir "_tracked.zip"
git archive --format=zip --output=$archivePath HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }

Expand-Archive -Path $archivePath -DestinationPath $StagingDir -Force
Remove-Item $archivePath

# 3) Copy in non-tracked, but runtime-required assets
#    (in-images/ already contains the styles/ subdir, so it covers both.)
$extras = @(
    "utils\model\weights",      # .pth files (gitignored as *.pth)
    "utils\data\in-images"      # sample inputs + styles (gitignored as *.jpg)
)
foreach ($rel in $extras) {
    $src = Join-Path $RepoRoot $rel
    if (-not (Test-Path $src)) {
        Write-Warning "Missing: $rel (skipped)"
        continue
    }
    $dst = Join-Path $StagingDir $rel
    New-Item -ItemType Directory -Path (Split-Path $dst -Parent) -Force | Out-Null
    Copy-Item $src $dst -Recurse -Force
    Write-Host "  + $rel"
}

# 4) Optional: pre-rendered example experiment outputs if present
$expDir = Join-Path $RepoRoot "analysis\experiments"
if (Test-Path $expDir) {
    $dst = Join-Path $StagingDir "analysis\experiments"
    Copy-Item $expDir $dst -Recurse -Force
    Write-Host "  + analysis/experiments/ (sample outputs)"
}

# 5) Strip pycache / DS_Store / log noise that may have slipped in
Get-ChildItem $StagingDir -Recurse -Force -Include "__pycache__","*.pyc",".DS_Store","*.log","*.log.json" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 6) Create the final zip
$outPath = Join-Path $RepoRoot $OutputName
if (Test-Path $outPath) { Remove-Item $outPath -Force }
Write-Host "Compressing → $OutputName ..."
Compress-Archive -Path "$StagingDir\*" -DestinationPath $outPath -CompressionLevel Optimal

# 7) Cleanup staging
Remove-Item $StagingDir -Recurse -Force

$size = (Get-Item $outPath).Length / 1MB
Write-Host ("Done. {0} ({1:N1} MB)" -f $OutputName, $size)
