<#
.SYNOPSIS
  Reproducibly builds the offline Windows desktop app (desktop/): Next.js standalone UI + embedded CPython + Electron installer.
.DESCRIPTION
  Output: desktop/dist/Reality Engine-Setup-<version>.exe (NSIS) and desktop/dist/win-unpacked/ (runnable without installing).
  Version comes from pyproject.toml ([project].version) -- the single source. Needs: Node 20+, npm, a host Python of the SAME
  minor version as the embedded one (used only to download wheels), internet (build time only; the app itself is offline).
  Not code-signed. Not published.
#>
param(
  [string]$PythonEmbedVersion = "3.14.7",
  [switch]$SkipInstaller   # stage + unpacked dir only (faster)
)
$ErrorActionPreference = "Stop"
$sw = [Diagnostics.Stopwatch]::StartNew()
$root = Split-Path -Parent $PSScriptRoot
$desktop = Join-Path $root "desktop"
$stage = Join-Path $desktop "build"
$cache = Join-Path $desktop ".cache"

function Step($m) { Write-Host "`n=== $m  [$([int]$sw.Elapsed.TotalSeconds)s]" -ForegroundColor Cyan }
function Run($exe, [string[]]$a, $cwd = $root) {
  Push-Location $cwd
  try { & $exe @a; if ($LASTEXITCODE -ne 0) { throw "$exe $($a -join ' ') failed ($LASTEXITCODE)" } } finally { Pop-Location }
}

# 0. version (single source: pyproject.toml)
$version = (Select-String -Path (Join-Path $root "pyproject.toml") -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
if (-not $version) { throw "cannot read version from pyproject.toml" }
Step "Version $version"

# host python must match embedded minor (wheel ABI)
$hostPy = (& python -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
$embedMinor = ($PythonEmbedVersion -split '\.')[0..1] -join '.'
if ($hostPy -ne $embedMinor) { throw "host python is $hostPy but embedded python is $embedMinor; wheels would not match" }

# 1. UI (Next standalone)
Step "Building UI (next build, standalone)"
$fe = Join-Path $root "frontend"
Run npm @("ci", "--no-audit", "--no-fund") $fe
$env:REALITY_DESKTOP = "1"
$env:NEXT_TELEMETRY_DISABLED = "1"
Run npm @("run", "build") $fe
Remove-Item Env:REALITY_DESKTOP
$ui = Join-Path $stage "ui"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force $ui | Out-Null
Copy-Item -Recurse -Force (Join-Path $fe ".next\standalone\*") $ui
New-Item -ItemType Directory -Force (Join-Path $ui ".next\static") | Out-Null
Copy-Item -Recurse -Force (Join-Path $fe ".next\static\*") (Join-Path $ui ".next\static")
if (Test-Path (Join-Path $fe "public")) { Copy-Item -Recurse -Force (Join-Path $fe "public") (Join-Path $ui "public") }
if (-not (Test-Path (Join-Path $ui "server.js"))) { throw "standalone server.js missing" }

# 2. Embedded Python + deps + engine sources
Step "Staging embedded Python $PythonEmbedVersion"
New-Item -ItemType Directory -Force $cache | Out-Null
$zip = Join-Path $cache "python-$PythonEmbedVersion-embed-amd64.zip"
if (-not (Test-Path $zip)) {
  Invoke-WebRequest "https://www.python.org/ftp/python/$PythonEmbedVersion/python-$PythonEmbedVersion-embed-amd64.zip" -OutFile $zip
}
$py = Join-Path $stage "python"
Expand-Archive $zip $py
$pth = Get-ChildItem $py -Filter "python*._pth" | Select-Object -First 1
# Isolated mode (what ._pth implies): site-packages + the engine sources, relative to python.exe (so `app` = resources/python/app).
@("python$($embedMinor -replace '\.','').zip", ".", "Lib\site-packages", "app", "import site") | Set-Content -Encoding ascii $pth.FullName

Step "Installing locked Python dependencies (python-requirements.lock; regenerate from .in with pip freeze --path) into the embedded runtime"
$sp = Join-Path $py "Lib\site-packages"
Run python @("-m", "pip", "install", "--no-warn-script-location", "--only-binary=:all:", "--target", $sp, "--no-deps", "-r", (Join-Path $desktop "python-requirements.lock"), "--quiet")
# trim: caches, test suites, bytecode are rebuilt-free/needless
Get-ChildItem $sp -Recurse -Directory -Include "__pycache__", "tests", "test" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Step "Copying engine sources"
$appDir = Join-Path $py "app"
New-Item -ItemType Directory -Force $appDir | Out-Null
# top-level import packages of the engine (pyproject [tool.setuptools.packages.find] include) + those apps.api imports at runtime
$pkgs = "engine","world_ir","provenance","events","evidence","reconstruction","perception","exporters","sdk","apps","uncertainty","worldstore","registration","trajectories","procedural","synthetic","assistant","config"
foreach ($p in $pkgs) {
  $src = Join-Path $root $p
  if (Test-Path $src) {
    & robocopy $src (Join-Path $appDir $p) /E /XD __pycache__ tests .pytest_cache /XF *.pyc /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy $p failed ($LASTEXITCODE)" }
  }
}
$global:LASTEXITCODE = 0

Step "Import check of the staged runtime"
Run (Join-Path $py "python.exe") @("-c", "import apps.api.main, sys; print('api import ok', sys.version.split()[0])") $stage

# 3. Electron
Step "Installing Electron toolchain"
Run npm @("install", "--no-audit", "--no-fund") $desktop
Step "Packaging (electron-builder)"
$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
$targets = if ($SkipInstaller) { @("--dir") } else { @("--win", "nsis", "dir") }
Run npx (@("electron-builder") + $targets + @("--config", "electron-builder.yml", "-c.extraMetadata.version=$version")) $desktop

Step "Done"
$out = Join-Path $desktop "dist"
Get-ChildItem $out -Filter "*.exe" | ForEach-Object { "{0}  {1:N1} MB" -f $_.Name, ($_.Length / 1MB) }
$unpacked = Join-Path $out "win-unpacked"
if (Test-Path $unpacked) { "win-unpacked  {0:N1} MB" -f ((Get-ChildItem $unpacked -Recurse -File | Measure-Object Length -Sum).Sum / 1MB) }
"Total build time: {0:N0}s" -f $sw.Elapsed.TotalSeconds
