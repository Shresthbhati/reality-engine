<#
.SYNOPSIS
  Smoke-tests the BUILT desktop app (desktop/dist/win-unpacked) headlessly:
    1. launch, wait for health, create a world + room through the API, check the UI root, check data landed in the data dir
       and nothing was written to the install dir, quit gracefully, verify no leftover processes/ports;
    2. crash recovery: hard-kill the shell only (taskkill /F, no /T), then relaunch and verify the orphaned API/UI from the
       dead run were reaped (the earlier "orphaned worker holding port 8100" bug class), then quit and verify clean.
.PARAMETER UseAppData
  Use the real %APPDATA%\RealityEngine data dir instead of a throwaway temp dir (removed afterwards only if this script created it).
#>
param(
  [string]$Exe = (Join-Path (Split-Path -Parent $PSScriptRoot) "desktop\dist\win-unpacked\Reality Engine.exe"),
  [switch]$UseAppData,
  [int]$TimeoutSec = 150
)
$ErrorActionPreference = "Stop"
$script:fail = @()
function Check($ok, $msg) { if ($ok) { Write-Host "PASS  $msg" -ForegroundColor Green } else { Write-Host "FAIL  $msg" -ForegroundColor Red; $script:fail += $msg } }
if (-not (Test-Path $Exe)) { throw "built app not found: $Exe (run scripts/build_desktop.ps1)" }
$installDir = Split-Path -Parent $Exe

$createdDataDir = $false
if ($UseAppData) {
  $data = Join-Path $env:APPDATA "RealityEngine"
  $createdDataDir = -not (Test-Path $data)
} else {
  $data = Join-Path ([IO.Path]::GetTempPath()) ("re-smoke-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
  $env:REALITY_DESKTOP_DATA_DIR = $data
}
$env:REALITY_DESKTOP_HEADLESS = "1"
Write-Host "exe:      $Exe"
Write-Host "data dir: $data"

function Snapshot($dir) { Get-ChildItem $dir -Recurse -File -Force | ForEach-Object { "$($_.FullName)|$($_.Length)|$($_.LastWriteTimeUtc.Ticks)" } }
function AppProcesses { @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path -like "$installDir*" }) }
function PidAlive($p) { [bool](Get-Process -Id $p -ErrorAction SilentlyContinue) }

function Start-App {
  $p = Start-Process -FilePath $Exe -PassThru
  $rt = Join-Path $data "runtime.json"
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline -and -not $p.HasExited) {
    if (Test-Path $rt) { try { $r = Get-Content $rt -Raw | ConvertFrom-Json; if ($r.ready -and $r.mainPid -eq $p.Id) { return @{ Proc = $p; Runtime = $r } } } catch {} }
    Start-Sleep -Milliseconds 500
  }
  return @{ Proc = $p; Runtime = $null }
}

function Quit-App($app) {
  New-Item -ItemType File -Force (Join-Path $data "quit.flag") | Out-Null
  [void]$app.Proc.WaitForExit(15000)
  Start-Sleep -Seconds 2
}

$before = Snapshot $installDir
$baselineProcs = (AppProcesses).Count
Check ($baselineProcs -eq 0) "no pre-existing processes from this install dir (found $baselineProcs)"

# ---------------------------------------------------------------- scenario 1: happy path
Write-Host "`n--- scenario 1: launch / API / UI / quit"
$sw = [Diagnostics.Stopwatch]::StartNew()
$app = Start-App
Check ($null -ne $app.Runtime) "app reported ready within ${TimeoutSec}s (took $([int]$sw.Elapsed.TotalSeconds)s)"
if (-not $app.Runtime) { Get-Content (Join-Path $data "logs\*.log") -Tail 30 -ErrorAction SilentlyContinue; throw "app never became ready; see $data\logs" }
$r = $app.Runtime
$apiPort = $r.apiPort; $uiPort = $r.uiPort
Write-Host "ports: api=$apiPort ui=$uiPort  pids: main=$($r.mainPid) api=$($r.apiPid) ui=$($r.uiPid)"
Check ($apiPort -ne 8100 -and $apiPort -ne 3000 -and $uiPort -ne 8100 -and $uiPort -ne 3000) "ports are dynamic, not the dev 8100/3000"
$api = "http://127.0.0.1:$apiPort"
try {
  $h = Invoke-RestMethod "$api/api/health"
  Check ($h.status -eq "ok") "GET /api/health -> $($h | ConvertTo-Json -Compress)"

  $w = Invoke-RestMethod "$api/api/worlds" -Method Post -ContentType "application/json" -Body '{"name":"Smoke World"}'
  Check ($w.id -like "wld*") "POST /api/worlds -> id $($w.id)"
  $room = Invoke-RestMethod "$api/api/worlds/$($w.id)/rooms" -Method Post -ContentType "application/json" `
    -Body '{"name":"Smoke Room","width":4.0,"depth":3.5,"height":2.6,"openings":[{"wall":"south","kind":"door","lateral_offset":1.5,"width":0.9,"bottom":0.0,"top":2.1}]}'
  Check ([bool]$room.room_entity_id) "POST /api/worlds/{id}/rooms -> room_entity_id $($room.room_entity_id), $(@($room.entity_ids_added).Count) entities"
  $list = Invoke-RestMethod "$api/api/worlds"
  Check (@($list.items | Where-Object { $_.id -eq $w.id }).Count -eq 1) "created world is listed"

  $sys = Invoke-RestMethod "$api/api/system/verification"
  $state = @{}; foreach ($a in $sys.runtime) { $state[$a.name] = $a.status }
  Write-Host ("optional components as the packaged app reports them: " + (($state.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ", "))
  Check ($state["torch (MiDaS / Mask R-CNN)"] -eq "UNAVAILABLE") "torch honestly reported UNAVAILABLE (not bundled; host user-site not leaking in)"
  Check ($state["ifcopenshell (IFC export)"] -eq "UNAVAILABLE") "ifcopenshell honestly reported UNAVAILABLE (not bundled)"

  $ui = Invoke-WebRequest "http://127.0.0.1:$uiPort/" -UseBasicParsing
  Check ($ui.StatusCode -eq 200) "UI root GET / -> $($ui.StatusCode) ($($ui.Content.Length) bytes)"
  $uiWorlds = Invoke-RestMethod "http://127.0.0.1:$uiPort/api/worlds"
  Check (@($uiWorlds.items | Where-Object { $_.id -eq $w.id }).Count -ge 1) "UI server proxy /api/worlds reaches the API and sees the created world"
  $pre = Invoke-WebRequest -Uri "$api/api/health" -Method Options -Headers @{ Origin = "http://127.0.0.1:$uiPort"; "Access-Control-Request-Method" = "GET" } -UseBasicParsing
  Check ($pre.Headers["Access-Control-Allow-Origin"] -eq "http://127.0.0.1:$uiPort") "CORS allows the UI origin"

  Check (Test-Path (Join-Path $data "data\app.db")) "SQLite DB in data dir"
  $ws = @(Get-ChildItem (Join-Path $data "data\worldstore") -Recurse -File -ErrorAction SilentlyContinue)
  Check ($ws.Count -gt 0) "worldstore files written in data dir ($($ws.Count) files)"
  Check (Test-Path (Join-Path $data "logs\api.log")) "logs written to <data>\logs"
} finally {
  Quit-App $app
}
$left = AppProcesses
Check ($left.Count -eq 0) "graceful quit leaves no processes (found: $(@($left | ForEach-Object { "$($_.ProcessName):$($_.Id)" }) -join ', '))"
foreach ($p in $left) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
$listening = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in @($apiPort, $uiPort) })
Check ($listening.Count -eq 0) "api/ui ports $apiPort/$uiPort no longer listening"
Check (-not (Test-Path (Join-Path $data "runtime.json"))) "runtime.json cleaned up on exit"

# ---------------------------------------------------------------- scenario 2: shell hard-killed, next start reaps orphans
Write-Host "`n--- scenario 2: shell killed hard (no tree kill), relaunch must reap orphans"
$app2 = Start-App
Check ($null -ne $app2.Runtime) "second launch ready (data persisted from run 1)"
if ($app2.Runtime) {
  $o = $app2.Runtime
  $persisted = Invoke-RestMethod "http://127.0.0.1:$($o.apiPort)/api/worlds"
  Check (@($persisted.items | Where-Object { $_.id -eq $w.id }).Count -eq 1) "world created in run 1 survived the restart"
  Stop-Process -Id $o.mainPid -Force
  # Windows lets the dead shell's pid linger briefly; a user cannot relaunch inside that window, so wait for it to clear.
  $t = [Diagnostics.Stopwatch]::StartNew()
  while ($t.Elapsed.TotalSeconds -lt 20 -and (PidAlive $o.mainPid)) { Start-Sleep -Milliseconds 500 }
  $orphanApi = PidAlive $o.apiPid; $orphanUi = PidAlive $o.uiPid
  Write-Host "after hard kill of shell (+$([int]$t.Elapsed.TotalSeconds)s): api child alive=$orphanApi, ui child alive=$orphanUi (alive = real orphans for the reaper to catch)"
  $app3 = Start-App
  Check ($null -ne $app3.Runtime) "relaunch after crash is ready"
  Check (-not (PidAlive $o.apiPid) -and -not (PidAlive $o.uiPid)) "orphaned api/ui from the killed run were reaped on relaunch"
  Quit-App $app3
}
$left = AppProcesses
Check ($left.Count -eq 0) "no processes left after scenario 2 (found: $(@($left | ForEach-Object { "$($_.ProcessName):$($_.Id)" }) -join ', '))"
foreach ($p in $left) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }

# ---------------------------------------------------------------- install dir untouched
$after = Snapshot $installDir
$diff = Compare-Object $before $after
Check (-not $diff) "install dir unchanged (all writes went to the data dir)$(if ($diff) { ': ' + (($diff | Select-Object -First 3 | ForEach-Object InputObject) -join '; ') })"

if ($script:fail.Count) { Write-Host "`n--- main.log tail"; Get-Content (Join-Path $data "logs\main.log") -Tail 30 -ErrorAction SilentlyContinue }
if ($UseAppData) { if ($createdDataDir) { Remove-Item -Recurse -Force $data -ErrorAction SilentlyContinue } }
else { Remove-Item -Recurse -Force $data -ErrorAction SilentlyContinue; Remove-Item Env:REALITY_DESKTOP_DATA_DIR }
Remove-Item Env:REALITY_DESKTOP_HEADLESS
if ($script:fail.Count) { Write-Host "`nSMOKE FAILED ($($script:fail.Count)): $($script:fail -join ' | ')" -ForegroundColor Red; exit 1 }
Write-Host "`nSMOKE PASSED" -ForegroundColor Green
