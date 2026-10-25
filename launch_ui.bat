@echo off
setlocal EnableExtensions
title Reality Engine Launcher
cd /d "%~dp0"

:: Usage:  launch_ui.bat [stop] [--no-browser]
::   Env overrides: API_PORT (default 8100), UI_PORT (default 3000)
if not defined API_PORT set "API_PORT=8100"
if not defined UI_PORT set "UI_PORT=3000"
set "API_URL=http://localhost:%API_PORT%"
set "UI_URL=http://localhost:%UI_PORT%"
set "OPEN=1"
for %%a in (%*) do (
    if /i "%%a"=="stop" goto :stop
    if /i "%%a"=="--no-browser" set "OPEN=0"
)

echo ================================================================
echo           REALITY ENGINE - SPATIAL WORKSTATION
echo ================================================================
echo.

:: ---- toolchain ------------------------------------------------------------
where node >nul 2>nul || call :die "Node.js is not on PATH. Install Node.js 20+ from https://nodejs.org and retry."
for /f %%v in ('node -p "process.versions.node.split('.')[0]"') do set "NODEMAJOR=%%v"
if %NODEMAJOR% LSS 20 call :die "Node.js %NODEMAJOR% found; Node.js 20 or newer is required."

:: Python: this folder's .venv, else the main checkout's .venv (git worktrees), else create one.
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
for /f "delims=" %%g in ('git rev-parse --path-format^=absolute --git-common-dir 2^>nul') do for %%p in ("%%g\..") do set "MAINROOT=%%~fp"
if not defined PY if defined MAINROOT if exist "%MAINROOT%\.venv\Scripts\python.exe" set "PY=%MAINROOT%\.venv\Scripts\python.exe"
if not defined PY (
    echo [INFO] No virtual environment found - creating .venv ...
    set "BASEPY=python"
    where py >nul 2>nul && set "BASEPY=py -3"
    %BASEPY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul || call :die "Python 3.10+ is required. Install it from https://www.python.org and retry."
    %BASEPY% -m venv .venv || call :die "Could not create .venv."
    set "PY=%~dp0.venv\Scripts\python.exe"
)
"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul || call :die "Python 3.10+ is required (found: %PY%)."

"%PY%" -c "import fastapi, uvicorn, sqlalchemy" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing engine Python packages into %PY% ... ^(first run only^)
    "%PY%" -m pip install -e ".[dev]" || call :die "pip install failed - see the output above."
)

if not exist "frontend\node_modules\" (
    echo [INFO] First-time setup: installing UI dependencies...
    pushd frontend
    call npm install || (popd & call :die "npm install failed - see the output above.")
    popd
)

:: ---- start servers (each only if not already answering) ---------------------
if not exist logs mkdir logs
:: Worktree: node_modules is a symlink into the main checkout; Turbopack needs a root that contains it.
dir /al /b frontend 2>nul | findstr /x /c:"node_modules" >nul && if defined MAINROOT set "REALITY_TURBOPACK_ROOT=%MAINROOT%"
set "REALITY_BACKEND_URL=%API_URL%"
set "CORS_ORIGINS=%UI_URL%"

call :needstart "Engine API" %API_PORT% %API_URL%/api/health && (
    echo [INFO] Starting Engine API on port %API_PORT% ...
    start "Reality Engine API" /min cmd /c ""%PY%" -m uvicorn apps.api.main:app --port %API_PORT% >> "%~dp0logs\api.log" 2>&1"
)
call :needstart "Studio UI" %UI_PORT% %UI_URL% && (
    echo [INFO] Starting Studio UI on port %UI_PORT% ...
    start "Reality Engine UI" /min cmd /c "cd /d "%~dp0frontend" && npm run dev -- --port %UI_PORT% >> "%~dp0logs\ui.log" 2>&1"
)

echo [INFO] Waiting for the engine API...
call :wait %API_URL%/api/health 180 || call :fail "Engine API" logs\api.log
echo [INFO] Waiting for the Studio UI (first start compiles, can take a minute)...
call :wait %UI_URL% 240 || call :fail "Studio UI" logs\ui.log

echo [OK] Reality Engine is up: %UI_URL%
if "%OPEN%"=="1" start "" %UI_URL%
echo.
echo Stop it with:  launch_ui.bat stop     (logs: logs\api.log, logs\ui.log)
exit /b 0

:: ---- stop -----------------------------------------------------------------
:stop
echo Stopping Reality Engine...
taskkill /FI "WINDOWTITLE eq Reality Engine API*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq Reality Engine UI*" /T /F >nul 2>nul
:: Catch anything still listening (e.g. servers started outside this launcher).
powershell -NoProfile -Command "foreach($p in %API_PORT%,%UI_PORT%){ Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }"
echo Stopped.
exit /b 0

:: ---- helpers ----------------------------------------------------------------
:: :needstart <name> <port> <health-url>  -> errorlevel 0 if it must be started, 1 if already up
:needstart
call :wait %~3 1 && (echo [INFO] %~1 already running on port %~2.& exit /b 1)
netstat -ano | findstr /R /C:":%~2 .*LISTENING" >nul && call :die "Port %~2 is held by another program that is not %~1. Free it, or change API_PORT / UI_PORT."
exit /b 0

:: :wait <url> <seconds>  -> errorlevel 0 once the URL answers, 1 on timeout
:wait
set /a "_left=%~2"
:waitloop
curl.exe -s -f -m 2 -o NUL "%~1" && exit /b 0
set /a "_left-=1"
if %_left% LEQ 0 exit /b 1
ping -n 2 127.0.0.1 >nul
goto :waitloop

:: :fail <name> <logfile>  -> show the log tail and exit
:fail
echo [ERROR] %~1 did not come up. Last log lines (%~2):
powershell -NoProfile -Command "if (Test-Path '%~2') { Get-Content '%~2' -Tail 15 }"
if not defined NOPAUSE pause
exit 1

:: :die <message>
:die
echo [ERROR] %~1
if not defined NOPAUSE pause
exit 1
