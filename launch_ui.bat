@echo off
setlocal
title Reality Engine Launcher
color 0B
cd /d "%~dp0"

echo ================================================================
echo           REALITY ENGINE - SPATIAL WORKSTATION
echo ================================================================
echo.

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js is not found in PATH. Install Node.js 20+ and retry.
    pause
    exit /b 1
)

for /f %%v in ('node -p "process.versions.node.split('.')[0]"') do set "NODEMAJOR=%%v"
if %NODEMAJOR% LSS 20 (
    echo [ERROR] Node.js %NODEMAJOR% found; Node.js 20 or newer is required. Install it from https://nodejs.org and retry.
    pause
    exit /b 1
)

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer was not found ("%PY%"). Install it from https://www.python.org and retry.
    pause
    exit /b 1
)

"%PY%" -c "import fastapi, uvicorn, sqlalchemy" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] The engine's Python packages are missing for "%PY%".
    echo         In this folder run:
    echo             python -m venv .venv
    echo             .venv\Scripts\activate
    echo             pip install -e ".[dev]"
    pause
    exit /b 1
)

if not exist "frontend\node_modules\" (
    echo [INFO] First-time setup: installing UI dependencies...
    pushd frontend
    call npm install
    popd
)

:: Each server is started only if it is not already answering.
call :up http://localhost:8100/api/health 1
if errorlevel 1 (
    echo [INFO] Starting engine API on http://localhost:8100 ...
    start "Reality Engine API" /min cmd /k ""%PY%" -m uvicorn apps.api.main:app --port 8100"
)
call :up http://localhost:3000 1
if errorlevel 1 (
    echo [INFO] Starting Studio UI on http://localhost:3000 ...
    start "Reality Engine UI" /min cmd /k "cd /d "%~dp0frontend" && npm run dev"
)

echo [INFO] Waiting for the engine...
call :up http://localhost:8100/api/health 60
if errorlevel 1 (
    echo [ERROR] The API did not come up on port 8100. Open the "Reality Engine API" window to see why.
    pause
    exit /b 1
)
echo [INFO] Waiting for the UI (first start compiles, can take a minute)...
call :up http://localhost:3000 120
if errorlevel 1 (
    echo [ERROR] The UI did not come up on port 3000. Open the "Reality Engine UI" window to see why.
    pause
    exit /b 1
)

echo [OK] Opening http://localhost:3000
start "" http://localhost:3000
echo.
echo Reality Engine is running. Close the "Reality Engine API" and "Reality Engine UI" windows to stop it.
timeout /t 5 >nul
exit /b 0

:: :up <url> <seconds>  -> errorlevel 0 once the URL answers, 1 on timeout
:up
powershell -NoProfile -Command "$end=(Get-Date).AddSeconds(%2); do { try { Invoke-WebRequest '%1' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { Start-Sleep -Milliseconds 700 } } while ((Get-Date) -lt $end); exit 1"
exit /b %errorlevel%
