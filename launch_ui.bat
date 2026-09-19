@echo off
setlocal EnableDelayedExpansion
title Reality Engine Workstation Launcher
color 0B

echo ================================================================
echo           REALITY ENGINE - SPATIAL WORKSTATION
echo ================================================================
echo.
echo Launching Reality Engine Professional UI...
echo.

cd /d "%~dp0frontend"

where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not found in PATH!
    echo Please install Node.js 18+ to run Reality Engine.
    pause
    exit /b 1
)

if not exist "node_modules\" (
    echo [INFO] Installing dependencies, first-time setup...
    call npm install
)

echo [INFO] Opening default browser to http://localhost:3000...
start http://localhost:3000

:: Check if port 3000 is already running
netstat -ano | findstr ":3000" | findstr "LISTENING" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo.
    echo [INFO] Reality Engine server is already running and active on port 3000.
    echo [INFO] The UI has been opened in your browser.
    echo.
    echo Press any key to close this launcher window...
    pause >nul
    exit /b 0
)

echo [INFO] Starting Reality Engine dev server on http://localhost:3000...
echo.
echo  - Press Ctrl+C in this terminal window to stop
echo.
call npm run dev
