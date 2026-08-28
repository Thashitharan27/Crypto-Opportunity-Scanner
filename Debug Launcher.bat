@echo off
setlocal
cd /d "%~dp0"
title Crypto Strategy Lab - Debug Launcher

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0app.py"
    set "exit_code=%errorlevel%"
    if not "%exit_code%"=="0" pause
    exit /b %exit_code%
)

echo The backtester's Python environment was not found.
echo Expected: %~dp0.venv\Scripts\python.exe
echo.
echo Please reinstall the project requirements, then try again.
pause
exit /b 1
