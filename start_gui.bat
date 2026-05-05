@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

where python >nul 2>&1
if %errorlevel% neq 0 (
  if not exist ".venv\Scripts\python.exe" (
    echo Python not found in PATH. Install Python 3.9+ or create .venv.
    pause
    exit /b 1
  )
)

"%PY%" run_gui.py
if errorlevel 1 pause
