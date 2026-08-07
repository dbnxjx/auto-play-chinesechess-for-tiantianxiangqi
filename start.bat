@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment not found: .venv\Scripts\python.exe
    echo Create it with: py -3.12 -m venv .venv
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m app.main
pause
