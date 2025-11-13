@echo off
setlocal

REM Save current script directory
set "SCRIPT_DIR=%~dp0"

REM Check if venv exists
if not exist "%SCRIPT_DIR%venv\" (
    echo Creating virtual environment...
    python -m venv "%SCRIPT_DIR%venv"
    
    echo Installing requirements...
    call "%SCRIPT_DIR%venv\Scripts\activate.bat"
    pip install --upgrade pip
    pip install -r "%SCRIPT_DIR%requirements.txt"
)

REM Activate the virtual environment
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

REM Run the Python script and log output
python "%SCRIPT_DIR%phase1.py" > "%SCRIPT_DIR%Errorlog.txt" 2>&1
