@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0venv\Scripts\python.exe"
set "VENV_UVICORN=%~dp0venv\Scripts\uvicorn.exe"
set "VENV_STREAMLIT=%~dp0venv\Scripts\streamlit.exe"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found at %~dp0venv
    pause
    exit /b 1
)

set "PYTHONPATH=%~dp0;%PYTHONPATH%"

echo ======================================================================
echo  AI-Powered Unified Service Management (USM) System
echo ======================================================================
echo Starting FastAPI REST Backend on http://localhost:8000 ...
start "USM FastAPI Backend" "%VENV_UVICORN%" api.main:app --host 127.0.0.1 --port 8000 --reload

timeout /t 2 /nobreak >nul

echo Starting Streamlit Portal on http://localhost:8501 ...
start "USM Streamlit Frontend" "%VENV_STREAMLIT%" run app.py --server.port 8501

echo.
echo ======================================================================
echo  Services started in separate windows!
echo   - FastAPI Backend:       http://localhost:8000
echo   - Interactive API Docs:  http://localhost:8000/docs
echo   - Streamlit Portal:      http://localhost:8501
echo ======================================================================
pause
