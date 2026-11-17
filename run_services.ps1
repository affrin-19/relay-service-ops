# AI-Powered Unified Service Management (USM) System - PowerShell Launcher
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
$VenvUvicorn = Join-Path $ScriptDir "venv\Scripts\uvicorn.exe"
$VenvStreamlit = Join-Path $ScriptDir "venv\Scripts\streamlit.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at $VenvPython"
    exit 1
}

$env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🌟 AI-Powered Unified Service Management (USM) System" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "Starting FastAPI REST Backend on http://localhost:8000 ..." -ForegroundColor Green
$apiProcess = Start-Process -FilePath $VenvUvicorn -ArgumentList "api.main:app --host 127.0.0.1 --port 8000 --reload" -PassThru

Start-Sleep -Seconds 2

Write-Host "Starting Streamlit Portal on http://localhost:8501 ..." -ForegroundColor Green
$uiProcess = Start-Process -FilePath $VenvStreamlit -ArgumentList "run app.py --server.port 8501" -PassThru

Write-Host ""
Write-Host "Services are running:" -ForegroundColor Yellow
Write-Host "  • FastAPI Backend:       http://localhost:8000" -ForegroundColor White
Write-Host "  • Interactive API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host "  • Streamlit Portal:      http://localhost:8501" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C or close the process windows to stop." -ForegroundColor Gray
