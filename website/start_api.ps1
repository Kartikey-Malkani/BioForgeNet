# VeriSight BioSecure — API Startup Script
# Run this instead of bare `uvicorn` to avoid Windows encoding issues
# with Unicode characters in model loading logs.

$env:PYTHONUTF8       = "1"
$env:PYTHONIOENCODING = "utf-8"

Set-Location $PSScriptRoot

Write-Host "Starting VeriSight BioSecure API on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

uvicorn api.app:app --host 127.0.0.1 --port 8000
