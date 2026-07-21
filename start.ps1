$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null

$env:DISABLE_TTS = "true"
$env:PYTHONPATH = $root
$env:API_URL = "http://127.0.0.1:8000"

Write-Host "Stopping old processes on ports 8000 and 8501..."
Get-NetTCPConnection -LocalPort 8000,8501 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

Write-Host "Starting backend on http://127.0.0.1:8000 ..."
Start-Process python `
    -ArgumentList "-m","uvicorn","backend.main:app","--host","127.0.0.1","--port","8000" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\backend.log" `
    -RedirectStandardError "$root\logs\backend.err" `
    -WindowStyle Hidden

$ready = $false
for ($i = 1; $i -le 60; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            Write-Host "Backend is ready."
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    Write-Host "Backend failed to start. Check logs\backend.err"
    Get-Content "$root\logs\backend.err" -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Starting frontend on http://127.0.0.1:8501 ..."
Start-Process python `
    -ArgumentList "-m","streamlit","run","frontend/app.py","--server.port","8501","--server.address","127.0.0.1","--browser.gatherUsageStats","false" `
    -WorkingDirectory $root `
    -RedirectStandardOutput "$root\logs\frontend.log" `
    -RedirectStandardError "$root\logs\frontend.err" `
    -WindowStyle Hidden

Start-Sleep -Seconds 3
Write-Host ""
Write-Host "Ready!"
Write-Host "  Frontend: http://127.0.0.1:8501"
Write-Host "  Backend:  http://127.0.0.1:8000/docs"
Write-Host "  Logs:     $root\logs\"
