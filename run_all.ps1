# run_all.ps1 — pipeline tuần tự! pino -> pinn -> evaluate sau khi fno xong
$ErrorActionPreference = "Continue"
Set-Location "D:\CFD_cylinder_2d"

Write-Output "waiting for FNO to finish..."
while ($true) {
    $alive = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.Id -ne $PID }
    # kiem tra co process python (fno) nao dang chay
    $fno_alive = $false
    try {
        $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        foreach ($x in $p) { if ($x.CommandLine -match "fno.py") { $fno_alive = $true } }
    } catch { $fno_alive = $false }
    if (-not $fno_alive) { break }
    Start-Sleep -Seconds 30
}
Write-Output "FNO finished. Checking preds file..."
if (-not (Test-Path "output\preds_fno.npz")) { Write-Output "WARNING: preds_fno.npz missing!" }

Write-Output "=== PINO starting ==="
python pino.py > output\pino_log.txt 2>&1
Write-Output "PINO exit: $LASTEXITCODE"

Write-Output "=== PINN starting ==="
python pinn.py > output\pinn_log.txt 2>&1
Write-Output "PINN exit: $LASTEXITCODE"

Write-Output "=== EVALUATE starting ==="
python evaluate.py --rollout > output\evaluate_log.txt 2>&1
Write-Output "EVALUATE exit: $LASTEXITCODE"

Write-Output "=== ALL DONE ==="
Get-Content output\pino_log.txt -Tail 3
Get-Content output\pinn_log.txt -Tail 3
Get-Content output\evaluate_log.txt -Tail 12
