# One-click setup and run for track3_sort.py (armfarm14)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "== armfarm14 Track 3 sort ==" -ForegroundColor Cyan

# 1. Python
$py = $null
foreach ($c in @("py", "python", "python3")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break }
}
if (-not $py) { Write-Host "Python not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and rerun." -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

# 2. venv + viam-sdk (first run only)
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    & $py -m venv .venv
}
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -c "import viam" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing viam-sdk (one time)..."
    & $vpy -m pip install --quiet --upgrade pip
    & $vpy -m pip install --quiet -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed (see above)." -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }
}

# 3. Credentials (kept only in this window)
$env:VIAM_MACHINE_ADDRESS = "armfarm14-main.310sld03v2.viam.cloud"
if (Test-Path "$PSScriptRoot\viam_keys.ps1") { . "$PSScriptRoot\viam_keys.ps1" }
# Fully autonomous: PatchCore only (no Claude agent in the loop).
# To re-enable: $env:AGENT_REVIEW="all"; $env:AGENT_SCREEN="on"; . .\anthropic_key.ps1
$env:AGENT_REVIEW = "off"
$env:AGENT_SCREEN = "off"
Write-Host "Agent review: off (fully autonomous)" -ForegroundColor Cyan
if (-not $env:VIAM_API_KEY_ID) { $env:VIAM_API_KEY_ID = Read-Host "API key ID (Viam app > armfarm14 > CONNECT > API keys)" }
if (-not $env:VIAM_API_KEY) {
    $sec = Read-Host "API key" -AsSecureString
    $env:VIAM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}

# 4. Dry run, then ask before moving
while ($true) {
    Write-Host "`n--- Dry run (arm does not move) ---" -ForegroundColor Cyan
    & $vpy track3_sort.py
    $ans = Read-Host "`nSort for real? Hand near the E-stop. [y = pick / r = re-detect / q = quit]"
    if ($ans -eq "y") {
        Write-Host "--- Sorting ---" -ForegroundColor Yellow
        & $vpy track3_sort.py --execute
        $again = Read-Host "`nRun again? [y/N]"
        if ($again -ne "y") { break }
    } elseif ($ans -eq "r") { continue } else { break }
}
Read-Host "Done. Press Enter to close"
