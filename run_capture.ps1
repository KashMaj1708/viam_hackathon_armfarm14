# Capture anomaly training images for track3_sort.py (armfarm14)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$vpy = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "Run run_track3.bat once first (it sets up Python)." -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }
$env:VIAM_MACHINE_ADDRESS = "armfarm14-main.310sld03v2.viam.cloud"
if (Test-Path "$PSScriptRoot\viam_keys.ps1") { . "$PSScriptRoot\viam_keys.ps1" }
while ($true) {
    $label = Read-Host "Label for the items now in the neutral zone [good / bad / manual-good / manual-bad / upload / q = quit]"
    if ($label -eq "upload") { & $vpy track3_sort.py --upload; continue }
    if ($label -eq "manual-good") { & $vpy track3_sort.py --manual good; continue }
    if ($label -eq "manual-bad") { & $vpy track3_sort.py --manual bad; continue }
    if ($label -ne "good" -and $label -ne "bad") { break }
    & $vpy track3_sort.py --capture $label
    Write-Host "Images so far:" (Get-ChildItem train_images -Recurse -File | Measure-Object).Count
}
Read-Host "Done. Press Enter to close"
