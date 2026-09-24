# run_signal_rebuild.ps1
#
# Windows PowerShell launcher for run_signal_rebuild.py.  Creates a private
# Python environment in .venv, installs numpy, scipy, MNE-Python and
# pymatreader into it, then runs the signal-level test.
#
# Run from this folder:
#     powershell -ExecutionPolicy Bypass -File .\run_signal_rebuild.ps1 -DeleteAfter
#
# Parameters:
#   -Sessions       t1 (default, 111 recordings, 12.7 GB) or t1,t2 (153, 17.4 GB)
#   -MaxRecordings  stop after this many recordings (0 = all)
#   -DataDir        folder that already holds a copy of ds003775, or where to put it
#   -DeleteAfter    delete each file once analysed, so at most one is on disk
#   -StatsOnly      recompute the tests from results\signal_rebuild_rows.csv
#
# An interrupted run resumes where it stopped.  When it finishes, send back
# results\signal_rebuild.json.

param(
    [string]$Sessions = "t1",
    [int]$MaxRecordings = 0,
    [string]$DataDir = "",
    [switch]$DeleteAfter,
    [switch]$StatsOnly
)

Set-Location -Path $PSScriptRoot

$py = $null
$pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $py = "py"
    $pyArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $py = "python"
}
if ($py -eq $null) {
    Write-Host "Python 3.9 or newer is needed. Install it from https://www.python.org/downloads/"
    Write-Host "and tick 'Add python.exe to PATH' in the installer, then run this again."
    exit 1
}
$ok = & $py @pyArgs -c "import sys; print(sys.version_info >= (3, 9))" 2>$null
if ("$ok".Trim() -ne "True") {
    Write-Host "The Python found is missing or older than 3.9. Install 3.9+ from https://www.python.org/downloads/"
    exit 1
}

$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Write-Host "Creating .venv ..."
    & $py @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "Could not create the environment."; exit 1 }
}

Write-Host "Installing packages ..."
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet -r requirements_signal.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Package installation failed."; exit 1 }

$runArgs = @("run_signal_rebuild.py", "--sessions", $Sessions)
if ($MaxRecordings -gt 0) { $runArgs += @("--max-recordings", "$MaxRecordings") }
if ($DataDir -ne "") { $runArgs += @("--data-dir", $DataDir) }
if ($DeleteAfter) { $runArgs += "--delete-after" }
if ($StatsOnly) { $runArgs += "--stats-only" }

& $vpy @runArgs
if ($LASTEXITCODE -ne 0) { Write-Host "The analysis stopped with an error; run the same command again to resume."; exit 1 }
Write-Host ""
Write-Host "Done. Send back results\signal_rebuild.json"
