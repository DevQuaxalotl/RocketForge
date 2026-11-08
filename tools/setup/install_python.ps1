# Installs the Python that RocketForge embeds for wandb metrics and the debug viewer. Nothing is installed without a "yes".
#
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_python.ps1         # asks before installing
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_python.ps1 -Yes    # no prompt, for scripted use
#
# The configure links against Python's headers and import library, and training imports wandb from that same install.
# When a usable Python is already on PATH this only adds wandb if it is missing.
# Otherwise it installs Python for the current user from python.org (no administrator rights) and adds it to PATH.

param(
    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonVersion = "3.12.10"
$installerUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe"
$installDir = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"
# The bundled pybind11 (2.12) supports Python 3.9 through 3.12.
$minMinor = 9
$maxMinor = 12

function Confirm-Install([string]$question) {
    if ($Yes) { return }
    $answer = Read-Host "$question [y/N]"
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host "Cancelled. Nothing was installed."
        exit 1
    }
}

function Install-Wandb([string]$python) {
    & $python -m pip install --quiet --disable-pip-version-check wandb
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install wandb failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
    Write-Host "wandb installed. Run 'wandb login' once before training with metrics, or train with --no-metrics."
}

# Runs Python code and returns its stdout, or $null when it fails.
# Windows PowerShell turns a native command's redirected stderr into error records, which "Stop" would throw on.
function Invoke-PythonProbe([string]$python, [string]$code) {
    $ErrorActionPreference = "Continue"
    $output = & $python -c $code 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return "$output"
}

# The Microsoft Store "python" alias is found on PATH but fails to run, so it reads as no Python.
$existing = Get-Command python -ErrorAction SilentlyContinue
$existingMinor = $null
if ($existing) {
    $output = Invoke-PythonProbe $existing.Source "import sys; print(sys.version_info[0], sys.version_info[1])"
    if ($output -match '^3 (\d+)$') { $existingMinor = [int]$Matches[1] }
}

if ($null -ne $existingMinor -and $existingMinor -ge $minMinor -and $existingMinor -le $maxMinor) {
    Write-Host "Python 3.$existingMinor is already installed at $($existing.Source)."
    if ($null -ne (Invoke-PythonProbe $existing.Source "import wandb")) {
        Write-Host "wandb is already installed."
        exit 0
    }
    Write-Host "This will install wandb (training metrics) into that Python with pip."
    Confirm-Install "Install wandb now?"
    Install-Wandb $existing.Source
    exit 0
}

if ($null -ne $existingMinor) {
    Write-Host ("Found Python 3.$existingMinor at $($existing.Source), outside the supported 3.$minMinor-3.$maxMinor; " +
        "installing Python $pythonVersion alongside it.")
}

Write-Host "This will install:"
Write-Host "  - Python $pythonVersion for the current user, from $installerUrl"
Write-Host "    into $installDir, added to PATH (about 100 MB)"
Write-Host "  - wandb (training metrics), with pip"

Confirm-Install "Install Python now?"

$installer = Join-Path $env:TEMP "python-$pythonVersion-amd64.exe"
& "$env:SystemRoot\System32\curl.exe" --fail --location --retry 3 --output $installer $installerUrl
if ($LASTEXITCODE -ne 0) { Write-Error "Downloading the Python installer failed (curl exit $LASTEXITCODE)"; exit $LASTEXITCODE }

try {
    $installArgs = @(
        "/quiet", "InstallAllUsers=0", "PrependPath=1", "TargetDir=`"$installDir`"",
        "Include_dev=1", "Include_pip=1", "Include_launcher=1", "InstallLauncherAllUsers=0",
        "Include_test=0", "Include_doc=0"
    )
    $process = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
    if ($process.ExitCode -ne 0) { Write-Error "The Python installer failed (exit $($process.ExitCode))"; exit $process.ExitCode }
} finally {
    Remove-Item -Force $installer -ErrorAction SilentlyContinue
}

$installedPython = Join-Path $installDir "python.exe"
if (-not (Test-Path $installedPython)) {
    Write-Error "The Python installer reported success but $installedPython is missing."
    exit 1
}
Install-Wandb $installedPython

Write-Host "Python $pythonVersion installed at $installDir."
Write-Host "Open a new terminal so PATH includes it before configuring RocketForge."
