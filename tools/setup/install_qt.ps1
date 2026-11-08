# Installs the Qt that the RocketForge GUI and its Qt SQL tests build against. Nothing is installed without a "yes".
#
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_qt.ps1                    # asks before installing to C:\Qt
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_qt.ps1 -Yes               # no prompt, for scripted use
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_qt.ps1 -OutputDir D:\Qt   # elsewhere; then set QTDIR
#
# Downloads official Qt binaries with aqtinstall, which needs Python 3.9 or later but no Qt account.
# Only the qtbase modules ForgeGUI links (Widgets, Sql) are fetched, plus what windeployqt uses and copies next to the exe.
# aqtinstall runs from a temporary virtual environment, so the system Python is left untouched.

param(
    [switch]$Yes,
    [string]$OutputDir = "C:\Qt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$qtVersion = "6.10.2"
$qtArch = "win64_msvc2022_64"
$aqtVersion = "3.3.0"
# windeployqt needs lconvert from qttools to bundle the translations.
$archives = @("qtbase", "qttools", "qttranslations", "d3dcompiler_47", "opengl32sw")

$OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)

# aqt names the install folder after the arch without its "win64_" prefix.
$qtPrefix = Join-Path $OutputDir "$qtVersion\msvc2022_64"
$qtConfig = Join-Path $qtPrefix "lib\cmake\Qt6Widgets\Qt6WidgetsConfig.cmake"

if (Test-Path $qtConfig) {
    Write-Host "Qt $qtVersion is already installed at $qtPrefix."
    exit 0
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python 3.9 or later is required to run aqtinstall. Install it from https://www.python.org/downloads/ and re-run."
    exit 1
}

# The Microsoft Store "python" alias is found on PATH but fails here, so this also catches it.
& $python.Source -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "aqtinstall needs Python 3.9 or later; '$($python.Source)' is older or not a working Python."
    exit 1
}

Write-Host "This will install:"
Write-Host "  - Qt $qtVersion (${qtArch}: $($archives -join ', '))"
Write-Host "    into $qtPrefix (about 430 MB)"
Write-Host "  using aqtinstall $aqtVersion in a temporary Python environment."

if (-not $Yes) {
    $answer = Read-Host "Install Qt now? [y/N]"
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host "Cancelled. Nothing was installed."
        exit 1
    }
}

$venv = Join-Path $env:TEMP "rocketforge-aqt-$([guid]::NewGuid().ToString('N'))"
try {
    & $python.Source -m venv $venv
    if ($LASTEXITCODE -ne 0) { Write-Error "Creating the Python virtual environment failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

    $venvPython = Join-Path $venv "Scripts\python.exe"
    & $venvPython -m pip install --quiet --disable-pip-version-check "aqtinstall==$aqtVersion"
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install aqtinstall failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

    # aqt writes aqtinstall.log to the working directory; running from the venv discards it with the venv.
    Push-Location $venv
    try {
        & $venvPython -m aqt install-qt windows desktop $qtVersion $qtArch --outputdir $OutputDir --archives @archives
        if ($LASTEXITCODE -ne 0) { Write-Error "aqt install-qt failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
} finally {
    if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
}

if (-not (Test-Path $qtConfig)) {
    Write-Error "aqt reported success but $qtConfig is missing."
    exit 1
}

Write-Host "Qt $qtVersion installed at $qtPrefix."
if ($qtPrefix -ne "C:\Qt\$qtVersion\msvc2022_64") {
    Write-Host "This is not the default C:\Qt path. Set QTDIR=$qtPrefix, which the CMake configure reads."
}
Write-Host "Build the GUI with: powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -BuildGui"
