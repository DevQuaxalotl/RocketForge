# Installs Build Tools for Visual Studio 2022 with the C++ workload: the compiler RocketForge builds with on Windows.
# This is the compiler, CMake and Ninja without the Visual Studio IDE. Nothing is installed without a "yes".
#
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_build_tools.ps1         # asks before installing
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_build_tools.ps1 -Yes    # no prompt, for scripted use
#
# Does nothing when any VS2022 edition with the C++ tools is already installed.
# The installer asks for administrator rights and shows its own progress window.

param(
    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "..\build\VsDevEnv.ps1")

$existing = Find-Vs2022CppInstall
if ($existing) {
    Write-Host "Visual Studio 2022 with the C++ build tools is already installed at $existing."
    exit 0
}

# A VS2022 install without the C++ workload is better extended than duplicated by a second install.
$vswhere = Join-Path $vsInstallerDir "vswhere.exe"
if (Test-Path $vswhere) {
    $withoutCpp = @(& $vswhere -nologo -latest -products * -version "[17.0,18.0)" -property installationPath)
    if ($withoutCpp.Count -gt 0 -and $withoutCpp[0]) {
        Write-Host "Visual Studio 2022 is installed at $($withoutCpp[0]) but lacks the C++ build tools."
        Write-Host "Open the Visual Studio Installer, choose Modify, and add the 'Desktop development with C++' workload."
        exit 1
    }
}

$bootstrapperUrl = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
$installArgs = @(
    "--add", "Microsoft.VisualStudio.Workload.VCTools",
    "--add", "Microsoft.VisualStudio.Component.VC.CMake.Project",
    "--includeRecommended", "--passive", "--wait", "--norestart"
)

Write-Host "This will install:"
Write-Host "  - Build Tools for Visual Studio 2022, from $bootstrapperUrl"
Write-Host "    with the C++ workload (MSVC, Windows SDK, CMake, Ninja)."
Write-Host "  This needs administrator rights and several GB of disk space."

if (-not $Yes) {
    $answer = Read-Host "Install the build tools now? [y/N]"
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host "Cancelled. Nothing was installed."
        exit 1
    }
}

$installer = Join-Path $env:TEMP "vs_BuildTools.exe"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $bootstrapperUrl -OutFile $installer -UseBasicParsing

# The bootstrapper is a GUI program, so the call operator would return before it finishes.
$process = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
$rebootRequired = $process.ExitCode -eq 3010
if ($process.ExitCode -ne 0 -and -not $rebootRequired) {
    Write-Error "The Visual Studio installer failed (exit $($process.ExitCode))."
    exit $process.ExitCode
}

$installed = Find-Vs2022CppInstall
if (-not $installed) {
    Write-Error "The installer finished but vswhere still finds no VS2022 C++ build tools."
    exit 1
}

Write-Host "Build tools installed at $installed."
if ($rebootRequired) { Write-Host "Restart Windows before building." }
Write-Host "RocketForge also needs the CUDA 12.6 toolkit and libtorch; see README.md."
