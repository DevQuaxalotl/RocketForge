# Locates the VS2022 C++ build tools. Dot-source it: . (Join-Path $PSScriptRoot "VsDevEnv.ps1"), or "..\build\VsDevEnv.ps1" from a sibling folder.
#
# Any VS2022 product counts (Community, Professional, Enterprise or Build Tools), on any drive,
# as long as it has the x64 C++ compiler. Newer Visual Studio versions are skipped
# because the CUDA 12.6 toolkit RocketForge builds with rejects their compiler.

$vsInstallerDir = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"

# VsDevCmd.bat runs a bare vswhere.exe from the Installer folder,
# which fails with "not recognized" when NoDefaultCurrentDirectoryInExePath is set.
if (Test-Path $vsInstallerDir) { $env:PATH = "$vsInstallerDir;$env:PATH" }

# Returns the newest matching VS2022 installation path, or $null when there is none.
function Find-Vs2022CppInstall {
    $vswhere = Join-Path $vsInstallerDir "vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $null }

    # Without -products *, vswhere skips Build Tools installs.
    $paths = @(& $vswhere -nologo -latest -products * -version "[17.0,18.0)" `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
    if ($LASTEXITCODE -ne 0 -or $paths.Count -eq 0 -or -not $paths[0]) { return $null }
    return $paths[0]
}

# Returns the path to vcvars64.bat, or writes an error naming the fix and exits.
function Get-VcvarsPath {
    $installPath = Find-Vs2022CppInstall
    if ($installPath) {
        $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvars) { return $vcvars }
    }

    Write-Error ("Could not find Visual Studio 2022 with the C++ build tools. " +
        "Run tools\setup\install_build_tools.ps1, or add the 'Desktop development with C++' workload in the Visual Studio Installer.")
    exit 1
}
