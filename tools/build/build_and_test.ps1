# Build ForgeTrainingTests and (optionally) ForgeGUI + ForgeConsoleTrainer in the forge preset's tree, then run the tests.
# Run from any directory — the script locates the repo root automatically.
# Requires Visual Studio 2022 (any edition, or Build Tools) with the C++ workload: tools/setup/install_build_tools.ps1 installs it.
# GUI build additionally requires Qt 6.10.2 at C:/Qt/6.10.2/msvc2022_64 (or wherever QTDIR points): tools/setup/install_qt.ps1 installs it.
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1                          # configure + build tests + run tests
#   powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -SkipConfig              # skip the forge configure (faster when no new files added)
#   powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -BuildGui                # also build ForgeGUI and its ForgeConsoleTrainer
#   powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -BuildGui -SkipConfig    # build GUI + skip the configure
#   powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -Filter GAE              # run only tests whose name contains "GAE"
#   powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -CTest                   # run through CTest (per-test process + timeout; slower, catches hangs)
#   powershell -ExecutionPolicy Bypass -File .\tools\build\build_and_test.ps1 -CTest -Filter "^GAE\."  # with -CTest, -Filter is a CTest regex (-R)

param(
    [switch]$SkipConfig,
    [switch]$BuildGui,
    [string]$Filter = "",
    [switch]$CTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

. (Join-Path $PSScriptRoot "VsDevEnv.ps1")
$vcvarsPath = Get-VcvarsPath

$buildDir = Join-Path $repoRoot "out\build\forge"

# Run a command inside the VS2022 x64 developer environment.
# Output goes directly to the console; check $LASTEXITCODE in the caller.
function Invoke-VsCmd {
    param([string]$Command)
    cmd /c "`"$vcvarsPath`" && $Command"
}

# Prints the engines line CMake wrote at configure time, so a build that skips the configure still reports it.
function Write-EnginesLine {
    $enginesFile = Join-Path $buildDir "engines.txt"
    if (Test-Path $enginesFile) { Write-Host (Get-Content $enginesFile -TotalCount 1) -ForegroundColor Cyan }
}

Push-Location $repoRoot
try {
    if (-not $SkipConfig) {
        Write-Host "`n==> Configuring forge preset..." -ForegroundColor Cyan
        Invoke-VsCmd "cmake --preset forge"
        if ($LASTEXITCODE -ne 0) { Write-Error "cmake configure (forge) failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
    }
    Write-EnginesLine

    # --- Tests build ---

    Write-Host "`n==> Building ForgeTrainingTests..." -ForegroundColor Cyan
    Invoke-VsCmd "cmake --build --preset build-tests --target ForgeTrainingTests"
    if ($LASTEXITCODE -ne 0) { Write-Error "Build (ForgeTrainingTests) failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

    $testExe = Join-Path $buildDir "ForgeTrainingTests.exe"
    if (-not (Test-Path $testExe)) {
        Write-Error "Test executable not found: $testExe"
        exit 1
    }

    Write-Host "`n==> Running tests..." -ForegroundColor Cyan
    if ($CTest) {
        # The label limits the run to the target built above: bot test exes share this build tree but aren't built here.
        # The tests preset fails the run when -R matches nothing.
        $ctestCommand = "ctest --preset tests -L ForgeTrainingTests"
        if ($Filter) { $ctestCommand += " -R `"$Filter`"" }
        Invoke-VsCmd $ctestCommand
    } else {
        # Plain text matches as a substring of Suite.Test; text with gtest pattern syntax (*, ?, :, or a leading -) passes through as a gtest filter.
        $gtestFilter = if (-not $Filter) { "" } elseif ($Filter -match '[\*\?:]|^-') { $Filter } else { "*$Filter*" }

        $gtestArgs = @("--gtest_brief=1")
        if ($gtestFilter) {
            $selected = & $testExe --gtest_list_tests "--gtest_filter=$gtestFilter" | Where-Object { $_ -match '^\s{2}\S' }
            if (-not $selected) {
                Write-Error "Filter '$Filter' matched no tests."
                exit 1
            }

            $gtestArgs += "--gtest_filter=$gtestFilter"
        }

        & $testExe @gtestArgs
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # --- GUI build (optional) ---

    if ($BuildGui) {
        if (-not (Test-Path (Join-Path $buildDir "ForgeGUI"))) {
            Write-Error "ForgeGUI isn't in this build because the configure didn't find Qt. Install it with tools/setup/install_qt.ps1 or set QTDIR, then rerun without -SkipConfig."
            exit 1
        }

        # When new files are added via GLOB, the AUTOMOC cache can go stale even after a fresh
        # configure — new Q_OBJECT classes compile fine but their moc symbols are never linked,
        # producing cryptic LNK2001 errors. Clearing the autogen dir forces a full AUTOMOC
        # re-scan (~1s overhead) and only runs on full-configure builds where new files may exist.
        if (-not $SkipConfig) {
            $autogenDir = Join-Path $buildDir "ForgeGUI\ForgeGUI_autogen"
            if (Test-Path $autogenDir) {
                Write-Host "`n==> Clearing AUTOMOC cache (new files may have been added)..." -ForegroundColor Cyan
                Remove-Item -Recurse -Force $autogenDir
            }
        }

        # ForgeConsoleTrainer shares the GigaLearn DLL rebuilt here,
        # so a stale exe can pass old struct layouts across the DLL boundary and crash at runtime.
        Write-Host "`n==> Building ForgeGUI + ForgeConsoleTrainer..." -ForegroundColor Cyan
        Invoke-VsCmd "cmake --build --preset build-gui --target ForgeGUI ForgeConsoleTrainer"
        if ($LASTEXITCODE -ne 0) { Write-Error "Build (ForgeGUI + ForgeConsoleTrainer) failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

        Write-Host "`n==> ForgeGUI + ForgeConsoleTrainer build succeeded." -ForegroundColor Green
    }
} finally {
    Pop-Location
}
