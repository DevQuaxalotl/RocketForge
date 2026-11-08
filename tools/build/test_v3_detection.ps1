# Checks that CMake detects whether RocketSim v3 can be built and reports why not, by configuring the forge preset (console only) in throwaway build dirs.
# Run from any directory; needs Visual Studio 2022 with the C++ workload.
#
#   powershell -ExecutionPolicy Bypass -File .\tools\build\test_v3_detection.ps1
#
# Two test-only cache options in cmake/DetectRocketSimV3.cmake make detection behave as if a dependency were missing,
# so the checks pass on any machine whatever is installed:
#   ROCKETFORGE_V3_TEST_HIDE_CARGO=ON      act as if cargo is not installed
#   ROCKETFORGE_V3_TEST_HIDE_SUBMODULE=ON  act as if the v3 submodule is empty and cannot be fetched
#
# The build dirs (out/build/v3-detect-<case>) are deleted when the script ends.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

. (Join-Path $PSScriptRoot "VsDevEnv.ps1")
$vcvarsPath = Get-VcvarsPath

# Runs cmake inside the x64 developer environment and returns its exit code and combined output.
function Invoke-Configure {
    param([string]$BuildDir, [string[]]$Definitions)
    $arguments = (@("--preset", "forge", "-B", "`"$BuildDir`"", "-DBUILD_GUI=OFF", "-DROCKETFORGE_BUILD_TESTS=OFF") + $Definitions) -join " "
    $output = & cmd /c "`"$vcvarsPath`" >nul && cmake $arguments 2>&1" | Out-String
    $exitCode = $LASTEXITCODE

    # CMake wraps long warning and error text, so compare with whitespace collapsed.
    return [pscustomobject]@{ ExitCode = $exitCode; Output = $output; FlatOutput = ($output -replace "\s+", " ") }
}

$headerRelativePath = "ForgeTraining\RLGym\Sim\generated\EngineConfig.h"
$submoduleReason = "RocketSim v3 isn't built: its source (the ForgeTraining/RLGym/RocketSimV3/upstream submodule) is missing."
$noRustReason = "RocketSim v3 isn't built: Rust isn't installed."

$cases = @(
    [pscustomobject]@{ Name = "hidden-cargo"; Definitions = @("-DROCKETFORGE_V3_TEST_HIDE_CARGO=ON"); ExpectConfigure = $true; ExpectedReason = $noRustReason },
    [pscustomobject]@{ Name = "hidden-submodule"; Definitions = @("-DROCKETFORGE_V3_TEST_HIDE_SUBMODULE=ON"); ExpectConfigure = $true; ExpectedReason = $submoduleReason },
    [pscustomobject]@{ Name = "forced-on"; Definitions = @("-DROCKETFORGE_ENABLE_V3=ON", "-DROCKETFORGE_V3_TEST_HIDE_CARGO=ON"); ExpectConfigure = $false; ExpectedReason = $noRustReason }
)

$buildRoot = Join-Path $repoRoot "out\build"
$failures = 0

Push-Location $repoRoot
try {
    foreach ($case in $cases) {
        Write-Host "`n==> Case $($case.Name)" -ForegroundColor Cyan
        $buildDir = Join-Path $buildRoot "v3-detect-$($case.Name)"
        if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }

        $result = Invoke-Configure -BuildDir $buildDir -Definitions $case.Definitions
        $problems = @()

        if ($case.ExpectConfigure) {
            if ($result.ExitCode -ne 0) { $problems += "configure failed (exit $($result.ExitCode))" }
            if ($result.FlatOutput -notmatch [regex]::Escape($case.ExpectedReason)) { $problems += "configure output lacks the expected reason" }

            $headerPath = Join-Path $buildDir $headerRelativePath
            if (-not (Test-Path $headerPath)) {
                $problems += "generated header is missing"
            } else {
                $header = Get-Content -Raw $headerPath
                if ($header -notmatch "RF_ENGINE_V3_AVAILABLE 0") { $problems += "header does not report v3 as unavailable" }
                if ($header -notmatch [regex]::Escape($case.ExpectedReason)) { $problems += "header lacks the expected reason" }
            }
        } else {
            if ($result.ExitCode -eq 0) { $problems += "configure succeeded but should have failed" }
            if ($result.FlatOutput -notmatch [regex]::Escape($case.ExpectedReason)) { $problems += "error output lacks the expected reason" }
        }

        if ($problems.Count -eq 0) {
            Write-Host "PASS $($case.Name)" -ForegroundColor Green
        } else {
            $failures++
            Write-Host "FAIL $($case.Name): $($problems -join '; ')" -ForegroundColor Red
            Write-Host $result.Output
        }
    }
} finally {
    foreach ($case in $cases) {
        $buildDir = Join-Path $buildRoot "v3-detect-$($case.Name)"
        if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
    }
    Pop-Location
}

Write-Host "`n$($cases.Count - $failures)/$($cases.Count) cases passed."
if ($failures -gt 0) { exit 1 }
