# Waits for a running ForgeGUI to exit, re-configures and rebuilds the forge tree, then starts the GUI again.
# ForgeGUI launches this after "New Bot", because a new bot is compiled in and only shows up after a re-configure and rebuild.
# It hands itself off to a new process outside the GUI's job (see -Detached), which shows the build in its own console window.
# On failure that window stays open with the error, since the script runs detached from any terminal.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\tools\build\rebuild_and_relaunch.ps1 -WaitPid <gui pid> -Relaunch <path to ForgeGUI.exe>

param(
    [Parameter(Mandatory = $true)][int]$WaitPid,
    [Parameter(Mandatory = $true)][string]$Relaunch,
    [switch]$Detached
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# A GUI launched from an IDE runs in a kill-on-close job object, and every child it starts joins that job, so a plain child
# would die with the GUI. WMI creates the process itself, outside any job; this first instance only hands off and exits.
if (-not $Detached) {
    $commandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" +
        " -WaitPid $WaitPid -Relaunch `"$Relaunch`" -Detached"
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]1 }
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $commandLine
        CurrentDirectory = $repoRoot
        ProcessStartupInformation = $startup
    }
    if ($result.ReturnValue -ne 0) {
        Write-Error "Could not start the rebuild (Win32_Process.Create returned $($result.ReturnValue))."
        exit 1
    }
    exit 0
}

# Keeps the window open so the error can be read; skipped when input is redirected so automated runs don't hang.
function Wait-BeforeClose {
    if (-not [Console]::IsInputRedirected) { Read-Host "Press Enter to close" | Out-Null }
}

try {
    . (Join-Path $PSScriptRoot "VsDevEnv.ps1")
    $vcvarsPath = Get-VcvarsPath

    function Invoke-VsCmd {
        param([string]$Command)
        cmd /c "`"$vcvarsPath`" && $Command"
        if ($LASTEXITCODE -ne 0) { throw "'$Command' failed (exit $LASTEXITCODE)" }
    }

    Write-Host "==> Waiting for RocketForge (PID $WaitPid) to exit..." -ForegroundColor Cyan
    # The GUI exe stays locked until its process ends, so the build can't start earlier.
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue

    Push-Location $repoRoot
    try {
        Write-Host "`n==> Configuring forge preset..." -ForegroundColor Cyan
        Invoke-VsCmd "cmake --preset forge"

        Write-Host "`n==> Building ForgeGUI + ForgeConsoleTrainer..." -ForegroundColor Cyan
        Invoke-VsCmd "cmake --build --preset build-gui"
    } finally {
        Pop-Location
    }

    Write-Host "`n==> Build succeeded. Starting RocketForge..." -ForegroundColor Green
    $builtTree = Join-Path $repoRoot "out\build\forge"
    if (-not ([System.IO.Path]::GetFullPath($Relaunch)).StartsWith($builtTree, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Host "WARNING: $Relaunch is not under $builtTree, so the relaunched exe is not the one just rebuilt." -ForegroundColor Yellow
    }
    Start-Process -FilePath $Relaunch
} catch {
    Write-Host "`nRebuild failed: $($_.Exception.Message)" -ForegroundColor Red
    Wait-BeforeClose
    exit 1
}
