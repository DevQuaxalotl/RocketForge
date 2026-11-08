# Installs the Rust toolchain RocketForge needs to build RocketSim v3. Nothing is installed without a "yes".
#
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_rust.ps1         # asks before installing
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_rust.ps1 -Yes    # no prompt, for scripted use
#
# When Rust is already installed this only adds the pinned toolchain. It relies on the VS2022 C++ build tools
# RocketForge already requires and does not install them.

param(
    [switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$toolchain = "1.99.0"

if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host "Rust is already installed; adding toolchain $toolchain if it is missing."
    & rustup toolchain install $toolchain --profile minimal
    if ($LASTEXITCODE -ne 0) { Write-Error "rustup toolchain install failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
    exit 0
}

Write-Host "This will install:"
Write-Host "  - rustup, from https://win.rustup.rs/x86_64"
Write-Host "  - Rust $toolchain (minimal profile)"
Write-Host "  into $env:USERPROFILE\.rustup and $env:USERPROFILE\.cargo (about 300 MB)."

if (-not $Yes) {
    $answer = Read-Host "Install Rust now? [y/N]"
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host "Cancelled. Nothing was installed."
        exit 1
    }
}

$installer = Join-Path $env:TEMP "rustup-init.exe"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $installer -UseBasicParsing

& $installer -y --profile minimal --default-toolchain $toolchain
if ($LASTEXITCODE -ne 0) { Write-Error "rustup-init failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

# rustup-init updates the user PATH, not this process's.
$cargo = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
& $cargo --version
if ($LASTEXITCODE -ne 0) { Write-Error "cargo --version failed after the install (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "Rust installed. Rebuild RocketForge (tools/build/build_and_test.ps1 -BuildGui) to enable RocketSim v3."
