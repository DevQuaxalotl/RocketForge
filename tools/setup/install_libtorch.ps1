# Installs the libtorch build that RocketForge compiles and links against. Nothing is installed without a "yes".
#
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_libtorch.ps1                        # asks before installing to %LOCALAPPDATA%\libtorch
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_libtorch.ps1 -Yes                   # no prompt, for scripted use
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_libtorch.ps1 -Variant Cpu           # CPU-only build, even with an NVIDIA GPU
#   powershell -ExecutionPolicy Bypass -File .\tools\setup\install_libtorch.ps1 -InstallDir D:\libtorch # elsewhere; then configure with -DLIBTORCH_ROOT
#
# Fetches the official release archive from download.pytorch.org.
# -Variant Auto picks the CUDA 12.6 build when an NVIDIA GPU is present and the CPU build otherwise.
# The CUDA build needs the CUDA 12.6 toolkit to configure; the CPU build needs no CUDA at all and trains on the CPU.
# The version must stay in step with the Dockerfile's libtorch (same version, cu129 there).

param(
    [switch]$Yes,
    [ValidateSet("Auto", "Cuda", "Cpu")]
    [string]$Variant = "Auto",
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "libtorch")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$torchVersion = "2.13.0"
$builds = @{
    Cuda = @{ Tag = "cu126"; DownloadGB = 5.2; InstalledGB = 7.7; Description = "CUDA 12.6" }
    Cpu  = @{ Tag = "cpu";   DownloadGB = 0.2; InstalledGB = 0.5; Description = "CPU only" }
}

if ($Variant -eq "Auto") {
    $nvidiaGpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
    if ($nvidiaGpu) {
        $Variant = "Cuda"
        Write-Host "Found $($nvidiaGpu.Name); using the CUDA build of libtorch (-Variant Cpu overrides this)."
    } else {
        $Variant = "Cpu"
        Write-Host "No NVIDIA GPU found; using the CPU-only build of libtorch, so training runs on the CPU."
    }
}
$build = $builds[$Variant]
$fullVersion = "$torchVersion+$($build.Tag)"
$downloadUrl = "https://download.pytorch.org/libtorch/$($build.Tag)/libtorch-win-shared-with-deps-$torchVersion%2B$($build.Tag).zip"

$InstallDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($InstallDir)
$torchConfig = Join-Path $InstallDir "share\cmake\Torch\TorchConfig.cmake"
$versionFile = Join-Path $InstallDir "build-version"

if (Test-Path $torchConfig) {
    $installedVersion = if (Test-Path $versionFile) { (Get-Content $versionFile -Raw).Trim() } else { "unknown" }
    if ($installedVersion -eq $fullVersion) {
        Write-Host "libtorch $fullVersion is already installed at $InstallDir."
        exit 0
    }
    Write-Error ("libtorch $installedVersion is installed at $InstallDir, but this would install $fullVersion. " +
        "Remove that folder and re-run, or pass -InstallDir to install elsewhere.")
    exit 1
}
if ((Test-Path $InstallDir) -and (Get-ChildItem $InstallDir -Force | Select-Object -First 1)) {
    Write-Error "$InstallDir exists but holds no libtorch. Empty or remove it, or pass -InstallDir to install elsewhere."
    exit 1
}

# Download and extraction happen on the install drive, so the final move is a rename rather than a multi-GB copy.
$installParent = Split-Path $InstallDir -Parent
New-Item -ItemType Directory -Force $installParent | Out-Null
$freeGB = (Get-PSDrive -Name (Split-Path $InstallDir -Qualifier).TrimEnd(':')).Free / 1GB
$neededGB = $build.DownloadGB + $build.InstalledGB
if ($freeGB -lt $neededGB) {
    Write-Error ("Installing needs about $neededGB GB free on $(Split-Path $InstallDir -Qualifier) " +
        "(the $($build.DownloadGB) GB archive plus $($build.InstalledGB) GB extracted); only $([math]::Round($freeGB, 1)) GB is free.")
    exit 1
}

Write-Host "This will install:"
Write-Host "  - libtorch $fullVersion (shared, $($build.Description))"
Write-Host "    from $downloadUrl"
Write-Host "    into $InstallDir (a $($build.DownloadGB) GB download, $($build.InstalledGB) GB once extracted)"

if (-not $Yes) {
    $answer = Read-Host "Install libtorch now? [y/N]"
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host "Cancelled. Nothing was installed."
        exit 1
    }
}

# A fixed archive name lets an interrupted download resume on the next run (curl -C -).
$archive = Join-Path $installParent "libtorch-$torchVersion-$($build.Tag)-download.zip"
$staging = Join-Path $installParent "rocketforge-libtorch-$([guid]::NewGuid().ToString('N'))"

# Windows' own curl.exe and tar.exe: Invoke-WebRequest and Expand-Archive take far longer on an archive this size.
& "$env:SystemRoot\System32\curl.exe" --fail --location --retry 3 --continue-at - --output $archive $downloadUrl
if ($LASTEXITCODE -ne 0) {
    Write-Error "Downloading libtorch failed (curl exit $LASTEXITCODE). Re-run to resume the download."
    exit $LASTEXITCODE
}

try {
    New-Item -ItemType Directory $staging | Out-Null
    Write-Host "Extracting..."
    & "$env:SystemRoot\System32\tar.exe" -xf $archive -C $staging
    if ($LASTEXITCODE -ne 0) {
        # A corrupt archive would otherwise be resumed (and kept) on every re-run.
        Remove-Item -Force $archive
        Write-Error "Extracting libtorch failed (tar exit $LASTEXITCODE). The archive was deleted; re-run to download it again."
        exit $LASTEXITCODE
    }

    $extracted = Join-Path $staging "libtorch"
    if (-not (Test-Path (Join-Path $extracted "share\cmake\Torch\TorchConfig.cmake"))) {
        Write-Error "The archive did not contain libtorch\share\cmake\Torch\TorchConfig.cmake."
        exit 1
    }
    if (Test-Path $InstallDir) { Remove-Item $InstallDir }
    Move-Item $extracted $InstallDir
} finally {
    if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
}
Remove-Item -Force $archive

Write-Host "libtorch $fullVersion installed at $InstallDir."
if ($InstallDir -ne (Join-Path $env:LOCALAPPDATA "libtorch")) {
    Write-Host "This is not the default %LOCALAPPDATA%\libtorch path. Configure with -DLIBTORCH_ROOT=$InstallDir."
}
if ($Variant -eq "Cuda" -and -not $env:CUDA_PATH_V12_6) {
    Write-Host ("The CUDA build also needs the CUDA 12.6 toolkit to configure, and it is not installed: " +
        "https://developer.nvidia.com/cuda-12-6-0-download-archive")
}
