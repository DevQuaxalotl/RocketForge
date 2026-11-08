<#
.SYNOPSIS
  Build the RocketForge training image and push it to a private Docker Hub repo.

.DESCRIPTION
  Runs `docker build`, tags the result as <User>/<Repo>:<Tag> and (optionally)
  :latest, then pushes. The Docker build is fully containerized Linux and does
  NOT require the VS2022 x64 dev environment that native CMake builds need.

  A build failure aborts before any tag/push, so nothing broken reaches a pod.

.PARAMETER User
  Docker Hub username (the image namespace). Defaults to $env:DOCKERHUB_USER.

.PARAMETER Repo
  Repository name. Default: rocketforge.

.PARAMETER Tag
  Image tag. Default: <BuildFolder>-<StartSegment> (e.g. ExampleBuild-v1.01),
  resolved from `ForgeConsoleTrainer build-info` for the selected agent/build.
  Falls back to the timestamp yyyyMMdd-HHmm when the console isn't built or
  resolution fails. An explicit -Tag always wins.

.PARAMETER Build
  Optional build name or Models folder name to pass to push_models.ps1 when
  seeding R2 after the image push. This only changes the R2 upload; the image
  still trains the build from its baked forge_settings.json.

.PARAMETER Agent
  Agent whose build supplies the auto tag and whose models are pushed after the image.
  This does not change what the image trains: it still trains the agent selected in its baked forge_app_state.json.
  Default: the GUI-selected agent in forge_app_state.json.

.PARAMETER PlainProgress
  Forwarded to push_models.ps1: one-line rclone stats instead of -P's in-place redraw.

.PARAMETER SkipLatest
  If set, do not also tag/push :latest.

.EXAMPLE
  ./tools/remote/push_training_image.ps1 -User myname -Tag example-reward-v2
#>
param(
    [string]$User = $env:DOCKERHUB_USER,
    [string]$Repo = "rocketforge",
    [string]$Tag  = (Get-Date -Format "yyyyMMdd-HHmm"),
    [string]$Agent = "",
    [string]$Build = "",
    [switch]$SkipLatest,
    [switch]$SkipModels,
    [switch]$PlainProgress
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($User)) {
    throw "Docker Hub user not set. Pass -User <name> or set `$env:DOCKERHUB_USER."
}

# Repo root = parent of this script's tools/ directory.
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $RepoRoot
try {
    $local  = "rocketforge"
    $remote = "$User/$Repo"

    # When -Tag wasn't explicitly passed, derive it from the selected build +
    # starting segment (build-info's TAG). Falls back to the timestamp default if
    # the console isn't built or resolution fails.
    if (-not $PSBoundParameters.ContainsKey('Tag')) {
        # Newest binary wins: a stale preset's exe may resolve the tag from outdated code.
        $exe = Get-ChildItem -Path (Join-Path $RepoRoot "out\build") -Recurse -Filter "ForgeConsoleTrainer.exe" -ErrorAction SilentlyContinue |
               Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($exe) {
            $buildInfoArgs = @()
            if (-not [string]::IsNullOrWhiteSpace($Agent)) { $buildInfoArgs += @("--agent", $Agent) }

            $info = @{}
            & $exe.FullName build-info @buildInfoArgs | ForEach-Object {
                if ($_ -match '^([A-Z_]+)=(.*)$') { $info[$matches[1]] = $matches[2] }
            }
            if (-not [string]::IsNullOrWhiteSpace($info['TAG'])) {
                $Tag = $info['TAG']
                Write-Host "==> Auto-tag from selected build: $Tag" -ForegroundColor Cyan
            } else {
                Write-Warning "build-info returned no TAG; using timestamp tag $Tag."
            }
        } else {
            Write-Warning "ForgeConsoleTrainer.exe not found; using timestamp tag $Tag."
        }
    }

    # The build context drops .git, so the image can't fetch the RocketSim v3 submodule itself.
    Write-Host "==> Checking out the RocketSim v3 submodule ..." -ForegroundColor Cyan
    git submodule update --init ForgeTraining/RLGym/RocketSimV3/upstream
    if ($LASTEXITCODE -ne 0) { throw "git submodule update failed (exit $LASTEXITCODE)." }

    Write-Host "==> Building $local from $RepoRoot ..." -ForegroundColor Cyan
    docker build -t $local .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed (exit $LASTEXITCODE)." }

    Write-Host "==> Tagging and pushing ${remote}:$Tag" -ForegroundColor Cyan
    docker tag $local "${remote}:$Tag"
    if ($LASTEXITCODE -ne 0) { throw "docker tag failed." }
    docker push "${remote}:$Tag"
    if ($LASTEXITCODE -ne 0) { throw "docker push failed - are you logged in? Run: docker login -u $User" }

    if (-not $SkipLatest) {
        Write-Host "==> Tagging and pushing ${remote}:latest" -ForegroundColor Cyan
        docker tag $local "${remote}:latest"
        if ($LASTEXITCODE -ne 0) { throw "docker tag :latest failed." }
        docker push "${remote}:latest"
        if ($LASTEXITCODE -ne 0) { throw "docker push :latest failed." }
    }

    if (-not $SkipModels) {
        Write-Host "==> Pushing current models to R2 (use -SkipModels to skip)" -ForegroundColor Cyan
        try {
            # A hashtable, not an array: an array splatted into a script binds "-Agent" etc. positionally as values.
            $pushModelsArgs = @{}
            if (-not [string]::IsNullOrWhiteSpace($Agent)) {
                $pushModelsArgs.Agent = $Agent
            }
            if (-not [string]::IsNullOrWhiteSpace($Build)) {
                $pushModelsArgs.Build = $Build
            }
            if ($PlainProgress) {
                $pushModelsArgs.PlainProgress = $true
            }
            & (Join-Path $PSScriptRoot "push_models.ps1") @pushModelsArgs
        }
        catch {
            Write-Warning "Models push failed (image push still succeeded): $($_.Exception.Message)"
        }
    }

    Write-Host ""
    Write-Host "Done. Deploy on Vast.ai (recommended) or RunPod:" -ForegroundColor Green
    Write-Host "    docker.io/${remote}:$Tag" -ForegroundColor Green
    Write-Host "    See Documentation/RemoteTraining/RemoteTraining.md to pick a host." -ForegroundColor Green
}
finally {
    Pop-Location
}
