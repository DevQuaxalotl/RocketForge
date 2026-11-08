<#
.SYNOPSIS
  Push the files the selected agent's stage queue needs to Cloudflare R2 so a
  training pod can seed (continue) from them.

.DESCRIPTION
  Resolves the agent (the -Agent override, else lastSelectedAgent from forge_app_state.json)
  and asks the built ForgeConsoleTrainer (`sync-manifest`) which files its stage queue touches:
  build metadata, every segment.json, the segments the queue continues or creates,
  the lineage's recent-policy and skill-benchmark pools,
  and the model folders selected as static opponents (imported models and source segments alike).
  Pushes exactly that set with ONE `rclone copy --filter-from` rooted at Bots/.
  Loads R2 credentials from ~/.rocketforge_env and uses --update
  (non-destructive: never deletes remote files, never overwrites a newer remote file).
  Safe to run repeatedly.

.PARAMETER Agent
  Agent name to push. Default: the GUI-selected agent in forge_app_state.json.

.PARAMETER Build
  Optional build name to push instead of forge_settings.json's buildName.

.PARAMETER Bucket
  R2 bucket. Default: $env:R2_BUCKET, else "rocketforge-checkpoints".

.PARAMETER ImportedModels
  Push the agent's whole "Imported Models" tree instead of a build. Use this to place
  a frozen model (an RLBotPack bot, say) on R2 for agents that have no build of their
  own. A build push already carries the imported models its static opponents name, so
  this is only needed to seed one ahead of that selection.

.PARAMETER PlainProgress
  Print rclone progress as periodic one-line stats instead of -P's in-place redraw,
  for output that is piped into a log (the GUI's Remote Sync panel).

.EXAMPLE
  ./tools/remote/push_models.ps1
  ./tools/remote/push_models.ps1 -Agent example-agent
  ./tools/remote/push_models.ps1 -Agent example-agent -Build MyBuild
  ./tools/remote/push_models.ps1 -Agent Nexto -ImportedModels
#>
param(
    [string]$Agent  = "",
    [string]$Build  = "",
    [string]$Bucket = "",
    [switch]$ImportedModels,
    [switch]$PlainProgress
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# --stats lines log at INFO by default, below rclone's NOTICE log level, so raise them to NOTICE.
$RcloneProgress = if ($PlainProgress) { @('--stats', '2s', '--stats-one-line', '--stats-log-level', 'NOTICE') } else { @('-P') }

# Writes the trainer's rclone filter list to a temp file and returns its path.
# rclone reads the file byte-for-byte, so it must be UTF-8 without a BOM.
function Write-SyncManifest {
    param([string]$Exe, [string[]]$TrainerArgs)

    $lines = & $Exe sync-manifest @TrainerArgs
    if ($LASTEXITCODE -ne 0) { throw "sync-manifest failed (exit $LASTEXITCODE)." }
    if (-not $lines) { throw "sync-manifest printed an empty filter list." }

    $path = Join-Path ([System.IO.Path]::GetTempPath()) ("rocketforge_sync_" + [System.IO.Path]::GetRandomFileName() + ".txt")
    [System.IO.File]::WriteAllText($path, (($lines -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding($false)))
    return $path
}

Push-Location $RepoRoot
try {
    # Load R2 credentials from the env file (same convention as pull_models.ps1).
    $envFile = Join-Path $env:USERPROFILE ".rocketforge_env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match '^\s*([^#=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
            }
        }
    }

    if ([string]::IsNullOrWhiteSpace($Bucket)) {
        $Bucket = if ($env:R2_BUCKET) { $env:R2_BUCKET } else { "rocketforge-checkpoints" }
    }

    # Newest binary wins: a stale preset's exe may lack the subcommands this script needs.
    $exe = Get-ChildItem -Path (Join-Path $RepoRoot "out\build") -Recurse -Filter "ForgeConsoleTrainer.exe" -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $exe) {
        throw "ForgeConsoleTrainer.exe not found under out/build. Build the console first (cmake --preset forge, then cmake --build --preset build-console)."
    }

    # models-path only needs the agent name, so it resolves agents that have no build
    # (and no forge_settings.json) at all — e.g. a public bot that only holds imported models.
    if ($ImportedModels) {
        $modelsRel = if ([string]::IsNullOrWhiteSpace($Agent)) {
            & $exe.FullName models-path
        } else {
            & $exe.FullName models-path --agent $Agent
        }
        $modelsRel = ([string]$modelsRel).Trim()
        if ([string]::IsNullOrWhiteSpace($modelsRel)) {
            throw "models-path did not return a Models path for agent '$Agent'."
        }

        $importedRel   = "$modelsRel/Imported Models"
        $localImported = Join-Path $RepoRoot ($importedRel -replace '/', '\')
        if (-not (Test-Path $localImported)) {
            throw "Local imported models path does not exist: $localImported"
        }

        Write-Host "==> Pushing $importedRel -> r2:$Bucket/$importedRel" -ForegroundColor Cyan
        rclone copy $localImported "r2:$Bucket/$importedRel" --update --transfers 8 @RcloneProgress
        if ($LASTEXITCODE -ne 0) { throw "rclone copy (imported models) failed (exit $LASTEXITCODE)." }

        Write-Host "Done. R2 now has the imported models for: $importedRel" -ForegroundColor Green
        return
    }

    $manifestArgs = @()
    if (-not [string]::IsNullOrWhiteSpace($Agent)) { $manifestArgs += @("--agent", $Agent) }
    if (-not [string]::IsNullOrWhiteSpace($Build)) {
        $manifestArgs += @("--build", $Build)
        Write-Host "==> Using build override '$Build'" -ForegroundColor Cyan

        # -Build only changes what is pushed; a pod still trains forge_settings.json's buildName.
        $modelsRel = if ([string]::IsNullOrWhiteSpace($Agent)) {
            (& $exe.FullName models-path | Select-Object -Last 1).Trim()
        } else {
            (& $exe.FullName models-path --agent $Agent | Select-Object -Last 1).Trim()
        }
        $localSettings = Join-Path $RepoRoot ((($modelsRel -replace '/Models$', '') + "/forge_settings.json") -replace '/', '\')
        if (Test-Path $localSettings) {
            try {
                $settingsBuild = [string]((Get-Content $localSettings -Raw | ConvertFrom-Json).buildName)
                if (-not [string]::IsNullOrWhiteSpace($settingsBuild) -and $settingsBuild -ne $Build) {
                    Write-Warning "-Build '$Build' only overrides the build being pushed. forge_settings.json still says buildName '$settingsBuild', so a pod will train that build, not '$Build'."
                }
            }
            catch {
                Write-Warning "Could not inspect forge_settings.json buildName: $($_.Exception.Message)"
            }
        }
    }

    $filterFile = Write-SyncManifest $exe.FullName $manifestArgs
    try {
        $localBots = Join-Path $RepoRoot "Bots"
        Write-Host "==> Pushing training manifest: $localBots -> r2:$Bucket/Bots" -ForegroundColor Cyan
        rclone copy $localBots "r2:$Bucket/Bots" --filter-from $filterFile --update --transfers 8 --checkers 16 @RcloneProgress
        if ($LASTEXITCODE -ne 0) { throw "rclone copy failed (exit $LASTEXITCODE)." }
    }
    finally {
        Remove-Item $filterFile -ErrorAction SilentlyContinue
    }

    Write-Host "Done. R2 now has the latest local files for the selected build's stage queue." -ForegroundColor Green
}
finally {
    Pop-Location
}
