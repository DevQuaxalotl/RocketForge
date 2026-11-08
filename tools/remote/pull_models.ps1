<#
.SYNOPSIS
  Pull trained checkpoints from Cloudflare R2 back to the local Bots tree.

.DESCRIPTION
  The mirror of push_models.ps1.
  Loads R2 credentials from ~/.rocketforge_env and runs `rclone copy --update`
  (non-destructive: never deletes local files, never overwrites a newer local file).
  Safe to run repeatedly.

  With no arguments it pulls the entire Bots tree.
  -Agent pulls that agent's whole Models tree plus forge_settings.json.
  -Agent -TrainingOnly pulls only what the agent's stage queue needs,
  as listed by the built ForgeConsoleTrainer (`sync-manifest`):
  forge_settings.json first (it names the segments a remote run created),
  then the build's metadata, then the queue's segments,
  the lineage's recent-policy and skill-benchmark pools, and the static opponents.
  It then copies the segment ids the remote run recorded into the local forge_settings.json,
  keeping every other local edit.

.PARAMETER Agent
  Pull only this agent. Required with -TrainingOnly. Default: pull the whole Bots tree.

.PARAMETER TrainingOnly
  With -Agent: pull only the files the agent's stage queue needs
  (three passes: forge_settings.json, build metadata, then the full manifest).

.PARAMETER TempCheckpoints
  With -TrainingOnly: also pull the newest remote autosave (temp_checkpoints/) of each queue segment,
  so a running pod's latest model can be tested locally, or a crashed run recovered.
  The GUI (RLBot export, Debug tab, benchmark) and a local resume pick it up when it is newer than the durable checkpoint.

.PARAMETER AllTempCheckpoints
  With -TempCheckpoints: pull every remote autosave of each queue segment instead of only the newest,
  so an earlier timestep is available to roll back to.
  The GUI and a local resume still use the newest autosave;
  to fall back, delete the newer timestep folders from the segment's local temp_checkpoints/
  (a later -TempCheckpoints pull brings the newest one back).

.PARAMETER Build
  Optional build name to pull instead of forge_settings.json's buildName. With -Agent
  alone it narrows the Models pull to that build's folder + builds.json.

.PARAMETER Bucket
  R2 bucket. Default: $env:R2_BUCKET, else "rocketforge-checkpoints".

.PARAMETER PlainProgress
  Print rclone progress as periodic one-line stats instead of -P's in-place redraw,
  for output that is piped into a log (the GUI's Remote Sync panel).

.EXAMPLE
  ./tools/remote/pull_models.ps1
  ./tools/remote/pull_models.ps1 -Agent example-agent
  ./tools/remote/pull_models.ps1 -Agent example-agent -TrainingOnly
  ./tools/remote/pull_models.ps1 -Agent example-agent -TrainingOnly -TempCheckpoints
  ./tools/remote/pull_models.ps1 -Agent example-agent -TrainingOnly -TempCheckpoints -AllTempCheckpoints
  ./tools/remote/pull_models.ps1 -Agent example-agent -Build MyBuild
#>
param(
    [string]$Agent  = "",
    [string]$Build  = "",
    [string]$Bucket = "",
    [switch]$TrainingOnly,
    [switch]$TempCheckpoints,
    [switch]$AllTempCheckpoints,
    [switch]$PlainProgress
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# --stats lines log at INFO by default, below rclone's NOTICE log level, so raise them to NOTICE.
$RcloneProgress = if ($PlainProgress) { @('--stats', '2s', '--stats-one-line', '--stats-log-level', 'NOTICE') } else { @('-P') }

function ConvertTo-BuildFolderName {
    param([string]$Name)

    $builder = [System.Text.StringBuilder]::new()
    $lastWasSeparator = $false

    foreach ($ch in $Name.ToCharArray()) {
        if ([char]::IsLetterOrDigit($ch)) {
            [void]$builder.Append($ch)
            $lastWasSeparator = $false
        }
        elseif ($ch -eq [char]'.' -and $builder.Length -gt 0 -and -not $lastWasSeparator) {
            [void]$builder.Append($ch)
            $lastWasSeparator = $false
        }
        elseif (-not $lastWasSeparator) {
            [void]$builder.Append('_')
            $lastWasSeparator = $true
        }
    }

    $value = $builder.ToString().Trim([char[]]@('_', '.'))
    if ([string]::IsNullOrWhiteSpace($value)) { return "V1" }
    return $value
}

function Resolve-BuildFolder {
    param([string]$ModelsRel, [string]$BuildName)

    $registryPath = Join-Path $RepoRoot (("$ModelsRel/builds.json") -replace '/', '\')
    if (Test-Path $registryPath) {
        try {
            $registry = Get-Content $registryPath -Raw | ConvertFrom-Json
            foreach ($entry in @($registry.builds)) {
                $entryBuildName = [string]$entry.build_name
                $entryFolderName = [string]$entry.folder_name
                if ($entryBuildName -eq $BuildName -or $entryFolderName -eq $BuildName) {
                    if (-not [string]::IsNullOrWhiteSpace($entryFolderName)) {
                        return $entryFolderName
                    }
                    return ConvertTo-BuildFolderName $entryBuildName
                }
            }
        }
        catch {
            Write-Warning "Could not read $registryPath to resolve -Build '$BuildName': $($_.Exception.Message)"
        }
    }

    return ConvertTo-BuildFolderName $BuildName
}

# Writes rclone filter lines to a temp file and returns its path.
# rclone reads the file byte-for-byte, so it must be UTF-8 without a BOM.
function Write-FilterFile {
    param([string[]]$Lines)

    if (-not $Lines) { throw "Filter list is empty." }

    $path = Join-Path ([System.IO.Path]::GetTempPath()) ("rocketforge_sync_" + [System.IO.Path]::GetRandomFileName() + ".txt")
    [System.IO.File]::WriteAllText($path, (($Lines -join "`n") + "`n"), (New-Object System.Text.UTF8Encoding($false)))
    return $path
}

# Consumes and deletes the filter file, whether or not rclone succeeded.
function Invoke-FilteredPull {
    param([string]$FilterFile, [string]$Label)

    try {
        $localBots = Join-Path $RepoRoot "Bots"
        Write-Host "==> Pulling $Label`: r2:$Bucket/Bots -> $localBots" -ForegroundColor Cyan
        rclone copy "r2:$Bucket/Bots" $localBots --filter-from $FilterFile --update --transfers 8 --checkers 16 @RcloneProgress
        if ($LASTEXITCODE -ne 0) { throw "rclone copy ($Label) failed (exit $LASTEXITCODE)." }
    }
    finally {
        Remove-Item $FilterFile -ErrorAction SilentlyContinue
    }
}

# rclone filter patterns treat these characters as glob syntax.
function ConvertTo-RcloneGlobLiteral {
    param([string]$Path)

    return ($Path -replace '([\\*?\[\]{}])', '\$1')
}

function Get-NewestLocalCheckpoint {
    param([string]$CheckpointsRoot)

    if (-not (Test-Path -LiteralPath $CheckpointsRoot)) { return $null }

    $newest = Get-ChildItem -LiteralPath $CheckpointsRoot -Directory | Where-Object { $_.Name -match '^\d+$' } |
              ForEach-Object { [UInt64]$_.Name } | Sort-Object -Descending | Select-Object -First 1
    return $newest
}

# Groups autosave timesteps by temp_checkpoints root from `rclone lsf -R --files-only` output, each list ascending.
# Files, not directories: a folder only counts once POLICY.lt has landed,
# which skips folders the pod has only started uploading.
function Get-TempCheckpointsByRoot {
    param([string[]]$RemoteFiles)

    $timestepsByRoot = @{}
    foreach ($file in $RemoteFiles) {
        if ($file -notmatch '^(?<root>.+/temp_checkpoints)/(?<ts>\d+)/POLICY\.lt$') { continue }

        if (-not $timestepsByRoot.ContainsKey($matches.root)) {
            $timestepsByRoot[$matches.root] = [System.Collections.Generic.List[UInt64]]::new()
        }
        $timestepsByRoot[$matches.root].Add([UInt64]$matches.ts)
    }

    foreach ($root in @($timestepsByRoot.Keys)) {
        $timestepsByRoot[$root] = @($timestepsByRoot[$root] | Sort-Object)
    }

    return $timestepsByRoot
}

# Pulls the remote autosaves of each queue segment: only the newest unless $All,
# which keeps earlier timesteps available to roll back to.
function Invoke-TempCheckpointPull {
    param([string]$Exe, [string[]]$ManifestArgs, [bool]$All)

    $tempFilter = & $Exe sync-manifest @ManifestArgs --temp-checkpoints
    if ($LASTEXITCODE -ne 0) { throw "sync-manifest --temp-checkpoints failed (exit $LASTEXITCODE)." }

    # The manifest escapes glob characters in ids; undo that to get the literal segment id.
    $segmentIds = @($tempFilter | ForEach-Object {
        if ($_ -match '/segments/(?<id>[^/]+)_\*/temp_checkpoints/') { $matches.id -replace '\\(.)', '$1' }
    })

    $listFilter = Write-FilterFile $tempFilter
    try {
        $remoteFiles = rclone lsf -R --files-only --filter-from $listFilter "r2:$Bucket/Bots"
        if ($LASTEXITCODE -ne 0) { throw "rclone lsf (temp checkpoints) failed (exit $LASTEXITCODE)." }
    }
    finally {
        Remove-Item $listFilter -ErrorAction SilentlyContinue
    }

    $timestepsByRoot = Get-TempCheckpointsByRoot @($remoteFiles)

    if ($timestepsByRoot.Count -gt 0) {
        $rules = @($timestepsByRoot.Keys | Sort-Object | ForEach-Object {
            $root = $_
            $selected = if ($All) { $timestepsByRoot[$root] } else { @($timestepsByRoot[$root])[-1] }
            $selected | ForEach-Object { "+ /" + (ConvertTo-RcloneGlobLiteral "$root/$_") + "/**" }
        })
        $rules += "- **"
        $label = if ($All) { "all autosaves" } else { "newest autosaves" }
        Invoke-FilteredPull (Write-FilterFile $rules) $label
    }

    foreach ($segmentId in $segmentIds) {
        $root = $timestepsByRoot.Keys | Where-Object { $_ -match "/segments/$([regex]::Escape($segmentId))_[^/]+/temp_checkpoints$" } | Select-Object -First 1
        if (-not $root) {
            Write-Host "  ${segmentId}: no remote autosave" -ForegroundColor Yellow
            continue
        }

        $timesteps = @($timestepsByRoot[$root])
        $pulledText = if ($All -and $timesteps.Count -gt 1) {
            "$($timesteps.Count) autosaves $($timesteps[0])..$($timesteps[-1])"
        }
        else {
            "autosave $($timesteps[-1])"
        }

        $localCheckpoints = Join-Path $RepoRoot ((("Bots/" + $root) -replace '/temp_checkpoints$', '/checkpoints') -replace '/', '\')
        $durable = Get-NewestLocalCheckpoint $localCheckpoints
        $durableText = if ($null -ne $durable) { "$durable" } else { "none" }
        Write-Host "  ${segmentId}: $pulledText (durable $durableText)" -ForegroundColor Green
    }
}

Push-Location $RepoRoot
try {
    # Load R2 credentials from the env file (same convention as push_models.ps1).
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

    if ($TrainingOnly -and [string]::IsNullOrWhiteSpace($Agent)) {
        throw "-TrainingOnly requires -Agent <name> so the narrow pull cannot target the wrong agent."
    }

    if ($TempCheckpoints -and -not $TrainingOnly) {
        throw "-TempCheckpoints requires -TrainingOnly: it extends the training-only pull with the queue segments' newest autosaves."
    }

    if ($AllTempCheckpoints -and -not $TempCheckpoints) {
        throw "-AllTempCheckpoints requires -TempCheckpoints: it widens the autosave pull from the newest to every remote timestep."
    }

    if ([string]::IsNullOrWhiteSpace($Agent)) {
        $rel = "Bots"
        $localPath = Join-Path $RepoRoot $rel
        Write-Host "==> Pulling r2:$Bucket/$rel -> $rel" -ForegroundColor Cyan
        rclone copy "r2:$Bucket/$rel" $localPath --update --transfers 8 --checkers 16 @RcloneProgress
        if ($LASTEXITCODE -ne 0) { throw "rclone copy failed (exit $LASTEXITCODE)." }
        Write-Host "Done. Local checkpoints updated for: $rel" -ForegroundColor Green
        return
    }

    # Newest binary wins: a stale preset's exe may lack the subcommands this script needs.
    $exe = Get-ChildItem -Path (Join-Path $RepoRoot "out\build") -Recurse -Filter "ForgeConsoleTrainer.exe" -ErrorAction SilentlyContinue |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $exe) {
        throw "ForgeConsoleTrainer.exe not found under out/build. Build the console first (cmake --preset forge, then cmake --build --preset build-console)."
    }

    $modelsRel = (& $exe.FullName models-path --agent $Agent | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($modelsRel)) {
        throw "models-path did not return a Models path for agent '$Agent'."
    }
    $agentRel    = $modelsRel -replace '/Models$', ''
    $settingsRel = "$agentRel/forge_settings.json"
    $botsRelativeModels = $modelsRel -replace '^Bots/', ''
    $botsRelativeAgent  = $agentRel  -replace '^Bots/', ''

    # forge_settings.json first: it carries createdSegmentId (the segment a remote run created or continued),
    # which sync-manifest reads to decide what the queue needs. It may not exist remotely yet, so no hard-fail.
    $localSettings = Join-Path $RepoRoot ($settingsRel -replace '/', '\')
    Write-Host "==> Pulling r2:$Bucket/$settingsRel" -ForegroundColor Cyan
    rclone copyto "r2:$Bucket/$settingsRel" $localSettings --update @RcloneProgress

    if (-not $TrainingOnly) {
        $rules = @("+ /$botsRelativeAgent/forge_settings.json")
        if ([string]::IsNullOrWhiteSpace($Build)) {
            $rules += "+ /$botsRelativeModels/**"
        }
        else {
            # Registry first so a renamed-folder build still resolves (it may not exist remotely yet, so no hard-fail).
            $localRegistry = Join-Path $RepoRoot (("$modelsRel/builds.json") -replace '/', '\')
            rclone copyto "r2:$Bucket/$modelsRel/builds.json" $localRegistry --update @RcloneProgress
            $buildFolder = Resolve-BuildFolder $modelsRel $Build
            Write-Host "==> Using build override '$Build' -> folder '$buildFolder'" -ForegroundColor Cyan
            $rules += "+ /$botsRelativeModels/builds.json"
            $rules += "+ /$botsRelativeModels/$buildFolder/**"
        }
        $rules += "- **"

        Invoke-FilteredPull (Write-FilterFile $rules) "agent '$Agent'"
        Write-Host "Done. Local checkpoints updated for agent: $Agent" -ForegroundColor Green
        return
    }

    $manifestArgs = @("--agent", $Agent)
    if (-not [string]::IsNullOrWhiteSpace($Build)) {
        $manifestArgs += @("--build", $Build)
        Write-Host "==> Using build override '$Build'" -ForegroundColor Cyan
    }

    # The manifest must follow the remote settings, which record the segments the remote run created.
    # The --update pull above keeps a local copy that was re-saved after the push (e.g. by the GUI),
    # and that copy still predicts segment ids instead of naming the real ones.
    $remoteSettings = Join-Path ([System.IO.Path]::GetTempPath()) ("rocketforge_settings_" + [System.IO.Path]::GetRandomFileName() + ".json")
    rclone copyto "r2:$Bucket/$settingsRel" $remoteSettings
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $remoteSettings)) {
        $manifestArgs += @("--file", $remoteSettings)
    }
    else {
        Write-Warning "No remote forge_settings.json for '$Agent'; building the manifest from the local settings."
    }

    try {
        # Metadata pass first:
        # the full manifest walks parent chains through segment.json files a remote run may have created since the last pull.
        $metadata = & $exe.FullName sync-manifest @manifestArgs --metadata-only
        if ($LASTEXITCODE -ne 0) { throw "sync-manifest --metadata-only failed (exit $LASTEXITCODE)." }
        Invoke-FilteredPull (Write-FilterFile $metadata) "build metadata"

        $full = & $exe.FullName sync-manifest @manifestArgs
        if ($LASTEXITCODE -ne 0) { throw "sync-manifest failed (exit $LASTEXITCODE)." }
        Invoke-FilteredPull (Write-FilterFile $full) "training manifest"

        if ($TempCheckpoints) {
            Invoke-TempCheckpointPull $exe.FullName $manifestArgs $AllTempCheckpoints.IsPresent
        }

        # The local settings stay on disk when they are newer, so give them the segments the remote run recorded;
        # otherwise the local queue would not treat the pulled segments as its own.
        if ($manifestArgs -contains "--file") {
            & $exe.FullName merge-segment-ids --agent $Agent --from $remoteSettings
            if ($LASTEXITCODE -ne 0) { throw "merge-segment-ids failed (exit $LASTEXITCODE)." }
        }
    }
    finally {
        Remove-Item -LiteralPath $remoteSettings -ErrorAction SilentlyContinue
    }

    Write-Host "Done. Local files updated for the stage queue of agent: $Agent" -ForegroundColor Green
}
finally {
    Pop-Location
}
