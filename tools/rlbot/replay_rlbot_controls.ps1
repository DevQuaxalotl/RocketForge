param(
    [Parameter(Mandatory = $true)]
    [int]$SnapshotTick,
    [Parameter(Mandatory = $true)]
    [string]$TraceCsv,
    [int]$MaxTicks = 240,
    [string]$Agent = "example-agent",
    [Parameter(Mandatory = $true)]
    [string]$SegmentDir,
    [string]$SetupPath = "",
    [string]$CheckpointPath = "",
    [int]$BotIndex = -1,
    [long]$CarId = -1,
    [string]$OutCsv = "",
    [int]$ControlInputDelay = 0,
    [switch]$UseLastInput,
    [switch]$FocusPolicy,
    [int]$CompareOffset = 1,
    [double]$PositionTolerance = 2.0,
    [double]$VelocityTolerance = 10.0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TrainerPath = Join-Path $RepoRoot "out\build\forge\ForgeConsoleTrainer.exe"

if ([string]::IsNullOrWhiteSpace($SetupPath)) {
    $rlbotSetup = Join-Path $SegmentDir "rlbot\agent_setup.json"
    $trainingSetup = Join-Path $SegmentDir "training_setup.json"
    $SetupPath = if (Test-Path -LiteralPath $rlbotSetup) { $rlbotSetup } else { $trainingSetup }
}

if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
    $checkpointRoot = Join-Path $SegmentDir "checkpoints"
    $checkpoint = @(Get-ChildItem -LiteralPath $checkpointRoot -Directory |
        Where-Object { $_.Name -match "^\d+$" } |
        Sort-Object { [long]$_.Name } -Descending |
        Select-Object -First 1)
    if ($checkpoint.Count -gt 0) {
        $CheckpointPath = $checkpoint[0].FullName
    }
}

if ([string]::IsNullOrWhiteSpace($OutCsv)) {
    $outDir = Join-Path $RepoRoot "out\analysis\rlbot_control_replay"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $sourceSuffix = if ($FocusPolicy) { "_focus_policy" } elseif ($UseLastInput) { "_lastinput" } else { "" }
    $delaySuffix = if ($ControlInputDelay -ne 0) { "_delay{0}" -f $ControlInputDelay } else { "" }
    $OutCsv = Join-Path $outDir ("control_replay_tick{0}{1}{2}.csv" -f $SnapshotTick, $sourceSuffix, $delaySuffix)
}

if (-not (Test-Path -LiteralPath $TrainerPath)) {
    throw "ForgeConsoleTrainer.exe not found: $TrainerPath"
}
if ([string]::IsNullOrWhiteSpace($TraceCsv) -or -not (Test-Path -LiteralPath $TraceCsv)) {
    throw "Trace CSV not found. Pass -TraceCsv or capture an RLBot trace first."
}
if (-not (Test-Path -LiteralPath $SetupPath)) {
    throw "Setup path not found: $SetupPath"
}
if ([string]::IsNullOrWhiteSpace($CheckpointPath) -or -not (Test-Path -LiteralPath $CheckpointPath)) {
    throw "Checkpoint path not found: $CheckpointPath"
}

$argsList = @(
    "action-trace",
    "--agent", $Agent,
    "--setup", $SetupPath,
    "--checkpoint-path", $CheckpointPath,
    "--snapshot-csv", $TraceCsv,
    "--snapshot-tick", "$SnapshotTick",
    "--control-replay-csv", $TraceCsv,
    "--max-ticks", "$MaxTicks",
    "--episodes", "1",
    "--out", $OutCsv
)
if ($BotIndex -ge 0) {
    $argsList += @("--snapshot-bot-index", "$BotIndex")
}
if ($CarId -ge 0) {
    $argsList += @("--snapshot-car-id", "$CarId")
}
if ($ControlInputDelay -ne 0) {
    $argsList += @("--control-replay-input-delay-ticks", "$ControlInputDelay")
}
if ($UseLastInput) {
    $argsList += @("--control-replay-use-last-input")
}
if ($FocusPolicy) {
    $argsList += @("--control-replay-focus-policy")
}

& $TrainerPath @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Control replay failed."
}

$replayRows = @(Import-Csv -LiteralPath $OutCsv)
if ($replayRows.Count -eq 0) {
    throw "Control replay produced no rows: $OutCsv"
}

$traceRows = @(Import-Csv -LiteralPath $TraceCsv)
$tracedCarId = $replayRows[0].car_id
$tracedBotIndex = $replayRows[0].bot_index
$expectedByTick = @{}
foreach ($row in $traceRows) {
    if ($row.car_id -eq $tracedCarId -and $row.bot_index -eq $tracedBotIndex) {
        $expectedByTick[[int]$row.physics_tick] = $row
    }
}

$firstMismatch = $null
$comparedRows = 0
foreach ($row in $replayRows) {
    $expectedTick = [int]$row.physics_tick + $CompareOffset
    if (-not $expectedByTick.ContainsKey($expectedTick)) {
        continue
    }
    $expected = $expectedByTick[$expectedTick]
    $comparedRows++

    $posError = [Math]::Sqrt(
        [Math]::Pow(([double]$row.x - [double]$expected.x), 2) +
        [Math]::Pow(([double]$row.y - [double]$expected.y), 2) +
        [Math]::Pow(([double]$row.z - [double]$expected.z), 2))
    $velError = [Math]::Sqrt(
        [Math]::Pow(([double]$row.vx - [double]$expected.vx), 2) +
        [Math]::Pow(([double]$row.vy - [double]$expected.vy), 2) +
        [Math]::Pow(([double]$row.vz - [double]$expected.vz), 2))

    $stateMismatch =
        $row.on_ground -ne $expected.on_ground -or
        $row.obs_on_ground -ne $expected.obs_on_ground -or
        $row.is_jumping -ne $expected.is_jumping -or
        $row.is_flipping -ne $expected.is_flipping -or
        $row.has_jumped -ne $expected.has_jumped -or
        $row.has_flipped -ne $expected.has_flipped -or
        $row.has_double_jumped -ne $expected.has_double_jumped

    if ($stateMismatch -or $posError -gt $PositionTolerance -or $velError -gt $VelocityTolerance) {
        $firstMismatch = [pscustomobject]@{
            ReplayTick = [int]$row.physics_tick
            ExpectedRlbotTick = $expectedTick
            GameTimeRemaining = $row.game_time_remaining
            PositionError = [Math]::Round($posError, 3)
            VelocityError = [Math]::Round($velError, 3)
            ReplayOnGround = $row.on_ground
            RlbotOnGround = $expected.on_ground
            ReplayWheels = $row.wheels_contact_bits
            RlbotWheels = $expected.wheels_contact_bits
            ReplayJumping = $row.is_jumping
            RlbotJumping = $expected.is_jumping
            ReplayFlipping = $row.is_flipping
            RlbotFlipping = $expected.is_flipping
        }
        break
    }
}

Write-Host "Control replay CSV: $OutCsv"
Write-Host "Compared offset state rows: $comparedRows (offset: $CompareOffset)"
if ($firstMismatch) {
    Write-Host "First notable divergence:"
    $firstMismatch | Format-List
} else {
    Write-Host "No notable divergence found within the configured tolerances."
}
