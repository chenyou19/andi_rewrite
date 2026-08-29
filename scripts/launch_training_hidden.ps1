param(
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$RunName
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe'
$trainScript = Join-Path $repoRoot 'scripts\train.py'
$configPath = (Resolve-Path $Config).Path
$runRoot = Join-Path $repoRoot (Join-Path 'outputs\runs' $RunName)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Required ANDi Python is missing: $python"
}
$gitSafeDirectory = "safe.directory=$repoRoot"
$gitCommit = (& git -c $gitSafeDirectory -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitCommit)) {
    throw "Unable to resolve the Git commit for $repoRoot"
}
$gitStatus = @(& git -c $gitSafeDirectory -C $repoRoot status --short)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to resolve the Git worktree status for $repoRoot"
}
if (Test-Path -LiteralPath $runRoot) {
    $existing = @(Get-ChildItem -LiteralPath $runRoot -Force)
    if ($existing.Count -gt 0) {
        throw "Run directory is non-empty; refusing to overwrite: $runRoot"
    }
} else {
    New-Item -ItemType Directory -Path $runRoot | Out-Null
}

$snapshot = Join-Path $runRoot 'config_snapshot.yaml'
$stdoutPath = Join-Path $runRoot 'stdout.log'
$stderrPath = Join-Path $runRoot 'stderr.log'
$metadataPath = Join-Path $runRoot 'launch_metadata.json'
$setupJsonPath = Join-Path $runRoot 'setup_report.json'
$setupMarkdownPath = Join-Path $runRoot 'setup_report.md'
Copy-Item -LiteralPath $configPath -Destination $snapshot

$configSha256 = (Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant()
$startedAt = (Get-Date).ToString('o')
$process = Start-Process `
    -FilePath $python `
    -ArgumentList @($trainScript, '--config', $snapshot, '--fit') `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$metadata = [ordered]@{
    status = 'LAUNCHED'
    run_name = $RunName
    pid = $process.Id
    start_time = $startedAt
    python_executable = $python
    training_entrypoint = $trainScript
    source_config = $configPath
    config_snapshot = $snapshot
    config_sha256 = $configSha256
    git_commit = $gitCommit
    git_worktree_status_at_launch = $gitStatus
    stdout = $stdoutPath
    stderr = $stderrPath
    hidden_window = $true
}
$metadata | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $metadataPath -Encoding utf8

$setup = [ordered]@{
    run_name = $RunName
    launch_metadata = $metadataPath
    config_snapshot = $snapshot
    metrics_csv = (Join-Path $runRoot 'training_metrics.csv')
    final_training_report_json = (Join-Path $runRoot 'training_report.json')
    final_training_report_markdown = (Join-Path $runRoot 'training_report.md')
    first_epoch_gate = 'train_loss and validation_loss finite; process alive; no OOM/NaN in logs'
}
$setup | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $setupJsonPath -Encoding utf8
@(
    '# Training Setup Report'
    ''
    "- Run: $RunName"
    "- PID: $($process.Id)"
    "- Start time: $startedAt"
    "- Python: $python"
    "- Git commit: $gitCommit"
    "- Config SHA-256: $configSha256"
    "- Metrics: $(Join-Path $runRoot 'training_metrics.csv')"
    "- stdout: $stdoutPath"
    "- stderr: $stderrPath"
    ''
    'The first-epoch gate requires finite train/validation losses, a live GPU process, and no OOM/NaN log evidence.'
) | Set-Content -LiteralPath $setupMarkdownPath -Encoding utf8

$metadata | ConvertTo-Json -Depth 6
