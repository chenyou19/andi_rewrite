param(
    [int]$PollSeconds = 60,
    [switch]$ResumeCompletedEvals
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'C:\Users\E-118-3\miniconda3\envs\ANDi\python.exe'
$runner = Join-Path $repoRoot 'scripts\run_brats21_251_lemon_comparison.py'
$outputRoot = Join-Path $repoRoot 'outputs\comparisons\brats21_251_lemon_models_matched_noise'
$nativeConfig = Join-Path $repoRoot 'configs\eval_brats21_251_native_lemon_model_matched_noise.yaml'
$crossConfig = Join-Path $repoRoot 'configs\eval_brats21_251_lemon_brats21_noise_model.yaml'

if ($PollSeconds -lt 10) {
    throw 'PollSeconds must be at least 10.'
}
foreach ($path in @($python, $runner, $nativeConfig, $crossConfig)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}

$gitSafeDirectory = "safe.directory=$repoRoot"
$gitCommit = (& git -c $gitSafeDirectory -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($gitCommit)) {
    throw "Unable to resolve the Git commit for $repoRoot"
}

if (Test-Path -LiteralPath $outputRoot) {
    $existing = @(Get-ChildItem -LiteralPath $outputRoot -Force)
    if ($existing.Count -gt 0 -and -not $ResumeCompletedEvals) {
        throw "Comparison output directory is non-empty; refusing to overwrite: $outputRoot"
    }
    if ($ResumeCompletedEvals) {
        $statusPath = Join-Path $outputRoot 'orchestration_status.json'
        $oldMetadataPath = Join-Path $outputRoot 'orchestrator_launch_metadata.json'
        foreach ($path in @($statusPath, $oldMetadataPath)) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Resume metadata is missing: $path"
            }
        }
        $oldStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        $oldMetadata = Get-Content -LiteralPath $oldMetadataPath -Raw | ConvertFrom-Json
        if ($oldStatus.status -ne 'BLOCKED' -or $null -ne $oldStatus.active_child_pid) {
            throw "Resume requires a BLOCKED orchestration with no active child."
        }
        if ($null -ne (Get-Process -Id ([int]$oldMetadata.pid) -ErrorAction SilentlyContinue)) {
            throw "Previous orchestrator PID is still alive: $($oldMetadata.pid)"
        }
        $requiredNative = @(
            (Join-Path $outputRoot 'native_lemon\ANDi.csv'),
            (Join-Path $outputRoot 'native_lemon\ANDi_mf.csv'),
            (Join-Path $outputRoot 'native_lemon\inference_metrics_summary.csv'),
            (Join-Path $outputRoot 'native_lemon\inference_report.json')
        )
        foreach ($path in $requiredNative) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Completed native evaluation output is missing: $path"
            }
        }
        foreach ($path in @(
            (Join-Path $outputRoot 'lemon_brats21_noise\ANDi.csv'),
            (Join-Path $outputRoot 'comparison_summary.csv'),
            (Join-Path $outputRoot 'comparison_report.md')
        )) {
            if (Test-Path -LiteralPath $path) {
                throw "Resume target already exists; refusing to overwrite: $path"
            }
        }
    }
} else {
    if ($ResumeCompletedEvals) {
        throw "Resume requested but comparison output root does not exist: $outputRoot"
    }
    New-Item -ItemType Directory -Path $outputRoot | Out-Null
}

$logStem = if ($ResumeCompletedEvals) { 'orchestrator_resume' } else { 'orchestrator' }
$stdoutPath = Join-Path $outputRoot "$($logStem)_stdout.log"
$stderrPath = Join-Path $outputRoot "$($logStem)_stderr.log"
$metadataPath = Join-Path $outputRoot "$($logStem)_launch_metadata.json"
foreach ($path in @($stdoutPath, $stderrPath, $metadataPath)) {
    if (Test-Path -LiteralPath $path) {
        throw "Orchestrator resume artifact already exists; refusing to overwrite: $path"
    }
}
$startedAt = (Get-Date).ToString('o')
$runnerArguments = @($runner, '--poll-seconds', [string]$PollSeconds)
if ($ResumeCompletedEvals) {
    $runnerArguments += '--resume-completed-evals'
}
$process = Start-Process `
    -FilePath $python `
    -ArgumentList $runnerArguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$metadata = [ordered]@{
    status = 'LAUNCHED'
    pid = $process.Id
    start_time = $startedAt
    python_executable = $python
    runner = $runner
    poll_seconds = $PollSeconds
    resume_completed_evals = [bool]$ResumeCompletedEvals
    git_commit = $gitCommit
    native_eval_config = $nativeConfig
    native_eval_config_sha256 = (Get-FileHash -LiteralPath $nativeConfig -Algorithm SHA256).Hash.ToLowerInvariant()
    cross_eval_config = $crossConfig
    cross_eval_config_sha256 = (Get-FileHash -LiteralPath $crossConfig -Algorithm SHA256).Hash.ToLowerInvariant()
    stdout = $stdoutPath
    stderr = $stderrPath
    status_file = (Join-Path $outputRoot 'orchestration_status.json')
    hidden_window = $true
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadataPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 5
