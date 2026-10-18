param(
    [ValidateSet('prepare','run','status','merge')]
    [string]$Command = 'prepare',
    [string]$Python = 'python',
    [string]$JobDir = '.\jobs\pml_topmetal_au100_metal8_100',
    [ValidateRange(1,1000000)][int]$Samples = 100,
    [ValidateRange(0,64)][int]$Jobs = 0,
    [ValidateRange(0,64)][int]$Workers = 0,
    [ValidateRange(0,64)][int]$Concurrency = 0,
    [ValidateRange(0,1024)][int]$Threads = 0,
    [string]$Lumapi = '',
    [ValidateRange(1,10000)][int]$PackEvery = 10
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$jobRoot = [System.IO.Path]::GetFullPath($JobDir)
$configPath = Join-Path $jobRoot 'job.json'
$layoutPath = Join-Path $jobRoot 'parallel_plan.json'
$mergedPath = Join-Path $jobRoot 'merged'
$pythonInfo = & $Python -c 'import sys,json,deta_generator2; from deta_generator2.resources import cpu_inventory; info=cpu_inventory(); print(json.dumps(dict(python=sys.executable,version=deta_generator2.__version__,package_file=deta_generator2.__file__,cpus=info[''budget_cpus''],cpu_inventory=info)))'
if ($LASTEXITCODE -ne 0) { throw 'Cannot load deta_generator2 in the selected Python environment.' }
$pythonInfo = $pythonInfo | ConvertFrom-Json
if ([version]$pythonInfo.version -lt [version]'2.1.0') { throw 'Install deta-generator2 2.1.0 or newer first.' }
$pythonExe = $pythonInfo.python
if ($Jobs -gt 0) {
    if ($Concurrency -gt 0 -and $Concurrency -ne $Jobs) { throw '-Jobs and -Concurrency must agree; specify only -Jobs.' }
    $Concurrency = $Jobs
}
$explicitConcurrency = $Concurrency -gt 0
if ($Concurrency -eq 0) { $Concurrency = [Math]::Min(4,$Samples) }
if ($Workers -eq 0) {
    if (Test-Path -LiteralPath $layoutPath) {
        $Workers = (Get-Content -LiteralPath $layoutPath -Raw | ConvertFrom-Json).workers
    } else { $Workers = [Math]::Min([Math]::Min(64,$Samples),$Concurrency*4) }
}
if (-not $explicitConcurrency) { $Concurrency = [Math]::Min($Concurrency,$Workers) }
if ($Workers -gt $Samples -or $Concurrency -gt $Workers) {
    throw "Requested $Concurrency active jobs but task has $Workers shards and $Samples samples. Use a new JobDir for more shards."
}
if ($Threads -eq 0) { $Threads = [Math]::Max(1,[Math]::Floor($pythonInfo.cpus/$Concurrency)) }
if ($Concurrency * $Threads -gt $pythonInfo.cpus) {
    throw "$Concurrency jobs x $Threads threads exceeds the $($pythonInfo.cpus) usable CPUs. Omit Threads for automatic allocation."
}

function Invoke-Generator([string[]]$GeneratorArgs) {
    if ($Lumapi -and $GeneratorArgs -notcontains '--lumapi') { $GeneratorArgs += @('--lumapi',$Lumapi) }
    & $pythonExe -m deta_generator2 @GeneratorArgs
    if ($LASTEXITCODE -ne 0) { throw "Generator failed: $($GeneratorArgs -join ' ')" }
}

function Assert-Task {
    if (-not (Test-Path -LiteralPath $configPath) -or -not (Test-Path -LiteralPath $layoutPath)) {
        throw 'Run -Command prepare once in a new job directory first.'
    }
    $taskConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $layout = Get-Content -LiteralPath $layoutPath -Raw | ConvertFrom-Json
    if ($taskConfig.sample_count -ne $Samples -or $layout.samples -ne $Samples -or $layout.workers -ne $Workers) {
        throw 'Sample count and shard count are fixed. Use a new JobDir to change them.'
    }
    if ($taskConfig.simulation.metal_position -ne 'above_substrate' -or
        $taskConfig.simulation.substrate_termination -ne 'pml' -or
        $taskConfig.simulation.mesh_policy -ne 'metal_only' -or $taskConfig.simulation.metal_mesh_nm -ne 8 -or
        $taskConfig.simulation.metal_thickness_nm -ne 100 -or $taskConfig.simulation.remove_metal) {
        throw 'This launcher requires the SiO2 -> Au100 -> As2Se3 / metal-only 8 nm task; old jobs cannot be resumed in v2.'
    }
}

function Show-Progress([switch]$WriteSnapshot) {
    try {
        $monitorArgs = @('-m','deta_generator2.monitor','--job-dir',$jobRoot,'--expected-jobs',"$Concurrency")
        if ($WriteSnapshot) { $monitorArgs += '--write-snapshot' }
        $monitorOutput = & $pythonExe @monitorArgs
        if ($LASTEXITCODE -ne 0) { throw 'Native process observation unavailable' }
        foreach ($line in $monitorOutput) { Write-Host $line }
        return
    } catch { Write-Warning 'Detailed parallel monitor unavailable; see shard logs. Generation is continuing.' }
    # Do not fall back to Get-Content on live manifests: ordinary Windows read
    # handles can deny DELETE sharing and interfere with their atomic replace.
}

function Merge-Results {
    $mergeArgs = @('merge','--config',$configPath,'--output',$mergedPath,'--shards')
    for ($index=0; $index -lt $Workers; $index++) { $mergeArgs += (Join-Path $jobRoot "shard$index") }
    Invoke-Generator $mergeArgs
    $summary = Get-Content -LiteralPath (Join-Path $mergedPath 'summary.json') -Raw | ConvertFrom-Json
    if ($summary.success -ne $Samples) { throw "Only $($summary.success)/$Samples samples were merged. Rerun missing/failed shards before training." }
    Write-Host "Dataset ready: $(Join-Path $mergedPath 'dataset.npz')"
}

# Start-Process joins ArgumentList into one Windows command line. Quote each
# argument using the Windows CRT rules so directories with spaces work too.
function Quote-WindowsArgument([string]$Value) {
    return '"' + (($Value -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
}

if ($Command -eq 'prepare') {
    if (Test-Path -LiteralPath $configPath) {
        Assert-Task
    } else {
        $initArgs = @('init','--job-dir',$jobRoot,'--preset','au-metal','--metal-mesh-nm','8',
                      '--samples',"$Samples",'--threads',"$Threads")
        if ($Lumapi) { $initArgs += @('--lumapi',$Lumapi) }
        Invoke-Generator $initArgs
        @{ samples=$Samples; workers=$Workers; created_at=(Get-Date).ToString('o') } |
            ConvertTo-Json | Set-Content -LiteralPath $layoutPath -Encoding UTF8
    }
    Invoke-Generator @('plan','--config',$configPath)
    Write-Host "Prepared $Samples samples, $Workers queued shards, up to $Concurrency concurrent samples x $Threads threads. No FDTD run started."
    exit 0
}

Assert-Task
if ($Command -eq 'status') { $null = Show-Progress; exit 0 }
if ($Command -eq 'merge') { Merge-Results; exit 0 }

New-Item -ItemType Directory -Force -Path (Join-Path $jobRoot 'parallel_logs') | Out-Null
$active = @()
$nextIndex = 0
$failedWorkers = @()
$lastDisplay = [datetime]::MinValue
$controllerLock = $null
$pidPath = Join-Path $jobRoot 'parallel_workers.json'
$runtimeWorkers = @()
Write-Host "Running up to $Concurrency concurrent samples, $Threads threads per sample."
Write-Host "Package $($pythonInfo.version): $($pythonInfo.package_file)"
Write-Host "CPU budget: $($pythonInfo.cpus); queued shards: $Workers; requested solver threads: $($Concurrency*$Threads)."
try {
    try {
        $controllerLock = [System.IO.File]::Open((Join-Path $jobRoot '.parallel-controller.lock'),
            [System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
    } catch { throw 'Another parallel controller is using this task.' }
    if (Test-Path -LiteralPath $pidPath) {
        $previousWorkers = (Get-Content -LiteralPath $pidPath -Raw | ConvertFrom-Json).workers
        foreach ($previousWorker in $previousWorkers) {
            $previousProcess = Get-Process -Id $previousWorker.pid -ErrorAction SilentlyContinue
            $sameIdentity = $null -ne $previousProcess -and $previousProcess.ProcessName -eq $previousWorker.process_name
            if ($sameIdentity -and $null -ne $previousWorker.started_at_unix) {
                $currentStart = ([DateTimeOffset]$previousProcess.StartTime.ToUniversalTime()).ToUnixTimeSeconds()
                $sameIdentity = [Math]::Abs($currentStart-$previousWorker.started_at_unix) -le 2
            }
            if ($sameIdentity) {
                throw "A previous worker is still running (PID $($previousWorker.pid)); wait for it before restarting."
            }
        }
    }
    @{ package_version=$pythonInfo.version; package_file=$pythonInfo.package_file; python=$pythonExe;
       cpu_inventory=$pythonInfo.cpu_inventory; requested_concurrency=$Concurrency; threads_per_sample=$Threads;
       shard_count=$Workers; samples=$Samples; started_at=(Get-Date).ToUniversalTime().ToString('o') } |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $jobRoot 'parallel_launch.json') -Encoding UTF8
    while ($nextIndex -lt $Workers -or $active.Count -gt 0) {
        while ($nextIndex -lt $Workers -and $active.Count -lt $Concurrency) {
            $workerArgs = @('-m','deta_generator2.worker','generate','--config',$configPath,'--processes','1','--threads',"$Threads",
                '--shard-count',"$Workers",'--shard-index',"$nextIndex",'--output',(Join-Path $jobRoot "shard$nextIndex"),
                '--pack-every',"$PackEvery",'--fail-fast')
            if ($Lumapi) { $workerArgs += @('--lumapi',$Lumapi) }
            $argumentLine = ($workerArgs | ForEach-Object { Quote-WindowsArgument $_ }) -join ' '
            $stdout = Join-Path $jobRoot "parallel_logs\shard$nextIndex.stdout.log"
            $stderr = Join-Path $jobRoot "parallel_logs\shard$nextIndex.stderr.log"
            $process = Start-Process -FilePath $pythonExe -ArgumentList $argumentLine -WorkingDirectory $jobRoot -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput $stdout -RedirectStandardError $stderr
            # Windows PowerShell 5.1 needs a retained native handle to expose
            # ExitCode reliably after a redirected child process has exited.
            $null = $process.Handle
            $runtimeWorkers = @($runtimeWorkers | Where-Object { $_.pid -ne $process.Id })
            $runtimeWorkers += @{ index=$nextIndex; pid=$process.Id; process_name=$process.ProcessName;
                started_at_unix=([DateTimeOffset]$process.StartTime.ToUniversalTime()).ToUnixTimeSeconds() }
            @{ workers=@($runtimeWorkers) } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $pidPath -Encoding UTF8
            $active += [pscustomobject]@{ Index=$nextIndex; Process=$process }
            Write-Host "Started shard $nextIndex, PID $($process.Id). Log: $stdout"
            $nextIndex++
        }
        $remaining = @()
        foreach ($worker in $active) {
            $worker.Process.Refresh()
            if ($worker.Process.HasExited) {
                $worker.Process.WaitForExit()
                $workerExitCode = $worker.Process.ExitCode
                if ($null -eq $workerExitCode -or $workerExitCode -ne 0) { $failedWorkers += $worker.Index }
                Write-Host "Shard $($worker.Index) exited with code $workerExitCode."
            } else { $remaining += $worker }
        }
        $active = $remaining
        if (((Get-Date)-$lastDisplay).TotalSeconds -ge 30 -or $active.Count -eq 0) {
            $null = Show-Progress -WriteSnapshot
            $lastDisplay = Get-Date
        }
        if ($active.Count -gt 0) { Start-Sleep -Seconds 5 }
    }
} finally {
    # Interrupting this monitor must not silently terminate native simulations.
    foreach ($worker in $active) {
        if (-not $worker.Process.HasExited) {
            Write-Warning "Shard $($worker.Index) is still running (PID $($worker.Process.Id)). This script did not stop it."
        }
    }
    if ($null -ne $controllerLock) { $controllerLock.Dispose() }
}
if ($failedWorkers.Count -gt 0) {
    throw "Failed shards: $($failedWorkers -join ', '). Results are preserved. Inspect their logs; rerun after resolving the error."
}
Merge-Results
