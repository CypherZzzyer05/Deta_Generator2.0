param(
    [ValidateSet('prepare','run','status','merge')][string]$Command = 'prepare',
    [string]$Python = 'python',
    [string]$JobDir = '.\jobs\jc220_topmetal_au100_metal8_100',
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
$arguments = @('-m','deta_generator2.dynamic','--action',$Command,'--job-dir',$JobDir,
               '--samples',"$Samples",'--pack-every',"$PackEvery")
if ($Jobs -gt 0) { $arguments += @('--jobs',"$Jobs") }
if ($Concurrency -gt 0) { $arguments += @('--concurrency',"$Concurrency") }
if ($Workers -gt 0) { $arguments += @('--workers',"$Workers") }
if ($Threads -gt 0) { $arguments += @('--threads',"$Threads") }
if ($Lumapi) { $arguments += @('--lumapi',$Lumapi) }
& $Python @arguments
exit $LASTEXITCODE
