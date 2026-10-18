param(
    [ValidateSet('init','doctor','preflight','plan','generate','pack','merge','status')]
    [string]$Command = 'status',
    [string]$Python = 'python',
    [string]$Config = '.\jobs\as2se3\job.json',
    [string]$JobDir = '.\jobs\as2se3',
    [ValidateSet('au-metal','ag-metal','bare-metal','au-4','au-8','ag-4','ag-8','bare-4','bare-8')]
    [string]$Preset = 'au-metal',
    [double]$MetalMeshNm = 8,
    [int]$Samples = 12,
    [int]$Threads = 8,
    [string]$Lumapi = '',
    [string]$Output = '',
    [int]$Limit = 0,
    [switch]$FailFast,
    [int]$ShardCount = 1,
    [int]$ShardIndex = 0,
    [string[]]$Shards = @()
)
$ErrorActionPreference = 'Stop'
$generatorArgs = @('-m','deta_generator2',$Command)
if ($Command -eq 'init') {
    $generatorArgs += @('--job-dir',$JobDir,'--preset',$Preset,'--samples',"$Samples",'--threads',"$Threads")
    if ($Preset -like '*-metal') { $generatorArgs += @('--metal-mesh-nm',"$MetalMeshNm") }
} else {
    $generatorArgs += @('--config',$Config,'--threads',"$Threads")
}
if ($Lumapi) { $generatorArgs += @('--lumapi',$Lumapi) }
if ($Output) { $generatorArgs += @('--output',$Output) }
if ($Limit -gt 0) { $generatorArgs += @('--limit',"$Limit") }
if ($FailFast) { $generatorArgs += '--fail-fast' }
if ($ShardCount -gt 1 -or $ShardIndex -gt 0) {
    $generatorArgs += @('--shard-count',"$ShardCount",'--shard-index',"$ShardIndex")
}
if ($Shards.Count -gt 0) { $generatorArgs += '--shards'; $generatorArgs += $Shards }
& $Python @generatorArgs
exit $LASTEXITCODE
