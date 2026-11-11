[CmdletBinding()]
param(
    [switch]$KeepInfrastructure
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$processFile = Join-Path $projectRoot "run\processes.json"

if (Test-Path $processFile -PathType Leaf) {
    $processes = Get-Content $processFile -Raw | ConvertFrom-Json
    foreach ($processId in @($processes.api, $processes.worker)) {
        if ($processId) {
            Stop-Process -Id $processId -ErrorAction SilentlyContinue
        }
    }
    Remove-Item $processFile -Force
}

if (-not $KeepInfrastructure) {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($docker) {
        & docker compose --env-file (Join-Path $projectRoot ".env") -f (Join-Path $projectRoot "infra\compose.yml") down
        if ($LASTEXITCODE -ne 0) {
            throw "Docker infrastructure failed to stop cleanly."
        }
    }
}
Write-Host "SecKnow has stopped. Persistent data and Docker volumes were retained."
