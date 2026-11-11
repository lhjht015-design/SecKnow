[CmdletBinding()]
param(
    [ValidateSet("strict-lexical", "competition", "enhanced", "llm")]
    [string]$Mode = "competition",
    [switch]$NoInfrastructure,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logRoot = Join-Path $projectRoot "logs"
$processFile = Join-Path $projectRoot "run\processes.json"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python -PathType Leaf)) {
    throw "Python environment is missing. Run scripts\bootstrap.ps1 first."
}
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "run"), $logRoot | Out-Null

if (-not $NoInfrastructure -and $Mode -ne "strict-lexical") {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        if ($Mode -eq "competition") {
            Write-Warning "Docker is unavailable; competition mode will use SQLite/TF-IDF/NetworkX fallback."
        }
        else {
            throw "$Mode mode requires Docker infrastructure. Install/start Docker or pass -NoInfrastructure explicitly."
        }
    }
    else {
        & docker compose --env-file (Join-Path $projectRoot ".env") -f (Join-Path $projectRoot "infra\compose.yml") up -d
        if ($LASTEXITCODE -ne 0) { throw "Docker infrastructure failed to start." }
    }
}
$env:SECKNOW_MODE = $Mode
$api = saps -FilePath $python -ArgumentList "-m", "secknow.main" -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
$worker = saps -FilePath $python -ArgumentList "-m", "secknow.worker" -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
@{ api = $api.Id; worker = $worker.Id; mode = $Mode } | ConvertTo-Json | Set-Content $processFile -Encoding utf8
Write-Host "SecKnow is running in $Mode mode at http://127.0.0.1:8765"
Write-Host "Use scripts\stop.ps1 to stop the API and worker."
if (-not $NoBrowser) { saps "http://127.0.0.1:8765" }
