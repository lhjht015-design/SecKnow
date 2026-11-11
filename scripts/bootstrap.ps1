[CmdletBinding()]
param(
    [ValidateSet("strict-lexical", "competition", "enhanced", "llm")]
    [string]$Profile = "competition",
    [string]$PythonVersion = "3.11.15",
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCommand) {
        throw "uv is required. Install it from the official Astral distribution, then rerun this script."
    }
    $uv = $uvCommand.Source
    $env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".uv-python"

    & $uv python install $PythonVersion
    if ($LASTEXITCODE -ne 0) {
        throw "Python $PythonVersion installation failed."
    }
    $pythonRuntime = (& $uv python find $PythonVersion).Trim()
    if (-not $pythonRuntime -or -not (Test-Path $pythonRuntime -PathType Leaf)) {
        throw "uv did not return a usable Python runtime."
    }

    if (-not (Test-Path ".env" -PathType Leaf)) {
        Copy-Item ".env.example" ".env"
    }

    # Parsing support is part of the platform in every mode. Semantic packages
    # are only added for modes that can use BGE-M3/Milvus.
    $extras = @("--extra", "dev", "--extra", "ocr", "--extra", "parse")
    if ($Profile -ne "strict-lexical") {
        $extras += @("--extra", "semantic")
    }
    & $uv sync --frozen --python $pythonRuntime @extras
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependency installation failed."
    }

    & ".\.venv\Scripts\python.exe" -B -m secknow.cli init
    if ($LASTEXITCODE -ne 0) {
        throw "SecKnow initialization failed."
    }

    if (-not $SkipFrontend) {
        $npm = Get-Command npm -ErrorAction SilentlyContinue
        if (-not $npm) {
            throw "Node.js 20.19+ is required to build the frontend."
        }
        Push-Location "frontend"
        try {
            & npm ci --ignore-scripts
            if ($LASTEXITCODE -ne 0) {
                throw "Frontend dependency installation failed."
            }
            & npm run build
            if ($LASTEXITCODE -ne 0) {
                throw "Frontend build failed."
            }
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "SecKnow is initialized for profile: $Profile"
}
finally {
    Pop-Location
}
