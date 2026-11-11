[CmdletBinding()]
param(
    [ValidateSet("strict-lexical", "competition")]
    [string]$Profile = "strict-lexical",
    [string]$Output = "offline",
    [string]$TesseractRoot = "",
    [string]$TesseractVersion = "",
    [string]$TesseractSha256Manifest = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputRoot = Join-Path $projectRoot $Output
$wheelhouse = Join-Path $outputRoot "wheelhouse"
$npmCache = Join-Path $outputRoot "npm-cache"
New-Item -ItemType Directory -Force -Path $wheelhouse, $npmCache | Out-Null

Push-Location $projectRoot
try {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        throw "uv is required to prepare the Python wheelhouse."
    }
    $requirementFile = Join-Path $outputRoot "requirements.txt"
    $extras = @("--extra", "dev", "--extra", "ocr")
    if ($Profile -eq "competition") {
        $extras += @("--extra", "parse", "--extra", "semantic")
    }
    if ($TesseractRoot) {
        if (-not $TesseractVersion) {
            throw "TesseractVersion is required when TesseractRoot is provided."
        }
        if (-not $TesseractSha256Manifest) {
            throw "TesseractSha256Manifest is required when TesseractRoot is provided."
        }
        $resolvedTesseract = (Resolve-Path $TesseractRoot).Path
        $resolvedManifest = (Resolve-Path $TesseractSha256Manifest).Path
        $tesseractExe = Join-Path $resolvedTesseract "tesseract.exe"
        $engData = Join-Path $resolvedTesseract "tessdata\eng.traineddata"
        $chiData = Join-Path $resolvedTesseract "tessdata\chi_sim.traineddata"
        $osdData = Join-Path $resolvedTesseract "tessdata\osd.traineddata"
        foreach ($required in @($tesseractExe, $engData, $chiData, $osdData)) {
            if (-not (Test-Path $required -PathType Leaf)) {
                throw "Required Tesseract asset is missing: $required"
            }
        }
        $versionOutput = & $tesseractExe --version 2>&1 | Select-Object -First 1
        if ($LASTEXITCODE -ne 0 -or $versionOutput -notmatch [regex]::Escape($TesseractVersion)) {
            throw "Tesseract executable does not match requested version $TesseractVersion."
        }
        $expectedHashes = @{}
        foreach ($line in Get-Content $resolvedManifest) {
            if ($line -match '^([0-9a-fA-F]{64})\s+(.+)$') {
                $relativeName = $Matches[2].Trim().Replace("/", "\")
                if ($expectedHashes.ContainsKey($relativeName)) {
                    throw "Duplicate Tesseract manifest entry: $relativeName"
                }
                $expectedHashes[$relativeName] = $Matches[1].ToLowerInvariant()
            }
        }
        foreach ($required in @("tesseract.exe", "tessdata\eng.traineddata", "tessdata\chi_sim.traineddata", "tessdata\osd.traineddata")) {
            if (-not $expectedHashes.ContainsKey($required)) {
                throw "Tesseract manifest does not cover required asset: $required"
            }
        }
        $runtimeFiles = Get-ChildItem $resolvedTesseract -Recurse -File
        foreach ($file in $runtimeFiles) {
            $relativeName = $file.FullName.Substring($resolvedTesseract.Length + 1)
            if (-not $expectedHashes.ContainsKey($relativeName)) {
                throw "Tesseract manifest does not cover runtime file: $relativeName"
            }
            $actualHash = (Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actualHash -ne $expectedHashes[$relativeName]) {
                throw "Tesseract hash mismatch: $relativeName"
            }
        }
        if ($runtimeFiles.Count -ne $expectedHashes.Count) {
            throw "Tesseract manifest contains files that are not present in the runtime directory."
        }
    }
    & uv export --frozen --no-emit-project @extras --output-file $requirementFile
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency export failed."
    }
    & uv run --no-project --with pip python -m pip download --dest $wheelhouse --requirement $requirementFile
    if ($LASTEXITCODE -ne 0) {
        throw "Wheelhouse download failed."
    }

    Push-Location "frontend"
    try {
        & npm ci --ignore-scripts --cache $npmCache
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend cache preparation failed."
        }
        & npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed."
        }
    }
    finally {
        Pop-Location
    }

    Copy-Item "models\manifest.json" (Join-Path $outputRoot "model-manifest.json") -Force
    if ($TesseractRoot) {
        $ocrRoot = Join-Path $outputRoot "runtime\tesseract"
        if (Test-Path $ocrRoot) {
            throw "Offline Tesseract output already exists: $ocrRoot"
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $ocrRoot -Parent) | Out-Null
        Copy-Item $resolvedTesseract $ocrRoot -Recurse
        Copy-Item $resolvedManifest (Join-Path $ocrRoot "SHA256SUMS") -Force
        Set-Content (Join-Path $ocrRoot "VERSION") $TesseractVersion -Encoding ascii
    }
    $hashTargets = @($wheelhouse)
    if ($TesseractRoot) {
        $hashTargets += (Join-Path $outputRoot "runtime\tesseract")
    }
    Get-ChildItem $hashTargets -Recurse -File | Get-FileHash -Algorithm SHA256 |
        Sort-Object Path |
        ForEach-Object { "{0}  {1}" -f $_.Hash.ToLowerInvariant(), $_.Path.Substring($outputRoot.Length + 1).Replace("\", "/") } |
        Set-Content (Join-Path $outputRoot "SHA256SUMS") -Encoding ascii
    Write-Host "Offline assets prepared in $outputRoot. Model weights and Docker image archives must be imported separately after hash verification."
}
finally {
    Pop-Location
}
